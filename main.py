import argparse
import re
import json
import os
import random
import torch
from vllm import LLM, SamplingParams

# 从你的文件导入 Handler 逻辑
from datasets_loader import get_dataset_handler

# ==========================================
# 1. Prompts 定义 (三种生成策略)
# ==========================================

OUTPUT_FORMAT_INSTRUCTIONS = """
**Strict Output Format:**
You MUST output EXACTLY {num_variants} blocks using the exact markdown structure below. Do not stop until all variants are generated.

### Variant [Number]
**Design:** [Explain the specific strategy used for this variant]
**Question:**
[The problem description in LaTeX]
**Final Answer:** \\boxed{{calculated_result}}
"""

# 策略 1：平行难度 (Parallel)
GEN_SYSTEM_PROMPT_PARALLEL = """You are an expert mathematics problem setter. Your task is to generate EXACTLY **{num_variants}** Structurally Isomorphic but Surface-Distinct variants of the provided reference question.

**Design Principle:**
Generate problems that strictly maintain the **same difficulty level** and **mathematical structure** as the Reference Question. Do not make them easier or harder. They should serve as parallel practice problems to reinforce the specific logic required.
""" + OUTPUT_FORMAT_INSTRUCTIONS

# 策略 2：真实爬坡 (Staircase - 从简到难)
GEN_SYSTEM_PROMPT_STAIRCASE = """You are an expert mathematics problem setter. Your task is to generate EXACTLY **{num_variants}** variants of the provided reference question, acting as stepping stones.

**The Difficulty Staircase:**
- Variant 1: A highly simplified version of the reference question (e.g., fewer dimensions, removed constraints, simpler numbers).
- Progressively increase the complexity and restore constraints for subsequent variants.
- Variant {num_variants}: Should be close to, but still slightly easier than the original reference question.
""" + OUTPUT_FORMAT_INSTRUCTIONS

# 策略 3：负重训练 (Overload - 越来越难)
GEN_SYSTEM_PROMPT_OVERLOAD = """You are an expert mathematics problem setter. Your task is to generate EXACTLY **{num_variants}** variants of the provided reference question, deliberately pushing the boundaries of difficulty.

**The Overload Trajectory:**
- Variant 1: Must be **strictly harder** than the original reference question right from the start. Do not start at the same difficulty. Introduce immediate complexity (e.g., adding tricky constraints, increasing dimensions, or blending in related theorems).
- Progressively increase the mathematical complexity and computational burden for each subsequent variant.
- Variant {num_variants}: Should be an extreme challenge, significantly more complex than both the original reference question and Variant 1.
""" + OUTPUT_FORMAT_INSTRUCTIONS

# 求解 Prompt 保持一致 (Step 2 共用)
SOLVE_SYSTEM_PROMPT = """You are a mathematical reasoning engine. 
Solve the following question step-by-step. You will be provided with previous related problems and their full reasoning as context. Use their logic to solve the current NEW question.

CRITICAL: End your response with: Final Answer: \\boxed{{result}}"""

# ==========================================
# 2. 核心类定义
# ==========================================
class QwenChainGenerator:
    def __init__(self, model_name, tp_size=1):
        print(f"Loading Model: {model_name}...")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=0.90,
            enable_prefix_caching=True, # 开启前缀缓存加速链式推理
            trust_remote_code=True,
            max_model_len=32768
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        self.common_stop = ["<|im_end|>", "<|endoftext|>"]
        self.solve_params = SamplingParams(
            temperature=0.7, 
            max_tokens=8192, 
            stop=self.common_stop
        )
        self.gen_params = SamplingParams(
            temperature=0.8, 
            max_tokens=4096,
            stop=self.common_stop
        )

    def _extract_answer(self, text):
        """平衡括号提取器，严格处理 LaTeX \\boxed"""
        match = re.search(r'\\boxed\{', text)
        if not match: return "N/A"
        start_idx = match.end()
        opened, res = 1, ""
        for i in range(start_idx, len(text)):
            if text[i] == '{': opened += 1
            elif text[i] == '}':
                opened -= 1
                if opened == 0: break
            res += text[i]
        return res if opened == 0 else "N/A"

    def _parse_variants(self, text):
        """使用 Markdown 标题进行鲁棒解析"""
        variants = []
        blocks = re.split(r'###\s*Variant\s*\d+', text, flags=re.IGNORECASE)
        
        for block in blocks[1:]:
            q_match = re.search(r'\*\*Question:\*\*(.*?)\*\*Final Answer:\*\*', block, re.DOTALL | re.IGNORECASE)
            if q_match:
                q_text = q_match.group(1).strip()
                variants.append({
                    "question": q_text, 
                    "full_cot": "", 
                    "extracted_answer": "N/A"
                })
        return variants

    def run_pipeline(self, questions, num_variants=3, strategy="staircase", similarity_map=None):
        tasks = []

        # ==========================================
        # Step 1: 准备变体 (生成 or 题库抽样)
        # ==========================================
        if strategy == "sample":
            print(f">>> Step 1: Retrieving top-{num_variants} variants from dataset for {len(questions)} items...")
            for q in questions:
                # 随机抽取指定数量的题目
                sampled_qs = similarity_map[q]
                
                variants_list = []
                for sq in sampled_qs:
                    variants_list.append({
                        "question": sq,
                        "full_cot": "",
                        "extracted_answer": "N/A"
                    })
                
                tasks.append({
                    "original_q": q,
                    "variants": variants_list,
                    "full_history_context": "", 
                    "final_solution": "",
                    "total_generated_tokens": 0, # 抽样不需要生成 token
                    "step_tokens": {"generation": 0, "solving_steps": []}
                })

        else:
            # 根据策略选择对应的 Prompt
            if strategy == "parallel":
                base_prompt = GEN_SYSTEM_PROMPT_PARALLEL
            elif strategy == "overload":
                base_prompt = GEN_SYSTEM_PROMPT_OVERLOAD
            else: # 默认 staircase
                base_prompt = GEN_SYSTEM_PROMPT_STAIRCASE
                
            formatted_prompt = base_prompt.format(num_variants=num_variants)

            gen_prompts = []
            for q in questions:
                msgs = [
                    {"role": "system", "content": formatted_prompt},
                    {"role": "user", "content": f"Reference Question: {q}\n\nRemember, you MUST generate EXACTLY {num_variants} variants using the required format."}
                ]
                gen_prompts.append(self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            
            print(f">>> Step 1: Generating {strategy.upper()} variants for {len(questions)} items...")
            gen_outputs = self.llm.generate(gen_prompts, self.gen_params)
            
            for i, out in enumerate(gen_outputs):
                gen_tokens = len(out.outputs[0].token_ids)
                variants_extracted = self._parse_variants(out.outputs[0].text)
                
                print(f"[Monitor] Original Q{i+1}: Expected {num_variants}, Extracted {len(variants_extracted)} variants.")
                
                tasks.append({
                    "original_q": questions[i],
                    "variants": variants_extracted,
                    "full_history_context": "", 
                    "final_solution": "",
                    "total_generated_tokens": gen_tokens,
                    "step_tokens": {"generation": gen_tokens, "solving_steps": []}
                })

        # ==========================================
        # Step 2: 链式分层并行求解 (各策略共用)
        # ==========================================
        for step in range(num_variants + 1):
            is_final = (step == num_variants)
            current_batch_prompts, mapping = [], []

            for t_idx, task in enumerate(tasks):
                if not is_final:
                    if step < len(task["variants"]):
                        target_q = task["variants"][step]["question"]
                        v_idx = step
                    else: continue # 容错处理：如果提取的变体不够，直接跳过当前层的这道题
                else:
                    target_q = task["original_q"]
                    v_idx = None

                user_msg = f"### PREVIOUS SOLVED VARIANTS (CONTEXT):\n{task['full_history_context'] if task['full_history_context'] else 'None.'}\n\n### CURRENT NEW QUESTION:\n{target_q}"
                
                solve_msgs = [
                    {"role": "system", "content": SOLVE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ]
                current_batch_prompts.append(self.tokenizer.apply_chat_template(solve_msgs, tokenize=False, add_generation_prompt=True))
                mapping.append((t_idx, v_idx))

            if not current_batch_prompts: continue

            print(f">>> Step 2.{step}: Solving level {step} (Batch size: {len(current_batch_prompts)})")
            responses = self.llm.generate(current_batch_prompts, self.solve_params)

            for i, res in enumerate(responses):
                t_idx, v_idx = mapping[i]
                cot = res.outputs[0].text.strip()
                ans = self._extract_answer(cot)
                solve_tokens = len(res.outputs[0].token_ids)
                
                tasks[t_idx]["total_generated_tokens"] += solve_tokens
                tasks[t_idx]["step_tokens"]["solving_steps"].append(solve_tokens)

                if v_idx is not None:
                    tasks[t_idx]["variants"][v_idx]["full_cot"] = cot
                    tasks[t_idx]["variants"][v_idx]["extracted_answer"] = ans
                    q_text = tasks[t_idx]["variants"][v_idx]["question"]
                    # 将这一步的题目、推理和答案加入上下文，供下一步使用
                    tasks[t_idx]["full_history_context"] += f"\n[Variant {v_idx+1}]\nQuestion: {q_text}\nReasoning: {cot}\nAnswer: \\boxed{{{ans}}}\n"
                else:
                    tasks[t_idx]["final_solution"] = cot
                    tasks[t_idx]["final_extracted_answer"] = ans

        return tasks

# ==========================================
# 3. 主程序入口
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to Qwen/Llama model")
    parser.add_argument("--dataset", type=str, default="aime2024", help="gsm8k, math, gpqa, etc.")
    parser.add_argument("--num_variants", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Limit questions for quick testing")
    parser.add_argument("--chunk_size", type=int, default=200, help="Process N questions at a time to save memory")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing results")
    parser.add_argument("--tp_size", type=int, default=1, help="Number of GPUs to use for tensor parallelism")
    
    # 策略选择参数 (支持四种模式)
    parser.add_argument("--strategy", type=str, choices=["staircase", "parallel", "overload", "sample"], default="staircase", 
                        help="Choose generation strategy: 'staircase', 'parallel', 'overload', or 'sample' (random from dataset).")
    args = parser.parse_args()

    # 根据策略动态命名输出文件
    output_filename = f"results/{args.model.replace('/', '_')}_{args.dataset}_v{args.num_variants}_{args.strategy}.json"
    
    # 防重复执行检查
    if os.path.exists(output_filename) and not args.overwrite:
        print(f"⚠️ 检测到结果文件已存在: {output_filename}")
        print("实验已完成过，为节省算力，程序自动退出。请加上 --overwrite 参数来强制重跑。")
        return

    # 1. 加载数据集
    handler = get_dataset_handler(args.dataset)
    all_questions, all_answers, unique_questions = handler.load_data()
    
    # 如果有限制数量，进行切片
    if args.limit is not None:
        all_questions = all_questions[:args.limit]
        all_answers = all_answers[:args.limit]

    # 2. 初始化生成器
    generator = QwenChainGenerator(args.model, tp_size=args.tp_size)

    similarity_map = {}
    if args.strategy == "sample":
        print(f"\n>>> Precomputing SentenceTransformer similarity matrix for {len(unique_questions)} unique questions...")
        from sentence_transformers import SentenceTransformer, util
        
        embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # Use the pristine, unique_questions list directly!
        embeddings = embedder.encode(unique_questions, convert_to_tensor=True)
        
        k_variants = min(args.num_variants, len(unique_questions) - 1)
        for i, q in enumerate(unique_questions):
            cos_scores = util.cos_sim(embeddings[i], embeddings)[0]
            cos_scores[i] = -9999 # Mask self
            
            top_results = torch.topk(cos_scores, k=k_variants)
            similarity_map[q] = [unique_questions[idx] for idx in top_results[1]]
        print(">>> Similarity matrix precomputation complete!\n")

    # 3. 分片处理 (Chunking)
    final_results = []
    correct_count = 0
    total_tokens_all_questions = 0

    for i in range(0, len(all_questions), args.chunk_size):
        chunk_qs = all_questions[i : i + args.chunk_size]
        chunk_as = all_answers[i : i + args.chunk_size]
        
        print(f"\n=== Processing Chunk {i//args.chunk_size + 1} (Size: {len(chunk_qs)}, Strategy: {args.strategy}) ===")
        # 传入 strategy 和全量题库
        chunk_outputs = generator.run_pipeline(chunk_qs, args.num_variants, strategy=args.strategy, similarity_map=similarity_map)

        # 验证结果
        for j, res in enumerate(chunk_outputs):
            is_correct = handler.compare_answer(res["final_solution"], chunk_as[j])
            if is_correct: correct_count += 1

            total_tokens_all_questions += res["total_generated_tokens"]
            
            res["ground_truth"] = str(chunk_as[j])
            res["is_correct"] = is_correct
            final_results.append(res)

    # 4. 统计与保存
    accuracy = correct_count / len(all_questions) if all_questions else 0
    avg_tokens = total_tokens_all_questions / len(all_questions) if all_questions else 0

    print(f"\n{args.strategy.capitalize()} Pipeline Finished!")
    print(f"Total processed: {len(all_questions)}")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Average Tokens per Question: {avg_tokens:.1f}")

    # 自动创建 results 目录
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open("results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "model": args.model, 
            "dataset": args.dataset, 
            "num_variants": args.num_variants, 
            "strategy": args.strategy, # 记录跑的是哪个策略
            "accuracy": accuracy, 
            "avg_generated_tokens": avg_tokens
        }, ensure_ascii=False) + "\n")
        
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {output_filename}")

if __name__ == "__main__":
    main()
