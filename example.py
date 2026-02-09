import argparse
import re
import json
import os
from vllm import LLM, SamplingParams

# 从你的文件导入 Handler 逻辑
from datasets_loader import get_dataset_handler

# ==========================================
# 1. Prompt 定义 (严格遵循你的要求)
# ==========================================
GEN_SYSTEM_PROMPT = """You are an expert mathematics problem setter. Your task is to generate **{num_variants}** Structurally Isomorphic but Surface-Distinct problems.

**The Difficulty Staircase:**
1. Variant 1: Base level. 
2. Variant 2 to {num_variants}: Progressively increase complexity (e.g., adding constraints, increasing dimensions).

**Strict Output Format:**
[[VARIANT_START_ID]]
**[Design]:** (strategy)
**<question>**
(The problem in LaTeX)
**</question>**
**Final Answer:** \\boxed{{calculated_result}}
[[VARIANT_END_ID]]"""

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
            max_model_len=16384
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        self.common_stop = ["<|im_end|>", "<|endoftext|>"]
        self.solve_params = SamplingParams(
            temperature=0.7, 
            max_tokens=4096, # 求解单个问题通常不需要8192，设为2048更稳
            stop=self.common_stop
        )
        self.gen_params = SamplingParams(
            temperature=0.8, 
            max_tokens=4096,
            # presence_penalty=1.1,
            stop=self.common_stop
        )

    def _extract_answer(self, text):
        """平衡括号提取器，严格处理 LaTeX \boxed"""
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
        pattern = re.compile(r'\[\[VARIANT_START_ID\]\](.*?)\[\[VARIANT_END_ID\]\]', re.DOTALL)
        blocks = pattern.findall(text)
        variants = []
        for block in blocks:
            q_match = re.search(r'<question>\s*(.*?)\s*</question>', block, re.DOTALL | re.IGNORECASE)
            if q_match:
                q_text = q_match.group(1).strip()
                clean_q = re.split(r'Final Answer:', q_text, flags=re.IGNORECASE)[0].strip()
                variants.append({
                    "question": clean_q, 
                    "full_cot": "", 
                    "extracted_answer": "N/A"
                })
        return variants

    def run_pipeline(self, questions, num_variants=3):
        # --- Step 1: 批量生成变体 ---
        gen_prompts = []
        for q in questions:
            msgs = [
                {"role": "system", "content": GEN_SYSTEM_PROMPT.format(num_variants=num_variants)},
                {"role": "user", "content": f"Reference Question: {q}"}
            ]
            gen_prompts.append(self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        
        print(f">>> Step 1: Generating variants for {len(questions)} items...")
        gen_outputs = self.llm.generate(gen_prompts, self.gen_params)
        
        tasks = []
        for i, out in enumerate(gen_outputs):
            tasks.append({
                "original_q": questions[i],
                "variants": self._parse_variants(out.outputs[0].text),
                "full_history_context": "", 
                "final_solution": ""
            })

        # --- Step 2: 链式分层并行求解 ---
        # 利用 vLLM Prefix Caching，每一层 Step 都会重用前一层的 KV Cache
        
        for step in range(num_variants + 1):
            is_final = (step == num_variants)
            current_batch_prompts, mapping = [], []

            for t_idx, task in enumerate(tasks):
                if not is_final:
                    if step < len(task["variants"]):
                        target_q = task["variants"][step]["question"]
                        v_idx = step
                    else: continue
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

                if v_idx is not None:
                    tasks[t_idx]["variants"][v_idx]["full_cot"] = cot
                    tasks[t_idx]["variants"][v_idx]["extracted_answer"] = ans
                    q_text = tasks[t_idx]["variants"][v_idx]["question"]
                    # 更新 Full History，供下一层 Step 引用
                    tasks[t_idx]["full_history_context"] += f"\n[Variant {v_idx+1}]\nQuestion: {q_text}\nReasoning: {cot}\nAnswer: \\boxed{{{ans}}}\n"
                else:
                    tasks[t_idx]["final_solution"] = cot

        return tasks

# ==========================================
# 3. 主程序入口
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to Qwen/Llama model")
    parser.add_argument("--dataset", type=str, default="aime2024", help="gsm8k, math, gpqa, etc.")
    parser.add_argument("--num_variants", type=int, default=1)
    # parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=200, help="Process N questions at a time to save memory")
    args = parser.parse_args()

    # 1. 加载数据集
    handler = get_dataset_handler(args.dataset)
    all_questions, all_answers = handler.load_data()
    # all_questions = all_questions[:args.limit]
    # all_answers = all_answers[:args.limit]

    # 2. 初始化生成器
    generator = QwenChainGenerator(args.model)

    # 3. 分片处理 (Chunking) 以保证大规模运行的稳定性
    final_results = []
    correct_count = 0

    for i in range(0, len(all_questions), args.chunk_size):
        chunk_qs = all_questions[i : i + args.chunk_size]
        chunk_as = all_answers[i : i + args.chunk_size]
        
        print(f"\n=== Processing Chunk {i//args.chunk_size + 1} (Size: {len(chunk_qs)}) ===")
        chunk_outputs = generator.run_pipeline(chunk_qs, args.num_variants)

        # 验证结果
        for j, res in enumerate(chunk_outputs):
            is_correct = handler.compare_answer(res["final_solution"], chunk_as[j])
            if is_correct: correct_count += 1
            
            res["ground_truth"] = str(chunk_as[j])
            res["is_correct"] = is_correct
            final_results.append(res)

    # 4. 统计与保存
    accuracy = correct_count / len(all_questions) if all_questions else 0
    print(f"\nPipeline Finished!")
    print(f"Total processed: {len(all_questions)}")
    print(f"Overall Accuracy: {accuracy:.2%}")
    with open("results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"model": args.model, "dataset": args.dataset, "num_variants": args.num_variants, "accuracy": accuracy}, ensure_ascii=False) + "\n")
    output_filename = f"results/{args.model.replace('/', '_')}_{args.dataset}_v{args.num_variants}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {output_filename}")

if __name__ == "__main__":
    main()