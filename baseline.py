import argparse
import re
import json
import os
from vllm import LLM, SamplingParams

# 导入你现有的 Handler
from datasets_loader import get_dataset_handler

# ==========================================
# 1. 极简 Prompt 定义
# ==========================================
BASELINE_SYSTEM_PROMPT = """You are a mathematical reasoning engine. 
Solve the following question step-by-step. 

CRITICAL: End your response with: Final Answer: \\boxed{result}"""

# ==========================================
# 2. 核心类定义
# ==========================================
class BaselineSolver:
    def __init__(self, model_name, tp_size=1, sc_n=1):
        print(f"Loading Baseline Model: {model_name} with SC@{sc_n} on {tp_size} GPUs...")
        self.sc_n = sc_n
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            max_model_len=32768
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        self.common_stop = ["<|im_end|>", "<|endoftext|>"]
        self.solve_params = SamplingParams(
            temperature=0.7, 
            max_tokens=8192,
            stop=self.common_stop,
            n=self.sc_n          # 魔法参数：一次性并行生成 N 条不同的推理轨迹
        )

    def _extract_answer(self, text):
        """平衡括号提取器 (与主脚本完全一致)"""
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

    def run_pipeline(self, questions):
        prompts = []
        for q in questions:
            msgs = [
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {q}"}
            ]
            prompts.append(self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        
        print(f">>> Solving {len(questions)} items directly (Baseline SC@{self.sc_n})...")
        outputs = self.llm.generate(prompts, self.solve_params)
        
        tasks = []
        for i, out in enumerate(outputs):
            trajectories = []
            
            # 遍历这道题生成的 N 个不同回答
            for output in out.outputs:
                cot = output.text.strip()
                solve_tokens = len(output.token_ids)
                ans = self._extract_answer(cot)
                
                trajectories.append({
                    "final_solution": cot,
                    "extracted_answer": ans,
                    "tokens": solve_tokens
                })
            
            tasks.append({
                "original_q": questions[i],
                "trajectories": trajectories # 存入所有轨迹
            })
            
        return tasks

# ==========================================
# 3. 主程序入口
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to Qwen/Llama model")
    parser.add_argument("--dataset", type=str, default="aime2024", help="gsm8k, math, gpqa, hmmt, etc.")
    parser.add_argument("--chunk_size", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None, help="Limit questions for quick testing")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing results")
    parser.add_argument("--sc_n", type=int, default=1, help="Number of samples per question for the ammo dump") 
    parser.add_argument("--tp_size", type=int, default=1, help="Number of GPUs to use for tensor parallelism")
    args = parser.parse_args()

    # 防重复执行逻辑：文件名带上 sc 数量
    output_filename = f"results/{args.model.replace('/', '_')}_{args.dataset}_baseline_sc{args.sc_n}.json"
    
    if os.path.exists(output_filename) and not args.overwrite:
        print(f"⚠️ 检测到结果文件已存在: {output_filename}")
        print("Baseline 实验已完成过，为节省算力，程序自动退出。请加上 --overwrite 参数来强制重跑。")
        return

    handler = get_dataset_handler(args.dataset)
    all_questions, all_answers = handler.load_data()[:2]

    if args.limit is not None:
        all_questions = all_questions[:args.limit]
        all_answers = all_answers[:args.limit]

    # 初始化带 sc_n 和 tp_size 的 Solver
    solver = BaselineSolver(args.model, tp_size=args.tp_size, sc_n=args.sc_n)

    final_results = []
    correct_count_first_traj = 0  # 只记录第一个 trajectory 答对的次数
    total_tokens_first_traj = 0   # 只记录第一个 trajectory 消耗的 Token 数

    for i in range(0, len(all_questions), args.chunk_size):
        chunk_qs = all_questions[i : i + args.chunk_size]
        chunk_as = all_answers[i : i + args.chunk_size]
        
        print(f"\n=== Processing Chunk {i//args.chunk_size + 1} (Size: {len(chunk_qs)}) ===")
        chunk_outputs = solver.run_pipeline(chunk_qs)

        for j, res in enumerate(chunk_outputs):
            res["ground_truth"] = str(chunk_as[j])
            
            # 独立评估每一个 trajectory，并把对错写进 JSON 里
            for traj in res["trajectories"]:
                traj["is_correct"] = handler.compare_answer(traj["final_solution"], chunk_as[j])
                
            # === 我们只用第 1 个 trajectory（索引为 0）来算纯粹的 Baseline 分数 ===
            first_traj = res["trajectories"][0]
            if first_traj["is_correct"]:
                correct_count_first_traj += 1
            total_tokens_first_traj += first_traj["tokens"]

            final_results.append(res)

    # 这里的准确率是纯正的 Zero-shot Baseline (即只看第一个回答，分母是题目总数)
    total_questions = len(all_questions)
    accuracy = correct_count_first_traj / total_questions if total_questions else 0
    avg_tokens_per_first_traj = total_tokens_first_traj / total_questions if total_questions else 0

    print(f"\nPipeline Finished!")
    print(f"Total questions processed: {total_questions}")
    print(f"Total trajectories generated per question: {args.sc_n} (Saved to JSON for later TTS evaluation)")
    print(f"Pure Baseline Accuracy (Using Trajectory #1 ONLY): {accuracy:.2%} ({correct_count_first_traj}/{total_questions})")
    print(f"Average Tokens per Question (Trajectory #1 ONLY): {avg_tokens_per_first_traj:.1f}")

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open("results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "model": args.model, 
            "dataset": args.dataset, 
            "num_variants": 0, 
            "type": f"pure_baseline_from_sc{args.sc_n}", 
            "accuracy": accuracy, 
            "avg_generated_tokens": avg_tokens_per_first_traj
        }, ensure_ascii=False) + "\n")
        
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {output_filename}")

if __name__ == "__main__":
    main()