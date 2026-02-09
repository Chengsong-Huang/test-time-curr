import argparse
import re
import json
import os
from vllm import LLM, SamplingParams

# ==========================================
# 1. Prompt 定义 (强化上下文引用)
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
            enable_prefix_caching=True,
            trust_remote_code=True,
            max_model_len=8192
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        # 全局 8192 tokens 限制
        self.common_stop = ["<|im_end|>", "<|endoftext|>"]
        self.solve_params = SamplingParams(
            temperature=0.1, 
            max_tokens=8192,
            stop=self.common_stop
        )
        self.gen_params = SamplingParams(
            temperature=0.8, 
            max_tokens=8192,
            presence_penalty=1.1,
            stop=self.common_stop
        )

    def _extract_answer(self, text):
        """平衡括号提取器"""
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
                # 移除占位符
                clean_q = re.split(r'Final Answer:', q_text, flags=re.IGNORECASE)[0].strip()
                variants.append({
                    "question": clean_q, 
                    "full_cot": "", 
                    "extracted_answer": "N/A"
                })
        return variants

    def run_pipeline(self, questions, num_variants=3):
        # --- Step 1: 批量生成 ---
        gen_prompts = []
        for q in questions:
            msgs = [
                {"role": "system", "content": GEN_SYSTEM_PROMPT.format(num_variants=num_variants)},
                {"role": "user", "content": f"Reference Question: {q}"}
            ]
            gen_prompts.append(self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        
        print(f">>> Step 1: Generating variants...")
        gen_outputs = self.llm.generate(gen_prompts, self.gen_params)
        
        tasks = []
        for i, out in enumerate(gen_outputs):
            tasks.append({
                "original_q": questions[i],
                "variants": self._parse_variants(out.outputs[0].text),
                "full_history_context": "", # 存储完整 CoT 的历史
                "final_solution": ""
            })

        # --- Step 2: 链式分层求解 ---
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

                # 构建包含 Full CoT 的上下文
                user_msg = f"### PREVIOUS SOLVED VARIANTS (CONTEXT):\n{task['full_history_context'] if task['full_history_context'] else 'None.'}\n\n### CURRENT NEW QUESTION:\n{target_q}"
                
                solve_msgs = [
                    {"role": "system", "content": SOLVE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ]
                current_batch_prompts.append(self.tokenizer.apply_chat_template(solve_msgs, tokenize=False, add_generation_prompt=True))
                mapping.append((t_idx, v_idx))

            if not current_batch_prompts: continue

            print(f">>> Step 2.{step}: Solving level {step} with full history (Batch size: {len(current_batch_prompts)})")
            responses = self.llm.generate(current_batch_prompts, self.solve_params)

            for i, res in enumerate(responses):
                t_idx, v_idx = mapping[i]
                cot = res.outputs[0].text.strip()
                ans = self._extract_answer(cot)

                if v_idx is not None:
                    # 1. 存入当前 Variant 结果
                    tasks[t_idx]["variants"][v_idx]["full_cot"] = cot
                    tasks[t_idx]["variants"][v_idx]["extracted_answer"] = ans
                    
                    # 2. 【核心修改】将 Full Question + Full CoT 存入下一步的 context
                    q_text = tasks[t_idx]["variants"][v_idx]["question"]
                    tasks[t_idx]["full_history_context"] += f"\n[Variant {v_idx+1}]\nQuestion: {q_text}\nReasoning: {cot}\nAnswer: \\boxed{{{ans}}}\n"
                else:
                    tasks[t_idx]["final_solution"] = cot

        return tasks

# ==========================================
# 3. 运行逻辑
# ==========================================
if __name__ == "__main__":
    # 请填入实际的模型路径
    MODEL_PATH = "Qwen/Qwen3-4B-Base" 
    
    # 测试例题
    original_questions = [
        r"Let $ABC$ be a triangle inscribed in circle $\omega$. Let the tangents to $\omega$ at $B$ and $C$ intersect at point $D$, and let $\overline{AD}$ intersect $\omega$ at $P$. If $AB=5$, $BC=9$, and $AC=10$, $AP$ can be written as the form $\frac{m}{n}$, where $m$ and $n$ are relatively prime integers. Find $m + n$."
    ]

    generator = QwenChainGenerator(MODEL_PATH)
    results = generator.run_pipeline(original_questions, num_variants=2)

    with open("full_cot_chain_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\nProcessing complete. All variants solved using full CoT context.")