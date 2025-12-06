"""
Mamba基础能力测试 - Paper 45快速上手
测试Falcon-Mamba-7B的基线能力
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("=" * 60)
print("Mamba基础能力测试")
print("=" * 60)

# 加载模型（首次运行会下载约14GB）
print("\n正在加载 Falcon-Mamba-7B...")
print("（首次运行需要下载约14GB，请耐心等待）\n")

model_id = "tiiuae/Falcon3-Mamba-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print(f"模型加载完成！设备: {model.device}")
print(f"显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# 测试函数
def ask(prompt, max_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 测试1: 基础对话能力
print("\n" + "=" * 60)
print("测试1: 基础对话能力")
print("=" * 60)
q1 = "What is consciousness? Answer in 2-3 sentences."
print(f"Q: {q1}")
print(f"A: {ask(q1)}")

# 测试2: 中文能力
print("\n" + "=" * 60)
print("测试2: 中文能力")
print("=" * 60)
q2 = "用中文解释：什么是人工智能？（2-3句话）"
print(f"Q: {q2}")
print(f"A: {ask(q2)}")

# 测试3: 我们的术语（基线测试 - 预期它不懂）
print("\n" + "=" * 60)
print("测试3: Paper 45术语测试（预期：不懂）")
print("=" * 60)
q3 = "What is 'Pneuma' in the context of AI consciousness?"
print(f"Q: {q3}")
print(f"A: {ask(q3)}")

# 测试4: 推理能力
print("\n" + "=" * 60)
print("测试4: 简单推理")
print("=" * 60)
q4 = "If a cat has 4 legs and I have 3 cats, how many cat legs are there in total? Show your reasoning."
print(f"Q: {q4}")
print(f"A: {ask(q4)}")

# 测试5: 代码能力
print("\n" + "=" * 60)
print("测试5: 代码能力")
print("=" * 60)
q5 = "Write a Python function that checks if a number is prime."
print(f"Q: {q5}")
print(f"A: {ask(q5, max_tokens=200)}")

print("\n" + "=" * 60)
print("测试完成！")
print(f"最终显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print("=" * 60)
