"""
RWKV-6-World-14B 基础能力测试
测试O(1)复杂度的RWKV架构
"""

import os
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0'  # 5090的compute_120架构太新，RWKV CUDA内核还不支持，先用纯PyTorch

import torch

print("=" * 60)
print("RWKV-6-World-14B 基础能力测试")
print("=" * 60)

# 检查模型文件是否存在
MODEL_PATH = os.path.expanduser("~/.cache/rwkv/RWKV-x060-World-14B-v2.1-20240719-ctx4096.pth")

if not os.path.exists(MODEL_PATH):
    print(f"\n模型文件不存在: {MODEL_PATH}")
    print("\n请先下载模型:")
    print("1. 访问 https://huggingface.co/BlinkDL/rwkv-6-world/tree/main")
    print("2. 下载 RWKV-x060-World-14B-v2.1-20240719-ctx4096.pth (约28GB)")
    print(f"3. 放到 {os.path.dirname(MODEL_PATH)}/")
    print("\n或者用命令下载:")
    print("mkdir -p ~/.cache/rwkv")
    print("cd ~/.cache/rwkv")
    print("wget https://huggingface.co/BlinkDL/rwkv-6-world/resolve/main/RWKV-x060-World-14B-v2.1-20240719-ctx4096.pth")
    exit(1)

print(f"\n正在加载 RWKV-6-World-14B...")
print(f"模型路径: {MODEL_PATH}")
print("（首次加载较慢，请耐心等待）\n")

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

# 加载模型 - fp16策略
model = RWKV(model=MODEL_PATH, strategy='cuda fp16')
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

print(f"模型加载完成！")
print(f"显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# 生成参数
gen_args = PIPELINE_ARGS(
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    alpha_frequency=0.25,
    alpha_presence=0.25,
    token_ban=[],
    token_stop=[0, 261],  # 停止符
    chunk_len=256
)

def ask(prompt, max_tokens=150):
    """简单问答"""
    ctx = f"User: {prompt}\n\nAssistant:"

    def callback(text):
        pass  # 不打印中间结果

    response = pipeline.generate(ctx, token_count=max_tokens, args=gen_args, callback=callback)
    # 去掉prompt部分
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    return response

# 测试1: 基础对话能力
print("\n" + "=" * 60)
print("测试1: 基础对话能力")
print("=" * 60)
q1 = "What is consciousness? Answer in 2-3 sentences."
print(f"Q: {q1}")
print(f"A: {ask(q1)}")

# 测试2: 中文能力（RWKV-World专门训过多语言）
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
