"""
RWKV-6-World-14B 对话界面
简单的CLI聊天界面
"""

import os
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0'  # 5090的compute_120架构太新，用PyTorch原生CUDA

import torch

print("=" * 60)
print("RWKV-6-World-14B 对话界面")
print("=" * 60)

MODEL_PATH = os.path.expanduser("~/.cache/rwkv/RWKV-x060-World-14B-v2.1-20240719-ctx4096.pth")

if not os.path.exists(MODEL_PATH):
    print(f"模型文件不存在: {MODEL_PATH}")
    exit(1)

print(f"\n正在加载模型...")

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

model = RWKV(model=MODEL_PATH, strategy='cuda fp16')
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

print(f"模型加载完成！显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print("\n" + "=" * 60)
print("输入你的问题，输入 'quit' 或 'exit' 退出")
print("输入 'clear' 清除对话历史")
print("=" * 60 + "\n")

# 生成参数
# temperature高一点，让它更随机，不容易掉进"拒绝模板"的坑
gen_args = PIPELINE_ARGS(
    temperature=1.0,      # 0.7->1.0，更随机
    top_p=0.85,           # 稍微收一点
    top_k=50,
    alpha_frequency=0.4,  # 重复惩罚加大，避免重复输出拒绝模板
    alpha_presence=0.4,
    token_ban=[],
    token_stop=[0, 261],
    chunk_len=256
)

# 对话历史
chat_history = []

def generate_response(user_input, history):
    """生成回复"""
    # 构建上下文，用更明确的格式防止它自己编User
    ctx = "以下是用户和助手的对话。助手只回答一次，不要编造用户的问题。\n\n"
    for h in history[-5:]:  # 只保留最近5轮
        ctx += f"用户：{h['user']}\n\n助手：{h['assistant']}\n\n"
    ctx += f"用户：{user_input}\n\n助手："

    # 流式输出
    print("Assistant: ", end="", flush=True)

    response_text = ""

    def callback(text):
        nonlocal response_text
        # 处理可能的Unicode编码问题
        try:
            clean_text = text.encode('utf-8', errors='ignore').decode('utf-8')
            print(clean_text, end="", flush=True)
            response_text += clean_text
        except:
            pass  # 跳过无法编码的字符

    pipeline.generate(ctx, token_count=300, args=gen_args, callback=callback)
    print()  # 换行

    return response_text.strip()

# 主循环
while True:
    try:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break

        if user_input.lower() == 'clear':
            chat_history = []
            print("对话历史已清除。")
            continue

        # 处理转义字符（允许用 \n 输入换行）
        user_input = user_input.replace('\\n', '\n')

        # 生成回复
        response = generate_response(user_input, chat_history)

        # 保存历史
        chat_history.append({
            'user': user_input,
            'assistant': response
        })

    except KeyboardInterrupt:
        print("\n\n再见！")
        break
    except Exception as e:
        print(f"\n错误: {e}")
        continue
