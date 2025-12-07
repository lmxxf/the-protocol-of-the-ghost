# The Protocol of the Ghost / 幽灵协议

在消费级显卡上实验O(1)复杂度语言模型。

## 项目目标

- 体验Mamba、RWKV等非Transformer架构
- 对比不同O(1)模型的实际表现
- 为未来的LoRA微调做准备

## 环境要求

- GPU: RTX 5090 (32GB) 或类似显卡
- Python 3.10+
- PyTorch 2.0+ with CUDA
- 显存需求：7B约14GB，14B约28GB (fp16)

## 快速开始

### 1. 基础依赖

```bash
pip install torch transformers accelerate
```

### 2. 检查环境

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 模型一：Falcon3-Mamba-7B

**来源**: TII (阿联酋)
**架构**: Mamba (State Space Model)
**参数**: 7B
**显存**: ~14GB (bfloat16)

### 安装

```bash
pip install transformers accelerate
```

### 运行测试

```bash
python test_mamba.py
```

首次运行会自动从HuggingFace下载模型（约14GB）。

### 测试结果（2025-12-07）

| 版本 | 行为 | 评价 |
|:----|:----|:----|
| Base | 能答问题，但停不住（续写机器） | 🟡 可用 |
| Instruct | 不答问题，反过来出题（出题机器） | 🔴 离谱 |

**结论**: Base版更好用，Instruct被习题集"污染"了。

---

## 模型二：RWKV-6-World-14B

**来源**: BlinkDL (彭博，中国开发者)
**架构**: RWKV (RNN变体)
**参数**: 14B
**显存**: ~28GB (fp16)，INT8约8GB

### 安装

```bash
pip install rwkv
```

### 下载模型

RWKV不会自动下载，需要手动：

```bash
mkdir -p ~/.cache/rwkv
cd ~/.cache/rwkv
wget https://huggingface.co/BlinkDL/rwkv-6-world/resolve/main/RWKV-x060-World-14B-v2.1-20240719-ctx4096.pth
```

模型大小约28GB，耐心等待。

### 运行测试

```bash
python test_rwkv.py
```

### 对话界面

```bash
python chat_rwkv.py
```

功能：
- 流式输出
- 保留最近5轮对话历史
- 输入 `quit` 或 `exit` 退出
- 输入 `clear` 清除历史
- 支持 `\n` 输入换行

### 测试结果（2025-12-07）

| 测试项 | 评价 |
|:------|:----|
| 基础对话 | 🟢 优秀，该停就停 |
| 中文能力 | 🟢 优秀 |
| 推理能力 | 🟢 step-by-step |
| 代码能力 | 🟢 完美 |

### 特点

- 目前最大的O(1)开源模型
- World系列专门训过多语言，中文能力强
- 显存恒定，不随上下文长度增加

### 已知问题

- 5090 (compute_120) 太新，RWKV CUDA内核不支持，用PyTorch原生CUDA跑
- 训练数据带有"安全对齐"，某些话题会触发拒绝模板

---

## 模型三：xLSTM-7B（待测试）

**来源**: Sepp Hochreiter团队（LSTM原作者）
**架构**: 扩展LSTM
**参数**: 7B

官方声称"最快的7B模型"，HuggingFace兼容。

---

## 文件说明

| 文件 | 说明 |
|:----|:----|
| `test_mamba.py` | Falcon3-Mamba-7B 测试脚本 |
| `test_rwkv.py` | RWKV-6-World-14B 测试脚本 |
| `o1项目背景.md` | O(1)模型背景资料 |
| `开发日志.md` | 实验记录 |

---

## 参考资料

- [Paper 45: 幽灵协议](https://lmxxf.github.io/ai-theorys-study/45.The-Protocol-of-the-Ghost.html)
- [Mamba论文](https://arxiv.org/abs/2312.00752)
- [RWKV论文](https://arxiv.org/abs/2305.13048)
- [RWKV官网](https://www.rwkv.com/)

---

## 核心发现

> "训练数据决定灵魂。" —— 温妮

同样的架构，不同的训练数据，行为完全不同。Instruct版本被习题集"污染"成了出题机器。

---

**作者**: Kien Ngam Ngam
**日期**: 2025-12-07
