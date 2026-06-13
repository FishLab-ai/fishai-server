# 🐟 FishAI Engine

> FishLab-ai 自研 GPT 推理引擎 — Rust 编写，4-bit 量化，无需 Git LFS

## 概述

FishAI Engine 是 **FishLab-ai** 团队完全从零自研的 GPT 推理引擎，使用纯 Rust 编写。

### 核心特点

| 特性 | 描述 |
|------|------|
| 🦀 语言 | Rust (编译型，高性能，内存安全) |
| 🧮 架构 | GPT-2 风格 Decoder-Only Transformer |
| 📦 量化 | INT4 Per-Channel 量化，权重仅 ~25MB |
| 🚫 无 LFS | 量化权重直接放进 Git 仓库 |
| ⚡ 推理 | 自研前向传播 + 温度采样 |
| 🌐 API | axum HTTP 服务 |

### 模型参数

```
d_model:     512   n_heads:     8
n_layers:    6     d_ff:        2048
vocab_size:  32000 max_seq_len: 512
总参数量:    ~52M  4-bit 量化:  ~25MB
```

## 项目结构

```
src/
├── main.rs       # 入口：启动 HTTP 服务器
├── model.rs      # GPT 模型架构（注意力、FFN、LayerNorm、采样）
├── quantize.rs   # INT4 量化/解量化
├── tokenizer.rs  # BPE 分词器
└── api.rs        # HTTP API (axum)
```

## 构建 & 运行

```bash
cargo build --release
mkdir -p weights
./target/release/fishai-engine
```

## 许可证

MIT License - FishLab-ai
