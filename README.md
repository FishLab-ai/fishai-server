# 🐟 FishAI Engine v2

> FishLab-ai 自研 GPT 推理引擎 — Rust 编写，LLaMA-style 架构，混合精度量化，无需 Git LFS

## v2 架构升级

| 特性 | v1 (GPT-2) | v2 (LLaMA-style) |
|------|-----------|------------------|
| 位置编码 | Learned Position Embedding | **RoPE** (零参数，更好长度外推) |
| 激活函数 | GELU | **SwiGLU** (更强表达力) |
| 归一化 | LayerNorm | **RMSNorm** (更快更简) |
| 注意力 | MHA | **GQA** (省 7% 参数 + 50% KV 缓存) |
| 偏置 | 有 | **无偏置** (现代发现冗余) |
| LM Head | 独立 | **权重绑定** (省 24-38% 参数) |
| 量化 | INT4 统一 | **混合精度** (FP16+INT8+INT4) |

## 核心特点

| 特性 | 描述 |
|------|------|
| 🦀 语言 | Rust (编译型，高性能，内存安全) |
| 🧮 架构 | LLaMA-style Decoder-Only Transformer |
| 📦 量化 | 混合精度 (Embed/Norm FP16, Q/K INT8, FFN INT4) |
| 🚫 无 LFS | 量化权重直接放进 Git 仓库 |
| ⚡ 推理 | 自研 GQA + RoPE + SwiGLU 前向传播 |
| 🌐 API | axum HTTP 服务 (SSE 流式) |

## 模型参数

```
架构:     RoPE + SwiGLU + RMSNorm + GQA + WeightTying + NoBias
d_model:  512       n_heads:      8 (Q) / 4 (KV, GQA)
n_layers: 6         d_ff:         1408 (8/3 × d_model, SwiGLU)
vocab:    32000     max_seq_len:  512
参数量:   ~38M (权重绑定后)
量化大小: ~12 MB (混合精度)
```

## 项目结构

```
src/
├── main.rs       # 入口：启动 HTTP 服务器
├── model.rs      # FishAI v2 模型 (RoPE+GQA+SwiGLU+RMSNorm+WeightTying)
├── quantize.rs   # 混合精度量化 (INT4/INT8/FP16)
├── tokenizer.rs  # BPE 分词器
└── api.rs        # HTTP API (axum)
```

## API 端点

```
POST /api/chat         - 对话生成
POST /api/chat/stream  - 流式对话生成
GET  /api/model        - 模型信息
GET  /health           - 健康检查
```

## 构建 & 运行

```bash
cargo build --release
mkdir -p weights
./target/release/fishai-engine
```

## 许可证

MIT License - FishLab-ai
