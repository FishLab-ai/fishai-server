//! FishAI Engine v2 — 小体积最聪明的自研 Transformer
//!
//! v2 架构升级 (对标 LLaMA/Phi):
//! - RoPE (Rotary Position Embedding) — 零参数位置编码
//! - SwiGLU 激活函数 — 比 GELU 更强表达力
//! - RMSNorm — 比 LayerNorm 更快更简
//! - GQA (Grouped Query Attention) — 省 7% 参数 + 50% KV 缓存
//! - 权重绑定 — Embed 与 LM Head 共享, 省 24-38% 参数
//! - 无偏置 — 现代发现 bias 在 RMSNorm+Residual 下冗余
//! - 混合精度量化 — 关键层 FP16, 其余 INT4
//!
//! FishAI v2 — 小体积, 最聪明

pub mod model;
pub mod quantize;
pub mod tokenizer;
pub mod api;
