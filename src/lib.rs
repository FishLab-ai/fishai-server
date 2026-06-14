//! FishAI Engine v3 — 小体积最聪明的自研 Transformer 推理引擎
//!
//! v3 核心升级 (对比 v2):
//! - KV Cache: 推理从 O(n²) 降到 O(n)
//! - 真流式 SSE: 逐 token 推送
//! - HuggingFace BPE 分词器: 真正的 BPE 合并
//! - Top-k/Top-p 采样: 可控文本生成
//! - 群组量化: 每 128 元素共享 scale/zp
//! - RoPE Scaling: 支持 Linear/YaRN 外推
//! - 多模型尺寸: S (~34M) / M (~400M) / L (~1.5B)
//! - 二进制权重格式: 替代 JSON
//!
//! FishAI v3 — 小体积, 最聪明

pub mod api;
pub mod bench;
pub mod model;
pub mod quantize;
pub mod tokenizer;
