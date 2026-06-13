//! FishAI Engine - FishLab-ai 自研 GPT 推理引擎
//!
//! 完全自研的 Transformer 架构实现：
//! - GPT-2 风格 Decoder-Only Transformer
//! - 多头自注意力机制 (Multi-Head Self-Attention)
//! - 前馈神经网络 (Feed-Forward Network)
//! - 层归一化 (Layer Normalization)
//! - 4-bit 整数量化 (INT4 Quantization)
//! - BPE 分词器
//!
//! FishAI — 小而能干，源自 FishLab-ai

pub mod model;
pub mod quantize;
pub mod tokenizer;
pub mod api;
