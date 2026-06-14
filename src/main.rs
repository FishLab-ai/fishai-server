//! FishAI Engine v3 主入口
//!
//! 启动流程:
//! 1. 加载模型配置 (支持 S/M/L 预设)
//! 2. 加载权重 (自动检测 JSON/二进制格式)
//! 3. 加载分词器 (支持 HuggingFace JSON / byte-level 回退)
//! 4. 运行快速基准测试
//! 5. 启动 HTTP API 服务

mod api;
mod bench;
mod model;
mod quantize;
mod tokenizer;

use api::AppState;
use model::{GPTWeights, ModelConfig, RopeScalingType};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokenizer::FishAITokenizer;

const PORT: u16 = 3031;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    println!("╔══════════════════════════════════════════════════╗");
    println!("║       🐟 FishAI Engine v3.0.0                  ║");
    println!("║   小体积最聪明 — FishLab-ai 自研 (Rust)         ║");
    println!("║   KV Cache + 真流式 + BPE + Top-k/p + RoPE缩放 ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    // ──── 加载模型配置 ────
    let model_size = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "small".to_string());
    let config = ModelConfig::from_name(&model_size);

    println!("[架构] RoPE + SwiGLU + RMSNorm + GQA + WeightTying + NoBias");
    println!("[配置] 模型: {} (~{}M 参数)", config.model_name(), config.total_params() / 1_000_000);
    println!("[配置] 混合精度量化后: {:.1} MB", config.quantized_size_mb());
    println!("[配置] GQA: {} Q heads × {} KV heads (group_size={})",
        config.n_heads, config.n_kv_heads, config.n_groups());
    println!("[配置] SwiGLU d_ff: {} (8/3 × d_model)", config.d_ff);
    println!("[配置] 上下文长度: {}", config.max_seq_len);
    println!("[配置] 词汇表大小: {}", config.vocab_size);
    println!("[配置] 权重绑定: {}", if config.weight_tying { "是" } else { "否" });
    println!("[配置] RoPE 缩放: factor={}, type={:?}",
        config.rope_scaling_factor, config.rope_scaling_type);
    println!();

    // ──── 加载权重 ────
    let weights = load_weights(&config);
    println!();

    // ──── 加载分词器 ────
    let tokenizer = load_tokenizer();
    println!();

    // ──── 快速基准测试 ────
    if let Some(ref w) = weights {
        println!("[基准] 运行快速基准测试...");
        let report = bench::quick_bench(w, &tokenizer);
        println!("[基准] TTFT: {:.2} ms, 吞吐量: {:.2} tokens/s",
            report.ttft_ms, report.tokens_per_second);
        println!();
    }

    // ──── 启动服务 ────
    let state = Arc::new(Mutex::new(AppState {
        weights,
        tokenizer,
        config,
    }));

    let app = api::create_router(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], PORT));
    println!("[服务] 🐟 FishAI Engine v3 启动于 http://0.0.0.0:{}", PORT);
    println!("[服务] API 端点:");
    println!("  POST /api/chat         - 对话 (完整返回)");
    println!("  POST /api/chat/stream  - 流式对话 (SSE 逐 token 推送)");
    println!("  GET  /api/model        - 模型信息");
    println!("  GET  /health           - 健康检查");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// 加载模型权重 (自动检测格式)
fn load_weights(config: &ModelConfig) -> Option<GPTWeights> {
    // 尝试二进制格式 (优先，更高效)
    let bin_paths = ["weights/model.fishai", "weights/model.bin", "weights/model_q4.bin"];
    for path in bin_paths {
        match GPTWeights::load_from_binary(path) {
            Ok(w) => {
                println!("[加载] ✅ 从二进制文件加载权重: {}", path);
                return Some(w);
            }
            Err(_) => continue,
        }
    }

    // 尝试 JSON 格式
    let json_paths = ["weights/model_q4.json", "weights/model.json"];
    for path in json_paths {
        match GPTWeights::load_from_json(path) {
            Ok(w) => {
                println!("[加载] ✅ 从 JSON 文件加载权重: {}", path);
                return Some(w);
            }
            Err(_) => continue,
        }
    }

    // 通用加载
    match GPTWeights::load_from_file("weights/model") {
        Ok(w) => {
            println!("[加载] ✅ 自动检测格式加载成功");
            Some(w)
        }
        Err(_) => {
            println!("[加载] ⚠️  未找到预训练权重，使用随机初始化 (demo 模式)");
            Some(GPTWeights::random_init(config))
        }
    }
}

/// 加载分词器 (支持 HuggingFace JSON / byte-level 回退)
fn load_tokenizer() -> FishAITokenizer {
    // 尝试 HuggingFace tokenizer JSON
    let tokenizer_paths = [
        "weights/tokenizer.json",
        "weights/tokenizer.model",
        "tokenizer.json",
    ];

    for path in tokenizer_paths {
        if std::path::Path::new(path).exists() {
            match FishAITokenizer::from_file(path) {
                Ok(t) => {
                    println!("[分词器] ✅ 从文件加载: {} (vocab_size={}, backend={})",
                        path, t.vocab_size, t.backend_name());
                    return t;
                }
                Err(_) => continue,
            }
        }
    }

    // 回退到 byte-level
    let tokenizer = FishAITokenizer::new_byte_fallback(32000);
    println!("[分词器] ⚠️  未找到分词器文件，使用 byte-level 回退 (vocab_size={})", tokenizer.vocab_size);
    tokenizer
}
