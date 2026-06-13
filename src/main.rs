mod api;
mod model;
mod quantize;
mod tokenizer;

use api::AppState;
use model::{GPTWeights, ModelConfig};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokenizer::BPETokenizer;

const PORT: u16 = 3031;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    println!("╔══════════════════════════════════════════════╗");
    println!("║       🐟 FishAI Engine v2.0.0               ║");
    println!("║   小体积最聪明 — FishLab-ai 自研 (Rust)      ║");
    println!("║   RoPE + SwiGLU + RMSNorm + GQA + WeightTie ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();

    let config = ModelConfig::default();
    println!("[v2 架构] RoPE + SwiGLU + RMSNorm + GQA + WeightTying + NoBias");
    println!("[配置] 模型参数量: ~{}M", config.total_params() / 1_000_000);
    println!("[配置] 混合精度量化后: {:.1} MB", config.quantized_size_mb());
    println!("[配置] GQA: {} Q heads × {} KV heads (group_size={})",
        config.n_heads, config.n_kv_heads, config.n_groups());
    println!("[配置] SwiGLU d_ff: {} (8/3 × d_model)", config.d_ff);
    println!("[配置] 上下文长度: {}", config.max_seq_len);
    println!("[配置] 词汇表大小: {}", config.vocab_size);
    println!("[配置] 权重绑定: {}", if config.weight_tying { "是" } else { "否" });
    println!();

    let weights = match GPTWeights::load_from_file("weights/model_q4.json") {
        Ok(w) => {
            println!("[加载] ✅ 预训练权重加载成功");
            Some(w)
        }
        Err(_) => {
            println!("[加载] ⚠️  未找到预训练权重，使用随机初始化 (demo 模式)");
            Some(GPTWeights::random_init(&config))
        }
    };

    let tokenizer = match BPETokenizer::load_from_file("weights/tokenizer.json") {
        Ok(t) => {
            println!("[加载] ✅ 分词器加载成功");
            t
        }
        Err(_) => {
            println!("[加载] ⚠️  未找到分词器，使用基础 byte-level 分词器");
            BPETokenizer::new()
        }
    };

    println!();

    let state = Arc::new(Mutex::new(AppState {
        weights,
        tokenizer,
        config,
    }));

    let app = api::create_router(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], PORT));
    println!("[服务] 🐟 FishAI Engine v2 启动于 http://0.0.0.0:{}", PORT);
    println!("[服务] API 端点:");
    println!("  POST /api/chat         - 对话");
    println!("  POST /api/chat/stream  - 流式对话");
    println!("  GET  /api/model        - 模型信息");
    println!("  GET  /health           - 健康检查");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
