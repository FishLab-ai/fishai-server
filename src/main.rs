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

    println!("╔══════════════════════════════════════╗");
    println!("║       🐟 FishAI Engine v0.1.0        ║");
    println!("║   FishLab-ai 自研 GPT 推理引擎 (Rust) ║");
    println!("╚══════════════════════════════════════╝");
    println!();

    let config = ModelConfig::default();
    println!("[配置] 模型参数量: ~{}M", config.total_params() / 1_000_000);
    println!("[配置] 量化后大小: {:.1} MB", config.quantized_size_mb());
    println!("[配置] 量化方案: INT4 Per-Channel");
    println!("[配置] 上下文长度: {}", config.max_seq_len);
    println!("[配置] 词汇表大小: {}", config.vocab_size);
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
    println!("[服务] 🐟 FishAI Engine 启动于 http://0.0.0.0:{}", PORT);
    println!("[服务] API 端点:");
    println!("  POST /api/chat         - 对话");
    println!("  POST /api/chat/stream  - 流式对话");
    println!("  GET  /api/model        - 模型信息");
    println!("  GET  /health           - 健康检查");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
