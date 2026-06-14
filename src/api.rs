//! FishAI v3 HTTP API 服务器 — 真流式推送
//!
//! v3 核心升级 (对比 v2):
//! 1. SSE (Server-Sent Events) 真流式推送 — 每 token 实时发送
//! 2. top_k / top_p 参数 — 可控采样策略
//! 3. model_size 参数 — 动态选择 S/M/L 配置
//! 4. tokio channel — 生成线程 → SSE 通道实时传输
//!
//! API 端点:
//! - POST /api/chat         — 对话 (完整返回)
//! - POST /api/chat/stream  — 流式对话 (SSE 逐 token 推送)
//! - GET  /api/model        — 模型信息
//! - GET  /health           — 健康检查

use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::Stream;
use tower_http::cors::CorsLayer;

use crate::model::{GPTWeights, ModelConfig};
use crate::tokenizer::FishAITokenizer;

// ═══════════════════════ 应用状态 ═══════════════════════

/// 应用全局状态
pub struct AppState {
    /// 模型权重 (可能未加载)
    pub weights: Option<GPTWeights>,
    /// 分词器
    pub tokenizer: FishAITokenizer,
    /// 模型配置
    pub config: ModelConfig,
}

// ═══════════════════════ 请求/响应结构 ═══════════════════════

/// 对话请求
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    /// 用户消息
    pub message: String,
    /// 采样温度 (0.01 - 2.0, 默认 0.7)
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// 最大生成 token 数 (1 - 2048, 默认 128)
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Top-k 采样 (0 = 不限制, 默认 50)
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Top-p (Nucleus) 采样 (0.0 - 1.0, 默认 0.9)
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// 模型尺寸 (small/medium/large, 默认 small)
    #[serde(default = "default_model_size")]
    pub model_size: String,
    /// 会话 ID (可选)
    #[serde(default)]
    pub conversation_id: Option<String>,
}

fn default_temperature() -> f32 { 0.7 }
fn default_max_tokens() -> usize { 128 }
fn default_top_k() -> usize { 50 }
fn default_top_p() -> f32 { 0.9 }
fn default_model_size() -> String { "small".to_string() }

/// 对话响应
#[derive(Debug, Serialize)]
pub struct ChatResponse {
    /// 生成的回复文本
    pub reply: String,
    /// 生成的 token 数量
    pub tokens_generated: usize,
    /// 模型名称
    pub model: String,
}

/// SSE 流式 token 事件
#[derive(Debug, Serialize)]
pub struct StreamTokenEvent {
    /// token ID
    pub token_id: usize,
    /// token 文本
    pub text: String,
    /// 是否结束
    pub finished: bool,
}

/// 模型信息
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub architecture: String,
    pub params: String,
    pub quantized_size: String,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub quantization: String,
    pub features: Vec<String>,
    pub kv_cache: String,
    pub sampling: String,
}

/// 健康检查响应
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model_loaded: bool,
    pub version: String,
}

// ═══════════════════════ 路由 ═══════════════════════

/// 创建 API 路由
pub fn create_router(state: Arc<Mutex<AppState>>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/model", get(model_info))
        .route("/api/chat", post(chat))
        .route("/api/chat/stream", post(chat_stream))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

// ═══════════════════════ 处理函数 ═══════════════════════

/// 健康检查
async fn health(
    State(state): State<Arc<Mutex<AppState>>>,
) -> Json<HealthResponse> {
    let state = state.lock().await;
    Json(HealthResponse {
        status: "ok".to_string(),
        model_loaded: state.weights.is_some(),
        version: "3.0.0".to_string(),
    })
}

/// 模型信息
async fn model_info(
    State(state): State<Arc<Mutex<AppState>>>,
) -> Json<ModelInfo> {
    let state = state.lock().await;
    Json(ModelInfo {
        name: format!("FishAI-v3 ({})", state.config.model_name()),
        version: "3.0.0".to_string(),
        architecture: "LLaMA-style (RoPE+SwiGLU+RMSNorm+GQA+WeightTie+NoBias)".to_string(),
        params: format!("~{}M", state.config.total_params() / 1_000_000),
        quantized_size: format!("{:.1} MB", state.config.quantized_size_mb()),
        vocab_size: state.config.vocab_size,
        max_seq_len: state.config.max_seq_len,
        quantization: "Mixed-Precision (FP16+INT4+INT8+GroupQuant)".to_string(),
        features: vec![
            "RoPE".into(),
            "SwiGLU".into(),
            "RMSNorm".into(),
            "GQA".into(),
            "WeightTying".into(),
            "NoBias".into(),
            "MixedPrecision".into(),
            "KVCache".into(),
            "TopKTopP".into(),
            "RoPEScaling".into(),
            "GroupQuantization".into(),
        ],
        kv_cache: "Per-layer KV Cache (O(n) generation)".into(),
        sampling: "Top-k + Top-p (Nucleus) + Temperature".into(),
    })
}

/// 普通对话 (完整返回)
async fn chat(
    State(state): State<Arc<Mutex<AppState>>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, StatusCode> {
    let state = state.lock().await;

    if state.weights.is_none() {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    let weights = state.weights.as_ref().unwrap();
    let tokenizer = &state.tokenizer;

    // 编码输入
    let tokens = tokenizer.encode(&req.message);
    let max_tokens = req.max_tokens.min(2048);

    // 使用 KV Cache 生成
    let output_tokens = crate::model::generate_with_cache(
        &tokens,
        weights,
        max_tokens,
        req.temperature,
        req.top_k,
        req.top_p,
    );

    // 解码新生成的 token
    let new_tokens = &output_tokens[tokens.len()..];
    let reply = tokenizer.decode(new_tokens);

    Ok(Json(ChatResponse {
        reply,
        tokens_generated: new_tokens.len(),
        model: format!("FishAI-v3 ({})", weights.config.model_name()),
    }))
}

/// 流式对话 (SSE 逐 token 推送)
/// 使用 tokio channel 实现真正的流式传输:
/// 生成线程 → unbounded channel → SSE stream → 客户端
async fn chat_stream(
    State(state): State<Arc<Mutex<AppState>>>,
    Json(req): Json<ChatRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // 创建通道 — 发送 Result<Event, Infallible>
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();

    // 获取权重和分词器的克隆
    let state = state.lock().await;
    let weights = state.weights.clone();
    let tokenizer = state.tokenizer.clone();
    let max_tokens = req.max_tokens.min(2048);
    let temperature = req.temperature;
    let top_k = req.top_k;
    let top_p = req.top_p;
    drop(state);

    // 在独立线程中生成 token
    tokio::spawn(async move {
        if weights.is_none() {
            let event_data = serde_json::to_string(&StreamTokenEvent {
                token_id: 0,
                text: "[模型未加载]".to_string(),
                finished: true,
            }).unwrap_or_default();
            let _ = tx.send(Ok(Event::default().data(event_data)));
            return;
        }

        let weights = weights.unwrap();
        let tokens = tokenizer.encode(&req.message);

        // 使用流式生成回调
        crate::model::generate_streaming(
            &tokens,
            &weights,
            max_tokens,
            temperature,
            top_k,
            top_p,
            |token_id| {
                let text = tokenizer.decode_token(token_id);
                let event = StreamTokenEvent {
                    token_id,
                    text,
                    finished: false,
                };
                let data = serde_json::to_string(&event).unwrap_or_default();
                let _ = tx.send(Ok(Event::default().data(data)));
            },
        );

        // 发送结束事件
        let done_event = StreamTokenEvent {
            token_id: 0,
            text: String::new(),
            finished: true,
        };
        let done_data = serde_json::to_string(&done_event).unwrap_or_default();
        let _ = tx.send(Ok(Event::default().data(done_data).event("done")));
    });

    // 将 channel 转换为 SSE 流
    let stream = UnboundedReceiverStream::new(rx);
    Sse::new(stream).keep_alive(KeepAlive::default())
}

// ═══════════════════════ 测试 ═══════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_request_defaults() {
        let json = r#"{"message": "hello"}"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.message, "hello");
        assert!((req.temperature - 0.7).abs() < 0.01);
        assert_eq!(req.max_tokens, 128);
        assert_eq!(req.top_k, 50);
        assert!((req.top_p - 0.9).abs() < 0.01);
        assert_eq!(req.model_size, "small");
    }

    #[test]
    fn test_chat_request_custom() {
        let json = r#"{
            "message": "test",
            "temperature": 0.5,
            "max_tokens": 256,
            "top_k": 10,
            "top_p": 0.8,
            "model_size": "medium"
        }"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert!((req.temperature - 0.5).abs() < 0.01);
        assert_eq!(req.max_tokens, 256);
        assert_eq!(req.top_k, 10);
        assert!((req.top_p - 0.8).abs() < 0.01);
        assert_eq!(req.model_size, "medium");
    }

    #[test]
    fn test_stream_token_event_serialization() {
        let event = StreamTokenEvent {
            token_id: 42,
            text: "你好".to_string(),
            finished: false,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("42"));
        assert!(json.contains("你好"));
    }

    #[test]
    fn test_model_info_structure() {
        let config = ModelConfig::small();
        let info = ModelInfo {
            name: format!("FishAI-v3 ({})", config.model_name()),
            version: "3.0.0".to_string(),
            architecture: "LLaMA-style".to_string(),
            params: format!("~{}M", config.total_params() / 1_000_000),
            quantized_size: format!("{:.1} MB", config.quantized_size_mb()),
            vocab_size: config.vocab_size,
            max_seq_len: config.max_seq_len,
            quantization: "Mixed-Precision".to_string(),
            features: vec!["KVCache".into()],
            kv_cache: "Per-layer".into(),
            sampling: "Top-k+Top-p".into(),
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("FishAI-S"));
    }
}
