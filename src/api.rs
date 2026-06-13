//! HTTP API 服务器 (v2)

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

use crate::model::{GPTWeights, ModelConfig};
use crate::tokenizer::BPETokenizer;

/// 应用状态
pub struct AppState {
    pub weights: Option<GPTWeights>,
    pub tokenizer: BPETokenizer,
    pub config: ModelConfig,
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub message: String,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub conversation_id: Option<String>,
}

fn default_temperature() -> f32 { 0.7 }
fn default_max_tokens() -> usize { 128 }

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub reply: String,
    pub tokens_generated: usize,
    pub model: String,
}

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
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model_loaded: bool,
}

pub fn create_router(state: Arc<Mutex<AppState>>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/model", get(model_info))
        .route("/api/chat", post(chat))
        .route("/api/chat/stream", post(chat_stream))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health(
    State(state): State<Arc<Mutex<AppState>>>,
) -> Json<HealthResponse> {
    let state = state.lock().await;
    Json(HealthResponse {
        status: "ok".to_string(),
        model_loaded: state.weights.is_some(),
    })
}

async fn model_info(
    State(state): State<Arc<Mutex<AppState>>>,
) -> Json<ModelInfo> {
    let state = state.lock().await;
    Json(ModelInfo {
        name: "FishAI-v2".to_string(),
        version: "2.0.0".to_string(),
        architecture: "LLaMA-style (RoPE+SwiGLU+RMSNorm+GQA+WeightTie+NoBias)".to_string(),
        params: format!("~{}M", state.config.total_params() / 1_000_000),
        quantized_size: format!("{:.1} MB", state.config.quantized_size_mb()),
        vocab_size: state.config.vocab_size,
        max_seq_len: state.config.max_seq_len,
        quantization: "Mixed-Precision (FP16+INT4)".to_string(),
        features: vec![
            "RoPE".into(),
            "SwiGLU".into(),
            "RMSNorm".into(),
            "GQA".into(),
            "WeightTying".into(),
            "NoBias".into(),
            "MixedPrecision".into(),
        ],
    })
}

async fn chat(
    State(state): State<Arc<Mutex<AppState>>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, StatusCode> {
    let state = state.lock().await;

    if state.weights.is_none() {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    let weights = state.weights.as_ref().unwrap();
    let tokens = state.tokenizer.encode(&req.message);
    let output_tokens = crate::model::generate(
        &tokens,
        weights,
        req.max_tokens.min(512),
        req.temperature,
    );
    let new_tokens = &output_tokens[tokens.len()..];
    let reply = state.tokenizer.decode(new_tokens);

    Ok(Json(ChatResponse {
        reply,
        tokens_generated: new_tokens.len(),
        model: "FishAI-v2".to_string(),
    }))
}

async fn chat_stream(
    State(state): State<Arc<Mutex<AppState>>>,
    Json(req): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let state = state.lock().await;

    if let Some(weights) = &state.weights {
        let tokens = state.tokenizer.encode(&req.message);
        let output_tokens = crate::model::generate(
            &tokens,
            weights,
            req.max_tokens.min(512),
            req.temperature,
        );
        let new_tokens = &output_tokens[tokens.len()..];
        let reply = state.tokenizer.decode(new_tokens);

        Json(ChatResponse {
            reply,
            tokens_generated: new_tokens.len(),
            model: "FishAI-v2".to_string(),
        })
    } else {
        Json(ChatResponse {
            reply: "模型未加载".to_string(),
            tokens_generated: 0,
            model: "FishAI-v2".to_string(),
        })
    }
}
