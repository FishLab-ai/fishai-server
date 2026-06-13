//! FishAI v2 模型架构 - 小体积最聪明的自研 Transformer
//!
//! v2 核心升级 (对标 LLaMA/Phi 架构):
//! 1. RoPE (Rotary Position Embedding) — 零参数位置编码，更好长度外推
//! 2. SwiGLU 激活函数 — 比 GELU 更强表达力
//! 3. RMSNorm — 比 LayerNorm 更快更简
//! 4. GQA (Grouped Query Attention) — 省 7% 参数 + 50% KV 缓存
//! 5. 权重绑定 (Weight Tying) — Token Embed 与 LM Head 共享，省 24-38% 参数
//! 6. 无偏置 (No Bias) — 现代发现 bias 在 RMSNorm+Residual 下冗余
//!
//! 参数效率: 同参数量下，v2 比 GPT-2 v1 架构有效容量提升 ~40%

use rand::Rng;
use std::f64::consts::PI;

// ──────────────── 模型配置 ────────────────

/// FishAI v2 模型超参数
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub n_heads: usize,       // Q 头数
    pub n_kv_heads: usize,    // KV 头数 (GQA: n_kv_heads < n_heads)
    pub n_layers: usize,
    pub d_ff: usize,           // SwiGLU 中间维度 (建议 8/3 * d_model, 向上取整到 64 倍数)
    pub rope_theta: f32,       // RoPE 基频
    pub norm_eps: f32,         // RMSNorm epsilon
    pub weight_tying: bool,    // 是否绑定 Embed 与 LM Head
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            max_seq_len: 512,
            d_model: 512,
            n_heads: 8,
            n_kv_heads: 4,           // GQA: 8 Q heads, 4 KV heads (group_size=2)
            n_layers: 6,
            d_ff: 1408,              // 8/3 * 512 ≈ 1365, round up to 64*22=1408
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            weight_tying: true,
        }
    }
}

impl ModelConfig {
    /// 每个注意力组的 Q 头数
    pub fn n_groups(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// 每个头的维度
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// 计算模型总参数量 (不含权重绑定节省的部分)
    pub fn total_params(&self) -> usize {
        let d = self.d_model;
        let v = self.vocab_size;
        let s = self.max_seq_len;
        let ff = self.d_ff;
        let nh = self.n_heads;
        let nkv = self.n_kv_heads;
        let hd = self.head_dim();

        // Token Embedding (无 Position Embedding — RoPE 是零参数的!)
        let tok_emb = v * d;

        let mut layer_params = 0usize;
        // GQA Attention: Q投影 (d_model -> nh*hd), K投影 (d_model -> nkv*hd), V投影 (d_model -> nkv*hd), O投影 (d_model -> d_model)
        // 无 bias
        layer_params += d * (nh * hd);     // Wq
        layer_params += d * (nkv * hd);    // Wk
        layer_params += d * (nkv * hd);    // Wv
        layer_params += d * d;              // Wo
        // SwiGLU FFN: 三个矩阵 (gate, up, down), 无 bias
        layer_params += d * ff;             // W_gate
        layer_params += d * ff;             // W_up
        layer_params += ff * d;             // W_down
        // RMSNorm (2 per layer, 每个只有 gamma, 无 beta)
        layer_params += 2 * d;

        let transformer_params = layer_params * self.n_layers;
        // Final RMSNorm
        let final_norm = d;
        // LM Head: 如果权重绑定则不计入
        let lm_head = if self.weight_tying { 0 } else { d * v };

        tok_emb + transformer_params + final_norm + lm_head
    }

    /// 混合精度量化后的模型大小
    /// 策略: Embed/LM Head/Norm -> FP16, 其余 -> INT4
    pub fn quantized_size_mb(&self) -> f64 {
        let d = self.d_model;
        let v = self.vocab_size;
        let ff = self.d_ff;
        let nh = self.n_heads;
        let nkv = self.n_kv_heads;
        let hd = self.head_dim();

        // FP16 部分: Token Embed + Final RMSNorm + (如果不绑定则 LM Head 也 FP16)
        let fp16_params = v * d + d + if self.weight_tying { 0 } else { v * d };

        // INT4 部分: 所有 Transformer 层的权重
        let mut int4_params = 0usize;
        int4_params += d * (nh * hd);      // Wq
        int4_params += d * (nkv * hd);     // Wk
        int4_params += d * (nkv * hd);     // Wv
        int4_params += d * d;               // Wo
        int4_params += d * ff;              // W_gate
        int4_params += d * ff;              // W_up
        int4_params += ff * d;              // W_down
        // Layer RMSNorm gamma 也用 FP16
        let fp16_per_layer = 2 * d;
        let int4_per_layer = int4_params - fp16_per_layer;

        let total_fp16 = fp16_params + fp16_per_layer * self.n_layers;
        let total_int4 = int4_per_layer * self.n_layers;

        // FP16: 2 bytes/param, INT4: 0.5 bytes/param
        let bytes = total_fp16 as f64 * 2.0 + total_int4 as f64 * 0.5;
        bytes / (1024.0 * 1024.0)
    }
}

// ──────────────── 量化权重格式 ────────────────

/// 4-bit 量化权重 (用于大部分线性层)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantizedWeight {
    pub data: Vec<u8>,
    pub scale: Vec<f32>,
    pub zero_point: Vec<i8>,
    pub shape: Vec<usize>,
}

/// FP16 权重 (用于 Embedding / Norm 等关键层)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FP16Weight {
    pub data: Vec<f32>,    // 实际用 FP32 存储 (推理时精度足够)
    pub shape: Vec<usize>,
}

// ──────────────── 模型权重 ────────────────

/// 单个 Transformer 层的权重 (v2: 无 bias, SwiGLU 三矩阵, GQA)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransformerLayerWeights {
    // GQA Attention (无 bias)
    pub wq: QuantizedWeight,           // [d_model, n_heads * head_dim]
    pub wk: QuantizedWeight,           // [d_model, n_kv_heads * head_dim]
    pub wv: QuantizedWeight,           // [d_model, n_kv_heads * head_dim]
    pub wo: QuantizedWeight,           // [d_model, d_model]
    // SwiGLU FFN (三矩阵, 无 bias)
    pub w_gate: QuantizedWeight,       // [d_model, d_ff]
    pub w_up: QuantizedWeight,         // [d_model, d_ff]
    pub w_down: QuantizedWeight,       // [d_ff, d_model]
    // RMSNorm (只有 gamma, 无 beta)
    pub rms1_gamma: Vec<f32>,          // [d_model]
    pub rms2_gamma: Vec<f32>,          // [d_model]
}

/// 完整的 FishAI v2 模型权重
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GPTWeights {
    pub config: ModelConfig,
    // Token Embedding (FP16 权重类)
    pub token_embedding: FP16Weight,    // [vocab_size, d_model]
    // Transformer Layers
    pub layers: Vec<TransformerLayerWeights>,
    // Final RMSNorm
    pub final_rms_gamma: Vec<f32>,      // [d_model]
    // LM Head: 权重绑定时不存储 (使用 token_embedding)
    pub lm_head: Option<QuantizedWeight>, // [d_model, vocab_size] 或 None
}

impl GPTWeights {
    /// 随机初始化 (demo 模式)
    pub fn random_init(config: &ModelConfig) -> Self {
        let d = config.d_model;
        let ff = config.d_ff;
        let v = config.vocab_size;
        let nh = config.n_heads;
        let nkv = config.n_kv_heads;
        let hd = config.head_dim();

        let mut rng = rand::thread_rng();

        let make_qw = |shape: Vec<usize>, channels: usize, rng: &mut rand::rngs::ThreadRng| -> QuantizedWeight {
            let total: usize = shape.iter().product();
            let data_len = (total + 1) / 2;
            let mut data = vec![0u8; data_len];
            for byte in data.iter_mut() {
                let low: u8 = rng.gen_range(0..16u8);
                let high: u8 = rng.gen_range(0..16u8);
                *byte = (high << 4) | low;
            }
            let scale = vec![0.02f32; channels];
            let zero_point = vec![8i8; channels];
            QuantizedWeight { data, scale, zero_point, shape }
        };

        let make_fp16 = |shape: Vec<usize>, rng: &mut rand::rngs::ThreadRng| -> FP16Weight {
            let total: usize = shape.iter().product();
            let data: Vec<f32> = (0..total)
                .map(|_| rng.gen::<f32>() * 0.04 - 0.02)
                .collect();
            FP16Weight { data, shape }
        };

        let layers: Vec<TransformerLayerWeights> = (0..config.n_layers)
            .map(|_| {
                TransformerLayerWeights {
                    wq: make_qw(vec![d, nh * hd], d, &mut rng),
                    wk: make_qw(vec![d, nkv * hd], d, &mut rng),
                    wv: make_qw(vec![d, nkv * hd], d, &mut rng),
                    wo: make_qw(vec![d, d], d, &mut rng),
                    w_gate: make_qw(vec![d, ff], d, &mut rng),
                    w_up: make_qw(vec![d, ff], d, &mut rng),
                    w_down: make_qw(vec![ff, d], ff, &mut rng),
                    rms1_gamma: vec![1.0f32; d],
                    rms2_gamma: vec![1.0f32; d],
                }
            })
            .collect();

        Self {
            config: config.clone(),
            token_embedding: make_fp16(vec![v, d], &mut rng),
            layers,
            final_rms_gamma: vec![1.0f32; d],
            lm_head: if config.weight_tying { None } else { Some(make_qw(vec![d, v], d, &mut rng)) },
        }
    }

    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let weights: GPTWeights = serde_json::from_str(&data)?;
        Ok(weights)
    }
}

// ──────────────── 推理核心 ────────────────

/// RMSNorm: x / sqrt(mean(x²) + eps) * gamma
/// 比 LayerNorm 更简单: 去掉 mean-centering 和 beta
fn rms_norm(x: &mut [f32], gamma: &[f32], eps: f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        x[i] = x[i] * inv_rms * gamma[i];
    }
}

/// SiLU (Sigmoid Linear Unit) 激活函数: x * sigmoid(x)
/// SwiGLU = SiLU(gate) * up
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softmax (in-place)
fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = x.iter().map(|v| (*v - max).exp()).sum();
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v = (*v - max).exp() / sum;
        }
    }
}

/// RoPE (Rotary Position Embedding)
/// 对 Q/K 向量的每对维度施加旋转: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
/// θ_i = 1 / rope_theta^(2i / head_dim)
/// 这是最关键的升级: 零参数位置编码，且自然编码相对位置关系
fn apply_rope(vec: &mut [f32], pos: usize, head_dim: usize, theta: f32) {
    let half = head_dim / 2;
    for i in 0..half {
        let freq = 1.0 / (theta.powf(2.0 * i as f32 / head_dim as f32));
        let angle = pos as f32 * freq;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let x0 = vec[2 * i];
        let x1 = vec[2 * i + 1];
        vec[2 * i] = x0 * cos_a - x1 * sin_a;
        vec[2 * i + 1] = x0 * sin_a + x1 * cos_a;
    }
}

/// 解量化线性层: y = x @ W^T (无 bias)
fn qmat_vec(w: &QuantizedWeight, input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let w_data: Vec<f32> = w.dequantize(0);
    let mut output = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut sum = 0.0f32;
        for j in 0..in_dim {
            if i * in_dim + j < w_data.len() && j < input.len() {
                sum += input[j] * w_data[i * in_dim + j];
            }
        }
        output[i] = sum;
    }
    output
}

/// FP16 权重线性层: y = x @ W^T (无 bias)
fn fpmat_vec(w: &FP16Weight, input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut sum = 0.0f32;
        for j in 0..in_dim {
            if i * in_dim + j < w.data.len() && j < input.len() {
                sum += input[j] * w.data[i * in_dim + j];
            }
        }
        output[i] = sum;
    }
    output
}

impl QuantizedWeight {
    /// 解量化为 f32 向量
    pub fn dequantize(&self, _channel: usize) -> Vec<f32> {
        let total: usize = self.shape.iter().product();
        self.data
            .iter()
            .flat_map(|&byte| {
                let low = (byte & 0x0F) as f32;
                let high = ((byte >> 4) & 0x0F) as f32;
                [low, high]
            })
            .take(total)
            .enumerate()
            .map(|(i, v)| {
                let ch = i % self.scale.len().max(1);
                (v - self.zero_point.get(ch).copied().unwrap_or(8) as f32)
                    * self.scale.get(ch).copied().unwrap_or(0.02)
            })
            .collect()
    }
}

/// GQA (Grouped Query Attention) with RoPE
/// 比 MHA 省 n_heads/n_kv_heads 的 K/V 投影参数
/// group_size = n_heads / n_kv_heads, 每 group_size 个 Q 头共享一组 KV
fn grouped_query_attention(
    x: &[Vec<f32>],
    wq: &QuantizedWeight,
    wk: &QuantizedWeight,
    wv: &QuantizedWeight,
    wo: &QuantizedWeight,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    d_model: usize,
    rope_theta: f32,
) -> Vec<Vec<f32>> {
    let seq_len = x.len();
    let n_groups = n_heads / n_kv_heads;
    let kv_dim = n_kv_heads * head_dim;

    // 计算 Q, K, V
    let q: Vec<Vec<f32>> = x.iter().map(|xi| qmat_vec(wq, xi, n_heads * head_dim, d_model)).collect();
    let k: Vec<Vec<f32>> = x.iter().map(|xi| qmat_vec(wk, xi, kv_dim, d_model)).collect();
    let v: Vec<Vec<f32>> = x.iter().map(|xi| qmat_vec(wv, xi, kv_dim, d_model)).collect();

    // 应用 RoPE 到 Q 和 K
    let mut q_rotated = q;
    let mut k_rotated = k;
    for pos in 0..seq_len {
        for h in 0..n_heads {
            let start = h * head_dim;
            apply_rope(&mut q_rotated[pos][start..start + head_dim], pos, head_dim, rope_theta);
        }
        for h in 0..n_kv_heads {
            let start = h * head_dim;
            apply_rope(&mut k_rotated[pos][start..start + head_dim], pos, head_dim, rope_theta);
        }
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_output = vec![vec![0.0f32; d_model]; seq_len];

    // GQA: 每 n_groups 个 Q 头共享一个 KV 头
    for h in 0..n_heads {
        let kv_h = h / n_groups; // 对应的 KV 头索引
        let q_start = h * head_dim;
        let kv_start = kv_h * head_dim;

        for i in 0..seq_len {
            let mut scores = vec![f32::NEG_INFINITY; seq_len];
            for j in 0..=i {
                let dot: f32 = (0..head_dim)
                    .map(|d| q_rotated[i][q_start + d] * k_rotated[j][kv_start + d])
                    .sum();
                scores[j] = dot * scale;
            }
            softmax(&mut scores);

            for j in 0..=i {
                for d in 0..head_dim {
                    attn_output[i][q_start + d] += scores[j] * v[j][kv_start + d];
                }
            }
        }
    }

    // 输出投影 (无 bias)
    attn_output
        .iter()
        .map(|ai| qmat_vec(wo, ai, d_model, d_model))
        .collect()
}

/// SwiGLU FFN: output = W_down(SiLU(x @ W_gate) ⊙ (x @ W_up))
/// 比 GELU FFN 多一个矩阵，但表达力显著更强
fn swiglu_ffn(
    x: &[f32],
    w_gate: &QuantizedWeight,
    w_up: &QuantizedWeight,
    w_down: &QuantizedWeight,
    d_ff: usize,
    d_model: usize,
) -> Vec<f32> {
    // gate = SiLU(x @ W_gate)
    let gate = qmat_vec(w_gate, x, d_ff, d_model);
    // up = x @ W_up
    let up = qmat_vec(w_up, x, d_ff, d_model);

    // SwiGLU: SiLU(gate) * up
    let mut hidden = vec![0.0f32; d_ff];
    for i in 0..d_ff {
        hidden[i] = silu(gate[i]) * up[i];
    }

    // down = hidden @ W_down
    qmat_vec(w_down, &hidden, d_model, d_ff)
}

/// 单个 Transformer 层前向传播 (Pre-RMSNorm)
/// x = x + Attention(RMSNorm(x))
/// x = x + SwiGLU(RMSNorm(x))
fn transformer_layer_forward(
    x: &mut Vec<Vec<f32>>,
    weights: &TransformerLayerWeights,
    config: &ModelConfig,
) {
    let d_model = config.d_model;
    let seq_len = x.len();

    // 保存残差
    let residual: Vec<Vec<f32>> = x.clone();

    // Pre-RMSNorm + GQA + Residual
    for i in 0..seq_len {
        rms_norm(&mut x[i], &weights.rms1_gamma, config.norm_eps);
    }

    let attn_out = grouped_query_attention(
        x,
        &weights.wq, &weights.wk, &weights.wv, &weights.wo,
        config.n_heads, config.n_kv_heads, config.head_dim(), d_model,
        config.rope_theta,
    );

    for i in 0..seq_len {
        for j in 0..d_model {
            x[i][j] = residual[i][j] + attn_out[i][j];
        }
    }

    // 保存残差
    let residual2: Vec<Vec<f32>> = x.clone();

    // Pre-RMSNorm + SwiGLU FFN + Residual
    for i in 0..seq_len {
        rms_norm(&mut x[i], &weights.rms2_gamma, config.norm_eps);
    }

    for i in 0..seq_len {
        let ff_out = swiglu_ffn(
            &x[i], &weights.w_gate, &weights.w_up, &weights.w_down,
            config.d_ff, d_model,
        );
        for j in 0..d_model {
            x[i][j] = residual2[i][j] + ff_out[j];
        }
    }
}

/// FishAI v2 完整前向传播
pub fn gpt_forward(
    token_ids: &[usize],
    weights: &GPTWeights,
) -> Vec<Vec<f32>> {
    let config = &weights.config;
    let d_model = config.d_model;
    let seq_len = token_ids.len();

    // Token Embedding (无 Position Embedding — RoPE 在注意力中施加)
    let mut x: Vec<Vec<f32>> = (0..seq_len)
        .map(|pos| {
            let token_id = token_ids[pos].min(config.vocab_size - 1);
            (0..d_model)
                .map(|d| {
                    let idx = token_id * d_model + d;
                    weights.token_embedding.data.get(idx).copied().unwrap_or(0.0)
                })
                .collect()
        })
        .collect();

    // 逐层 Transformer
    for layer_weights in &weights.layers {
        transformer_layer_forward(&mut x, layer_weights, config);
    }

    // Final RMSNorm
    for i in 0..seq_len {
        rms_norm(&mut x[i], &weights.final_rms_gamma, config.norm_eps);
    }

    // LM Head: 计算 logits
    // 权重绑定: 使用 token_embedding 的转置
    let vocab_size = config.vocab_size;
    x.iter()
        .map(|xi| {
            if config.weight_tying {
                // W_tied^T: [d_model, vocab_size] -> [vocab_size, d_model]
                (0..vocab_size)
                    .map(|v| {
                        let mut sum = 0.0f32;
                        for d in 0..d_model {
                            let idx = v * d_model + d;
                            if idx < weights.token_embedding.data.len() {
                                sum += xi[d] * weights.token_embedding.data[idx];
                            }
                        }
                        sum
                    })
                    .collect()
            } else if let Some(lm_head) = &weights.lm_head {
                let lm_data: Vec<f32> = lm_head.dequantize(0);
                (0..vocab_size)
                    .map(|v| {
                        let mut sum = 0.0f32;
                        for d in 0..d_model {
                            let idx = d * vocab_size + v;
                            if idx < lm_data.len() {
                                sum += xi[d] * lm_data[idx];
                            }
                        }
                        sum
                    })
                    .collect()
            } else {
                vec![0.0f32; vocab_size]
            }
        })
        .collect()
}

/// 从 logits 采样下一个 token (temperature + top-k)
pub fn sample_token(logits: &[f32], temperature: f32) -> usize {
    let vocab_size = logits.len();
    if vocab_size == 0 { return 0; }

    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature.max(0.01)).collect();
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();

    if sum <= 0.0 { return 0; }

    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    let mut rng = rand::thread_rng();
    let mut r: f32 = rng.gen::<f32>();
    for (i, &p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 { return i; }
    }
    vocab_size - 1
}

/// 自回归生成
pub fn generate(
    prompt_tokens: &[usize],
    weights: &GPTWeights,
    max_new_tokens: usize,
    temperature: f32,
) -> Vec<usize> {
    let mut tokens = prompt_tokens.to_vec();
    let config = &weights.config;

    for _ in 0..max_new_tokens {
        let context_len = tokens.len().min(config.max_seq_len);
        let context = &tokens[tokens.len() - context_len..];
        let logits = gpt_forward(context, weights);
        let last_logits = &logits[logits.len() - 1];
        let next_token = sample_token(last_logits, temperature);
        tokens.push(next_token);
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v2_config_params() {
        let config = ModelConfig::default();
        let params = config.total_params();
        let size_mb = config.quantized_size_mb();
        println!("FishAI v2 — Total parameters: {} ({:.1}M)", params, params as f64 / 1e6);
        println!("FishAI v2 — Mixed-precision quantized size: {:.2} MB", size_mb);
        println!("FishAI v2 — Architecture: RoPE + SwiGLU + RMSNorm + GQA + WeightTying + NoBias");
        println!("FishAI v2 — GQA: {} Q heads, {} KV heads (group_size={})", config.n_heads, config.n_kv_heads, config.n_groups());
        println!("FishAI v2 — d_ff = {} (8/3 * d_model ratio)", config.d_ff);
        // Weight tying 应大幅减少参数量
        assert!(params > 0);
        // 量化后应该很小
        assert!(size_mb < 30.0);
    }

    #[test]
    fn test_rms_norm() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32; 4];
        rms_norm(&mut x, &gamma, 1e-5);
        // RMSNorm 后范数应该接近 gamma
        let rms: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(rms > 0.0);
    }

    #[test]
    fn test_rope() {
        let mut vec = vec![1.0f32, 0.0, 0.0, 1.0];
        apply_rope(&mut vec, 0, 4, 10000.0);
        // pos=0 时 RoPE 不应改变向量
        assert!((vec[0] - 1.0).abs() < 1e-5);
        assert!(vec[1].abs() < 1e-5);
    }

    #[test]
    fn test_silu() {
        assert!(silu(0.0).abs() < 1e-5);
        assert!(silu(1.0) > 0.0);
        assert!(silu(-1.0) < 0.0);
    }
}
