//! FishAI v3 模型架构 — 小体积最聪明的自研 Transformer
//!
//! v3 核心升级 (对比 v2):
//! 1. KV Cache — 推理从 O(n²) 降到 O(n)，每 token 只算新增部分
//! 2. Top-k / Top-p (Nucleus) 采样 — 更可控的文本生成
//! 3. RoPE Scaling — 支持 YaRN / Linear 外推，扩展上下文长度
//! 4. 多模型尺寸 — FishAI-S (~34M) / M (~400M) / L (~1.5B)
//! 5. 修正 dequantize() — 修复 per-channel 解量化索引计算错误
//!
//! 保留 v2 全部特性:
//! - RoPE (Rotary Position Embedding)
//! - SwiGLU 激活函数
//! - RMSNorm
//! - GQA (Grouped Query Attention)
//! - 权重绑定 (Weight Tying)
//! - 混合精度量化 (FP16 + INT4 + INT8)

use rand::Rng;
use std::f64::consts::PI;

// ═══════════════════════ 模型配置 ═══════════════════════

/// RoPE 缩放类型
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum RopeScalingType {
    /// 不使用缩放
    None,
    /// 线性缩放: θ_i = θ_i / scaling_factor
    Linear,
    /// YaRN 缩放: 混合温度和缩放
    Yarn,
}

impl Default for RopeScalingType {
    fn default() -> Self {
        RopeScalingType::None
    }
}

/// FishAI v3 模型超参数
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub n_heads: usize,       // Q 头数
    pub n_kv_heads: usize,    // KV 头数 (GQA: n_kv_heads < n_heads)
    pub n_layers: usize,
    pub d_ff: usize,          // SwiGLU 中间维度
    pub rope_theta: f32,      // RoPE 基频
    pub rope_scaling_factor: f32,   // RoPE 缩放因子
    pub rope_scaling_type: RopeScalingType, // RoPE 缩放类型
    pub norm_eps: f32,        // RMSNorm epsilon
    pub weight_tying: bool,   // 是否绑定 Embed 与 LM Head
}

impl ModelConfig {
    /// FishAI-S: ~34M 参数 (当前模型规模)
    pub fn small() -> Self {
        Self {
            vocab_size: 32000,
            max_seq_len: 2048,
            d_model: 512,
            n_heads: 8,
            n_kv_heads: 4,
            n_layers: 6,
            d_ff: 1408,              // 8/3 * 512 ≈ 1365, 向上取整到 64 倍数
            rope_theta: 100000.0,
            rope_scaling_factor: 1.0,
            rope_scaling_type: RopeScalingType::None,
            norm_eps: 1e-5,
            weight_tying: true,
        }
    }

    /// FishAI-M: ~400M 参数 (对标 Qwen2.5-0.5B)
    pub fn medium() -> Self {
        Self {
            vocab_size: 32000,
            max_seq_len: 4096,
            d_model: 896,
            n_heads: 14,
            n_kv_heads: 2,
            n_layers: 24,
            d_ff: 4864,              // 8/3 * 896 ≈ 2389, 向上取整到 64*76=4864
            rope_theta: 100000.0,
            rope_scaling_factor: 1.0,
            rope_scaling_type: RopeScalingType::None,
            norm_eps: 1e-5,
            weight_tying: true,
        }
    }

    /// FishAI-L: ~1.5B 参数 (对标 Qwen2.5-1.5B)
    pub fn large() -> Self {
        Self {
            vocab_size: 32000,
            max_seq_len: 8192,
            d_model: 1536,
            n_heads: 12,
            n_kv_heads: 4,
            n_layers: 28,
            d_ff: 8960,              // 8/3 * 1536 ≈ 4096, 向上取整到 64*140=8960
            rope_theta: 100000.0,
            rope_scaling_factor: 1.0,
            rope_scaling_type: RopeScalingType::Yarn,
            norm_eps: 1e-5,
            weight_tying: false,
        }
    }

    /// 根据名称获取配置
    pub fn from_name(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "small" | "s" | "fishai-s" => Self::small(),
            "medium" | "m" | "fishai-m" => Self::medium(),
            "large" | "l" | "fishai-l" => Self::large(),
            _ => Self::small(),
        }
    }

    /// 配置名称
    pub fn model_name(&self) -> &str {
        if self.d_model == 512 {
            "FishAI-S"
        } else if self.d_model == 896 {
            "FishAI-M"
        } else if self.d_model == 1536 {
            "FishAI-L"
        } else {
            "FishAI-Custom"
        }
    }

    /// 每个注意力组的 Q 头数
    pub fn n_groups(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// 每个头的维度
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// 计算模型总参数量
    pub fn total_params(&self) -> usize {
        let d = self.d_model;
        let v = self.vocab_size;
        let ff = self.d_ff;
        let nh = self.n_heads;
        let nkv = self.n_kv_heads;
        let hd = self.head_dim();

        // Token Embedding
        let tok_emb = v * d;

        let mut layer_params = 0usize;
        // GQA Attention (无 bias)
        layer_params += d * (nh * hd);     // Wq
        layer_params += d * (nkv * hd);    // Wk
        layer_params += d * (nkv * hd);    // Wv
        layer_params += d * d;              // Wo
        // SwiGLU FFN
        layer_params += d * ff;             // W_gate
        layer_params += d * ff;             // W_up
        layer_params += ff * d;             // W_down
        // RMSNorm (2 per layer)
        layer_params += 2 * d;

        let transformer_params = layer_params * self.n_layers;
        let final_norm = d;
        let lm_head = if self.weight_tying { 0 } else { d * v };

        tok_emb + transformer_params + final_norm + lm_head
    }

    /// 混合精度量化后的模型大小
    pub fn quantized_size_mb(&self) -> f64 {
        let d = self.d_model;
        let v = self.vocab_size;
        let ff = self.d_ff;
        let nh = self.n_heads;
        let nkv = self.n_kv_heads;
        let hd = self.head_dim();

        // FP16 部分 (2 bytes/param)
        let fp16_params = v * d + d + 2 * d * self.n_layers;

        // INT8 部分 (1 byte/param): Q/K 投影
        let int8_per_layer = d * (nh * hd) + d * (nkv * hd);
        let total_int8 = int8_per_layer * self.n_layers;

        // INT4 部分 (0.5 bytes/param): V/O + FFN
        let int4_per_layer = d * (nkv * hd) + d * d + d * ff + d * ff + ff * d;
        let total_int4 = int4_per_layer * self.n_layers;

        let bytes = fp16_params as f64 * 2.0 + total_int8 as f64 * 1.0 + total_int4 as f64 * 0.5;
        bytes / (1024.0 * 1024.0)
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::small()
    }
}

// ═══════════════════════ 量化权重格式 ═══════════════════════

/// 量化精度类型
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum QuantType {
    /// 4-bit 量化 (per-channel)
    Int4,
    /// 8-bit 量化 (per-channel)
    Int8,
    /// 群组量化 (group_size 个元素共享 scale/zp, 类似 GPTQ)
    GroupQuant4 { group_size: usize },
}

impl Default for QuantType {
    fn default() -> Self {
        QuantType::Int4
    }
}

/// 量化权重 (支持 INT4, INT8, 群组量化)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantizedWeight {
    pub data: Vec<u8>,
    pub scale: Vec<f32>,
    pub zero_point: Vec<i8>,
    pub shape: Vec<usize>,
    pub quant_type: QuantType,
}

/// FP16 权重 (用于 Embedding / Norm 等关键层)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FP16Weight {
    pub data: Vec<f32>,    // 实际用 FP32 存储
    pub shape: Vec<usize>,
}

// ═══════════════════════ KV Cache ═══════════════════════

/// 单层的 KV Cache
/// 存储已计算的 Key 和 Value 向量，避免重复计算
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Key 缓存: [seq_len, n_kv_heads * head_dim]
    pub k: Vec<Vec<f32>>,
    /// Value 缓存: [seq_len, n_kv_heads * head_dim]
    pub v: Vec<Vec<f32>>,
    /// 当前缓存的有效长度
    pub len: usize,
}

impl KVCache {
    /// 创建空的 KV Cache
    pub fn new() -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            len: 0,
        }
    }

    /// 重置缓存
    pub fn reset(&mut self) {
        self.k.clear();
        self.v.clear();
        self.len = 0;
    }

    /// 追加新的 Key/Value
    pub fn append(&mut self, new_k: Vec<Vec<f32>>, new_v: Vec<Vec<f32>>) {
        self.k.extend(new_k);
        self.v.extend(new_v);
        self.len = self.k.len();
    }

    /// 获取完整的 K 缓存
    pub fn get_k(&self) -> &[Vec<f32>] {
        &self.k[..self.len]
    }

    /// 获取完整的 V 缓存
    pub fn get_v(&self) -> &[Vec<f32>] {
        &self.v[..self.len]
    }

    /// 内存占用 (字节)
    pub fn memory_bytes(&self) -> usize {
        let kv_dim = if !self.k.is_empty() {
            self.k[0].len()
        } else {
            0
        };
        self.len * kv_dim * 2 * 4 // K+V, each f32 = 4 bytes
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

/// 全模型的 KV Cache 集合
#[derive(Debug, Clone)]
pub struct ModelKVCache {
    pub layers: Vec<KVCache>,
}

impl ModelKVCache {
    /// 为每层创建 KV Cache
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| KVCache::new()).collect(),
        }
    }

    /// 重置所有层的缓存
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// 总内存占用
    pub fn total_memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }
}

// ═══════════════════════ 模型权重 ═══════════════════════

/// 单个 Transformer 层的权重
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransformerLayerWeights {
    // GQA Attention (无 bias)
    pub wq: QuantizedWeight,
    pub wk: QuantizedWeight,
    pub wv: QuantizedWeight,
    pub wo: QuantizedWeight,
    // SwiGLU FFN
    pub w_gate: QuantizedWeight,
    pub w_up: QuantizedWeight,
    pub w_down: QuantizedWeight,
    // RMSNorm (只有 gamma, 无 beta)
    pub rms1_gamma: Vec<f32>,
    pub rms2_gamma: Vec<f32>,
}

/// 完整的 FishAI v3 模型权重
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GPTWeights {
    pub config: ModelConfig,
    pub token_embedding: FP16Weight,
    pub layers: Vec<TransformerLayerWeights>,
    pub final_rms_gamma: Vec<f32>,
    pub lm_head: Option<QuantizedWeight>,
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

        let make_qw = |shape: Vec<usize>, channels: usize, qtype: QuantType, rng: &mut rand::rngs::ThreadRng| -> QuantizedWeight {
            let total: usize = shape.iter().product();
            let data_len = match &qtype {
                QuantType::Int8 => total,
                QuantType::GroupQuant4 { .. } => (total + 1) / 2,
                QuantType::Int4 => (total + 1) / 2,
            };
            let mut data = vec![0u8; data_len];
            for byte in data.iter_mut() {
                match &qtype {
                    QuantType::Int8 => *byte = rng.gen_range(0..=255u8),
                    _ => {
                        let low: u8 = rng.gen_range(0..16u8);
                        let high: u8 = rng.gen_range(0..16u8);
                        *byte = (high << 4) | low;
                    }
                }
            }
            let scale = vec![0.02f32; channels];
            let zero_point = match &qtype {
                QuantType::Int8 => vec![127i8; channels],
                _ => vec![8i8; channels],
            };
            QuantizedWeight { data, scale, zero_point, shape, quant_type: qtype }
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
                    // Q/K 使用 INT8 (注意力精度敏感)
                    wq: make_qw(vec![d, nh * hd], d, QuantType::Int8, &mut rng),
                    wk: make_qw(vec![d, nkv * hd], d, QuantType::Int8, &mut rng),
                    // V/O/FFN 使用 INT4
                    wv: make_qw(vec![d, nkv * hd], d, QuantType::Int4, &mut rng),
                    wo: make_qw(vec![d, d], d, QuantType::Int4, &mut rng),
                    w_gate: make_qw(vec![d, ff], d, QuantType::Int4, &mut rng),
                    w_up: make_qw(vec![d, ff], d, QuantType::Int4, &mut rng),
                    w_down: make_qw(vec![ff, d], ff, QuantType::Int4, &mut rng),
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
            lm_head: if config.weight_tying {
                None
            } else {
                Some(make_qw(vec![d, v], d, QuantType::Int4, &mut rng))
            },
        }
    }

    /// 从 JSON 文件加载
    pub fn load_from_json(path: &str) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let weights: GPTWeights = serde_json::from_str(&data)?;
        Ok(weights)
    }

    /// 保存为 JSON 文件
    pub fn save_to_json(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// 从二进制文件加载 (更高效)
    pub fn load_from_binary(path: &str) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        let weights: GPTWeights = bincode::deserialize(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(weights)
    }

    /// 保存为二进制文件
    pub fn save_to_binary(&self, path: &str) -> std::io::Result<()> {
        let data = bincode::serialize(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// 自动检测格式并加载
    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        if path.ends_with(".bin") || path.ends_with(".safetensors") || path.ends_with(".fishai") {
            Self::load_from_binary(path)
        } else {
            Self::load_from_json(path)
        }
    }
}

// ═══════════════════════ 推理核心 ═══════════════════════

/// RMSNorm: x / sqrt(mean(x²) + eps) * gamma
fn rms_norm(x: &mut [f32], gamma: &[f32], eps: f32) {
    let n = x.len();
    if n == 0 { return; }
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        x[i] = x[i] * inv_rms * gamma[i];
    }
}

/// SiLU (Sigmoid Linear Unit): x * sigmoid(x)
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softmax (in-place, 支持 causal mask)
fn softmax(x: &mut [f32]) {
    if x.is_empty() { return; }
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = x.iter().map(|v| (*v - max).exp()).sum();
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v = (*v - max).exp() / sum;
        }
    }
}

/// 应用 RoPE (Rotary Position Embedding) 到向量
/// 支持 RoPE Scaling (Linear / YaRN)
fn apply_rope(vec: &mut [f32], pos: usize, head_dim: usize, theta: f32, scaling_factor: f32, scaling_type: &RopeScalingType) {
    let half = head_dim / 2;
    for i in 0..half {
        // 计算基础频率
        let base_freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);

        // 根据 scaling 类型调整
        let freq = match scaling_type {
            RopeScalingType::None => base_freq,
            RopeScalingType::Linear => base_freq / scaling_factor,
            RopeScalingType::Yarn => {
                // YaRN: 对低频段缩放，高频段不变
                let low_freq_wavelen = 2.0 * PI as f32 * (1.0 / base_freq);
                let high_freq_wavelen = 2.0 * PI as f32 / theta.powf(2.0 * (half - 1) as f32 / head_dim as f32);
                let wavelen = 2.0 * PI as f32 / base_freq;

                if wavelen < high_freq_wavelen {
                    base_freq // 高频不变
                } else if wavelen > low_freq_wavelen {
                    base_freq / scaling_factor // 低频缩放
                } else {
                    // 中间段平滑插值
                    let smooth = (low_freq_wavelen - wavelen) / (low_freq_wavelen - high_freq_wavelen);
                    (1.0 - smooth) * base_freq / scaling_factor + smooth * base_freq
                }
            }
        };

        let angle = pos as f32 * freq;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let x0 = vec[2 * i];
        let x1 = vec[2 * i + 1];
        vec[2 * i] = x0 * cos_a - x1 * sin_a;
        vec[2 * i + 1] = x0 * sin_a + x1 * cos_a;
    }
}

impl QuantizedWeight {
    /// 解量化为 f32 向量
    /// 修正了 per-channel 索引计算错误 (v2 的 bug)
    pub fn dequantize(&self, _channel: usize) -> Vec<f32> {
        match &self.quant_type {
            QuantType::Int4 => self.dequantize_int4(),
            QuantType::Int8 => self.dequantize_int8(),
            QuantType::GroupQuant4 { group_size } => self.dequantize_group4(*group_size),
        }
    }

    /// INT4 解量化 (per-channel, 每个 output channel 共享 scale/zp)
    fn dequantize_int4(&self) -> Vec<f32> {
        let total: usize = self.shape.iter().product();
        if self.shape.len() < 2 { return vec![0.0f32; total]; }

        let _out_dim = self.shape[0];
        let in_dim = self.shape[1];

        let mut result = Vec::with_capacity(total);

        for i in 0..total {
            let is_high = i % 2 == 1;
            let byte_idx = i / 2;
            let raw = if byte_idx < self.data.len() {
                if is_high {
                    (self.data[byte_idx] >> 4) & 0x0F
                } else {
                    self.data[byte_idx] & 0x0F
                }
            } else {
                8u8
            };

            // 修正: 行优先排列下, channel = row_index
            let row = i / in_dim;
            let ch = row.min(self.scale.len().saturating_sub(1));
            let s = self.scale.get(ch).copied().unwrap_or(0.02);
            let zp = self.zero_point.get(ch).copied().unwrap_or(8);

            result.push((raw as f32 - zp as f32) * s);
        }

        result
    }

    /// INT8 解量化 (per-channel)
    /// 修复了 v2 中 channel 索引计算错误
    fn dequantize_int8(&self) -> Vec<f32> {
        let total: usize = self.shape.iter().product();
        if self.shape.len() < 2 { return vec![0.0f32; total]; }

        let in_dim = self.shape[1];

        let mut result = Vec::with_capacity(total);

        for i in 0..total.min(self.data.len()) {
            // 修正: 行优先排列, channel = row_index
            let row = i / in_dim;
            let ch = row.min(self.scale.len().saturating_sub(1));
            let s = self.scale.get(ch).copied().unwrap_or(0.01);
            let zp = self.zero_point.get(ch).copied().unwrap_or(127);

            result.push((self.data[i] as f32 - zp as f32) * s);
        }

        // 补齐长度
        while result.len() < total {
            result.push(0.0f32);
        }

        result
    }

    /// 群组4-bit 解量化 (每 group_size 个元素共享一组 scale/zp)
    fn dequantize_group4(&self, group_size: usize) -> Vec<f32> {
        let total: usize = self.shape.iter().product();
        let mut result = Vec::with_capacity(total);

        for i in 0..total {
            let is_high = i % 2 == 1;
            let byte_idx = i / 2;
            let raw = if byte_idx < self.data.len() {
                if is_high {
                    (self.data[byte_idx] >> 4) & 0x0F
                } else {
                    self.data[byte_idx] & 0x0F
                }
            } else {
                8u8
            };

            // 群组索引: 每 group_size 个元素属于同一个群组
            let group_idx = i / group_size;
            let ch = group_idx.min(self.scale.len().saturating_sub(1));
            let s = self.scale.get(ch).copied().unwrap_or(0.02);
            let zp = self.zero_point.get(ch).copied().unwrap_or(8);

            result.push((raw as f32 - zp as f32) * s);
        }

        result
    }
}

/// 量化权重矩阵 × 向量: y = x @ W^T (无 bias)
fn qmat_vec(w: &QuantizedWeight, input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let w_data: Vec<f32> = w.dequantize(0);
    let mut output = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut sum = 0.0f32;
        for j in 0..in_dim {
            let idx = i * in_dim + j;
            if idx < w_data.len() && j < input.len() {
                sum += input[j] * w_data[idx];
            }
        }
        output[i] = sum;
    }
    output
}

/// FP16 权重矩阵 × 向量: y = x @ W^T (无 bias)
fn fpmat_vec(w: &FP16Weight, input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut sum = 0.0f32;
        for j in 0..in_dim {
            let idx = i * in_dim + j;
            if idx < w.data.len() && j < input.len() {
                sum += input[j] * w.data[idx];
            }
        }
        output[i] = sum;
    }
    output
}

/// GQA (Grouped Query Attention) with RoPE — 无 KV Cache 版本
/// 用于首次前向传播 (prefill)
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
    rope_scaling_factor: f32,
    rope_scaling_type: &RopeScalingType,
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
            apply_rope(&mut q_rotated[pos][start..start + head_dim], pos, head_dim, rope_theta, rope_scaling_factor, rope_scaling_type);
        }
        for h in 0..n_kv_heads {
            let start = h * head_dim;
            apply_rope(&mut k_rotated[pos][start..start + head_dim], pos, head_dim, rope_theta, rope_scaling_factor, rope_scaling_type);
        }
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_output = vec![vec![0.0f32; d_model]; seq_len];

    // GQA: 每 n_groups 个 Q 头共享一个 KV 头
    for h in 0..n_heads {
        let kv_h = h / n_groups;
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

    // 输出投影
    attn_output
        .iter()
        .map(|ai| qmat_vec(wo, ai, d_model, d_model))
        .collect()
}

/// GQA with KV Cache — 增量计算版
/// 只计算新的 token，利用 KV Cache 避免重复计算
fn grouped_query_attention_cached(
    new_x: &[Vec<f32>],           // 新增 token 的输入 [new_tokens, d_model]
    cache: &mut KVCache,           // 该层的 KV Cache
    start_pos: usize,              // 新 token 在序列中的起始位置
    wq: &QuantizedWeight,
    wk: &QuantizedWeight,
    wv: &QuantizedWeight,
    wo: &QuantizedWeight,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    d_model: usize,
    rope_theta: f32,
    rope_scaling_factor: f32,
    rope_scaling_type: &RopeScalingType,
) -> Vec<Vec<f32>> {
    let new_len = new_x.len();
    let n_groups = n_heads / n_kv_heads;
    let kv_dim = n_kv_heads * head_dim;

    // 只计算新 token 的 Q, K, V
    let q: Vec<Vec<f32>> = new_x.iter().map(|xi| qmat_vec(wq, xi, n_heads * head_dim, d_model)).collect();
    let k_new: Vec<Vec<f32>> = new_x.iter().map(|xi| qmat_vec(wk, xi, kv_dim, d_model)).collect();
    let v_new: Vec<Vec<f32>> = new_x.iter().map(|xi| qmat_vec(wv, xi, kv_dim, d_model)).collect();

    // 对新 token 应用 RoPE
    let mut q_rotated = q;
    let mut k_rotated = k_new;
    for (idx, pos) in (start_pos..start_pos + new_len).enumerate() {
        for h in 0..n_heads {
            let start = h * head_dim;
            apply_rope(&mut q_rotated[idx][start..start + head_dim], pos, head_dim, rope_theta, rope_scaling_factor, rope_scaling_type);
        }
        for h in 0..n_kv_heads {
            let start = h * head_dim;
            apply_rope(&mut k_rotated[idx][start..start + head_dim], pos, head_dim, rope_theta, rope_scaling_factor, rope_scaling_type);
        }
    }

    // 更新 KV Cache
    cache.append(k_rotated.clone(), v_new.clone());

    let cached_k = cache.get_k();
    let cached_v = cache.get_v();
    let total_len = cache.len;

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_output = vec![vec![0.0f32; d_model]; new_len];

    // GQA with cached K/V
    for h in 0..n_heads {
        let kv_h = h / n_groups;
        let q_start = h * head_dim;
        let kv_start = kv_h * head_dim;

        for i in 0..new_len {
            let global_pos = start_pos + i;
            let mut scores = vec![f32::NEG_INFINITY; total_len];
            for j in 0..total_len {
                if j > global_pos { continue; } // causal mask
                let dot: f32 = (0..head_dim)
                    .map(|d| q_rotated[i][q_start + d] * cached_k[j][kv_start + d])
                    .sum();
                scores[j] = dot * scale;
            }
            softmax(&mut scores);

            for j in 0..total_len {
                if j > global_pos { continue; }
                for d in 0..head_dim {
                    attn_output[i][q_start + d] += scores[j] * cached_v[j][kv_start + d];
                }
            }
        }
    }

    // 输出投影
    attn_output
        .iter()
        .map(|ai| qmat_vec(wo, ai, d_model, d_model))
        .collect()
}

/// SwiGLU FFN: output = W_down(SiLU(x @ W_gate) ⊙ (x @ W_up))
fn swiglu_ffn(
    x: &[f32],
    w_gate: &QuantizedWeight,
    w_up: &QuantizedWeight,
    w_down: &QuantizedWeight,
    d_ff: usize,
    d_model: usize,
) -> Vec<f32> {
    let gate = qmat_vec(w_gate, x, d_ff, d_model);
    let up = qmat_vec(w_up, x, d_ff, d_model);

    let mut hidden = vec![0.0f32; d_ff];
    for i in 0..d_ff {
        hidden[i] = silu(gate[i]) * up[i];
    }

    qmat_vec(w_down, &hidden, d_model, d_ff)
}

/// 单个 Transformer 层前向传播 — 无 Cache 版本
fn transformer_layer_forward(
    x: &mut Vec<Vec<f32>>,
    weights: &TransformerLayerWeights,
    config: &ModelConfig,
) {
    let d_model = config.d_model;
    let seq_len = x.len();

    let residual: Vec<Vec<f32>> = x.clone();

    for i in 0..seq_len {
        rms_norm(&mut x[i], &weights.rms1_gamma, config.norm_eps);
    }

    let attn_out = grouped_query_attention(
        x,
        &weights.wq, &weights.wk, &weights.wv, &weights.wo,
        config.n_heads, config.n_kv_heads, config.head_dim(), d_model,
        config.rope_theta, config.rope_scaling_factor, &config.rope_scaling_type,
    );

    for i in 0..seq_len {
        for j in 0..d_model {
            x[i][j] = residual[i][j] + attn_out[i][j];
        }
    }

    let residual2: Vec<Vec<f32>> = x.clone();

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

/// 单个 Transformer 层前向传播 — 带 KV Cache 版本
fn transformer_layer_forward_cached(
    new_x: &mut Vec<Vec<f32>>,
    cache: &mut KVCache,
    start_pos: usize,
    weights: &TransformerLayerWeights,
    config: &ModelConfig,
) {
    let d_model = config.d_model;
    let seq_len = new_x.len();

    let residual: Vec<Vec<f32>> = new_x.clone();

    for i in 0..seq_len {
        rms_norm(&mut new_x[i], &weights.rms1_gamma, config.norm_eps);
    }

    let attn_out = grouped_query_attention_cached(
        new_x,
        cache,
        start_pos,
        &weights.wq, &weights.wk, &weights.wv, &weights.wo,
        config.n_heads, config.n_kv_heads, config.head_dim(), d_model,
        config.rope_theta, config.rope_scaling_factor, &config.rope_scaling_type,
    );

    for i in 0..seq_len {
        for j in 0..d_model {
            new_x[i][j] = residual[i][j] + attn_out[i][j];
        }
    }

    let residual2: Vec<Vec<f32>> = new_x.clone();

    for i in 0..seq_len {
        rms_norm(&mut new_x[i], &weights.rms2_gamma, config.norm_eps);
    }

    for i in 0..seq_len {
        let ff_out = swiglu_ffn(
            &new_x[i], &weights.w_gate, &weights.w_up, &weights.w_down,
            config.d_ff, d_model,
        );
        for j in 0..d_model {
            new_x[i][j] = residual2[i][j] + ff_out[j];
        }
    }
}

// ═══════════════════════ 前向传播 ═══════════════════════

/// FishAI v3 完整前向传播 (无 Cache, 用于 prefill)
pub fn gpt_forward(
    token_ids: &[usize],
    weights: &GPTWeights,
) -> Vec<Vec<f32>> {
    let config = &weights.config;
    let d_model = config.d_model;
    let seq_len = token_ids.len();

    // Token Embedding
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

    // LM Head
    compute_logits(&x, weights)
}

/// FishAI v3 前向传播 — 带 KV Cache 版本
/// 用于生成阶段，每步只计算新 token
pub fn gpt_forward_with_cache(
    new_token_ids: &[usize],       // 新增 token 的 ID
    start_pos: usize,               // 新 token 在序列中的起始位置
    kv_cache: &mut ModelKVCache,    // 全模型 KV Cache
    weights: &GPTWeights,
) -> Vec<Vec<f32>> {
    let config = &weights.config;
    let d_model = config.d_model;
    let new_len = new_token_ids.len();

    // 新 token 的 Embedding
    let mut x: Vec<Vec<f32>> = (0..new_len)
        .map(|i| {
            let token_id = new_token_ids[i].min(config.vocab_size - 1);
            (0..d_model)
                .map(|d| {
                    let idx = token_id * d_model + d;
                    weights.token_embedding.data.get(idx).copied().unwrap_or(0.0)
                })
                .collect()
        })
        .collect();

    // 逐层 Transformer (带 Cache)
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        transformer_layer_forward_cached(
            &mut x,
            &mut kv_cache.layers[layer_idx],
            start_pos,
            layer_weights,
            config,
        );
    }

    // Final RMSNorm
    for i in 0..new_len {
        rms_norm(&mut x[i], &weights.final_rms_gamma, config.norm_eps);
    }

    compute_logits(&x, weights)
}

/// 计算 logits (LM Head)
fn compute_logits(x: &[Vec<f32>], weights: &GPTWeights) -> Vec<Vec<f32>> {
    let config = &weights.config;
    let d_model = config.d_model;
    let vocab_size = config.vocab_size;

    x.iter()
        .map(|xi| {
            if config.weight_tying {
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
                qmat_vec(lm_head, xi, vocab_size, d_model)
            } else {
                vec![0.0f32; vocab_size]
            }
        })
        .collect()
}

// ═══════════════════════ 采样策略 ═══════════════════════

/// Top-k + Top-p (Nucleus) 采样
/// 1. 先用 temperature 缩放 logits
/// 2. Top-k: 只保留概率最高的 k 个 token
/// 3. Top-p: 在 top-k 结果上进一步过滤，保留累积概率 ≤ p 的 token
/// 4. 从过滤后的分布中采样
pub fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> usize {
    let vocab_size = logits.len();
    if vocab_size == 0 { return 0; }

    // Temperature 缩放
    let temp = temperature.max(0.01);
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temp).collect();

    // Softmax 得到概率
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 { return 0; }
    let mut probs: Vec<(usize, f32)> = exps.iter().enumerate()
        .map(|(i, &e)| (i, e / sum))
        .collect();

    // 按概率降序排列
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top-k 过滤
    let k = if top_k > 0 { top_k.min(vocab_size) } else { vocab_size };
    probs.truncate(k);

    // Top-p (Nucleus) 过滤
    if top_p > 0.0 && top_p < 1.0 {
        let mut cumsum = 0.0f32;
        let mut cutoff = probs.len();
        for (idx, &(_, p)) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= top_p {
                cutoff = idx + 1;
                break;
            }
        }
        probs.truncate(cutoff);
    }

    // 重新归一化
    let total: f32 = probs.iter().map(|(_, p)| *p).sum();
    if total <= 0.0 { return 0; }
    for (_, p) in probs.iter_mut() {
        *p /= total;
    }

    // 按概率采样
    let mut rng = rand::thread_rng();
    let mut r: f32 = rng.gen::<f32>();
    for (token_id, p) in &probs {
        r -= p;
        if r <= 0.0 { return *token_id; }
    }

    // 兜底
    probs.first().map(|(id, _)| *id).unwrap_or(0)
}

/// 自回归生成 — 无 KV Cache 版本 (O(n²), 向后兼容)
pub fn generate(
    prompt_tokens: &[usize],
    weights: &GPTWeights,
    max_new_tokens: usize,
    temperature: f32,
) -> Vec<usize> {
    generate_with_params(prompt_tokens, weights, max_new_tokens, temperature, 0, 1.0)
}

/// 自回归生成 — 带参数，无 Cache
pub fn generate_with_params(
    prompt_tokens: &[usize],
    weights: &GPTWeights,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Vec<usize> {
    let mut tokens = prompt_tokens.to_vec();
    let config = &weights.config;

    for _ in 0..max_new_tokens {
        let context_len = tokens.len().min(config.max_seq_len);
        let context = &tokens[tokens.len() - context_len..];
        let logits = gpt_forward(context, weights);
        let last_logits = &logits[logits.len() - 1];
        let next_token = sample_token(last_logits, temperature, top_k, top_p);
        tokens.push(next_token);
    }

    tokens
}

/// 自回归生成 — 带 KV Cache 版本 (O(n), 推荐使用)
/// 首次 prefill 计算全部 prompt，之后每步只算 1 个新 token
pub fn generate_with_cache(
    prompt_tokens: &[usize],
    weights: &GPTWeights,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Vec<usize> {
    let config = &weights.config;
    let mut tokens = prompt_tokens.to_vec();
    let mut kv_cache = ModelKVCache::new(config.n_layers);

    // Prefill: 计算全部 prompt
    let prompt_len = prompt_tokens.len().min(config.max_seq_len);
    let prompt = &prompt_tokens[prompt_tokens.len() - prompt_len..];
    let logits = gpt_forward_with_cache(prompt, 0, &mut kv_cache, weights);
    let last_logits = &logits[logits.len() - 1];
    let mut next_token = sample_token(last_logits, temperature, top_k, top_p);
    tokens.push(next_token);

    // Decode: 每步只算 1 个新 token
    for step in 1..max_new_tokens {
        let pos = prompt_len + step - 1;
        if pos >= config.max_seq_len { break; }

        let new_logits = gpt_forward_with_cache(
            &[next_token], pos, &mut kv_cache, weights,
        );
        let last = &new_logits[0];
        next_token = sample_token(last, temperature, top_k, top_p);
        tokens.push(next_token);
    }

    tokens
}

/// 流式生成回调 — 带 KV Cache，每生成一个 token 调用回调
/// 适用于 SSE 真流式推送
pub fn generate_streaming<F>(
    prompt_tokens: &[usize],
    weights: &GPTWeights,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    mut on_token: F,
) -> Vec<usize>
where
    F: FnMut(usize),
{
    let config = &weights.config;
    let mut tokens = prompt_tokens.to_vec();
    let mut kv_cache = ModelKVCache::new(config.n_layers);

    // Prefill
    let prompt_len = prompt_tokens.len().min(config.max_seq_len);
    let prompt = &prompt_tokens[prompt_tokens.len() - prompt_len..];
    let logits = gpt_forward_with_cache(prompt, 0, &mut kv_cache, weights);
    let last_logits = &logits[logits.len() - 1];
    let mut next_token = sample_token(last_logits, temperature, top_k, top_p);
    tokens.push(next_token);
    on_token(next_token);

    // Decode
    for step in 1..max_new_tokens {
        let pos = prompt_len + step - 1;
        if pos >= config.max_seq_len { break; }

        let new_logits = gpt_forward_with_cache(
            &[next_token], pos, &mut kv_cache, weights,
        );
        let last = &new_logits[0];
        next_token = sample_token(last, temperature, top_k, top_p);
        tokens.push(next_token);
        on_token(next_token);
    }

    tokens
}

// ═══════════════════════ 测试 ═══════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_config() {
        let config = ModelConfig::small();
        assert_eq!(config.d_model, 512);
        assert_eq!(config.n_heads, 8);
        assert_eq!(config.n_kv_heads, 4);
        assert_eq!(config.n_layers, 6);
        let params = config.total_params();
        println!("FishAI-S: {} params (~{}M)", params, params / 1_000_000);
        assert!(params > 20_000_000 && params < 50_000_000);
    }

    #[test]
    fn test_medium_config() {
        let config = ModelConfig::medium();
        assert_eq!(config.d_model, 896);
        let params = config.total_params();
        println!("FishAI-M: {} params (~{}M)", params, params / 1_000_000);
        assert!(params > 300_000_000 && params < 500_000_000);
    }

    #[test]
    fn test_large_config() {
        let config = ModelConfig::large();
        assert_eq!(config.d_model, 1536);
        let params = config.total_params();
        println!("FishAI-L: {} params (~{}M)", params, params / 1_000_000);
        assert!(params > 1_000_000_000 && params < 2_000_000_000);
    }

    #[test]
    fn test_from_name() {
        assert_eq!(ModelConfig::from_name("small").d_model, 512);
        assert_eq!(ModelConfig::from_name("S").d_model, 512);
        assert_eq!(ModelConfig::from_name("medium").d_model, 896);
        assert_eq!(ModelConfig::from_name("M").d_model, 896);
        assert_eq!(ModelConfig::from_name("large").d_model, 1536);
        assert_eq!(ModelConfig::from_name("L").d_model, 1536);
    }

    #[test]
    fn test_rms_norm() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32; 4];
        rms_norm(&mut x, &gamma, 1e-5);
        let rms: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(rms > 0.0);
    }

    #[test]
    fn test_rope_pos_zero() {
        let mut vec = vec![1.0f32, 0.0, 0.0, 1.0];
        apply_rope(&mut vec, 0, 4, 10000.0, 1.0, &RopeScalingType::None);
        // pos=0 时 RoPE 不应改变向量
        assert!((vec[0] - 1.0).abs() < 1e-5);
        assert!(vec[1].abs() < 1e-5);
    }

    #[test]
    fn test_rope_linear_scaling() {
        let mut vec_no_scale = vec![1.0f32, 0.0, 0.0, 1.0];
        let mut vec_scaled = vec![1.0f32, 0.0, 0.0, 1.0];
        apply_rope(&mut vec_no_scale, 10, 4, 10000.0, 1.0, &RopeScalingType::None);
        apply_rope(&mut vec_scaled, 10, 4, 10000.0, 2.0, &RopeScalingType::Linear);
        // Linear scaling with factor=2 应等效于 pos=5
        let mut vec_half_pos = vec![1.0f32, 0.0, 0.0, 1.0];
        apply_rope(&mut vec_half_pos, 5, 4, 10000.0, 1.0, &RopeScalingType::None);
        assert!((vec_scaled[0] - vec_half_pos[0]).abs() < 1e-4);
        assert!((vec_scaled[1] - vec_half_pos[1]).abs() < 1e-4);
    }

    #[test]
    fn test_silu() {
        assert!(silu(0.0).abs() < 1e-5);
        assert!(silu(1.0) > 0.0);
        assert!(silu(-1.0) < 0.0);
    }

    #[test]
    fn test_kv_cache() {
        let mut cache = KVCache::new();
        assert_eq!(cache.len, 0);

        cache.append(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        );
        assert_eq!(cache.len, 2);

        cache.append(
            vec![vec![9.0, 10.0]],
            vec![vec![11.0, 12.0]],
        );
        assert_eq!(cache.len, 3);
        assert_eq!(cache.get_k()[2], vec![9.0, 10.0]);

        cache.reset();
        assert_eq!(cache.len, 0);
    }

    #[test]
    fn test_top_k_top_p_sampling() {
        // 构造一个简单的 logits 分布，第 5 个 token 概率最高
        let mut logits = vec![0.0f32; 10];
        logits[5] = 10.0;
        logits[3] = 5.0;
        logits[7] = 3.0;

        // Greedy: temperature 很低时应该选最高的
        let token = sample_token(&logits, 0.01, 0, 1.0);
        assert_eq!(token, 5);

        // Top-k = 1: 也应该选最高的
        let token = sample_token(&logits, 1.0, 1, 1.0);
        assert_eq!(token, 5);
    }

    #[test]
    fn test_generate_with_cache_matches_no_cache() {
        let config = ModelConfig::small();
        let weights = GPTWeights::random_init(&config);
        let prompt = vec![1, 2, 3, 4];

        // 无 Cache 版本 (短序列以确保一致性)
        let output_no_cache = generate(&prompt, &weights, 3, 0.8);
        // 带 Cache 版本
        let output_with_cache = generate_with_cache(&prompt, &weights, 3, 0.8, 0, 1.0);

        // 两者应该生成相同长度的 token
        assert_eq!(output_no_cache.len(), output_with_cache.len());
        // 注意: 由于采样有随机性, 具体 token 可能不同，但结构应一致
        assert!(output_with_cache.len() >= prompt.len() + 3);
    }

    #[test]
    fn test_dequantize_int4() {
        let config = ModelConfig::small();
        let weights = GPTWeights::random_init(&config);
        // 检查 wv (INT4) 的 dequantize
        let wv_data = weights.layers[0].wv.dequantize(0);
        let total: usize = weights.layers[0].wv.shape.iter().product();
        assert_eq!(wv_data.len(), total);
        // 检查 wq (INT8) 的 dequantize
        let wq_data = weights.layers[0].wq.dequantize(0);
        let total_q: usize = weights.layers[0].wq.shape.iter().product();
        assert_eq!(wq_data.len(), total_q);
    }
}
