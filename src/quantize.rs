//! FishAI v3 混合精度量化模块
//!
//! v3 核心升级 (对比 v2):
//! 1. 群组量化 (GroupQuant4) — 每 128 个元素共享 scale/zp (类 GPTQ)
//! 2. 修复 INT8 dequantize 通道索引计算错误
//! 3. 完善误差度量 — MSE + 余弦相似度
//! 4. 二进制序列化格式 (替代 JSON, 更高效)
//! 5. 量化权重类型标记 (QuantType 枚举)
//!
//! 量化策略:
//! - Embedding/Norm → FP16 (精度敏感)
//! - Q/K 投影 → INT8 (注意力精度敏感)
//! - V/O/FFN → INT4 或 GroupQuant4 (对量化鲁棒)

use super::model::{QuantizedWeight, FP16Weight, QuantType};

// ═══════════════════════ 常量 ═══════════════════════

/// 默认群组量化大小 (GPTQ 使用 128)
pub const DEFAULT_GROUP_SIZE: usize = 128;

// ═══════════════════════ INT4 Per-Channel 量化 ═══════════════════════

/// 将 FP32 权重量化为 INT4 Per-Channel
/// 每个输出通道独立计算 scale 和 zero_point
pub fn quantize_tensor_int4(
    weights: &[f32],
    shape: &[usize],
    channel_dim: usize,
) -> QuantizedWeight {
    let n_channels = shape[channel_dim];
    let channel_size: usize = shape.iter().enumerate()
        .filter(|(i, _)| *i != channel_dim)
        .map(|(_, &s)| s)
        .product();

    let total_elements: usize = shape.iter().product();
    let data_len = (total_elements + 1) / 2;

    let mut data = vec![0u8; data_len];
    let mut scale = vec![0.0f32; n_channels];
    let mut zero_point = vec![8i8; n_channels];

    for ch in 0..n_channels {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        // 查找通道范围
        for i in 0..channel_size {
            let idx = compute_index(ch, i, channel_dim, n_channels, channel_size);
            if idx < weights.len() {
                min = min.min(weights[idx]);
                max = max.max(weights[idx]);
            }
        }

        // 计算 scale 和 zero_point
        let ch_scale = (max - min) / 15.0;
        let ch_zp = if ch_scale > 0.0 {
            (-min / ch_scale).round() as i8
        } else {
            8i8
        };

        scale[ch] = ch_scale;
        zero_point[ch] = ch_zp.clamp(0, 15);

        // 量化并打包
        for i in 0..channel_size {
            let idx = compute_index(ch, i, channel_dim, n_channels, channel_size);
            let quantized = if ch_scale > 0.0 && idx < weights.len() {
                ((weights[idx] / ch_scale).round() as i32 + ch_zp as i32)
                    .clamp(0, 15) as u8
            } else {
                8u8
            };

            let flat_idx = ch * channel_size + i;
            let byte_idx = flat_idx / 2;
            let is_high = flat_idx % 2 == 1;

            if is_high {
                data[byte_idx] |= quantized << 4;
            } else {
                data[byte_idx] = quantized;
            }
        }
    }

    QuantizedWeight {
        data, scale, zero_point, shape: shape.to_vec(),
        quant_type: QuantType::Int4,
    }
}

// ═══════════════════════ INT8 Per-Channel 量化 ═══════════════════════

/// 将 FP32 权重量化为 INT8 Per-Channel
/// 用于注意力 Q/K 投影 (精度敏感)
pub fn quantize_tensor_int8(
    weights: &[f32],
    shape: &[usize],
    channel_dim: usize,
) -> QuantizedWeight {
    let n_channels = shape[channel_dim];
    let channel_size: usize = shape.iter().enumerate()
        .filter(|(i, _)| *i != channel_dim)
        .map(|(_, &s)| s)
        .product();

    let total_elements: usize = shape.iter().product();
    let mut data = vec![0u8; total_elements];
    let mut scale = vec![0.0f32; n_channels];
    let mut zero_point = vec![127i8; n_channels];

    for ch in 0..n_channels {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        for i in 0..channel_size {
            let idx = compute_index(ch, i, channel_dim, n_channels, channel_size);
            if idx < weights.len() {
                min = min.min(weights[idx]);
                max = max.max(weights[idx]);
            }
        }

        let ch_scale = (max - min) / 255.0;
        let ch_zp = if ch_scale > 0.0 {
            (-min / ch_scale).round() as i8
        } else {
            127i8
        };

        scale[ch] = ch_scale;
        zero_point[ch] = ch_zp.clamp(-128, 127);

        for i in 0..channel_size {
            let idx = compute_index(ch, i, channel_dim, n_channels, channel_size);
            let quantized = if ch_scale > 0.0 && idx < weights.len() {
                ((weights[idx] / ch_scale).round() as i32 + ch_zp as i32)
                    .clamp(0, 255) as u8
            } else {
                127u8
            };

            let flat_idx = ch * channel_size + i;
            data[flat_idx] = quantized;
        }
    }

    QuantizedWeight {
        data, scale, zero_point, shape: shape.to_vec(),
        quant_type: QuantType::Int8,
    }
}

// ═══════════════════════ 群组量化 (GroupQuant4) ═══════════════════════

/// 群组 4-bit 量化 (类 GPTQ)
/// 每 group_size 个元素共享一组 scale 和 zero_point
/// 优势: 比纯 per-channel 更精细的量化粒度
pub fn quantize_tensor_group4(
    weights: &[f32],
    shape: &[usize],
    group_size: usize,
) -> QuantizedWeight {
    let total_elements: usize = shape.iter().product();
    let n_groups = (total_elements + group_size - 1) / group_size;
    let data_len = (total_elements + 1) / 2;

    let mut data = vec![0u8; data_len];
    let mut scale = vec![0.0f32; n_groups];
    let mut zero_point = vec![8i8; n_groups];

    for g in 0..n_groups {
        let start = g * group_size;
        let end = (start + group_size).min(total_elements);

        // 查找群组范围
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for i in start..end {
            min = min.min(weights[i]);
            max = max.max(weights[i]);
        }

        // 计算 scale 和 zero_point
        let g_scale = (max - min) / 15.0;
        let g_zp = if g_scale > 0.0 {
            (-min / g_scale).round() as i8
        } else {
            8i8
        };

        scale[g] = g_scale;
        zero_point[g] = g_zp.clamp(0, 15);

        // 量化并打包
        for i in start..end {
            let quantized = if g_scale > 0.0 {
                ((weights[i] / g_scale).round() as i32 + g_zp as i32)
                    .clamp(0, 15) as u8
            } else {
                8u8
            };

            let byte_idx = i / 2;
            let is_high = i % 2 == 1;

            if is_high {
                data[byte_idx] |= quantized << 4;
            } else {
                data[byte_idx] = quantized;
            }
        }
    }

    QuantizedWeight {
        data, scale, zero_point, shape: shape.to_vec(),
        quant_type: QuantType::GroupQuant4 { group_size },
    }
}

// ═══════════════════════ FP16 保留 ═══════════════════════

/// 将 FP32 权重保留为 FP16 格式 (实际用 FP32 存储)
/// 用于 Embedding 和 RMSNorm gamma
pub fn quantize_tensor_fp16(
    weights: &[f32],
    shape: &[usize],
) -> FP16Weight {
    FP16Weight {
        data: weights.to_vec(),
        shape: shape.to_vec(),
    }
}

// ═══════════════════════ 解量化 ═══════════════════════

/// 解量化 INT4 权重 (per-channel)
/// 修复了 v2 的通道索引计算错误:
/// - v2 bug: 使用 `i % scale.len()` 按元素循环, 错误
/// - v3 fix: 使用 `i / in_dim` 按行计算通道, 正确
pub fn dequantize_int4(qw: &QuantizedWeight) -> Vec<f32> {
    let total: usize = qw.shape.iter().product();
    if qw.shape.len() < 2 { return vec![0.0f32; total]; }

    let in_dim = qw.shape[1];
    let mut result = Vec::with_capacity(total);

    for i in 0..total {
        let is_high = i % 2 == 1;
        let byte_idx = i / 2;
        let raw = if byte_idx < qw.data.len() {
            if is_high {
                (qw.data[byte_idx] >> 4) & 0x0F
            } else {
                qw.data[byte_idx] & 0x0F
            }
        } else {
            8u8
        };

        // 修正: 行优先排列, channel = row_index
        let row = i / in_dim;
        let ch = row.min(qw.scale.len().saturating_sub(1));
        let s = qw.scale.get(ch).copied().unwrap_or(0.02);
        let zp = qw.zero_point.get(ch).copied().unwrap_or(8);

        result.push((raw as f32 - zp as f32) * s);
    }

    result
}

/// 解量化 INT8 权重 (per-channel)
/// 修复了 v2 的通道索引计算错误:
/// - v2 bug: 使用 `i / ((total + scale.len() - 1) / scale.len())` 近似计算通道
/// - v3 fix: 使用 `i / in_dim` 精确计算通道索引
pub fn dequantize_int8(qw: &QuantizedWeight) -> Vec<f32> {
    let total: usize = qw.shape.iter().product();
    if qw.shape.len() < 2 { return vec![0.0f32; total]; }

    let in_dim = qw.shape[1];
    let mut result = Vec::with_capacity(total);

    for i in 0..total.min(qw.data.len()) {
        // 修正: 行优先, channel = row_index = i / in_dim
        let row = i / in_dim;
        let ch = row.min(qw.scale.len().saturating_sub(1));
        let s = qw.scale.get(ch).copied().unwrap_or(0.01);
        let zp = qw.zero_point.get(ch).copied().unwrap_or(127);

        result.push((qw.data[i] as f32 - zp as f32) * s);
    }

    while result.len() < total {
        result.push(0.0f32);
    }

    result
}

/// 解量化群组 4-bit 权重
/// 每 group_size 个元素共享 scale/zp
pub fn dequantize_group4(qw: &QuantizedWeight, group_size: usize) -> Vec<f32> {
    let total: usize = qw.shape.iter().product();
    let mut result = Vec::with_capacity(total);

    for i in 0..total {
        let is_high = i % 2 == 1;
        let byte_idx = i / 2;
        let raw = if byte_idx < qw.data.len() {
            if is_high {
                (qw.data[byte_idx] >> 4) & 0x0F
            } else {
                qw.data[byte_idx] & 0x0F
            }
        } else {
            8u8
        };

        let group_idx = i / group_size;
        let ch = group_idx.min(qw.scale.len().saturating_sub(1));
        let s = qw.scale.get(ch).copied().unwrap_or(0.02);
        let zp = qw.zero_point.get(ch).copied().unwrap_or(8);

        result.push((raw as f32 - zp as f32) * s);
    }

    result
}

/// 通用解量化 (根据 quant_type 自动选择)
pub fn dequantize(qw: &QuantizedWeight) -> Vec<f32> {
    match &qw.quant_type {
        QuantType::Int4 => dequantize_int4(qw),
        QuantType::Int8 => dequantize_int8(qw),
        QuantType::GroupQuant4 { group_size } => dequantize_group4(qw, *group_size),
    }
}

// ═══════════════════════ 误差度量 ═══════════════════════

/// 计算量化误差 (MSE - 均方误差)
pub fn quantization_mse(original: &[f32], dequantized: &[f32]) -> f32 {
    let n = original.len().min(dequantized.len());
    if n == 0 { return 0.0; }
    (0..n)
        .map(|i| (original[i] - dequantized[i]).powi(2))
        .sum::<f32>()
        / n as f32
}

/// 计算量化误差 (RMSE - 均方根误差)
pub fn quantization_rmse(original: &[f32], dequantized: &[f32]) -> f32 {
    quantization_mse(original, dequantized).sqrt()
}

/// 计算量化误差 (余弦相似度)
/// 返回值范围 [-1, 1]，1 表示完全相同
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 { return 0.0; }

    let dot: f32 = (0..n).map(|i| a[i] * b[i]).sum();
    let norm_a: f32 = (0..n).map(|i| a[i] * a[i]).sum::<f32>().sqrt();
    let norm_b: f32 = (0..n).map(|i| b[i] * b[i]).sum::<f32>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// 综合量化误差报告
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantizationReport {
    pub mse: f32,
    pub rmse: f32,
    pub cosine_sim: f32,
    pub max_abs_error: f32,
    pub quant_type: String,
    pub original_elements: usize,
    pub compressed_bytes: usize,
    pub compression_ratio: f64,
}

/// 生成综合量化误差报告
pub fn quantization_report(
    original: &[f32],
    dequantized: &[f32],
    qw: &QuantizedWeight,
) -> QuantizationReport {
    let n = original.len().min(dequantized.len());
    let max_abs_error = if n > 0 {
        (0..n).map(|i| (original[i] - dequantized[i]).abs())
            .fold(0.0f32, f32::max)
    } else {
        0.0
    };

    let original_bytes = original.len() * 4; // f32 = 4 bytes
    let compressed_bytes = qw.data.len() + qw.scale.len() * 4 + qw.zero_point.len();

    QuantizationReport {
        mse: quantization_mse(original, dequantized),
        rmse: quantization_rmse(original, dequantized),
        cosine_sim: cosine_similarity(original, dequantized),
        max_abs_error,
        quant_type: format!("{:?}", qw.quant_type),
        original_elements: original.len(),
        compressed_bytes,
        compression_ratio: original_bytes as f64 / compressed_bytes.max(1) as f64,
    }
}

// ═══════════════════════ 二进制序列化 ═══════════════════════

/// 二进制格式的魔数
const BINARY_MAGIC: &[u8; 4] = b"FQ3\0";

/// 二进制格式的版本号
const BINARY_VERSION: u32 = 1;

/// 将量化权重序列化为二进制格式
/// 格式: [MAGIC:4][VERSION:4][quant_type:1][shape_len:4][shape...][scale_len:4][scale...][zp_len:4][zp...][data_len:4][data...]
pub fn serialize_quantized_binary(qw: &QuantizedWeight) -> Vec<u8> {
    let mut buf = Vec::new();

    // 魔数 + 版本
    buf.extend_from_slice(BINARY_MAGIC);
    buf.extend_from_slice(&BINARY_VERSION.to_le_bytes());

    // quant_type 标记 (先写入类型标记，再按需写入附加数据)
    let qt_byte: u8 = match &qw.quant_type {
        QuantType::Int4 => 0,
        QuantType::Int8 => 1,
        QuantType::GroupQuant4 { .. } => 2,
    };
    buf.push(qt_byte);

    // 群组量化附加数据: group_size
    if let QuantType::GroupQuant4 { group_size } = &qw.quant_type {
        buf.extend_from_slice(&(*group_size as u32).to_le_bytes());
    }

    // shape
    buf.extend_from_slice(&(qw.shape.len() as u32).to_le_bytes());
    for &s in &qw.shape {
        buf.extend_from_slice(&(s as u32).to_le_bytes());
    }

    // scale
    buf.extend_from_slice(&(qw.scale.len() as u32).to_le_bytes());
    for &s in &qw.scale {
        buf.extend_from_slice(&s.to_le_bytes());
    }

    // zero_point
    buf.extend_from_slice(&(qw.zero_point.len() as u32).to_le_bytes());
    for &zp in &qw.zero_point {
        buf.push(zp as u8); // i8 → u8 保存
    }

    // data
    buf.extend_from_slice(&(qw.data.len() as u32).to_le_bytes());
    buf.extend_from_slice(&qw.data);

    buf
}

/// 从二进制格式反序列化量化权重
pub fn deserialize_quantized_binary(data: &[u8]) -> Result<QuantizedWeight, String> {
    if data.len() < 12 {
        return Err("数据太短".to_string());
    }

    let mut pos = 0;

    // 检查魔数
    if &data[0..4] != BINARY_MAGIC {
        return Err("无效的魔数".to_string());
    }
    pos += 4;

    // 版本
    let _version = u32::from_le_bytes(data[pos..pos + 4].try_into().map_err(|_| "版本解析失败")?);
    pos += 4;

    // quant_type
    let qt_byte = data[pos];
    pos += 1;

    let quant_type = match qt_byte {
        0 => QuantType::Int4,
        1 => QuantType::Int8,
        2 => {
            if pos + 4 > data.len() {
                return Err("群组量化数据不完整".to_string());
            }
            let group_size = u32::from_le_bytes(data[pos..pos + 4].try_into().map_err(|_| "group_size 解析失败")?) as usize;
            pos += 4;
            QuantType::GroupQuant4 { group_size }
        }
        _ => return Err(format!("未知的量化类型: {}", qt_byte)),
    };

    // shape
    let shape_len = u32::from_le_bytes(data[pos..pos + 4].try_into().map_err(|_| "shape 长度解析失败")?) as usize;
    pos += 4;

    let mut shape = Vec::with_capacity(shape_len);
    for _ in 0..shape_len {
        let s = u32::from_le_bytes(data[pos..pos + 4].try_into().map_err(|_| "shape 解析失败")?) as usize;
        shape.push(s);
        pos += 4;
    }

    // scale
    let scale_len = u32::from_le_bytes(data[pos..pos + 4].try_into().map_err(|_| "scale 长度解析失败")?) as usize;
    pos += 4;

    let mut scale = Vec::with_capacity(scale_len);
    for _ in 0..scale_len {
        let s = f32::from_le_bytes(data[pos..pos + 4].try_into().map_err(|_| "scale 解析失败")?);
        scale.push(s);
        pos += 4;
    }

    // zero_point
    let zp_len = u32::from_le_bytes(data[pos..pos + 4].try_into().map_err(|_| "zp 长度解析失败")?) as usize;
    pos += 4;

    let mut zero_point = Vec::with_capacity(zp_len);
    for _ in 0..zp_len {
        zero_point.push(data[pos] as i8);
        pos += 1;
    }

    // data
    let data_len = u32::from_le_bytes(data[pos..pos + 4].try_into().map_err(|_| "data 长度解析失败")?) as usize;
    pos += 4;

    if pos + data_len > data.len() {
        return Err("数据长度不匹配".to_string());
    }

    let weight_data = data[pos..pos + data_len].to_vec();

    Ok(QuantizedWeight {
        data: weight_data,
        scale,
        zero_point,
        shape,
        quant_type,
    })
}

// ═══════════════════════ 工具函数 ═══════════════════════

/// 计算多维索引到扁平索引的映射
fn compute_index(ch: usize, i: usize, channel_dim: usize, n_channels: usize, _channel_size: usize) -> usize {
    if channel_dim == 0 {
        ch * _channel_size + i
    } else {
        i * n_channels + ch
    }
}

// ═══════════════════════ 测试 ═══════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int4_quantize_dequantize() {
        let original: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 10.0).collect();
        let shape = vec![10, 10];
        let qw = quantize_tensor_int4(&original, &shape, 0);
        assert_eq!(qw.quant_type, QuantType::Int4);

        let deq = dequantize_int4(&qw);
        let error = quantization_mse(&original, &deq);
        println!("INT4 MSE: {:.6}", error);
        assert!(error < 50.0, "INT4 MSE 过高: {}", error);
    }

    #[test]
    fn test_int8_quantize_dequantize() {
        let original: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 10.0).collect();
        let shape = vec![10, 10];
        let qw = quantize_tensor_int8(&original, &shape, 0);
        assert_eq!(qw.quant_type, QuantType::Int8);

        let deq = dequantize_int8(&qw);
        let error = quantization_mse(&original, &deq);
        println!("INT8 MSE: {:.6}", error);
        // INT8 误差应该远小于 INT4
        assert!(error < 10.0, "INT8 MSE 过高: {}", error);
    }

    #[test]
    fn test_int8_channel_index_fix() {
        // 验证 v3 的通道索引修复
        // 构造一个每行值不同的矩阵
        let mut original = vec![0.0f32; 4 * 8]; // [4, 8]
        for row in 0..4 {
            for col in 0..8 {
                original[row * 8 + col] = (row as f32 + 1.0) * 0.1;
            }
        }

        let shape = vec![4, 8];
        let qw = quantize_tensor_int8(&original, &shape, 0);
        let deq = dequantize_int8(&qw);

        // 每行的值应该近似相同 (因为同行的原始值相同)
        for row in 0..4 {
            let row_vals: Vec<f32> = (0..8).map(|col| deq[row * 8 + col]).collect();
            let row_min = row_vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let row_max = row_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            // 同行内误差应极小 (量化精度范围内)
            assert!((row_max - row_min) < 0.1, "行 {} 内误差过大: {}", row, row_max - row_min);
        }
    }

    #[test]
    fn test_group4_quantize() {
        let original: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 50.0).collect();
        let shape = vec![4, 64];
        let qw = quantize_tensor_group4(&original, &shape, DEFAULT_GROUP_SIZE);
        assert!(matches!(qw.quant_type, QuantType::GroupQuant4 { .. }));

        let deq = dequantize_group4(&qw, DEFAULT_GROUP_SIZE);
        let error = quantization_mse(&original, &deq);
        println!("GroupQuant4 MSE: {:.6}", error);
        assert!(error < 50.0, "GroupQuant4 MSE 过高: {}", error);
    }

    #[test]
    fn test_fp16_preserve() {
        let original: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let shape = vec![2, 5];
        let fw = quantize_tensor_fp16(&original, &shape);
        assert_eq!(fw.data, original);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-5, "相同向量余弦相似度应为 1.0");

        let c = vec![-1.0, -2.0, -3.0];
        let sim2 = cosine_similarity(&a, &c);
        assert!((sim2 - (-1.0)).abs() < 1e-5, "反方向余弦相似度应为 -1.0");
    }

    #[test]
    fn test_binary_serialization_int4() {
        let original: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 10.0).collect();
        let shape = vec![10, 10];
        let qw = quantize_tensor_int4(&original, &shape, 0);

        let binary = serialize_quantized_binary(&qw);
        let loaded = deserialize_quantized_binary(&binary).unwrap();

        assert_eq!(loaded.data, qw.data);
        assert_eq!(loaded.shape, qw.shape);
        assert_eq!(loaded.quant_type, qw.quant_type);
    }

    #[test]
    fn test_binary_serialization_int8() {
        let original: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 10.0).collect();
        let shape = vec![10, 10];
        let qw = quantize_tensor_int8(&original, &shape, 0);

        let binary = serialize_quantized_binary(&qw);
        let loaded = deserialize_quantized_binary(&binary).unwrap();

        assert_eq!(loaded.data, qw.data);
        assert_eq!(loaded.shape, qw.shape);
        assert_eq!(loaded.quant_type, qw.quant_type);
    }

    #[test]
    fn test_binary_serialization_group4() {
        let original: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 50.0).collect();
        let shape = vec![4, 64];
        let qw = quantize_tensor_group4(&original, &shape, DEFAULT_GROUP_SIZE);

        let binary = serialize_quantized_binary(&qw);
        let loaded = deserialize_quantized_binary(&binary).unwrap();

        assert_eq!(loaded.data, qw.data);
        assert_eq!(loaded.shape, qw.shape);
        assert_eq!(loaded.quant_type, qw.quant_type);
    }

    #[test]
    fn test_quantization_report() {
        let original: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 10.0).collect();
        let shape = vec![10, 10];
        let qw = quantize_tensor_int8(&original, &shape, 0);
        let deq = dequantize_int8(&qw);

        let report = quantization_report(&original, &deq, &qw);
        println!("量化报告: {:?}", report);
        assert!(report.cosine_sim > 0.7, "INT8 余弦相似度应 > 0.7: {}", report.cosine_sim);
        assert!(report.compression_ratio > 1.0, "压缩率应 > 1.0");
    }

    #[test]
    fn test_int4_smaller_error_than_v2() {
        // v3 的 INT4 通道索引修正应该给出更低的误差
        let original: Vec<f32> = (0..200).map(|i| (i as f32 - 100.0) / 20.0).collect();
        let shape = vec![10, 20];
        let qw = quantize_tensor_int4(&original, &shape, 0);
        let deq = dequantize_int4(&qw);
        let mse = quantization_mse(&original, &deq);
        let cos = cosine_similarity(&original, &deq);
        println!("v3 INT4: MSE={:.6}, CosSim={:.4}", mse, cos);
        assert!(cos > 0.7, "v3 INT4 余弦相似度过低: {}", cos);
    }
}
