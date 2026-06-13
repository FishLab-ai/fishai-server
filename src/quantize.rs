//! FishAI v2 混合精度量化模块
//!
//! 核心升级:
//! - 关键层 (Embedding, RMSNorm gamma) 保留 FP16 精度
//! - 线性层权重使用 INT4 Per-Channel 量化
//! - Q/K 投影倾向 INT8 (注意力精度更敏感)
//! - FFN 权重可用 INT4 (对量化更鲁棒)
//!
//! 预期: 3-4× 压缩率，困惑度损失 < 1%

use super::model::{QuantizedWeight, FP16Weight};

/// 将 FP32 权重量化为 INT4 Per-Channel
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
    let mut zero_point = vec![0i8; n_channels];

    for ch in 0..n_channels {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        for i in 0..channel_size {
            let idx = if channel_dim == 0 {
                ch * channel_size + i
            } else {
                i * n_channels + ch
            };
            if idx < weights.len() {
                min = min.min(weights[idx]);
                max = max.max(weights[idx]);
            }
        }

        let ch_scale = (max - min) / 15.0;
        let ch_zp = if ch_scale > 0.0 {
            (-min / ch_scale).round() as i8
        } else {
            8i8
        };

        scale[ch] = ch_scale;
        zero_point[ch] = ch_zp.clamp(0, 15);

        for i in 0..channel_size {
            let idx = if channel_dim == 0 {
                ch * channel_size + i
            } else {
                i * n_channels + ch
            };

            let quantized = if ch_scale > 0.0 {
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

    QuantizedWeight { data, scale, zero_point, shape: shape.to_vec() }
}

/// 将 FP32 权重量化为 INT8 Per-Channel (用于注意力 Q/K 投影)
/// 比 INT4 精度更高，适合注意力精度敏感的场景
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
    // INT8: 每个值占 1 byte (用 data 直接存储)
    let mut data = vec![0u8; total_elements];
    let mut scale = vec![0.0f32; n_channels];
    let mut zero_point = vec![0i8; n_channels];

    for ch in 0..n_channels {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        for i in 0..channel_size {
            let idx = if channel_dim == 0 {
                ch * channel_size + i
            } else {
                i * n_channels + ch
            };
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
            let idx = if channel_dim == 0 {
                ch * channel_size + i
            } else {
                i * n_channels + ch
            };

            let quantized = if ch_scale > 0.0 {
                ((weights[idx] / ch_scale).round() as i32 + ch_zp as i32)
                    .clamp(0, 255) as u8
            } else {
                127u8
            };

            let flat_idx = ch * channel_size + i;
            data[flat_idx] = quantized;
        }
    }

    // 用 QuantizedWeight 格式存储，但 shape 标记为 INT8
    QuantizedWeight { data, scale, zero_point, shape: shape.to_vec() }
}

/// 将 FP32 权重保留为 FP16 格式 (实际用 FP32 存储)
pub fn quantize_tensor_fp16(
    weights: &[f32],
    shape: &[usize],
) -> FP16Weight {
    // 简单保留 FP32 (实际部署可转为 f16)
    FP16Weight {
        data: weights.to_vec(),
        shape: shape.to_vec(),
    }
}

/// 解量化 INT4 权重
pub fn dequantize_int4(qw: &QuantizedWeight) -> Vec<f32> {
    let total: usize = qw.shape.iter().product();
    qw.data
        .iter()
        .flat_map(|&byte| {
            let low = (byte & 0x0F) as f32;
            let high = ((byte >> 4) & 0x0F) as f32;
            [low, high]
        })
        .take(total)
        .enumerate()
        .map(|(i, v)| {
            let ch = i % qw.scale.len().max(1);
            (v - qw.zero_point.get(ch).copied().unwrap_or(8) as f32)
                * qw.scale.get(ch).copied().unwrap_or(0.02)
        })
        .collect()
}

/// 解量化 INT8 权重
pub fn dequantize_int8(qw: &QuantizedWeight) -> Vec<f32> {
    let total: usize = qw.shape.iter().product();
    qw.data
        .iter()
        .take(total)
        .enumerate()
        .map(|(i, &v)| {
            let ch = i / ((total + qw.scale.len() - 1) / qw.scale.len().max(1));
            let ch_clamped = ch.min(qw.scale.len() - 1);
            (v as f32 - qw.zero_point[ch_clamped] as f32) * qw.scale[ch_clamped]
        })
        .collect()
}

/// 计算量化误差 (MSE)
pub fn quantization_error(original: &[f32], dequantized: &[f32]) -> f32 {
    let n = original.len().min(dequantized.len());
    let mse: f32 = (0..n)
        .map(|i| (original[i] - dequantized[i]).powi(2))
        .sum::<f32>()
        / n as f32;
    mse
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int4_quantize() {
        let original: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 10.0).collect();
        let shape = [10, 10];
        let qw = quantize_tensor_int4(&original, &shape, 0);
        let deq = dequantize_int4(&qw);
        let error = quantization_error(&original, &deq);
        assert!(error < 1.0, "INT4 error too high: {}", error);
    }

    #[test]
    fn test_int8_quantize() {
        let original: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 10.0).collect();
        let shape = [10, 10];
        let qw = quantize_tensor_int8(&original, &shape, 0);
        let deq = dequantize_int8(&qw);
        let error = quantization_error(&original, &deq);
        // INT8 误差应该远小于 INT4
        assert!(error < 0.1, "INT8 error too high: {}", error);
    }

    #[test]
    fn test_fp16_preserve() {
        let original: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let shape = [2, 5];
        let fw = quantize_tensor_fp16(&original, &shape);
        assert_eq!(fw.data, original);
    }
}
