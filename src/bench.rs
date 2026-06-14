//! FishAI v3 推理基准测试模块
//!
//! 测量指标:
//! 1. TTFT (Time-To-First-Token) — 首 token 延迟
//! 2. Tokens/Second — 生成吞吐量
//! 3. 内存占用 — KV Cache + 权重内存
//! 4. Prefill vs Decode 延迟对比
//!

use std::time::Instant;

use crate::model::{GPTWeights, ModelConfig, ModelKVCache, generate_with_cache, gpt_forward, gpt_forward_with_cache, sample_token};
use crate::tokenizer::FishAITokenizer;

// ═══════════════════════ 基准测试报告 ═══════════════════════

/// 基准测试报告
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchReport {
    /// 模型名称
    pub model_name: String,
    /// 模型参数量 (百万)
    pub params_m: usize,
    /// 量化后大小 (MB)
    pub quantized_size_mb: f64,
    /// Prompt 长度
    pub prompt_len: usize,
    /// 生成 token 数
    pub generated_tokens: usize,
    /// TTFT: 首 token 延迟 (毫秒)
    pub ttft_ms: f64,
    /// Prefill 延迟 (毫秒) — 处理 prompt 的时间
    pub prefill_ms: f64,
    /// Decode 平均延迟 (毫秒/token) — 生成单个 token 的时间
    pub decode_ms_per_token: f64,
    /// 总延迟 (毫秒)
    pub total_ms: f64,
    /// 生成吞吐量 (tokens/second)
    pub tokens_per_second: f64,
    /// KV Cache 内存占用 (字节)
    pub kv_cache_bytes: usize,
    /// 模型权重估算内存 (字节)
    pub weights_estimated_bytes: usize,
    /// 是否使用 KV Cache
    pub used_kv_cache: bool,
    /// 采样参数
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl std::fmt::Display for BenchReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════╗")?;
        writeln!(f, "║       🐟 FishAI v3 Benchmark Report             ║")?;
        writeln!(f, "╚══════════════════════════════════════════════════╝")?;
        writeln!(f, "模型:        {} ({}M 参数)", self.model_name, self.params_m)?;
        writeln!(f, "量化大小:    {:.1} MB", self.quantized_size_mb)?;
        writeln!(f, "─────────────────────────────────────────────────")?;
        writeln!(f, "Prompt 长度: {}", self.prompt_len)?;
        writeln!(f, "生成 Token:  {}", self.generated_tokens)?;
        writeln!(f, "─────────────────────────────────────────────────")?;
        writeln!(f, "TTFT:        {:.2} ms", self.ttft_ms)?;
        writeln!(f, "Prefill:     {:.2} ms", self.prefill_ms)?;
        writeln!(f, "Decode:      {:.2} ms/token", self.decode_ms_per_token)?;
        writeln!(f, "总延迟:      {:.2} ms", self.total_ms)?;
        writeln!(f, "吞吐量:      {:.2} tokens/s", self.tokens_per_second)?;
        writeln!(f, "─────────────────────────────────────────────────")?;
        writeln!(f, "KV Cache:    {:.2} MB", self.kv_cache_bytes as f64 / (1024.0 * 1024.0))?;
        writeln!(f, "权重估算:    {:.2} MB", self.weights_estimated_bytes as f64 / (1024.0 * 1024.0))?;
        writeln!(f, "KV Cache:    {}", if self.used_kv_cache { "✅ 启用" } else { "❌ 未启用" })?;
        writeln!(f, "─────────────────────────────────────────────────")?;
        writeln!(f, "采样参数:    temperature={}, top_k={}, top_p={}", self.temperature, self.top_k, self.top_p)
    }
}

// ═══════════════════════ 基准测试函数 ═══════════════════════

/// 运行完整基准测试 (带 KV Cache)
pub fn run_benchmark(
    weights: &GPTWeights,
    tokenizer: &FishAITokenizer,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> BenchReport {
    let config = &weights.config;

    // 准备 prompt
    let prompt = "FishAI is a powerful language model";
    let prompt_tokens = tokenizer.encode(prompt);
    let prompt_len = prompt_tokens.len();

    // 估算权重内存
    let weights_bytes = (config.quantized_size_mb() * 1024.0 * 1024.0) as usize;

    // ──── 带 KV Cache 的基准测试 ────
    let mut kv_cache = ModelKVCache::new(config.n_layers);

    // Prefill 阶段
    let prefill_start = Instant::now();
    let logits = gpt_forward_with_cache(&prompt_tokens, 0, &mut kv_cache, weights);
    let prefill_elapsed = prefill_start.elapsed();
    let prefill_ms = prefill_elapsed.as_secs_f64() * 1000.0;

    // TTFT = prefill 时间
    let ttft_ms = prefill_ms;

    // 首 token 采样
    let last_logits = &logits[logits.len() - 1];
    let mut next_token = sample_token(last_logits, temperature, top_k, top_p);

    // Decode 阶段
    let decode_start = Instant::now();
    let mut generated = 1;
    for step in 1..max_new_tokens {
        let pos = prompt_len + step - 1;
        if pos >= config.max_seq_len { break; }

        let new_logits = gpt_forward_with_cache(
            &[next_token], pos, &mut kv_cache, weights,
        );
        let last = &new_logits[0];
        next_token = sample_token(last, temperature, top_k, top_p);
        generated += 1;
    }
    let decode_elapsed = decode_start.elapsed();
    let decode_ms = decode_elapsed.as_secs_f64() * 1000.0;
    let decode_ms_per_token = if generated > 1 { decode_ms / (generated - 1) as f64 } else { 0.0 };

    let total_ms = prefill_ms + decode_ms;
    let tokens_per_second = if decode_ms > 0.0 {
        (generated - 1) as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };

    let kv_cache_bytes = kv_cache.total_memory_bytes();

    BenchReport {
        model_name: config.model_name().to_string(),
        params_m: config.total_params() / 1_000_000,
        quantized_size_mb: config.quantized_size_mb(),
        prompt_len,
        generated_tokens: generated,
        ttft_ms,
        prefill_ms,
        decode_ms_per_token,
        total_ms,
        tokens_per_second,
        kv_cache_bytes,
        weights_estimated_bytes: weights_bytes,
        used_kv_cache: true,
        temperature,
        top_k,
        top_p,
    }
}

/// 运行无 KV Cache 的基准测试 (对比用)
pub fn run_benchmark_no_cache(
    weights: &GPTWeights,
    tokenizer: &FishAITokenizer,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> BenchReport {
    let config = &weights.config;
    let prompt = "FishAI is a powerful language model";
    let prompt_tokens = tokenizer.encode(prompt);
    let prompt_len = prompt_tokens.len();
    let weights_bytes = (config.quantized_size_mb() * 1024.0 * 1024.0) as usize;

    let total_start = Instant::now();

    // 完整 prefill
    let prefill_start = Instant::now();
    let logits = gpt_forward(&prompt_tokens, weights);
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = prefill_ms;

    let last_logits = &logits[logits.len() - 1];
    let first_token = sample_token(last_logits, temperature, top_k, top_p);

    // 逐 token 生成 (无 Cache, 每次重算整个序列)
    let decode_start = Instant::now();
    let mut tokens = prompt_tokens.clone();
    tokens.push(first_token);
    let mut generated = 1;

    for _ in 1..max_new_tokens {
        let context_len = tokens.len().min(config.max_seq_len);
        let context = &tokens[tokens.len() - context_len..];
        let new_logits = gpt_forward(context, weights);
        let last = &new_logits[new_logits.len() - 1];
        let next = sample_token(last, temperature, top_k, top_p);
        tokens.push(next);
        generated += 1;
    }

    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let decode_ms_per_token = if generated > 1 { decode_ms / (generated - 1) as f64 } else { 0.0 };
    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

    let tokens_per_second = if decode_ms > 0.0 {
        (generated - 1) as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };

    BenchReport {
        model_name: config.model_name().to_string(),
        params_m: config.total_params() / 1_000_000,
        quantized_size_mb: config.quantized_size_mb(),
        prompt_len,
        generated_tokens: generated,
        ttft_ms,
        prefill_ms,
        decode_ms_per_token,
        total_ms,
        tokens_per_second,
        kv_cache_bytes: 0,
        weights_estimated_bytes: weights_bytes,
        used_kv_cache: false,
        temperature,
        top_k,
        top_p,
    }
}

/// 对比测试: KV Cache vs 无 Cache
pub fn run_comparison_benchmark(
    weights: &GPTWeights,
    tokenizer: &FishAITokenizer,
    max_new_tokens: usize,
) -> BenchComparison {
    let with_cache = run_benchmark(weights, tokenizer, max_new_tokens, 0.7, 50, 0.9);
    let without_cache = run_benchmark_no_cache(weights, tokenizer, max_new_tokens, 0.7, 50, 0.9);

    let speedup = if without_cache.decode_ms_per_token > 0.0 {
        without_cache.decode_ms_per_token / with_cache.decode_ms_per_token
    } else {
        0.0
    };

    BenchComparison {
        with_cache,
        without_cache,
        speedup,
    }
}

/// 对比报告
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchComparison {
    pub with_cache: BenchReport,
    pub without_cache: BenchReport,
    /// KV Cache 加速倍数
    pub speedup: f64,
}

impl std::fmt::Display for BenchComparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════╗")?;
        writeln!(f, "║    🐟 FishAI v3 KV Cache 对比报告               ║")?;
        writeln!(f, "╚══════════════════════════════════════════════════╝")?;
        writeln!(f, "")?;
        writeln!(f, "─── 带 KV Cache ───")?;
        write!(f, "{}", self.with_cache)?;
        writeln!(f, "")?;
        writeln!(f, "─── 无 KV Cache ───")?;
        write!(f, "{}", self.without_cache)?;
        writeln!(f, "")?;
        writeln!(f, "─── 对比 ───")?;
        writeln!(f, "加速倍数:    {:.2}x", self.speedup)?;
        writeln!(f, "Decode 优化: {:.2} ms → {:.2} ms/token",
            self.without_cache.decode_ms_per_token,
            self.with_cache.decode_ms_per_token)
    }
}

// ═══════════════════════ 快速基准测试 ═══════════════════════

/// 运行快速基准测试 (少量 token, 快速获得结果)
pub fn quick_bench(weights: &GPTWeights, tokenizer: &FishAITokenizer) -> BenchReport {
    run_benchmark(weights, tokenizer, 10, 0.7, 50, 0.9)
}

// ═══════════════════════ 测试 ═══════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;

    #[test]
    fn test_bench_report_display() {
        let report = BenchReport {
            model_name: "FishAI-S".to_string(),
            params_m: 34,
            quantized_size_mb: 12.0,
            prompt_len: 10,
            generated_tokens: 64,
            ttft_ms: 50.0,
            prefill_ms: 50.0,
            decode_ms_per_token: 10.0,
            total_ms: 690.0,
            tokens_per_second: 100.0,
            kv_cache_bytes: 1024 * 1024,
            weights_estimated_bytes: 12 * 1024 * 1024,
            used_kv_cache: true,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
        };
        let display = format!("{}", report);
        assert!(display.contains("FishAI-S"));
        assert!(display.contains("100.00 tokens/s"));
    }

    #[test]
    fn test_quick_bench() {
        let config = ModelConfig::small();
        let weights = GPTWeights::random_init(&config);
        let tokenizer = FishAITokenizer::new_byte_fallback(config.vocab_size);

        let report = quick_bench(&weights, &tokenizer);
        assert!(report.generated_tokens > 0);
        assert!(report.ttft_ms >= 0.0);
        assert!(report.tokens_per_second >= 0.0);
        assert!(report.used_kv_cache);
        println!("{}", report);
    }
}
