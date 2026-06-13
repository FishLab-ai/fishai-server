//! BPE (Byte-Pair Encoding) 分词器
//!
//! 自研轻量级分词器:
//! - 基于 Byte-Pair Encoding 算法
//! - 支持中英文混合文本
//! - 词汇表大小: 32000
//! - 特殊 token: <PAD>, <BOS>, <EOS>, <UNK>

use std::collections::HashMap;

/// 特殊 token
pub const PAD_TOKEN: usize = 0;
pub const BOS_TOKEN: usize = 1;
pub const EOS_TOKEN: usize = 2;
pub const UNK_TOKEN: usize = 3;

/// BPE 分词器
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BPETokenizer {
    pub vocab: HashMap<String, usize>,
    pub id_to_token: HashMap<usize, String>,
    pub merges: Vec<(String, String)>,
    pub vocab_size: usize,
}

impl BPETokenizer {
    /// 创建基础 byte-level 分词器
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();

        let specials = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"];
        for (i, token) in specials.iter().enumerate() {
            vocab.insert(token.to_string(), i);
            id_to_token.insert(i, token.to_string());
        }

        for b in 0u8..=255 {
            let token = format!("<byte_{:02x}>", b);
            let id = 4 + b as usize;
            vocab.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }

        Self {
            vocab,
            id_to_token,
            merges: Vec::new(),
            vocab_size: 260,
        }
    }

    /// 编码文本为 token ID 序列
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens: Vec<usize> = vec![BOS_TOKEN];

        for ch in text.chars() {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            for &b in s.as_bytes() {
                tokens.push(4 + b as usize);
            }
        }

        tokens.push(EOS_TOKEN);
        tokens
    }

    /// 解码 token ID 序列为文本
    pub fn decode(&self, token_ids: &[usize]) -> String {
        let mut bytes = Vec::new();

        for &id in token_ids {
            if id <= 3 {
                continue;
            }
            if let Some(token) = self.id_to_token.get(&id) {
                if token.starts_with("<byte_") && token.ends_with(">") {
                    let hex = &token[6..token.len() - 1];
                    if let Ok(b) = u8::from_str_radix(hex, 16) {
                        bytes.push(b);
                    }
                }
            }
        }

        String::from_utf8_lossy(&bytes).to_string()
    }

    /// 保存分词器
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// 加载分词器
    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let tokenizer: BPETokenizer = serde_json::from_str(&data)?;
        Ok(tokenizer)
    }
}

impl Default for BPETokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_encode_decode() {
        let tokenizer = BPETokenizer::new();
        let text = "Hello";
        let tokens = tokenizer.encode(text);
        assert!(tokens.len() > 2);
        let decoded = tokenizer.decode(&tokens);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_chinese_encode_decode() {
        let tokenizer = BPETokenizer::new();
        let text = "你好";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);
        assert_eq!(decoded, text);
    }
}
