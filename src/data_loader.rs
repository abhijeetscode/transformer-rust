use crate::config::ENCODING_NAME;
use candle_core::{Device, Result, Tensor};
use std::fs::read_to_string;
use tiktoken::get_encoding;

pub fn get_train_test_data(filename: &str, split_perc: f64) -> Result<(Vec<u32>, Vec<u32>)> {
    if !(split_perc > 0.0 && split_perc <= 1.0) {
        return Err(candle_core::Error::msg(
            "split_perc must be between 0 and 1",
        ));
    }
    let encoding = get_encoding(ENCODING_NAME)
        .ok_or_else(|| candle_core::Error::msg(format!("unknown encoding: {}", ENCODING_NAME)))?;
    let mut tokens: Vec<u32> = Vec::new();
    for line in read_to_string(filename)?.lines() {
        tokens.extend(encoding.encode(line));
    }
    let train_size: usize = (split_perc * tokens.len() as f64) as usize;
    let train_data: Vec<u32> = tokens[0..train_size].to_vec();
    let test_data: Vec<u32> = tokens[train_size..].to_vec();
    Ok((train_data, test_data))
}

pub fn get_batch(
    data: &[u32],
    starts: &[usize],
    context_size: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut xs: Vec<Tensor> = Vec::new();
    let mut ys: Vec<Tensor> = Vec::new();
    for &s in starts {
        xs.push(Tensor::new(&data[s..s + context_size], device)?);
        ys.push(Tensor::new(&data[s + 1..s + context_size + 1], device)?);
    }
    Ok((Tensor::stack(&xs, 0)?, Tensor::stack(&ys, 0)?))
}
