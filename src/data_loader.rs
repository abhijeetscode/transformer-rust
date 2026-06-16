use crate::tokenizer::{load_tokenizer, train_tokenizer};
use candle_core::{Device, Result, Tensor};
use std::fs::{read_to_string, write};

const TRAIN_SPLIT_PATH: &str = "./dataset/train_split.txt";

fn to_err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::msg(e.to_string())
}

pub fn get_data_splits(
    filename: &str,
    train_perc: f64,
    eval_perc: f64,
) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    if !(train_perc > 0.0 && eval_perc > 0.0 && train_perc + eval_perc < 1.0) {
        return Err(candle_core::Error::msg(
            "need train_perc>0, eval_perc>0, and train_perc+eval_perc<1",
        ));
    }
    let text = read_to_string(filename)?;

    // Split the RAW text at two points, so the tokenizer is only ever trained on
    // the training portion (no eval/test leakage). Snap up to a UTF-8 char boundary.
    let char_boundary = |frac: f64| {
        let mut i = (frac * text.len() as f64) as usize;
        while i < text.len() && !text.is_char_boundary(i) {
            i += 1;
        }
        i
    };
    let i1 = char_boundary(train_perc);
    let i2 = char_boundary(train_perc + eval_perc);

    let train_text = &text[..i1];
    let eval_text = &text[i1..i2];
    let test_text = &text[i2..];

    // Train the BPE tokenizer on the training split only, then load it back.
    write(TRAIN_SPLIT_PATH, train_text)?;
    train_tokenizer(TRAIN_SPLIT_PATH).map_err(to_err)?;
    let tokenizer = load_tokenizer().map_err(to_err)?;

    // Encode each split into one flat token stream.
    let encode = |t: &str| -> Result<Vec<u32>> {
        Ok(tokenizer
            .encode(t, false)
            .map_err(to_err)?
            .get_ids()
            .to_vec())
    };
    let train_data = encode(train_text)?;
    let eval_data = encode(eval_text)?;
    let test_data = encode(test_text)?;

    println!(
        "vocab: {}, train: {}, eval: {}, test: {} tokens",
        tokenizer.get_vocab_size(true),
        train_data.len(),
        eval_data.len(),
        test_data.len(),
    );
    Ok((train_data, eval_data, test_data))
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
