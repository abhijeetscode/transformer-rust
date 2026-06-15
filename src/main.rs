mod attention;
mod config;
mod data_loader;
mod gpt_model;
mod transformer_block;

use crate::config::ENCODING_NAME;
use crate::gpt_model::GPTModel;
use candle_core::DType;
use candle_core::DType::F32;
use candle_core::Result;
use candle_core::Tensor;

use candle_nn::VarBuilder;
use candle_nn::VarMap;
use candle_nn::loss::cross_entropy;
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

use config::{BATCH_SIZE, CONTEXT_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS};

use config::device;
use data_loader::{get_batch, get_train_test_data};
use rand::seq::SliceRandom;
use tiktoken::get_encoding;

fn evaluate_model(gpt_model: &GPTModel, data: &Vec<u32>, mask: &Tensor) -> Result<f32> {
    let starts: Vec<usize> = (0..data.len() - CONTEXT_SIZE - 1)
        .step_by(CONTEXT_SIZE)
        .collect();
    let mut total_loss: f32 = 0.0;
    let mut num_batch: usize = 0;
    for chunk in starts.chunks(3) {
        let (xs, ys) = get_batch(data, chunk, CONTEXT_SIZE, device())?;
        let logits: Tensor = gpt_model.forward(&xs, mask)?;
        let (b, l, v) = logits.dims3()?;
        let logits: Tensor = logits.reshape((b * l, v))?;
        let ys = ys.reshape(b * l)?;
        let loss = cross_entropy(&logits, &ys)?;
        total_loss += loss.to_scalar::<f32>()?;
        num_batch += 1
    }
    Ok(total_loss / num_batch as f32)
}
fn main() -> Result<()> {
    let (train_data, test_data) = get_train_test_data("./dataset/tiny_shakespere.txt", 0.9)?;

    println!(
        "train_data len: {}, test_data len: {}",
        train_data.len(),
        test_data.len()
    );
    let encoding = get_encoding(ENCODING_NAME)
        .ok_or_else(|| candle_core::Error::msg(format!("unknown encoding: {}", ENCODING_NAME)))?;
    let vocab_size = encoding.vocab_size();
    let max_seq_length: usize = CONTEXT_SIZE;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device());
    let gpt_model = GPTModel::new(
        vocab_size,
        EMBED_DIM,
        NUM_HEADS,
        max_seq_length,
        NUM_LAYERS,
        &vb,
    )?;
    let num_params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    println!("Num Params = {}", num_params);
    return Ok(());
    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 3e-4,
            ..Default::default()
        },
    )?;
    let starts: Vec<usize> = (0..train_data.len() - CONTEXT_SIZE - 1)
        .step_by(CONTEXT_SIZE)
        .collect();

    let max_epochs = 50;
    let patience = 3;
    let mut best_val = f32::INFINITY;
    let mut stale = 0;
    let blocked: Tensor = Tensor::tril2(CONTEXT_SIZE, F32, device())?.eq(0.0)?;
    let mask_additive: Tensor = blocked.where_cond(
        &Tensor::full(f32::NEG_INFINITY, (CONTEXT_SIZE, CONTEXT_SIZE), device())?,
        &Tensor::zeros((CONTEXT_SIZE, CONTEXT_SIZE), F32, device())?,
    )?;

    for epoch in 0..max_epochs {
        let mut rng = rand::rng();
        let mut shuffled = starts.clone();
        shuffled.shuffle(&mut rng);
        let mut loss_epoch = 0.0;
        let mut num_batches = 0usize;
        for chunk in shuffled.chunks(BATCH_SIZE) {
            let (xs, ys) = get_batch(&train_data, chunk, CONTEXT_SIZE, device())?;

            let logits: Tensor = gpt_model.forward(&xs, &mask_additive)?;
            let (b, t, v) = logits.dims3()?;
            let logits = logits.reshape((b * t, v))?;
            let ys: Tensor = ys.reshape(b * t)?;
            let loss: Tensor = cross_entropy(&logits, &ys)?;
            loss_epoch += loss.to_scalar::<f32>()?;
            opt.backward_step(&loss)?;
            num_batches += 1;
        }
        let train_loss: f32 = loss_epoch / num_batches as f32;
        let val_loss: f32 = evaluate_model(&gpt_model, &test_data, &mask_additive)?;

        println!(
            "Epoch {} Train Loss {} Val Loss {}",
            epoch + 1,
            train_loss,
            val_loss
        );

        // early stopping — save only the best model
        if val_loss < best_val {
            best_val = val_loss;
            stale = 0;
            varmap.save("best_model.safetensors")?;
            println!("new best (val {val_loss:.4}), saved best_model.safetensors");
        } else {
            stale += 1;
            println!("  no improvement ({stale}/{patience})");
            if stale >= patience {
                println!("early stopping — val loss not improving");
                break;
            }
        }
    }

    Ok(())
}
