mod attention;
mod config;
mod gpt_model;
mod transformer_block;

use attention::MultiheadMaskAttention;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use config::DEVICE;

fn main() -> Result<()> {
    let x_tensor = Tensor::randn(0f32, 1f32, (1, 10, 1024), &DEVICE)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let xx = MultiheadMaskAttention::new(2, 1024, &vb)?;
    let tensor = xx.forward(&x_tensor)?;
    Ok(())
}
