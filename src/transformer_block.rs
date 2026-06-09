use crate::attention::MultiheadMaskAttention;
use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, LayerNormConfig, Linear, Module, VarBuilder, layer_norm, linear};

struct Ffn {
    l1: Linear,
    l2: Linear,
    embed_dim: usize,
}
impl Ffn {
    fn new(embed_dim: usize, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            l1: linear(embed_dim, 4 * embed_dim, vb.pp("l1"))?,
            l2: linear(4 * embed_dim, embed_dim, vb.pp("l2"))?,
            embed_dim,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.l1.forward(x)?;
        let out = out.gelu()?;
        let out = self.l2.forward(&out)?;
        Ok(out)
    }
}

pub struct TransformerBlock {
    multi_head_masked_attention: MultiheadMaskAttention,
    lnorm1: LayerNorm,
    ffn: Ffn,
    lnorm2: LayerNorm,
}
impl TransformerBlock {
    pub fn new(vb: &VarBuilder, embed_dim: usize, num_heads: usize) -> Result<Self> {
        Ok(Self {
            multi_head_masked_attention: MultiheadMaskAttention::new(
                num_heads,
                embed_dim,
                &vb.pp("attn"),
            )?,
            lnorm1: layer_norm(embed_dim, LayerNormConfig::default(), vb.pp("layer_norm1"))?,
            ffn: Ffn::new(embed_dim, &vb.pp("FFN"))?,
            lnorm2: layer_norm(embed_dim, LayerNormConfig::default(), vb.pp("layer_norm2"))?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shortcut = x;
        let h = self.lnorm1.forward(x)?;
        let attn: Tensor = self.multi_head_masked_attention.forward(&h)?;
        let out = (attn + shortcut)?;

        let shortcut = &out;
        let out2 = self.lnorm2.forward(&out)?;
        let out3 = self.ffn.forward(&out2)?;
        let out = (shortcut + out3)?;
        Ok(out)
    }
}
