use candle_core::{DType, Result, Tensor};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{Linear, Module, VarBuilder, linear};

use crate::config::DEVICE;

struct MaskedAttentionHead {
    q: Linear,
    k: Linear,
    v: Linear,
    head_dim: usize,
}
impl MaskedAttentionHead {
    fn new(embed_dim: usize, head_dim: usize, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            head_dim,
            q: linear(embed_dim, head_dim, vb.pp("q"))?,
            k: linear(embed_dim, head_dim, vb.pp("k"))?,
            v: linear(embed_dim, head_dim, vb.pp("v"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // B = Batch Size, L = Max Seq Length, E = embed_dim, O = head dim (generally E//number of heads)
        // x is (B, L, E)
        let q = self.q.forward(x)?; // (B, L, O)
        let k = self.k.forward(x)?; // (B, L, O)
        let v = self.v.forward(x)?; // (B, L, O)
        let attn_scores = (q.matmul(&k.transpose(1, 2)?)? / (self.head_dim as f64).sqrt())?; // (B, L, L) -> Attention score are always L*L
        let (batch_size, seq_length, _) = attn_scores.dims3()?;
        let mask: Tensor = Tensor::tril2(seq_length, DType::U8, &DEVICE)?.eq(0u8)?;
        let mask = mask.broadcast_as((batch_size, seq_length, seq_length))?;
        let neg_inf_tensor: Tensor = Tensor::full(
            f32::NEG_INFINITY,
            (batch_size, seq_length, seq_length),
            &DEVICE,
        )?
        .to_dtype(attn_scores.dtype())?;
        let attn_scores = mask.where_cond(&neg_inf_tensor, &attn_scores)?; // Masked positions get -inf, unmasked positions remain unchanged    
        let attn_weights = softmax_last_dim(&attn_scores)?;
        let context = attn_weights.matmul(&v)?; // (B, L, O)
        Ok(context)
    }
}

pub struct MultiheadMaskAttention {
    num_heads: usize,
    heads: Vec<MaskedAttentionHead>,
    out: Linear,
}
impl MultiheadMaskAttention {
    pub fn new(num_heads: usize, embed_dim: usize, vb: &VarBuilder) -> Result<Self> {
        assert!(
            embed_dim.is_multiple_of(num_heads),
            "embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        );
        let head_dim = embed_dim / num_heads;
        let mut heads: Vec<MaskedAttentionHead> = Vec::new();
        for i in 0..num_heads {
            heads.push(MaskedAttentionHead::new(embed_dim, head_dim, &vb.pp(i))?);
        }
        let out: Linear = linear(embed_dim, embed_dim, vb.pp("out"))?;

        Ok(Self {
            num_heads,
            heads,
            out,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut tensors: Vec<Tensor> = Vec::new();
        for i in 0..self.num_heads {
            tensors.push(self.heads[i].forward(x)?); // each (Batch Size, Seq length, Head dim)
        }
        let tensor = Tensor::cat(&tensors, 2)?;
        let tensor = self.out.forward(&tensor)?;
        Ok(tensor)
    }
}
