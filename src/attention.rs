use candle_core::{Result, Tensor};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{Linear, Module, VarBuilder, linear};

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
    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // B = Batch Size, L = Max Seq Length, E = embed_dim, O = head dim (generally E//number of heads)
        // x is (B, L, E), mask is (max_seq_length, max_seq_length) additive: 0 allowed / -inf future
        let q = self.q.forward(x)?; // (B, L, O)
        let k = self.k.forward(x)?; // (B, L, O)
        let v = self.v.forward(x)?; // (B, L, O)
        let attn_scores = q.matmul(&k.transpose(1, 2)?)?; // (B, L, L) -> Attention score are always L*L
        let (_, seq_length, _) = attn_scores.dims3()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scaled_attn_scores: Tensor = (attn_scores * scale)?;
        // slice the prebuilt mask to this batch's seq length, then add (broadcast over batch)
        let m = mask.narrow(0, 0, seq_length)?.narrow(1, 0, seq_length)?; // (L, L)
        let masked: Tensor = scaled_attn_scores.broadcast_add(&m)?; // future positions -> -inf
        let attn_weights = softmax_last_dim(&masked)?;
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
    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let mut tensors: Vec<Tensor> = Vec::new();
        for i in 0..self.num_heads {
            tensors.push(self.heads[i].forward(x, mask)?); // each (Batch Size, Seq length, Head dim)
        }
        let tensor = Tensor::cat(&tensors, 2)?;
        let tensor = self.out.forward(&tensor)?;
        Ok(tensor)
    }
}
