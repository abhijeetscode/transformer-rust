use crate::transformer_block::TransformerBlock;
use candle_core::{Result, Tensor};
use candle_nn::{
    Embedding, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder, embedding, layer_norm,
    linear,
};

pub struct GPTModel {
    vocab_size: usize,
    embed_dim: usize,
    num_heads: usize,
    max_seq_length: usize,
    token_embedding: Embedding,
    position_embedding: Embedding,
    transformer_blocks: Vec<TransformerBlock>,
    lnorm: LayerNorm,
    output_layer: Linear,
    num_layers: usize,
}
impl GPTModel {
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        num_heads: usize,
        max_seq_length: usize,
        num_layers: usize,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let mut trf_blocks: Vec<TransformerBlock> = Vec::new();
        for i in 0..num_layers {
            trf_blocks.push(TransformerBlock::new(&vb.pp(i), embed_dim, num_heads)?);
        }
        Ok(Self {
            vocab_size,
            embed_dim,
            num_heads,
            max_seq_length,
            token_embedding: embedding(vocab_size, embed_dim, vb.pp("token_embedding"))?,
            position_embedding: embedding(max_seq_length, embed_dim, vb.pp("position_embedding"))?,
            transformer_blocks: trf_blocks,
            lnorm: layer_norm(embed_dim, LayerNormConfig::default(), vb.pp("lnorm"))?,
            output_layer: linear(embed_dim, vocab_size, vb.pp("output_layer"))?,
            num_layers,
        })
    }
    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (_, seq_length) = x.dims2()?;
        let pos_ids = Tensor::arange(0u32, seq_length as u32, x.device())?;
        let mut enriched_embedding: Tensor = self
            .token_embedding
            .forward(x)?
            .broadcast_add(&self.position_embedding.forward(&pos_ids)?)?;

        for blk in self.transformer_blocks.iter() {
            enriched_embedding = blk.forward(&enriched_embedding, mask)?;
        }
        enriched_embedding = self.lnorm.forward(&enriched_embedding)?;
        enriched_embedding = self.output_layer.forward(&enriched_embedding)?;
        Ok(enriched_embedding)
    }
}
