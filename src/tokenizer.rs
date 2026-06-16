use crate::config::VOCAB_SIZE;
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Result, Tokenizer, TokenizerBuilder};

pub const TOKENIZER_PATH: &str = "tokenizer.json";

/// Train a byte-level BPE tokenizer on `train_file` and save it to `TOKENIZER_PATH`.
///
/// Base alphabet is the 256 bytes, so the BPE learns `VOCAB_SIZE - 256 - <specials>`
/// merges. Only `<|endoftext|>` is added — a decoder-only LM needs no pad/unk/mask,
/// and byte-level can encode any input so OOV is impossible.
pub fn train_tokenizer(train_file: &str) -> Result<()> {
    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(VOCAB_SIZE)
        .min_frequency(2)
        .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
        .special_tokens(vec![AddedToken::from(String::from("<|endoftext|>"), true)])
        .build();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(NFC))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()?;

    tokenizer
        .train_from_files(&mut trainer, vec![train_file.to_string()])?
        .save(TOKENIZER_PATH, false)?;

    Ok(())
}

/// Load the trained tokenizer from `TOKENIZER_PATH`.
pub fn load_tokenizer() -> Result<Tokenizer> {
    Tokenizer::from_file(TOKENIZER_PATH)
}
