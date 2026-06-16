use candle_core::Device;
use std::sync::OnceLock;

pub fn device() -> &'static Device {
    static DEVICE: OnceLock<Device> = OnceLock::new();
    // DEVICE.get_or_init(|| Device::Cpu)
    DEVICE.get_or_init(|| Device::new_metal(0).expect("metal init fail"))
}
pub const CONTEXT_SIZE: usize = 256;
pub const EMBED_DIM: usize = 256;
pub const NUM_HEADS: usize = 8;
pub const NUM_LAYERS: usize = 4;
pub const BATCH_SIZE: usize = 16;
pub const VOCAB_SIZE: usize = 1024;
