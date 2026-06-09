use candle_core::Device;
use std::sync::OnceLock;

pub fn device() -> &'static Device {
    static DEVICE: OnceLock<Device> = OnceLock::new();
    // DEVICE.get_or_init(|| Device::Cpu)
    DEVICE.get_or_init(|| Device::new_metal(0).expect("metal init fail"))
}
pub const CONTEXT_SIZE: usize = 1024;
pub const ENCODING_NAME: &str = "r50k_base";
