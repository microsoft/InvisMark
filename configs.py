from dataclasses import dataclass
from typing import Tuple

# TODO: reorganize and split the config for different use cases.
@dataclass
class ModelConfig:
    log_dir: str = './logs/'
    ckpt_path: str = './ckpts/'
    saved_ckpt_path: str = ''
    world_size: int = 1
    lr: float = 0.0002
    num_epochs: int = 50
    log_interval: int = 400
    num_encoded_bits: int = 128
    image_shape: Tuple[int, int] = (256, 256)
    num_down_levels: int = 4
    num_initial_channels: int = 32
    batch_size: int = 32
    beta_min: float = 0.0001
    beta_max: float = 10.0
    beta_start_epoch: float = 1
    beta_epochs: int = 15
    warmup_epochs: int = 1
    discriminator_feature_dim: int = 16
    num_discriminator_layers: int = 4
    watermark_hidden_dim: int = 16
    psnr_threshold: float = 55.0
    enc_mode: str = "ecc" 
    ecc_t: int = 16
    ecc_m: int = 8
    num_classes: int = 2
    beta_transform: float = 0.5
    num_noises: int = 2
    noise_start_epoch: int = 20
