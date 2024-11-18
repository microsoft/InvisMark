import logging
import sys
import os
from collections import defaultdict

import model
import noise
import metrics
import utils

import numpy as np
from focal_frequency_loss import FocalFrequencyLoss as ffl
import lpips
from PIL import ImageFile
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch.nn.functional as thf

# Necessary for avoiding reading truncated images from dataloader.
ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class Watermark(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()
        
        self.config = config
        self.device = device
        self.encoder = model.Encoder(config).to(device)
        self.discriminator = model.DisResNet(config).to(device)
        self.decoder = model.Extractor(config).to(device)
        self.noiser =  noise.Noiser(num_transforms=1, device=self.device).to(device)
        self.opt_encoder = torch.optim.AdamW(self.encoder.parameters(), lr=config.lr)
        self.opt_decoder = torch.optim.AdamW(self.decoder.parameters(), lr=config.lr)
        self.opt_discriminator = torch.optim.AdamW(self.discriminator.parameters(), lr=config.lr)
        # m should be adjusted according to the number of encoded bits.
        if self.config.enc_mode == "ecc":
            self.bchecc = model.BCHECC(t=config.ecc_t, m=config.ecc_m)
            logger.info(f"enc_bits: {self.config.num_encoded_bits}, data_bytes: {self.bchecc.data_bytes}")
    
        # lpips: image should be RGB, IMPORTANT: normalized to [-1,1]
        self.lpips_loss_fn = lpips.LPIPS(net='vgg')
        self.lpips_loss_fn.cuda()
        self.ffl_fn = ffl(loss_weight=1.0, alpha=1.0)
        self.bce_loss_fn = nn.BCELoss()
        self.cur_epoch = 0
        self.cur_step = 0
        self.train_encoder = True
    
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.losses = defaultdict(float)
        self.eval_psnr = 0.0
        self.train_bit_accuracy = defaultdict(float)

        self.transform = transforms.Compose([
                transforms.Resize(self.config.image_shape),
            ])

    def train(self, train_data, eval_data=None, ckpt_path=None):
        if ckpt_path:
            logger.info(f"Loading model from ckpt: {ckpt_path}")
            self.load_model(ckpt_path)
            self.config.num_noises = self.config.num_noises
            logger.info(f"Loaded model from epoch num_noises:{self.config.num_noises, self.config.beta_transform}")

        fixed_batch = next(iter(train_data))
        for i in range(self.config.num_epochs):
            logger.info(f"Training for epoch: {self.cur_epoch}, beta_quality: {self._update_beta()}, train_encoder: {self.train_encoder}")
            if self.eval_psnr > self.config.psnr_threshold:
                self.train_encoder = False

            if i < self.config.warmup_epochs:
                self._train_one_epoch(train_data, fixed_batch=fixed_batch)
            else:
                self._train_one_epoch(train_data)

            if eval_data: self._validate(eval_data)
            self._save_model()
            self.cur_epoch += 1

    def eval(self, ckpt_path, input_images, secrets=None):
        self.load_model(ckpt_path)
        if secrets:
            assert secrets.shape[0] == input_images.shape[0], "Secrets and inputs need share the same batch dim."
        else:
            secrets = torch.randint(0, 2, (input_images.shape[0], self.config.num_encoded_bits), 
                    device = self.device).type(torch.float32)
        enc_images = self.encoder(input_images, secrets)
        dec_secrets = self.decoder(enc_images)
        return enc_images, secrets, dec_secrets

    def _encode(self, inputs, secret):
        # resize to smaller resolution for encoding.
        resize_inputs = self.transform(inputs).to(self.device)            
        encoded_output = self.encoder(resize_inputs, secret)
        orig_diff = transforms.Resize(inputs.shape[-2:])(encoded_output - resize_inputs).to('cpu')
        output = torch.clamp(inputs + orig_diff, min=-1.0, max=1.0)
        return output.to(self.device), resize_inputs, encoded_output
    
    def _decode(self, images):
        trans_images = self.transform(images)
        return self.decoder(trans_images)

    def _update_beta(self):
        cur_beta_epoch = min(max(0, self.cur_epoch - self.config.beta_start_epoch), self.config.beta_epochs-1)
        beta_schedule = np.logspace(np.log10(self.config.beta_min), np.log10(self.config.beta_max), self.config.beta_epochs)
        return beta_schedule[cur_beta_epoch]

    def _train_one_epoch(self, dataloader, fixed_batch=None):
        for data in dataloader:
            self.encoder.train()
            self.decoder.train()
            if self.train_encoder:
                self.opt_encoder.zero_grad()   
            self.opt_decoder.zero_grad()
            self.opt_discriminator.zero_grad()
            self.cur_step += 1
            if fixed_batch: data = fixed_batch

            # Generate random bit array for watermarking.
            secret = self._generate_secret(data[0].shape[0], self.device)
            total_loss = self._loss_fn(data[0], secret)
            total_loss.backward()
    
            if self.train_encoder:
                self.opt_encoder.step()
            self.opt_decoder.step()
            self.opt_discriminator.step()
            
            logger.info(f"""epoch: {self.cur_epoch}, cur_step: {self.cur_step}, total_loss: {total_loss}""")
            if self.cur_step % self.config.log_interval == 0:
                metrics = self._calculate_metric(data[0], secret)
                self._log_metrics(metrics, "Train")
                self._log_metrics(self.losses, "Train")

    def _loss_fn(self, data, secret):
        if self.train_encoder:
            final_output, enc_inputs, enc_output = self._encode(data, secret)
        else:
            with torch.no_grad():
                final_output, enc_inputs, enc_output = self._encode(data, secret)
        extracted_secret = self._decode(final_output)

        # TODO: update the image reconstruction loss using the orig_image and final_output.
        self.losses['mse_loss'] = utils.compute_reconstruction_loss(enc_inputs, enc_output, self.device, recon_type='yuv').mean()
        self.losses['lpips_loss'] = self.lpips_loss_fn(enc_inputs, enc_output).mean()
        self.losses['bce_loss'] = self.bce_loss_fn(extracted_secret, secret)
        self.losses['ffl_loss'] = self.ffl_fn(enc_inputs, enc_output).mean()
        self.losses['discriminator_loss'] = torch.ones(1, device=self.device)
        
        # Enable GAN loss after beta ramp up.
        if self.cur_epoch >=  self.config.beta_start_epoch:
            real_loss = -torch.mean(self.discriminator(enc_inputs))
            fake_loss = torch.mean(self.discriminator(enc_output))
            self.losses['discriminator_loss'] += real_loss + fake_loss
            # adding noise in the discriminator loss

        if not self.train_encoder or self.cur_epoch >= self.config.noise_start_epoch:
            sorted_keys = [[k] for k, _ in sorted(self.train_bit_accuracy.items(), key=lambda x: x[1])]
            # select top-k important noises to include in the loss function.
            for key in [None] + sorted_keys[:self.config.num_noises]:
                trans_output = self.noiser(final_output, key)
                if key is not None:
                    logger.info(f"Adding noise: {key}")
                extracted_secret = self._decode(trans_output)
                # TODO: Adjust the importance of different transformation
                self.losses['bce_loss'] += self.config.beta_transform * self.bce_loss_fn(extracted_secret, secret)

        if self.train_encoder:
            beta_quality = self._update_beta()
            return beta_quality * (self.losses['mse_loss'] + self.losses['lpips_loss'] + \
                self.losses['discriminator_loss'] + self.losses['ffl_loss']) + self.losses['bce_loss']
        else:
            return self.losses['bce_loss']


    def _generate_secret(self, batch_size, device):
        if self.config.enc_mode == "uuid":
            bits, _ = utils.uuid_to_bits(batch_size)
        elif self.config.enc_mode == "ecc":
            assert self.config.num_encoded_bits == 256, "Encode 256 bits in ecc mode"
            bits = self.bchecc.batch_encode(batch_size)
        else:
            raise "secret enc_mode is not supported! choose between uuid and ecc."
        return bits[:, :self.config.num_encoded_bits].to(device)
    
    @torch.no_grad()
    def _calculate_metric(self, orig_images, secrets, prefix='Train'):
        self.encoder.eval()
        self.decoder.eval()
        final_output, _, _= self._encode(orig_images, secrets)
        metric = defaultdict(float)
        # Image pixel values should be within [-1, 1], i.e. data_range = 2.0
        metric['psnr'] = metrics.image_psnr(orig_images, final_output.cpu())
        metric['ssim'] = metrics.image_ssim(final_output.cpu(), orig_images)
        for key in noise.supported_transforms(final_output.shape[-2:]):
            trans_output = self.noiser(final_output, [key])
            extracted_secret = self._decode(trans_output)
            bit_accuracy_trans = metrics.bit_accuracy(secrets, extracted_secret)
            metric[f'BitAcc-{key}'] = bit_accuracy_trans
            if prefix == 'Train': self.train_bit_accuracy[key] = bit_accuracy_trans
            if self.config.enc_mode == 'ecc':
                cor_secret = self.bchecc.batch_decode_ecc(extracted_secret).cpu()
                metric[f'DataBitAcc-{key}'] = metrics.bit_accuracy(
                    cor_secret[:, :-self.bchecc.bch.ecc_bytes*8],
                    secrets[:, :-self.bchecc.bch.ecc_bytes*8].cpu())
        return metric

    def _log_metrics(self, metrics, prefix='Train'):
        for key in metrics:
            self.writer.add_scalar(f"{prefix}/{key}", metrics[key], self.cur_step)

    def _log_images(self, orig_images, secrets, prefix='Train'):
        final_output, _, _= self._encode(orig_images, secrets)
        grid = vutils.make_grid(orig_images[0], normalize=True, value_range=(-1., 1.))
        self.writer.add_image(f"{prefix}/input_images", grid, self.cur_step)
        grid = vutils.make_grid(final_output[0], normalize=True, value_range=(-1., 1.))
        self.writer.add_image(f"{prefix}/encoded_images", grid, self.cur_step)         
        grid = vutils.make_grid(10.0*(orig_images[0]-final_output[0].cpu()), normalize=True, value_range=(-1., 1.))
        self.writer.add_image(f"{prefix}/image_diff_x10", grid, self.cur_step)


    @torch.no_grad()
    def _validate(self, eval_data, num_batches=100):
        avg_metrics = defaultdict(float)
        for i, eval_batch in enumerate(eval_data):
            secrets = self._generate_secret(eval_batch[0].shape[0], self.device)
            batch_metric = self._calculate_metric(eval_batch[0], secrets, "Eval")
            for k, v in batch_metric.items(): avg_metrics[k] += v
            if i>=num_batches: break
        for k in avg_metrics:
            avg_metrics[k] = avg_metrics[k] / (i + 1.0)
        self.eval_psnr = avg_metrics['psnr']
        self._log_metrics(avg_metrics, "Eval")
        self._log_images(eval_batch[0], secrets, "Eval")
    

    def load_model(self, ckpt_path):
        state_dict = torch.load(ckpt_path)
        logger.info(f"Loading model from epoch:{state_dict['cur_epoch']}")
        self.encoder.load_state_dict(state_dict['encoder_state_dict'])
        self.encoder.train()
        self.decoder.load_state_dict(state_dict['decoder_state_dict'])
        self.decoder.train()
        self.discriminator.load_state_dict(state_dict['discriminator_state_dict'])
        self.discriminator.train()
        self.opt_encoder.load_state_dict(state_dict['opt_encoder_state_dict'])
        self.opt_decoder.load_state_dict(state_dict['opt_decoder_state_dict'])
        self.opt_discriminator.load_state_dict(state_dict['opt_discriminator_state_dict'])
        self.cur_epoch = state_dict['cur_epoch']
        self.cur_step = state_dict['cur_step']
        self.config = state_dict['config']
    
    def _save_model(self):
        if not os.path.exists(self.config.ckpt_path):
            os.makedirs(self.config.ckpt_path)
        torch.save({
                'encoder_state_dict': self.encoder.state_dict(), 
                'decoder_state_dict': self.decoder.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'opt_encoder_state_dict': self.opt_encoder.state_dict(),
                'opt_decoder_state_dict': self.opt_decoder.state_dict(),
                'opt_discriminator_state_dict': self.opt_discriminator.state_dict(),
                'cur_epoch': self.cur_epoch,
                'cur_step': self.cur_step,
                'config': self.config,
                }, f"{self.config.ckpt_path}/model-{self.cur_epoch:04d}.ckpt")
