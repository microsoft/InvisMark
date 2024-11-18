import logging 
import random

import torch
import numpy as np
from torch import nn
from torchvision.transforms import v2
import kornia.augmentation as aug

logger = logging.getLogger(__name__)


# Medium level noise
def supported_transforms(image_size):
    return {
    'HFlip': aug.RandomHorizontalFlip(p=1.0),
    'VFlip': aug.RandomVerticalFlip(p=1.0),
    "Rotation":  aug.RandomRotation(degrees=[0.0, 10.0], p=1.0),
    #"Rotation":  aug.RandomRotation(degrees=[0.0, 45.0], p=1.0),
    'Perspective': aug.RandomPerspective(distortion_scale=0.1, p=1.0),
    'RandomResizedCrop':  aug.RandomResizedCrop(size=image_size, scale=(0.75, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
    #'RandomResizedCrop':  aug.RandomResizedCrop(size=image_size, scale=(0.5, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),

    'Brighter':  aug.RandomBrightness(brightness=(1.25, 1.25)), # 0.5-1.5
    'Darker':  aug.RandomBrightness(brightness=(0.75, 0.75)), # 0.5-1.5
    'Contrast_p': aug.RandomContrast(contrast=(1.25, 1.25)), # 0.5 - 1.5
    'Contrast_m': aug.RandomContrast(contrast=(0.75, 0.75)), # 0.5 - 1.5
    'Saturation_p': aug.RandomSaturation(saturation=(1.25, 1.25)), #
    'Saturation_m': aug.RandomSaturation(saturation=(0.75, 0.75)), #
    'BoxBlur': aug.RandomBoxBlur(kernel_size=(5, 5), p= 1.0), # 7.
    'GaussianBlur': aug.RandomGaussianBlur(kernel_size=(5, 5), sigma=(1.0, 1.5), p=1.0), # 7, 0.1-2.0
    'GaussianNoise': aug.RandomGaussianNoise(mean=0.0, std=0.04, p=1.0), # 0.08
    "Jpeg": v2.JPEG(quality=[50, 50]), # it is expected to have dtype uint8, on CPU, and have [..., 3 or 1, H, W] shape 40
    'RandomErasing': aug.RandomErasing(scale=(0.02, 0.1), ratio=(0.5, 1.5), p=1.0),
    "Jiggle":  aug.ColorJiggle(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    "Posterize": aug.RandomPosterize(bits=4),
    "RGBShift": aug.RandomRGBShift(r_shift_limit=0.05, g_shift_limit=0.05, b_shift_limit=0.05, p=1.0),
    }

class Noiser(nn.Module):
    def __init__(self, num_transforms, device):
        super().__init__()
        self.device = device
        self.num_transforms = num_transforms
        keys = list(supported_transforms((256, 256)).keys())
        self.geo_transforms = keys[:5]
        self.pert_transforms = keys[5:]

    def forward(self, input, noises=None):
        if noises: 
            for key in noises:
                input = self.apply_noise(input, key)
            return input
        for key in random.choices(self.geo_transforms, k = self.num_transforms):
            input = self.apply_noise(input, key)
        for key in random.choices(self.pert_transforms, k = self.num_transforms):
            input = self.apply_noise(input, key)
        return input
    
    def apply_noise(self, input, key):
        transforms = supported_transforms(input.shape[-2:])
        if key not in transforms:
            raise Exception(f"{key} is not a supported image transformation.")
        # JPEG dtype uint8, on CPU, and have [..., 3 or 1, H, W] shape
        if key == "Jpeg":
            input = ((input + 1.0)*127.5).to(torch.uint8).to('cpu')
            output = transforms[key](input) 
            output = (output / 127.5 - 1.0).to(self.device)
            return output
        # Kornia assumes image data range [0, 1]
        else:
            input = (input + 1.0) / 2.0
            output = transforms[key](input)
            return output * 2.0 - 1.0
