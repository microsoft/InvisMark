import numpy as np
import struct
import uuid
import torch
from kornia import color


def compute_reconstruction_loss(inputs, reconstructions, device, recon_type='rgb'):
    if recon_type == 'rgb':
        rec_loss = torch.abs(inputs - reconstructions).mean(dim=[1,2,3])
    elif recon_type == 'yuv':
        reconstructions_yuv = color.rgb_to_yuv((reconstructions + 1) / 2)
        inputs_yuv = color.rgb_to_yuv((inputs + 1) / 2)
        yuv_loss = torch.mean((reconstructions_yuv - inputs_yuv)**2, dim=[2,3])
        yuv_scale = torch.tensor([1,100,100]).unsqueeze(1).float().to(device) # [3,1]
        rec_loss = torch.mm(yuv_loss, yuv_scale).squeeze(1)
    else:
        raise ValueError(f"Unknown recon type {recon_type}")
    return rec_loss

def uuid_to_bits(batch_size):
    uid = [uuid.uuid4() for _ in range(batch_size)]
    seq = np.array([[n for n in u.bytes] for u in uid], dtype=np.uint8)
    bits = torch.Tensor(np.unpackbits(seq, axis=1)).to(torch.float32)
    strs = [str(u) for u in uid]
    return bits, strs


def uuid_to_bytes(batch_size):
    return [uuid.uuid4().bytes for _ in range(batch_size)]


def bits_to_uuid(bits, threshold=0.5):
        bits = np.array(bits) >= threshold
        nums = np.packbits(bits.astype(np.int64), axis=-1)
        res = []
        for j in range(nums.shape[0]):
            bstr = b''
            for i in range(nums.shape[1]):
                bstr += struct.pack('>B', nums[j][i])
            res.append(str(uuid.UUID(bytes=bstr)))
        return res
