
import torch
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure


def bit_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5):
    assert y_true.size() == y_pred.size()
    return torch.Tensor([(y_pred >= threshold).eq(y_true >= 0.5).sum().float().item() / y_pred.numel()])


def image_psnr(preds, targets, data_range=2.0):
    psnr =  PeakSignalNoiseRatio(data_range=data_range)
    return psnr(preds, targets)
                

def image_ssim(preds, targets, data_range=2.0):
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range)
    return ssim(preds, targets)