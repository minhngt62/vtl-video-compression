import torch.nn.functional as F
from torch import Tensor
import torch
from typing import Iterable
from pytorch_msssim import ms_ssim, ssim

def psnr_fn(
    output_lst: Iterable[Tensor], 
    target_lst: Iterable[Tensor]
):
    psnr_lst = []
    for output, target in zip(output_lst, target_lst):
        l2_loss = F.mse_loss(output.detach(), target.detach(), reduction="mean")
        psnr = -10 * torch.log10(l2_loss)
        psnr = psnr.view(1, 1).expand(output.size(0), -1)
        psnr_lst.append(psnr)
    psnr = torch.cat(psnr_lst, dim=1)
    return psnr

def msssim_fn(
    output_lst: Iterable[Tensor], 
    target_lst: Iterable[Tensor]
):
    msssim_list = []
    for output, target in zip(output_lst, target_lst):
        if output.size(-2) >= 160:
            msssim = ms_ssim(output.float().detach(), target.detach(), data_range=1, size_average=True)
        else:
            msssim = torch.tensor(0).to(output.device)
        msssim_list.append(msssim.view(1))
    msssim = torch.cat(msssim_list, dim=0) #(num_stage)
    msssim = msssim.view(1, -1).expand(output_lst[-1].size(0), -1) #(batchsize, num_stage)
    return msssim

def loss_fn(pred, target, alpha):
    return alpha * torch.mean(torch.abs(pred - target)) + (1 - alpha) * (1 - ssim(pred, target, data_range=1, size_average=True))