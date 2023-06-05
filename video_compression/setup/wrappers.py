import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from dahuffman import HuffmanCodec

from ..models import Nerv, PositionalEncoding
from .metrics import loss_fn, psnr_fn
from .compress import quantize_weights

class LtNerv(L.LightningModule):
    def __init__(
        self,
        stem_dim_num="512_1",
        fc_hw_dim="9_16_26",
        pe_embed="1.25_40",
        stride_list=[5, 2, 2, 2, 2],
        expansion=1,
        reduction=2,
        lower_width=96,
        num_blocks=1,
        bias=True,
        sin_res=True,
        sigmoid=True,

        lr0=5e-4,
        betas=(0.5, 0.999),
        weight_decay=0,
        warmup_epochs=30, # 0.2 * 150
        loss_alpha=0.7,

        quant_bit=8,
        quant_axis=0,     
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr0, self.betas, self.weight_decay = lr0, betas, weight_decay
        self.warmup_epochs = warmup_epochs
        self.loss_alpha = loss_alpha
        self.quant_bit, self.quant_axis = quant_bit, quant_axis

        self.pe = PositionalEncoding(pe_embed=pe_embed)
        self.model = Nerv(
            stem_dim_num=stem_dim_num,
            fc_hw_dim=fc_hw_dim,
            embed_length=self.pe.embed_length,
            stride_list=stride_list,
            expansion=expansion,
            reduction=reduction,
            lower_width=lower_width,
            num_blocks=num_blocks,
            bias=bias,
            sin_res=sin_res,
            sigmoid=sigmoid
        )
    
    def forward(self, x):
        embedding = self.pe(x)
        return self.model(embedding)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(), 
            betas=self.betas,
            weight_decay=self.weight_decay,
            lr=self.lr0
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.warmup_epochs, T_mult=1
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        frame_inds, frames = batch
        pred_frames = self.model(self.pe(frame_inds))
        frames = [F.adaptive_avg_pool2d(frames, x.shape[-2:]) for x in pred_frames]
        
        losses = [loss_fn(pred, target, alpha=self.loss_alpha) for pred, target in zip(pred_frames, frames)]
        loss = sum(losses)

        psnr = psnr_fn(pred_frames, frames)
        
        self.log_dict({"%s_loss" % mode: loss, "%s_psnr" % mode: psnr}, prog_bar=True)
        return loss, psnr

    def training_step(self, batch, batch_idx):
        loss, psnr = self._calculate_loss(batch)
        return {"loss": loss, "psnr": psnr}

    def validation_step(self, batch, batch_idx):
        quantized_ckt, _ = quantize_weights(self.model, self.quant_bit, self.quant_axis)
        self.model.load_state_dict(quantized_ckt)

        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")