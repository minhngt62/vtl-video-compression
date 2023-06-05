import torch.utils.data as data
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelPruning

from typing import Any, Dict, Optional, Union
import os
from pathlib import Path
PRJ_ROOT = Path(__file__).parent.parent.parent.resolve()

def train(
    model: L.LightningModule,
    dataset: data.Dataset,
    compress_params: Dict[str, Any] = dict(
        parameters_to_prune=None,
        prune_ratio=0.4,
        prune_at_epochs=[0.],
    ),
    trainer_params: Dict[str, Any] = dict(
        default_root_dir=os.path.join(PRJ_ROOT, "configs", "nerv"),
        max_epochs=150,
        log_every_n_steps=15,
    ),
    loader_params: Dict[str, Any] = dict(
        batch_size=1,
        shuffle=False,
        num_worker=4,
        pin_memory=True,
        drop_last=False,
    )
):  
    prune_at_epochs = [int(i * trainer_params["max_epochs"]) for i in compress_params["prune_at_epochs"]]
    def compute_amount(epoch):
        if len(prune_at_epochs) == 1:
            if epoch == prune_at_epochs[0]:
                return compress_params["prune_ratio"]
        else:
            base_prune_ratio = compress_params["prune_ratio"] ** (1 / len(prune_at_epochs))
            for i in range(len(prune_at_epochs)-1):
                if prune_at_epochs[i] <= epoch < prune_at_epochs[i+1]:
                    return base_prune_ratio ** (i+2)

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        callbacks=[
            ModelCheckpoint(mode="max", monitor="val_psnr"),
            LearningRateMonitor("epoch"),
            ModelPruning(
                parameters_to_prune=compress_params["parameters_to_prune"],
                pruning_fn="l1_unstructured",
                amount=compute_amount,
                use_global_unstructured=True
            )
        ],
        log_every_n_steps=5,
        **trainer_params,
    )
    trainer.logger._log_graph = False  # if True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # optional logging argument that we don't need

    # build data loaders
    train_loader = data.DataLoader(dataset, **loader_params)
    val_loader = data.DataLoader(dataset, **loader_params)

    # train the model
    L.seed_everything(42)
    trainer.fit(model, train_loader, val_loader) 

    # test the model
    model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    test_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    return model, test_result
    