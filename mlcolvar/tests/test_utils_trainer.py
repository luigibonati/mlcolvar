import lightning
import torch

from torch.optim.lr_scheduler import StepLR

from mlcolvar.cvs import AutoEncoderCV
from mlcolvar.data import DictDataset, DictModule
from mlcolvar.utils.trainer import MetricsCallback, SimpleMetricsCallback


def test_metrics_callbacks():

    X = torch.rand((100, 2))
    dataset = DictDataset({"data": X})
    datamodule = DictModule(dataset)

    model = AutoEncoderCV([2, 2, 1])
    metrics = SimpleMetricsCallback()
    trainer = lightning.Trainer(
        max_epochs=1,
        log_every_n_steps=2,
        logger=None,
        enable_checkpointing=False,
        callbacks=metrics,
    )
    trainer.fit(model, datamodule)

    model = AutoEncoderCV([2, 2, 1])
    metrics = MetricsCallback()
    trainer = lightning.Trainer(
        max_epochs=1,
        log_every_n_steps=2,
        logger=None,
        enable_checkpointing=False,
        callbacks=metrics,
    )
    trainer.fit(model, datamodule)

    model = AutoEncoderCV(
        [2, 2, 1],
        options={
            "lr_scheduler": {
                "scheduler": StepLR,
                "step_size": 1,
                "gamma": 0.5,
            }
        },
    )
    metrics = MetricsCallback()
    trainer = lightning.Trainer(
        max_epochs=1,
        log_every_n_steps=2,
        logger=None,
        enable_checkpointing=False,
        callbacks=metrics,
    )
    trainer.fit(model, datamodule)
    assert "lr" in metrics.metrics
