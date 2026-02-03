from lightning import Callback
import copy


class SimpleMetricsCallback(Callback):
    """Lightning callback which append logged metrics to a list.
    The metrics are recorded at the end of each validation epoch.
    """

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            metrics = copy.deepcopy(trainer.callback_metrics)
            self.metrics.append(metrics)


class MetricsCallback(Callback):
    """Lightning callback which saves logged metrics into a dictionary.
    The metrics are recorded at the end of each validation epoch.
    """

    def __init__(self):
        super().__init__()
        self.metrics = {"epoch": []}

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if not trainer.sanity_checking:
            self.metrics["epoch"].append(trainer.current_epoch)
            for key, val in metrics.items():
                val = val.item()
                if key in self.metrics:
                    self.metrics[key].append(val)
                else:
                    self.metrics[key] = [val]
            has_scheduler = bool(getattr(trainer, "lr_scheduler_configs", None))
            if has_scheduler and "lr" not in metrics and trainer.optimizers:
                lrs = [pg["lr"] for opt in trainer.optimizers for pg in opt.param_groups]
                lr_val = lrs[0] if len(lrs) == 1 else lrs
                if "lr" in self.metrics:
                    self.metrics["lr"].append(lr_val)
                else:
                    self.metrics["lr"] = [lr_val]


def test_metrics_callbacks():
    import torch
    import lightning
    from mlcolvar.cvs import AutoEncoderCV
    from mlcolvar.data import DictDataset, DictModule

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
