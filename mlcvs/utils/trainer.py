from pytorch_lightning import Callback
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
    """Lightning which saves logged metrics into a dictionary. 
       The metrics are recorded at the end of each validation epoch.
    """

    def __init__(self):
        super().__init__()
        self.metrics = { 'epoch' : []}  

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if not trainer.sanity_checking:
            self.metrics['epoch'].append(trainer.current_epoch)
            for key, val in metrics.items():
                val = val.item()
                if key in self.metrics:
                    self.metrics[key].append(val)
                else:
                    self.metrics[key] = [val]