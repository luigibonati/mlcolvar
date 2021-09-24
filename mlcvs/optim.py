"""Utils for NN optimization."""

__all__ = ["EarlyStopping"]

import torch

class EarlyStopping():
    """
    Early stopping the training when the loss does not improve after
    a number of epochs.
    
    Examples
    --------
    early_stopping = EarlyStopping(patience=10,min_delta=0,consecutive=True,log=False,save_best=True)
    while not early_stopping.early_stop:
        train_loss = ...
        valid_loss = ...
        early_stopping(valid_loss,model)

    best_valid_loss = early_stopping.best_valid
    best_model = early_stopping.best_model 
    """

    def __init__(self, patience=5, min_delta=0, consecutive=True, log = False, save_best=True):
        """
        Attributes
        ----------
        early_stop : bool
            check whether to stop the training
        best_model : model / parameters
            Saved model corresponding to best loss
        best_loss : torch.Tensor
            Current best validation loss

        Parameters
        ----------
        patience : int
            how many epochs to wait before stopping when loss is
               not improving (default = 5)
        min_delta : float, optional 
            minimum difference between new loss and old loss to be considered as an improvement (default = 0)
        consecutive: bool, optional
            whether to consider cumulative or consecutive patience
        log: bool, optional
            print info when counter increases
        save_best: bool, optional
            store the best model
       
        """
        self.patience = patience
        self.min_delta = min_delta
        self.consecutive=consecutive
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.log = log
        self.save_best_model= save_best
        self.best_model = None
        self.best_epoch = None
        
    def __call__(self, val_loss, model=None, epoch=None):

        # IF first epoch: initialize
        if self.best_loss is None:
            self.best_loss = val_loss

        # IF valid_loss decreases
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset the counter when starts decreasing again
            if self.consecutive:
                self.counter = 0
            # save model if corresponding option is enabled
            if self.save_best_model:
                self.best_model = model
                if epoch is not None:
                    self.best_epoch = epoch
                    
        # IF valid_loss increases
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.log:
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`:

    .. math:: `\mathtt{new_lr}` = `\mathtt{old_lr}` * `\mathtt{factor}`
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.9, log= False
    ):
        """
        Parameters
        ----------
        optimizer : torch.optimizer
            the optimizer we are using
        patience : int, optional 
            how many epochs to wait before updating the lr (default = 5)
        min_lr: float, optional
            least lr value to reduce to while updating (defaul = 1e-6)
        factor: float, optional 
            factor by which the lr should be updated (default = 0.9)
        log: bool, optional
            print verbose info
        
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=log
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)   