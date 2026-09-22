import lightning
import numpy as np
import torch

from mlcolvar.cvs.unsupervised.autoencoder import AutoEncoderCV
from mlcolvar.data import DictDataset, DictModule


def test_autoencodercv():
    in_features, out_features = 8, 2
    layers = [in_features, 6, 4, out_features]

    # initialize via dictionary
    opts = {
        "encoder": {"activation": "relu"},
    }
    model = AutoEncoderCV(encoder_layers=layers, options=opts)
    print(model)

    # train
    print("train 1 - no weights")
    X = torch.randn(100, in_features)
    dataset = DictDataset({"data": X})
    datamodule = DictModule(dataset)
    trainer = lightning.Trainer(
        max_epochs=1, log_every_n_steps=2, logger=None, enable_checkpointing=False
    )
    trainer.fit(model, datamodule)
    # model.eval()
    X_hat = model(X)

    # test export of decoder_model
    decoder_model = model.get_decoder(return_normalization=True)
    # print(model.encode_decode(X) - decoder_model(X_hat))

    # train with weights
    print("train 2 - weights")
    dataset = DictDataset(
        {"data": torch.randn(100, in_features), "weights": np.arange(100)}
    )
    datamodule = DictModule(dataset)
    trainer = lightning.Trainer(
        max_epochs=1, log_every_n_steps=2, logger=None, enable_checkpointing=False
    )
    trainer.fit(model, datamodule)

    # train with different input and ouput
    print("train 3 - timelagged")
    dataset = DictDataset(
        {"data": torch.randn(100, in_features), "target": torch.randn(100, in_features)}
    )
    datamodule = DictModule(dataset)
    trainer = lightning.Trainer(
        max_epochs=1, log_every_n_steps=2, logger=None, enable_checkpointing=False
    )
    trainer.fit(model, datamodule)