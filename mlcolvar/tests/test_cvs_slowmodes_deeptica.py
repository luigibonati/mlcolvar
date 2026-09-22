import lightning
import torch

from mlcolvar.core.nn import FeedForward
from mlcolvar.core.nn.graph.schnet import SchNetModel
from mlcolvar.cvs.timelagged.deeptica import DeepTICA
from mlcolvar.data import DictModule
from mlcolvar.data.graph.utils import create_test_graph_input
from mlcolvar.utils.timelagged import create_timelagged_dataset

def test_deep_tica():
    # create dataset
    X = torch.randn((10000, 2))
    dataset = create_timelagged_dataset(X, lag_time=1)
    datamodule = DictModule(dataset, batch_size=10000)

    # create cv
    print()
    print('NORMAL')
    print()
    layers = [2, 10, 10, 2]
    model = DeepTICA(layers, n_cvs=1)

    # change loss options
    model.loss_fn.mode = "sum2"

    # create trainer and fit
    trainer = lightning.Trainer(
        max_epochs=1, log_every_n_steps=2, logger=None, enable_checkpointing=False
    )
    trainer.fit(model, datamodule)

    model.eval()
    with torch.no_grad():
        s = model(X).numpy()
    print(X.shape, "-->", s.shape)


    print()
    print('EXTERNAL')
    print()
    ff_model = FeedForward(layers=layers)
    model = DeepTICA(ff_model, n_cvs=1)

    # change loss options
    model.loss_fn.mode = "sum2"

    # create trainer and fit
    trainer = lightning.Trainer(
        max_epochs=1, log_every_n_steps=2, logger=None, enable_checkpointing=False
    )
    trainer.fit(model, datamodule)

    model.eval()
    with torch.no_grad():
        s = model(X).numpy()
    print(X.shape, "-->", s.shape)

    
    # gnn external
    print()
    print('GNN')
    print()

    gnn_model = SchNetModel(n_out=2, cutoff=0.1, atomic_numbers=[1, 8])
    model = DeepTICA(gnn_model, n_cvs=1)

    # change loss options
    model.loss_fn.mode = "sum2"

    # create trainer and fit
    trainer = lightning.Trainer(
        max_epochs=1, log_every_n_steps=2, logger=False, enable_checkpointing=False, enable_model_summary=False,
    )

    dataset = create_test_graph_input(output_type='dataset', n_samples=200, n_states=2)
    lagged_dataset = create_timelagged_dataset(dataset, logweights=torch.randn(len(dataset)))
    
    datamodule = DictModule(dataset=lagged_dataset)
    trainer.fit(model, datamodule)

    model.eval()
    with torch.no_grad():
        example_input_graph_test = create_test_graph_input(output_type='example', n_atoms=4, n_samples=3, n_states=2)
        s = model(example_input_graph_test).numpy()
    print(X.shape, "-->", s.shape)