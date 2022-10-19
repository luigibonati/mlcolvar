'''
Test for Deep-TDA module
'''

# Import package, test suite, and other packages as needed
import pytest

import numpy as np

import torch
from mlcvs.tda import deep_tda as test_tda
from mlcvs.utils.io import dataloader_from_file

import string
alphabet = list(string.ascii_uppercase)

# set global variables
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(precision=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_names=['state_A.dat', 'state_B.dat', 'state_C.dat']
n_input = 28

# test for 2 and 3 states in 1D and 2D
@pytest.mark.parametrize("states_and_cvs", [ [2, 1], [3, 1], [3, 2] ])
def test_deeptda(states_and_cvs):
    # get the number of states and cvs for the test run
    states_num = states_and_cvs[0]
    cvs_num = states_and_cvs[1]

    # define targets for the test cases
    # WARNING: these are just unit values for testing! DON'T USE THEM FOR REAL TRAINING !
    if cvs_num == 1:
        if states_num == 2:
            target_centers = [-1, 1]
            target_sigmas = [0.1, 0.1]
        elif states_num == 3:
            target_centers = [-1, 0, 1]
            target_sigmas = [0.1, 0.1, 0.1]
    elif cvs_num == 2:
        target_centers = [[-1, -1], [0, 1], [1, -1]]
        target_sigmas = [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]

    target_centers, target_sigmas = np.array(target_centers), np.array(target_sigmas)
    # initialize loader from file
    train_loader, valid_loader = dataloader_from_file(states_num=states_num,
                                                      files_folder='mlcvs/tests/data/3states_model',
                                                      file_names=['state_A.dat', 'state_B.dat', 'state_C.dat'],
                                                      n_input=28,
                                                      max_rows=1000,
                                                      from_column=1,
                                                      silent=True)

    # initialize simple test model
    model = test_tda.DeepTDA_CV(layers=[n_input, 24, 12],
                                states_num=states_num,
                                cvs_num=cvs_num,
                                target_centers=target_centers,
                                target_sigmas=target_sigmas,
                                device = device
                               )

    # initialize a fast optimizer (lr=0.005)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    model.set_optimizer(opt)

    # quickly train the CV
    model.fit(train_loader, valid_loader, nepochs=100, log_every=10000, standardize_outputs=False)

    # get the results
    results_centers = np.zeros_like(target_centers, dtype=float)
    results_sigmas = np.zeros_like(target_sigmas, dtype=float)
    with torch.no_grad():
        X = next(iter(train_loader))[0].to(device)
        y = next(iter(train_loader))[1].to(device)
        H = model.forward(X)
        for i in range(states_num):
            H_red = H[torch.nonzero(y == i).view(-1)].cpu().numpy()
            results_centers[i] = np.mean(H_red, 0)
            results_sigmas[i] = np.std(H_red, 0)

    # get rough estimate of the errors
    dev_centers = np.sqrt((results_centers - target_centers)**2)
    dev_sigmas = np.sqrt((results_sigmas - target_sigmas) ** 2) * 10 # scale to unit
    # assert we are roughly within a 3% error, this training is nonsense anyway
    check = dev_centers > 3e-2
    assert not np.any(check)
    check = dev_sigmas > 3e-2
    assert not np.any(check)
