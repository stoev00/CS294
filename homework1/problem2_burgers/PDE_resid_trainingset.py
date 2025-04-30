import torch
import torch.nn.functional as nnF
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from data import BurgersDataset
from model import ConvNet2D
from pde import burgers_pde_residual, burgers_data_loss
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pde_training():
    burgers_train = BurgersDataset(
        'data/Burgers_train_1000_visc_0.01.mat', train=True)
    burgers_validation = BurgersDataset(
        'data/Burgers_test_50_visc_0.01.mat', train=False)

    # Hyperparameters
    lr = 5e-3
    batch_size = 16
    epochs = 40

    # Setup optimizer, model, data loader etc.
    # TODO
    train_loader = DataLoader(burgers_train, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(burgers_validation, batch_size=batch_size, shuffle=False)

    alpha = 0
    model = ConvNet2D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    training_losses = []
    pde_resid_loss = []
    #loss_pde_only = burgers_pde_residual(inputs[:,:,:,0], inputs[:,:,:,1], target_val)
    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        running_loss_pde =0
        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.float()
            loss_pde_only = burgers_pde_residual(inputs[:, :, :, 0], inputs[:, :, :, 1], targets)
            #print(inputs.shape)
            print(loss_pde_only)

            running_loss_pde += loss_pde_only

        pde_resid_loss.append(running_loss_pde)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
    np.save(f'pde_resid_loss.npy',np.array(pde_resid_loss))
    # Validation Loop
    # TODO
    print(f'PDE Residual:',np.mean(pde_resid_loss))

if __name__ == '__main__':
    torch.manual_seed(0)
    pde_training()
