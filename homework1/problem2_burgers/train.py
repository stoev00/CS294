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


def train():
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


    model = ConvNet2D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    training_losses = []
    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.float()
            print(inputs.shape)
            optimizer.zero_grad()
            predicted = model(inputs)

            #print(targets.shape)
            loss1 = burgers_data_loss(predicted, targets)

            loss2 = burgers_pde_residual(inputs[:,:,:,0], inputs[:,:,:,1], predicted)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(train_loader.dataset)
        training_losses.append(epoch_loss)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
    #np.save(f'training-losses.npy', np.array(training_losses))
    #torch.save(model, 'model.pth')
    # Validation Loop
    # TODO
    model.eval()
    val_loss = 0.0
    num_val_batches = 0
    with torch.no_grad():
        for x_val, target_val in validation_loader:
            x_val, target_val = x_val.to(device), target_val.to(device)
            x_val = x_val.float()
            val_pred = model(x_val)
            loss_val = burgers_data_loss(val_pred, target_val)
            val_loss += loss_val.item()
            num_val_batches += 1

    print(f"Validation Loss: {val_loss / num_val_batches:.4f}")
    #np.save(f'training-rmses.npy', np.array(val_loss / num_val_batches))
    '''
    plt.figure()
    plt.plot(range(1, epochs), training_losses, label='Data Loss')
    plt.title('Data Loss vs. Epoch')
    plt.xlabel('Epochs')r
    plt.ylabel('Data Loss')
    plt.legend()

    plt.figure()
    plt.contourf()
    plt.plot(x_val[1,:,:], target_val[1,:,:], label='Data Loss')
    plt.title('Data Loss vs. Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Data Loss')
    plt.legend()
    '''
if __name__ == '__main__':
    torch.manual_seed(0)
    train()
