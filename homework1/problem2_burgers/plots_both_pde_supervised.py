import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as nnF
import torch.optim as optim

from train import *
from pde import burgers_data_loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 3
burgers_validation = BurgersDataset(
        'data/Burgers_test_50_visc_0.01.mat', train=False)
training_losses = np.load("training-both_losses.npy")
#model = torch.load('model.pth')
model = torch.load('model_alpha_0_00067.pth',map_location=torch.device('cpu'))
model.eval()
validation_loader = DataLoader(burgers_validation, batch_size=batch_size, shuffle=False)
#print(validation_loader)
val_loss = 0.0
num_val_batches = 0

with torch.no_grad():
    for x_val, target_val in validation_loader:
        x_val, target_val = x_val.to(device), target_val.to(device)
        x_val = x_val.float()
        val_pred = model(x_val)
        print(x_val.shape)
        loss_val = burgers_data_loss(val_pred, target_val)
        val_loss += loss_val.item()
        num_val_batches += 1
        break
print(val_pred.shape,target_val.shape)
print(num_val_batches)

fig, (ax1, ax2) = plt.subplots(1,2)
countour1 = ax1.contourf(val_pred[0,:,:])
countour1_t = ax2.contourf(target_val[0,:,:])
ax1.set_title('Predicted Solution PDE and Supervision, alpha = 1')
ax2.set_title('Target Solution')
plt.tight_layout()
plt.show(block=False)  # Use non-blocking show

fig2, (ax1, ax2) = plt.subplots(1,2)
countour2 = ax1.contourf(val_pred[1,:,:])
countour2_t = ax2.contourf(target_val[1,:,:])
ax1.set_title('Predicted Solution PDE and Supervision, alpha = 1')
ax2.set_title('Target Solution')
plt.tight_layout()
plt.show(block=False)

fig3, (ax1, ax2) = plt.subplots(1,2)
countour3 = ax1.contourf(val_pred[2,:,:])
countour3_t = ax2.contourf(target_val[2,:,:])
ax1.set_title('Predicted Solution PDE and Supervision, alpha = 1')
ax2.set_title('Target Solution')
plt.tight_layout()
plt.show()