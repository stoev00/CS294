import torch
import numpy as np
import argparse
from schnet import SchNet
from lmdb_dataset import LmdbDataset, data_list_collater
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import L2MAELoss, rotate_3d_coordinates
import wandb
def train(data_dir, size, r_max, batch_size, lr, max_epochs, device):

    #W&B Run
    wandb.login()
    run = wandb.init(
    # Set the project where this run will be logged
    project="dl-physics-hw2-schnet",
    name=f"aspirin_{size}_r_max={r_max}",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": max_epochs,
        "batch_size": batch_size,
        "r_max": r_max,
        "size": size
    }
    )

    # data
    train_dir = f"{data_dir}/{size}/train"
    val_dir = f"{data_dir}/{size}/val"
    test_dir = f"{data_dir}/{size}/test"
    train_dataset = LmdbDataset({'src': train_dir})
    val_dataset = LmdbDataset({'src': val_dir})
    test_dataset = LmdbDataset({'src': test_dir})
    train_dataloader = DataLoader(train_dataset, collate_fn=data_list_collater, \
                                batch_size = batch_size, shuffle = False)
    val_dataloader = DataLoader(val_dataset, collate_fn=data_list_collater, \
                                batch_size = batch_size, shuffle = False)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_list_collater, \
                                batch_size = batch_size, shuffle = False)
    # model
    model = SchNet(cutoff = r_max)
    model = model.to(device)

    #optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)    
    force_loss = L2MAELoss()

    epoch = 0
    while epoch < max_epochs and optimizer.param_groups[0]['lr'] > 1e-7:
        model.train()
        print(f"Train Epoch {epoch + 1}")
        for batch in tqdm(train_dataloader):
            #TODO: fill in the training loop
            
            print(batch)
            atomic_numbers = batch['atomic_numbers']
            positions = batch['pos']
            indices = batch['batch']
            data_forces = batch['force']
            optimizer.zero_grad()
            predictedE, predictedF = model(atomic_numbers, positions, indices)
            
            loss = force_loss(predictedF[0], data_forces)
            loss.backward()
            optimizer.step()
            wandb.log({"train_force_loss": loss, "lr": optimizer.param_groups[0]['lr']})
        loss = 0
        val_loss = 0
        model.eval()
        print(f"Val Epoch {epoch + 1}")
        for batch in val_dataloader:
            #TODO: fill in validation loop
            atomic_numbers = batch['atomic_numbers']
            positions = batch['pos']
            indices = batch['batch']
            data_forces = batch['force']
            energy_pred, forces_pred = model(atomic_numbers, positions, indices)
            loss = force_loss(forces_pred, data_forces)
            val_loss += loss
            wandb.log({"val_force_loss": loss})
        mean_val_loss = val_loss / len(val_dataloader)
        scheduler.step(mean_val_loss)
        
        epoch +=1
    
    #Final Test
    test_losses = []
    for batch in test_dataloader:
        #TODO: fill in test loop
        test_losses.append(loss.mean().item())
    wandb.log({"test_force_loss": sum(test_losses) / len(test_losses)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--size', type=str, default='1k', help='Size of dataset')
    parser.add_argument('--r_max', type=float, default=5.0, help='Cutoff (A) for radius graph construction')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default="/Users/alankstoev/Documents/VS Code/CS294/homework2/homework2/problem4-schnet 2/aspirin", help='Directory of data')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Number of epochs')


    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(args.data_dir, args.size, args.r_max, args.batch_size, args.lr, args.max_epochs, device)
