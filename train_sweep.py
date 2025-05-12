import wandb
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from CNN import CIFAR10Net
from datetime import datetime
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available

def train_and_track():
    run = wandb.init()
    config = run.config

    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Load train split
    train_artifact = run.use_artifact("cifar10-train-tensor:latest", type="dataset")
    train_path = train_artifact.download()
    train_dataset = torch.load(f"{train_path}/train_data.pt", weights_only=False)

    # Load val split
    val_artifact = run.use_artifact("cifar10-val-tensor:latest", type="dataset")
    val_path = val_artifact.download()
    val_dataset = torch.load(f"{val_path}/val_data.pt", weights_only=False)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Init model
    model = CIFAR10Net().to(device)

    # Optimizer
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    # Loss function
    criterion = nn.NLLLoss()

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        train_losses = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        print(f"[{datetime.utcnow()}] Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save the final model
    model_path = "BEST_CNN.pth"
    torch.save(model.state_dict(), model_path)

    # Log to Weights & Biases
    artifact = wandb.Artifact("BEST_CNN", type="model")
    artifact.add_file(model_path)
    run.log_artifact(artifact)
    print("Model logged to W&B as artifact.")

    run.finish()

if __name__ == "__main__":
    train_and_track()
