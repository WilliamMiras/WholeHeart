import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from model import EchoNetConvMamba, loss_fn
from dataset import EchoNetDataset, collate_fn
import os


def train(start_epoch=0):
    # Device setup: Leverage GPU if available (RTX 3070Ti)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EchoNetConvMamba().to(device)  # Instantiate and move to GPU

    # Checkpoint directory management
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Resume from latest epoch checkpoint if it exists
    latest_checkpoint = None
    if start_epoch == 0:  # Only auto-resume on fresh start
        for file in os.listdir(checkpoint_dir):
            if file.startswith("checkpoint_epoch_") and file.endswith(".pth"):
                epoch_num = int(file.split("_")[2].split(".")[0])
                if latest_checkpoint is None or epoch_num > latest_checkpoint[1]:
                    latest_checkpoint = (file, epoch_num)

    if latest_checkpoint:
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, latest_checkpoint[0])))
        start_epoch = latest_checkpoint[1] + 1
        print(f"Loaded checkpoint from epoch {latest_checkpoint[1]}, starting at epoch {start_epoch}")

    # Data pipeline: Load EchoNet train split with normalization
    transform = torch.nn.Sequential(
        torch.nn.functional.normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])  # ImageNet stats
    )
    train_dataset = EchoNetDataset(root_dir="./echonet_data/", split="train",
                                   transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                              num_workers=2, collate_fn=collate_fn, pin_memory=True)

    # Optimizer and mixed precision setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = GradScaler()  # For FP16 training
    accumulation_steps = 4  # Effective batch size = 2 * 4 = 8

    # Training loop: 50 epochs
    num_epochs = 50
    for epoch in range(start_epoch, num_epochs):
        model.train()  # Set model to training mode
        optimizer.zero_grad()  # Clear gradients
        running_loss = 0.0

        # Iterate over batches (~5,000 per epoch with batch_size=2)
        for i, (frames, efs, masks) in enumerate(train_loader):
            frames, efs, masks = frames.to(device), efs.to(device), masks.to(device)

            # Mixed precision forward pass
            with autocast():
                ef_pred, seg_pred = model(frames)  # [B], [B, F, 1, H, W]
                loss = loss_fn(ef_pred, efs, seg_pred, masks)  # Combined EF + seg loss
                loss = loss / accumulation_steps  # Scale for accumulation

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()

            # Step optimizer after accumulating gradients
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Logging
            running_loss += loss.item() * accumulation_steps
            if i % 10 == 9:
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

            # Mid-epoch checkpoint every 500 batches
            if (i + 1) % 500 == 0:
                mid_file = f"checkpoints/checkpoint_epoch_{epoch + 1}_batch_{i + 1}.pth"
                torch.save(model.state_dict(), mid_file)
                print(f"Saved mid-epoch checkpoint at batch {i + 1}")

        # End-of-epoch checkpoint, overwrite previous epoch
        new_file = f"checkpoints/checkpoint_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), new_file)
        print(f"Saved checkpoint for epoch {epoch + 1}")

        if epoch > start_epoch:
            old_file = f"checkpoints/checkpoint_epoch_{epoch}.pth"
            if os.path.exists(old_file):
                os.remove(old_file)
                print(f"Deleted old checkpoint: {old_file}")


if __name__ == "__main__":
    train()