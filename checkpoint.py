import torch
import os

def save_checkpoint(epoch, model, optimizer, losses, rank, filename="checkpoint.pth"):
    """Saves the training state to a checkpoint file."""
    if rank == 0:  # Only save on the root process
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
        }
        torch.save(state, filename)
        print(f"Saved checkpoint at epoch {epoch + 1} to {filename}")

def load_checkpoint(model, optimizer, device, rank, filename="checkpoint.pth"):
    """Loads the training state from a checkpoint file."""
    start_epoch = 0
    losses = []
    if os.path.exists(filename):
        # All ranks load the checkpoint to ensure model and optimizer are synchronized.
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        losses = checkpoint['losses']
        if rank == 0:
            print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}. Resuming training from epoch {start_epoch + 1}.")
    else:
        if rank == 0:
            print("No checkpoint found, starting training from scratch.")
    return start_epoch, losses
