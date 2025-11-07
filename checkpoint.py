import torch
import os
import wandb

def save_checkpoint(epoch, model, optimizer, epoch_losses, epoch_numbers, output_dir, filename="ch_learn_model.pth"):
    """Saves the training state to a checkpoint file using wandb.Artifacts."""
    # Save model locally first
    checkpoint_path = output_dir / filename
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch_losses': epoch_losses,
        'epoch_numbers': epoch_numbers,
    }, checkpoint_path)
    
    # Create a wandb Artifact
    artifact = wandb.Artifact(name="ch_learn_model", type="model")
    artifact.add_file(str(checkpoint_path))
    wandb.log_artifact(artifact)
    print(f"Saved checkpoint artifact at epoch {epoch + 1}")

def load_checkpoint(model, optimizer, device, output_dir, filename="ch_learn_model.pth"):
    """Loads the training state from a checkpoint file using wandb.Artifacts."""
    start_epoch = 0
    epoch_losses = []
    epoch_numbers = []
    checkpoint_path = output_dir / filename

    if wandb.run:
        try:
            # Download the latest model artifact
            artifact = wandb.use_artifact('ch_learn_model:latest')
            artifact_path = artifact.download()
            artifact_checkpoint_path = os.path.join(artifact_path, filename)

            checkpoint = torch.load(artifact_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            epoch_losses = checkpoint.get('epoch_losses', [])
            epoch_numbers = checkpoint.get('epoch_numbers', [])
            print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}. Resuming training from epoch {start_epoch + 1}.")
        except Exception as e:
            print(f"Could not load wandb artifact: {e}. Starting training from scratch.")
    elif checkpoint_path.exists():
        # Fallback to local checkpoint if wandb not active or artifact not found
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        epoch_losses = checkpoint.get('epoch_losses', [])
        epoch_numbers = checkpoint.get('epoch_numbers', [])
        print(f"Loaded local checkpoint from epoch {checkpoint['epoch'] + 1}. Resuming training from epoch {start_epoch + 1}.")
    else:
        print("No checkpoint found, starting training from scratch.")
            
    return start_epoch, epoch_losses, epoch_numbers