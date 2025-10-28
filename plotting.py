import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import subprocess
import torch

def plot_dfdc_vs_c(dfdc_net, device, filename="dfdc_vs_c.png"):
    """
    Plots the learned free energy derivative (dfdc) against the concentration (c).

    :param dfdc_net: The trained PyTorch model for df/dc.
    :param device: The PyTorch device (e.g., 'cpu' or 'cuda').
    :param filename: The name of the file to save the plot to.
    """
    try:
        c_values = np.linspace(0, 1, 200).reshape(-1, 1)
        c_tensor = torch.from_numpy(c_values).to(device)
        with torch.no_grad():
            dfdc_values = dfdc_net(c_tensor).cpu().numpy()

        plt.figure(figsize=(6, 4))
        plt.plot(c_values, dfdc_values)
        plt.xlabel("Concentration (c)")
        plt.ylabel("Free Energy Derivative (df/dc)")
        plt.title("Learned Free Energy Derivative")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()
        print(f"Saved df/dc vs c plot to {filename}")
    except Exception as e:
        print(f"Could not save df/dc vs c plot: {e}")


def plot_combined_final_timestep(preds_collection, epochs_collection, target_final_global, filename="c_final_epochs_combined.png"):
    """
    Creates a combined plot showing the final-timestep predictions from several epochs against the ground truth.

    :param preds_collection: A list of numpy arrays, where each array is a prediction from a checkpoint epoch.
    :param epochs_collection: A list of epoch numbers corresponding to the predictions.
    :param target_final_global: A numpy array with the ground truth data for the final timestep.
    :param filename: The name of the file to save the plot to.
    """
    if len(preds_collection) > 0:
        try:
            x = np.arange(preds_collection[0].size)
            plt.figure(figsize=(8,5))
            # plot each collected prediction
            for arr, ep in zip(preds_collection, epochs_collection):
                plt.plot(x, arr, label=f'Pred (ep {ep})', lw=1, alpha=0.9)
            # overlay ground truth (final time)
            if target_final_global is not None:
                plt.plot(x, target_final_global, label='Ground truth (final time)', color='k', lw=2)
            plt.xlabel("DOF index")
            plt.ylabel("c")
            plt.title("Final timestep: predictions (multiple epochs) vs ground truth")
            plt.legend(ncol=2, fontsize='small')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(filename, dpi=200)
            plt.close()
            print(f"Saved combined final-timestep plot to {filename}")
        except Exception as e:
            print(f"Could not save combined final-timestep plot: {e}")

def create_video(frames_list, frames_dir, video_fname="c_final_comparison.mp4", fps=10):
    """
    Assembles frames into a video.

    :param frames_list: A list of paths to the image frames.
    :param frames_dir: The directory where the frames are stored.
    :param video_fname: The name of the output video file.
    :param fps: The frames per second for the video.
    """
    if len(frames_list) > 0:
        video_fname = "c_final_comparison.mp4"
        try:
            with imageio.get_writer(video_fname, fps=fps) as writer:
                for f in frames_list:
                    img = imageio.imread(f)
                    writer.append_data(img)
            print(f"Saved video to {video_fname} using imageio")
        except Exception as e_img:
            print(f"imageio failed ({e_img}), trying ffmpeg fallback...")
            try:
                cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i",
                       os.path.join(frames_dir, "frame_epoch_%04d.png"),
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", video_fname]
                subprocess.run(cmd, check=True)
                print(f"Saved video to {video_fname} using ffmpeg")
            except Exception as e_ff:
                print(f"ffmpeg fallback failed: {e_ff}")

def save_comparison_image(epoch, pred_global, target_global, filename_prefix="c_final_epoch_"):
    """
    Saves a PNG image comparing the prediction and ground truth for a given epoch.
    """
    try:
        x = np.arange(pred_global.size)
        plt.figure(figsize=(8,5))
        plt.plot(x, pred_global, label=f'Pred (ep {epoch})', lw=1, alpha=0.9)
        plt.plot(x, target_global, label='Ground truth (final time)', color='k', lw=2)
        plt.xlabel("DOF index")
        plt.ylabel("c")
        plt.title(f"Final-timestep comparison at epoch {epoch}")
        plt.legend(ncol=2, fontsize='small')
        plt.grid(True)
        plt.tight_layout()
        fname_img = f"{filename_prefix}{epoch}.png"
        plt.savefig(fname_img, dpi=200)
        plt.close()
        print(f"Saved comparison image to {fname_img}")
    except Exception as e:
        print(f"Could not save comparison image for epoch {epoch}: {e}")

def save_video_frame(epoch, pred_global, target_global, loss_epoch, frames_dir, frames_list, filename=None):
    """
    Saves a PNG image frame for the video.
    """
    try:
        x = np.arange(pred_global.size)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, pred_global, label=f'Pred (ep {epoch})', lw=1)
        ax.plot(x, target_global, label='Ground truth (final time)', color='k', lw=1.5, alpha=0.9)
        ax.set_xlabel("DOF index")
        ax.set_ylabel("c")
        ax.set_title("Final timestep: prediction vs ground truth")
        ax.legend(loc="upper left", fontsize='small')
        # epoch text (top-right)
        ax.text(0.98, 0.98, f"Epoch {epoch}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        # loss text (bottom-right)
        ax.text(0.98, 0.02, f"Loss = {loss_epoch:.3e}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        ax.grid(True)
        plt.tight_layout()
        if filename:
            frame_fname = filename
        else:
            frame_fname = os.path.join(frames_dir, f"frame_epoch_{epoch:04d}.png")
            frames_list.append(frame_fname)
        fig.savefig(frame_fname, dpi=150)
        plt.close(fig)
        print(f"Saved frame for epoch {epoch} -> {frame_fname}")
    except Exception as e:
        print(f"Could not create frame for epoch {epoch}: {e}")

def plot_loss_over_epochs(losses, filename="loss_vs_epoch.png"):
    """
    Plots the training loss versus the epoch number.

    :param losses: A list of loss values, one for each epoch.
    :param filename: The name of the file to save the plot to.
    """
    try:
        plt.figure(figsize=(6,4))
        plt.plot(range(1, len(losses)+1), losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (J)")
        plt.yscale('log')
        plt.title("Training loss vs epoch")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()
        print(f"Saved loss plot to {filename}")
    except Exception as e:
        print(f"Could not save loss plot: {e}")
