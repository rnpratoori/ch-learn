import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import subprocess
import torch

def plot_dfdc_vs_c(dfdc_net, device):
    """
    Plots the learned free energy derivative (dfdc) against the concentration (c).
    Returns the matplotlib figure.

    :param dfdc_net: The trained PyTorch model for df/dc.
    :param device: The PyTorch device (e.g., 'cpu' or 'cuda').
    :return: A matplotlib.figure.Figure object.
    """
    try:
        c_values = np.linspace(0, 1, 200).reshape(-1, 1)
        c_tensor = torch.from_numpy(c_values).to(device)
        with torch.no_grad():
            dfdc_values = dfdc_net(c_tensor).cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(c_values, dfdc_values)
        ax.set_xlabel("Concentration (c)")
        ax.set_ylabel("Free Energy Derivative (df/dc)")
        ax.set_title("Learned Free Energy Derivative")
        ax.grid(True)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Could not create df/dc vs c plot: {e}")
        return None


def plot_f_vs_c(f_net, device):
    """
    Plots the learned free energy (f) against the concentration (c).
    Returns the matplotlib figure.

    :param f_net: The trained PyTorch model for f.
    :param device: The PyTorch device (e.g., 'cpu' or 'cuda').
    :return: A matplotlib.figure.Figure object.
    """
    try:
        c_values = np.linspace(0, 1, 200).reshape(-1, 1)
        c_tensor = torch.from_numpy(c_values).to(device)
        with torch.no_grad():
            f_values = f_net(c_tensor).cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(c_values, f_values)
        ax.set_xlabel("Concentration (c)")
        ax.set_ylabel("Free Energy (f)")
        ax.set_title("Learned Free Energy")
        ax.grid(True)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Could not create f vs c plot: {e}")
        return None


def plot_combined_final_timestep(preds_collection, epochs_collection, target_final_global):
    """
    Creates a combined plot showing the final-timestep predictions from several epochs against the ground truth.
    Returns the matplotlib figure.

    :param preds_collection: A list of numpy arrays, where each array is a prediction from a checkpoint epoch.
    :param epochs_collection: A list of epoch numbers corresponding to the predictions.
    :param target_final_global: A numpy array with the ground truth data for the final timestep.
    :return: A matplotlib.figure.Figure object.
    """
    if len(preds_collection) > 0:
        try:
            x = np.arange(preds_collection[0].size)
            fig, ax = plt.subplots(figsize=(8,5))
            # plot each collected prediction
            for arr, ep in zip(preds_collection, epochs_collection):
                ax.plot(x, arr, label=f'Pred (ep {ep})', lw=1, alpha=0.9)
            # overlay ground truth (final time)
            if target_final_global is not None:
                ax.plot(x, target_final_global, label='Ground truth (final time)', color='k', lw=2)
            ax.set_xlabel("DOF index")
            ax.set_ylabel("c")
            ax.set_title("Final timestep: predictions (multiple epochs) vs ground truth")
            ax.legend(ncol=2, fontsize='small')
            ax.grid(True)
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Could not create combined final-timestep plot: {e}")
            return None
    return None



def plot_multi_timestep_comparison(epoch, comparison_data):
    """
    Creates a figure with predictions and targets at multiple timesteps on the same plot.
    """
    num_plots = len(comparison_data)
    if num_plots == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, num_plots))

    for i, (timestep, pred, target) in enumerate(comparison_data):
        pred_np = pred.dat.data_ro
        target_np = target.dat.data_ro
        x = np.arange(pred_np.size)

        ax.plot(x, pred_np, label=f'Pred (t={timestep + 1})', color=colors[i])
        ax.plot(x, target_np, label=f'Targ (t={timestep + 1})', color=colors[i], linestyle='--')

    ax.set_xlabel("DOF index")
    ax.set_ylabel("c")
    ax.set_title(f"Epoch {epoch} - Multi-timestep Comparison")
    ax.legend(ncol=2, fontsize='small')
    ax.grid(True)
    plt.tight_layout()
    return fig