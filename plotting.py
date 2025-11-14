import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_nn_output_vs_c(net, device, ylabel, title):
    """
    Plots the output of a given neural network against the concentration (c).
    Returns the matplotlib figure.

    :param net: The trained PyTorch model.
    :param device: The PyTorch device (e.g., 'cpu' or 'cuda').
    :param ylabel: The label for the y-axis.
    :param title: The title for the plot.
    :return: A matplotlib.figure.Figure object.
    """
    try:
        c_values = np.linspace(0, 1, 200).reshape(-1, 1)
        c_tensor = torch.from_numpy(c_values).to(device)
        with torch.no_grad():
            output_values = net(c_tensor).cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(c_values, output_values)
        ax.set_xlabel("Concentration (c)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Could not create nn output vs c plot: {e}")
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



from mpl_toolkits.mplot3d import Axes3D

def plot_multi_timestep_comparison_2d(epoch, comparison_data):
    """
    Creates a 2D figure with predictions and targets at multiple timesteps on the same plot.
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
    ax.set_title(f"Epoch {epoch} - 2D Multi-timestep Comparison")
    ax.legend(ncol=2, fontsize='small')
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_multi_timestep_comparison_3d(epoch, comparison_data):
    """
    Creates a 3D figure with predictions and targets at multiple timesteps.
    X-axis: DOF index, Y-axis: Time, Z-axis: Concentration (c)
    """
    if len(comparison_data) == 0:
        return None

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare data for surface plots
    x_coords = np.arange(comparison_data[0][1].dat.data_ro.size)
    t_coords = np.array([d[0] for d in comparison_data])
    
    X, T = np.meshgrid(x_coords, t_coords)
    
    C_pred = np.array([d[1].dat.data_ro for d in comparison_data])
    C_targ = np.array([d[2].dat.data_ro for d in comparison_data])

    # Plot the prediction and target surfaces
    ax.plot_surface(X, T, C_pred, cmap='viridis', alpha=0.7, rstride=1, cstride=5, label='Prediction')
    ax.plot_surface(X, T, C_targ, cmap='autumn', alpha=0.7, rstride=1, cstride=5, label='Target')

    ax.set_xlabel("DOF index")
    ax.set_ylabel("Timestep")
    ax.set_zlabel("Concentration (c)")
    ax.set_title(f"Epoch {epoch} - 3D Space-Time Comparison")
    
    # Create proxy artists for legend
    pred_proxy = plt.Rectangle((0, 0), 1, 1, fc="blue")
    targ_proxy = plt.Rectangle((0, 0), 1, 1, fc="red")
    ax.legend([pred_proxy, targ_proxy], ['Prediction', 'Target'])

    plt.tight_layout()
    return fig


def plot_loss_vs_epochs(epochs, losses, output_path):
    """
    Plots the training loss against epochs and saves it to a file.

    :param epochs: A list of epoch numbers.
    :param losses: A list of loss values.
    :param output_path: The path to save the plot image.
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, losses, label='Training Loss')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss vs. Epochs")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
    except Exception as e:
        print(f"Could not create loss vs epochs plot: {e}")