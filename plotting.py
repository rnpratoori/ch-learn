import matplotlib.pyplot as plt
import plotly.graph_objects as go

import numpy as np
import torch

def plot_nn_output_vs_c(net, device, ylabel, title):
    """
    Plots the output of a given neural network against the concentration (c).
    Returns the Plotly figure.

    :param net: The trained PyTorch model.
    :param device: The PyTorch device (e.g., 'cpu' or 'cuda').
    :param ylabel: The label for the y-axis.
    :param title: The title for the plot.
    :return: A plotly.graph_objects.Figure object.
    """
    try:
        c_values = np.linspace(0, 1, 200).reshape(-1, 1)
        c_tensor = torch.from_numpy(c_values).to(device)
        with torch.no_grad():
            output_values = net(c_tensor).cpu().numpy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c_values.flatten(), y=output_values.flatten(), mode='lines'))
        fig.update_layout(
            title=title,
            xaxis_title="Concentration (c)",
            yaxis_title=ylabel
        )
        return fig
    except Exception as e:
        print(f"Could not create nn output vs c plot: {e}")
        return None


def plot_combined_final_timestep(preds_collection, epochs_collection, target_final_global):
    """
    Creates a combined plot showing the final-timestep predictions from several epochs against the ground truth.
    Returns the Plotly figure.

    :param preds_collection: A list of numpy arrays, where each array is a prediction from a checkpoint epoch.
    :param epochs_collection: A list of epoch numbers corresponding to the predictions.
    :param target_final_global: A numpy array with the ground truth data for the final timestep.
    :return: A plotly.graph_objects.Figure object.
    """
    if len(preds_collection) > 0:
        try:
            fig = go.Figure()
            x = np.arange(preds_collection[0].size)
            
            # plot each collected prediction
            for arr, ep in zip(preds_collection, epochs_collection):
                fig.add_trace(go.Scatter(x=x, y=arr, mode='lines', name=f'Pred (ep {ep})', line=dict(width=1), opacity=0.9))

            # overlay ground truth (final time)
            if target_final_global is not None:
                fig.add_trace(go.Scatter(x=x, y=target_final_global, mode='lines', name='Ground truth (final time)', line=dict(color='black', width=2)))

            fig.update_layout(
                title="Final timestep: predictions (multiple epochs) vs ground truth",
                xaxis_title="DOF index",
                yaxis_title="c",
                legend=dict(font=dict(size=10))
            )
            return fig
        except Exception as e:
            print(f"Could not create combined final-timestep plot: {e}")
            return None
    return None



def plot_loss_vs_epochs(epochs, losses, output_path, min_loss=None):
    """
    Plots the training loss against epochs and saves it to a file.

    :param epochs: A list of epoch numbers.
    :param losses: A list of loss values.
    :param output_path: The path to save the plot image.
    :param min_loss: The minimum loss achieved, to draw a horizontal line.
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, losses, label='Training Loss')
        
        if min_loss is not None:
            ax.axhline(min_loss, color='r', linestyle='--', label=f'Min Loss: {min_loss:.6e}')
            
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