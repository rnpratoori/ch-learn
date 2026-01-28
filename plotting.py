import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import torch

def plot_nn_output_vs_c(net, device, ylabel, title):
    """
    Plots the output of a given neural network against the concentration (c).
    Returns the Plotly figure.
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
            yaxis_title=ylabel,
            template="plotly_white"
        )
        return fig
    except Exception as e:
        print(f"Could not create nn output vs c plot: {e}")
        return None


def plot_combined_final_timestep(preds_collection, epochs_collection, target_final_global):
    """
    Creates a combined plot showing the final-timestep predictions from several epochs against the ground truth.
    """
    if len(preds_collection) > 0:
        try:
            fig = go.Figure()
            x = np.arange(preds_collection[0].size)
            
            for arr, ep in zip(preds_collection, epochs_collection):
                fig.add_trace(go.Scatter(x=x, y=arr, mode='lines', name=f'Pred (ep {ep})', line=dict(width=1), opacity=0.9))

            if target_final_global is not None:
                fig.add_trace(go.Scatter(x=x, y=target_final_global, mode='lines', name='Ground truth (final time)', line=dict(color='black', width=2)))

            fig.update_layout(
                title="Final timestep: predictions (multiple epochs) vs ground truth",
                xaxis_title="DOF index",
                yaxis_title="c",
                template="plotly_white",
                legend=dict(font=dict(size=10))
            )
            return fig
        except Exception as e:
            print(f"Could not create combined final-timestep plot: {e}")
            return None
    return None


def plot_loss_vs_epochs(epochs, losses, output_path, min_loss=None):
    """
    Plots the training loss against epochs using Matplotlib and saves as PNG.
    """
    try:
        fig, ax = plt.subplots()
        ax.plot(epochs, losses, '-', label='Training Loss')
        
        if min_loss is not None:
            ax.axhline(y=min_loss, color='r', linestyle='--', label=f"Min Loss: {min_loss:.6e}")
            
        ax.set(xlabel="Epoch", ylabel="Loss (log scale)", title="Loss vs. Epochs")
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()
        
        # Ensure extension is .png
        path = str(output_path)
        if not path.endswith('.png'):
            path = path.rsplit('.', 1)[0] + '.png'
        
        fig.savefig(path)
        plt.close(fig)
        return fig
    except Exception as e:
        print(f"Could not create loss vs epochs plot: {e}")
        return None


def plot_multi_timestep_comparison_2d(epoch, comparison_data, title=None):
    """
    Creates a 2D Plotly figure with predictions and targets at multiple timesteps.
    """
    if not comparison_data:
        return None
    
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    
    for i, (timestep, pred_np, target_np) in enumerate(comparison_data):
        x = np.arange(pred_np.size)
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(x=x, y=pred_np, mode='lines', 
                                 name=f'Pred (t={timestep + 1})', 
                                 line=dict(color=color)))
        fig.add_trace(go.Scatter(x=x, y=target_np, mode='lines', 
                                 name=f'Targ (t={timestep + 1})', 
                                 line=dict(color=color, dash='dash')))

    fig.update_layout(
        title=title or f"Epoch {epoch} - 2D Multi-timestep Comparison",
        xaxis_title="DOF index",
        yaxis_title="c",
        template="plotly_white",
        legend=dict(font=dict(size=8), orientation="h")
    )
    return fig


def plot_multi_timestep_comparison_3d(epoch, comparison_data, title=None):
    """
    Creates a 3D Plotly figure with predictions and targets as surfaces.
    """
    if not comparison_data:
        return None

    x_coords = np.arange(comparison_data[0][1].size)
    t_coords = np.array([d[0] for d in comparison_data])
    
    C_pred = np.array([d[1] for d in comparison_data])
    C_targ = np.array([d[2] for d in comparison_data])

    fig = go.Figure()

    # Prediction surface
    fig.add_trace(go.Surface(x=x_coords, y=t_coords, z=C_pred, 
                             name='Prediction', colorscale='Viridis', showscale=False, opacity=0.8))
    
    # Target surface
    fig.add_trace(go.Surface(x=x_coords, y=t_coords, z=C_targ, 
                             name='Target', colorscale='Hot', showscale=False, opacity=0.6))

    fig.update_layout(
        title=title or f"Epoch {epoch} - 3D Space-Time Comparison",
        scene=dict(
            xaxis_title='DOF Index',
            yaxis_title='Timestep',
            zaxis_title='Concentration (c)'
        ),
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig