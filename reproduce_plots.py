import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from pathlib import Path
import argparse
import shutil



def plot_nn_output_animation(c_values, all_nn_outputs, ylabel, output_path):
    """
    Creates an animated Plotly plot of the neural network output vs. concentration for each epoch.
    Saves the animation as an HTML file.
    """
    try:
        fig = go.Figure()

        # Calculate true df/dc
        c = c_values.flatten()
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-10
        c_safe = np.clip(c, epsilon, 1 - epsilon)
        true_dfdc = 1 - 2 * c_safe + (np.log(c_safe) - np.log(1 - c_safe)) / 5

        # Add trace for true df/dc
        fig.add_trace(
            go.Scatter(
                x=c,
                y=true_dfdc,
                name="True df/dc",
                mode='lines',
                line=dict(color='black', dash='dash')
            )
        )

        # Add traces for each epoch
        for nn_output_data in all_nn_outputs:
            epoch = nn_output_data['epoch']
            nn_output_values = nn_output_data['output']
            fig.add_trace(
                go.Scatter(
                    x=c_values.flatten(),
                    y=nn_output_values.flatten(),
                    name=f"Epoch {epoch}",
                    visible=False,
                    mode='lines'
                )
            )

        # Make the first trace visible
        if len(fig.data) > 1:
            fig.data[1].visible = True

        # Create and add slider
        steps = []
        for i, nn_output_data in enumerate(all_nn_outputs):
            epoch = nn_output_data['epoch']
            # Set visibility for all traces
            visibility = [True] + [False] * len(all_nn_outputs)
            visibility[i+1] = True
            
            step = dict(
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Learned {ylabel} vs. Concentration (Epoch {epoch})"}],
                label=str(epoch)
            )
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Epoch: "},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders,
            title=f"Learned {ylabel} vs. Concentration (Epoch {all_nn_outputs[0]['epoch']})",
            xaxis_title="Concentration (c)",
            yaxis_title=ylabel,
        )

        pio.write_html(fig, output_path)
        print(f"Saved nn output animation to {output_path}")

    except Exception as e:
        print(f"Could not create nn output animation: {e}")




# def plot_multi_timestep_comparison_2d_from_data(epoch, comparison_data, output_path):
#     """
#     Creates a 2D figure with predictions and targets at multiple timesteps on the same plot from saved data.
#     """
#     num_plots = len(comparison_data)
#     if num_plots == 0:
#         return
# 
#     fig, ax = plt.subplots(figsize=(10, 6))
#     
#     colors = plt.cm.viridis(np.linspace(0, 1, num_plots))
# 
#     for i, (timestep, pred_np, target_np) in enumerate(comparison_data):
#         x = np.arange(pred_np.size)
#         ax.plot(x, pred_np, label=f'Pred (t={timestep + 1})', color=colors[i])
#         ax.plot(x, target_np, label=f'Targ (t={timestep + 1})', color=colors[i], linestyle='--')
# 
#     ax.set_xlabel("DOF index")
#     ax.set_ylabel("c")
#     ax.set_title(f"Epoch {epoch} - 2D Multi-timestep Comparison (from saved data)")
#     ax.legend(ncol=2, fontsize='small')
#     ax.grid(True)
#     plt.tight_layout()
#     fig.savefig(output_path)
#     plt.close(fig)
#     print(f"Saved 2D multi-timestep comparison plot to {output_path}")


def plot_simulation_data_3d(all_epochs_data, output_path):
    """
    Creates an interactive 3D Plotly plot of the simulation data (pred, targ, error)
    with a slider to select the epoch and a dropdown to select the surface.
    Saves the plot as an HTML file.
    """
    if len(all_epochs_data) == 0:
        print("No simulation data to write to Plotly file.")
        return

    try:
        fig = go.Figure()

        # Determine dimensions from the first epoch
        first_epoch_data = all_epochs_data[0]['data']
        if not first_epoch_data:
            print("Epoch data is empty, cannot determine dimensions.")
            return

        num_sim_timesteps = len(first_epoch_data)
        num_dofs = first_epoch_data[0][1].size

        x_coords = np.arange(num_dofs)

        # Add traces for each epoch
        for epoch_data in all_epochs_data:
            epoch_num = epoch_data['epoch']
            
            if len(epoch_data['data']) != num_sim_timesteps:
                continue

            t_coords = np.array([d[0] for d in epoch_data['data']])
            C_pred = np.array([d[1] for d in epoch_data['data']])
            C_targ = np.array([d[2] for d in epoch_data['data']])
            C_err = C_pred - C_targ
            
            X, T = np.meshgrid(x_coords, t_coords)

            # Add surfaces for prediction, target, and error for each epoch
            fig.add_trace(go.Surface(z=C_pred, x=X, y=T, name=f'Prediction', colorscale=[[0, 'blue'], [1, 'blue']], showscale=False, visible=False))
            fig.add_trace(go.Surface(z=C_targ, x=X, y=T, name=f'Target', colorscale=[[0, 'green'], [1, 'green']], showscale=False, visible=False))
            fig.add_trace(go.Surface(z=C_err, x=X, y=T, name=f'Error', colorscale=[[0, 'red'], [1, 'red']], showscale=False, visible=False))

        # Make the first epoch's prediction trace visible
        if len(fig.data) > 0:
            fig.data[0].visible = True

        # Create and add slider
        steps = []
        for i in range(len(all_epochs_data)):
            epoch_num = all_epochs_data[i]['epoch']
            
            visibility = [False] * len(fig.data)
            visibility[i*3] = True
            
            step = dict(
                method="restyle",
                args=[{"visible": visibility}],
                label=str(epoch_num)
            )
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Epoch: "},
            pad={"t": 50},
            steps=steps,
        )]

        updatemenus = [
            dict(
                active=0,
                buttons=list([
                    dict(label="Prediction",
                         method="restyle",
                         args=[{"visible": [True, False, False] * len(all_epochs_data)}]),
                    dict(label="Target",
                         method="restyle",
                         args=[{"visible": [False, True, False] * len(all_epochs_data)}]),
                    dict(label="Prediction and Target",
                         method="restyle",
                         args=[{"visible": [True, True, False] * len(all_epochs_data)}]),
                    dict(label="Error",
                         method="restyle",
                         args=[{"visible": [False, False, True] * len(all_epochs_data)}]),
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
        
        initial_visibility = [True, False, False] * len(all_epochs_data)
        for i in range(len(fig.data)):
            fig.data[i].visible = initial_visibility[i]
        
        fig.update_layout(
            sliders=sliders,
            updatemenus=updatemenus,
            title_text=f"Cahn-Hilliard Simulation Data (Epoch {all_epochs_data[0]['epoch']})",
            scene=dict(
                xaxis_title='DOF index',
                yaxis_title='Timestep',
                zaxis_title='Concentration (c)'),
        )

        pio.write_html(fig, output_path)
        print(f"Saved simulation data 3D plot to {output_path}")

    except Exception as e:
        print(f"Could not create simulation data 3D plot: {e}")


def plot_multi_timestep_comparison_3d_from_data(epoch, comparison_data, output_path):
    """
    Creates a 3D figure with predictions and targets at multiple timesteps from saved data.
    """
    if len(comparison_data) == 0:
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare data for surface plots
    x_coords = np.arange(comparison_data[0][1].size)
    t_coords = np.array([d[0] for d in comparison_data])
    
    X, T = np.meshgrid(x_coords, t_coords)
    
    C_pred = np.array([d[1] for d in comparison_data])
    C_targ = np.array([d[2] for d in comparison_data])

    # Plot the prediction and target surfaces
    ax.plot_surface(X, T, C_pred, cmap='viridis', alpha=0.7, rstride=1, cstride=5, label='Prediction')
    ax.plot_surface(X, T, C_targ, cmap='autumn', alpha=0.7, rstride=1, cstride=5, label='Target')

    ax.set_xlabel("DOF index")
    ax.set_ylabel("Timestep")
    ax.set_zlabel("Concentration (c)")
    ax.set_title(f"Epoch {epoch} - 3D Space-Time Comparison (from saved data)")
    
    # Create proxy artists for legend
    pred_proxy = plt.Rectangle((0, 0), 1, 1, fc="blue")
    targ_proxy = plt.Rectangle((0, 0), 1, 1, fc="red")
    ax.legend([pred_proxy, targ_proxy], ['Prediction', 'Target'])

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved 3D multi-timestep comparison plot to {output_path}")


def plot_spacetime_error_3d(epoch, comparison_data, output_path):
    """
    Creates a 3D surface plot of the difference (error) between predictions 
    and targets at multiple timesteps from saved data.
    """
    if len(comparison_data) == 0:
        return
    
    try:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Prepare data for surface plots
        x_coords = np.arange(comparison_data[0][1].size)
        t_coords = np.array([d[0] for d in comparison_data])
        
        X, T = np.meshgrid(x_coords, t_coords)
        
        C_pred = np.array([d[1] for d in comparison_data])
        C_targ = np.array([d[2] for d in comparison_data])
        
        C_diff = C_pred - C_targ

        # Plot the error surface
        surf = ax.plot_surface(X, T, C_diff, cmap='coolwarm', rstride=1, cstride=5)

        ax.set_xlabel("DOF index")
        ax.set_ylabel("Timestep")
        ax.set_zlabel("Error (Prediction - Target)")
        ax.set_title(f"Epoch {epoch} - 3D Space-Time Error (from saved data)")
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved 3D space-time error plot to {output_path}")
    except Exception as e:
        print(f"Could not create 3D space-time error plot: {e}")


def reproduce_plots(npz_path):
    """
    Loads data from an .npz file and reproduces the plots generated by ch_learn.py.
    """
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        print(f"Error: File not found at {npz_path}")
        return

    # Create output directory for plots, removing old plots
    plot_output_dir = npz_path.parent / "reproduced_plots"
    if plot_output_dir.exists():
        shutil.rmtree(plot_output_dir)
    plot_output_dir.mkdir()
    print(f"Saving plots to {plot_output_dir}")

    data = np.load(npz_path, allow_pickle=True)

    # --- Find min loss epoch ---
    min_loss_idx = -1
    min_loss_epoch = -1
    if 'epoch_losses' in data and len(data['epoch_losses']) > 0:
        min_loss_idx = np.argmin(data['epoch_losses'])
        if 'epoch_numbers' in data:
            min_loss_epoch = data['epoch_numbers'][min_loss_idx]
            print(f"\nMinimum loss of {data['epoch_losses'][min_loss_idx]} found at epoch {min_loss_epoch}.")
        else:
            # Fallback if epoch_numbers is not there for some reason
            min_loss_epoch = min_loss_idx + 1
            print(f"\nMinimum loss of {data['epoch_losses'][min_loss_idx]} found at index {min_loss_idx} (epoch number may vary).")
    else:
        print("\nLoss data not found, cannot determine minimum loss epoch.")

    # --- 2. Write nn output vs c for all saved epochs to Plotly animation ---
    print("\nGenerating nn output vs c animation for all saved epochs...")
    if 'c_values_nn' in data and 'all_nn_outputs' in data and 'nn_output_label' in data:
        ylabel = str(data['nn_output_label'])
        c_vals = data['c_values_nn']
        all_nn_outputs = data['all_nn_outputs']
        output_path = plot_output_dir / "nn_output_vs_c.html"
        
        plot_nn_output_animation(c_vals, all_nn_outputs, ylabel, output_path)

    elif 'c_values_nn' in data and 'nn_output_values' in data and 'nn_output_label' in data: # Backwards compatibility
        print("Found old format 'nn_output_values'. Plotting for final model state only.")
        ylabel = str(data['nn_output_label'])
        c_vals = data['c_values_nn']
        # Create a structure that the new animation function can understand
        all_nn_outputs = [{'epoch': data['epochs_collection'][-1], 'output': data['nn_output_values']}]
        output_path = plot_output_dir / "nn_output_vs_c.html"
        plot_nn_output_animation(c_vals, all_nn_outputs, ylabel, output_path)

    else:
        print("Skipping nn output animation: Data not found in .npz file.")

    # --- 4. Write multi-timestep comparisons to Plotly 3D plot ---
    print("\nGenerating multi-timestep comparison 3D plot...")
    if 'all_epochs_comparison_data' in data:
        all_epochs_data = data['all_epochs_comparison_data']

        if len(all_epochs_data) > 1000:
            print(f"Found {len(all_epochs_data)} epochs, downsampling to 1000 for the plot.")
            indices = np.linspace(0, len(all_epochs_data) - 1, 1000, dtype=int)
            all_epochs_data = [all_epochs_data[i] for i in indices]

        output_path = plot_output_dir / "simulation_data_3d.html"
        plot_simulation_data_3d(all_epochs_data, output_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce plots from ch_learn.py's .npz output.")
    parser.add_argument("npz_file", type=str, help="Path to the post_processing_data.npz file.")
    args = parser.parse_args()

    reproduce_plots(args.npz_file)
