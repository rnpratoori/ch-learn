import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from pathlib import Path
import argparse
import shutil



def plot_nn_output_animation(c_values, all_nn_outputs, ylabel, output_path, chi=None, N1=None, N2=None, epoch_list=None):
    """
    Creates an animated Plotly plot of the neural network output vs. concentration for each epoch.
    Saves the animation as an HTML file.
    """
    try:
        fig = go.Figure()

        # Calculate the analytic reference curve from the physics metadata saved
        # in the NPZ file, falling back to the historical defaults if needed.
        c = c_values.flatten()
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-10
        c_safe = np.clip(c, epsilon, 1 - epsilon)
        if chi is not None and N1 is not None and N2 is not None:
            # Use dynamic physics parameters.
            if "Energy" in str(ylabel):
                # Formula for f: c*Log[c]/N1+(1-c)*Log[1-c]/N2+chi*c*(1-c).
                # f is identifiable only up to an additive constant in some
                # learning paths, but this plot shows the absolute analytic
                # reference for orientation.
                true_values = (c_safe * np.log(c_safe))/N1 + ((1 - c_safe) * np.log(1 - c_safe))/N2 + chi * c_safe * (1 - c_safe)
                label_name = "True f(c)"
                # Align theoretical minimum with learned minimum for better visual comparison (optional, but helpful as f is defined up to constant)
                # Here we just plot the theoretical absolute value.
            else:
                # Formula for df/dc: chi - 2* c *chi + 1/N1 - 1/N2 - Log[1 - c]/N2 + Log[c]/N1
                true_values = chi - 2 * chi * c_safe + 1/N1 - 1/N2 - (np.log(1 - c_safe))/N2 + (np.log(c_safe))/N1
                label_name = "True df/dc"
                print("Using true df/dc formula")
            print(f"Using dynamic {label_name} with chi={chi}, N1={N1}, N2={N2}")
        else:
             # Fallback to old hardcoded formula
             true_values = 1 - 2 * c_safe + (np.log(c_safe) - np.log(1 - c_safe)) / 5
             label_name = "True df/dc (Fallback)"
             print("Using fallback true df/dc formula")

        # Add trace for true values
        fig.add_trace(
            go.Scatter(
                x=c,
                y=true_values,
                name=label_name,
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
    Creates interactive 3D Plotly plots of the simulation data (pred, targ, error).
    Saves the plots as separate HTML files for each view.
    """
    if len(all_epochs_data) == 0:
        print("No simulation data to write to Plotly file.")
        return

    output_dir = output_path.parent
    base_name = output_path.stem.replace('_3d', '')

    plot_configs = [
        {'label': 'Prediction', 'visible_pattern': [True, False, False], 'filename': f'{base_name}_prediction_3d.html'},
        {'label': 'Target', 'visible_pattern': [False, True, False], 'filename': f'{base_name}_target_3d.html'},
        {'label': 'Prediction and Target', 'visible_pattern': [True, True, False], 'filename': f'{base_name}_prediction_and_target_3d.html'},
        {'label': 'Error', 'visible_pattern': [False, False, True], 'filename': f'{base_name}_error_3d.html'}
    ]

    for config in plot_configs:
        try:
            fig = go.Figure()

            # Determine dimensions from the first epoch
            first_epoch_data = all_epochs_data[0]['data']
            if not first_epoch_data:
                print("Epoch data is empty, cannot determine dimensions.")
                continue

            num_sim_timesteps = len(first_epoch_data)
            num_dofs = first_epoch_data[0][1].size
            x_coords = np.arange(num_dofs)

            # Add traces for each epoch
            for epoch_data in all_epochs_data:
                if len(epoch_data['data']) != num_sim_timesteps:
                    continue

                t_coords = np.array([d[0] for d in epoch_data['data']])
                C_pred = np.array([d[1] for d in epoch_data['data']])
                C_targ = np.array([d[2] for d in epoch_data['data']])
                C_err = np.abs(C_pred - C_targ)
                
                X, T = np.meshgrid(x_coords, t_coords)

                # Add surfaces for prediction, target, and error for each epoch
                fig.add_trace(go.Surface(z=C_pred, x=X, y=T, name='Prediction', colorscale=[[0, "red"], [1, "red"]], showscale=False, visible=False))
                fig.add_trace(go.Surface(z=C_targ, x=X, y=T, name='Target', colorscale=[[0, "blue"], [1, "blue"]], showscale=False, visible=False))
                fig.add_trace(go.Surface(z=C_err, x=X, y=T, name='Error', colorscale='Bluered', cmin=0, cmax=1, colorbar=dict(title='Error'), showscale=True, visible=False))

            # Create and add slider for epochs
            steps = []
            traces_per_epoch = 3
            
            for i, epoch_data in enumerate(all_epochs_data):
                epoch_num = epoch_data['epoch']
                visibility = [False] * len(fig.data)
                
                # Apply visibility pattern for the current epoch
                for trace_idx, is_visible in enumerate(config['visible_pattern']):
                    if is_visible:
                        visibility[i * traces_per_epoch + trace_idx] = True

                step = dict(
                    method="update",
                    args=[{"visible": visibility},
                          {"title.text": f"Cahn-Hilliard Simulation: {config['label']} (Epoch {epoch_num})"}],
                    label=str(epoch_num)
                )
                steps.append(step)

            sliders = [dict(
                active=0,
                currentvalue={"prefix": "Epoch: "},
                pad={"t": 50},
                steps=steps,
            )]

            # Set initial visibility for the first epoch
            initial_visibility = [False] * len(fig.data)
            traces_to_make_visible = [idx for idx, is_vis in enumerate(config['visible_pattern']) if is_vis]
            for trace_idx in traces_to_make_visible:
                if trace_idx < len(fig.data):
                    initial_visibility[trace_idx] = True
            
            for i in range(len(fig.data)):
                fig.data[i].visible = initial_visibility[i]
            
            z_axis_title = 'Concentration (c)'
            if config['label'] == 'Error':
                z_axis_title = 'Absolute Error |c_pred - c_targ|'

            fig.update_layout(
                sliders=sliders,
                title_text=f"Cahn-Hilliard Simulation: {config['label']} (Epoch {all_epochs_data[0]['epoch']})",
                scene=dict(
                    xaxis_title='DOF index',
                    yaxis_title='Timestep',
                    zaxis_title=z_axis_title
                ),
            )

            current_output_path = output_dir / config['filename']
            pio.write_html(fig, current_output_path)
            print(f"Saved simulation data 3D plot to {current_output_path}")

        except Exception as e:
            print(f"Could not create simulation data 3D plot for {config['label']}: {e}")


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
    ax.plot_surface(X, T, C_pred, color='red', alpha=0.7, rstride=1, cstride=5, label='Prediction')
    ax.plot_surface(X, T, C_targ, color='blue', alpha=0.7, rstride=1, cstride=5, label='Target')

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
        surf = ax.plot_surface(X, T, C_diff, color='green', rstride=1, cstride=5)

        ax.set_xlabel("DOF index")
        ax.set_ylabel("Timestep")
        ax.set_zlabel("Error (Prediction - Target)")
        ax.set_title(f"Epoch {epoch} - 3D Space-Time Error (from saved data)")


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

    # Recreate the plot directory to avoid mixing fresh figures with outputs
    # generated from an older NPZ schema or a different training run.
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
        
        plot_nn_output_animation(c_vals, all_nn_outputs, ylabel, output_path,
                                 chi=float(data.get('chi')),
                                 N1=float(data.get('N1')),
                                 N2=float(data.get('N2')))

    elif 'c_values_nn' in data and 'nn_output_values' in data and 'nn_output_label' in data: # Backwards compatibility
        print("Found old format 'nn_output_values'. Plotting for final model state only.")
        ylabel = str(data['nn_output_label'])
        c_vals = data['c_values_nn']
        # Create a structure that the new animation function can understand
        all_nn_outputs = [{'epoch': data['epochs_collection'][-1], 'output': data['nn_output_values']}]
        output_path = plot_output_dir / "nn_output_vs_c.html"
        plot_nn_output_animation(c_vals, all_nn_outputs, ylabel, output_path,
                                 chi=float(data.get('chi', 1.0)) if 'chi' in data else None,
                                 N1=float(data.get('N1', 5.0)) if 'N1' in data else None,
                                 N2=float(data.get('N2', 5.0)) if 'N2' in data else None)

    else:
        print("Skipping nn output animation: Data not found in .npz file.")

    # --- 4. Write multi-timestep comparisons to Plotly 3D plot ---
    print("\nGenerating multi-timestep comparison 3D plot...")
    if 'all_epochs_comparison_data' in data:
        all_epochs_data = data['all_epochs_comparison_data']

        if len(all_epochs_data) > 1000:
            print(f"Found {len(all_epochs_data)} epochs, downsampling to 1000 for the plot.")
            # Plotly becomes unwieldy with thousands of surfaces; uniform
            # downsampling keeps the beginning and end of training visible.
            indices = np.linspace(0, len(all_epochs_data) - 1, 1000, dtype=int)
            all_epochs_data = [all_epochs_data[i] for i in indices]

        output_path = plot_output_dir / "simulation_data_3d.html"
        plot_simulation_data_3d(all_epochs_data, output_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce plots from ch_learn.py's .npz output.")
    parser.add_argument("npz_file", type=str, help="Path to the post_processing_data.npz file.")
    args = parser.parse_args()

    reproduce_plots(args.npz_file)
