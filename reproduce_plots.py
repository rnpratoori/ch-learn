import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import argparse
import shutil

# Assuming plotting.py is in the same directory or accessible
from plotting import plot_combined_final_timestep, plot_loss_vs_epochs

def write_nn_output_tecplot(c_values, all_nn_outputs, ylabel, output_path):
    """
    Writes the neural network output vs. concentration to a Tecplot ASCII file.
    Each epoch is written as a separate zone.
    """
    try:
        with open(output_path, 'w') as f:
            f.write(f'TITLE = "Learned {ylabel} vs. Concentration"\n')
            f.write(f'VARIABLES = "c", "{ylabel}"\n')

            for nn_output_data in all_nn_outputs:
                epoch = nn_output_data['epoch']
                nn_output_values = nn_output_data['output']
                
                num_points = len(c_values)
                f.write(f'\nZONE T="Epoch {epoch}", I={num_points}, DATAPACKING=POINT\n')
                
                for i in range(num_points):
                    f.write(f"{c_values[i][0]:.8f} {nn_output_values[i][0]:.8f}\n")
        
        print(f"Saved nn output Tecplot file to {output_path}")
    except Exception as e:
        print(f"Could not create nn output Tecplot file: {e}")




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


def write_simulation_data_tecplot(all_epochs_data, output_path):
    """
    Writes the full simulation data (pred, targ, error) to a single
    Tecplot ASCII file with a 3D structured grid (I=dof, J=sim_timestep, K=epoch).
    """
    if len(all_epochs_data) == 0:
        print("No simulation data to write to Tecplot file.")
        return

    try:
        # Determine dimensions
        num_epochs = len(all_epochs_data)
        
        # Get data from the first epoch and first timestep to determine other dims
        first_epoch_data = all_epochs_data[0]['data']
        if not first_epoch_data:
            print("Epoch data is empty, cannot determine dimensions.")
            return
            
        num_sim_timesteps = len(first_epoch_data)
        num_dofs = first_epoch_data[0][1].size # pred_np size

        print(f"Writing simulation data to Tecplot file with dimensions: I={num_dofs}, J={num_sim_timesteps}, K={num_epochs}")

        with open(output_path, 'w') as f:
            f.write('TITLE = "Cahn-Hilliard Simulation Data"\n')
            f.write('VARIABLES = "x", "t_sim", "epoch", "prediction", "target", "error"\n')
            f.write(f'ZONE T="Simulation Data", I={num_dofs}, J={num_sim_timesteps}, K={num_epochs}, DATAPACKING=POINT\n')

            # Loop in K, J, I order (epoch, sim_timestep, dof)
            for k, epoch_data in enumerate(all_epochs_data):
                epoch_num = epoch_data['epoch']
                
                # Check if this epoch's data has the expected number of timesteps
                if len(epoch_data['data']) != num_sim_timesteps:
                    print(f"Warning: Epoch {epoch_num} has {len(epoch_data['data'])} timesteps, expected {num_sim_timesteps}. Skipping.")
                    continue

                for j, (sim_timestep, pred_np, target_np) in enumerate(epoch_data['data']):
                    
                    # Check if this timestep's data has the expected number of dofs
                    if pred_np.size != num_dofs or target_np.size != num_dofs:
                        print(f"Warning: Data size mismatch in epoch {epoch_num}, sim_timestep {sim_timestep}. Skipping timestep.")
                        continue
                        
                    error_np = pred_np - target_np
                    x_coords = np.arange(num_dofs) # Represents DOF index

                    for i in range(num_dofs):
                        f.write(f"{x_coords[i]} {sim_timestep} {epoch_num} {pred_np[i]:.8f} {target_np[i]:.8f} {error_np[i]:.8f}\n")
        
        print(f"Saved simulation data Tecplot file to {output_path}")

    except Exception as e:
        print(f"Could not create simulation data Tecplot file: {e}")


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


    # --- 1. Plot combined final timestep (REMOVED as per user request) ---

    # --- Generate plots for min loss epoch ---
    if min_loss_epoch != -1:
        print(f"\nGenerating additional plots for minimum loss epoch {min_loss_epoch}...")

        # --- 1a. Plot combined final timestep for min loss epoch ---
        if 'preds_collection' in data and 'epochs_collection' in data:
            epoch_in_collection_idx = np.where(data['epochs_collection'] == min_loss_epoch)[0]
            if len(epoch_in_collection_idx) > 0:
                idx = epoch_in_collection_idx[0]
                print("Generating combined final timestep plot for min loss epoch...")
                min_loss_fig = plot_combined_final_timestep(
                    [data['preds_collection'][idx]],
                    [data['epochs_collection'][idx]],
                    data['target_final_global']
                )
                if min_loss_fig:
                    min_loss_fig.savefig(plot_output_dir / f"combined_final_timestep_min_loss_epoch_{min_loss_epoch}.png")
                    plt.close(min_loss_fig)
                    print(f"Saved combined final timestep plot for min loss epoch.")
            else:
                print("Could not find prediction data for min loss epoch in preds_collection.")

    # --- 2. Write nn output vs c for all saved epochs to Tecplot ---
    print("\nGenerating nn output vs c Tecplot file for all saved epochs...")
    if 'c_values_nn' in data and 'all_nn_outputs' in data and 'nn_output_label' in data:
        ylabel = str(data['nn_output_label'])
        c_vals = data['c_values_nn']
        all_nn_outputs = data['all_nn_outputs']
        output_path = plot_output_dir / "nn_output_vs_c.dat"
        
        write_nn_output_tecplot(c_vals, all_nn_outputs, ylabel, output_path)

    elif 'c_values_nn' in data and 'nn_output_values' in data and 'nn_output_label' in data: # Backwards compatibility
        print("Found old format 'nn_output_values'. Plotting for final model state only.")
        ylabel = str(data['nn_output_label'])
        c_vals = data['c_values_nn']
        # Create a structure that the new tecplot writer can understand
        all_nn_outputs = [{'epoch': data['epochs_collection'][-1], 'output': data['nn_output_values']}]
        output_path = plot_output_dir / "nn_output_vs_c.dat"
        write_nn_output_tecplot(c_vals, all_nn_outputs, ylabel, output_path)

    else:
        print("Skipping nn output tecplot file: Data not found in .npz file.")

    # --- 3. Plot loss vs epochs ---
    print("\nGenerating loss vs epochs plot...")
    if 'epoch_losses' in data and 'epoch_numbers' in data:
        plot_loss_vs_epochs(
            data['epoch_numbers'],
            data['epoch_losses'],
            plot_output_dir / "loss_vs_epochs.png"
        )
    else:
        print("Skipping loss plot: Data not found in .npz file.")

    # --- 4. Write multi-timestep comparisons to Tecplot file ---
    print("\nGenerating multi-timestep comparison Tecplot file...")
    if 'all_epochs_comparison_data' in data:
        all_epochs_data = data['all_epochs_comparison_data']
        output_path = plot_output_dir / "simulation_data.dat"
        write_simulation_data_tecplot(all_epochs_data, output_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce plots from ch_learn.py's .npz output.")
    parser.add_argument("npz_file", type=str, help="Path to the post_processing_data.npz file.")
    args = parser.parse_args()

    reproduce_plots(args.npz_file)
