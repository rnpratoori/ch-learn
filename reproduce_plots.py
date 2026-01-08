import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import argparse
import plotly.graph_objects as go

# Assuming plotting.py is in the same directory or accessible
from plotting import plot_combined_final_timestep, plot_multi_timestep_comparison_2d, plot_multi_timestep_comparison_3d, plot_loss_vs_epochs

def plot_nn_output_vs_c_from_data(c_values, nn_output_values, ylabel, title, output_path):
    """
    Plots nn output vs c from saved data using Plotly.
    """
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c_values.flatten(), y=nn_output_values.flatten(), mode='lines'))
        fig.update_layout(
            title=title,
            xaxis_title="Concentration (c)",
            yaxis_title=ylabel,
            template="plotly_white"
        )
        fig.write_html(str(output_path).replace('.png', '.html'))
        print(f"Saved nn output plot to {output_path.with_suffix('.html')}")
    except Exception as e:
        print(f"Could not create nn output vs c plot: {e}")

def reproduce_plots(npz_path):
    """
    Loads data from an .npz file and reproduces the plots as interactive HTML.
    """
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        print(f"Error: File not found at {npz_path}")
        return

    # Create output directory for plots
    plot_output_dir = npz_path.parent / "reproduced_plots"
    plot_output_dir.mkdir(exist_ok=True)
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
            min_loss_epoch = min_loss_idx + 1
            print(f"\nMinimum loss of {data['epoch_losses'][min_loss_idx]} found at index {min_loss_idx}.")
    else:
        print("\nLoss data not found, cannot determine minimum loss epoch.")


    # --- 1. Plot combined final timestep ---
    print("Generating combined final timestep plot...")
    if 'preds_collection' in data and 'epochs_collection' in data and 'target_final_global' in data:
        combined_fig = plot_combined_final_timestep(
            data['preds_collection'],
            data['epochs_collection'],
            data['target_final_global']
        )
        if combined_fig:
            combined_fig.write_html(plot_output_dir / "combined_final_timestep.html")
            print(f"Saved combined final timestep plot.")

    # --- Generate plots for min loss epoch ---
    if min_loss_epoch != -1:
        print(f"\nGenerating additional plots for minimum loss epoch {min_loss_epoch}...")

        if 'preds_collection' in data and 'epochs_collection' in data:
            epoch_in_collection_idx = np.where(data['epochs_collection'] == min_loss_epoch)[0]
            if len(epoch_in_collection_idx) > 0:
                idx = epoch_in_collection_idx[0]
                min_loss_fig = plot_combined_final_timestep(
                    [data['preds_collection'][idx]],
                    [data['epochs_collection'][idx]],
                    data['target_final_global']
                )
                if min_loss_fig:
                    min_loss_fig.write_html(plot_output_dir / f"combined_final_timestep_min_loss_epoch_{min_loss_epoch}.html")
                    print(f"Saved combined final timestep plot for min loss epoch.")

    # --- 2. Plot nn output vs c ---
    print("\nGenerating nn output vs c plot...")
    if 'c_values_nn' in data and 'all_nn_outputs' in data and 'nn_output_label' in data:
        ylabel = str(data['nn_output_label'])
        # Use final output
        nn_outputs = data['all_nn_outputs']
        if len(nn_outputs) > 0:
            final_output = nn_outputs[-1]['output']
            title = f"Learned {ylabel} vs. Concentration (Final Epoch)"
            plot_nn_output_vs_c_from_data(
                data['c_values_nn'],
                final_output,
                ylabel,
                title,
                plot_output_dir / "nn_output_vs_c.html"
            )

    # --- 3. Plot loss vs epochs ---
    print("\nGenerating loss vs epochs plot...")
    if 'epoch_losses' in data and 'epoch_numbers' in data:
        min_l = np.min(data['epoch_losses'])
        plot_loss_vs_epochs(
            data['epoch_numbers'],
            data['epoch_losses'],
            plot_output_dir / "loss_vs_epochs.html",
            min_loss=min_l
        )

    # --- 4. Plot multi-timestep comparisons ---
    print("\nGenerating multi-timestep comparison plots...")
    if 'all_epochs_comparison_data' in data:
        all_epochs_data = data['all_epochs_comparison_data']
        num_epochs_from_data = data['epoch_numbers'][-1] if 'epoch_numbers' in data else len(all_epochs_data)

        # Plot freq
        video_frame_save_freq = max(1, num_epochs_from_data // 100)
        
        for epoch_data in all_epochs_data:
            epoch = epoch_data['epoch']
            plot_now = ((epoch + 1) % video_frame_save_freq == 0) or \
                       ((epoch + 1) == num_epochs_from_data) or \
                       ((epoch + 1) == min_loss_epoch)

            if plot_now:
                filename_suffix = f"_epoch_{epoch+1}"
                if (epoch + 1) == min_loss_epoch:
                    filename_suffix = f"_min_loss_epoch_{epoch+1}"

                fig2d = plot_multi_timestep_comparison_2d(epoch + 1, epoch_data['data'])
                if fig2d:
                    fig2d.write_html(plot_output_dir / f"multi_ts_comparison_2d{filename_suffix}.html")
                
                fig3d = plot_multi_timestep_comparison_3d(epoch + 1, epoch_data['data'])
                if fig3d:
                    fig3d.write_html(plot_output_dir / f"multi_ts_comparison_3d{filename_suffix}.html")
    else:
        print("Skipping multi-timestep plots: Data not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce plots from ch_learn.py's .npz output.")
    parser.add_argument("npz_file", type=str, help="Path to the post_processing_data.npz file.")
    args = parser.parse_args()

    reproduce_plots(args.npz_file)
