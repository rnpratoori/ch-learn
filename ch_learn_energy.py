from firedrake import *
from firedrake.adjoint import *   # provides ReducedFunctional, Control, etc.
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from plotting import plot_combined_final_timestep, plot_dfdc_vs_c, plot_multi_timestep_comparison, plot_f_vs_c
from simulation import setup_firedrake, solve_one_step, load_target_data
from checkpoint import save_checkpoint, load_checkpoint

# ----------------------
# Hyperparameters
# ----------------------
config = {
    "learning_rate": 1e-3,
    "epochs": 100,
    "seed": 12,
}

checkpoint_filename = "ch_learn_energy_model.pth"

# ----------------------
# PyTorch model
# ----------------------
class FEnergy(nn.Module):
    def __init__(self):
        super(FEnergy, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 50), # input c and t
            nn.LeakyReLU(),
            nn.Linear(50, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, c):
        return self.mlp(c)

# Seeds
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])
torch.set_default_dtype(torch.float64)
# Instantiate network and optimizer
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using PyTorch device: {device}")
f_net = FEnergy().to(device)
f_net.double()
optimizer = optim.Adam(f_net.parameters(), lr=config["learning_rate"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

# ----------------------
# Problem setup
# ----------------------
lmbda = 5e-2
dt = 1e-2
T = 1e0
M = 1.0

num_timesteps = int(T / dt)

mesh, V, W, u_ic = setup_firedrake()

u = Function(W, name="Solution")
c, mu = split(u)
v = TestFunction(W)
c_test, mu_test = split(v)

# ----------------------
# Load target (from PVD via pyvista)
# ----------------------
c_target_list, _, _ = load_target_data(num_timesteps, V, None, 0)

# ----------------------
# Training using firedrake-adjoint
# ----------------------
num_epochs = config["epochs"]

video_frame_save_freq = num_epochs/100
checkpoint_freq = num_epochs/20
plot_loss_freq = num_epochs/20
dfdc_plot_freq = num_epochs/100
vtk_out = VTKFile("ch_learn_adjoint.pvd")

# ----------------------
# Argument Parsing
# ----------------------
parser = argparse.ArgumentParser(description='Cahn-Hilliard learning script.')
parser.add_argument('--no-resume', action='store_true', help='Start training from scratch, ignoring any checkpoints.')
args = parser.parse_args()

# Initialize wandb
wandb.init(project="ch_learn", config=config)

# Load from checkpoint if available and not disabled
start_epoch = 0
if not args.no_resume:
    start_epoch = load_checkpoint(f_net, optimizer, device, filename=checkpoint_filename)
else:
    print("Starting training from scratch as requested by --no-resume flag.")
preds_collection = []    # list of ndarray, each is full global DOF vector for a checkpoint epoch
epochs_collection = []   # corresponding epoch numbers
target_final_global = None

get_working_tape().progress_bar = ProgressBar

min_loss = float('inf')

for epoch in range(start_epoch, num_epochs):
    # clear previous tape
    get_working_tape().clear_tape()

    epoch_t0 = time.perf_counter()

    continue_annotation()

    # reset solution
    u_curr = u_ic.copy(deepcopy=True)

    # --- FORWARD PASS ---
    simulation_start = time.perf_counter()
    c_inputs = []
    dfdc_outputs = []
    J_adj = 0.0  # Adjoint functional for firedrake-adjoint
    J_total_log = 0.0  # Scalar loss for logging
    comparison_data = []

    for i in range(num_timesteps):
        c_curr = u_curr.sub(0)
        c_snapshot = Function(V, name=f"c_snapshot_{i}")
        c_snapshot.assign(c_curr)
        c_inputs.append(c_snapshot)

        c_vec = c_curr.dat.data_ro.copy().astype(np.float64)
        c_tensor = torch.from_numpy(c_vec.reshape(-1, 1)).to(device).requires_grad_(True)
        
        # Predict f and compute df/dc using autograd
        f_tensor = f_net(c_tensor)
        dfdc_tensor = torch.autograd.grad(f_tensor.sum(), c_tensor, create_graph=True)[0]
        
        dfdc_np = dfdc_tensor.detach().cpu().numpy().reshape(-1)

        dfdc_f = Function(V, name=f"dfdc_pred_{i}")
        dfdc_f.dat.data[:] = dfdc_np
        dfdc_outputs.append(dfdc_f)

        u_next = solve_one_step(u_curr, dfdc_f, u, c, mu, c_test, mu_test, dt, M, lmbda)
        u_curr.assign(u_next)

        if (i + 1) % 20 == 0 or (i + 1) == num_timesteps:
            comparison_data.append((i, u_curr.sub(0).copy(deepcopy=True), c_target_list[i]))

        # --- LOSS CALCULATION (FFT) ---
        u_curr_np = u_curr.sub(0).dat.data_ro
        target_np = c_target_list[i].dat.data_ro
        u_tensor = torch.tensor(u_curr_np, device=device, requires_grad=True)
        t_tensor = torch.tensor(target_np, device=device)

        fft_u = torch.fft.fft(u_tensor)
        fft_t = torch.fft.fft(t_tensor)
        loss_i = 0.5 * torch.mean(torch.abs(fft_u - fft_t)**2)
        
        weight = 1.0 if i <= 20 else 1.0
        
        (weight * loss_i).backward()
        grad_u_tensor = u_tensor.grad

        g_i = Function(V)
        g_i.dat.data[:] = grad_u_tensor.cpu().numpy()

        J_adj += assemble(inner(g_i, u_curr.sub(0)) * dx)
        J_total_log += weight * loss_i.item()

        if epoch == num_epochs - 1:
            t = (i + 1) * dt
            vtk_out.write(project(u_curr.sub(0), V, name="Volume Fraction"), time=t)

    simulation_time = time.perf_counter() - simulation_start

    pause_annotation()

    # --- ADJOINT GRADIENT ---
    adjoint_grad_start = time.perf_counter()
    controls = [Control(d) for d in dfdc_outputs]
    rf = ReducedFunctional(J_adj, controls)

    dJ_dcs = rf.derivative()
    adjoint_grad_time = time.perf_counter() - adjoint_grad_start

    # --- PYTORCH BACKPROPAGATION ---
    backprop_start = time.perf_counter()
    optimizer.zero_grad()

    for i in range(num_timesteps):
        c_i_vec = c_inputs[i].dat.data_ro.copy().astype(np.float64)
        c_i_tensor = torch.from_numpy(c_i_vec.reshape(-1, 1)).to(device).requires_grad_(True)

        # Predict f and compute df/dc to reconstruct the graph
        f_i_tensor = f_net(c_i_tensor)
        dfdc_i_tensor = torch.autograd.grad(f_i_tensor.sum(), c_i_tensor, create_graph=True)[0]

        dJ_dcs_i_np = dJ_dcs[i].dat.data_ro.copy()
        sens_i_tensor = torch.from_numpy(dJ_dcs_i_np.astype(np.float64)).reshape(-1, 1).to(device)

        # Backpropagate the sensitivity from the adjoint pass through the df/dc calculation
        dfdc_i_tensor.backward(gradient=sens_i_tensor)

    optimizer.step()
    backprop_time = time.perf_counter() - backprop_start

    # --- LOGGING ---
    elapsed_time = time.perf_counter() - epoch_t0
    loss_epoch = J_total_log
    scheduler.step(loss_epoch)
    save_min_loss_plots_now = False

    print(f"Epoch {epoch+1}/{num_epochs} finished in {elapsed_time:.2f} s, J={J_total_log:.6e}")
    wandb.log({"loss": loss_epoch, "epoch": epoch})

    if loss_epoch < min_loss:
        min_loss = loss_epoch
        print(f"New minimum loss: {min_loss:.6e}. Saving plots.")
        save_min_loss_plots_now = True

    # --- CHECKPOINTING ---
    if (epoch + 1) % checkpoint_freq == 0 or epoch == num_epochs - 1:
        save_checkpoint(epoch, f_net, optimizer, filename=checkpoint_filename)

    # --- Visualization ---
    save_video_frame_now = ((epoch + 1) % video_frame_save_freq == 0) or (epoch == num_epochs - 1)
    save_dfdc_plot_now = ((epoch + 1) % dfdc_plot_freq == 0) or (epoch == num_epochs - 1)
    if save_video_frame_now or save_min_loss_plots_now or save_dfdc_plot_now:
        pred_global = u_curr.sub(0).dat.data_ro.copy().astype(np.float64)
        target_global = c_target_list[-1].dat.data_ro.copy().astype(np.float64)

        preds_collection.append(pred_global.copy())
        epochs_collection.append(epoch + 1)
        if target_final_global is None:
            target_final_global = target_global.copy()

        if save_video_frame_now:
            multi_ts_video_frame_fig = plot_multi_timestep_comparison(epoch + 1, comparison_data)
            if multi_ts_video_frame_fig:
                wandb.log({"video_frame": wandb.Image(multi_ts_video_frame_fig)}, commit=False)
                plt.close(multi_ts_video_frame_fig)

        if save_dfdc_plot_now:
            f_fig = plot_f_vs_c(f_net, device)
            if f_fig:
                wandb.log({"f_plot": wandb.Image(f_fig)}, commit=False)
                plt.close(f_fig)

        if save_min_loss_plots_now:
            f_fig_min = plot_f_vs_c(f_net, device)
            if f_fig_min:
                wandb.log({"f_plot_min_loss": wandb.Image(f_fig_min)}, commit=False)
                plt.close(f_fig_min)
            
            multi_ts_fig = plot_multi_timestep_comparison(epoch + 1, comparison_data)
            if multi_ts_fig:
                wandb.log({"multi_timestep_comparison_min_loss": wandb.Image(multi_ts_fig)}, commit=False)
                plt.close(multi_ts_fig)

print("Training finished (adjoint update).")
# Create one combined plot with all collected checkpoint epochs (only last timestep)
combined_fig = plot_combined_final_timestep(preds_collection, epochs_collection, target_final_global)
if combined_fig:
    wandb.log({"combined_final_timestep_plot": wandb.Image(combined_fig)})
    plt.close(combined_fig)

# Plot the learned f curve
f_fig_final = plot_f_vs_c(f_net, device)
if f_fig_final:
    wandb.log({"f_plot_final": wandb.Image(f_fig_final)})
    plt.close(f_fig_final)
# Save data for post-processing
np.savez("post_processing_data.npz",
         preds_collection=np.array(preds_collection),
         epochs_collection=np.array(epochs_collection),
         target_final_global=target_final_global)

wandb.finish()
