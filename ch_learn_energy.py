from firedrake import *
from firedrake.adjoint import *   # provides ReducedFunctional, Control, etc.
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from plotting import plot_combined_final_timestep, plot_nn_output_vs_c, plot_multi_timestep_comparison_2d, plot_multi_timestep_comparison_3d, plot_loss_vs_epochs
from simulation import solve_one_step, load_target_data
from checkpoint import save_checkpoint, load_checkpoint
import os
from pathlib import Path

output_dir = Path(os.getenv("OUTPUT_DIR", "."))
output_dir.mkdir(parents=True, exist_ok=True)

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
# Argument Parsing
# ----------------------
parser = argparse.ArgumentParser(description='Cahn-Hilliard learning script (Energy version).')
parser.add_argument('--no-resume', action='store_true', help='Start training from scratch, ignoring any checkpoints.')
args = parser.parse_args()

# ----------------------
# Problem setup
# ----------------------
lmbda = 5e-2
dt = 1e-2
T = 1e0
M = 1.0
num_timesteps = int(T / dt)

# Create the mesh and function spaces
mesh = UnitIntervalMesh(100)
V = FunctionSpace(mesh, "Lagrange", 1)
W = V * V

# Load the target data
c_target_list, _, _ = load_target_data(num_timesteps, V, None, 0)

# Setup the initial condition
u_ic = Function(W, name="Initial_condition")
print("Setting initial condition from the first timestep of the target data.")
u_ic.sub(0).assign(c_target_list[0])
u_ic.sub(1).assign(0.0)

u = Function(W, name="Solution")
c, mu = split(u)
v = TestFunction(W)
c_test, mu_test = split(v)

# Initialize wandb
wandb.init(project="ch_learn", config=config)

# Add frequency definitions
num_epochs = config["epochs"]
video_frame_save_freq = num_epochs/100
checkpoint_freq = num_epochs/20
plot_loss_freq = num_epochs/20
dfdc_plot_freq = num_epochs/100
npz_save_freq = num_epochs/10 # Save every 10% of epochs
vtk_out = VTKFile(output_dir / "ch_learn_adjoint.pvd")

# Load from checkpoint if available and not disabled
start_epoch = 0
if not args.no_resume:
    start_epoch = load_checkpoint(f_net, optimizer, device, output_dir, filename=checkpoint_filename)
else:
    print("Starting training from scratch as requested by --no-resume flag.")
preds_collection = []    # list of ndarray, each is full global DOF vector for a checkpoint epoch
epochs_collection = []   # corresponding epoch numbers
target_final_global = None
all_epochs_comparison_data = []



min_loss = float('inf')
epoch_losses = []
epoch_numbers = []

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

    # --- Post-processing and data collection for this epoch ---
    processed_comparison_data = []
    for i, u_pred, c_targ in comparison_data:
        processed_comparison_data.append(
            (i, u_pred.dat.data_ro.copy(), c_targ.dat.data_ro.copy())
        )
    all_epochs_comparison_data.append({'epoch': epoch, 'data': processed_comparison_data})


    # --- LOGGING ---
    elapsed_time = time.perf_counter() - epoch_t0
    loss_epoch = J_total_log
    scheduler.step(loss_epoch)
    save_min_loss_plots_now = False

    epoch_losses.append(loss_epoch)
    epoch_numbers.append(epoch + 1)

    print(f"Epoch {epoch+1}/{num_epochs} finished in {elapsed_time:.2f} s, J={J_total_log:.6e}")
    wandb.log({"loss": loss_epoch, "epoch": epoch})

    if loss_epoch < min_loss:
        min_loss = loss_epoch
        print(f"New minimum loss: {min_loss:.6e}. Saving plots.")
        save_min_loss_plots_now = True

    # --- CHECKPOINTING ---
    if (epoch + 1) % checkpoint_freq == 0 or epoch == num_epochs - 1:
        save_checkpoint(epoch, f_net, optimizer, epoch_losses, epoch_numbers, output_dir, filename=checkpoint_filename)

    # --- PLOT LOSS ---
    if (epoch + 1) % plot_loss_freq == 0 or epoch == num_epochs - 1:
        plot_loss_vs_epochs(epoch_numbers, epoch_losses, output_dir / "loss_vs_epochs_energy.png")

    # --- Visualization ---
    # All wandb plotting is disabled in favor of reproduce_plots.py
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
            pass # wandb plotting disabled

        if save_dfdc_plot_now:
            pass # wandb plotting disabled

        if save_min_loss_plots_now:
            pass # wandb plotting disabled

    # --- NPZ Checkpointing ---
    if (epoch + 1) % npz_save_freq == 0 or epoch == num_epochs - 1:
        print(f"Saving .npz data at epoch {epoch + 1}...")
        # Generate nn output vs c plot data
        c_values_nn = np.linspace(0, 1, 200).reshape(-1, 1)
        c_tensor_nn = torch.from_numpy(c_values_nn).to(device)
        with torch.no_grad():
            nn_output_values = f_net(c_tensor_nn).cpu().numpy()

        np.savez(output_dir / "post_processing_energy.npz",
                 preds_collection=np.array(preds_collection),
                 epochs_collection=np.array(epochs_collection),
                 target_final_global=target_final_global,
                 all_epochs_comparison_data=np.array(all_epochs_comparison_data, dtype=object),
                 c_values_nn=c_values_nn,
                 nn_output_values=nn_output_values,
                 epoch_losses=np.array(epoch_losses),
                 epoch_numbers=np.array(epoch_numbers),
                 nn_output_label=np.array("Free Energy (f)"))
        
        # Save the npz file to wandb
        wandb.save(str(output_dir / "post_processing_energy.npz"))

print("Training finished (adjoint update).")

wandb.finish()
