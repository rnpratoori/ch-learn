from firedrake import *
from firedrake.adjoint import *   # provides ReducedFunctional, Control, etc.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import meshio
import pyvista as pv
from mpi4py import MPI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
import os
import subprocess
from plotting import plot_combined_final_timestep, create_video, save_comparison_image, save_video_frame, plot_loss_over_epochs
from simulation import setup_firedrake, solve_one_step, load_target_data
from checkpoint import save_checkpoint, load_checkpoint

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ----------------------
# PyTorch model
# ----------------------
class FEDerivative(nn.Module):
    def __init__(self):
        super(FEDerivative, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 20), # input c and t
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, c):
        return self.mlp(c)

# Seeds
torch.manual_seed(12)
np.random.seed(12)
torch.set_default_dtype(torch.float64)
# Instantiate network and optimizer
# Use CUDA only when running with a single process (size == 1).
# If multiple MPI ranks are used, default to CPU unless you have per-rank GPUs.
if torch.cuda.is_available() and size == 1:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
if rank == 0:
    print(f"Using PyTorch device: {device} (rank {rank}/{size})")
dfdc_net = FEDerivative().to(device)
dfdc_net.double()
optimizer = optim.Adam(dfdc_net.parameters(), lr=5e-3)

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
c_target_list, counts, displs = load_target_data(num_timesteps, V, comm, rank)

# ----------------------
# Training using firedrake-adjoint
# ----------------------
num_epochs = 20000
comparison_image_save_freq = 1000
video_frame_save_freq = 100
checkpoint_freq = 100
vtk_out = VTKFile("ch_learn_adjoint.pvd")

# Load from checkpoint if available
start_epoch, losses = load_checkpoint(dfdc_net, optimizer, device, rank)
if rank == 0:
    preds_collection = []    # list of ndarray, each is full global DOF vector for a checkpoint epoch
    epochs_collection = []   # corresponding epoch numbers
    target_final_global = None
    # frames directory & list for assembling video (rank 0 only)
    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)
    frames_list = []

if rank == 0:
    get_working_tape().progress_bar = ProgressBar

for epoch in range(start_epoch, num_epochs):
    # synchronize and start epoch timing (use perf_counter for better resolution)
    comm.Barrier()
    epoch_t0 = time.perf_counter()

    continue_annotation()

    # reset solution
    u_curr = u_ic.copy(deepcopy=True)

    # --- FORWARD PASS ---
    # Run the forward model and store the neural network inputs (c) and outputs (dfdc) for each time step.
    simulation_start = time.perf_counter()
    c_inputs = []
    dfdc_outputs = []
    J_total = 0.0

    for i in range(num_timesteps):
        c_curr = u_curr.sub(0)
        c_snapshot = Function(V, name=f"c_snapshot_{i}")
        c_snapshot.assign(c_curr)
        c_inputs.append(c_snapshot)

        c_vec = c_curr.dat.data_ro.copy().astype(np.float64)
        with torch.no_grad():
            c_tensor = torch.from_numpy(c_vec.reshape(-1, 1)).to(device)
            dfdc_np = dfdc_net(c_tensor).cpu().numpy().reshape(-1)

        dfdc_f = Function(V, name=f"dfdc_pred_{i}")
        dfdc_f.dat.data[:] = dfdc_np
        dfdc_outputs.append(dfdc_f)

        u_next = solve_one_step(u_curr, dfdc_f, u, c, mu, c_test, mu_test, dt, M, lmbda)
        u_curr.assign(u_next)

        # --- LOSS CALCULATION ---
        J_total += assemble(0.5 * (u_curr.sub(0) - c_target_list[i])**2 * dx)

        # Write this timestep only for the final epoch (match ch_fh.py behavior)
        if epoch == num_epochs - 1:
            t = (i + 1) * dt
            vtk_out.write(project(u_curr.sub(0), V, name="Volume Fraction"), time=t)

    # synchronize and reduce forward simulation time (max across ranks)
    comm.Barrier()
    simulation_time_local = time.perf_counter() - simulation_start
    simulation_time = comm.allreduce(simulation_time_local, op=MPI.MAX)

    pause_annotation()

    # --- ADJOINT GRADIENT ---
    # Compute the gradient of the loss with respect to the network outputs (dfdc) using the adjoint method.
    comm.Barrier()
    adjoint_grad_start = time.perf_counter()
    controls = [Control(d) for d in dfdc_outputs]
    rf = ReducedFunctional(J_total, controls)

    # derivative() will return a list of gradients, one for each control
    dJ_dcs = rf.derivative()
    # synchronize and reduce adjoint timing (max across ranks)
    comm.Barrier()
    adjoint_grad_time_local = time.perf_counter() - adjoint_grad_start
    adjoint_grad_time = comm.allreduce(adjoint_grad_time_local, op=MPI.MAX)

    # --- PYTORCH BACKPROPAGATION ---
    # synchronize and time PyTorch backprop (including optimizer.step)
    comm.Barrier()
    backprop_start = time.perf_counter()
    optimizer.zero_grad()

    for i in range(num_timesteps):
        # Recompute the network output for this step with autograd enabled
        c_i_vec = c_inputs[i].dat.data_ro.copy().astype(np.float64)
        c_i_tensor = torch.from_numpy(c_i_vec.reshape(-1, 1)).to(device)
        dfdc_i_tensor = dfdc_net(c_i_tensor)


        dJ_dcs_i_np = dJ_dcs[i].dat.data_ro.copy()
        sens_i_tensor = torch.from_numpy(dJ_dcs_i_np.astype(np.float64)).reshape(-1, 1).to(device)

        # Form scalar loss for this time step and backpropagate
        # The gradients will be accumulated in the network parameters
        scalar_for_backprop = torch.sum(dfdc_i_tensor * sens_i_tensor)
        scalar_for_backprop.backward()

    # Update the network weights once with the accumulated gradients
    optimizer.step()
    # synchronize and reduce backprop timing (max across ranks)
    comm.Barrier()
    backprop_time_local = time.perf_counter() - backprop_start
    backprop_time = comm.allreduce(backprop_time_local, op=MPI.MAX)

    # --- LOGGING ---
    # synchronize and reduce epoch timing (max across ranks)
    comm.Barrier()
    elapsed_local = time.perf_counter() - epoch_t0
    elapsed_time = comm.allreduce(elapsed_local, op=MPI.MAX)
    # Record scalar loss for this epoch (assemble returns a global scalar)
    loss_epoch = float(J_total)
    if rank == 0:
        losses.append(loss_epoch)
        print(f"Epoch {epoch+1}/{num_epochs} finished in {elapsed_time:.2f} s, J={float(J_total):.6e}")

    # --- CHECKPOINTING ---
    if (epoch + 1) % checkpoint_freq == 0 or epoch == num_epochs - 1:
        save_checkpoint(epoch, dfdc_net, optimizer, losses, rank)

    
    # --- Print timings ---
    # if rank == 0:
    #     print(f"--- Epoch {epoch+1} Timings ---")
    #     print(f"  Firedrake Simulation:  {simulation_time:.4f}s")
    #     print(f"  Adjoint Gradient:      {adjoint_grad_time:.4f}s")
    #     print(f"  PyTorch Backprop:      {backprop_time:.4f}s")
    #     print(f"  ---------------------------")
    #     total_timed = simulation_time + adjoint_grad_time + backprop_time
    #     print(f"  Total Timed Sections:   {total_timed:.4f}s")
    #     print(f"  Total Epoch Time:         {elapsed_time:.4f}s")

    # --- Visualization ---
    # Periodically gather data to the root process to save plots and video frames.
    save_comparison_image_now = ((epoch + 1) % comparison_image_save_freq == 0) or (epoch == num_epochs - 1)
    save_video_frame_now = ((epoch + 1) % video_frame_save_freq == 0) or (epoch == num_epochs - 1)
    if save_comparison_image_now or save_video_frame_now:
        # local arrays (each rank)
        pred_local = u_curr.sub(0).dat.data_ro.copy().astype(np.float64)
        target_local = c_target_list[-1].dat.data_ro.copy().astype(np.float64)

        # prepare recv buffers on rank 0 (global size = sum(counts))
        if rank == 0:
            global_n = counts.sum()
            pred_global = np.empty(global_n, dtype=np.float64)
            target_global = np.empty(global_n, dtype=np.float64)
        else:
            pred_global = None
            target_global = None

        # Gatherv from all ranks into rank 0
        comm.Gatherv([pred_local, MPI.DOUBLE],
                     [pred_global, counts, displs, MPI.DOUBLE],
                     root=0)
        comm.Gatherv([target_local, MPI.DOUBLE],
                     [target_global, counts, displs, MPI.DOUBLE],
                     root=0)

        # Rank 0: save per-25 image (preserved) and/or a frame for the video
        if rank == 0:
            # Store the gathered prediction for the final combined plot.
            preds_collection.append(pred_global.copy())
            epochs_collection.append(epoch + 1)
            if target_final_global is None:
                target_final_global = target_global.copy()

            if save_comparison_image_now:
                save_comparison_image(epoch + 1, pred_global, target_global)

            if save_video_frame_now:
                try:
                    loss_text = loss_epoch  # available in scope
                except NameError:
                    loss_text = float(J_total)
                save_video_frame(epoch + 1, pred_global, target_global, loss_text, frames_dir, frames_list)

if rank == 0:
    print("Training finished (adjoint update).")
    # Create one combined plot with all collected checkpoint epochs (only last timestep)
    plot_combined_final_timestep(preds_collection, epochs_collection, target_final_global)

    # Assemble frames into a video (rank 0 only).
    create_video(frames_list, frames_dir)

    # Plot loss vs epoch (rank 0 only; saved to PNG)
    plot_loss_over_epochs(losses)