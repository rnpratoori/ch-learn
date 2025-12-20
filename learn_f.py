"""
Cahn-Hilliard Learning Script (Energy Version)
Learns the free energy f(c) using neural networks and adjoint methods.
"""

from firedrake import *
from firedrake.adjoint import *
import os

# Set threading limits to avoid conflicts with MPI
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch

import time
import wandb
import matplotlib
matplotlib.use("Agg")

from training_utils import (
    parse_arguments, setup_device, setup_output_dir, initialize_training
)
from models.energy import FEnergy
from simulation import CHSolver, load_target_data
from checkpoint import save_checkpoint
from plotting import plot_loss_vs_epochs

# Limit PyTorch threads
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def setup_problem(num_timesteps):
    """Setup the Cahn-Hilliard problem: mesh, function spaces, and target data."""
    # Create mesh and function spaces
    mesh = IntervalMesh(200, 2)
    V = FunctionSpace(mesh, "Lagrange", 1)
    W = V * V
    
    # Load target data
    c_target_list, _, _ = load_target_data(num_timesteps, V, None, 0)
    
    # Setup initial condition
    u_ic = Function(W, name="Initial_condition")
    print("Setting initial condition from the first timestep of the target data.")
    u_ic.sub(0).assign(c_target_list[0])
    u_ic.sub(1).assign(0.0)
    
    # Create solution and test functions
    u = Function(W, name="Solution")
    c, mu = split(u)
    v = TestFunction(W)
    c_test, mu_test = split(v)
    
    return V, W, u_ic, u, c, mu, c_test, mu_test, c_target_list


def compute_loss_and_gradient(u_curr, target, device, weight=1.0):
    """Compute FFT-based loss and its gradient."""
    u_curr_np = u_curr.sub(0).dat.data_ro
    target_np = target.dat.data_ro
    
    u_tensor = torch.tensor(u_curr_np, device=device, requires_grad=True)
    t_tensor = torch.tensor(target_np, device=device)
    
    fft_u = torch.fft.fft(u_tensor)
    fft_t = torch.fft.fft(t_tensor)
    loss = 0.5 * torch.mean(torch.abs(fft_u - fft_t)**2)
    
    (weight * loss).backward()
    grad_u_tensor = u_tensor.grad
    
    return weight * loss.item(), grad_u_tensor


def train_epoch(epoch, num_epochs, model, optimizer, device, u_ic, u, c_target_list, 
                V, W, dt, M, lmbda, num_timesteps, vtk_out, ch_solver, use_wandb=True):
    """Execute one training epoch."""
    # Clear previous tape
    get_working_tape().clear_tape()
    
    epoch_t0 = time.perf_counter()
    continue_annotation()
    
    # Reset solution
    u_curr = u_ic.copy(deepcopy=True)
    
    # --- FORWARD PASS ---
    c_inputs = []
    dfdc_outputs = []
    J_adj = 0.0  # Adjoint functional
    J_total_log = 0.0  # Scalar loss for logging
    comparison_data = []

    for i in range(num_timesteps):
        c_curr = u_curr.sub(0)
        
        # Snapshot for backprop
        c_snapshot = Function(V, name=f"c_snapshot_{i}")
        c_snapshot.assign(c_curr)
        c_inputs.append(c_snapshot)
        
        # Neural network prediction: predict f(c), then compute df/dc via autograd
        c_vec = c_curr.dat.data_ro.copy().astype(np.float64)
        c_tensor = torch.from_numpy(c_vec.reshape(-1, 1)).to(device).requires_grad_(True)
        
        # Predict f and compute df/dc using autograd
        f_tensor = model(c_tensor)
        dfdc_tensor = torch.autograd.grad(f_tensor.sum(), c_tensor, create_graph=True)[0]
        
        dfdc_np = dfdc_tensor.detach().cpu().numpy().reshape(-1)
        
        # Create dfdc Function for solver and adjoint tracking
        dfdc_f = Function(V, name=f"dfdc_pred_{i}")
        dfdc_f.dat.data[:] = dfdc_np
        dfdc_outputs.append(dfdc_f)
        
        # Solve one timestep using CHSolver
        u_next = ch_solver.solve_step(u_curr, dfdc_f, u)
        u_curr.assign(u_next)
        
        # Store comparison data for visualization
        if (i + 1) % 20 == 0 or (i + 1) == num_timesteps:
            comparison_data.append((i, u_curr.sub(0).copy(deepcopy=True), c_target_list[i]))
        
        # --- LOSS CALCULATION ---
        loss_val, grad_u_tensor = compute_loss_and_gradient(u_curr, c_target_list[i], device)
        
        # Inject gradient into Firedrake adjoint
        g_i = Function(V)
        g_i.dat.data[:] = grad_u_tensor.cpu().numpy()
        
        J_adj += assemble(inner(g_i, u_curr.sub(0)) * dx)
        J_total_log += loss_val
        
        # Write VTK output on final epoch
        if epoch == num_epochs - 1:
            t = (i + 1) * dt
            vtk_out.write(project(u_curr.sub(0), V, name="Volume Fraction"), time=t)
    
    pause_annotation()
    
    # --- ADJOINT GRADIENT ---
    controls = [Control(d) for d in dfdc_outputs]
    rf = ReducedFunctional(J_adj, controls)
    dJ_dcs = rf.derivative()
    
    # --- PYTORCH BACKPROPAGATION (with autograd for df/dc) ---
    optimizer.zero_grad()
    
    for i in range(num_timesteps):
        c_i_vec = c_inputs[i].dat.data_ro.copy().astype(np.float64)
        c_i_tensor = torch.from_numpy(c_i_vec.reshape(-1, 1)).to(device).requires_grad_(True)
        
        # Predict f and compute df/dc to reconstruct the graph
        f_i_tensor = model(c_i_tensor)
        dfdc_i_tensor = torch.autograd.grad(f_i_tensor.sum(), c_i_tensor, create_graph=True)[0]
        
        dJ_dcs_i_np = dJ_dcs[i].dat.data_ro.copy()
        sens_i_tensor = torch.from_numpy(dJ_dcs_i_np.astype(np.float64)).reshape(-1, 1).to(device)
        
        # Backpropagate the sensitivity through the df/dc calculation
        dfdc_i_tensor.backward(gradient=sens_i_tensor)
    
    optimizer.step()
    
    # Process comparison data for return
    processed_comparison_data = [
        (i, u_pred.dat.data_ro.copy(), c_targ.dat.data_ro.copy())
        for i, u_pred, c_targ in comparison_data
    ]
    
    elapsed_time = time.perf_counter() - epoch_t0
    
    return J_total_log, elapsed_time, u_curr, processed_comparison_data


def save_npz_data(output_dir, epoch, preds_collection, epochs_collection, 
                  target_final_global, all_epochs_comparison_data, 
                  epoch_losses, epoch_numbers, model, device, use_wandb, all_nn_outputs):
    """Save post-processing data to .npz file."""
    print(f"Saving .npz data at epoch {epoch}...")
    c_values_nn = np.linspace(0, 1, 200).reshape(-1, 1)
    c_tensor_nn = torch.from_numpy(c_values_nn).to(device)
    with torch.no_grad():
        nn_output_values = model(c_tensor_nn).cpu().numpy()

    # Append current nn_output to the collection
    all_nn_outputs.append({'epoch': epoch, 'output': nn_output_values})
    
    npz_path = output_dir / "post_processing_energy.npz"
    np.savez(npz_path,
             preds_collection=np.array(preds_collection),
             epochs_collection=np.array(epochs_collection),
             target_final_global=target_final_global,
             all_epochs_comparison_data=np.array(all_epochs_comparison_data, dtype=object),
             c_values_nn=c_values_nn,
             all_nn_outputs=np.array(all_nn_outputs, dtype=object),
             epoch_losses=np.array(epoch_losses),
             epoch_numbers=np.array(epoch_numbers),
             nn_output_label=np.array("Free Energy (f)"))
    
    if use_wandb:
        wandb.save(str(npz_path))


def main():
    args = parse_arguments()
    
    # Override epochs if profiling
    if args.profile:
        print("Profiling mode enabled: reducing epochs to 2")
        args.epochs = 2
    
    output_dir = setup_output_dir(args)
    device = setup_device(args)
    
    # Problem parameters
    dt = 1e-3
    T = 1e-1
    M = 1.0
    lmbda = 5e-2
    num_timesteps = int(T / dt)
    
    # Setup problem
    V, W, u_ic, u, c, mu, c_test, mu_test, c_target_list = setup_problem(num_timesteps)
    
    # Initialize CH solver
    ch_solver = CHSolver(W, dt, M, lmbda)
    
    # Create model
    model = FEnergy()
    
    # Initialize training
    model, optimizer, scheduler, start_epoch, epoch_losses, epoch_numbers = initialize_training(
        args, model, device, output_dir
    )
    
    # Constants
    num_epochs = args.epochs
    checkpoint_freq = max(1, num_epochs // 20)
    
    # Save and plot frequency logic
    base_freq = max(1, num_epochs // 20)
    save_and_plot_freq = min(base_freq, 100)
    print(f"Data and plots will be saved every {save_and_plot_freq} epochs.")

    plot_loss_freq = save_and_plot_freq
    npz_save_freq = save_and_plot_freq
    
    vtk_out = VTKFile(str(output_dir / "ch_learn_energy_adjoint.pvd"))
    use_wandb = not args.no_wandb
    
    # Training state
    preds_collection = []
    epochs_collection = []
    target_final_global = None
    all_epochs_comparison_data = []
    min_loss = float('inf')
    all_nn_outputs = []
    
    # Resume NPZ data if needed
    npz_path = output_dir / "post_processing_energy.npz"
    if start_epoch > 0 and npz_path.exists():
        print(f"Resuming from checkpoint, loading existing .npz data from {npz_path}")
        with np.load(npz_path, allow_pickle=True) as data:
            preds_collection = list(data.get('preds_collection', []))
            epochs_collection = list(data.get('epochs_collection', []))
            target_final_global = data.get('target_final_global')
            all_epochs_comparison_data = list(data.get('all_epochs_comparison_data', []))
            all_nn_outputs = list(data.get('all_nn_outputs', []))

    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(start_epoch, num_epochs):
        loss_epoch, elapsed_time, u_curr, processed_comparison_data = train_epoch(
            epoch, num_epochs, model, optimizer, device, u_ic, u, c_target_list,
            V, W, dt, M, lmbda, num_timesteps, vtk_out, ch_solver, use_wandb
        )
        
        old_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store losses
        epoch_losses.append(loss_epoch)
        epoch_numbers.append(epoch + 1)
        all_epochs_comparison_data.append({'epoch': epoch, 'data': processed_comparison_data})
        
        # Logging
        if use_wandb:
            wandb.log({"loss": loss_epoch, "epoch": epoch})
            if scheduler is not None:
                 wandb.log({"learning_rate": current_lr, "epoch": epoch})

        if loss_epoch < min_loss:
            min_loss = loss_epoch
            print(f"Epoch {epoch+1}/{num_epochs} finished in {elapsed_time:.2f} s, J={loss_epoch:.6e}")
            print(f"New minimum loss: {min_loss:.6e}")
            if scheduler is not None and current_lr != old_lr:
                print(f"Learning rate updated to {current_lr:.6e}")
        
        # Checkpointing
        if (epoch + 1) % checkpoint_freq == 0 or epoch == num_epochs - 1:
            save_checkpoint(epoch, model, optimizer, scheduler, epoch_losses, epoch_numbers, output_dir)
            
        # Plotting
        if (epoch + 1) % plot_loss_freq == 0 or epoch == num_epochs - 1:
            plot_loss_vs_epochs(epoch_numbers, epoch_losses, output_dir / "loss_vs_epochs.png", min_loss=min_loss)
            
        # Data collection
        pred_global = u_curr.sub(0).dat.data_ro.copy().astype(np.float64)
        target_global = c_target_list[-1].dat.data_ro.copy().astype(np.float64)
        preds_collection.append(pred_global)
        epochs_collection.append(epoch + 1)
        
        if target_final_global is None:
            target_final_global = target_global.copy()
            
        # NPZ Save
        if (epoch + 1) % npz_save_freq == 0 or epoch == num_epochs - 1:
            save_npz_data(output_dir, epoch + 1, preds_collection, epochs_collection,
                          target_final_global, all_epochs_comparison_data,
                          epoch_losses, epoch_numbers, model, device, use_wandb, all_nn_outputs)

    print("Training finished.")
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
