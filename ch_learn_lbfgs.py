"""
Cahn-Hilliard Learning Script
Learns the free energy derivative using neural networks and adjoint methods.
"""

from firedrake import *
from firedrake.adjoint import *
import os

# Set threading limits to avoid conflicts with MPI
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

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
from plotting import plot_loss_vs_epochs
from simulation import solve_one_step, load_target_data, CHSolver
from checkpoint import save_checkpoint, load_checkpoint
from pathlib import Path

# Limit PyTorch threads
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# ----------------------
# PyTorch Model
# ----------------------
class FEDerivative(nn.Module):
    """Neural network to approximate the free energy derivative df/dc."""
    
    def __init__(self, hidden_size=50):
        super(FEDerivative, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, c):
        return self.mlp(c)


# ----------------------
# Setup Functions
# ----------------------
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Cahn-Hilliard learning script.')
    parser.add_argument('--epochs', type=int, default=5000, 
                        help='Number of training epochs.')
    parser.add_argument('--learning-rate', type=float, default=1.0, 
                        help='Learning rate for optimizer.')
    parser.add_argument('--seed', type=int, default=12, 
                        help='Random seed for reproducibility.')
    parser.add_argument('--no-resume', action='store_true', 
                        help='Start training from scratch, ignoring checkpoints.')
    parser.add_argument('--no-wandb', action='store_true', 
                        help='Disable Weights & Biases logging.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results.')
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling mode (reduces epochs to 2).')
    parser.add_argument('--cpu', action='store_true',
                        help='Force usage of CPU for PyTorch even if CUDA is available.')
    parser.add_argument('--max-iter', type=int, default=20,
                        help='Maximum number of LBFGS iterations per optimization step.')
    return parser.parse_args()


def setup_device(args):
    """Setup PyTorch device (CUDA or CPU)."""
    if not args.cpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using PyTorch device: {device}")
    return device


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


def initialize_training(args, device, output_dir):
    """Initialize model, optimizer, and wandb."""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_default_dtype(torch.float64)
    
    # Create model and optimizer
    model = FEDerivative().to(device)
    model.double()
    optimizer = optim.LBFGS(model.parameters(), lr=args.learning_rate, max_iter=args.max_iter, history_size=100, line_search_fn='strong_wolfe')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)
    
    # Initialize wandb
    if not args.no_wandb:
        config = {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "seed": args.seed,
        }
        wandb.init(project="ch_learn", config=config)
    
    # Load checkpoint if available
    start_epoch = 0
    epoch_losses = []
    epoch_numbers = []
    
    if not args.no_resume:
        start_epoch, epoch_losses, epoch_numbers = load_checkpoint(
            model, optimizer, device, output_dir
        )
        # Re-apply LBFGS hyperparameters after loading checkpoint
        for param_group in optimizer.param_groups:
            param_group['max_iter'] = args.max_iter
            param_group['history_size'] = 100
            param_group['line_search_fn'] = 'strong_wolfe'
            param_group['max_eval'] = int(args.max_iter * 1.25)
            param_group['tolerance_grad'] = 1e-07
            param_group['tolerance_change'] = 1e-09
    else:
        print("Starting training from scratch as requested by --no-resume flag.")
    
    return model, optimizer, scheduler, start_epoch, epoch_losses, epoch_numbers


# ----------------------
# Training Loop
# ----------------------
def train_epoch(epoch, num_epochs, model, optimizer, device, u_ic, u, c, mu, 
                c_test, mu_test, c_target_list, V, W, dt, M, lmbda, num_timesteps,
                vtk_out, ch_solver=None, use_wandb=True, step_optimizer=True):
    """Execute one training epoch."""
    # Clear previous tape
    get_working_tape().clear_tape()
    optimizer.zero_grad()
    
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
        c_snapshot = Function(V, name=f"c_snapshot_{i}")
        c_snapshot.assign(c_curr)
        c_inputs.append(c_snapshot)
        
        # Neural network prediction
        c_vec = c_curr.dat.data_ro.copy().astype(np.float64)
        with torch.no_grad():
            c_tensor = torch.from_numpy(c_vec.reshape(-1, 1)).to(device)
            dfdc_np = model(c_tensor).cpu().numpy().reshape(-1)
        
        # Create dfdc Function for adjoint tracking (still needed for Control)
        dfdc_f = Function(V, name=f"dfdc_pred_{i}")
        dfdc_f.dat.data[:] = dfdc_np
        dfdc_outputs.append(dfdc_f)
        
        # Solve one timestep using CHSolver (reuses compiled form!)
        if ch_solver is not None:
            u_next = ch_solver.solve_step(u_curr, dfdc_f, u)
        else:
            # Fallback to old method
            u_next = solve_one_step(u_curr, dfdc_f, u, c, mu, c_test, mu_test, dt, M, lmbda)
        u_curr.assign(u_next)
        
        # Store comparison data
        if (i + 1) % 200 == 0 or (i + 1) == num_timesteps:
            comparison_data.append((i, u_curr.sub(0).copy(deepcopy=True), c_target_list[i]))
        
        # --- LOSS CALCULATION (FFT) ---
        u_curr_np = u_curr.sub(0).dat.data_ro
        target_np = c_target_list[i].dat.data_ro
        u_tensor = torch.tensor(u_curr_np, device=device, requires_grad=True)
        t_tensor = torch.tensor(target_np, device=device)
        
        fft_u = torch.fft.fft(u_tensor)
        fft_t = torch.fft.fft(t_tensor)
        loss_i = 0.5 * torch.mean(torch.abs(fft_u - fft_t)**2)
        
        weight = 5.0 if i <= 40 else 1.0
        (weight * loss_i).backward()
        grad_u_tensor = u_tensor.grad
        
        g_i = Function(V)
        g_i.dat.data[:] = grad_u_tensor.cpu().numpy()
        
        J_adj += assemble(inner(g_i, u_curr.sub(0)) * dx)
        J_total_log += weight * loss_i.item()
        
        # Write VTK output on final epoch
        if epoch == num_epochs - 1 and step_optimizer:
            t = (i + 1) * dt
            vtk_out.write(project(u_curr.sub(0), V, name="Volume Fraction"), time=t)
    
    pause_annotation()
    
    # --- ADJOINT GRADIENT ---
    controls = [Control(d) for d in dfdc_outputs]
    rf = ReducedFunctional(J_adj, controls)
    dJ_dcs = rf.derivative()
    
    # --- PYTORCH BACKPROPAGATION ---
    # optimizer.zero_grad() - Moved to start of function
    
    for i in range(num_timesteps):
        c_i_vec = c_inputs[i].dat.data_ro.copy().astype(np.float64)
        c_i_tensor = torch.from_numpy(c_i_vec.reshape(-1, 1)).to(device)
        dfdc_i_tensor = model(c_i_tensor)
        
        dJ_dcs_i_np = dJ_dcs[i].dat.data_ro.copy()
        sens_i_tensor = torch.from_numpy(dJ_dcs_i_np.astype(np.float64)).reshape(-1, 1).to(device)
        
        scalar_for_backprop = torch.sum(dfdc_i_tensor * sens_i_tensor)
        scalar_for_backprop.backward()
    
    if step_optimizer:
        optimizer.step()
    
    # --- Post-processing ---
    processed_comparison_data = []
    for i, u_pred, c_targ in comparison_data:
        processed_comparison_data.append(
            (i, u_pred.dat.data_ro.copy(), c_targ.dat.data_ro.copy())
        )
    
    elapsed_time = time.perf_counter() - epoch_t0
    loss_epoch = J_total_log
    
    return loss_epoch, elapsed_time, u_curr, processed_comparison_data


def train(args, model, optimizer, scheduler, start_epoch, epoch_losses, epoch_numbers,
          V, W, u_ic, u, c, mu, c_test, mu_test, c_target_list, device, output_dir):
    """Main training loop."""
    # Problem parameters
    lmbda = 5e-2
    dt = 1e-3
    T = 1e-1
    M = 1.0
    num_timesteps = int(T / dt)
    num_epochs = args.epochs
    
    # Frequency parameters
    checkpoint_freq = max(1, num_epochs // 20)
    plot_loss_freq = max(1, num_epochs // 20)
    npz_save_freq = max(1, num_epochs // 10)
    
    # VTK output
    vtk_out = VTKFile(output_dir / "ch_learn_adjoint.pvd")
    
    # Create CHSolver once for reuse across all epochs
    from simulation import CHSolver
    ch_solver = CHSolver(W, dt, M, lmbda)
    
    # Training state
    preds_collection = []
    epochs_collection = []
    target_final_global = None
    all_epochs_comparison_data = []
    min_loss = float('inf')

    npz_path = output_dir / "post_processing_data.npz"
    if start_epoch > 0 and npz_path.exists():
        print(f"Resuming from checkpoint, loading existing .npz data from {npz_path}")
        with np.load(npz_path, allow_pickle=True) as data:
            preds_collection = list(data.get('preds_collection', []))
            epochs_collection = list(data.get('epochs_collection', []))
            target_final_global = data.get('target_final_global')
            all_epochs_comparison_data = list(data.get('all_epochs_comparison_data', []))
    
    use_wandb = not args.no_wandb
    
    for epoch in range(start_epoch, num_epochs):
        if isinstance(optimizer, torch.optim.LBFGS):

            
            # Retrieve metrics from LAST evaluation in the step?
            # LBFGS doesn't return the metrics, so we just run one forward pass for logging?
            # Or we can capture them from the closure using nonlocals.
            # However, for simplicity and to avoid side-effect issues, let's just re-evaluate 
            # or (better) accept that 'loss_epoch' is the one returned by the final closure call is inaccessible cleanly without nonlocal.
            # Let's use the nonlocal approach.
            
            loss_captured = None
            elapsed_captured = None
            u_curr_captured = None
            processed_data_captured = None

            def closure_captured():
                nonlocal loss_captured, elapsed_captured, u_curr_captured, processed_data_captured
                optimizer.zero_grad()
                l, e, u_c, p_d = train_epoch(
                    epoch, num_epochs, model, optimizer, device, u_ic, u, c, mu,
                    c_test, mu_test, c_target_list, V, W, dt, M, lmbda, num_timesteps,
                    vtk_out, ch_solver, use_wandb, step_optimizer=False
                )
                loss_captured = l
                elapsed_captured = e
                u_curr_captured = u_c
                processed_data_captured = p_d
                return l

            optimizer.step(closure_captured)
            loss_epoch, elapsed_time, u_curr, processed_comparison_data = loss_captured, elapsed_captured, u_curr_captured, processed_data_captured

        else:
            loss_epoch, elapsed_time, u_curr, processed_comparison_data = train_epoch(
                epoch, num_epochs, model, optimizer, device, u_ic, u, c, mu,
                c_test, mu_test, c_target_list, V, W, dt, M, lmbda, num_timesteps,
                vtk_out, ch_solver, use_wandb, step_optimizer=True
            )
        
        # Update scheduler
        scheduler.step(loss_epoch)
        
        # Store losses
        epoch_losses.append(loss_epoch)
        epoch_numbers.append(epoch + 1)
        
        # Store comparison data
        all_epochs_comparison_data.append({'epoch': epoch, 'data': processed_comparison_data})
        
        # Logging
        print(f"Epoch {epoch+1}/{num_epochs} finished in {elapsed_time:.2f} s, J={loss_epoch:.6e}")
        if use_wandb:
            wandb.log({"loss": loss_epoch, "epoch": epoch})
        
        # Track minimum loss
        if loss_epoch < min_loss:
            min_loss = loss_epoch
            print(f"New minimum loss: {min_loss:.6e}")
        
        # --- CHECKPOINTING ---
        if (epoch + 1) % checkpoint_freq == 0 or epoch == num_epochs - 1:
            save_checkpoint(epoch, model, optimizer, epoch_losses, epoch_numbers, output_dir)
        
        # --- PLOT LOSS ---
        if (epoch + 1) % plot_loss_freq == 0 or epoch == num_epochs - 1:
            plot_loss_vs_epochs(epoch_numbers, epoch_losses, output_dir / "loss_vs_epochs.png")
        
        # --- Store predictions ---
        pred_global = u_curr.sub(0).dat.data_ro.copy().astype(np.float64)
        target_global = c_target_list[-1].dat.data_ro.copy().astype(np.float64)
        
        preds_collection.append(pred_global.copy())
        epochs_collection.append(epoch + 1)
        if target_final_global is None:
            target_final_global = target_global.copy()
        
        # --- NPZ Checkpointing ---
        if (epoch + 1) % npz_save_freq == 0 or epoch == num_epochs - 1:
            print(f"Saving .npz data at epoch {epoch + 1}...")
            c_values_nn = np.linspace(0, 1, 200).reshape(-1, 1)
            c_tensor_nn = torch.from_numpy(c_values_nn).to(device)
            with torch.no_grad():
                nn_output_values = model(c_tensor_nn).cpu().numpy()
            
            np.savez(output_dir / "post_processing_data.npz",
                     preds_collection=np.array(preds_collection),
                     epochs_collection=np.array(epochs_collection),
                     target_final_global=target_final_global,
                     all_epochs_comparison_data=np.array(all_epochs_comparison_data, dtype=object),
                     c_values_nn=c_values_nn,
                     nn_output_values=nn_output_values,
                     epoch_losses=np.array(epoch_losses),
                     epoch_numbers=np.array(epoch_numbers),
                     nn_output_label=np.array("df/dc"))
            
            if use_wandb:
                wandb.save(str(output_dir / "post_processing_data.npz"))
    
    print("Training finished.")
    if use_wandb:
        wandb.finish()


# ----------------------
# Main Function
# ----------------------
def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Override epochs if profiling
    if args.profile:
        print("Profiling mode enabled: reducing epochs to 2")
        args.epochs = 2
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(os.getenv("OUTPUT_DIR", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = setup_device(args)
    
    # Problem parameters
    dt = 1e-3
    T = 1e-1
    num_timesteps = int(T / dt)
    
    # Setup problem
    V, W, u_ic, u, c, mu, c_test, mu_test, c_target_list = setup_problem(num_timesteps)
    
    # Initialize training
    model, optimizer, scheduler, start_epoch, epoch_losses, epoch_numbers = initialize_training(
        args, device, output_dir
    )
    
    # Train
    train(args, model, optimizer, scheduler, start_epoch, epoch_losses, epoch_numbers,
          V, W, u_ic, u, c, mu, c_test, mu_test, c_target_list, device, output_dir)


if __name__ == "__main__":
    main()
