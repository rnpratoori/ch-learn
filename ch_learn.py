# ch_learn_adjoint_sketch.py
# Sketch showing how to use firedrake-adjoint to compute sensitivities of the
# PDE-constrained loss w.r.t. the *Firedrake* field dF/dc (dfdc_f), then backpropagate
# those sensitivities into a PyTorch neural network that predicts dF/dc.
#
# IMPORTANT: this is a *sketch* / best-effort integration. You must have
# firedrake-adjoint (pyadjoint) installed and available in the same Python
# environment as Firedrake. APIs can change; treat this as a working template
# you may need to adapt to your exact firedrake-adjoint version.

from firedrake import *
from firedrake_adjoint import *   # provides ReducedFunctional, Control, etc.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import meshio
import pyvista as pv

# ----------------------
# PyTorch model
# ----------------------
class FEDerivative(nn.Module):
    def __init__(self):
        super(FEDerivative, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 20),
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

# Instantiate network and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using PyTorch device: {device}")
dfdc_net = FEDerivative().to(device)
optimizer = optim.Adam(dfdc_net.parameters(), lr=1e-3)

# ----------------------
# Problem setup
# ----------------------
lmbda = 5e-2
dt = 1e-2
T = 1e0
M = 1.0

num_timesteps = int(T / dt)
if num_timesteps > 2000:
    num_timesteps = 2000

mesh = UnitIntervalMesh(100)
V = FunctionSpace(mesh, "Lagrange", 1)
W = V * V

# Solution variable (trial), test functions
u = Function(W, name="Solution")
c, mu = split(u)
v = TestFunction(W)
c_test, mu_test = split(v)

# initial condition
rng = np.random.default_rng(12)
u_ic = Function(W, name="Initial_condition")
num_dofs = u_ic.sub(0).dat.data.shape[0]
ic = np.zeros((num_dofs, 2))
ic[:, 0] = 0.5 + 0.2 * (0.5 - rng.random(num_dofs))
ic[:, 1] = 0
u_ic.sub(0).dat.data[:] = ic[:, 0]
u_ic.sub(1).dat.data[:] = ic[:, 1]

# one-step solver (as before)
def solve_one_step(u_old, dfdc_f):
    u_ = Function(W, name="Solution_Old")
    u_.assign(u_old)
    c_ = u_.sub(0)
    mu_ = u_.sub(1)

    F0 = (inner(c, c_test) - inner(c_, c_test)) * dx + (dt/2) * M * dot(grad(mu + mu_), grad(c_test)) * dx
    F1 = inner(mu, mu_test) * dx - inner(dfdc_f, mu_test) * dx - lmbda**2 * dot(grad(c), grad(mu_test)) * dx
    F = F0 + F1

    # Solve nonlinear/linear system for u (holds global Function `u`)
    solve(F == 0, u, solver_parameters={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    })
    return u

# ----------------------
# Load target (from PVD via pyvista)
# ----------------------
print("Loading target from PVD (pyvista)...")
reader = pv.get_reader("ch_fh.pvd")
data = reader.read()
last_step = data[-1]
values = np.asarray(last_step.point_data["Volume Fraction"]).reshape(-1)

print("Mesh points:", len(last_step.points))
print("Point data arrays:", last_step.point_data.keys())
print("Cell data arrays:", last_step.cell_data.keys())
print("Firedrake DOFs:", V.dim())

c_target = Function(V, name="Target_c")
if values.shape[0] != c_target.dat.data.shape[0]:
    raise RuntimeError("Mismatch between VTU point-data length and Firedrake DOFs")
c_target.dat.data[:] = values
print("Target loaded")

# ----------------------
# Training using firedrake-adjoint
# ----------------------
num_epochs = 200
vtk_out = VTKFile("ch_learn_adjoint.pvd")

for epoch in range(num_epochs):
    start_time = time.time()

    # reset solution
    u_curr = u_ic.copy(deepcopy=True)

    # --- 1. PyTorch Forward Pass ---
    # forward_pass_start = time.time()
    c0 = u_curr.sub(0)
    c_vec0 = c0.dat.data_ro.copy().astype(np.float32)
    with torch.no_grad():
        c_tensor0 = torch.from_numpy(c_vec0.reshape(-1, 1)).to(device)
        dfdc_np0 = dfdc_net(c_tensor0).cpu().numpy().reshape(-1)

    dfdc_f = Function(V, name="dfdc_pred")
    dfdc_f.dat.data[:] = dfdc_np0
    # forward_pass_time = time.time() - forward_pass_start

    # --- 2. Firedrake Simulation ---
    # simulation_start = time.time()
    # Forward integrate using the current dfdc_f (here we keep dfdc fixed in time for simplicity)
    for i in range(num_timesteps):
        u_next = solve_one_step(u_curr, dfdc_f)
        u_curr.assign(u_next)

        # Progress
        # if (i + 1) % 200 == 0 or i == num_timesteps - 1:
        #     print(f" Epoch {epoch+1}/{num_epochs}, timestep {i+1}/{num_timesteps}")
    # simulation_time = time.time() - simulation_start

    # --- 3. Loss Calculation ---
    # loss_calc_start = time.time()
    # Compute PDE-constrained loss (Firedrake scalar)
    c_final = u_curr.sub(0)
    J = assemble(0.5 * (c_final - c_target)**2 * dx)
    # loss_calc_time = time.time() - loss_calc_start

    # --- 4. Adjoint Gradient ---
    # adjoint_grad_start = time.time()
    # Create ReducedFunctional of J w.r.t. the Firedrake control dfdc_f
    # Control expects a Firedrake Function
    rf = ReducedFunctional(J, Control(dfdc_f))

    # Compute gradient of J w.r.t dfdc_f (returns a Firedrake Function)
    dJ_ddfdc = rf.derivative()

    # Extract numpy array from Firedrake Function
    dJ_ddfdc_np = dJ_ddfdc.dat.data_ro.copy()
    # adjoint_grad_time = time.time() - adjoint_grad_start

    # --- 5. PyTorch Backpropagation ---
    # backprop_start = time.time()
    # Recompute dfdc_pred with autograd enabled
    c_vec = c0.dat.data_ro.copy().astype(np.float32)
    c_tensor = torch.from_numpy(c_vec.reshape(-1, 1)).to(device)
    c_tensor.requires_grad = False
    dfdc_tensor = dfdc_net(c_tensor)  # shape (N,1)

    # form scalar loss = inner(dfdc_tensor.flatten(), dJ_ddfdc_np)
    sens_tensor = torch.from_numpy(dJ_ddfdc_np.astype(np.float32)).reshape(-1, 1).to(device)
    scalar_for_backprop = torch.sum(dfdc_tensor * sens_tensor)

    # Backpropagate to get gradients for NN parameters
    optimizer.zero_grad()
    scalar_for_backprop.backward()
    optimizer.step()
    # backprop_time = time.time() - backprop_start

    # --- Logging + write ---
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{num_epochs} finished in {elapsed_time:.2f} s, J={float(J):.6e}")

    # --- Print timings ---
    # print(f"--- Epoch {epoch+1} Timings ---")
    # print(f"  1. PyTorch Forward Pass:  {forward_pass_time:.4f}s")
    # print(f"  2. Firedrake Simulation:  {simulation_time:.4f}s")
    # print(f"  3. Loss Calculation:      {loss_calc_time:.4f}s")
    # print(f"  4. Adjoint Gradient:      {adjoint_grad_time:.4f}s")
    # print(f"  5. PyTorch Backprop:      {backprop_time:.4f}s")
    # print(f"  ---------------------------")
    # total_timed = forward_pass_time + simulation_time + loss_calc_time + adjoint_grad_time + backprop_time
    # print(f"  Total Timed Sections:   {total_timed:.4f}s")
    # print(f"  Total Epoch Time:         {epoch_time:.4f}s")


    # with vtk_out:
    vtk_out.write(project(c_final, V, name="Volume Fraction"), time=epoch+1)

print("Training finished (adjoint update).")

# NOTES:
# - We kept dfdc_f fixed in time and dependent only on the initial c for simplicity. In general you
#   may want dfdc to be evaluated at each timestep from the current c; in that case, you must
#   either treat all time-dependent dfdc evaluations as controls or construct a wrapper that
#   accumulates their influence. That is more complicated but follows the same pattern.
# - The key idea here is the two-step chain-rule:
#     1) use firedrake-adjoint to get dJ / d (dfdc_f) as a vector of sensitivities on DOFs
#     2) form the inner product of those sensitivities with the PyTorch network outputs and
#        backpropagate that scalar through PyTorch to update NN weights.
# - This sketch assumes the mapping between dfdc_f.dat ordering and the PyTorch network outputs
#   is consistent (we used direct assignment). If you evaluate the NN at quadrature points or
#   a reduced set of points, the mapping must be constructed carefully.
# - firedrake-adjoint must be installed; APIs (ReducedFunctional, Control) may differ slightly
#   between versions. If rf.derivative() fails, consult the pyadjoint docs for the correct call.
