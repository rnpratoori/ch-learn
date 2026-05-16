from firedrake import *
import numpy as np
import time
import sys
import gmsh

# -----------------------------------------------------------------------------
# MPI
# -----------------------------------------------------------------------------
# Firedrake runs this script under MPI even for single-process jobs. Keep the
# rank and communicator size handy so progress messages are emitted once while
# Firedrake handles distributed mesh/vector data.
comm = COMM_WORLD
rank = comm.rank
size = comm.size

if rank == 0:
    print("=" * 60, flush=True)
    print("3D Split Cahn-Hilliard | BDF2 | Firedrake", flush=True)
    print(f"MPI size = {size}", flush=True)
    print("=" * 60, flush=True)

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
# Flory-Huggins free energy density:
#   f(c) = c log(c)/N1 + (1-c) log(1-c)/N2 + chi c(1-c)
# with lambda setting the diffuse-interface penalty.
lmbda = Constant(5e-2)
chi   = Constant(1.0)
N1    = Constant(5.0)
N2    = Constant(5.0)
Mmob  = Constant(1.0)

dt = Constant(5e-6)
T  = 1e-1
num_steps = int(T/float(dt))

# -----------------------------------------------------------------------------
# Mesh and function spaces
# -----------------------------------------------------------------------------
# The fine checkpoint supplies the initial condition; it is projected below onto
# the working mesh so the solve can run at the desired resolution.
with CheckpointFile('check_128.h5', 'r') as f_fine:
    mesh_fine = f_fine.load_mesh()

V_fine = FunctionSpace(mesh_fine, "CG", 1)
W_fine = V_fine * V_fine

u_fine = Function(W_fine)
c_fine,  mu_fine  = split(u_fine)

np.random.seed(12345 + rank)

c0_fine = Function(V_fine)
local_size = c0_fine.dat.data_ro.shape[0]
c0_fine.dat.data[:] = 0.5 + 0.05 * (np.random.rand(local_size) - 0.5)

u_fine.sub(0).assign(c0_fine)
u_fine.sub(1).assign(0.0)

with CheckpointFile('check_128.h5', 'r') as f:
    mesh = f.load_mesh()

# Use a mixed CG1 space so concentration c and chemical potential mu are solved
# together as one nonlinear system.
V = FunctionSpace(mesh, "CG", 1)
W = V * V

if rank == 0:
    print(f"Cells: {mesh.num_cells()}, DOFs: {W.dim()}", flush=True)

# -----------------------------------------------------------------------------
# Unknowns and history
# -----------------------------------------------------------------------------
u   = Function(W, name="u")      # (c^{n+1}, mu^{n+1})
u_n = Function(W, name="u_n")    # (c^n, mu^n)

c,  mu  = split(u)
c_n, mu_n = split(u_n)

v = TestFunction(W)
c_test, mu_test = split(v)

# -----------------------------------------------------------------------------
# Coarse initialization
# -----------------------------------------------------------------------------
# Project the fine mixed state into the working mixed space and use it as both
# the current and previous timestep state.
u_n.project(u_fine)

u.assign(u_n)

if rank == 0:
    cmin = c0.dat.data.min()
    cmax = c0.dat.data.max()
    print(f"Initial c range (local): [{cmin:.4f}, {cmax:.4f}]", flush=True)

# -----------------------------------------------------------------------------
# Free energy and chemical potential
# -----------------------------------------------------------------------------
c_var = variable(c)
f = c_var*ln(c_var)/N1 + (1 - c_var)*ln(1 - c_var)/N2 + chi*c_var*(1 - c_var)
dfdc = diff(f, c_var)

# -----------------------------------------------------------------------------
# Residuals
# -----------------------------------------------------------------------------
# BDF2/CN-style CH residual blocks:
# F_c is the conservative mass balance using midpoint mobility in mu.
# F_mu enforces mu = df/dc - lambda^2 Laplacian(c) in weak form.
F_c = (inner(c, c_test) - inner(c_n, c_test)) * dx + (dt/2) * Mmob * dot(grad(mu + mu_n), grad(c_test)) * dx
F_mu = inner(mu, mu_test) * dx - inner(dfdc, mu_test) * dx - lmbda**2 * dot(grad(c), grad(mu_test)) * dx

F = F_c + F_mu

# -----------------------------------------------------------------------------
# Jacobian
# -----------------------------------------------------------------------------
# Keep the explicit Jacobian available for solver experimentation, even though
# the current problem construction lets Firedrake derive it automatically.
J = derivative(F, u)

problem = NonlinearVariationalProblem(F, u)

# -----------------------------------------------------------------------------
# Solver parameters (Robust Monolithic)
# -----------------------------------------------------------------------------
solver_parameters = {
    # ... (Keep your snes parameters) ...
    "snes_type": "newtonls",
    "snes_linesearch_type": "basic",
    "snes_rtol": 1e-6,
    "snes_atol": 1e-10,
    "snes_max_it": 20,
    "snes_monitor": None,
    "snes_converged_reason": None,

    # ... (Keep KSP parameters) ...
    "ksp_type": "gmres",
    "ksp_rtol": 1e-6,
    "ksp_converged_reason": None,

    # Additive Schwarz with local ILU is more scalable than a global direct
    # solve for this 3D split system.
    "pc_type": "asm",
    "pc_asm_overlap": 1,
    "sub_ksp_type": "preonly",
    "sub_pc_type": "ilu",
    "sub_pc_factor_levels": 1,

    # Shift near-zero pivots to avoid local ILU factorization failures.
    "sub_pc_factor_shift_type": "nonzero",
    "sub_pc_factor_shift_amount": 1e-10,
}

solver = NonlinearVariationalSolver(
    problem,
    solver_parameters=solver_parameters
)

# -----------------------------------------------------------------------------
# Time loop
# -----------------------------------------------------------------------------
with CheckpointFile("simulation_128.h5", 'w') as chk:
    # Save the mesh once so later function checkpoints can be loaded with the
    # same topology and coordinate field.
    chk.save_mesh(mesh)
    chk.save_function(u, name="solution", idx=0)

    t = 0.0
    timestep_index = 1

    if rank == 0:
        print("=" * 60, flush=True)
        print("Starting time integration (CN)", flush=True)
        print("=" * 60, flush=True)

# --- First step: Backward Euler startup ---
if rank == 0:
    print("Startup step (Backward Euler)", flush=True)

solver.solve()
u_nm1.assign(u_n)
u_n.assign(u)
t += float(dt)

# --- Main BDF2 loop ---
for n in range(1, num_steps):
    if rank == 0:
        print("Solving for t = ", t, "...", flush=True)
    solve(F == 0, u, solver_parameters={"ksp_type": "preonly", "pc_type": "lu", "convergence_criteria": "incremental", "pc_factor_mat_solver_type": "mumps"})
    u_.assign(u)
    t += dt
    n += 1
    # VTKFile is parallel-aware, but only rank 0 writes the file header.
    outfile.write(project(c_, V, name="Volume Fraction"), time=t)
