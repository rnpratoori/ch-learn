from firedrake import *
import numpy as np
import time
import sys
import gmsh

# -----------------------------------------------------------------------------
# MPI
# -----------------------------------------------------------------------------
comm = COMM_WORLD
rank = comm.rank
size = comm.size

if rank == 0:
    print("=" * 60, flush=True)
    print("3D Split Cahn–Hilliard | BDF2 | Firedrake", flush=True)
    print(f"MPI size = {size}", flush=True)
    print("=" * 60, flush=True)

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
lmbda = Constant(5e-2)
chi   = Constant(1.0)
N1    = Constant(5.0)
N2    = Constant(5.0)
Mmob  = Constant(1.0)

dt = Constant(1e-8)
T  = 1e-1
num_steps = int(10)

# -----------------------------------------------------------------------------
# Mesh and function spaces
# -----------------------------------------------------------------------------
with CheckpointFile('check_128.h5', 'r') as f:
    mesh = f.load_mesh()

V = FunctionSpace(mesh, "CG", 1)
W = V * V

if rank == 0:
    print(f"Cells: {mesh.num_cells()}, DOFs: {W.dim()}", flush=True)

# -----------------------------------------------------------------------------
# Unknowns and history
# -----------------------------------------------------------------------------
u   = Function(W, name="u")      # (c^{n+1}, mu^{n+1})
u_n = Function(W, name="u_n")    # (c^n, mu^n)
u_nm1 = Function(W, name="u_nm1")  # (c^{n-1}, mu^{n-1})

c,  mu  = split(u)
c_n, mu_n = split(u_n)
c_nm1, mu_nm1 = split(u_nm1)

v = TestFunction(W)
c_test, mu_test = split(v)

# -----------------------------------------------------------------------------
# MPI-safe random initialization
# -----------------------------------------------------------------------------
np.random.seed(12345 + rank)

c0 = Function(V)
local_size = c0.dat.data_ro.shape[0]
c0.dat.data[:] = 0.5 + 0.05 * (np.random.rand(local_size) - 0.5)

u_n.sub(0).assign(c0)
u_n.sub(1).assign(0.0)

u_nm1.assign(u_n)   # for BE startup
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
# BDF2 time derivative for c
F_c = inner((3*c - 4*c_n + c_nm1), c_test) * dx + (2*dt) * Mmob * dot(grad(mu), grad(c_test)) * dx

# Chemical potential definition
F_mu = inner(mu, mu_test) * dx - inner(dfdc, mu_test) * dx - lmbda**2 * dot(grad(c), grad(mu_test)) * dx

F = F_c + F_mu

# -----------------------------------------------------------------------------
# Jacobian
# -----------------------------------------------------------------------------
J = derivative(F, u)

# -----------------------------------------------------------------------------
# Nullspace Definition (The Fix)
# -----------------------------------------------------------------------------
# We need to tell the solver that the chemical potential (index 1 of W)
# has a constant nullspace (it "floats").

# 1. Create a vector in the mixed space W
null_vec = Function(W)

# 2. Set the 'c' component to 0 and 'mu' component to 1
#    (We are saying: "adding 1.0 to mu everywhere changes nothing")
null_vec.sub(1).assign(1.0)

# 3. Create an orthonormal basis from this vector
nullspace = VectorSpaceBasis([null_vec])
nullspace.orthonormalize()

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

    # --- THE CRITICAL FIX ---
    "pc_type": "asm",
    "pc_asm_overlap": 2,
    "sub_ksp_type": "preonly",
    "sub_pc_type": "ilu",
    "sub_pc_factor_levels": 1,
    
    # ADD THESE TWO LINES:
    "sub_pc_factor_shift_type": "nonzero",  # Prevents zero pivot crashes
    "sub_pc_factor_shift_amount": 1e-10,    # The size of the safety shift
}

solver = NonlinearVariationalSolver(
    problem,
    solver_parameters=solver_parameters,
    nullspace=nullspace
)

# -----------------------------------------------------------------------------
# Time loop
# -----------------------------------------------------------------------------
t = 0.0

if rank == 0:
    print("=" * 60, flush=True)
    print("Starting time integration (BDF2)", flush=True)
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
    # VTKFile is parallel-aware, but only rank 0 writes the file header
    outfile.write(project(c_, V, name="Volume Fraction"), time=t)
