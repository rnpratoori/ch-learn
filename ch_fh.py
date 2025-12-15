from firedrake import *
from firedrake.petsc import PETSc
import numpy as np

# Add MPI communicator
comm = COMM_WORLD
rank = comm.rank

# Model parameters
lmbda = 5e-2
chi = 1
N1 = 5
N2 = 5

# Simulation parameters
dt = 1e-3
T = 1e-1
N = T/dt

# Create mesh
mesh = IntervalMesh(200, 2)

# Define function space
V = FunctionSpace(mesh, "Lagrange", 1)
W = V*V

# Define functions
u = Function(W, name="Solution")
u_ = Function(W, name="Solution_Old")
c, mu = split(u)
c_, mu_ = split(u_)

v = TestFunction(W)
c_test, mu_test = split(v)

# Initial condition
rng = np.random.default_rng(11)
num_dofs = u.sub(0).dat.data.shape[0]
ic = np.zeros((num_dofs, 2))
ic[:, 0] = [0.5 + 0.2 * sin(pi*i/4) for i in range(num_dofs)]
ic[:, 1] = 0
u_.sub(0).dat.data[:] = ic[:, 0]  # First component
u_.sub(1).dat.data[:] = ic[:, 1]  # Second component
u.assign(u_)

# Define residuals
c = variable(c)
f = c*ln(c)/N1 + (1-c)*ln(1-c)/N2 + chi*c*(1-c)
dfdc = diff(f, c)

# Define mobility
M = 1

F0 = (inner(c, c_test) - inner(c_, c_test)) * dx + (dt/2) * M * dot(grad(mu + mu_), grad(c_test)) * dx
F1 = inner(mu, mu_test) * dx - inner(dfdc, mu_test) * dx - lmbda**2 * dot(grad(c), grad(mu_test)) * dx
F = F0 + F1

# Create nonlinear problem
problem = NonlinearVariationalProblem(F, u)

# Output
t = 0.0
n = 0
outfile = VTKFile("ch_fh.pvd")
outfile.write(project(c_, V, name="Volume Fraction"), time=t)

while (t < T):
    if rank == 0:
        print("Solving for t = ", t, "...")
    solve(F == 0, u, solver_parameters={"ksp_type": "preonly", "pc_type": "lu", "convergence_criteria": "incremental", "pc_factor_mat_solver_type": "mumps"})
    u_.assign(u)
    t += dt
    n += 1
    # VTKFile is parallel-aware, but only rank 0 writes the file header
    outfile.write(project(c_, V, name="Volume Fraction"), time=t)
