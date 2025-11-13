from firedrake import *
import numpy as np
import pyvista as pv
from mpi4py import MPI

def solve_one_step(u_old, dfdc_f, u, c, mu, c_test, mu_test, dt, M, lmbda):
    u_ = Function(u.function_space(), name="Solution_Old")
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

def load_target_data(num_timesteps, V, comm=None, rank=None):
    print("Loading target from PVD (pyvista)...")
    c_target_list = []

    for i in range(num_timesteps):
        reader = pv.get_reader(f"/work/mech-ai/rnp/ch-learn/ch_fh/ch_fh_{i}.vtu")
        data = reader.read()
        arr_global = data.point_data["Volume Fraction"].astype(np.float64)

        f = Function(V, name=f"target_{i}")
        f.dat.data[:] = arr_global
        c_target_list.append(f)
    return c_target_list, None, None