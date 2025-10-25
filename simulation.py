from firedrake import *
import numpy as np
import pyvista as pv

def setup_firedrake():
    mesh = UnitIntervalMesh(100)
    V = FunctionSpace(mesh, "Lagrange", 1)
    W = V * V

    # initial condition
    rng = np.random.default_rng(12)
    u_ic = Function(W, name="Initial_condition")
    num_dofs = u_ic.sub(0).dat.data.shape[0]
    ic = np.zeros((num_dofs, 2))
    ic[:, 0] = [0.5 + 0.2 * sin(pi*i/4) for i in range(num_dofs)]
    ic[:, 1] = 0
    u_ic.sub(0).dat.data[:] = ic[:, 0]
    u_ic.sub(1).dat.data[:] = ic[:, 1]
    return mesh, V, W, u_ic

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

def load_target_data(num_timesteps, V, comm, rank):
    if rank == 0:
        print("Loading target from PVD (pyvista)... (parallel-safe)")
    c_target_list = []

    # create a Function on each rank to query local size
    _local_example = Function(V)
    local_n = _local_example.dat.data.shape[0]
    counts = comm.allgather(local_n)           # list of local sizes per rank
    counts = np.array(counts, dtype=np.int32)
    displs = np.insert(np.cumsum(counts), 0, 0, axis=0)[:-1]

    for i in range(num_timesteps):
        if rank == 0:
            # read global array on rank 0 only
            reader = pv.get_reader(f"/home/rnp/firedrake/ch_learn/ch_fh/ch_fh_{i}.vtu")
            data = reader.read()
            arr_global = data.point_data["Volume Fraction"].astype(np.float64)
            # Sanity check: combined counts should match length of arr_global
            if arr_global.size != counts.sum():
                raise RuntimeError(f"Global DOF mismatch: arr size {arr_global.size} != sum(counts) {counts.sum()}")
        else:
            arr_global = None

        # prepare local recv buffer
        local_arr = np.empty(local_n, dtype=np.float64)

        # Scatterv: sendbuf=(arr_global, counts, displs, MPI.DOUBLE), recvbuf=local_arr
        comm.Scatterv([arr_global, counts, displs, MPI.DOUBLE], local_arr, root=0)

        # assign local part to a Firedrake Function (this is local data, correct shape)
        f = Function(V, name=f"target_{i}")
        f.dat.data[:] = local_arr
        c_target_list.append(f)
    return c_target_list, counts, displs