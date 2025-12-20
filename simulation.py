from firedrake import *
import numpy as np
import pyvista as pv



class CHSolver:
    """
    Reusable Cahn-Hilliard solver that builds forms once and reuses the solver object.
    This eliminates the UFL expression rebuilding overhead.
    """
    
    def __init__(self, W, dt, M, lmbda):
        """
        Initialize the solver with problem parameters.
        
        Args:
            W: Mixed function space (V * V)
            dt: Time step size
            M: Mobility coefficient
            lmbda: Interface width parameter
        """
        self.dt = dt
        self.M = M
        self.lmbda = lmbda
        
        # Create functions once - these will be reused
        self.u = Function(W, name="Solution")
        self.u_ = Function(W, name="Solution_Old")
        
        # Get sub-functions
        c, mu = split(self.u)
        c_, mu_ = split(self.u_)
        
        # Test functions
        v = TestFunction(W)
        c_test, mu_test = split(v)
        
        # Placeholder for dfdc - will be updated each timestep
        V = W.sub(0)
        self.dfdc_f = Function(V, name="dfdc")
        
        # Build form ONCE (not 1000 times per epoch!)
        F0 = (inner(c, c_test) - inner(c_, c_test)) * dx + \
             (dt/2) * M * dot(grad(mu + mu_), grad(c_test)) * dx
        F1 = inner(mu, mu_test) * dx - inner(self.dfdc_f, mu_test) * dx - \
             lmbda**2 * dot(grad(c), grad(mu_test)) * dx
        F = F0 + F1
        
        # Create solver ONCE with direct linear solver
        solver_parameters = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"
        }
        
        problem = NonlinearVariationalProblem(F, self.u)
        self.solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)
        
        print("CHSolver initialized - forms built once, solver ready for reuse")
    
    def solve_step(self, u_old, dfdc_f, u_target):
        """
        Solve one timestep.
        
        Args:
            u_old: Previous solution (Function)
            dfdc_f: Neural network prediction for df/dc (Function)
            u_target: Target solution Function to update
            
        Returns:
            Updated solution (Function)
        """
        # Update data in existing Functions (no form rebuilding!)
        self.u_.assign(u_old)
        self.dfdc_f.assign(dfdc_f)
        
        # Solve (reuses compiled form and solver)
        self.solver.solve()
        
        # Copy result to target
        u_target.assign(self.u)
        
        return u_target
    
    def get_dfdc_function(self):
        """Return the dfdc Function for use with adjoint."""
        return self.dfdc_f


def solve_one_step(u_old, dfdc_f, u, c, mu, c_test, mu_test, dt, M, lmbda):
    """
    Original solve function (kept for backward compatibility).
    Consider using CHSolver class for better performance.
    """
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
    
    # Pre-compute local-to-global index mapping based on coordinates
    # This is necessary for parallel execution where each rank only owns a part of the mesh
    x = SpatialCoordinate(V.mesh())
    # Interpolate x coordinate onto V
    x_fn = Function(V).interpolate(x[0])
    x_local = x_fn.dat.data_ro
    
    # Assuming uniform mesh on [0, 2] with 200 cells (matches problem setup)
    L = 2.0
    N = 200
    dx = L / N
    
    # Map coordinates to indices: index = round(x / dx)
    # We use rint to round to nearest integer
    indices = np.rint(x_local / dx).astype(int)
    
    # Clip indices to ensure they are within bounds (0 to 200 inclusive -> 201 points)
    # The global data has 201 points
    indices = np.clip(indices, 0, 200)

    for i in range(num_timesteps):
        reader = pv.get_reader(f"ch_fh/ch_fh_{i}.vtu")
        data = reader.read()
        arr_global = data.point_data["Volume Fraction"].astype(np.float64)

        f = Function(V, name=f"target_{i}")
        
        # Assign local data using the computed indices
        f.dat.data[:] = arr_global[indices]
        
        c_target_list.append(f)
    return c_target_list, None, None