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
        
        # These Functions own the mutable state used by the solver.  Updating
        # their data between steps is much cheaper than rebuilding UFL forms.
        self.u = Function(W, name="Solution")
        self.u_ = Function(W, name="Solution_Old")
        
        # Get sub-functions
        c, mu = split(self.u)
        c_, mu_ = split(self.u_)
        
        # Test functions
        v = TestFunction(W)
        c_test, mu_test = split(v)
        
        # Placeholder for the learned constitutive relation.  The weak form
        # keeps a symbolic reference to this Function, so assigning new values
        # here changes the coefficient without invalidating the compiled form.
        V = W.sub(0)
        self.dfdc_f = Function(V, name="dfdc")
        
        # Semi-implicit CH step:
        #   c - c_old = (dt/2) M Laplacian(mu + mu_old)
        #   mu = learned df/dc - lambda^2 Laplacian(c)
        # The previous mu enters only through self.u_, while learned df/dc is
        # injected through self.dfdc_f.
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
        
        print("CHSolver initialized - forms built once, solver ready for reuse", flush=True)
    
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
        # Update coefficient data in existing Functions so the cached solver
        # sees the new state and learned derivative for this timestep.
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

def load_target_data(data_dir, V, comm=None, rank=None):
    print("Loading target from PVD (pyvista)...", flush=True)
    from scipy.spatial import KDTree
    from pathlib import Path
    
    c_target_list = []
    
    # Prefer VTI files when present because they preserve structured grid
    # dimensions; otherwise fall back to VTU output from Firedrake/VTK.
    vtu_files = sorted(Path(data_dir).glob('*.vtu'))
    vti_files = sorted(Path(data_dir).glob('*.vti'))
    files = vti_files if vti_files else vtu_files
    
    if not files:
        raise ValueError(f"No .vtu or .vti files found in {data_dir}")

    # Read the first file to establish the coordinate mapping.  The mapping is
    # reused for every timestep, so this assumes all frames share one mesh.
    reader = pv.get_reader(str(files[0]))
    mesh_data = reader.read()
    
    if "Volume Fraction" not in mesh_data.point_data and "Volume Fraction" in mesh_data.cell_data:
        mesh_data = mesh_data.cell_data_to_point_data()
        
    vtk_points = mesh_data.points
    
    # Normalize VTK points to [0, 1] to match Firedrake's unit-domain meshes.
    # This lets MD outputs with physical box coordinates be compared against
    # nondimensional CH coordinates.
    vtk_points = (vtk_points - vtk_points.min(axis=0)) / (vtk_points.max(axis=0) - vtk_points.min(axis=0))
    
    # Build KDTree for nearest neighbor search
    print("Building KDTree for mesh mapping...", flush=True)
    tree = KDTree(vtk_points)
    
    # Get Firedrake DOF coordinates
    # Note: V.mesh().coordinates.dat.data_ro matches the .dat.data ordering for CG1
    fd_coords = V.mesh().coordinates.dat.data_ro
    
    # Nearest-neighbor transfer is intentionally simple here: it preserves the
    # observed values at grid/sample points and avoids introducing another
    # interpolation operator into the training loop.
    print("Mapping coordinates...", flush=True)
    _, indices = tree.query(fd_coords)

    print(f"Loading {len(files)} timesteps...", flush=True)
    for f_path in files:
        reader = pv.get_reader(str(f_path))
        data = reader.read()
        
        # Handle cell data conversion seamlessly
        if "Volume Fraction" not in data.point_data and "Volume Fraction" in data.cell_data:
            data = data.cell_data_to_point_data()
            
        arr_global = data.point_data["Volume Fraction"].astype(np.float64)

        f = Function(V, name=f"target_{len(c_target_list)}")
        
        # Assign data using the pre-computed mapping indices
        f.dat.data[:] = arr_global[indices]
        
        c_target_list.append(f)
        
    return c_target_list, None, None
