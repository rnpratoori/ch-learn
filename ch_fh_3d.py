from firedrake import *
import numpy as np
import random

def main():
    # 3D Unit Cube with 64 elements per side
    mesh = BoxMesh(64, 64, 64, 1.0, 1.0, 1.0)
    
    V = FunctionSpace(mesh, "Lagrange", 1)
    W = V * V

    # Parameters
    dt = 1e-3
    T = 1e-1
    M = 1.0
    lmbda = 5e-2
    num_timesteps = int(T / dt)

    # Flory-Huggins parameters
    N1 = 5.0
    N2 = 5.0
    chi = 1.0

    # Initial condition: small random fluctuations around 0.5
    u_ic = Function(W, name="Initial_condition")
    c_ic, mu_ic = split(u_ic)
    
    np.random.seed(42)
    # Using spatial coordinates to seed based on position could be one way,
    # but here we just fill the vector directly.
    c_vec = u_ic.sub(0).dat.data
    c_vec[:] = 0.5 + 0.01 * (np.random.rand(len(c_vec)) - 0.5)
    u_ic.sub(1).assign(0.0)

    # Solution variables
    u = Function(W, name="Solution")
    u.assign(u_ic)
    c, mu = split(u)
    
    u_ = Function(W)
    u_.assign(u_ic)
    c_, mu_ = split(u_)
    
    v = TestFunction(W)
    c_test, mu_test = split(v)

    # Chemical Potential: df/dc
    # f(c) = c/N1 * ln(c) + (1-c)/N2 * ln(1-c) + chi*c*(1-c)
    # df/dc = 1/N1*ln(c) + 1/N1 - 1/N2*ln(1-c) - 1/N2 + chi*(1-2c)
    # Using Variables to allow automatic differentiation if needed, but symbolic UFL is fine here.
    
    # Safe log to avoid NaN at c=0 or c=1, though dynamics should keep it inside (0, 1)
    # Adding a small epsilon for stability if needed, but standard CH often works fine
    # if dt is small enough and initial condition is safe.
    dfdc = (1.0/N1)*ln(c) + (1.0/N1) - (1.0/N2)*ln(1.0-c) - (1.0/N2) + chi*(1.0-2.0*c)

    # Weak form
    # F0: c conservation
    F0 = ( inner(c, c_test) - inner(c_, c_test) )*dx + (dt/2)*M*dot(grad(mu + mu_), grad(c_test))*dx
    
    # F1: Chemical potential definition
    F1 = inner(mu, mu_test)*dx - inner(dfdc, mu_test)*dx - lmbda**2*dot(grad(c), grad(mu_test))*dx
    
    F = F0 + F1

    # Solver
    problem = NonlinearVariationalProblem(F, u)
    solver_parameters = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_monitor": None
    }
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)

    # Output
    outfile = File("ch_fh_3d/ch_fh_3d.pvd")
    
    # Save initial condition
    u.sub(0).rename("Volume Fraction")
    outfile.write(u.sub(0), time=0.0)

    print(f"Starting simulation on 3D mesh: 64x64x64, {V.dim()} DOFs")

    for i in range(num_timesteps):
        t = (i + 1) * dt
        
        u_.assign(u)
        solver.solve()
        
        if (i+1) % 10 == 0:
            print(f"Timestep {i+1}/{num_timesteps} completed.")
            
        outfile.write(u.sub(0), time=t)

    print("Ground truth generation finished.")

if __name__ == "__main__":
    main()
