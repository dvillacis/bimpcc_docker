from cyipopt import Problem
import numpy as np
from scipy.sparse import coo_matrix,diags

class NonlinearProblem:
    def objective(self, x):
        # Objective: f(x) = 0.5 * (x1^2 + x2^2)
        return 0.5 * (x[0]**2 + x[1]**2)
    
    def gradient(self, x):
        # Gradient of the objective: df/dx = [x1, x2]
        return np.array([x[0], x[1]])
    
    def constraints(self, x):
        # Nonlinear constraint: g(x) = x1^2 + x2^2 - 1
        return np.array([x[0]**2 + x[1]**2 - 1])
    
    def jacobian(self, x):
        # Jacobian of the constraint: dg/dx = [2*x1, 2*x2]
        return np.array([2*x[0], 2*x[1]])
    
    def jacobianstructure(self):
        # Structure of the Jacobian: both elements are non-zero
        return (np.array([0, 0]), np.array([0, 1]))
    
    def hessianstructure(self):
        # Structure of the Hessian: lower triangular (including diagonal)
        return (np.array([0,1]), np.array([0,1]))
    
    def hessian(self, x, lagrange, obj_factor):
        # Hessian of the objective: [1, 0; 0, 1]
        hessian_obj = obj_factor*diags([1, 1])
        
        # Hessian of the constraint: lambda * [2, 0; 0, 2]
        hessian_constr = lagrange[0]*diags([2, 2])
        
        # Combine Hessian of objective and constraint
        hessian_values = hessian_obj.data + hessian_constr.data
        
        return hessian_values

# Set up the problem with bounds and constraints
nlp = Problem(
    n=2,  # Number of variables
    m=1,  # Number of constraints
    problem_obj=NonlinearProblem(),
    lb=[-10, -10],  # Lower bounds for variables
    ub=[10, 10],    # Upper bounds for variables
    cl=[0],  # Lower bounds for constraint
    cu=[0]   # Upper bounds for constraint (equality)
)

# Set options to use exact Hessian
nlp.add_option('jacobian_approximation', 'exact')
nlp.add_option('derivative_test', 'second-order')
nlp.add_option('hessian_approximation', 'exact')
nlp.add_option('print_level', 5)

# Initial guess for the solution
x0 = [1.0, 0.0]

# Solve the problem
solution = nlp.solve(x0)

# Print the solution
print("Solution:", solution)