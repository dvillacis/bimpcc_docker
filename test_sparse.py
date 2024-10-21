from cyipopt import Problem
import numpy as np
from scipy.sparse import coo_matrix, bmat

class TestProblem:
    def __init__(self):
        # Define the sparse block matrices for the constraint Jacobian in COO format
        self.A = coo_matrix(([1, 1], ([0, 0], [0, 1])), shape=(2, 2))  # Block for g1
        self.B = coo_matrix(([1, 1], ([1, 1], [0, 1])), shape=(2, 2))  # Block for g2
        self.C = bmat([[self.A,self.B]])  # Full Jacobian matrix
        self.H = coo_matrix((np.array([1.0, 2.0, 3.0, 4.0]),  # Non-zero values
                             (np.array([0, 1, 2, 3]),  # Row indices
                              np.array([0, 1, 2, 3]))),  # Column indices
                             shape=(4, 4))
    
    def objective(self, x):
        # Objective function: f(x) = 0.5 * (x1^2 + 2*x2^2 + 3*x3^2 + 4*x4^2)
        return 0.5 * (x[0]**2 + 2*x[1]**2 + 3*x[2]**2 + 4*x[3]**2)
    
    def gradient(self, x):
        # Gradient of the objective function
        return np.array([x[0], 2*x[1], 3*x[2], 4*x[3]])
    
    def constraints(self, x):
        # Linear constraints: g1(x) = x1 + x2 - 1 and g2(x) = x3 + x4 - 1
        return np.array([x[0] + x[1] - 1, x[2] + x[3] - 1])
    
    def jacobian(self, x):
        # Jacobian of the constraints in COO format
        # values_A = self.A.data  # Values for g1
        # values_B = self.B.data  # Values for g2
        # return np.concatenate([values_A, values_B]
        return self.C.data

    def jacobianstructure(self):
        # Row and column indices for the non-zero elements of the Jacobian
        # row_A, col_A = self.A.row, self.A.col
        # row_B, col_B = self.B.row, self.B.col
        # row_indices = np.concatenate([row_A, row_B])
        # col_indices = np.concatenate([col_A, col_B])
        # return (row_indices, col_indices)
        return self.C.row, self.C.col
    
    def hessianstructure(self):
        # Extract the row and column indices from the sparse lower triangular Hessian matrix (COO format)
        return (self.H.row, self.H.col)
    
    def hessian(self, x, lagrange, obj_factor):
        # Hessian of the objective function (already in sparse format)
        # Only provide the lower triangular part (in this case, just the diagonal)
        hessian_values = self.H.data * obj_factor
        
        return hessian_values

# Set up the problem with bounds and constraints
nlp = Problem(
    n=4,  # Number of variables
    m=2,  # Number of constraints
    problem_obj=TestProblem(),
    lb=[-10, -10, -10, -10],  # Lower bounds for variables
    ub=[10, 10, 10, 10],      # Upper bounds for variables
    cl=[0, 0],  # Lower bounds for constraints (equality)
    cu=[0, 0]   # Upper bounds for constraints (equality)
)

# Set the options to use the exact Jacobian
nlp.add_option('jacobian_approximation', 'exact')
nlp.add_option('derivative_test', 'second-order')
nlp.add_option('hessian_approximation', 'exact')
nlp.add_option('print_level', 5)

# Initial guess for the solution
x0 = [0.5, 0.5, 0.5, 0.5]

# Solve the problem
solution = nlp.solve(x0)

# Print the solution
print("Solution:", solution)