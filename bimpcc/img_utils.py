import numpy as np
from scipy.sparse import coo_matrix, bmat, diags
from numpy.linalg import svd

def gradient(x):
    """
    Compute the gradient of the image x using forward differences.
    Boundary conditions: Neumann (zero gradient at the boundary).
    
    Parameters:
    - x: 2D numpy array.
    
    Returns:
    - grad: 3D numpy array with shape (height, width, 2).
    """
    grad = np.zeros((x.shape[0], x.shape[1], 2), dtype=x.dtype)
    # Compute forward differences for x-gradient
    grad[:-1, :, 0] = x[1:, :] - x[:-1, :]
    grad[-1, :, 0] = 0  # Neumann boundary condition
    
    # Compute forward differences for y-gradient
    grad[:, :-1, 1] = x[:, 1:] - x[:, :-1]
    grad[:, -1, 1] = 0  # Neumann boundary condition
    
    return grad

def sparse_gradient(height, width):
    """
    Construct a sparse matrix that represents the gradient operator
    with Neumann boundary conditions (zero gradient at the boundary).
    
    Parameters:
    - height: The number of rows (height of the image).
    - width: The number of columns (width of the image).
    
    Returns:
    - Gx: Sparse matrix for x-gradient (horizontal gradient).
    - Gy: Sparse matrix for y-gradient (vertical gradient).
    """
    n = height * width  # Total number of pixels
    
    # x-gradient: Forward differences along the x-direction (rows)
    # Create diagonal matrices for forward differences
    diag_main = np.ones(n)    # Main diagonal
    diag_upper = -np.ones(n)  # Upper diagonal for forward differences
    
    # Adjust for boundary conditions (Neumann: zero gradient at the boundary)
    diag_upper[width-1::width] = 0  # No forward difference across row boundaries

    Gx = diags([diag_main, diag_upper], [0, 1], shape=(n, n), format="coo")
    
    # y-gradient: Forward differences along the y-direction (columns)
    diag_main_y = np.ones(n)
    diag_upper_y = -np.ones(n - width)  # Skip last row in y direction

    Gy = diags([diag_main_y, diag_upper_y], [0, width], shape=(n, n), format="coo")
    
    return Gx, Gy

def divergence(p):
    """
    Compute the divergence of p using backward differences.
    Boundary conditions: Neumann (zero gradient at the boundary).
    
    Parameters:
    - p: 3D numpy array with shape (height, width, 2).
    
    Returns:
    - div: 2D numpy array with shape (height, width).
    """
    div = np.zeros(p.shape[:2], dtype=p.dtype)
    # Compute backward differences for x-component
    div[1:, :] += p[:-1, :, 0]
    div[:-1, :] -= p[:-1, :, 0]
    
    # Compute backward differences for y-component
    div[:, 1:] += p[:, :-1, 1]
    div[:, :-1] -= p[:, :-1, 1]
    
    return div

def compute_rof_gap(u,p,nabla_u, div_p,l,image):
    """
    Compute the primal-dual gap for ROF denoising.
    
    Parameters:
    - nabla_u: Gradient of the denoised image.
    - div_p: Divergence of the dual variable.
    - image: Noisy image.
    
    Returns:
    - gap: Primal-dual gap.
    """
    nu = np.sqrt(nabla_u[:, :, 0]**2 + nabla_u[:, :, 1]**2)
    primal = l*np.sum(nu.ravel()) + np.linalg.norm((u - image).ravel())**2
    dual = -0.5*np.linalg.norm(div_p.ravel())**2 + (image.ravel())@(div_p.ravel())
    
    return primal-dual

def sparse_divergence(height, width):
    """
    Construct a sparse matrix that represents the divergence operator
    with Neumann boundary conditions.
    
    Parameters:
    - height: The number of rows (height of the image).
    - width: The number of columns (width of the image).
    
    Returns:
    - Dx: Sparse matrix for divergence of the x-component of the field.
    - Dy: Sparse matrix for divergence of the y-component of the field.
    """
    n = height * width
    
    # x-component: Backward differences along the x-direction (rows)
    diag_main_x = -np.ones(n)
    diag_lower_x = np.ones(n)
    
    # Adjust for boundary conditions (Neumann: zero gradient at the boundary)
    diag_main_x[width-1::width] = 0  # No backward difference across row boundaries

    Dx = diags([diag_main_x, diag_lower_x[:-1]], [0, -1], shape=(n, n), format="coo")
    
    # y-component: Backward differences along the y-direction (columns)
    diag_main_y = -np.ones(n)
    diag_lower_y = np.ones(n - width)

    Dy = diags([diag_main_y[:n], diag_lower_y], [0, -width], shape=(n, n), format="coo")
    
    return Dx, Dy

def chambolle_pock_rof_denoising(image, tau=0.01, sigma=0.125, theta=1.0, max_iter=10000, lambda_tv=1.0, verbose=True):
    """
    Perform total variation denoising using the Chambolle-Pock algorithm.

    Parameters:
    - image: Input noisy image (2D numpy array).
    - tau: Step size for the primal variable.
    - sigma: Step size for the dual variable.
    - theta: Over-relaxation parameter.
    - max_iter: Maximum number of iterations.
    - lambda_tv: Regularization parameter for TV.

    Returns:
    - u: Denoised image.
    - p: Dual variable.
    """

    # Initialize variables
    u = np.copy(image)  # Primal variable (denoised image)
    p = np.zeros((image.shape[0], image.shape[1], 2), dtype=image.dtype)  # Dual variable (for the gradient)
    u_bar = np.copy(u)  # Intermediate variable
    # gap = np.zeros(max_iter)  # Primal-dual gap


    for i in range(max_iter):
        # Update the dual variable (gradient ascent)
        grad_u_bar = gradient(u_bar)
        p_old = p.copy()
        p += sigma * grad_u_bar
        # Compute the norm and apply projection
        norm_p = np.maximum(1.0, np.sqrt(p[:, :, 0]**2 + p[:, :, 1]**2)/lambda_tv)
        p[:, :, 0] /= norm_p
        p[:, :, 1] /= norm_p

        # Update the primal variable (gradient descent)
        u_bar = u.copy()
        u = u - tau * divergence(p)
        u = (u+tau*image)/(1+tau)

        # Over-relaxation step
        u_bar = u + theta * (u - u_bar)

        # Compute the primal-dual gap
        # gap[i] = compute_rof_gap(u, p, gradient(u), divergence(p), lambda_tv, image)

        # termination criterion
        primal_residual = np.linalg.norm((u - u_bar).ravel())
        dual_residual = np.linalg.norm((p - p_old).ravel())
        # s = np.linalg.norm(u_bar - image + divergence(p))
        # print(f's: {s}')
        if primal_residual < 1e-4 and dual_residual < 1e-2:
            print(f"Converged after {i+1} iterations.")
            break
        
        if verbose and (i+1) % 100 == 0:
            print(f"Iteration {i+1}/{max_iter}: Primal residual = {primal_residual} | Dual residual = {dual_residual}")

    return u, p, gradient(u)

def compute_condition_number(A):
    _,s,_ = svd(A)

    # # Compute the largest and smallest singular values
    # u, s, vt = svds(hess, k=1, which='LM')  # Largest singular value (LM)
    # _, s_smallest, _ = svds(hess, k=1, which='SM')  # Smallest singular value (SM)

    # Condition number is the ratio of the largest to smallest singular value
    condition_number = max(s) / min(s)
    return condition_number

# Example usage
if __name__ == "__main__":
    from Dataset import get_dataset
    import matplotlib.pyplot as plt

    # Load a sample image
    dataset_name = "cameraman"
    size = 256
    utrue, unoisy = get_dataset(dataset_name, size, '../datasets').get_training_data()

    # Denoise the image
    tau = 0.01
    L = np.sqrt(8)
    sigma = 1 / (tau * L**2)
    denoised_image, dual_variable = chambolle_pock_rof_denoising(unoisy,tau,sigma,lambda_tv=0.5)

    # Display the results
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(utrue, cmap='gray')
    axs[0].set_title('True Image')
    axs[0].axis('off')
    axs[1].imshow(unoisy, cmap='gray')
    axs[1].set_title('Noisy Image')
    axs[1].axis('off')
    axs[2].imshow(denoised_image, cmap='gray')
    axs[2].set_title('Denoised Image')
    axs[2].axis('off')
    plt.savefig('../denoising_result.png')

    print("Denoising complete!")