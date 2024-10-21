import numpy as np
from scipy.sparse import spdiags, kron, diags



def generate_2D_gradient_matrices(N) -> tuple:
    '''
    Generate the gradient matrices for a 2D image

    Parameters:
    N: int
        Number of pixels in each dimension

    Returns:
    Kx: np.ndarray
        Gradient matrix in the x-direction
    Ky: np.ndarray
        Gradient matrix in the y-direction
    '''
    Kx_temp = spdiags([-np.ones(N), np.ones(N)], [0, 1], N-1, N, format='csr')
    Kx = kron(spdiags(np.ones(N), [0], N, N), Kx_temp, format='csr')
    Ky_temp = spdiags([-np.ones(N*(N-1))], [0], N*(N-1), N**2, format='csr')
    Ky = Ky_temp + spdiags([np.ones(N*(N-1)+N)], [N],
                           N*(N-1), N**2, format='csr')

    # CREAR K NO SPARSE
    # Convertir matrices dispersas a arrays densos
    Kx = Kx.toarray()  # Convertir Kx a array denso
    Ky = Ky.toarray()  # Convertir Ky a array denso

    # create K matrix
    K = np.empty((2*N*(N-1), Kx.shape[1]))

    # Llenar K alternando filas de Kx y Ky
    K[::2] = Kx  # Fila 1 de Kx, fila 3 de Kx, etc.
    K[1::2] = Ky  # Fila 2 de Ky, fila 4 de Ky, etc
    h = 1/(N-1)

    return h*Kx, h*Ky, h*K

def coef(gamma, rho, beta, delta_gamma, z):
    A = beta * delta_gamma - beta * z * (z / gamma + rho) ** (z - 1)
    B = -beta * z * (z - 1) * (z / gamma + rho) ** (z - 2)
    a = (B * rho - A)/(4*rho**3)
    b_tilde = (3 * A - 2 * B * rho)/(4*rho**2)
    return a, b_tilde

def j_prima_rho(t, beta, delta_gamma, z, gamma, rho):
    
    # Concatenar Ku_x y Ku_y para formar la matriz de     
        a, b_tilde = coef(gamma, rho, beta, delta_gamma, z)

        # Caso 1: t <= 1/gamma - rho
        if t <= 1/gamma - rho:
            return 0

        # Caso 2: (1/gamma - rho) < t <= (1/gamma + rho)
        elif (1/gamma - rho) < t <= (1/gamma + rho):
            return a * (t - 1/gamma + rho)**3 + b_tilde * (t - 1/gamma + rho)**2

        # Caso 3: t > 1/gamma + rho
        else:  # t > 1/gamma + rho
            return beta * delta_gamma - beta * z * (t + (z- 1)/gamma)**(z - 1)

def j_2prima_rho(t, beta, delta_gamma, z, gamma, rho):
    
    # Concatenar Ku_x y Ku_y para formar la matriz de     
        a, b_tilde = coef(gamma, rho, beta, delta_gamma, z)
        # Caso 1: t <= 1/gamma - rho
        if t <= 1/gamma - rho:
            return 0

        # Caso 2: (1/gamma - rho) < t <= (1/gamma + rho)
        elif (1/gamma - rho) < t <= (1/gamma + rho):
            return 3 * a * (t - 1/gamma + rho)**2 + 2 * b_tilde * (t - 1/gamma + rho)

        # Caso 3: t > 1/gamma + rho
        else:  # t > 1/gamma + rho
            return - beta * z * (z-1) * (t + (z- 1)/gamma)**(z - 2)
        
def j_prim_rho_beta(t, delta_gamma, z, gamma, rho):
    A_prima = delta_gamma - z * (z / gamma + rho) ** (z - 1)
    B_prima = -z * (z - 1) * (z / gamma + rho) ** (z - 2)
    a_prima = (B_prima * rho - A_prima)/(4*rho**3)
    b_tilde_prima = (3 * A_prima - 2 * B_prima * rho)/(4*rho**2)

    if t <= 1/gamma-rho:
        return 0

    elif (1/gamma - rho) < t <= (1/gamma + rho):
        return a_prima*(t-1/gamma+rho)**3+b_tilde_prima*(t-1/gamma+rho)**2

    else:
        return delta_gamma-z*(t+(z-1)/gamma)**(z-1)

def calc_norm_Ku(u, Kx, Ky):
     Ku_x = Kx@u
     Ku_y = Ky@u
     norm_Ku = np.sqrt(Ku_x**2 + Ku_y**2)
     return norm_Ku

def hes(u, Kx, Ky, K, beta, delta_gamma, gamma, rho, z, m):
    b = np.zeros(m)
    c = np.zeros(m)
    d = np.zeros(m)
    norm_Ku_vals = calc_norm_Ku(u, Kx, Ky)
    e = np.zeros(m)

    for i in range(m):
        norm_Ku_i = norm_Ku_vals[i]
        if norm_Ku_i != 0:
            j_rho_val = j_prima_rho(
                norm_Ku_i, beta, delta_gamma, z, gamma, rho)
            j_2prima_val = j_2prima_rho(
                norm_Ku_i, beta, delta_gamma, z, gamma, rho)
            j_prim_rho_beta_val = j_prim_rho_beta(
                norm_Ku_i, delta_gamma, z, gamma, rho)
            b[i] = j_2prima_val / norm_Ku_i ** 2
            c[i] = j_rho_val / norm_Ku_i ** 3
            d[i] = j_rho_val / norm_Ku_i
            e[i] = j_prim_rho_beta_val / norm_Ku_i

    diag_d = np.repeat(d, 2)
    diag_e = np.repeat(e, 2)
    Dd = diags(diag_d, offsets=0, shape=(2*m, 2*m), format='csr')
    W = np.empty((2*Kx.shape[0], Kx.shape[1]))
    De = diags(diag_e, offsets=0, shape=(2*m, 2*m), format='csr')
    W_matrix = np.empty((2*Kx.shape[0], Kx.shape[1]))

    for i in range(m):
        # Calculamos K1 y K2 para la fila i
        Ki = np.array([Kx[i, :], Ky[i, :]])

        # Calculamos K1@u para Kx_u y Ky_u
        Kx_u = Kx[i, :] @ u
        Ky_u = Ky[i, :] @ u

        # Creamos la fila correspondiente de W
        A = Ki.T @ (b[i] * (Ki @ u))
        B = Ki.T @ (c[i] * (Ki @ u))
        C_x = d[i] * Kx[i, :]
        C_y = d[i] * Ky[i, :]

        W_matrix[2*i, :] = Kx_u * (A - B) + C_x
        W_matrix[2*i+1, :] = Ky_u * (A - B) + C_y

    Kw_prima = K.T@W_matrix
    Kw_prima_beta = (K.T@De@K@u).reshape(-1, 1)

    return Kw_prima, Dd, Kw_prima_beta
