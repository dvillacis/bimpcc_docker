import numpy as np
from scipy.sparse import spdiags
import cyipopt
from bimpcc.utils import generate_2D_gradient_matrices
from bimpcc.L2TVMPCC import L2TVMPCC
from bimpcc.L2TVMPCC_init import L2TVMPCC_init
from bimpcc.solve_mpcc import solve_mpcc
from bimpcc.Dataset import get_dataset

from scipy.interpolate import RegularGridInterpolator, interp1d


def interpolate(u, zx, zy, alpha, r, delta, theta, sizeout):
    sizein = int(np.sqrt(len(u)))
    N = sizeout
    M = sizeout*(sizeout-1)

    # u
    matriz_u = u.reshape(sizein, sizein)

    # Definimos los valores originales de la imagen NxN
    x = np.linspace(0, 1, sizein)  # Coordenadas en el eje x
    y = np.linspace(0, 1, sizein)  # Coordenadas en el eje y

    # Creamos el interpolador usando RegularGridInterpolator
    f = RegularGridInterpolator((y, x), matriz_u, method='linear')

    # Creamos nuevas coordenadas para la imagen
    xnew = np.linspace(0, 1, N)  # Nuevas coordenadas en x (N puntos)
    ynew = np.linspace(0, 1, N)
    xnew_grid, ynew_grid = np.meshgrid(xnew, ynew, sparse=True)

    unew_matrix = f((ynew_grid, xnew_grid))

    unew = unew_matrix.ravel()

    # vector zx
    matriz_zx = zx.reshape(sizein, sizein-1)

    # Definimos los valores originales de la imagen
    xzx = np.linspace(0, 1, sizein-1)  # columnas
    yzx = np.linspace(0, 1, sizein)  # filas

    g1 = RegularGridInterpolator(
        (yzx, xzx), matriz_zx, bounds_error=False, fill_value=None)

    # Creamos nuevas coordenadas para la imagen
    xznewx = np.linspace(0, 1, N-1)
    yznewx = np.linspace(0, 1, N)
    xnewzx_grid, ynewzx_grid = np.meshgrid(xznewx, yznewx, sparse=True)
    zxnew_matrix = g1((ynewzx_grid, xnewzx_grid))
    zxnew = zxnew_matrix.ravel()

    # vector zy
    matriz_zy = zy.reshape(sizein, sizein-1)

    g2 = RegularGridInterpolator(
        (yzx, xzx), matriz_zy, bounds_error=False, fill_value=None)

    zynew_matrix = g2((ynewzx_grid, xnewzx_grid))
    zynew = zynew_matrix.ravel()

    # vector r
    matriz_r = r.reshape(sizein, sizein-1)

    g3 = RegularGridInterpolator(
        (yzx, xzx), matriz_r, bounds_error=False, fill_value=None)

    rnew_matrix = g3((ynewzx_grid, xnewzx_grid))
    rnew = rnew_matrix.ravel()

    # vector delta
    matriz_d = delta.reshape(sizein, sizein-1)

    g4 = RegularGridInterpolator(
        (yzx, xzx), matriz_d, bounds_error=False, fill_value=None)

    dnew_matrix = g4((ynewzx_grid, xnewzx_grid))
    dnew = dnew_matrix.ravel()

    # vector theta
    matriz_t = theta.reshape(sizein, sizein-1)

    g5 = RegularGridInterpolator(
        (yzx, xzx), matriz_t, bounds_error=False, fill_value=None)

    tnew_matrix = g5((ynewzx_grid, xnewzx_grid))
    tnew = tnew_matrix.ravel()

    xout = np.concatenate((unew, zxnew, zynew, alpha, rnew, dnew, tnew))

    return xout


def test(size, size_initial, dataset_name='cameraman'):
    # Paso 1: Resolver para N inicial
    utrue, unoisy = get_dataset(dataset_name, size_initial).get_training_data()

    Kx, Ky, _ = generate_2D_gradient_matrices(size_initial)
    M, N = Kx.shape
    P = 1

    u0 = unoisy.ravel()
    qx0 = 0.0*np.ones(M)
    qy0 = 0.0*np.ones(M)
    alpha0 = 0.0*np.ones(P)
    r0 = 1e-1*np.ones(M)
    delta0 = 0.0*np.ones(M)
    theta0 = 1e-5*np.ones(M)

    A = np.eye(N)
    Q = np.ones((M, P))

    lb_u = np.zeros(N)
    lb_qx = -1e20*np.ones(M)
    lb_qy = -1e20*np.ones(M)
    lb_alpha = 1e-10*np.ones(P)
    lb_r = 1e-10*np.ones(M)
    lb_delta = 1e-20*np.ones(M)
    lb_theta = -1e20*np.ones(M)
    lb = np.concatenate(
        (lb_u, lb_qx, lb_qy, lb_alpha, lb_r, lb_delta, lb_theta))

    ub_u = 1e20*np.ones(N)
    ub_qx = 1e20*np.ones(M)
    ub_qy = 1e20*np.ones(M)
    ub_alpha = 1e20*np.ones(P)
    ub_r = 1e20*np.ones(M)
    ub_delta = 1e20*np.ones(M)
    ub_theta = 1e20*np.ones(M)
    ub = np.concatenate(
        (ub_u, ub_qx, ub_qy, ub_alpha, ub_r, ub_delta, ub_theta))

    cl_1 = np.zeros(N)
    cl_2 = np.zeros(M)
    cl_3 = np.zeros(M)
    cl_4 = np.zeros(M)
    cl_5 = np.zeros(M)
    cl_6 = np.zeros(M)
    cl = np.concatenate((cl_1, cl_2, cl_3, cl_4, cl_5, cl_6))
    cl_init = np.concatenate((cl_1, cl_2, cl_3, cl_4, cl_5))

    cu_1 = np.zeros(N)
    cu_2 = np.zeros(M)
    cu_3 = np.zeros(M)
    cu_4 = np.zeros(M)
    cu_5 = np.zeros(M)
    cu_6 = 1e20*np.ones(M)
    cu = np.concatenate((cu_1, cu_2, cu_3, cu_4, cu_5, cu_6))
    cu_init = np.concatenate((cu_1, cu_2, cu_3, cu_4, cu_5))

    # Computar la solución inicial para N=35
    x0 = np.concatenate((u0, qx0, qy0, alpha0, r0, delta0, theta0))
    pinstance_init = L2TVMPCC_init(1e-3, A, Kx, Ky, Q, utrue, unoisy)
    nlp_init = cyipopt.Problem(
        n=len(x0),
        m=N+4*M,
        problem_obj=pinstance_init,
        lb=lb,
        ub=ub,
        cl=cl_init,
        cu=cu_init
    )
    print(f'Computing initial guess for size {size_initial}')
    nlp_init.add_option('print_level', 5)
    nlp_init.add_option('sb', 'yes')
    nlp_init.add_option('tol', 1e-2)
    x_, info_init = nlp_init.solve(x0)

    # Paso 2: Interpolación de N inicial a N grande
    size_out = size
    x_init = interpolate(x_[:N], x_[N:N+M], x_[N+M:N+2*M], x_[N+2*M:N+2*M+P],
                         x_[N+2*M+P:N+3*M+P], x_[N+3*M+P:N+4*M+P], x_[N+4*M+P:N+5*M+P], size_out)

    utrue_new, unoisy_new = get_dataset(
        dataset_name, size_out).get_training_data()
    Kx_new, Ky_new, _ = generate_2D_gradient_matrices(size_out)
    M_new, N_new = Kx_new.shape

    A_new = np.eye(N_new)
    Q_new = np.ones((M_new, P))
    lb_new = np.concatenate(
        (np.zeros(N_new), -1e20*np.ones(M_new), -1e20*np.ones(M_new),
         1e-10*np.ones(P), 1e-10*np.ones(M_new), 1e-20*np.ones(M_new), -1e20*np.ones(M_new)))

    ub_new = np.concatenate(
        (np.ones(N_new), 1e20*np.ones(M_new), 1e20*np.ones(M_new),
         1e20*np.ones(P), 1e20*np.ones(M_new), 1e20*np.ones(M_new), 1e20*np.ones(M_new)))

    cl_new = np.concatenate((np.zeros(N_new), np.zeros(M_new), np.zeros(
        M_new), np.zeros(M_new), np.zeros(M_new), np.zeros(M_new)))
    cu_new = np.concatenate((np.zeros(N_new), np.zeros(M_new), np.zeros(
        M_new), np.zeros(M_new), np.zeros(M_new), 1e20*np.ones(M_new)))

    # Resolver el problema con la interpolación en tamaño N=40
    vars, info = solve_mpcc(
        L2TVMPCC,
        x_init,
        lb_new,
        ub_new,
        cl_new,
        cu_new,
        A_new,
        Kx_new,
        Ky_new,
        Q_new,
        utrue_new,
        unoisy_new,
        pi_init=0.1,
        mu_init=0.1,
        tol=1e-3
    )

    return vars, info


if __name__ == '__main__':
    import argparse
    import cProfile
    from io import StringIO
    import pstats

    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='cameraman')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    vars, info = test(args.N, args.n, args.dataset)

    if args.save:
        print(
            f'Saving to results_ScalarTVDenoising_global/{args.dataset}_{args.N}.pkl')
        info.to_pickle(
            f'results_ScalarTVDenoising_global/{args.dataset}_{args.N}.pkl')
    # print(vars)
