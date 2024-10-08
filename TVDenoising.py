import numpy as np
from scipy.sparse import spdiags
import cyipopt
from bimpcc.utils import generate_2D_gradient_matrices
from bimpcc.L2TVMPCC import L2TVMPCC
from bimpcc.L2TVMPCC_pen import L2TVMPCC_pen
from bimpcc.L2TVMPCC_init import L2TVMPCC_init
from bimpcc.solve_mpcc import solve_mpcc
from bimpcc.Dataset import get_dataset

from scipy.interpolate import RegularGridInterpolator, interp1d

def interpolate(data, sizeout):
    print(f'sizeout: {sizeout}')
    u = data[0]
    zx = data[1]
    zy = data[2]
    alpha = data[3]
    r = data[4]
    delta = data[5]
    theta = data[6]
    
    sizein = int(np.sqrt(len(u)))
    N = sizeout
    M = sizeout*(sizeout-1)
    
    matriz_u = u.reshape(sizein, sizein)
    print(f'matriz_u: {matriz_u}')


    # Definimos los valores originales de la imagen NxN
    x = np.linspace(0,1,sizein)  # Coordenadas en el eje x
    y = np.linspace(0,1,sizein)  # Coordenadas en el eje y

    # Creamos el interpolador usando RegularGridInterpolator
    f = RegularGridInterpolator((x, y), matriz_u, method='linear')

    # Creamos nuevas coordenadas para la imagen
    xnew = np.linspace(0, 1, N)  # Nuevas coordenadas en x (N puntos)
    ynew = np.linspace(0, 1, N)
    xnew_grid, ynew_grid = np.meshgrid(xnew, ynew)

    unew_matrix = (f((xnew_grid, ynew_grid))).T

    unew = unew_matrix.ravel()
    
    # vector zx
    ext = np.linspace(0, 1, len(zx))
    zxnew = np.linspace(0, 1, M)
    interpolador = interp1d(ext, zx, kind='linear')
    zx_new = interpolador(zxnew)

    # vector zy
    zynew = np.linspace(0, 1, M)
    interpoladorzy = interp1d(ext, zy, kind='linear')
    zy_new = interpoladorzy(zynew)

    # vector r
    rnew = np.linspace(0, 1, M)
    interpoladorr = interp1d(ext, r, kind='linear')
    r_new = interpoladorr(rnew)

    # vector delta
    dnew = np.linspace(0, 1, M)
    interpoladord = interp1d(ext, delta, kind='linear')
    d_new = interpoladord(dnew)

    # vector theta
    tnew = np.linspace(0, 1, M)
    interpoladort = interp1d(ext, theta, kind='linear')
    t_new = interpoladord(tnew)
    
    xout = np.concatenate((unew, zx_new, zy_new, alpha, r_new, d_new, t_new))
    
    return xout

def test(size=10, dataset_name='cameraman'):

    utrue, unoisy = get_dataset(dataset_name, size).get_training_data()

    Kx, Ky, _ = generate_2D_gradient_matrices(size)
    M, N = Kx.shape
    P = 1

    u0 = unoisy.ravel()
    qx0 = 0.0*np.ones(M)
    qy0 = 0.0*np.ones(M)
    alpha0 = 0.01*np.ones(P)  # np.array([0.1])
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

    ub_u = np.ones(N)
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
    # cl_6 = np.zeros(M)
    cl = np.concatenate((cl_1, cl_2, cl_3, cl_4, cl_5))
    cl_init = np.concatenate((cl_1, cl_2, cl_3, cl_4, cl_5))

    cu_1 = np.zeros(N)
    cu_2 = np.zeros(M)
    cu_3 = np.zeros(M)
    cu_4 = np.zeros(M)
    cu_5 = np.zeros(M)
    # cu_6 = 1e20*np.ones(M)
    cu = np.concatenate((cu_1, cu_2, cu_3, cu_4, cu_5))
    cu_init = np.concatenate((cu_1, cu_2, cu_3, cu_4, cu_5))
    
    # if size <= 5:
    #     x0 = np.concatenate((u0, qx0, qy0, alpha0, r0, delta0, theta0))
    # else:
    #     variables,info = test(size//5, dataset_name)
    #     x0 = interpolate(variables,size)
    x0 = np.concatenate((u0, qx0, qy0, alpha0, r0, delta0, theta0))
            
    vars, info = solve_mpcc(
        L2TVMPCC_pen,
        x0,
        lb,
        ub,
        cl,
        cu,
        A,
        Kx,
        Ky,
        Q,
        utrue,
        unoisy,
        pi_init=0.1,
        mu_init=0.1,
        tol=1e-3
    )
    
    return vars,info


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='cameraman')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    vars,info = test(args.N, args.dataset)
    
    if args.save:
        print(f'Saving to results_ScalarTVDenoising_global/{args.dataset}_{args.N}.pkl')
        info.to_pickle(
            f'results_ScalarTVDenoising_global/{args.dataset}_{args.N}.pkl')
    print(vars)