import numpy as np
from scipy.sparse import spdiags
import cyipopt
from bimpcc.utils import generate_2D_gradient_matrices
from bimpcc.DCTVMPCC2 import DCTVMPCC2
from bimpcc.solve_mpcc import solve_mpcc
from bimpcc.Dataset import get_dataset


def test(size=5, dataset_name='cameraman'):
    # np.random.seed(0)
    # utrue = np.tril(0.8*np.ones((size, size)))
    # unoisy = utrue + 0.1*np.random.randn(size, size).clip(0,1)

    utrue, unoisy = get_dataset(dataset_name, size).get_training_data()

    Kx, Ky, K = generate_2D_gradient_matrices(size)
    # Kx = (1/(size**2))*spdiags([np.ones(size**2)], [0], size*(size-1), size**2)
    # Ky = (1/(size**2))*spdiags([np.ones(size**2)], [0], size*(size-1), size**2)
    # Kx = Kx - 1e-2*Kx_d
    # print(Kx.toarray())
    # print(f'Kx {Kx.shape}:\n{Kx.toarray()} - {np.linalg.cond(Kx.toarray())}')
    # print(f'Ky {Ky.shape}:\n{Ky.toarray()} - {np.linalg.cond(Ky.toarray())}')
    M, N = Kx.shape
    # Kx = (1/N*M)*Kx
    # Ky = (1/N*M)*Ky

    u0 = unoisy.ravel()
    # u0 = np.zeros(N)
    qx0 = 1e-5*np.ones(M) # zx0
    qy0 = 1e-5*np.ones(M) # zy0
    alpha0 = 1e-1*np.ones(1) # np.array([0.1]) # beta0
    r0 = 1e-1*np.ones(M) 
    delta0 = 1e-6*np.ones(M)
    theta0 = 1e-5*np.ones(M)
    
    x0 = np.concatenate((u0.ravel(), qx0, qy0, alpha0, r0, delta0, theta0))

    lb_u = np.zeros(N)
    lb_qx = -1e20*np.ones(M)
    lb_qy = -1e20*np.ones(M)
    lb_alpha = 1e-10*np.ones(1) 
    lb_r = 1e-10*np.ones(M)
    lb_delta = 1e-20*np.ones(M)
    lb_theta = -1e20*np.ones(M)
    lb = np.concatenate(
        (lb_u, lb_qx, lb_qy, lb_alpha, lb_r, lb_delta, lb_theta))

    ub_u = np.ones(N)
    ub_qx = 1e20*np.ones(M)
    ub_qy = 1e20*np.ones(M)
    ub_alpha = 1e20*np.ones(1) 
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
    # cl = np.concatenate((cl_2, cl_3))

    cu_1 = np.zeros(N)
    cu_2 = np.zeros(M)
    cu_3 = np.zeros(M)
    cu_4 = np.zeros(M)
    cu_5 = np.zeros(M)
    cu_6 = 1e20*np.ones(M)
    cu = np.concatenate((cu_1, cu_2, cu_3, cu_4, cu_5, cu_6))
    # cu = np.concatenate((cu_2, cu_3))

    vars, info = solve_mpcc(
        DCTVMPCC2,
        x0,
        lb,
        ub,
        cl,
        cu,
        Kx,
        Ky,
        K,
        utrue,
        unoisy,
        pi_init=0.1,
        mu_init=0.1,
        tol=1e-3
    )

    # print(f'Solution: {vars[0].reshape(size,size)}')
    # print(f'delta {vars[5]}')
    # print(f'unoisy: {unoisy}')
    
    return vars,info


if __name__ == '__main__':
    import argparse
    import cProfile
    from io import StringIO
    import pstats

    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='cameraman')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    #profiler = cProfile.Profile()
    #profiler.enable()
    vars,info = test(args.N, args.dataset)
    #profiler.disable()
    
    #s = StringIO()
    #sortby = 'cumulative'
    #ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print(s.getvalue())
    
    if args.save:
        print(f'Saving to results_ScalarTVqDenoising/{args.dataset}_{args.N}.pkl')
        info.to_pickle(
            f'results_ScalarTVqDenoising/{args.dataset}_{args.N}.pkl')
    # print(f'alpha: {alpha}')
    # if args.plot:
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(utrue, vmax=1, vmin=0)
    #     plt.title('True Image')
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(unoisy, vmax=1, vmin=0)
    #     plt.title('Noisy Image')
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(x.reshape(args.N, args.N), vmax=1, vmin=0)
    #     plt.title('Denoised Image')
    #     plt.show()
    print(vars)
