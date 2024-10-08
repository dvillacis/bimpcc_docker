import numpy as np
from scipy.sparse import spdiags
import cyipopt
from bimpcc.utils import generate_2D_gradient_matrices
from bimpcc.L2TVMPCC import L2TVMPCC
from bimpcc.L2TVMPCC_init import L2TVMPCC_init
from bimpcc.solve_mpcc import solve_mpcc
from bimpcc.Dataset import get_dataset


def test(size=5, dataset_name='cameraman'):
    # np.random.seed(0)
    # utrue = np.tril(0.8*np.ones((size, size)))
    # unoisy = utrue + 0.1*np.random.randn(size, size).clip(0,1)

    utrue, unoisy = get_dataset(dataset_name, size).get_training_data()

    Kx, Ky, _ = generate_2D_gradient_matrices(size)
    # Kx = (1/(size**2))*spdiags([np.ones(size**2)], [0], size*(size-1), size**2)
    # Ky = (1/(size**2))*spdiags([np.ones(size**2)], [0], size*(size-1), size**2)
    # Kx = Kx - 1e-2*Kx_d
    # print(Kx.toarray())
    # print(f'Kx {Kx.shape}:\n{Kx.toarray()} - {np.linalg.cond(Kx.toarray())}')
    # print(f'Ky {Ky.shape}:\n{Ky.toarray()} - {np.linalg.cond(Ky.toarray())}')
    M, N = Kx.shape
    # Kx = (1/N*M)*Kx
    # Ky = (1/N*M)*Ky
    P = 1

    u0 = unoisy.ravel()
    # u0 = np.zeros(N)
    qx0 = 0.0*np.ones(M)
    qy0 = 0.0*np.ones(M)
    alpha0 = 0.0*np.ones(P)  # np.array([0.1])
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
    
    
    # Initial guess computation
    x0 = np.concatenate((u0.ravel(), qx0, qy0, alpha0, r0, delta0, theta0))
    pinstance_init = L2TVMPCC_init(1e-4, A, Kx, Ky, Q, utrue, unoisy)
    nlp_init = cyipopt.Problem(
        n=len(x0),
        m=N+4*M,
        problem_obj=pinstance_init,
        lb=lb,
        ub=ub,
        cl=cl_init,
        cu=cu_init
    )
    print(f'Computing initial guess')
    # nlp_init.add_option('jacobian_approximation', 'finite-difference-values')
    # nlp_init.add_option('derivative_test', 'first-order')
    nlp_init.add_option('print_level', 5)
    nlp_init.add_option('sb', 'yes')
    nlp_init.add_option('tol', 1e-2)
    x_, info_init = nlp_init.solve(x0)

    # return x_, info_init
    vars, info = solve_mpcc(
        L2TVMPCC,
        x_,
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
        pi_init=1.0,
        mu_init=0.1,
        tol=1e-2
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
    
    # profiler = cProfile.Profile()
    # profiler.enable()
    vars,info = test(args.N, args.dataset)
    # profiler.disable()
    
    # s = StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
    
    if args.save:
        print(f'Saving to results_ScalarTVDenoising/{args.dataset}_{args.N}.pkl')
        info.to_pickle(
            f'results_ScalarTVDenoising/{args.dataset}_{args.N}.pkl')
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
