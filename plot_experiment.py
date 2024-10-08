import os
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from bimpcc.Dataset import get_dataset
import matplotlib.pyplot as plt


def generate_plot(experiment_file):
    parent_folder = os.path.basename(os.path.dirname(experiment_file))
    dataset_name,N = os.path.splitext(os.path.basename(experiment_file))[0].split('_')
    N = int(N)
    M = N*(N-1)
    
    results = pd.read_pickle(experiment_file)
    utrue,unoisy = get_dataset(dataset_name,int(N)).get_training_data()
    # alpha = results.iloc[-1]['x'][-1]
    x = results.iloc[-1]['x']
    rec = x[:(N**2)]
    rec = rec.reshape(N,N)
    alpha = x[N**2+2*M:N**2+2*M+1]
    print(alpha)
    
    fig,ax = plt.subplots(1,4,figsize=(14,4))
    ax[0].imshow(utrue,cmap='gray')
    ax[0].set_title('True Image')
    ax[0].axis('off')
    ax[1].imshow(unoisy,cmap='gray')
    ax[1].set_title('Noisy Image\nPSNR: {:.4f}'.format(psnr(utrue,unoisy)))
    ax[1].axis('off')
    ax[2].imshow(rec,cmap='gray')
    ax[2].set_title(f'Reconstructed Image\nPSNR: {psnr(utrue,rec):.4f}')
    ax[2].set_xlabel('alpha = {}'.format(N))
    ax[2].axis('off')
    ax[3].semilogy(results['mu'],'--')
    ax[3].semilogy(results['pi'],'--')
    ax[3].semilogy(results['comp'])
    ax[3].set_title('Complementarity Evolution')
    ax[3].legend(['mu','pi','comp'])
    ax[3].set_xlabel('Iteration')
    ax[3].grid()
    plt.savefig(f'{parent_folder}/{dataset_name}_{N}.png')

def print_statistics(experiment_file):
    results = pd.read_pickle(experiment_file)
    print('Statistics:')
    print('-----------')
    print('Number of iterations: {}'.format(results.shape[0]))
    print('Final mu: {}'.format(results.iloc[-1]['mu']))
    print('Final pi: {}'.format(results.iloc[-1]['pi']))
    print('Final complementarity: {}'.format(results.iloc[-1]['comp']))
    print('Final objective: {}'.format(results.iloc[-1]['obj_val']))
    print(f'Number of variables: {results.iloc[-1]["x"].shape[0]}')
    print(f'Number of constraints: {results.iloc[-1]["num_constraints"]}')
    print(f'Number of nonzeros in jacobian: {results.iloc[-1]["jac_nz"]}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot the results of a experiment pickle file')
    parser.add_argument('file', type=str, help='The path to the pickle file')
    # parser.add_argument('dataset', type=str, help='The dataset name')
    # parser.add_argument('N', type=int, help='The number of pixels in each dimension')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, help='Save the plot')
    parser.add_argument('--print_statistics', action=argparse.BooleanOptionalAction, help='Save the plot')
    args = parser.parse_args()
    if args.print_statistics:
        print_statistics(args.file)
    else:
        generate_plot(args.file)