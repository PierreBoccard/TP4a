from chainconsumer import ChainConsumer
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import corner

def plot_convergence_by_walker(samples_mcmc, param_mcmc, n_walkers, verbose = False):
    n_params = samples_mcmc.shape[1]
    n_step = int(samples_mcmc.shape[0] / n_walkers)

    chain = np.empty((n_walkers, n_step, n_params))

    for i in np.arange(n_params):
        samples = samples_mcmc[:, i].T
        chain[:, :, i] = samples.reshape((n_step, n_walkers)).T

    mean_pos = np.zeros((n_params, n_step))
    median_pos = np.zeros((n_params, n_step))
    std_pos = np.zeros((n_params, n_step))
    q16_pos = np.zeros((n_params, n_step))
    q84_pos = np.zeros((n_params, n_step))

    # chain = np.empty((nwalker, nstep, ndim), dtype = np.double)
    for i in np.arange(n_params):
        for j in np.arange(n_step):
            mean_pos[i][j] = np.mean(chain[:, j, i])
            median_pos[i][j] = np.median(chain[:, j, i])
            std_pos[i][j] = np.std(chain[:, j, i])
            q16_pos[i][j] = np.percentile(chain[:, j, i], 16.)
            q84_pos[i][j] = np.percentile(chain[:, j, i], 84.)

    fig, ax = plt.subplots(n_params, sharex=True, figsize=(16, 2 * n_params))
    if n_params == 1: ax = [ax]
    last = n_step
    burnin = int((9.*n_step) / 10.) #get the final value on the last 10% on the chain

    for i in range(n_params):
        if verbose :
            print(param_mcmc[i], '{:.4f} +/- {:.4f}'.format(median_pos[i][last - 1], (q84_pos[i][last - 1] - q16_pos[i][last - 1]) / 2))
        ax[i].plot(median_pos[i][:last], c='g')
        ax[i].axhline(np.median(median_pos[i][burnin:last]), c='r', lw=1)
        ax[i].fill_between(np.arange(last), q84_pos[i][:last], q16_pos[i][:last], alpha=0.4)
        ax[i].set_ylabel(param_mcmc[i], fontsize=10)
        ax[i].set_xlim(0, last)

    return fig


def main():
    parameters = ['$\log(M_{disk})$', 'a', r'$r_{0,halo}$', r'$\rho_0$', r'$\log(M_{bulb})$']
    #parameters = ['\log(M_{disk})', 'a', 'r_{0,halo}', '\rho_0', '\log(M_{bulb})']
    file = 'samples.pkl'
    chain = pkl.load(open(file, 'rb'))
    nwalkers = 50

    fig = plot_convergence_by_walker(chain, parameters, nwalkers, verbose=False)
    plt.show()

    x = int(input('Enter burnin :'))
    burnin = x * nwalkers
    chain = chain[burnin:, :]

    CC = ChainConsumer()
    CC.add_chain(chain, parameters=parameters)
    CC.configure(summary=True)
    CC.plotter.plot(display=True, filename= 'cornerplot.png')

if __name__ == '__main__':
    main()
