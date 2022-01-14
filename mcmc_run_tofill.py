import emcee
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
import velocity

D = 9200
i = 75
fname = 'ngc3198_rot_curve.txt'
v_data = np.loadtxt(fname, skiprows=2, usecols=0)*np.sin(np.pi*i/180)
sigma = np.loadtxt(fname, skiprows=2, usecols=1)*np.sin(np.pi*i/180)
r = np.loadtxt(fname, skiprows=2, usecols=2)* D * 0.00029088599995183


def log_likelihood(theta):
   
    
    M_disks, aa, r_0s, rho_0s, M_cms = theta
    M_disks, M_cms, rho_0s = 10**(M_disks), 10**(M_cms), 10**(rho_0s)
    v_fit = velocity.V_tot(r, M_cms, M_disks, aa, r_0s, rho_0s)
    LL = ((v_data-v_fit)/sigma)**2

    if M_cms < 10**6 or M_cms > 10**11:
        return -np.inf
    if M_disks < 10**10 or M_disks > 10**12:
        return -np.inf    
    if aa < 1 or aa > 60:
        return -np.inf
    if r_0s < 10 or r_0s > 100:
        return -np.inf
    if rho_0s < 0 or rho_0s > 10**12:
        return -np.inf                    
    #define your log likelihood here L, it should be proportionnal to -chi^2 /2
    L = -(0.5)*np.sum(LL)
    return L


def main():
    nwalkers = 100
    nsamples = 5000
    save_output = True
    startpos =  np.asarray([10.5,5.0,15.0,9.5,8.0])
    
    pos = startpos + startpos * 0.01 * np.random.randn(nwalkers, len(startpos)) # initialise the walkers
    nwalkers, ndim = pos.shape

    with Pool(processes=cpu_count()) as pool: #multiprocessing, remove you want to comput on a single core
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, pool=pool)
        sampler.run_mcmc(pos, nsamples, progress=True)

    flat_samples = sampler.get_chain(discard=0, flat=True)
    lhood_samples = sampler.get_log_prob(discard=0, flat=True)

    print(np.shape(flat_samples))
    print(flat_samples)

    if save_output :
        pkl_file = open('samples.pkl', 'wb')
        pkl_file2 = open('logLikelihood.pkl', 'wb')
        pickle.dump(flat_samples, pkl_file, protocol=-1)
        pkl_file.close()
        pickle.dump(lhood_samples, pkl_file2, protocol=-1)
        pkl_file2.close()

if __name__ == '__main__':
    main()