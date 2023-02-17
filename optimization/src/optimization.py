import numpy as np
from scipy.stats import norm
from tqdm.auto import tqdm
import torch
from .vae_model import VAE
from .oracle import Oracle
from .util import *
#from .encoding_utils import allowed
from .train_vae import start_training, evaluate
import time

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def run_greedy(save_path, data_config, opt_config, device, verbose=True):
    
    """
    Runs optimization with greedy single step walk
    """
    start = time.time()

    assert opt_config['type'] in ['greedy']
    n_mix = opt_config["n_mix"]
    iters = opt_config["iters"]
    sites = data_config["sites"]
    samples = opt_config["samples"]
    n_directions = opt_config['directions']

    #keeps tracks of means of scores, counts, and diversity, variances of socres, counts, and diversity, and input sequences respectively
    traj_means = np.zeros((iters, samples, 5))
    traj_vars = np.zeros((iters, samples, 5))
    traj_Xts = np.zeros((iters, samples, sites*12, n_mix))

    np.random.seed(opt_config['seed'])
    oracle = Oracle(data_config, opt_config)

    #is the last step not even saved?
    for t in range(iters):
        print('\n####  Iteration: ' + str(t+1) + '  ####')
        
        if t > 0:
            #all_Xt_new = np.zeros((samples, sites*12, n_mix, n_directions + 1))
            all_Xt_new = np.copy(all_Xt)
            all_means_new = np.zeros((samples, 5, n_mix, n_directions + 1))
            all_vars_new = np.zeros((samples, 5, n_mix, n_directions + 1))

            for j, index in enumerate(all_yt_max_idx):
                #which direction had the best mutation
                remainder = index % (n_directions + 1)
                #which mixed base library encoding had the best mutation (out of the mixed ones)
                k = int((index - remainder)/(n_directions + 1))
                #set the new starting point as the best single step from 
                all_Xt_new[j, :, k, 0] = all_Xt[j, :, k, remainder]    

            #need to recalculate
            all_Xt = all_Xt_new
            Xt = all_Xt[:, :, :, 0].astype(int)

            #check if more needs to be filled in here
            all_means_new[:, :, 0, 0], all_vars_new[:, :, 0, 0] = oracle.predict(Xt)

            all_means = all_means_new
            all_vars = all_vars_new
            means = all_means[:, :, 0, 0]
            vars = all_vars[:, :, 0, 0]

            all_yt_max = all_means[:, 0, 0, 0]

        else:
            #store the proposed steps
            #there will be one for each mixed base library
            all_means = np.zeros((samples, 5, n_mix, n_directions + 1))
            all_vars = np.zeros((samples, 5, n_mix, n_directions + 1))
            
            if "library" in data_config:
                #load optimized libraries
                if isinstance(data_config['library'], str):
                    if '.npy' in data_config['library']:
                        Xt = np.load('dclo/saved/' + data_config['library'], allow_pickle=True)
                    else:
                        results = np.load('dclo/saved/' + data_config["library"] + '/results.npy', allow_pickle=True)
                        Xts = results.item()['Xts']
                        Xt = Xts[-1, :, :]
                #load manual list as a starting point
                else:
                    Xt = np.zeros((len(data_config['library']), sites*12, 1))
                    for i, seq in enumerate(data_config['library']):
                        Xt[i] = seq2encoding(seq).T
            else:
                Xt = get_init_samples(samples, n_mix, sites)

            Xt = Xt.astype(int)

            all_Xt = np.zeros((samples, sites*12, n_mix, n_directions + 1))

            means, vars = oracle.predict(Xt)

            #check if more needs to be filled in here
            all_means[:, :, 0, 0] = means
            all_vars[:, :, 0, 0] = vars
            all_Xt[:, :, :, 0] = Xt
            all_yt_max = all_means[:, 0, 0, 0]

        traj_means[t, :, :] = means
        traj_vars[t, :, :] = vars
        traj_Xts[t, :, :, :] = Xt

        #need to fix this to print multiple sequences (right now it just sticks them next to each other)
        if verbose:
            for row, score, unweighted_score, raw_score, counts, diversity in zip(Xt[:,:,0], all_yt_max, all_means[:, 1, 0, 0], all_means[:, 2, 0, 0], all_means[:, 3, 0, 0], all_means[:, 4, 0, 0]):
                print("%s --- Weighted: %.3f, Unweighted: %.3f, Raw Simple: %.3f, Top Counts: %3.0f, Diversity: %3.0f" % (encoding2seq(row), score, unweighted_score, raw_score, counts, diversity))

        ### Evaluate oracle for the proposed steps ###
        #implified method, alternating optimization of each library

        #all_yt_max_idx = np.zeros((samples, n_mix))

        #all_Xt_old = np.copy(all_Xt)

        #test moves if its not the final iteration
        if t != iters - 1:
            with tqdm() as pbar:
                pbar.reset(n_directions*n_mix)
                pbar.set_description('Predicting through oracle')

                #should really only change one library at a time, not the best of both
                
                #np.random.seed(opt_config['seed'])

                for k in range(n_mix):
                    #all_means2 = np.copy(all_means)
                    #all_vars2 = np.copy(all_vars)
                    #all_Xt_test = np.copy(all_Xt_old)

                    #make the mutations
                    for i, encodings in enumerate(Xt):
                        row = encodings[:, k]

                        choices = allowed(row)
                        directions = np.random.choice(choices, n_directions, replace=False)

                        for j, index in enumerate(directions):
                            row_new = np.copy(row)
                            row_new = row_new.reshape(1, -1)
                            #invert the mixed based library at one position
                            row_new[0, index] = ~row_new[0, index] + 2

                            #for keeping track of all mutations
                            all_Xt[i, :, k, j+1] = row_new
                            #for testing only mutations to one sequence at a time
                            #all_Xt_test[i, :, k, j+1] = row_new
                    
                    #predict through the oracle
                    for j in range(n_directions):
                        #take all the originals but replace the row with the mutated on
                        #need to check to make sure this makes sense
                        Xt_new = np.copy(all_Xt[:, :, :, 0])
                        Xt_new[:, :, k] = all_Xt[:, :, k, j+1]

                        #print(Xt_new)
                        all_means[:, :, k, j+1], all_vars[:, :, k, j+1] = oracle.predict(Xt_new)
                        pbar.update()

            pbar.close 

            all_yt = all_means[:, 0, :, :]
            all_yt = all_yt.reshape(all_yt.shape[0], -1)
            all_yt_max_idx = np.argmax(all_yt, axis=1)
            #confused about this part, still need to evaluate both together    
    
    results = {'means': traj_means, 'vars': traj_vars, 'Xts': traj_Xts}
    np.save(save_path + '/results.npy', results)
    
    end = time.time()
    print('Time: ' + str(end-start))

    return results

# def run_dbas(save_path, data_config, vae_model_config, vae_train_config, opt_config, device, verbose=True, homoscedastic=False):
    
#     """
#     Runs weighted optimization with  dbas
#     """
    
#     assert opt_config['type'] in ['dbas']
#     iters = opt_config["iters"]
#     sites = data_config["sites"]
#     samples = opt_config["samples"]
#     uncertainty = opt_config["uncertainty"]

#     traj = np.zeros((iters, 10))
#     oracle_samples = np.zeros((iters, samples))
#     gt_samples = np.zeros((iters, samples))
#     oracle_max_seq = None
#     oracle_max = -np.inf
#     gt_of_oracle_max = -np.inf
#     y_star = -np.inf  

#     np.random.seed(opt_config['seed'])
#     oracle = Oracle(data_config, opt_config)
    
    
#     vae = VAE(vae_model_config, data_config).to(device)
#     # Initialize optimizer
#     vae.init_optimizer(vae_train_config)

#     for t in range(iters):
#         print('\n####  Iteration: ' + str(t+1) + '  ####')
#         ### Take random normal samples ###
#         zt = np.random.randn(samples, vae_model_config['z_dim'])
#         if t > 0:
#             Xt_p = vae.decode(torch.tensor(zt).float().to(device))
#             Xt_new = get_samples(Xt_p.detach().cpu().numpy())
#             if opt_config['append_new']:
#                 Xt = np.concatenate((Xt_new, Xt), axis = 0)
#             else:
#                 Xt = Xt_new

#             #can train the VAE with all the samples or just the new ones
#             #print(Xt.shape)
#             # print(Xt_new.shape)
#             # Xt = np.concatenate((Xt_new, Xt), axis = 0)
#             # print(Xt.shape)
#         else:
#             #np.random.seed(opt_config['seed'] + t)
#             Xt = get_init_samples(samples, sites)
        
#         ### Evaluate ground truth and oracle ###

#         means, vars = oracle.predict(Xt)

#         yt = means[:, 0]
#         counts = means[:, 1]
#         div = means[:, 2]

#         yt_var = vars[:, 0]
#         counts_var = vars[:, 1]
#         div_var = vars[:, 2]
        
#         ### Calculate weights for different schemes ###
#         if t > 0:
#             #weights for dbas
#             #finds the y value that represents the desired percentile
#             y_star_1 = np.percentile(yt, opt_config['quantile']*100)
            
#             if y_star_1 > y_star:
#                 y_star = y_star_1

#             print('Quantile Cutoff: %6.0f' % y_star)
#             #uses the survival function (1 - cdf), shouldn't it be the opposite?
#             #find what fraction of samples lie above y_star if the zs_distribution was modelled as a standard normal
#             ###in the original paper, highly uncertain weights are penalized###
#             #instead we penalize  the opposite (low-variance ZS score distributions), but really we should penalize diversity

#             if uncertainty == True:
#                 weights = norm.sf(y_star, loc=yt, scale=np.sqrt(yt_var))
#             else:
#             ###ignore the uncertainty of the weights###
#                 weights = norm.sf(y_star, loc=yt)
#         else:
#             weights = np.ones(yt.shape[0])

#         print('Sum of Weights: %3.0f' % np.sum(np.sum(weights)))

#         yt_max_idx = np.argmax(yt)
#         yt_max = yt[yt_max_idx]
        
#         if yt_max > oracle_max:
#             oracle_max = yt_max
#             div_max = div[yt_max_idx]
#             counts_max = counts[yt_max_idx]
#             try:
#                 oracle_max_seq = encoding2seq(Xt[yt_max_idx])
#             except IndexError:
#                 print(Xt[yt_max_idx])
        
#         #is this subsampling or just reordering initially?
#         if t == 0:
#             rand_idx = np.random.randint(0, len(yt), samples)
#             oracle_samples[t, :] = yt[rand_idx]
#         #is it even used later on?
#         if t > 0:
#             oracle_samples[t, :] = yt[-samples:]
        
#         #Keep track of training statistics
#         traj[t, 0] = np.max(yt)
#         traj[t, 1] = np.mean(yt)
        
#         #for now just use the counts for the other max
#         traj[t, 2] = counts[yt_max_idx]
#         traj[t, 3] = np.mean(counts)

#         traj[t, 4] = div[yt_max_idx]
#         traj[t, 5] = np.mean(div)
#         traj[t, 6] = np.std(yt)

#         traj[t, 7] = np.mean(np.sqrt(yt_var))
#         traj[t, 8] = np.mean(np.sqrt(counts_var))
#         traj[t, 9] = np.mean(np.sqrt(div_var))
        
#         ### Record and print results ##
#         if verbose:
#             print("Mean Score: %.3f,  Mean Counts: %4.0f, Mean Diversity: %4.0f" % (traj[t, 1], traj[t, 3], traj[t, 5]))
#             print("Std Score: %.3f, Std Counts: %5.0f, Std Diversity: %5.0f" % (traj[t, 7], traj[t, 8], traj[t, 9]))
#             print("Best Sequence of this Iteration: %s (Score: %.3f, Counts: %4.0f, Diversity: %4.0f)" % (encoding2seq(Xt[yt_max_idx]), traj[t, 0], traj[t, 2], traj[t, 4]))

#             print("Running Best: %s (Score: %.3f, Counts: %4.0f,  Diversity: %4.0f)" % (oracle_max_seq,oracle_max, counts_max, div_max))
            
#             # print(t, traj[t, 3], color.BOLD + str(traj[t, 4]) + color.END, traj[t, 5], traj[t, 6])
        
#         ### Train model ###
#         #changed code so that training starts in the first round
#         # if t == 0:
#         #     #set weights here
#         #     pass
#         #     # vae.encoder_.set_weights(vae_0.encoder_.get_weights())
#         #     # vae.decoder_.set_weights(vae_0.decoder_.get_weights())
#         #     # vae.vae_.set_weights(vae_0.vae_.get_weights())
#         # else:
        
#         #do not need to consider samples with a weight of zero (below the cutoff)
#         cutoff_idx = np.where(weights < opt_config['cutoff'])
#         Xt = np.delete(Xt, cutoff_idx, axis=0)
#         yt = np.delete(yt, cutoff_idx, axis=0)
#         weights = np.delete(weights, cutoff_idx, axis=0)

#         #keep the weights from the last iteration
#         vae = start_training(vae, Xt, save_path, data_config, vae_model_config, vae_train_config, device, weights)

#         print(evaluate(torch.tensor(Xt).float(), vae, device))

    
#     max_dict = {'oracle_max' : oracle_max, 
#                 'oracle_max_seq': oracle_max_seq,
#                 'diversity_max': div_max}
    
#     return traj, oracle_samples, max_dict
