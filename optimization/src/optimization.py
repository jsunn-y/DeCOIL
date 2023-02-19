import numpy as np
from tqdm.auto import tqdm
from .oracle import Oracle
from .util import *
import time

def run_greedy(save_path, data_config, opt_config, verbose=True):
    
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
                #load previously libraries
                if isinstance(data_config['library'], str):
                    #saved in encoding as npy
                    if '.npy' in data_config['library']:
                        Xt = np.load('optimization/saved/' + data_config['library'], allow_pickle=True)
                    #saved in the results file of a previous campaign
                    else:
                        results = np.load('optimization/saved/' + data_config["library"] + '/results.npy', allow_pickle=True)
                        Xts = results.item()['Xts']
                        Xt = Xts[-1, :, :]

                #load manual list of letters as a starting point
                else:
                    Xt = np.zeros((len(data_config['library']), sites*12, 1))
                    for i, seq in enumerate(data_config['library']):
                        Xt[i] = seq2encoding(seq).T
                
                assert Xt.shape[0] == opt_config['samples']
                assert Xt.shape[2] == opt_config['n_mix']
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
                print("%s --- Weighted: %.3f, Unweighted: %.3f, Raw Weighted Simple: %.3f, Top Counts: %3.0f, Unique: %3.0f" % (encoding2seq(row), score, unweighted_score, raw_score, counts, diversity))

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

                        all_means[:, :, k, j+1], all_vars[:, :, k, j+1] = oracle.predict(Xt_new)
                        pbar.update()

            pbar.close 

            all_yt = all_means[:, 0, :, :]
            all_yt = all_yt.reshape(all_yt.shape[0], -1) 
            all_yt_max_idx = np.argmax(all_yt, axis=1)
    
    results = {'means': traj_means, 'vars': traj_vars, 'Xts': traj_Xts}
    np.save(save_path + '/results.npy', results)
    
    end = time.time()
    print('Time: ' + str(end-start))

    return results