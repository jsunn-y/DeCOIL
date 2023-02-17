from multiprocessing import Pool
import numpy as np
import pandas as pd
import random
from .encoding_utils import *
from .seqtools import *

dist_function_dict = {
    'hamming' : hammingdistance,
    'manhattan' : manhattandistance,
    'euclidean' : euclideandistance
}

class Oracle():
    """Maps from a degenerate mixed base library to the zs score distribtuion. Maintains mappings that have already been calculated."""
    def __init__(self, data_config, opt_config, verbose=False):
        self.opt_config = opt_config
        self.seed = opt_config["seed"]
        self.verbose = verbose
        self.mappings = {}
        self.weight_type = opt_config["weight_type"]
        self.n_mix = opt_config["n_mix"]
        self.n_samples = data_config["samples"]
        self.sites = data_config["sites"]
        self.n_repeats = opt_config["num_repeats"]
        self.num_workers = opt_config["num_workers"]

        if 'full_coverage' in self.weight_type:
            self.sigma = opt_config["sigma"]
            self.dist_function = opt_config["dist_function"]
        
        self.samples_dict = {}

        df_unsorted = pd.read_csv("data/" + data_config['name'])

        #normalized ranking
        zs_name = data_config['zs_name']

        df = df_unsorted.sort_values(zs_name)

        #smaller value score is better
        df['ranked_zs'] = df[zs_name].rank(ascending=False)/len(df)
        self.cutoff = 1- opt_config['top_fraction']

        df_top = df.iloc[:16000]
        self.avg_zs = np.mean(df['ranked_zs'].values)
        
        #missing combos are now filled in 
        self.combo2zs_dict = dict(zip(df["Combo"].values, df["ranked_zs"].values))
        self.combo2zs_dict_top = dict(zip(df_top["Combo"].values, df_top["ranked_zs"].values))
        
        self.dict = self.combo2zs_dict
        self.exp_dict = dict(zip(df["Combo"].values, np.power(df["ranked_zs"].values, self.opt_config["zs_exp"])))

        if 'ESM2' in self.weight_type:
            embeddings = np.load('data/GB1/ESM2_all.npy')
            self.embdict = dict(zip(df_unsorted["Combo"].values, embeddings))
        elif 'ESM1b' in self.weight_type:
            embeddings = np.load('data/GB1/ESM1b_all.npy')
            self.embdict = dict(zip(df_unsorted["Combo"].values, embeddings))
        elif 'MSA_transformer' in self.weight_type:
            embeddings = np.load('data/GB1/MSA_transformer.npy')
            self.embdict = dict(zip(df_unsorted["Combo"].values, embeddings))
        
    def encoding2aas(self, encoding_list, seed, n_samples = 0):
        '''
        converts a numerical encoding of a mixed base library (or a set of multiple mixed base libraries) into a sampling of protein sequences
        '''
        
        if n_samples == 0: #default value, for training
            n_samples = self.n_samples
            repeats = self.n_repeats
        else: #non default value, for sampling afterward only
            repeats = 1

        n_samples_each = int(n_samples/encoding_list.shape[1])
        n_samples_each_all = n_samples_each*repeats
        all_aaseqs = np.full((repeats, n_samples), 'VDGV')
       
        for k, encoding in enumerate(encoding_list.T):
            encoding = encoding.reshape((self.sites, 12))
            choices = []
            random.seed(seed)

            for j, row in enumerate(encoding):
                #aaprobs_dict = self.full_dict[tuple(row)]
                #row_tuple = tuple(row)
                #if row_tuple in self.samples_dict.keys():
                    #print(current_process().name)
                    #choices.append(self.samples_dict[row_tuple]) #take based on the multiprocessing ID (need to make sure this distributes across all 80)
                    #pass
                #else:
                aaprobs_dict = mixedcodon2aaprobs(row)
                choices.append(random.choices(list(aaprobs_dict.keys()), weights=aaprobs_dict.values(), k=n_samples_each_all))

            aaseqs=[]
            for i in range(n_samples_each_all):
                aaseqs.append(choices[0][i] + choices[1][i] + choices[2][i] + choices[3][i])
            # aaseqs = np.apply_along_axis(''.join, 0, choices)
            #print(len(aaseqs))

            aaseqs = np.array(aaseqs).reshape((repeats, n_samples_each))
            all_aaseqs[:, k*n_samples_each : (k+1)*(n_samples_each)] = aaseqs

        return all_aaseqs

    def aas2zs(self, aaseqs):
        """
        map all protein sequences to their corresponding zero shot scores
        report stats about the distribution
        """
        uniques = []
        scores = []
        raw_scores = []

        if 'full' in self.weight_type:
            uniques = np.unique(aaseqs)
            uniques2 = [seq for seq in uniques if '*' not in seq]

            #loop through the sequences to be covered
            total_coverage, total_unweighted_coverage = self.get_coverage(uniques2, sigma=self.sigma)
            
            score_avg = total_coverage/self.n_samples
            unweighted_score_avg = total_unweighted_coverage/self.n_samples

            diversity = len(uniques2)
            uniques2 = np.array([self.dict[aa] for aa in uniques2])
            
            counts = len(uniques2[uniques2 > self.cutoff])
            raw_simple_score_avg = sum(uniques2)/self.n_samples
        else:
            for seq in aaseqs:
                if seq in self.dict.keys():
                    raw_score = self.dict[seq]
                    score = raw_score
                    #for softening the effect of zs score (don't wnat too close to the bottom or top)

                    score = self.exp_dict[seq]
                    
                    #sequence
                    if seq not in uniques:
                        uniques.append(seq)
                        scores.append(score)
                        raw_scores.append(raw_score)
                    #duplicate score
                    else:
                        uniques.append('-')
                        scores.append(0)
                        raw_scores.append(0)
                #assign the minimum for a stop codon
                elif '*' in seq:
                    uniques.append('-')
                    scores.append(0)
                    raw_scores.append(0)
                else:
                    print('Missing ZS score')
                    uniques.append('-')
                    scores.append(self.avg_zs)

            scores = np.array(scores)
            raw_scores = np.array(raw_scores)
            uniques = np.array(uniques)
            indices = np.argwhere(uniques != '-')

            score_avg = np.sum(scores)/self.n_samples
            raw_simple_score_avg = np.sum(raw_scores)/self.n_samples

            unique_scores = raw_scores[indices]
            diversity = len(indices)
            unweighted_score_avg = diversity/self.n_samples
            counts = len(unique_scores[unique_scores > self.cutoff])
        
        return score_avg, unweighted_score_avg, raw_simple_score_avg, counts, diversity, aaseqs
    
    # def encoding2zs(self, encoding):
    #     all_aas = self.encoding2aas(encoding)
    #     return self.aas2zs()

    def sample(self, encodings, n_samples, seed):
        '''
        samples amino acid sequences from a library encoding
        '''
        return self.encoding2aas(encodings, seed=seed, n_samples=n_samples)

    def predict(self, encodings): 
        '''
        Runs the oracle

        Arguments:
            encodings: array of size [batch_size x  12 * number of sites x n_mix]

        Outputs: the results of a single oracle prediction
        '''
        self.encodings = encodings
        self.batch_size = encodings.shape[0]
        
        #run the repeated encoding calculations in parallel 
        results = np.zeros((self.batch_size, self.n_repeats, 5))
        all_all_seqs = np.full((self.batch_size, self.n_repeats, self.n_samples), 'VDGV')
        
        with Pool(self.num_workers) as p:
            
            #all_results = p.map(self.predictor_all, [encodings]*self.repeats)

            for i, (result, all_seqs) in enumerate(p.map(self.predictor_all, list(self.encodings))):
                 results[i,:,:] = result
                 #print(result)
                 all_all_seqs[i,:,:] = all_seqs
            
            #could make this more efficient instead of unraveling all the strings
            # for i, encoding in enumerate(encodings):
            #     encoding = encoding.reshape((self.sites, 12))
            #     for j, row in enumerate(encoding):
            #         row = tuple(row)
            #         if row not in self.samples_dict.keys():
            #             self.samples_dict[row] = list(np.vectorize(lambda s: s[j])(all_all_seqs[i, :, :]).flatten())
        
        means = np.mean(results, axis = 1)
        vars =  np.var(results, axis = 1)

        #all_results = np.zeros((self.batch_size, 2, self.repeats))

        ### Other option is to run each row in parallel ###
        #probably less efficient because each row is so quick
        # with Pool(self.num_workers) as p:
        #      all_results = p.map(self.predictor, [row for row in encodings])
        
        # all_results = np.array(all_results)
        if self.verbose:
            return means, vars, all_all_seqs
        else:
            return means, vars

    # def predictor(self, encoding):
    #     return self.encoding2zs(encoding)

    def predictor_all(self, encoding):
        '''
        passes a given encoding (or mix of encodings) through the oracle
        '''
        results = np.zeros((self.n_repeats, 5))
        all_seqs = np.full((self.n_repeats, self.n_samples), 'VDGV')

        #print(encoding.shape)
        all_aaseqs = self.encoding2aas(encoding, seed=self.seed)

        for i, aaseqs in enumerate(all_aaseqs):
            output = self.aas2zs(aaseqs)
            results[i, :] = output[:5]
            all_seqs[i, :] = np.array(output[5], dtype=str).reshape(1, -1)
    
        return results, all_seqs
            #if encoding2seq(row) == 'WSHYYHMSWNYS':
            #    exit()
    
    def get_coverage(self, set, sigma=0.4):
        """
        calculates how well all_seqs (dictionary mapping strings to weights) is covered by set (a list of sequences)
        """
        total_coverage = 0
        total_unweighted_coverage = 0

        for seq, weight in self.exp_dict.items():
            distances = []
            if self.dist_function == 'manhattan' or self.dist_function == 'euclidean':
                emb = self.embdict[seq]
                
            for aseq in set:
                if self.dist_function == 'hamming':
                    distances.append(dist_function_dict[self.dist_function](seq, aseq))
                elif self.dist_function == 'manhattan' or self.dist_function == 'euclidean':
                    aemb = self.embdict[aseq]
                    distances.append(dist_function_dict[self.dist_function](emb, aemb))

            distances = np.array(distances)

            coverage = 1 - np.prod(1 - np.exp(-distances/sigma))

            total_coverage += weight * coverage
            total_unweighted_coverage += coverage
        return total_coverage, total_unweighted_coverage