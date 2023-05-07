from multiprocessing import Pool
import numpy as np
import pandas as pd
import random
from scipy.spatial import distance
from .encoding_utils import *
from .seqtools import *

def hammingdistance(seq1, seq2):
    """
    Returns the hamming distance between two sequences of equal length
    """
    return sum(s1 != s2 for s1, s2 in zip(seq1, seq2))

def manhattandistance(a, b):
    """
    Returns the manhattan distance between two sequences of equal length
    """
    return np.sum(np.absolute(a-b))

def euclideandistance(a, b):
    """
    Returns the euclidean distance between two sequences of equal length
    """
    return distance.euclidean(a, b)

dist_function_dict = {
    'hamming' : hammingdistance,
    'manhattan' : manhattandistance,
    'euclidean' : euclideandistance
}

class Oracle():
    """
    Maps from a degenerate codon library to the values provided by the predictive model.
    """
    def __init__(self, data_config: dict, opt_config: dict, verbose=False):
        """
        Args:
            data_config: dictionary of data configuration
            opt_config: dictionary of optimization configuration
        """
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

        if 'full' in self.weight_type:
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
            embeddings = np.load('data/ESM2_all.npy')
            self.embdict = dict(zip(df_unsorted["Combo"].values, embeddings))
        elif 'ESM1b' in self.weight_type:
            embeddings = np.load('data/ESM1b_all.npy')    
            self.embdict = dict(zip(df_unsorted["Combo"].values, embeddings))
        elif 'MSA_transformer' in self.weight_type:
            embeddings = np.load('data/MSA_transformer.npy')
            self.embdict = dict(zip(df_unsorted["Combo"].values, embeddings))
        elif 'georgiev' in self.weight_type:
            embeddings = np.load('data/georgiev_all.npy')
            self.embdict = dict(zip(df_unsorted["Combo"].values, embeddings))
        
    def encoding2aas(self, encoding_list: np.ndarray, seed: int, n_samples = 0) -> np.ndarray:
        """
        converts a numerical encoding of a mixed base library (or a set of multiple mixed base libraries) into a sampling of protein sequences
        Args:
            encoding_list: a numpy array of shape (n_sites * 12, n_mix) where n is the number of degenerate codon libraries, n_sites is the number of amino acid sites, and n_mix is the number of templates per library
            seed: the seed for the random number generator
            n_samples: the number of samples to take from each mixed base library, 0 means take the default number of samples during training
        Returns:
            a numpy array of shape (repeats, n_samples) where repeats is the number of repeated sampling desired
        """
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

    def aas2zs(self, aaseqs: np.ndarray) -> tuple:
        """
        map all protein sequences to their corresponding zero shot scores and
        report stats about the distribution
        Args:
            aaseqs: a numpy array of samples amino acid sequences
        Output:
            Tuple containing stats about the sampled sequences
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

    def predict(self, encodings: np.ndarray): 
        """
        Runs the oracle
        Args:
            encodings: array of size [n_libraries x  12 * number of sites x n_mix] corresponding to all degenerate codon libraries being optimized
        Outputs: 
            Tuple containing the results of a Oracle prediction
        """
        self.encodings = encodings
        self.batch_size = encodings.shape[0]
        
        #run the repeated encoding calculations in parallel 
        results = np.zeros((self.batch_size, self.n_repeats, 5))
        all_all_seqs = np.full((self.batch_size, self.n_repeats, self.n_samples), 'VDGV')
        
        with Pool(self.num_workers) as p:
            for i, (result, all_seqs) in enumerate(p.map(self.predictor_all, list(self.encodings))):
                 results[i,:,:] = result
                 all_all_seqs[i,:,:] = all_seqs
            
        
        means = np.mean(results, axis = 1)
        vars =  np.var(results, axis = 1)

        if self.verbose:
            return means, vars, all_all_seqs
        else:
            return means, vars


    def predictor_all(self, encoding: np.ndarray):
        """
        Passes a single degenerate codon library through the oracle.
        Args:
            array of size [12 * number of sites x n_mix] corresponding to a single degenerate codon library being optimized
        """
        results = np.zeros((self.n_repeats, 5))
        all_seqs = np.full((self.n_repeats, self.n_samples), 'VDGV')

        all_aaseqs = self.encoding2aas(encoding, seed=self.seed)

        for i, aaseqs in enumerate(all_aaseqs):
            output = self.aas2zs(aaseqs)
            results[i, :] = output[:5]
            all_seqs[i, :] = np.array(output[5], dtype=str).reshape(1, -1)
    
        return results, all_seqs
    
    def get_coverage(self, set: list, sigma=0.4):
        """
        calculates how well all_seqs (dictionary mapping strings to weights) is covered by set (a list of sequences)
        Args:
            set: a list of sequences
            sigma: the sigma parameter for the coverage function
        Output:
            Tuple containing the coverage of the set and the unweighted coverage of the set
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