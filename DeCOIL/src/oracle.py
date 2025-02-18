from multiprocessing import Pool
import numpy as np
import pandas as pd
import random
from scipy.spatial import distance
from .encoding_utils import *
from .seqtools import *
from .acquisition import Acquisition

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
        self.model_path = opt_config["model_path"]

        if 'full' in self.weight_type:
            self.sigma = opt_config["sigma"]
            self.dist_function = opt_config["dist_function"]
        
        self.samples_dict = {}
    
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
        all_aaseqs = np.full((repeats, n_samples), 'AAAAAAAA') #filler
       

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
        all_seqs = np.full((self.n_repeats, self.n_samples), 'AAAAAAAA')

        all_aaseqs = self.encoding2aas(encoding, seed=self.seed)
        acquisition = Acquisition(self.model_path)

        for i, aaseqs in enumerate(all_aaseqs):
            #TODO: process aaseqs into the correct batch input
            output = acquisition(aaseqs)
            results[i, :] = output[:5]
            all_seqs[i, :] = np.array(output[5], dtype=str).reshape(1, -1)
    
        return results, all_seqs
    
    