import json
import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from .models import *
from .Datasets import Dataset

from DeCOIL.src.oracle import Oracle
from DeCOIL.src.encoding_utils import *

def ndcg(y_true, y_pred):
    y_true_normalized = y_true - min(y_true)
    return ndcg_score(y_true_normalized.reshape(1, -1), y_pred.reshape(1, -1))

class MLDESim():
    """Class for training and evaluating MLDE models."""
    def __init__(self, save_path: str, encoding: str, model_class: str, n_samples: int, model_config: dict, data_config: dict, train_config: dict) -> None:
        """
        Args:
            save_path : path to save results
            encoding : encoding type
            model_class : model class
            n_samples : number of samples to train on
            model_config : model configuration
            data_config : data configuration
            train_config : training configuration
        """
        self.data_config = data_config
        self.train_config = train_config
        self.model_config = model_config
        self.save_path = save_path
        self.num_workers = train_config['num_workers']

        self.n_splits = train_config['n_splits']
        self.n_subsets = train_config['n_subsets']
        self.n_samples = n_samples

        self.n_solutions = data_config['n_solutions']
        self.library = data_config['library']
 
        self.save_model = False
        if "save_model" in train_config:
            self.save_model = train_config["save_model"]    

        self.model_class = model_class

        #string refers to optimized libraries
        if isinstance(self.library, str):
            self.dclibrary = True

            if '.npy' in self.library:
                self.final_encodings = np.load('dclo/saved/' + self.library, allow_pickle=True)
                with open('optimization/configs/defaults/DEFAULT.json', 'r') as f:
                    config = json.load(f)

                self.oracle = Oracle(config['data_config'], config['opt_config'], verbose = True)
            else:
                opt_results = np.load('optimization/saved/' + self.library + '/results.npy', allow_pickle=True)
                Xts = opt_results.item()['Xts']
                self.final_encodings = Xts[-1, :, :, :]

                with open('optimization/saved/' + self.library + '/' + self.library + '.json', 'r') as f:
                    config = json.load(f)

                self.oracle = Oracle(config['data_config'], config['opt_config'], verbose = True)
        
        #list of strings refers to list of encodings
        elif isinstance(self.library[0], str):
            self.n_sites = int(len(self.library[0])/3)
            self.dclibrary = True

            self.final_encodings = np.zeros((len(self.library), self.n_sites*12, 1))
            for i, seq in enumerate(self.library):
                self.final_encodings[i] = seq2encoding(seq).T

            with open('optimization/configs/defaults/DEFAULT.json', 'r') as f:
                config = json.load(f)

            self.oracle = Oracle(config['data_config'], config['opt_config'], verbose = True)
        
            assert self.n_solutions == self.final_encodings.shape[0]
        #list of numbers refers to random sampling cutoff
        else:
            self.dclibrary = False
            assert self.n_solutions == len(self.library)

        self.top_seqs = np.full((self.n_solutions, self.n_subsets, 500), 'VDGV')
        self.ndcgs = np.zeros((self.n_solutions, self.n_subsets))
        self.maxes = np.zeros((self.n_solutions, self.n_subsets))
        self.means = np.zeros((self.n_solutions, self.n_subsets))
        self.unique = np.zeros((self.n_solutions, self.n_subsets))
        self.labelled = np.zeros((self.n_solutions, self.n_subsets))

        # Sample and fix a random seed if not set in train_config
        self.seed = train_config["seed"]
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.fitness_df = pd.read_csv('data/' + data_config['name'])
        self.dataset = Dataset(dataframe = self.fitness_df)
        
        self.dataset.encode_X(encoding = encoding)

        self.X_train_all = np.array(self.dataset.X)
        #np.save('/home/jyang4/repos/DeCOIL/one_hot.npy', self.X_train_all)

        self.y_train_all = np.array(self.dataset.y)
        self.y_preds_all = np.zeros((self.dataset.N, self.n_subsets, self.n_splits))
        
        self.all_combos = self.dataset.all_combos
        self.n_sites = self.dataset.n_residues
        
    def train_all(self):
        '''
        Loops through all libraries to be sampled from (n_solutions) and for each solution trains n_subsets of models. Each model is an ensemble of n_splits models, each trained on 90% of the subset selected randomly.

        Output: results for each of the models
        '''
        with tqdm() as pbar:
            pbar.reset(self.n_solutions * self.n_subsets * self.n_splits)
            pbar.set_description('Training and evaluating')

            for k in range(self.n_solutions):
                
                if self.dclibrary == False:
                    cutoff = self.library[k]
                else:
                    sequences = self.final_encodings[k,:,:]     

                for j in range(self.n_subsets):
                    #need to check if the seeding process works the same way
                    
                    if self.dclibrary == True:
                        seqs = self.oracle.encoding2aas(sequences, self.seed + (k*self.n_subsets+j), self.n_samples).T
                        seqs = seqs.squeeze()
                    else:
                        seqs = self.dataset.sample_top(cutoff, self.n_samples, self.seed + (k*self.n_subsets+j))

                    uniques = np.unique(seqs)
                    self.unique[k, j] = len(uniques)
                    mask = self.dataset.get_mask(uniques)
                    self.labelled[k, j] = len(mask)
                    combos_train = []

                    if self.save_model:
                        save_dir = os.path.join(self.save_path, str(k), str(j))
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                    for i in range(self.n_splits):
                        if self.n_splits > 1:
                            #boostrap ensembling with 90% of the data
                            train_mask, validation_mask = train_test_split(mask, test_size=0.1, random_state=i)
                        else:
                            train_mask = mask
                            validation_mask = mask #used for validation if desired

                        X_train = self.X_train_all[train_mask]
                        y_train = self.y_train_all[train_mask]
                        combos_train += list(self.all_combos[train_mask])
                        X_validation = self.X_train_all[validation_mask]
                        y_validation = self.y_train_all[validation_mask]
                        y_preds, clf = self.train_single(X_train, y_train, X_validation, y_validation)

                        #remove this if you don't want to save the model
                        if self.save_model:
                            filename = 'split' + str(i) + '.model' 
                            clf.save_model(os.path.join(save_dir, filename))

                        self.y_preds_all[:, j, i] = y_preds
                        pbar.update()
                    
                    #need to redo some of these, loop was in wrong place? but it should be fine cause it gets replaced to the correct mean at the end
                    means = np.mean(self.y_preds_all, axis = 2)
                    y_preds = means[:, j]

                    self.maxes[k, j], self.means[k, j], self.top_seqs[k, j, :] = self.get_mlde_results(self.dataset.data, y_preds, uniques)
                    ndcg_value = ndcg(self.y_train_all, y_preds)
                    self.ndcgs[k, j] = ndcg_value
                    
        pbar.close 

        return self.top_seqs, self.maxes, self.means, self.ndcgs, self.unique, self.labelled
        
    def train_single(self, X_train: np.ndarray, y_train: np.ndarray, X_validation: np.ndarray, y_validation: np.ndarray):
        '''
        Trains a single supervised ML model. Returns the predictions on the training set and the trained model.
        '''
        if self.model_class == 'boosting':
            clf = get_model(
            self.model_class,
            model_kwargs={'nthread': self.num_workers})
            eval_set = [(X_validation, y_validation)]
            clf.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            clf = get_model(
            self.model_class,
            model_kwargs={})
            clf.fit(X_train, y_train)

        y_preds = clf.predict(self.X_train_all)
        
        return y_preds, clf
    
    def get_mlde_results(self, data2: pd.DataFrame, y_preds: np.ndarray, unique_seqs: list) -> tuple:
        """
        Calculates the MLDE results for a given set of predictions. Returns the max and mean of the top 96 sequences and the top 500 sequences.
        Args:
            data2: pandas dataframe with all sequences and fitness labels in the combinatorial space
            y_preds: the predictions on the training data
            unique_seqs: the unique sequences in the training data
        """
        data2['y_preds'] = y_preds
        
        ##optionally filter out the sequences in the training set
        #data2 = data2[~data2['Combo'].isin(unique_seqs)]

        sorted  = data2.sort_values(by=['y_preds'], ascending = False)

        top = sorted.iloc[:96,:]['fit']
        max = np.max(top)
        mean = np.mean(top)

        #save the top 500
        top_seqs = sorted.iloc[:500,:]['Combo'].values

        ##for checking how many predictions are in the training set
        #top_seqs_96 = sorted.iloc[:96,:]['Combo'].values
        #print(len(np.intersect1d(np.array(unique_seqs), top_seqs_96)))

        return max, mean, top_seqs