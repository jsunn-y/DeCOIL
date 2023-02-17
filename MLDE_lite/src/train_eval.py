
import json
import os
import random
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils import resample
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
#import matplotlib.pyplot as plt
#import seaborn as sns
from .models import *

#comment these out if not using any degenerate codon libraries
from dclo.src.oracle import Oracle
from ProtGraphR.src.Datasets import *
from dclo.src.encoding_utils import *

#comment these out if not using graphs
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Subset
from torch_geometric.loader import DataLoader
from ProtGraphR.src.Datasets import *
from ProtGraphR.src.load_dataset import load_dataset
from ProtGraphR.src.model import *

def ndcg(y_true, y_pred):
    y_true_normalized = y_true - min(y_true)
    return ndcg_score(y_true_normalized.reshape(1, -1), y_pred.reshape(1, -1))

dataset_dict = {
    'GB1' : GB1Dataset,
    'TrpB' : TrpBDataset,
}

class MLDESim():
    def __init__(self, save_path, encoding, model_class, n_samples, model_config, data_config, train_config, device) -> None:
        self.data_config = data_config
        self.train_config = train_config
        self.model_config = model_config
        self.save_path = save_path

        self.protein = data_config["protein"]
        if self.protein == 'GB1':
            self.sites = 4

        self.n_splits = train_config['n_splits']
        self.n_subsets = train_config['n_subsets']
        self.n_samples = n_samples

        self.n_solutions = data_config['n_solutions']
        self.library = data_config['library']
        #self.n_mix = data_config['n_mix']

        self.filter_AAs = False
        if "filter_AAs" in train_config:
            self.filter_AAs = train_config["filter_AAs"]  
        self.save_model = False
        if "save_model" in train_config:
            self.save_model = train_config["save_model"]    

        #optionally return the input seqs (only works for 384 samples)
        if "input_seqs" in train_config:
            self.inputs = train_config["input_seqs"]
            # if self.inputs == True:
            #     with open('/home/jyang4/repos/StARDUST/dclo/configs/coverage_DEFAULT.json', 'r') as f:
            #         config = json.load(f)
            #     self.coverage_oracle = Oracle(config['data_config'], config['opt_config'], verbose = True)

        self.model_class = model_class

        #string refers to optimized libraries
        if isinstance(self.library, str):
            self.dclibrary = True

            if '.npy' in self.library:
                
                self.final_encodings = np.load('dclo/saved/' + self.library, allow_pickle=True)
                with open('/home/jyang4/repos/StARDUST/dclo/configs/DEFAULT.json', 'r') as f:
                    config = json.load(f)

                self.oracle = Oracle(config['data_config'], config['opt_config'], verbose = True)
            else:
                opt_results = np.load('dclo/saved/' + self.library + '/results.npy', allow_pickle=True)
                Xts = opt_results.item()['Xts']
                self.final_encodings = Xts[-1, :, :, :]

                with open('/home/jyang4/repos/StARDUST/dclo/saved/' + self.library + '/' + self.library + '.json', 'r') as f:
                    config = json.load(f)

                self.oracle = Oracle(config['data_config'], config['opt_config'], verbose = True)
        #list of strings refers to list of encodings
        elif isinstance(self.library[0], str):
            self.dclibrary = True
            self.final_encodings = np.zeros((len(self.library), self.sites*12, 1))
            for i, seq in enumerate(self.library):
                self.final_encodings[i] = seq2encoding(seq).T
            #could make the sampling and coversion not require the oracle parameters
            #should really exist outside the class
            with open('/home/jyang4/repos/StARDUST/dclo/configs/DEFAULT.json', 'r') as f:
                config = json.load(f)

            self.oracle = Oracle(config['data_config'], config['opt_config'], verbose = True)
        #list of numbers refers to random sampling cutoff
        else:
            self.dclibrary = False
        
        self.top_seqs = np.full((self.n_solutions, self.n_subsets, 500), 'VDGV')
        self.ndcgs = np.zeros((self.n_solutions, self.n_subsets))
        self.maxes = np.zeros((self.n_solutions, self.n_subsets))
        self.means = np.zeros((self.n_solutions, self.n_subsets))
        self.unique = np.zeros((self.n_solutions, self.n_subsets))
        self.labelled = np.zeros((self.n_solutions, self.n_subsets))
        self.input_seqs = np.full((self.n_solutions, self.n_subsets, n_samples), 'VDGV')
        #self.coverage_results = np.zeros((self.n_solutions, self.n_subsets))

        # Sample and fix a random seed if not set in train_config
        self.seed = train_config["seed"]
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.protein == "GB1":
            self.fitness_df = pd.read_csv('/home/jyang4/repos/StARDUST/data/GB1/fitness.csv')
        elif self.protein == "TrpB":
            self.fitness_df = pd.read_csv('/home/jyang4/repos/StARDUST/data/tm9d8s/processed/' + data_config["data_name"][:4] + '_processed.csv')

        if model_class == 'gnn':
            #only works for GB1 for now
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            self.dataset, self.model_config = load_dataset(encoding, data_config, self.model_config)
            self.data_loader = DataLoader(self.dataset, batch_size=train_config['batch_size'], num_workers=train_config['num_workers'], shuffle=False)
            self.device = device
            self.y_train_all = self.fitness_df['fit'].values
        else:
            #could make this better by appending a keyword arg instead of requiring it
            zs_score = "Triad-FixedBb-dG" #name of ZS score
            if "zs_scores" in data_config:
                zs_score = data_config["zs_score"]
            self.dataset = dataset_dict[self.protein](dataframe = self.fitness_df, encoding = encoding, zs_score = zs_score)

            os.environ["CUDA_VISIBLE_DEVICES"] = str('/physical_device:GPU:' + str(device))
            #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                tf.config.set_visible_devices(gpus[device], 'GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu,True)

            if model_class == 'cnn':
                self.dataset.encode_X(flatten=False)
            else: 
                self.dataset.encode_X(flatten=True)

            self.X_train_all = np.array(self.dataset.X)
            self.all_combos = self.dataset.data['Combo'].values
            self.y_train_all = np.array(self.dataset.y)
        
        self.y_preds_all = np.zeros((len(self.dataset), self.n_subsets, self.n_splits)) 
        
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
                    
                    #assume there are always 100 solutions
                    # np.random.seed(self.seed + k)
                    # choices = np.random.choice(100, self.n_mix, replace=False)
                    # print(choices)
                    # sequences = [self.final_encodings[choice,:] for choice in choices]        

                for j in range(self.n_subsets):
                    #need to check if the seeding process works the same way
                    #X_resample, y_resample = resample(X_train_all, y_train_all, n_samples=n_samples, random_state = seed + j)
                    
                    if self.dclibrary == True:
                        seqs = self.oracle.sample(sequences, self.n_samples, self.seed + (k*self.n_subsets+j)).T
                        seqs = seqs.squeeze()
                    else:
                        seqs = self.dataset.sample_top(cutoff, self.n_samples, self.seed + (k*self.n_subsets+j))

                    uniques = np.unique(seqs)
                    self.unique[k, j] = len(uniques)
                    mask = self.dataset.get_mask(uniques)
                    self.labelled[k, j] = len(mask)
                    combos_train = []

                    # if self.coverage == True:
                    #     self.coverage_results[k, j] = self.coverage_oracle.get_coverage(uniques)
                    if self.inputs == True:
                        self.input_seqs[k, j, :] = np.array(uniques)

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

                        if self.model_class == 'gnn':
                            train_dataset = Subset(self.dataset, train_mask)
                            train_loader = DataLoader(train_dataset, batch_size=self.train_config['batch_size'], num_workers=self.train_config['num_workers'], shuffle=True)

                            y_preds = self.train_gnn(train_loader)
                        else:
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

                    self.maxes[k, j], self.means[k, j], self.top_seqs[k, j, :] = self.get_mlde_results(self.dataset.data, y_preds, combos_train)

                    #manually override the top seqs here
                    

                    #print(self.maxes[k, j])
                    ndcg_value = ndcg(self.y_train_all, y_preds)
                    self.ndcgs[k, j] = ndcg_value
                    #consider saving the model here
                    #print(f'{ndcg_value}')
                    #print(top_seqs[k, j, :])
                    
        pbar.close 
            #do not save the predictions, takes up too much storage (maybe only save the top 384 for the mean)
            #np.save(save_path + '/predictions.npy', y_preds_all)
        return self.top_seqs, self.maxes, self.means, self.ndcgs, self.unique, self.labelled, self.input_seqs
        
    def train_single(self, X_train, y_train, X_validation, y_validation):
        '''
        Trains a single supervised ML model (everything in scikit and tensorflow).
        '''
        if False:
        # if self.model_class == 'boosting':
            # model_params = {"booster": "gbtree",
            #     "tree_method": "exact",
            #     "nthread": 1,
            #     "objective": "reg:tweedie",
            #     "tweedie_variance_power": 1.5,
            #     "eval_metric": "tweedie-nloglik@1.5",
            #     "eta": 0.3,
            #         "max_depth": 6,
            #         "lambda": 1,
            #         "alpha": 0
            #     }
            # train_matrix = xgb.DMatrix(X_train, label = y_train)

            # # in reality we should use a validation set, not the test set
            # test_matrix = xgb.DMatrix(X_validation, label = y_validation)
            # evallist = [(train_matrix, "train"), (test_matrix, "test")]
            # bst = xgb.train(model_params, train_matrix, num_boost_round=1000, early_stopping_rounds=10, evals=evallist, verbose_eval=False)
            # best = bst.best_iteration

            # y_preds = bst.predict(xgb.DMatrix(self.X_train_all), iteration_range=(0,best))
            # #bst.save_model(save_path + f'/subset{j:02d}_split{i:02d}.json')
            pass
        else:
            sequence_length = X_train.shape[1] #number of sites
            vocab_size = num_tokens
            
            # model_kwargs = dict(cnn_num_filters=2,
            #                     cnn_kernel_size=1,
            #                     cnn_hidden_size=12,
            #                     cnn_batch_size=1,
            #                     cnn_num_epochs=1)
            
            #could clean this up a lot
            clf, flatten_inputs = get_model(
                self.model_class,
                sequence_length,
                vocab_size,
                model_kwargs={})

            if self.model_class == 'boosting':
                eval_set = [(X_validation, y_validation)]
                clf.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            else:
                clf.fit(X_train, y_train)

            #clfs.append(clf)
            y_preds = clf.predict(self.X_train_all)
            #filename = save_path + f'/subset{j:02d}_split{i:02d}.sav'
            #pickle.dump(clf, open(filename, 'wb'))
        
        return y_preds, clf

    def train_gnn(self, train_loader):
        '''
        Trains a single supervised GNN model (pytorch).
        '''
        model = SupervisedGNN(model_config=self.model_config).to(self.device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr = self.train_config['learning_rate'])

        #reduce the learning rate of the optimizer based on the training loss
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, verbose=True)

        # Start training
        pbar = tqdm()
        pbar.reset(len(train_loader)*self.train_config['num_epochs'])
        pbar.set_description('Training')

        for epoch in range(1, 1 + self.train_config['num_epochs']):
            loss = self.__gnn_one_epoch(model, self.device, train_loader, optimizer, pbar)
            scheduler.step(loss)

            if self.train_config['verbose']:
                tqdm.write(f'Epoch {epoch:02d}, loss {loss:.4f}')

        #evaluate the model
        #for now this is fine but may want to build a function with batching to be more efficient
        model.eval()
        ys = np.array([])
        y_preds = np.array([])

        pbar = tqdm()
        pbar.reset(total=len(self.data_loader))
        pbar.set_description('Evaluating')

        for step, batch in enumerate(self.data_loader):
            batch = batch.to(self.device)

            with torch.no_grad():
                y_pred = model(data=batch)
                y_pred = y_pred.cpu()
                y = batch.y.cpu()

            if y_preds.shape[0] == 0:
                y_preds = y_pred
                ys = y
            else:
                y_preds = np.concatenate([y_preds, y_pred], axis=0)
                #breakpoint()
                ys = np.concatenate([ys, y], axis=0)
            pbar.update()

        #save the model
        # torch.save(model.state_dict(), save_path + f'/subset{j:02d}_split{i:02d}.pt')
        return y_preds

    def __gnn_one_epoch(self, model, device, data_loader, optimizer, pbar):
        """Trains a GNN model for one epoch.

        Args
        - model: nn.Module, GNN model, already placed on device
        - device: torch.device
        - data_loader: pyg.loader.DataLoader
        - optimizer: torch.optim.Optimizer

        Returns: loss
        - avg_loss: float, avg loss across epoch
        """
        model.train()
        total_loss = 0
        total = 0

        # enumerate through both dataloaders
        # the first dataloader contains the graph object and the second contains the weakly supervised labels
        for step, batch in enumerate(data_loader):
            total += batch.num_graphs
            #batch.y = torch.from_numpy(np.stack(batch.y)[:, 0].astype(np.float32))
            batch.y = torch.from_numpy(np.stack(batch.y).astype(np.float32))
            batch = batch.to(device, non_blocking=True)

            y_pred = model(data=batch)
            loss = model.loss(y_pred, batch.y)
            total_loss += loss.item() * batch.num_graphs

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update()

        avg_loss = total_loss / total

        return avg_loss
    
    def get_mlde_results(self, data2, y_preds, combos_train):
        AA1s = np.unique([item[0] for item in combos_train])
        AA2s = np.unique([item[1] for item in combos_train])
        AA3s = np.unique([item[2] for item in combos_train])
        AA4s = np.unique([item[3] for item in combos_train])

        data2['y_preds'] = y_preds

        if self.filter_AAs:
            for column, AAs in zip(['AA1', 'AA2', 'AA3', 'AA4'], [AA1s, AA2s, AA3s, AA4s]):
                data2 = data2[data2[column].isin(AAs)]
        
        #print(len(data2))
        #take the top 96 to test
        sorted  = data2.sort_values(by=['y_preds'], ascending = False)

        top = sorted.iloc[:96,:]['fit']
        max = np.max(top)
        mean = np.mean(top)

        #save the top 500
        top_seqs = sorted.iloc[:500,:]['Combo'].values

        return max, mean, top_seqs