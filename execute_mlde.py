import argparse
import json
import os
from xmlrpc.client import Boolean
import sys

import time
from time import gmtime, strftime
import numpy as np
#import matplotlib.pyplot as plt

from MLDE_lite.src.train_eval import *

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(save_dir, 'log.txt'), 'a')
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    

# Script starts here.
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str,
                    required=False, default='',
                    help='config file for experiments')
parser.add_argument('--exp_name', type=str,
                    required=False, default='',
                    help='experiment name (default will be config folder name)')
parser.add_argument('-d', '--device', type=int,
                    required=False, default=0,
                    help='device to run the experiment on')
parser.add_argument('--extract', action='store_true',
                    help='whether to extract the features')

args = parser.parse_args()

# Get JSON config file
config_file = os.path.join(os.getcwd(), 'MLDE_lite', 'configs', args.config_file)

# Get experiment name
#Experiment should be either named the library optimization procedure or random sampling
exp_name = args.exp_name if len(args.exp_name) > 0 else args.config_file[:-5]

# Get save directory
save_dir = os.path.join(os.getcwd(), 'MLDE_lite', 'saved', exp_name)

# Create save folder
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#Redirect output to log file
# sys.stdout = Logger()
# sys.stdout = open(os.path.join(save_dir, 'log.txt'), 'w')

print('Config file:\t {}'.format(config_file))
print('Save directory:\t {}'.format(save_dir))

# Get device ID
if torch.cuda.is_available() and args.device >= 0:
    assert args.device < torch.cuda.device_count()
    device = 'cuda:{:d}'.format(args.device)
else:
    device = 'cpu'
print('Device:\t {}'.format(device))

# Load JSON config file
with open(config_file, 'r') as f:
    config = json.load(f)

#save the config file
with open(os.path.join(save_dir, args.config_file), 'w') as f:
    json.dump(config, f, indent=4)

# Start training
encodings = config['data_config']['encoding']
library = config['data_config']['library']

if isinstance(library, str):
    dclibrary = True
else:
    dclibrary = False
    assert len(library) == config['data_config']['n_solutions']

model_classes = config['model_config']['name']
n_sampless = config['train_config']['n_samples']

all_ndcgs = np.zeros((len(encodings), len(model_classes), len(n_sampless), config['data_config']['n_solutions'], config['train_config']['n_subsets']))
all_maxes = np.copy(all_ndcgs)
all_means = np.copy(all_ndcgs)
all_unique = np.copy(all_ndcgs)
all_labelled = np.copy(all_ndcgs)

all_top_seqs = np.full((len(encodings), len(model_classes), len(n_sampless), config['data_config']['n_solutions'], config['train_config']['n_subsets'], 500), 'VDGV')

#right now this only works with 384 sequences
all_input_seqs = np.full((len(encodings), len(model_classes), len(n_sampless), config['data_config']['n_solutions'], config['train_config']['n_subsets'], 384), 'VDGV')

for i, encoding in enumerate(encodings):
    for j, model_class in enumerate(model_classes):
        for k, n_samples in enumerate(n_sampless):
            
            #keep track of how long the computation took
            start = time.time()

            exp_name2 = encoding + '_' + model_class + '_' + str(n_samples)
            save_dir2 = os.path.join(os.getcwd(), 'MLDE_lite', 'saved', exp_name, exp_name2)

            print('\n###' + exp_name2 + '###')
            
            # Create save folder
            # if not os.path.exists(save_dir2):
            #     os.makedirs(save_dir2)

            mlde_sim = MLDESim(save_path=save_dir2,
                encoding = encoding, 
                model_class = model_class, 
                n_samples = n_samples, 
                model_config=config['model_config'],
                data_config=config['data_config'],
                train_config=config['train_config'],
                device = args.device)
            top_seqs, maxes, means, ndcgs, unique, labelled, input_seqs =  mlde_sim.train_all()
            
            all_top_seqs[i, j, k, :, :, :] = top_seqs
            all_ndcgs[i, j, k, :, :] = ndcgs
            all_maxes[i, j, k, :, :] = maxes
            all_means[i, j, k, :, :] = means
            all_unique[i, j, k, :, :] = unique
            all_labelled[i, j, k, :, :] = labelled

            if config['train_config']['input_seqs'] == True:
                all_input_seqs[i, j, k, :, :, :] = input_seqs

            end = time.time()
            print('Time: ' + str(end-start))

mlde_results = {}
mlde_results['top_seqs'], mlde_results['maxes'], mlde_results['means'], mlde_results['ndcgs'], mlde_results['unique'], mlde_results['labelled'], mlde_results['input_seqs'] = all_top_seqs, all_maxes, all_means, all_ndcgs, all_unique, all_labelled, all_input_seqs

np.save(os.path.join(save_dir, 'mlde_results.npy'), mlde_results)



