import argparse
import json
import os
import sys

from DeCOIL.src.optimization import run_greedy

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(save_dir, 'log.txt'), 'w')
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass    

# Script starts here.
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str,
                    required=False, default='',
                    help='config file for experiments')
parser.add_argument('--exp_name', type=str,
                    required=False, default='',
                    help='experiment name (default will be config folder name)')
args = parser.parse_args()

# Get JSON config file
config_file = os.path.join(os.getcwd(), 'DeCOIL', 'configs', args.config_file)

# Get experiment name
exp_name = args.exp_name if len(args.exp_name) > 0 else args.config_file[:-5]

# Get save directory
save_dir = os.path.join(os.getcwd(), 'DeCOIL', 'saved', exp_name)

# Create save folder
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#Redirect output to log file
sys.stdout = Logger()

print('Config file:\t {}'.format(config_file))
print('Save directory:\t {}'.format(save_dir))

# Load JSON config file
with open(config_file, 'r') as f:
    config = json.load(f)

#save the config file
with open(os.path.join(save_dir, args.config_file), 'w') as f:
    json.dump(config, f, indent=4)

# Start training
results = run_greedy(
    save_path=save_dir,
    data_config=config['data_config'],
    opt_config=config['opt_config']
)
