#!/bin/bash
python execute_optimization.py --config_file 'simple_exp010_GB1.json' &
python execute_optimization.py --config_file 'simple_exp1_GB1.json' &
python execute_optimization.py --config_file 'simple_exp25_GB1.json' &
python execute_optimization.py --config_file 'simple_exp010_mix2_GB1.json' &
python execute_optimization.py --config_file 'simple_exp1_mix2_GB1.json' &
python execute_optimization.py --config_file 'simple_exp25_mix2_GB1.json' &
python execute_optimization.py --config_file 'simple_exp1_TrpB.json' &
python execute_optimization.py --config_file 'simple_exp25_TrpB.json'
