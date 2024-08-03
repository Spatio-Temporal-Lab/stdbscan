import argparse
import json

from optimizer.utils import parse_args
from optimizer.optimize import Optimizer
from optimizer.utils import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter
import sys


def setup_configuration_space(knob_config_file, knob_num):
    """
    config space
    """
    KNOBS = initialize_knobs(knob_config_file, knob_num)
    knobs_list = []
    config_space = ConfigurationSpace()

    for name in KNOBS.keys():
        value = KNOBS[name]
        knob_type = value['type']
        if knob_type == 'enum':
            knob = CategoricalHyperparameter(name, [str(i) for i in value["enum_values"]], default_value= str(value['default']))
        elif knob_type == 'integer':
            min_val, max_val = value['min'], value['max']
            if KNOBS[name]['max'] > sys.maxsize:
                knob = UniformIntegerHyperparameter(name, int(min_val / 1000), int(max_val / 1000),
                                                    default_value=int(value['default'] / 1000))
            else:
                knob = UniformIntegerHyperparameter(name, min_val, max_val, default_value=value['default'])
        elif knob_type == 'float':
            min_val, max_val = value['min'], value['max']
            knob = UniformFloatHyperparameter(name, min_val, max_val, default_value=value['default'])
        else:
            raise ValueError('Invalid knob type!')

        knobs_list.append(knob)

    config_space.add_hyperparameters(knobs_list)

    return config_space

def initialize_knobs(knobs_config, num):
    global KNOBS
    global KNOB_DETAILS
    if num == -1:
        f = open(knobs_config)
        KNOB_DETAILS = json.load(f)
        KNOBS = list(KNOB_DETAILS.keys())
        f.close()
    else:
        f = open(knobs_config)
        knob_tmp = json.load(f)
        i = 0
        KNOB_DETAILS = {}
        while i < num:
            key = list(knob_tmp.keys())[i]
            KNOB_DETAILS[key] = knob_tmp[key]
            i = i + 1
        KNOBS = list(KNOB_DETAILS.keys())
        f.close()
    return KNOB_DETAILS



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.ini', help='config file')
    opt = parser.parse_args()
    args_db, args_tune = parse_args(opt.config)

    config_space = setup_configuration_space(args_db['knob_config_file'], int(args_db['knob_num']))

    reference_point = []
    op = Optimizer(step,
                   config_space,
                   num_objs=1,
                   optimizer_type=args_tune['optimize_method'],
                   max_runs=int(args_tune['max_runs']),
                   surrogate_type='prf',
                   history_bo_data=[],
                   acq_optimizer_type='local_random',  # 'random_scipy',#
                   selector_type=args_tune['selector_type'],
                   initial_runs=int(args_tune['initial_runs']),
                   incremental=args_tune['incremental'],
                   incremental_every=int(args_tune['incremental_every']),
                   incremental_num=int(args_tune['incremental_num']),
                   init_strategy='random_explore_first',
                   ref_point=reference_point,
                   task_id=args_tune['task_id'],
                   time_limit_per_trial=60 * 200,
                   num_hps_init=int(args_tune['initial_tunable_knob_num']),
                   num_metrics=1,
                   mean_var_file=args_tune['mean_var_file'],
                   batch_size=int(args_tune['batch_size']),
                   params='',
                   knob_config_file=args_db['knob_config_file'],
                   auto_optimizer_type=args_tune['auto_optimizer_type'],
                   hold_out_workload=args_db['workload'],
                   history_workload_data=list(),

                   )

    history = op.run()
