# This code is part of the VarQEC Experimental Module.
#
# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import pickle
import os


def compose_experiment_name(args):
    # flatten list of error strength
    noise_strength = '-'.join([str(1000 * n if 'thermal_relaxation' == args.noise else n) for n in args.noise_strength])
    # set up name containing setup information
    name = (f'{args.provider}_error={args.noise}={noise_strength}'
            f'{f'us-t1={1000*args.t1:.0f}us-t2={1000*args.t2:.0f}us' if 'thermal_relaxation' == args.noise else f''}'  # noqa
            f'_wires={args.wires_data}-{args.wires_ancilla}'
            f'{f'_layout={args.layout_iqm}' if 'iqm' == args.provider and 3 == args.wires_ancilla else f''}'  # noqa
            f'_blocks={args.blocks_encoding}'
            f'_epochs={args.epochs_encoding}'
            f'_seed={args.seed}')
    return name


def compose_hardware_result_name(provider: str, device: str, num_wires: int, delay: float, shots: int,
                                 path: str = 'results', submit: bool = False, retrieve: bool = False,
                                 layout: str = None):
    name = (f'{provider}_{device}_wires={num_wires}{f'_layout={layout}' if layout is not None else ''}'
            f'_delay={1000*delay:.1f}us_shots={shots}')
    location = os.path.join('experiment', path, name)
    if retrieve:  # job_ids already must be present
        if not os.path.isdir(location):
            raise RuntimeError(f'Experiment at {location} does not yet exist. Run the script with `--submit` first.')
    elif submit:  # setting up folder from scratch (if it does not yet exist)
        if os.path.isdir(location):
            raise RuntimeError(f'Experiment at {location} already exists. Delete or rename the folder inorder to '
                               f're-submit the experiment.')
        os.makedirs(location)
    else:
        if not os.path.isdir(location):
            raise RuntimeError(f'Experiment at {location} does not yet exist. Run the script with `--submit` and '
                               f'subsequently `--retrieve` first.')
    print(f'Saving results to {location}.')
    return location


def store_results(experiment_path, args, params_encoding, encoding_loss_logger, baseline_encoding_loss, total_time):
    baseline_encoding_avg, baseline_encoding_max = baseline_encoding_loss
    encoding_loss_avg, encoding_loss_max = encoding_loss_logger.get_training_data()
    encoding_val_avg, encoding_val_max = encoding_loss_logger.get_validation_data()
    result = {
        'args': vars(args),
        'params_encoding': params_encoding,
        'baseline_encoding_avg': baseline_encoding_avg, 'baseline_encoding_max': baseline_encoding_max,
        'encoding_loss_avg': encoding_loss_avg, 'encoding_loss_max': encoding_loss_max,
        'encoding_val_avg': encoding_val_avg, 'encoding_val_max': encoding_val_max, 'time': total_time
    }

    with open(experiment_path, 'wb') as ff:
        pickle.dump(result, ff)
    print(f'\nStored results to {experiment_path}.')

    return result
