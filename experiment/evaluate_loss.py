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

import os.path
import pickle
from typing import Callable
import numpy as np
import torch


GROUNDTRUTH_STATES = torch.tensor(np.array(
        [[[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],  # |0><0|
         [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],  # |1><1|
         [[0.5 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.5 + 0.0j]],  # |+><+|
         [[0.5 + 0.0j, -0.5 + 0.0j], [-0.5 + 0.0j, 0.5 + 0.0j]],  # |-><-|
         [[0.5 + 0.0j, 0.0 - 0.5j], [0.0 + 0.5j, 0.5 + 0.0j]],  # |+i><+i|
         [[0.5 + 0.0j, 0.0 + 0.5j], [0.0 - 0.5j, 0.5 + 0.0j]]]  # |-i><-i|
    ))


def evaluate_loss(args, loss: Callable, path: str, display: bool = True, loss_type: str = 'max'):
    states_baseline, states = load_states(wires=args.wires, resample=args.resample, path=path,
                                          establish_baseline=args.establish_baseline)

    if args.resample > 0:
        loss_baseline_avg, loss_baseline_max, loss_avg, loss_max = [], [], [], []
        for index in range(states_baseline.shape[0]):
            states_baseline_ = states_baseline[index]
            states_ = states[index] if not args.establish_baseline else None
            loss_baseline_avg_, loss_baseline_max_, loss_avg_, loss_max_ = _evaluate_loss(args, states_baseline_,
                                                                                          states_, loss)
            loss_baseline_avg.append(loss_baseline_avg_)
            loss_baseline_max.append(loss_baseline_max_)
            if not args.establish_baseline:
                loss_avg.append(loss_avg_)
                loss_max.append(loss_max_)
        loss_baseline_avg, loss_baseline_max = np.array(loss_baseline_avg), np.array(loss_baseline_max)
        with open(os.path.join(path, f'loss_baseline_resample={args.resample}.pkl'), 'wb') as ff:
            pickle.dump({'avg': loss_baseline_avg, 'max': loss_baseline_max}, ff)
        if not args.establish_baseline:
            loss_avg, loss_max = np.array(loss_avg), np.array(loss_max)
            with open(os.path.join(path, f'loss_resample={args.resample}.pkl'), 'wb') as ff:
                pickle.dump({'avg': loss_avg, 'max': loss_max}, ff)
        if display:
            print(loss_baseline_avg.shape)
            for patch in range(loss_baseline_avg.shape[1]):
                loss_baseline = loss_baseline_avg[:, patch] if 'avg' == loss_type else loss_baseline_max[:, patch]
                print(f'\nPatch #{patch}:')
                for wire in range(loss_baseline.shape[1]):
                    print(f'{f'Wire `d0`' if 0 == wire else f'Wire `a{wire - 1}`'}: '
                          f'median loss={np.median(loss_baseline[:, wire]):.3f}, '
                          f'10/90-percentile={np.percentile(loss_baseline[:, wire], 10):.3f}/'
                          f'{np.percentile(loss_baseline[:, wire], 90):.3f}')
                if not args.establish_baseline:
                    loss = loss_avg[:, patch] if 'avg' == loss_type else loss_max[:, patch]  # noqa
                    print(f'Encoding:  '
                          f'median loss={np.median(loss):.3f}, '
                          f'10/90-percentile={np.percentile(loss, 10):.3f}/{np.percentile(loss, 90):.3f}')
    else:
        loss_baseline_avg, loss_baseline_max, loss_avg, loss_max = _evaluate_loss(args, states_baseline, states, loss)
        with open(os.path.join(path, 'loss_baseline.pkl'), 'wb') as ff:
            pickle.dump({'avg': loss_baseline_avg, 'max': loss_baseline_max}, ff)
        if not args.establish_baseline:
            with open(os.path.join(path, 'loss.pkl'), 'wb') as ff:
                pickle.dump({'avg': loss_avg, 'max': loss_max}, ff)
        if display:
            for patch in range(loss_baseline_avg.shape[0]):
                loss_baseline = loss_baseline_avg[patch] if 'avg' == loss_type else loss_baseline_max[patch]
                print(f'\nPatch #{patch}: best={np.min(loss_baseline):.3f}, avg={np.average(loss_baseline):.3f}')
                print(f'Wire ', end='')
                for wire in range(loss_baseline.shape[0]):
                    print(f'{f'`d`' if 0 == wire else f'`a{wire - 1}`'}: loss={loss_baseline[wire]:.3f}'
                          f'{f', ' if wire != loss_baseline.shape[0] - 1 else ''}', end='')
                print()
                if not args.establish_baseline:
                    loss = loss_avg[patch] if 'avg' == loss_type else loss_max[patch]  # noqa
                    print(f'Encoding: loss={loss:.3f}')


def _evaluate_loss(args, states_baseline, states, loss):
    loss_baseline_avg, loss_baseline_max = [], []
    # iterate over the patches
    for state_baseline in states_baseline:
        loss_baseline_wire_avg, loss_baseline_wire_max = [], []
        # iterate over the physical wires in this patch
        for wire in range(state_baseline.shape[1]):
            state_baseline_wire = torch.tensor(state_baseline[:, wire])
            avg_, max_ = loss(rho=state_baseline_wire, rho_groundtruth=GROUNDTRUTH_STATES)
            loss_baseline_wire_avg.append(avg_.detach().numpy())
            loss_baseline_wire_max.append(max_.detach().numpy())
        loss_baseline_avg.append(loss_baseline_wire_avg)
        loss_baseline_max.append(loss_baseline_wire_max)
    loss_baseline_avg, loss_baseline_max = np.array(loss_baseline_avg), np.array(loss_baseline_max)

    if not args.establish_baseline:
        loss_avg, loss_max = [], []
        # iterate over the patches
        for state in states:
            state = torch.tensor(state)
            avg_, max_ = loss(rho=state, rho_groundtruth=GROUNDTRUTH_STATES)
            loss_avg.append(avg_.detach().numpy())
            loss_max.append(max_.detach().numpy())
        loss_avg, loss_max = np.array(loss_avg), np.array(loss_max)
    else:
        loss_avg, loss_max = None, None

    return loss_baseline_avg, loss_baseline_max, loss_avg, loss_max


def load_states(wires: int, resample: int, path: str, establish_baseline: bool = True):
    if not os.path.isdir(path):
        raise ValueError(f'The experiment at {path} does not exist.')
    if not os.path.isfile(os.path.join(path, f'states_baseline{f'_resample={resample}' if resample > 0 else ''}.pkl')):
        raise ValueError(f'The baseline states have not been extracted, run the script with `--retrieve` '
                         f'{f'and `--resample {resample}` ' if resample > 0 else ''}first.')
    with open(os.path.join(path, f'states_baseline{f'_resample={resample}' if resample > 0 else ''}.pkl'), 'rb') as ff:
        states_baseline = pickle.load(ff)
    if resample > 0:
        if resample != states_baseline.shape[0]:
            raise RuntimeError(f'The resample dimension was expected to be {resample}, '
                               f'but was {states_baseline.shape[0]}.')
    if 6 != states_baseline.shape[1 + int(resample > 0)]:
        raise RuntimeError(f'The basis_states dimension is expected to be 6, '
                           f'but was {states_baseline.shape[1 + int(resample > 0)]}.')
    if wires != states_baseline.shape[2 + int(resample > 0)]:
        raise RuntimeError(f'The wires dimension is expected to be {wires}, '
                           f'but was {states_baseline.shape[2 + int(resample > 0)]}.')
    if 2 != states_baseline.shape[3 + int(resample > 0)] or 2 != states_baseline.shape[4 + int(resample > 0)]:
        raise RuntimeError(f'The data dimensions are expected to be 2x2, but were '
                           f'{states_baseline.shape[3 + int(resample > 0)]}x{states_baseline.shape[4] + int(resample > 0)}.')  # noqa
    print(f'Loaded baseline states with {states_baseline.shape[0 + int(resample > 0)]} patches '
          f'and {states_baseline.shape[2 + int(resample > 0)]} wires per patch.')

    if not establish_baseline:
        if not os.path.isfile(os.path.join(path, f'states{f'_resample={resample}' if resample > 0 else ''}.pkl')):
            raise ValueError(f'The states have not been extracted, run the script with `--retrieve` '
                             f'{f'and `--resample {resample}` ' if resample > 0 else ''}first.')
        with open(os.path.join(path, f'states{f'_resample={resample}' if resample > 0 else ''}.pkl'), 'rb') as ff:
            states = pickle.load(ff)
        if resample > 0:
            if resample != states_baseline.shape[0]:
                raise RuntimeError(f'The resample dimension was expected to be {resample}, '
                                   f'but was {states_baseline.shape[0]}.')
        if 6 != states.shape[1 + int(resample > 0)]:
            raise RuntimeError(f'The basis_states dimension is expected to be 6, '
                               f'but was {states.shape[1 + int(resample > 0)]}.')
        if 2**wires != states.shape[2 + int(resample > 0)] or 2**wires != states.shape[3 + int(resample > 0)]:
            raise RuntimeError(f'The data dimensions are expected to be {2**wires}x{2**wires}, but were '
                               f'{states.shape[2 + int(resample > 0)]}x{states.shape[3 + int(resample > 0)]}.')
        if states_baseline.shape[0 + int(resample > 0)] != states.shape[0 + int(resample > 0)]:
            raise RuntimeError(f'Inconsistent number of patches between baseline states '
                               f'({states_baseline.shape[0 + int(resample > 0)]}) and states '
                               f'({states.shape[0 + int(resample > 0)]}).')
        print(f'Loaded states with {states.shape[0 + int(resample > 0)]} patches, each on {wires} wires.')
    else:
        states = None
    return states_baseline, states
