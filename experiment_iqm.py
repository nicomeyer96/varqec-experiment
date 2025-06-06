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
import warnings
from datetime import datetime
import argparse
import numpy as np

from iqm.qiskit_iqm import IQMProvider

from experiment.varqec_loss_function import distinguishability_loss
from configuration import get_physical_wires, compose_hardware_result_name
from experiment import (construct_experiments, construct_baseline_experiments, evaluate_loss,
                        compare_qiskit_pennylane_realizations)


# pre-trained models for IQM hardware for different system sizes (and layouts)
MODELS = {
    4: {'star': 'iqm_error=thermal_relaxation=2.0us-t1=48us-t2=20us_wires=1-3_layout=star_blocks=3_epochs=25',
        'square': 'iqm_error=thermal_relaxation=2.0us-t1=48us-t2=20us_wires=1-3_layout=square_blocks=4_epochs=25'},  # noqa
}


def run_on_iqm(args):

    # get physical wire allocations
    physical_wires, patches = get_physical_wires(num_wires=args.wires, provider='iqm', device=args.device,
                                                 layout=args.layout)

    # # determine result path
    path = compose_hardware_result_name(provider='iqm', device=args.device, num_wires=args.wires, delay=args.delay,
                                        shots=args.shots, path='baseline' if args.establish_baseline else 'results',
                                        submit=args.submit, retrieve=args.retrieve,
                                        layout=args.layout if 4 == args.wires else None)

    server_url = f'https://cocos.resonance.meetiqm.com/{args.device}'
    api_token = None
    if api_token is None:
        raise ValueError('Please add a valid API Token to access IQM Resonance.')

    service = IQMProvider(server_url, token=api_token)
    print(f'Set up provider ({datetime.now()}).')
    backend = service.get_backend()  # noqa
    print(f'Set up backend ({datetime.now()}).')

    # set up handles for baseline experiments (i.e. on physical wires)
    baseline_experiments = construct_baseline_experiments(delay=args.delay, backend=backend,
                                                          physical_wires=physical_wires,
                                                          virtualize_delay_with_rx=True, resample=args.resample > 0)

    if not args.establish_baseline:
        # see if model of desired size exists
        model = MODELS.get(args.wires, None)
        if 4 == args.wires:
            model = model.get(args.layout, None)  # noqa
        if model is None:
            raise ValueError(
                f'No model for {args.layout} layout on the IQM devices has been trained. New models have to be'
                f' linked via the `MODELS` global variable.')
        # validate that circuit realizations in Qiskit and PennyLane are identical
        compatible = compare_qiskit_pennylane_realizations(model=model, num_wires=args.wires, provider='iqm',
                                                           layout=args.layout)
        if not compatible:
            raise RuntimeError('Qiskit and PennyLane Ansätze are not identical.')
        # set up handles for experiments (i.e. with encoding circuits)
        experiments = construct_experiments(model=model, provider='iqm', delay=args.delay, backend=backend,  # noqa
                                            physical_wires=physical_wires, virtualize_delay_with_rx=True,
                                            resample=args.resample > 0, layout=args.layout)

    #####################################################
    # submit the experiments to hardware (non-blocking) #
    #####################################################
    if args.submit:
        job_ids_baseline, job_ids = {}, {}
        # iterate over all basis states from the 6-element spherical 2-design
        for basis_state in [0, 1, 2, 3, 4, 5]:
            print(f'Submitting baseline experiment for basis state #{basis_state} ({datetime.now()}).')
            # will automatically use session if not None, else defaults to individual submission via backend, use .run
            # method instead of Sampler Primitive
            with warnings.catch_warnings():  # suppresses harmless warnings on unknown backend option
                warnings.filterwarnings('ignore', category=UserWarning)
                job_id_baseline_ = baseline_experiments[basis_state].submit(backend=backend, shots=args.shots,
                                                                            backend_run=True)
            job_id_baseline = [jib_.job_id() for jib_ in job_id_baseline_]
            if 0 == len(job_id_baseline):
                print('Job id could not be identified, has to be added manually.')
            job_ids_baseline[basis_state] = job_id_baseline
            if not args.establish_baseline:
                print(f'Submitting experiment for basis state #{basis_state} ({datetime.now()}).')
                with warnings.catch_warnings():  # suppresses harmless warnings on unknown backend option
                    warnings.filterwarnings('ignore', category=UserWarning)
                    job_id_ = experiments[basis_state].submit(backend=backend, shots=args.shots, backend_run=True)  # noqa
                job_id = [ji_.job_id() for ji_ in job_id_]
                if 0 == len(job_id):
                    print('Job id could not be identified, has to be added manually.')
                job_ids[basis_state] = job_id

        # store the job ids
        with open(os.path.join(path, 'job_ids_baseline.pkl'), 'wb') as ff:
            pickle.dump(job_ids_baseline, ff)
        if not args.establish_baseline:
            with open(os.path.join(path, 'job_ids.pkl'), 'wb') as ff:
                pickle.dump(job_ids, ff)
        print(f'Stored job_ids to {path} ({datetime.now()}).')

    ####################################################
    # retrieve the experiment from hardware (blocking) #
    ####################################################
    if args.retrieve:
        # load the job ids
        if not os.path.isfile(os.path.join(path, 'job_ids_baseline.pkl')):
            raise RuntimeError(f'No baseline job ids found at {path}, please run the script with `--submit` flag first.')
        with open(os.path.join(path, 'job_ids_baseline.pkl'), 'rb') as ff:
            job_ids_baseline = pickle.load(ff)
        if not args.establish_baseline:
            if not os.path.isfile(os.path.join(path, 'job_ids.pkl')):
                raise RuntimeError(f'No job ids found at {path}, please run the script with `--submit` flag first.')
            with open(os.path.join(path, 'job_ids.pkl'), 'rb') as ff:
                job_ids = pickle.load(ff)
        print(f'Retrieved job_ids from {path} ({datetime.now()}).')
        print(f'This script might block until the respective jobs have been executed.')

        data_baseline, data = {}, {}
        states_baseline, states = [], []
        # iterate over all basis states from the 6-element spherical 2-design
        for basis_state in [0, 1, 2, 3, 4, 5]:
            job_id_baseline = job_ids_baseline.get(basis_state, None)
            if job_id_baseline is None:
                raise RuntimeError(f'Job id for baseline experiment on basis state #{basis_state} not found.')
            # Load jobs, retrieve and save raw results
            jobs_baseline = [backend.retrieve_job(jib) for jib in job_id_baseline]
            if 0 == args.resample:
                print(f'Retrieving baseline experiment for basis state #{basis_state} ({datetime.now()}).')
                # analyse results, i.e. perform tomography
                analysis_baseline = baseline_experiments[basis_state].retrieve(jobs=jobs_baseline, backend=backend)
                # extract the actual density matrices
                state_baseline = []
                for a in analysis_baseline.analysis_results():
                    if 'state' == a.name:
                        state_baseline.append(a.value.data)
                state_baseline = np.array(state_baseline)
                # reshape into format [patch, wire, 2, 2]
                state_baseline = np.reshape(state_baseline, (patches, args.wires, 2, 2))
                states_baseline.append(state_baseline)
            else:  # perform cluster re-sampling, different instance will be returned for each call to retrieve
                states_baseline_ = []
                for index in range(args.resample):
                    print(f'[{index + 1}/{args.resample}] '
                          f'Retrieving baseline experiment for basis state #{basis_state} ({datetime.now()}).')
                    # analyse results, i.e. perform tomography
                    analysis_baseline = baseline_experiments[basis_state].retrieve(jobs=jobs_baseline, backend=backend)
                    # extract the actual density matrices
                    state_baseline = []
                    for a in analysis_baseline.analysis_results():
                        if 'state' == a.name:
                            state_baseline.append(a.value.data)
                    state_baseline = np.array(state_baseline)
                    # reshape into format [patch, wire, 2, 2]
                    state_baseline = np.reshape(state_baseline, (patches, args.wires, 2, 2))
                    states_baseline_.append(state_baseline)
                states_baseline.append(states_baseline_)

            if not args.establish_baseline:
                job_id = job_ids.get(basis_state, None)
                if job_id is None:
                    raise RuntimeError(f'Job id for experiment on basis state #{basis_state} not found.')
                # Load jobs, retrieve and save raw results
                jobs = [backend.retrieve_job(ji) for ji in job_id]
                if 0 == args.resample:
                    print(f'Retrieving experiment for basis state #{basis_state} ({datetime.now()}).')
                    # analyse results, i.e. perform tomography
                    analysis = experiments[basis_state].retrieve(jobs=jobs, backend=backend)
                    # extract the actual density matrices
                    state = []
                    for a in analysis.analysis_results():
                        if 'state' == a.name:
                            state.append(a.value.data)
                    state = np.array(state)
                    states.append(state)
                else:
                    states_ = []
                    for index in range(args.resample):
                        print(f'[{index + 1}/{args.resample}] '
                              f'Retrieving experiment for basis state #{basis_state} ({datetime.now()}).')
                        # analyse results, i.e. perform tomography
                        analysis = experiments[basis_state].retrieve(jobs=jobs, backend=backend)
                        # extract the actual density matrices
                        state = []
                        for a in analysis.analysis_results():
                            if 'state' == a.name:
                                state.append(a.value.data)
                        state = np.array(state)
                        states_.append(state)
                    states.append(states_)

        # transpose baseline states from [basis_states, (resample), patches, wires_per_patch, 2, 2]
        # to [(resample), patches, basis_states, ...]
        states_baseline = np.array(states_baseline)
        if 5 == len(states_baseline.shape):
            states_baseline = np.transpose(states_baseline, (1, 0, 2, 3, 4))
        else:
            states_baseline = np.transpose(states_baseline, (1, 2, 0, 3, 4, 5))
        if not args.establish_baseline:
            # transpose states from [basis_states, (resample), patches, 2**wires, 2**wires]
            # to [(resample), patches, basis_states, ...]
            states = np.array(states)
            if 4 == len(states.shape):
                states = np.transpose(states, (1, 0, 2, 3))
            else:
                states = np.transpose(states, (1, 2, 0, 3, 4))
        # store raw results and states
        with open(os.path.join(path, f'states_baseline{f'_resample={args.resample}' if args.resample > 0 else ''}.pkl'),
                  'wb') as ff:
            pickle.dump(states_baseline, ff)
        if not args.establish_baseline:
            with open(os.path.join(path, f'states{f'_resample={args.resample}' if args.resample > 0 else ''}.pkl'),
                      'wb') as ff:
                pickle.dump(states, ff)
        print(f'Stored results to {path} ({datetime.now()})')


def evaluate(args):
    path = compose_hardware_result_name(provider='iqm', device=args.device, num_wires=args.wires, delay=args.delay,
                                        shots=args.shots, path='baseline' if args.establish_baseline else 'results',
                                        layout=args.layout if 4 == args.wires else None)
    evaluate_loss(args, loss=distinguishability_loss, path=path)


def parse_iqm():
    parser = argparse.ArgumentParser()
    # For training and testing
    parser.add_argument('--device', type=str, default='garnet', choices=['garnet'],
                        help='IQM device to perform experiments on.')
    parser.add_argument('--wires', type=int, default=4,
                        help='Number of physical wires for each logical patch.')
    parser.add_argument('--layout', type=str, default='square', choices=['square', 'star'],
                        help='Physical qubit alignment strategy for the logical patches '
                             '(only affects 4-wire setup, defaults to `square`).')
    parser.add_argument('--delay', type=float, default=0.001,
                        help='Delay to imply on all involved qubits before measuring [in ms].')
    parser.add_argument('--shots', type=int, default=1000,
                        help='Number of shots for executing the experiments.')
    parser.add_argument('--submit', action='store_true',
                        help='Submit the experiments to hardware.')
    parser.add_argument('--retrieve', action='store_true',
                        help='Retrieve the results from hardware (blocks if experiments still running).')
    parser.add_argument('--resample', type=int, default=0,
                        help='Number of repetitions for cluster re-sampling (by default not active).')
    parser.add_argument('--establish_baseline', action='store_true',
                        help='Only establish baseline, i.e. do not submit / evaluate encoding circuits.')
    args = parser.parse_args()
    if args.submit and args.retrieve:
        raise ValueError('The flags `--submit` and `--retrieve` are mutually exclusive.')
    return parser.parse_args()


if __name__ == '__main__':

    _args = parse_iqm()
    if _args.submit or _args.retrieve:  # generate / retrieve experimental data
        run_on_iqm(_args)
    else:
        evaluate(_args)
