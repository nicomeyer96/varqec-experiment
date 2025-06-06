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
from datetime import datetime
import time
import argparse
import numpy as np
import warnings

from qiskit_ibm_runtime import QiskitRuntimeService, Session

from experiment.varqec_loss_function import distinguishability_loss
from configuration import get_physical_wires, compose_hardware_result_name
from experiment import (construct_experiments, construct_baseline_experiments, evaluate_loss,
                        compare_qiskit_pennylane_realizations)


# pre-trained models for IBMQ hardware for different system sizes
MODELS = {
    3: 'ibmq_error=thermal_relaxation=10.0us-t1=180us-t2=120us_wires=1-2_blocks=3_epochs=25',
    4: 'ibmq_error=thermal_relaxation=10.0us-t1=180us-t2=120us_wires=1-3_blocks=4_epochs=25',
    5: 'ibmq_error=thermal_relaxation=10.0us-t1=180us-t2=120us_wires=1-4_blocks=6_epochs=25'
}


def run_on_ibmq(args):
    # get physical wire allocations
    physical_wires, patches = get_physical_wires(num_wires=args.wires, provider='ibmq', device=args.device)

    # determine result path
    path = compose_hardware_result_name(provider='ibmq', device=args.device, num_wires=args.wires, delay=args.delay,
                                        shots=args.shots, path='baseline' if args.establish_baseline else 'results',
                                        submit=args.submit, retrieve=args.retrieve)

    api_token = None
    if api_token is None:
        raise ValueError('Please add a valid API Token to access IBM Quantum.')
    service = QiskitRuntimeService(channel='ibm_quantum', token=api_token)
    print(f'Set up runtime service ({datetime.now()}).')
    backend = service.backend(f'ibm_{args.device}')
    print(f'Set up backend ({datetime.now()}).')

    # set up handles for baseline experiments (i.e. on physical wires)
    baseline_experiments = construct_baseline_experiments(delay=args.delay, backend=backend,
                                                          physical_wires=physical_wires, resample=args.resample > 0)

    if not args.establish_baseline:
        # see if model of desired size exists
        model = MODELS.get(args.wires, None)
        if model is None:
            raise ValueError(
                f'No model for {args.wires} qubits on the IBMQ devices has been trained. New models have to be'
                f' linked via the `MODELS` global variable.')
        # validate that circuit realizations in Qiskit and PennyLane are identical
        compatible = compare_qiskit_pennylane_realizations(model=model, num_wires=args.wires, provider='ibmq')
        if not compatible:
            raise RuntimeError('Qiskit and PennyLane Ansätze are not identical.')
        # set up handles for experiments (i.e. with encoding circuits)
        experiments = construct_experiments(model=model, provider='ibmq', delay=args.delay, backend=backend,
                                            physical_wires=physical_wires, resample=args.resample > 0)

    #####################################################
    # submit the experiments to hardware (non-blocking) #
    #####################################################
    if args.submit:
        max_time = 300 if args.establish_baseline else 7200  # maximum time for a session
        if args.use_session:
            # set a timeout of 5 minutes / 2 hours for baseline / full experiment
            session = Session(backend=backend, max_time=max_time)
            print(f'Set up session with a maximum duration of {max_time}s ({datetime.now()}).')
        else:  # run as individual jobs on backend
            session = None

        job_ids_baseline, job_ids = {}, {}
        # iterate over all basis states from the 6-element spherical 2-design
        for basis_state in [0, 1, 2, 3, 4, 5]:
            print(f'Submitting baseline experiment for basis state #{basis_state} ({datetime.now()}).')
            # will automatically use session if not None, else defaults to individual submission via backend
            job_id_baseline_ = baseline_experiments[basis_state].submit(session=session, backend=backend,
                                                                        shots=args.shots)
            job_id_baseline = [jib_.job_id() for jib_ in job_id_baseline_]
            job_ids_baseline[basis_state] = job_id_baseline
            if not args.establish_baseline:
                print(f'Submitting experiment for basis state #{basis_state} ({datetime.now()}).')
                job_id_ = experiments[basis_state].submit(session=session, backend=backend, shots=args.shots)  # noqa
                job_id = [ji_.job_id() for ji_ in job_id_]
                job_ids[basis_state] = job_id

        # store the job ids
        with open(os.path.join(path, 'job_ids_baseline.pkl'), 'wb') as ff:
            pickle.dump(job_ids_baseline, ff)
        if not args.establish_baseline:
            with open(os.path.join(path, 'job_ids.pkl'), 'wb') as ff:
                pickle.dump(job_ids, ff)
        print(f'Stored job_ids to {path} ({datetime.now()}).')

        # close session after some idle times
        if args.use_session:
            print(f'Waiting 1h before closing the session to ensure everything has been scheduled ({datetime.now()}).')
            time.sleep(3600)
            session.close()
            print(f'Session has been closed ({datetime.now()}).')

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

        states_baseline, states = [], []
        # iterate over all basis states from the 6-element spherical 2-design
        for basis_state in [0, 1, 2, 3, 4, 5]:
            job_id_baseline = job_ids_baseline.get(basis_state, None)
            if job_id_baseline is None:
                raise RuntimeError(f'Job id for baseline experiment on basis state #{basis_state} not found.')
            # Load jobs, retrieve and save raw results
            with warnings.catch_warnings():  # suppresses harmless warnings on different Qiskit versions
                warnings.filterwarnings('ignore', category=UserWarning)
                jobs_baseline = [service.job(jib) for jib in job_id_baseline]
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
                    print(f'[{index+1}/{args.resample}] '
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
                with warnings.catch_warnings():  # suppresses harmless warnings on different Qiskit versions
                    warnings.filterwarnings('ignore', category=UserWarning)
                    jobs = [service.job(ji) for ji in job_id]
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
                else:  # perform cluster re-sampling, different instance will be returned for each call to retrieve
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
        # store resulting states
        with open(os.path.join(path, f'states_baseline{f'_resample={args.resample}' if args.resample > 0 else ''}.pkl'),
                  'wb') as ff:
            pickle.dump(states_baseline, ff)
        if not args.establish_baseline:
            with open(os.path.join(path, f'states{f'_resample={args.resample}' if args.resample > 0 else ''}.pkl'),
                      'wb') as ff:
                pickle.dump(states, ff)
        print(f'Stored results to {path} ({datetime.now()})')


def evaluate(args):
    path = compose_hardware_result_name(provider='ibmq', device=args.device, num_wires=args.wires, delay=args.delay,
                                        shots=args.shots, path='baseline' if args.establish_baseline else 'results')
    evaluate_loss(args, loss=distinguishability_loss, path=path)


def parse_ibmq():
    parser = argparse.ArgumentParser()
    # For training and testing
    parser.add_argument('--device', type=str, default='marrakesh', choices=['marrakesh'],
                        help='IBMQ device to perform experiments on.')
    parser.add_argument('--wires', type=int, default=4,
                        help='Number of physical wires for each logical patch.')
    parser.add_argument('--delay', type=float, default=0.005,
                        help='Delay to imply on all involved qubits before measuring [in ms].')
    parser.add_argument('--shots', type=int, default=10000,
                        help='Number of shots for executing the experiments.')
    parser.add_argument('--submit', action='store_true',
                        help='Submit the experiments to hardware.')
    parser.add_argument('--use_session', action='store_true',
                        help='Use the IBMQ Session for running experiment instead of individual jobs.')
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

    _args = parse_ibmq()
    if _args.submit or _args.retrieve:  # generate / retrieve experimental data
        run_on_ibmq(_args)
    else:
        evaluate(_args)
