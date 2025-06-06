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

from qiskit import QuantumCircuit
import numpy as np
import os
import pickle

from configuration.platform_experiment import get_layout


def state_initialization_circuits(wires, delay: float = 0.0, virtualize_delay_with_rx: bool = False):
    """
    Prepares states corresponding to unitary two-design on topmost qubit.
    """
    circuits = [QuantumCircuit(len(wires)) for _ in range(6)]
    # |1> state
    circuits[1].x(0)
    # |+> state
    circuits[2].h(0)
    # |-> state
    circuits[3].x(0)
    circuits[3].h(0)
    # |+i> state
    circuits[4].h(0)
    circuits[4].s(0)
    # |-i> state
    circuits[5].h(0)
    circuits[5].sdg(0)
    for qc in circuits:
        qc.barrier(wires)
        # apply delay directly after state initialization (e.g. for benchmarking physical wires)
        if delay > 0.0:
            for wire in wires:
                if virtualize_delay_with_rx:
                    # As IQM devices do not support delay gates, and identities are pruned, virtualize the delay with
                    # application of a Rx(theta=0) gate.
                    # We work under the assumption, that a typical gate execution time is 30ns = 3e-5ms
                    gate_duration = 3e-5
                    number_gates = int(delay / gate_duration)
                    for _ in range(number_gates):
                        qc.rx(0.0, wire)
                else:
                    qc.delay(delay, wire, unit='ms')
            qc.barrier(wires)
    return circuits


def _encoding_circuits(wires, instance, connectivity, parameters_initial, parameters_block, delay: float,
                       provider: str = 'ibmq', virtualize_delay_with_rx: bool = False):
    # generate circuits with already prepared initial states
    circuits = state_initialization_circuits(wires)
    ############################################################
    # This reproduces the RandomizedEntanglingAnsatz in Qiskit #
    ############################################################
    if 'ibmq' == provider:
        for qc in circuits:  # apply to all circuits (containing potentially different initial states)
            # initial layer
            for wire, parameters_initial_ in enumerate(parameters_initial):
                qc.rz(parameters_initial_[0], wire)
                qc.rx(parameters_initial_[1], wire)
                qc.rz(parameters_initial_[2], wire)
            # randomized blocks, make sure to choose the same seed as in the PennyLane version
            rng = np.random.default_rng(seed=instance)
            for block, parameters_block_ in enumerate(parameters_block):
                target = rng.choice(wires)
                potential_controls = connectivity[target]
                control = rng.choice(potential_controls)
                # two-qubit gate
                qc.cz(control, target)
                # single-qubit gates on control
                qc.rz(parameters_block_[0], control)
                qc.rx(parameters_block_[1], control)
                qc.rz(parameters_block_[2], control)
                # single-qubit gate(s) on target
                qc.rz(parameters_block_[3], target)
                qc.rx(parameters_block_[4], target)
                qc.rz(parameters_block_[5], target)
    elif 'iqm' == provider:
        for qc in circuits:  # apply to all circuits (containing potentially different initial states)
            # initial layer
            for wire, parameters_initial_ in enumerate(parameters_initial):
                qc.r(parameters_initial_[0], parameters_initial_[1], wire)
            # randomized blocks, make sure to choose the same seed as in the PennyLane version
            rng = np.random.default_rng(seed=instance)
            for block, parameters_block_ in enumerate(parameters_block):
                target = rng.choice(wires)
                potential_controls = connectivity[target]
                control = rng.choice(potential_controls)
                # two-qubit gate
                qc.cz(control, target)
                # single-qubit gates on control
                qc.r(parameters_block_[0], parameters_block_[1], control)
                # single-qubit gate(s) on target
                qc.r(parameters_block_[2], parameters_block_[3], target)
    else:
        raise NotImplementedError(f'Provider {provider} is not available.')

    for qc in circuits:
        qc.barrier(wires)
        # apply delay directly after state initialization (e.g. for benchmarking physical wires)
        if delay > 0.0:
            for wire in wires:
                if virtualize_delay_with_rx:
                    # As IQM devices do not support delay gates, and identities are pruned, virtualize the delay with
                    # application of a Rx(theta=0) gate.
                    # We work under the assumption, that a typical gate execution time is 30ns = 3e-5ms
                    gate_duration = 3e-5
                    number_gates = int(delay / gate_duration)
                    for _ in range(number_gates):
                        qc.rx(0.0, wire)
                else:
                    qc.delay(delay, wire, unit='ms')
            qc.barrier(wires)

    return circuits, wires


def _load_model(model: str, path: str, provider: str, layout: str = None):
    if not os.path.exists(os.path.join(path, f'{model}.pkl')):
        raise ValueError(f'The model {model}.pkl could not be found at {path}.')
    with open(os.path.join(path, f'{model}.pkl'), 'rb') as ff:
        data = pickle.load(ff)
    num_wires = data.get('args').get('wires_data') + data.get('args').get('wires_ancilla')
    instance = data.get('args').get('seed_encoding')
    params_encoding_initial = data.get('params_encoding').get('params_encoding_initial')
    params_encoding_block = data.get('params_encoding').get('params_encoding_block')
    wires = list(range(num_wires))
    connectivity = get_layout(wires, provider=provider, layout=layout)
    return wires, instance, connectivity, params_encoding_initial, params_encoding_block


def load_circuits(model: str, provider: str, delay: float, path: str = os.path.join('experiment', 'models'),
                  layout: str = None, virtualize_delay_with_rx: bool = False):
    wires, instance, connectivity, params_encoding_initial, params_encoding_block = _load_model(model, path, provider,
                                                                                                layout=layout)
    return _encoding_circuits(wires, instance, connectivity, params_encoding_initial, params_encoding_block,
                              provider=provider, delay=delay, virtualize_delay_with_rx=virtualize_delay_with_rx)


if __name__ == '__main__':

    _circuit = load_circuits('ibmq_error=thermal_relaxation=10.0us-t1=180us-t2=120us_wires=1-3_blocks=4_epochs=25_seed=16')  # noqa
    for _qc in _circuit:
        print(_qc)
