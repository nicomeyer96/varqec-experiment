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
import pennylane as qml
import numpy as np
from configuration import get_layout, get_gateset
from .varqec_randomized_entangling_ansatz import RandomizedEntanglingAnsatz
from experiment.load_model import load_circuits
from qiskit_aer import StatevectorSimulator


def compare_qiskit_pennylane_realizations(model: str, num_wires: int, provider: str, layout: str = None,
                                          tolerance: float = 0.0001):
    """
    Tests if PennyLane and Qiskit realization of RandomizedEntanglingAnsatz are equivalent.
    """
    # load trained model
    with open(os.path.join('experiment', 'models', f'{model}.pkl'), 'rb') as ff:
        data = pickle.load(ff)
    instance = data.get('args').get('seed_encoding')
    params_encoding_initial = data.get('params_encoding').get('params_encoding_initial')
    params_encoding_block = data.get('params_encoding').get('params_encoding_block')

    wires = list(range(num_wires))
    connectivity = get_layout(wires, provider=provider, layout=layout)  # noqa
    (gates1q, gate2q), _, _ = get_gateset(provider)

    @qml.qnode(qml.device("default.qubit", wires=wires), interface='torch')
    def circuit():
        RandomizedEntanglingAnsatz(params_encoding_initial, params_encoding_block, instance=instance,
                                   wires=wires, gates1q=gates1q, gate2q=gate2q, connectivity=connectivity)
        return qml.state()

    # set up PennyLane circuit
    qc_pennylane = qml.draw(circuit, level=3)
    # print(qc_pennylane())

    # set up Qiskit circuit
    qc_qiskit = load_circuits(model=model, provider=provider, delay=0.0, layout=layout)[0][0]
    # print(qc_qiskit)

    # simulate PennyLane state
    state_pennylane = circuit().numpy()  # noqa

    # simulate Qiskit state
    simulator = StatevectorSimulator()
    state_qiskit = simulator.run(qc_qiskit).result().get_statevector().data
    # fix Qiskit reverse order
    indices = np.array([int(str(bin(b))[2:].zfill(num_wires)[::-1], 2) for b in list(range(2**num_wires))])
    state_qiskit = state_qiskit[indices]

    # return True if states are equal (up to some small threshold)
    return np.all(np.isclose(state_pennylane, state_qiskit, atol=tolerance))
