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

import pennylane as qml
import numpy as np
import torch


class RandomizedEntanglingAnsatz(qml.operation.Operation):
    """
    Encoding circuit operation, with a randomized entangling ansatz (setting `instance` ensures reproducibility).
    The number of blocks is determined by the first dimension of `parameters_block`.
    By default, the gates are instantiated by (C)Rot operations.

    The circuit can be visualized by embedding into a QNode, i.e.

        @qml.qnode(qml.device("default.qubit", wires=3))
        def circuit(parameters):
            EncodingCircuit(parameters, wires_data=0, wires_ancilla=[1, 2], seed=1)
            return qml.state()
        drawer = qml.draw(circuit, show_matrices=False, level=3)
        print(drawer(torch.rand(5, 3)))
    """

    # Can be defined on an arbitrary number of wires (at least two)
    num_wires = qml.operation.AnyWires

    # Note: The current mode does not support parameter-shift gradient computation, but the module could be extended
    #       by designing a respective `grad_recipe`. Other modes of differentiation are supported.
    grad_method = "A"
    grad_recipe = None

    # default gateset: Rot for single-qubit operations, Controlled-Rot for two-qubit operations
    gates1q_default = [qml.Rot]
    gate2q_default = qml.CRot

    def __init__(self,
                 parameters_initial,
                 parameters_block,
                 instance: int,
                 wires: list,
                 gates1q: list[qml.operation.Operation] = None,
                 gate2q: qml.operation.Operation = None,
                 connectivity: dict = None):

        # check for number of wires
        if len(wires) <= 1:
            raise ValueError("At least two wires have to be provided.")

        # check and set gates
        gates1q, gate2q = self._check_gates(gates1q, gate2q, wires, parameters_initial, parameters_block)

        # all wires that the operator acts on
        wires = qml.wires.Wires(wires)

        # check and set connectivity structure
        connectivity = self._check_connectivity(connectivity, wires)

        blocks = parameters_block.shape[0]
        # define non-trainable hyperparameters
        self._hyperparameters = {
            'instance': instance,
            'blocks': blocks,
            'gates1q': gates1q,
            'gate2q': gate2q,
            'connectivity': connectivity
        }

        # initialize the parent class
        super().__init__(parameters_initial, parameters_block, wires=wires,
                         id=f'#={blocks},instance={instance}')  # noqa

    @property
    def num_params(self):
        # set of initial parameters, set of block parameters
        return 2

    @property
    def ndim_params(self):
        # for initial parameters: (wires, parameters_per_wire), for block parameters: (blocks, parameters_per_block)
        return 2, 2

    @staticmethod
    def compute_decomposition(parameters_initial, parameters_block, wires, instance, blocks,  # noqa
                              gates1q, gate2q, connectivity):  # pylint: disable=arguments-differ  # noqa
        # set seed to fix ansatz instance
        rng = np.random.default_rng(seed=instance)
        op_list = []

        # initial layer with single-qubit operation on each qubit
        for wire_index, wire in enumerate(wires):
            start_index = 0
            for gate in gates1q:
                op_list.append(gate(*parameters_initial[wire_index, start_index:start_index+gate.num_params],
                                    wires=wire))
                start_index += gate.num_params

        # consecutive blocks, each layer contains one (randomly-placed) 2-qubit gate, followed by two single-qubit gates
        for block in range(blocks):
            # randomly select target and control qubit
            target = rng.choice(wires)
            potential_controls = connectivity[target]  # connected qubits
            control = rng.choice(potential_controls)

            # apply two-qubit gate, potentially followed by two-qubit error
            start_index = 0
            op_list.append(gate2q(*parameters_block[block, start_index:start_index+gate2q.num_params],
                                  wires=[control, target]))
            start_index += gate2q.num_params

            # apply first single-qubit gate (on control), potentially followed by single-qubit error
            for gate in gates1q:  # apply single-qubit gates to control
                op_list.append(gate(*parameters_block[block, start_index:start_index + gate.num_params],
                                    wires=control))
                start_index += gate.num_params

            # apply second single-qubit gate (on target), potentially followed by single-qubit error
            for gate in gates1q:  # apply single-qubit gates to target
                op_list.append(gate(*parameters_block[block, start_index:start_index + gate.num_params],
                                    wires=target))
                start_index += gate.num_params

        return op_list

    def _check_gates(self, gates1q, gate2q, wires, parameters_initial, parameters_block):
        """
        Extracts and sets gateset, checks parameter shapes.
        """
        # check and set initial parametrized gates
        if gates1q is None:  # use default gates
            gates1q = self.gates1q_default
        else:
            for g in gates1q:
                if 1 != g.num_wires:
                    raise ValueError(f'The gate {g} cannot be applied to 1 wire.')
        num_parameters_per_wire = np.sum([g.num_params for g in gates1q])
        if (len(wires), num_parameters_per_wire) != parameters_initial.shape:
            raise ValueError(f'The parameters for the initial layer have to be of shape '
                             f'[{len(wires)}, {num_parameters_per_wire}] (were {list(parameters_initial.shape)}).')

        # check and set block gates
        if gate2q is None:  # use default gates
            gate2q = self.gate2q_default
        else:
            if 2 != gate2q.num_wires:
                raise ValueError(f'The gate {gate2q} cannot be applied to 2 wires.')
        num_parameters_per_block = gate2q.num_params + 2 * num_parameters_per_wire
        if parameters_block.shape[1] != num_parameters_per_block:
            raise ValueError(f'The parameters for the blocks have to be of shape '
                             f'[_,{num_parameters_per_block}] (were {list(parameters_block.shape)}).')
        return gates1q, gate2q

    @staticmethod
    def _check_connectivity(connectivity, wires):
        """
        Checks if connectivity structure is valid, defaults to all-to-all connectivity.
        """
        # check connectivity structure
        if connectivity is None:  # default: set up all-to-all connectivity
            connectivity = {wire: [w for w in wires if w != wire] for wire in wires}
        else:
            keys = list(connectivity.keys())
            for key in keys:
                if key not in wires:
                    raise ValueError(f'The wire {key} from the connectivity dictionary is not in the list of wires.')
                if key in connectivity[key]:
                    raise ValueError(f'The wire {key} is listed as connected to itself.')
                if len(connectivity[key]) < 1:
                    raise ValueError(f'The wire {key} is not connected to any other wire.')
            for wire in wires:
                if wire not in keys:
                    raise ValueError(f'The wire {wire} from the list of wires is not in the connectivity dictionary.')
            for key in keys:
                values = connectivity[key]
                for value in values:
                    if key not in connectivity[value]:
                        raise ValueError(
                            f'The wire {key} is connected to {value}, but {value} is not connected to {key}.')  # noqa
        return connectivity


if __name__ == '__main__':
    _wires = [0, 1, 2]
    _blocks = 10
    _connectivity = {
        0: [1, 2],
        1: [0],
        2: [0]
    }

    @qml.qnode(qml.device("default.mixed", wires=_wires), interface='torch')
    def circuit(parameters_initial, parameters_block):
        RandomizedEntanglingAnsatz(parameters_initial, parameters_block,
                                   instance=42, wires=_wires, connectivity=_connectivity)
        qml.adjoint(
            RandomizedEntanglingAnsatz(parameters_initial, parameters_block,
                                       instance=42, wires=_wires, connectivity=_connectivity)
        )
        return qml.state()

    torch.manual_seed(42)
    _parameters_initial = torch.rand(len(_wires), 3, requires_grad=True)
    _parameters_block = torch.rand(_blocks, 9, requires_grad=True)

    _drawer = qml.draw(circuit, level=1, show_matrices=False)
    # _drawer = qml.draw(circuit, level=3, show_matrices=False)
    print(_drawer(_parameters_initial, _parameters_block))
    _state = circuit(_parameters_initial, _parameters_block)
    print(_state)
