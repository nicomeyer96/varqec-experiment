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

import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
import torch

# selected physical wires / patches for ibmq_manila device
PHYSICAL_WIRES_IBMQ_MARRAKESH = [
    [3, 2, 4, 16, 23],
    [11, 10, 12, 18, 31],
    [47, 46, 48, 57, 67],
    [61, 60, 62, 76, 81],
    [93, 94, 92, 79, 73],
    [107, 108, 106, 97, 87],
    [143, 144, 142, 136, 123],
    [151, 152, 150, 138, 131]
]

# selected physical wires / patches for iqm_garnet device (crystal topology, contains 4-qubit square and star alignment)
PHYSICAL_WIRES_IQM_GARNET = [
    [3, 2, 0, 4, 1],  # [4, 3, 1, 5, 2],  # opposed to the displayed layout, the internal qubit numbering starts with 0!
    [11, 6, 16, 10, 15],  # [12, 7, 17, 11, 16],
    [13, 12, 14, 17, 18],  # [14, 13, 15, 18, 19],
]


def get_physical_wires(num_wires: int, provider: str = 'ibmq', device: str = 'marrakesh', layout: str = 'star'):
    if 'ibmq_marrakesh' == f'{provider}_{device}':
        if num_wires > 5:
            raise ValueError(f'Supports a maximum of 5 wires, {num_wires} were selected.')
        return [physical_wires[:num_wires] for physical_wires in PHYSICAL_WIRES_IBMQ_MARRAKESH], 8
    elif 'iqm_garnet' == f'{provider}_{device}':
        if num_wires > 5:
            raise ValueError(f'Supports a maximum of 5 wires, {num_wires} were selected.')
        if 4 == num_wires and 'square' == layout:  # special case of layout that is allowed by square connectivity
            return [[physical_wires[0]] + physical_wires[2:] for physical_wires in PHYSICAL_WIRES_IQM_GARNET], 3
        else:
            return [physical_wires[:num_wires] for physical_wires in PHYSICAL_WIRES_IQM_GARNET], 3
    else:
        raise NotImplementedError(f'No physical wires defined for the device `{device}` on the provider `{provider}`.')


def get_layout(wires: list, provider: str = 'ibmq', layout: str = 'star'):
    if 'ibmq' == provider:
        match len(wires):
            case 3:
                """
                1 -- 0 -- 2
                """
                return {
                    wires[0]: [wires[1], wires[2]],
                    wires[1]: [wires[0]],
                    wires[2]: [wires[0]],
                }
            case 4:
                """
                1 -- 0 -- 2
                     |
                     3
                """
                return {
                    wires[0]: [wires[1], wires[2], wires[3]],
                    wires[1]: [wires[0]],
                    wires[2]: [wires[0]],
                    wires[3]: [wires[0]],
                }
            case 5:
                """
                1 -- 0 -- 2
                     |
                     3
                     |
                     4
                """
                return {
                    wires[0]: [wires[1], wires[2], wires[3]],
                    wires[1]: [wires[0]],
                    wires[2]: [wires[0]],
                    wires[3]: [wires[0], wires[4]],
                    wires[4]: [wires[3]]
                }
            case _:
                raise NotImplementedError(f'Currently only 3, 4, or 5 wires are supported for IBMQ '
                                          f'({len(wires)} were given).')
    elif 'iqm' == provider:
        match len(wires):
            case 3:
                """
                1 -- 0 -- 2
                """
                return {
                    wires[0]: [wires[1], wires[2]],
                    wires[1]: [wires[0]],
                    wires[2]: [wires[0]],
                }
            case 4:
                match layout:
                    case 'star':
                        """
                        1 -- 0 -- 2
                             |
                             3
                        """
                        return {
                            wires[0]: [wires[1], wires[2], wires[3]],
                            wires[1]: [wires[0]],
                            wires[2]: [wires[0]],
                            wires[3]: [wires[0]],
                        }
                    case 'square':
                        """
                            0
                          |   |
                        1       2
                          |   |
                            3
                        """
                        return {
                            wires[0]: [wires[1], wires[2]],
                            wires[1]: [wires[0], wires[3]],
                            wires[2]: [wires[0], wires[3]],
                            wires[3]: [wires[1], wires[2]],
                        }
                    case _:
                        raise NotImplementedError(f'The layout {layout} is not available for the IQM devices.')
            case 5:
                """
                                1
                                |   
                                0   
                              |   |
                            2       3
                              |   |
                                4
                            """
                return {
                    wires[0]: [wires[1], wires[2], wires[3]],
                    wires[1]: [wires[0]],
                    wires[2]: [wires[0], wires[4]],
                    wires[3]: [wires[0], wires[4]],
                    wires[4]: [wires[2], wires[3]]
                }
            case _:
                raise NotImplementedError(f'Currently only 3, 4, or 5 wires are supported for IQM '
                                          f'({len(wires)} were given).')

    else:
        raise NotImplementedError(f'Layout for provider {provider} not defined.')


class PRx(qml.operation.Operation):
    """
    Realizes the PRx gate implemented natively on the IQM devices
    (see https://github.com/amazon-braket/amazon-braket-examples/blob/main/examples/braket_features/IQM_Garnet_Native_Gates.ipynb).  # noqa
    It can be decomposed as PRx(phi,theta) = Rz(theta)Rx(phi)Rz(-theta).
    """
    # single-qubit gate
    num_wires = 1
    num_params = 2
    ndim_params = (0, 0)

    # gate is differentiable
    grad_method = "A"
    grad_recipe = None

    def __init__(self, phi, theta, wires):
        super().__init__(phi, theta, wires=wires)  # noqa

    @staticmethod
    def compute_decomposition(phi, theta, wires): # pylint: disable=arguments-differ  # noqa
        return [
            qml.RZ(-theta, wires=wires),
            qml.RX(phi, wires=wires),
            qml.RZ(theta, wires=wires)
        ]


def test_PRx():  # noqa
    from qiskit_aer import StatevectorSimulator

    # random parameters
    phi, theta = np.random.randn(2)

    # set up PennyLane realization of PRx gate
    @qml.qnode(qml.device("default.qubit", wires=[0]), interface='torch')
    def circuit(_phi, _theta):
        PRx(_phi, _theta, wires=[0])
        return qml.state()

    # set up Qiskit realization of PRx (`r`) gate
    qc = QuantumCircuit(1)
    qc.r(phi, theta, 0)

    # simulate PennyLane results
    state_pennylane = circuit(torch.tensor(phi, requires_grad=True), torch.tensor(theta, requires_grad=True))
    print(f'PennyLane State: {state_pennylane.detach().numpy()}')  # noqa

    # simulate Qiskit results
    simulator = StatevectorSimulator()
    state_qiskit = simulator.run(qc).result().get_statevector().data
    print(f'Qiskit State:    {state_qiskit}')


def get_gateset(provider: str = 'ibmq'):
    if 'ibmq' == provider:
        """
        Initial layer has 3 parameters per wire, each block layer has 2*3+0=6 parameters
        """
        return ([qml.RZ, qml.RX, qml.RZ], qml.CZ), (3, 6), '[Rz, Rx, Rz], CZ'
    elif 'iqm' == provider:
        """
        Initial layer has 2 parameters per wire, each block layer has 2*2+0=4 parameters
        """
        return ([PRx], qml.CZ), (2, 4), '[PRx], CZ'
    else:
        raise NotImplementedError(f'Gateset for provider {provider} not defined.')



if __name__ == '__main__':
    test_PRx()
    pass
