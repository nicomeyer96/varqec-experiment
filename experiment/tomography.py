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

from .qiskit_experiments_custom import MitigatedStateTomography, ParallelExperiment
from .load_model import state_initialization_circuits, load_circuits


def construct_baseline_experiments(delay: float, backend, physical_wires, virtualize_delay_with_rx: bool = False,
                                   resample: bool = False):
    # construct single-qubit state initialization circuits (one for each state of the 6-element spherical 2-design)
    physical_circuits = state_initialization_circuits([0], delay=delay,
                                                      virtualize_delay_with_rx=virtualize_delay_with_rx)
    # construct tomography experiments
    state_tomography_baseline_experiments = []
    for qc in physical_circuits:
        state_tomography_baseline_experiments.append(_construct_baseline_experiment(
            qc=qc, backend=backend,
            physical_wires_flattened=[physical_wire for physical_wires_ in physical_wires
                                      for physical_wire in physical_wires_],  # flatten for individual tomography
            resample=resample
        ))
    return state_tomography_baseline_experiments


def _construct_baseline_experiment(qc, backend, physical_wires_flattened, resample: bool):
    experiments = []
    for physical_wire in physical_wires_flattened:
        state_tomography = MitigatedStateTomography(qc, backend=backend, physical_qubits=[physical_wire],
                                                    resample=resample)
        state_tomography.set_transpile_options(optimization_level=0)  # already trained for specific gateset and layout
        experiments.append(state_tomography)
    # run all experiments in parallel (all operations are commuting)
    return ParallelExperiment(experiments)  # noqa


def construct_experiments(model: str, provider: str, delay: float, backend, physical_wires,
                          virtualize_delay_with_rx: bool = False, resample: bool = False, layout: str = None):
    # construct pre-trained encoding circuits (one for each state of the 6-element spherical 2-design)
    encoding_circuits, _ = load_circuits(model=model, provider=provider, delay=delay,
                                         virtualize_delay_with_rx=virtualize_delay_with_rx, layout=layout)
    # construct tomography experiments
    state_tomography_experiments = []
    for qc in encoding_circuits:
        state_tomography_experiments.append(_construct_experiment(
            qc=qc, backend=backend, physical_wires=physical_wires, resample=resample
        ))
    return state_tomography_experiments


def _construct_experiment(qc, backend, physical_wires, resample):
    experiments = []
    for patch_wires in physical_wires:
        state_tomography = MitigatedStateTomography(qc, backend=backend, physical_qubits=patch_wires, resample=resample)
        state_tomography.set_transpile_options(optimization_level=0)  # already trained for specific gateset and layout
        experiments.append(state_tomography)
    # run all experiments in parallel (all operations are commuting)
    return ParallelExperiment(experiments)  # noqa
