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
import torch


def distinguishability_loss(rho, rho_groundtruth, display: str = None, memory_efficient: bool = True):
    """
    Distinguishability loss for training encodings on specific noise models.
    Computes the absolute traces between pairs of states (provided as density matrices) and returns average and maximum.

    Note: The noise-free difference is subtracted element-wise, the loss function converges to 0.
    """

    number_density_matrices = rho.shape[0]
    if not number_density_matrices == rho_groundtruth.shape[0]:
        raise ValueError(f'The number of noisy ({number_density_matrices}) and groundtruth density matrices '
                         f'({rho_groundtruth.shape[0]}) does not match.')

    # computes the distinguishability loss without explicitly constructing all combinations of density matrices
    if memory_efficient:  # much more memory-efficient, only slightly slower
        pairwise_lost_distinguishability = []
        # iterate over upper triangle matrix (excluding diagonals)
        for row in range(0, number_density_matrices):
            for col in range(1 + row, number_density_matrices):
                # determine trace for groundtruth first
                trace_groundtruth = qml.math.trace_distance(rho_groundtruth[row], rho_groundtruth[col])

                # determine trace for noisy density matrices
                trace = qml.math.trace_distance(rho[row], rho[col])

                # subtract from the groundtruth traces to get the `loss in distinguishability`
                # (the values are in principle always positive, but the `relu` cleans some numerical inaccuracies)
                pairwise_lost_distinguishability.append(torch.nn.functional.relu(trace_groundtruth - trace))  # noqa

        # stack results and perform sanity check
        lost_distinguishability = torch.stack(pairwise_lost_distinguishability, dim=0)
        if lost_distinguishability.shape[0] != (number_density_matrices ** 2 - number_density_matrices) // 2:
            raise RuntimeError(f'Expected {(number_density_matrices ** 2 - number_density_matrices) // 2} unique pairs,'
                               f' but got {lost_distinguishability.shape[0]}.')

        # extract average and maximum value (take into account symmetry and diagonal elements, which are always 0)
        max_lost_distinguishability = torch.max(lost_distinguishability)  # noqa
        avg_lost_distinguishability = (2 * torch.sum(lost_distinguishability)) / (number_density_matrices ** 2)
    else:  # much more memory-consuming, only slightly faster
        # determine traces for groundtruth first -> construct all possible pairs
        rho_groundtruth_row = rho_groundtruth.repeat(number_density_matrices, 1, 1)
        rho_groundtruth_col = rho_groundtruth.repeat_interleave(number_density_matrices, dim=0)
        traces_groundtruth = qml.math.trace_distance(rho_groundtruth_row, rho_groundtruth_col)

        # determine traces for noisy density matrices -> construct all possible pairs
        rho_row = rho.repeat(number_density_matrices, 1, 1)
        rho_col = rho.repeat_interleave(number_density_matrices, dim=0)
        traces = qml.math.trace_distance(rho_row, rho_col)

        # subtract from the groundtruth traces to get the `loss in distinguishability`
        # (the values are in principle always positive, but the `relu` cleans some numerical inaccuracies)
        lost_distinguishability = torch.nn.functional.relu(traces_groundtruth - traces)  # noqa

        # return (and optionally show) the average + max value
        max_lost_distinguishability = torch.max(lost_distinguishability)  # noqa
        avg_lost_distinguishability = torch.mean(lost_distinguishability)  # noqa

    if display is not None:
        print(f'{display}: AVG={avg_lost_distinguishability.detach():.7f}, MAX={max_lost_distinguishability.detach():.7f}')  # noqa
    return avg_lost_distinguishability, max_lost_distinguishability
