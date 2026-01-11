"""
Vector Representation Module

Transforms timing-amplitude data into various mathematical representations
for pattern discovery through linear algebra and complex analysis.

This is the core mathematical innovation: treating each drum hit as a
2D vector (timing_deviation, amplitude) and analyzing the resulting
geometric structures.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import h5py

logger = logging.getLogger(__name__)


@dataclass
class VectorRepresentations:
    """Container for all vector representations of a groove."""

    # Core representation: N x 2 matrix of (deviation, amplitude) pairs
    hit_matrix: np.ndarray

    # Complex number representation: z_i = amplitude * exp(i * 2π * deviation / period)
    complex_sequence: np.ndarray

    # Trajectory matrix for phase space analysis
    trajectory_matrix: Optional[np.ndarray] = None

    # Lag matrices for autocorrelation analysis
    lag_matrices: Optional[dict] = None

    # Bar-structured 3D representation
    bar_tensor: Optional[np.ndarray] = None

    # Metadata
    num_hits: int = 0
    grid_interval_ms: float = 0.0

    def save_numpy(self, directory: Path) -> None:
        """Save representations as numpy files."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        np.save(directory / 'hit_matrix.npy', self.hit_matrix)
        np.save(directory / 'complex_sequence.npy', self.complex_sequence)

        if self.trajectory_matrix is not None:
            np.save(directory / 'trajectory_matrix.npy', self.trajectory_matrix)

        if self.bar_tensor is not None:
            np.save(directory / 'bar_tensor.npy', self.bar_tensor)

        logger.info(f"Saved numpy representations to {directory}")

    def save_hdf5(self, path: Path) -> None:
        """Save all representations to HDF5 file."""
        with h5py.File(path, 'w') as f:
            f.create_dataset('hit_matrix', data=self.hit_matrix)
            f.create_dataset('complex_sequence', data=self.complex_sequence)

            if self.trajectory_matrix is not None:
                f.create_dataset('trajectory_matrix', data=self.trajectory_matrix)

            if self.bar_tensor is not None:
                f.create_dataset('bar_tensor', data=self.bar_tensor)

            if self.lag_matrices is not None:
                lag_group = f.create_group('lag_matrices')
                for lag, matrix in self.lag_matrices.items():
                    lag_group.create_dataset(f'lag_{lag}', data=matrix)

            f.attrs['num_hits'] = self.num_hits
            f.attrs['grid_interval_ms'] = self.grid_interval_ms

        logger.info(f"Saved HDF5 representations to {path}")

    @classmethod
    def load_hdf5(cls, path: Path) -> 'VectorRepresentations':
        """Load representations from HDF5 file."""
        with h5py.File(path, 'r') as f:
            hit_matrix = f['hit_matrix'][:]
            complex_sequence = f['complex_sequence'][:]

            trajectory_matrix = None
            if 'trajectory_matrix' in f:
                trajectory_matrix = f['trajectory_matrix'][:]

            bar_tensor = None
            if 'bar_tensor' in f:
                bar_tensor = f['bar_tensor'][:]

            lag_matrices = None
            if 'lag_matrices' in f:
                lag_matrices = {
                    int(key.split('_')[1]): f['lag_matrices'][key][:]
                    for key in f['lag_matrices'].keys()
                }

            return cls(
                hit_matrix=hit_matrix,
                complex_sequence=complex_sequence,
                trajectory_matrix=trajectory_matrix,
                lag_matrices=lag_matrices,
                bar_tensor=bar_tensor,
                num_hits=f.attrs['num_hits'],
                grid_interval_ms=f.attrs['grid_interval_ms'],
            )


class GrooveVectorizer:
    """
    Transform timing-amplitude data into mathematical representations.

    The key insight is treating each drum hit as a 2D vector in
    (timing_deviation, amplitude) space. This enables:

    1. Geometric analysis: vector magnitudes, angles, clustering
    2. Complex analysis: phase relationships, circular statistics
    3. Phase space methods: trajectory analysis, attractors
    4. Matrix methods: PCA, eigenvalue analysis, correlation structure

    Parameters
    ----------
    normalize_deviation : bool, default=True
        If True, use normalized deviations (-0.5 to 0.5).
        If False, use millisecond deviations.
    """

    def __init__(self, normalize_deviation: bool = True):
        self.normalize_deviation = normalize_deviation

    def create_representations(
        self,
        deviations: np.ndarray,
        amplitudes: np.ndarray,
        grid_interval_ms: float,
        grid_positions: Optional[np.ndarray] = None,
        beats_per_bar: int = 4,
        grid_subdivision: int = 16,
    ) -> VectorRepresentations:
        """
        Create all vector representations from timing-amplitude data.

        Parameters
        ----------
        deviations : np.ndarray
            Timing deviations (normalized or ms, depending on normalize_deviation)
        amplitudes : np.ndarray
            Normalized amplitude values (0-1)
        grid_interval_ms : float
            Grid interval in milliseconds
        grid_positions : np.ndarray, optional
            Grid position indices for bar structuring
        beats_per_bar : int
            Beats per bar for bar tensor creation
        grid_subdivision : int
            Grid subdivision (e.g., 16 for 16th notes)

        Returns
        -------
        VectorRepresentations
        """
        n_hits = len(deviations)

        # 1. Core hit matrix: N x 2
        hit_matrix = self.create_hit_matrix(deviations, amplitudes)

        # 2. Complex sequence
        complex_seq = self.to_complex_sequence(
            deviations, amplitudes, grid_interval_ms
        )

        # 3. Trajectory matrix for phase space
        trajectory = self.create_trajectory_matrix(hit_matrix, lag=1)

        # 4. Lag matrices
        lag_matrices = {
            1: self.create_lag_matrix(hit_matrix, 1),
            2: self.create_lag_matrix(hit_matrix, 2),
            4: self.create_lag_matrix(hit_matrix, 4),
        }

        # 5. Bar tensor (if grid positions provided)
        bar_tensor = None
        if grid_positions is not None:
            bar_tensor = self.reshape_to_bars(
                deviations, amplitudes, grid_positions,
                beats_per_bar, grid_subdivision
            )

        return VectorRepresentations(
            hit_matrix=hit_matrix,
            complex_sequence=complex_seq,
            trajectory_matrix=trajectory,
            lag_matrices=lag_matrices,
            bar_tensor=bar_tensor,
            num_hits=n_hits,
            grid_interval_ms=grid_interval_ms,
        )

    def create_hit_matrix(
        self,
        deviations: np.ndarray,
        amplitudes: np.ndarray,
    ) -> np.ndarray:
        """
        Create the core N x 2 hit matrix.

        Each row is a vector [Δt, v] representing one drum hit.
        The resulting matrix can be analyzed using standard linear algebra.

        Returns
        -------
        np.ndarray
            Shape (N, 2) where columns are [deviation, amplitude]
        """
        return np.column_stack([deviations, amplitudes])

    def to_complex_sequence(
        self,
        deviations: np.ndarray,
        amplitudes: np.ndarray,
        grid_interval_ms: float,
    ) -> np.ndarray:
        """
        Convert hits to complex number representation.

        Each hit becomes: z = amplitude * exp(i * 2π * deviation / grid_interval)

        This maps timing deviation to phase angle, allowing:
        - Circular statistics on timing
        - Phase coherence analysis
        - Frequency domain methods

        The real part encodes "on-time amplitude" and imaginary part
        encodes "off-time amplitude" in a sense.

        Returns
        -------
        np.ndarray
            Complex array of length N
        """
        if self.normalize_deviation:
            # Deviation is already -0.5 to 0.5, so phase is -π to π
            phase = 2 * np.pi * deviations
        else:
            # Convert ms to phase
            phase = 2 * np.pi * deviations / grid_interval_ms

        return amplitudes * np.exp(1j * phase)

    def create_trajectory_matrix(
        self,
        hit_matrix: np.ndarray,
        lag: int = 1,
    ) -> np.ndarray:
        """
        Create trajectory matrix for phase space analysis.

        Each row is [Δt_n, v_n, Δt_{n+lag}, v_{n+lag}].

        This reveals:
        - Sequential dependencies between hits
        - Compensatory timing patterns
        - Attractor structures in groove

        Parameters
        ----------
        hit_matrix : np.ndarray
            N x 2 hit matrix
        lag : int
            Number of hits to skip for the second point

        Returns
        -------
        np.ndarray
            Shape (N-lag, 4)
        """
        n = len(hit_matrix)
        if n <= lag:
            return np.empty((0, 4))

        # Concatenate current and lagged states
        current = hit_matrix[:-lag]
        lagged = hit_matrix[lag:]

        return np.hstack([current, lagged])

    def create_lag_matrix(
        self,
        hit_matrix: np.ndarray,
        lag: int,
    ) -> np.ndarray:
        """
        Create lag-k matrix for autocorrelation analysis.

        Returns pairs of hit vectors separated by k positions.

        Returns
        -------
        np.ndarray
            Shape (N-lag, 2, 2) where [i, 0, :] is hit i and [i, 1, :] is hit i+lag
        """
        n = len(hit_matrix)
        if n <= lag:
            return np.empty((0, 2, 2))

        result = np.zeros((n - lag, 2, 2))
        result[:, 0, :] = hit_matrix[:-lag]
        result[:, 1, :] = hit_matrix[lag:]

        return result

    def reshape_to_bars(
        self,
        deviations: np.ndarray,
        amplitudes: np.ndarray,
        grid_positions: np.ndarray,
        beats_per_bar: int = 4,
        grid_subdivision: int = 16,
    ) -> Optional[np.ndarray]:
        """
        Reshape data into bar-structured 3D tensor.

        This enables analysis of:
        - Cross-bar consistency
        - Position-specific patterns (e.g., always late on beat 2)
        - Bar-to-bar evolution

        Returns
        -------
        np.ndarray or None
            Shape (num_bars, positions_per_bar, 2) or None if incomplete bars
        """
        grids_per_beat = grid_subdivision // 4
        positions_per_bar = grids_per_beat * beats_per_bar

        # Create a sparse representation first
        max_pos = int(grid_positions.max()) + 1
        full_dev = np.full(max_pos, np.nan)
        full_amp = np.full(max_pos, np.nan)

        for i, pos in enumerate(grid_positions):
            full_dev[pos] = deviations[i]
            full_amp[pos] = amplitudes[i]

        # Calculate number of complete bars
        num_bars = max_pos // positions_per_bar
        if num_bars == 0:
            return None

        # Trim to complete bars
        trim_length = num_bars * positions_per_bar
        full_dev = full_dev[:trim_length]
        full_amp = full_amp[:trim_length]

        # Reshape to (num_bars, positions_per_bar)
        dev_bars = full_dev.reshape(num_bars, positions_per_bar)
        amp_bars = full_amp.reshape(num_bars, positions_per_bar)

        # Stack to get (num_bars, positions_per_bar, 2)
        return np.stack([dev_bars, amp_bars], axis=-1)

    def compute_vector_statistics(
        self,
        hit_matrix: np.ndarray,
    ) -> dict:
        """
        Compute basic statistics on the hit matrix.

        Returns
        -------
        dict
            Statistics including means, stds, correlations, and geometric properties
        """
        deviations = hit_matrix[:, 0]
        amplitudes = hit_matrix[:, 1]

        # Basic statistics
        stats = {
            'n_hits': len(hit_matrix),
            'timing': {
                'mean': float(np.mean(deviations)),
                'std': float(np.std(deviations)),
                'min': float(np.min(deviations)),
                'max': float(np.max(deviations)),
                'median': float(np.median(deviations)),
            },
            'amplitude': {
                'mean': float(np.mean(amplitudes)),
                'std': float(np.std(amplitudes)),
                'min': float(np.min(amplitudes)),
                'max': float(np.max(amplitudes)),
                'median': float(np.median(amplitudes)),
            },
        }

        # Correlation
        if len(hit_matrix) > 2:
            corr = np.corrcoef(deviations, amplitudes)[0, 1]
            stats['correlation'] = float(corr) if not np.isnan(corr) else 0.0

            # Covariance matrix and eigenvalues
            cov_matrix = np.cov(hit_matrix.T)
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            stats['covariance_matrix'] = cov_matrix.tolist()
            stats['eigenvalues'] = eigenvalues.tolist()

            # Geometric properties
            # Vector magnitudes (distance from origin)
            magnitudes = np.sqrt(deviations**2 + amplitudes**2)
            stats['magnitude'] = {
                'mean': float(np.mean(magnitudes)),
                'std': float(np.std(magnitudes)),
            }

            # Vector angles (direction in timing-amplitude space)
            angles = np.arctan2(amplitudes, deviations)
            stats['angle'] = {
                'mean': float(np.mean(angles)),
                'std': float(np.std(angles)),
                'circular_mean': float(np.angle(np.mean(np.exp(1j * angles)))),
            }

            # Centroid
            stats['centroid'] = {
                'timing': float(np.mean(deviations)),
                'amplitude': float(np.mean(amplitudes)),
            }

        return stats

    def compute_complex_statistics(
        self,
        complex_sequence: np.ndarray,
    ) -> dict:
        """
        Compute statistics on the complex representation.

        Returns
        -------
        dict
            Statistics including phase coherence, mean resultant length, etc.
        """
        # Mean resultant length (phase coherence)
        mean_vector = np.mean(complex_sequence)
        mean_resultant_length = np.abs(mean_vector)
        mean_phase = np.angle(mean_vector)

        # Individual magnitudes and phases
        magnitudes = np.abs(complex_sequence)
        phases = np.angle(complex_sequence)

        # Circular variance
        circular_variance = 1 - mean_resultant_length / np.mean(magnitudes)

        return {
            'mean_resultant_length': float(mean_resultant_length),
            'mean_phase': float(mean_phase),
            'phase_coherence': float(mean_resultant_length / np.mean(magnitudes)),
            'circular_variance': float(circular_variance),
            'total_power': float(np.sum(magnitudes**2)),
            'magnitude_stats': {
                'mean': float(np.mean(magnitudes)),
                'std': float(np.std(magnitudes)),
            },
            'phase_stats': {
                'mean': float(np.mean(phases)),
                'std': float(np.std(phases)),
            },
        }

    def save_statistics(
        self,
        representations: VectorRepresentations,
        path: Path,
    ) -> None:
        """Compute and save all statistics to JSON."""
        stats = {
            'vector_stats': self.compute_vector_statistics(representations.hit_matrix),
            'complex_stats': self.compute_complex_statistics(representations.complex_sequence),
        }

        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved statistics to {path}")
