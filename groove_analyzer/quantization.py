"""
Quantization Module

Calculates timing deviations from ideal grid positions.
Converts raw onset times to (deviation, amplitude) pairs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QuantizationResult:
    """Results from grid quantization."""

    grid_positions: np.ndarray  # Nearest grid position index for each onset
    actual_times: np.ndarray  # Original onset times in seconds
    deviations_ms: np.ndarray  # Timing deviation in milliseconds
    deviations_normalized: np.ndarray  # Deviation as fraction of grid interval
    amplitudes_raw: np.ndarray  # Original amplitude values
    amplitudes_normalized: np.ndarray  # Normalized 0-1
    tempo_bpm: float
    grid_subdivision: int
    grid_interval_ms: float  # Time between grid positions in ms

    def __len__(self) -> int:
        return len(self.actual_times)

    @property
    def grid_times(self) -> np.ndarray:
        """Ideal grid times in seconds."""
        return self.grid_positions * (self.grid_interval_ms / 1000)

    def to_csv(self, path: Path) -> None:
        """Export quantization data to CSV."""
        import pandas as pd
        df = pd.DataFrame({
            'grid_position': self.grid_positions,
            'grid_time_s': self.grid_times,
            'actual_time_s': self.actual_times,
            'deviation_ms': self.deviations_ms,
            'deviation_normalized': self.deviations_normalized,
            'amplitude_raw': self.amplitudes_raw,
            'amplitude_normalized': self.amplitudes_normalized,
        })
        df.to_csv(path, index=False)
        logger.info(f"Saved quantization data to {path}")

    def save_params(self, path: Path) -> None:
        """Save quantization parameters to JSON."""
        params = {
            'tempo_bpm': self.tempo_bpm,
            'grid_subdivision': self.grid_subdivision,
            'grid_interval_ms': self.grid_interval_ms,
            'num_hits': len(self),
            'deviation_stats': {
                'mean_ms': float(np.mean(self.deviations_ms)),
                'std_ms': float(np.std(self.deviations_ms)),
                'min_ms': float(np.min(self.deviations_ms)),
                'max_ms': float(np.max(self.deviations_ms)),
            }
        }
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)


class GridQuantizer:
    """
    Calculate timing deviations from ideal grid positions.

    Aligns onset times to a metrical grid and computes how much
    each hit deviates from perfect quantization.

    Parameters
    ----------
    tempo_bpm : float
        Known tempo in beats per minute
    grid_subdivision : int, default=16
        Grid resolution per beat:
        - 4 = quarter notes
        - 8 = 8th notes
        - 16 = 16th notes
        - 32 = 32nd notes
    first_downbeat : float, default=0.0
        Time of first downbeat in seconds (for alignment)
    """

    def __init__(
        self,
        tempo_bpm: float,
        grid_subdivision: int = 16,
        first_downbeat: float = 0.0,
    ):
        self.tempo_bpm = tempo_bpm
        self.grid_subdivision = grid_subdivision
        self.first_downbeat = first_downbeat

        # Calculate grid interval
        beat_duration_s = 60.0 / tempo_bpm
        self.grid_interval_s = beat_duration_s / (grid_subdivision / 4)
        self.grid_interval_ms = self.grid_interval_s * 1000

        logger.info(
            f"Grid: {tempo_bpm} BPM, {grid_subdivision}th notes, "
            f"{self.grid_interval_ms:.2f}ms interval"
        )

    def quantize(
        self,
        onset_times: np.ndarray,
        onset_amplitudes: np.ndarray,
        amplitude_norm: Literal['minmax', 'zscore', 'none'] = 'minmax',
    ) -> QuantizationResult:
        """
        Quantize onset times to grid and calculate deviations.

        Parameters
        ----------
        onset_times : np.ndarray
            Onset times in seconds
        onset_amplitudes : np.ndarray
            Amplitude values for each onset
        amplitude_norm : str
            Amplitude normalization method:
            - 'minmax': Scale to [0, 1]
            - 'zscore': Standardize to mean=0, std=1
            - 'none': Keep raw values

        Returns
        -------
        QuantizationResult
        """
        # Adjust for first downbeat
        adjusted_times = onset_times - self.first_downbeat

        # Find nearest grid position for each onset
        grid_positions = np.round(adjusted_times / self.grid_interval_s).astype(int)

        # Calculate ideal grid times
        ideal_times = grid_positions * self.grid_interval_s

        # Calculate deviations
        deviations_s = adjusted_times - ideal_times
        deviations_ms = deviations_s * 1000

        # Normalize deviations as fraction of grid interval (-0.5 to +0.5)
        deviations_normalized = deviations_s / self.grid_interval_s

        # Normalize amplitudes
        amplitudes_normalized = self._normalize_amplitudes(
            onset_amplitudes, method=amplitude_norm
        )

        # Warn about potential issues
        self._check_quantization_quality(deviations_normalized, grid_positions)

        return QuantizationResult(
            grid_positions=grid_positions,
            actual_times=onset_times,
            deviations_ms=deviations_ms,
            deviations_normalized=deviations_normalized,
            amplitudes_raw=onset_amplitudes,
            amplitudes_normalized=amplitudes_normalized,
            tempo_bpm=self.tempo_bpm,
            grid_subdivision=self.grid_subdivision,
            grid_interval_ms=self.grid_interval_ms,
        )

    def _normalize_amplitudes(
        self,
        amplitudes: np.ndarray,
        method: str,
    ) -> np.ndarray:
        """Normalize amplitude values."""
        if method == 'minmax':
            min_val = amplitudes.min()
            max_val = amplitudes.max()
            if max_val - min_val > 1e-8:
                return (amplitudes - min_val) / (max_val - min_val)
            return np.ones_like(amplitudes) * 0.5
        elif method == 'zscore':
            std = amplitudes.std()
            if std > 1e-8:
                return (amplitudes - amplitudes.mean()) / std
            return np.zeros_like(amplitudes)
        else:
            return amplitudes.copy()

    def _check_quantization_quality(
        self,
        deviations_normalized: np.ndarray,
        grid_positions: np.ndarray,
    ) -> None:
        """Check for potential quantization issues."""
        # Check for hits very close to grid boundary
        boundary_hits = np.abs(np.abs(deviations_normalized) - 0.5) < 0.05
        if boundary_hits.sum() > 0:
            logger.warning(
                f"{boundary_hits.sum()} hits near grid boundary "
                "(may be assigned to wrong grid position)"
            )

        # Check for duplicate grid positions
        unique, counts = np.unique(grid_positions, return_counts=True)
        duplicates = (counts > 1).sum()
        if duplicates > 0:
            logger.warning(
                f"{duplicates} grid positions have multiple hits "
                "(flams or detection artifacts)"
            )

        # Check for large deviations
        large_dev = np.abs(deviations_normalized) > 0.4
        if large_dev.sum() > 0:
            logger.warning(
                f"{large_dev.sum()} hits have large deviations (>40% of grid interval)"
            )

    def estimate_tempo(
        self,
        onset_times: np.ndarray,
        tempo_range: tuple = (60, 200),
    ) -> float:
        """
        Estimate tempo from onset times using autocorrelation.

        Useful for verifying the provided tempo or detecting drift.

        Parameters
        ----------
        onset_times : np.ndarray
            Onset times in seconds
        tempo_range : tuple
            Min and max BPM to consider

        Returns
        -------
        float
            Estimated tempo in BPM
        """
        # Calculate inter-onset intervals
        iois = np.diff(onset_times)

        # Create histogram of IOIs
        min_ioi = 60.0 / tempo_range[1]  # Fastest tempo
        max_ioi = 60.0 / tempo_range[0]  # Slowest tempo

        # Find peaks in IOI distribution
        iois_valid = iois[(iois >= min_ioi / 4) & (iois <= max_ioi)]

        if len(iois_valid) == 0:
            logger.warning("Could not estimate tempo from onsets")
            return self.tempo_bpm

        # Use median IOI as estimate of subdivision
        median_ioi = np.median(iois_valid)

        # Convert to BPM (assuming 16th notes are most common)
        estimated_bpm = 60.0 / (median_ioi * 4)

        # Adjust to be within range
        while estimated_bpm < tempo_range[0]:
            estimated_bpm *= 2
        while estimated_bpm > tempo_range[1]:
            estimated_bpm /= 2

        logger.info(
            f"Estimated tempo: {estimated_bpm:.1f} BPM "
            f"(provided: {self.tempo_bpm} BPM)"
        )

        if abs(estimated_bpm - self.tempo_bpm) > 5:
            logger.warning(
                f"Tempo mismatch: estimated {estimated_bpm:.1f} vs "
                f"provided {self.tempo_bpm:.1f} BPM"
            )

        return estimated_bpm

    def calculate_swing_ratio(
        self,
        grid_positions: np.ndarray,
        deviations_ms: np.ndarray,
    ) -> Optional[float]:
        """
        Calculate swing ratio from 8th note pairs.

        Swing ratio = duration of long note / duration of short note
        Straight = 1.0, Triplet swing = 2.0, Heavy swing > 2.0

        Returns None if not enough 8th note pairs found.
        """
        # Find consecutive 8th note positions (for 16th grid)
        if self.grid_subdivision == 16:
            # 8th notes are at positions 0, 2, 4, 6, ...
            eighth_positions = grid_positions[grid_positions % 2 == 0]
            offbeat_positions = grid_positions[grid_positions % 2 == 1]

            if len(eighth_positions) < 4 or len(offbeat_positions) < 4:
                return None

            # Get deviations for on-beats and off-beats
            on_mask = grid_positions % 2 == 0
            off_mask = grid_positions % 2 == 1

            on_dev_mean = np.mean(deviations_ms[on_mask])
            off_dev_mean = np.mean(deviations_ms[off_mask])

            # Swing pushes off-beats late
            swing_offset_ms = off_dev_mean - on_dev_mean
            base_interval = self.grid_interval_ms

            # Calculate ratio
            long_duration = base_interval + swing_offset_ms
            short_duration = base_interval - swing_offset_ms

            if short_duration > 0:
                swing_ratio = long_duration / short_duration
                logger.info(f"Swing ratio: {swing_ratio:.2f}")
                return swing_ratio

        return None

    def get_grid_timeline(
        self,
        num_positions: int,
        beats_per_bar: int = 4,
    ) -> dict:
        """
        Generate a timeline of grid positions with beat/bar information.

        Returns
        -------
        dict
            Contains 'times', 'beat_in_bar', 'bar_number', 'is_downbeat'
        """
        positions = np.arange(num_positions)
        times = positions * self.grid_interval_s + self.first_downbeat

        grids_per_beat = self.grid_subdivision // 4
        grids_per_bar = grids_per_beat * beats_per_bar

        beat_in_bar = (positions // grids_per_beat) % beats_per_bar
        bar_number = positions // grids_per_bar
        is_downbeat = (positions % grids_per_bar) == 0

        return {
            'positions': positions,
            'times': times,
            'beat_in_bar': beat_in_bar,
            'bar_number': bar_number,
            'is_downbeat': is_downbeat,
        }
