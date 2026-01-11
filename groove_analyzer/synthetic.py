"""
Synthetic Groove Generator

Creates synthetic drum performances with known timing-amplitude patterns
for testing and validation of the analysis system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SyntheticGroove:
    """Synthetic groove data for testing."""

    # Grid positions (which 16th note each hit is on)
    grid_positions: np.ndarray

    # Timing deviations in normalized units (-0.5 to 0.5)
    deviations_normalized: np.ndarray

    # Timing deviations in milliseconds
    deviations_ms: np.ndarray

    # Amplitude values (0 to 1)
    amplitudes: np.ndarray

    # Metadata
    pattern_type: str
    tempo_bpm: float
    grid_subdivision: int
    num_bars: int
    beats_per_bar: int
    random_seed: Optional[int]

    @property
    def grid_interval_ms(self) -> float:
        """Time between grid positions in ms."""
        beat_ms = 60000 / self.tempo_bpm
        return beat_ms / (self.grid_subdivision / 4)

    @property
    def onset_times(self) -> np.ndarray:
        """Absolute onset times in seconds."""
        grid_times = self.grid_positions * (self.grid_interval_ms / 1000)
        return grid_times + (self.deviations_ms / 1000)

    def __len__(self) -> int:
        return len(self.grid_positions)


class SyntheticGrooveGenerator:
    """
    Generate synthetic drum performances with controlled patterns.

    Use these to validate that the analysis system correctly detects
    known patterns and to understand the visualization outputs.

    Parameters
    ----------
    tempo_bpm : float, default=120
        Tempo in beats per minute
    grid_subdivision : int, default=16
        Grid resolution (16 = 16th notes)
    beats_per_bar : int, default=4
        Time signature numerator
    random_seed : int, optional
        Seed for reproducibility
    """

    def __init__(
        self,
        tempo_bpm: float = 120,
        grid_subdivision: int = 16,
        beats_per_bar: int = 4,
        random_seed: Optional[int] = None,
    ):
        self.tempo_bpm = tempo_bpm
        self.grid_subdivision = grid_subdivision
        self.beats_per_bar = beats_per_bar
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

        # Calculate grid interval
        beat_ms = 60000 / tempo_bpm
        self.grid_interval_ms = beat_ms / (grid_subdivision / 4)

        self.grids_per_beat = grid_subdivision // 4
        self.grids_per_bar = self.grids_per_beat * beats_per_bar

    def generate(
        self,
        pattern_type: Literal[
            'perfect', 'swing', 'rushed', 'dragged', 'accent_downbeat',
            'random', 'compensatory', 'pocket', 'push_pull', 'fatback'
        ],
        num_bars: int = 8,
        hit_density: float = 1.0,
        noise_level: float = 0.0,
    ) -> SyntheticGroove:
        """
        Generate a synthetic groove pattern.

        Parameters
        ----------
        pattern_type : str
            Type of groove to generate:
            - 'perfect': Quantized grid, uniform velocity
            - 'swing': Late off-beats (jazz/shuffle feel)
            - 'rushed': Progressively early timing
            - 'dragged': Progressively late timing
            - 'accent_downbeat': Loud on downbeats, soft elsewhere
            - 'random': Random deviations (null hypothesis)
            - 'compensatory': Early hits followed by late (and vice versa)
            - 'pocket': Subtle consistent lay-back
            - 'push_pull': Alternating tension/release
            - 'fatback': Late snare (backbeat) feel

        num_bars : int
            Number of bars to generate
        hit_density : float
            Fraction of grid positions to fill (1.0 = all, 0.5 = half)
        noise_level : float
            Additional random noise to add (std in normalized units)

        Returns
        -------
        SyntheticGroove
        """
        total_grids = num_bars * self.grids_per_bar

        # Generate grid positions based on density
        if hit_density >= 1.0:
            grid_positions = np.arange(total_grids)
        else:
            n_hits = int(total_grids * hit_density)
            grid_positions = np.sort(
                np.random.choice(total_grids, n_hits, replace=False)
            )

        n_hits = len(grid_positions)

        # Generate pattern-specific deviations and amplitudes
        if pattern_type == 'perfect':
            deviations, amplitudes = self._perfect(grid_positions)

        elif pattern_type == 'swing':
            deviations, amplitudes = self._swing(grid_positions)

        elif pattern_type == 'rushed':
            deviations, amplitudes = self._rushed(grid_positions, total_grids)

        elif pattern_type == 'dragged':
            deviations, amplitudes = self._dragged(grid_positions, total_grids)

        elif pattern_type == 'accent_downbeat':
            deviations, amplitudes = self._accent_downbeat(grid_positions)

        elif pattern_type == 'random':
            deviations, amplitudes = self._random(grid_positions)

        elif pattern_type == 'compensatory':
            deviations, amplitudes = self._compensatory(grid_positions)

        elif pattern_type == 'pocket':
            deviations, amplitudes = self._pocket(grid_positions)

        elif pattern_type == 'push_pull':
            deviations, amplitudes = self._push_pull(grid_positions)

        elif pattern_type == 'fatback':
            deviations, amplitudes = self._fatback(grid_positions)

        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

        # Add noise if requested
        if noise_level > 0:
            deviations += np.random.normal(0, noise_level, n_hits)
            amplitudes += np.random.normal(0, noise_level * 0.5, n_hits)
            amplitudes = np.clip(amplitudes, 0.1, 1.0)

        # Clip deviations to valid range
        deviations = np.clip(deviations, -0.5, 0.5)

        # Convert to milliseconds
        deviations_ms = deviations * self.grid_interval_ms

        logger.info(
            f"Generated {pattern_type} pattern: {n_hits} hits, "
            f"{num_bars} bars, dev std={np.std(deviations_ms):.2f}ms"
        )

        return SyntheticGroove(
            grid_positions=grid_positions,
            deviations_normalized=deviations,
            deviations_ms=deviations_ms,
            amplitudes=amplitudes,
            pattern_type=pattern_type,
            tempo_bpm=self.tempo_bpm,
            grid_subdivision=self.grid_subdivision,
            num_bars=num_bars,
            beats_per_bar=self.beats_per_bar,
            random_seed=self.random_seed,
        )

    def _perfect(self, positions: np.ndarray) -> tuple:
        """Perfect quantization, uniform velocity."""
        n = len(positions)
        return np.zeros(n), np.ones(n) * 0.7

    def _swing(
        self,
        positions: np.ndarray,
        swing_amount: float = 0.25,
        swing_ratio: float = 2.0,
    ) -> tuple:
        """
        Swing feel: off-beats pushed late.

        swing_ratio of 2.0 = triplet swing
        swing_ratio of 1.5 = light swing
        """
        n = len(positions)
        deviations = np.zeros(n)
        amplitudes = np.ones(n) * 0.7

        for i, pos in enumerate(positions):
            beat_position = pos % self.grids_per_beat

            # Off-beats (positions 1, 3, 5, 7 for 16th notes)
            if beat_position % 2 == 1:
                deviations[i] = swing_amount
                amplitudes[i] = 0.6  # Slightly softer

            # Downbeats slightly louder
            if pos % self.grids_per_bar == 0:
                amplitudes[i] = 0.9

        return deviations, amplitudes

    def _rushed(self, positions: np.ndarray, total: int) -> tuple:
        """Progressively earlier timing (speeding up)."""
        n = len(positions)
        # Linear progression from 0 to -0.15 (15% early by the end)
        progress = positions / total
        deviations = -0.15 * progress
        amplitudes = np.ones(n) * 0.7
        return deviations, amplitudes

    def _dragged(self, positions: np.ndarray, total: int) -> tuple:
        """Progressively later timing (slowing down)."""
        n = len(positions)
        progress = positions / total
        deviations = 0.15 * progress
        amplitudes = np.ones(n) * 0.7
        return deviations, amplitudes

    def _accent_downbeat(self, positions: np.ndarray) -> tuple:
        """Strong accents on downbeats, ghost notes elsewhere."""
        n = len(positions)
        deviations = np.zeros(n)
        amplitudes = np.ones(n) * 0.4  # Default to ghost note level

        for i, pos in enumerate(positions):
            # Downbeat of bar
            if pos % self.grids_per_bar == 0:
                amplitudes[i] = 1.0
                deviations[i] = 0.02  # Slightly laid back

            # Backbeat (beats 2 and 4)
            elif pos % self.grids_per_beat == 0:
                beat_in_bar = (pos % self.grids_per_bar) // self.grids_per_beat
                if beat_in_bar in [1, 3]:
                    amplitudes[i] = 0.85

        return deviations, amplitudes

    def _random(self, positions: np.ndarray) -> tuple:
        """Random deviations and amplitudes (null hypothesis)."""
        n = len(positions)
        deviations = np.random.normal(0, 0.1, n)  # ~10% std
        amplitudes = np.random.uniform(0.3, 1.0, n)
        return deviations, amplitudes

    def _compensatory(self, positions: np.ndarray) -> tuple:
        """
        Compensatory timing: early hit followed by late hit.

        This creates negative lag-1 autocorrelation in timing.
        Tests if we can detect this temporal dependency.
        """
        n = len(positions)
        deviations = np.zeros(n)
        amplitudes = np.ones(n) * 0.7

        # Generate alternating pattern with some randomness
        for i in range(n):
            if i > 0:
                # Compensate for previous deviation
                deviations[i] = -0.6 * deviations[i-1] + np.random.normal(0, 0.05)
            else:
                deviations[i] = np.random.normal(0, 0.1)

        return deviations, amplitudes

    def _pocket(self, positions: np.ndarray) -> tuple:
        """
        Pocket feel: consistent slight lay-back behind the beat.

        All hits are slightly late, creating a relaxed feel.
        Includes correlation between amplitude and timing.
        """
        n = len(positions)
        base_layback = 0.08  # 8% behind

        # Louder hits are more "on top", softer hits lay back more
        amplitudes = np.random.uniform(0.4, 1.0, n)
        timing_amplitude_coupling = -0.1  # Negative correlation

        deviations = base_layback + timing_amplitude_coupling * (amplitudes - 0.7)
        deviations += np.random.normal(0, 0.03, n)  # Small variation

        return deviations, amplitudes

    def _push_pull(self, positions: np.ndarray) -> tuple:
        """
        Push-pull feel: alternating bars of slightly ahead/behind.

        Creates a wave-like tension and release pattern.
        """
        n = len(positions)
        deviations = np.zeros(n)
        amplitudes = np.ones(n) * 0.7

        for i, pos in enumerate(positions):
            bar_num = pos // self.grids_per_bar
            bar_phase = (pos % self.grids_per_bar) / self.grids_per_bar

            # Sinusoidal push/pull over 2-bar phrase
            phrase_phase = (bar_num % 2) + bar_phase
            deviations[i] = 0.1 * np.sin(np.pi * phrase_phase)

            # Amplitude follows inverse pattern (loud when pushing)
            amplitudes[i] = 0.6 + 0.2 * np.cos(np.pi * phrase_phase)

        return deviations, amplitudes

    def _fatback(self, positions: np.ndarray) -> tuple:
        """
        Fatback feel: snare (backbeat) pushed late for deep pocket.

        Classic R&B/funk drumming style.
        """
        n = len(positions)
        deviations = np.zeros(n)
        amplitudes = np.ones(n) * 0.6

        for i, pos in enumerate(positions):
            beat_in_bar = (pos % self.grids_per_bar) // self.grids_per_beat
            position_in_beat = pos % self.grids_per_beat

            # Kick drum positions (beat 1 and 3) - slightly early, loud
            if position_in_beat == 0 and beat_in_bar in [0, 2]:
                deviations[i] = -0.03
                amplitudes[i] = 0.9

            # Snare/backbeat positions (beat 2 and 4) - late, loud
            elif position_in_beat == 0 and beat_in_bar in [1, 3]:
                deviations[i] = 0.12  # Fat backbeat
                amplitudes[i] = 0.85

            # Hi-hat (all other 8th/16th positions)
            else:
                deviations[i] = np.random.normal(0, 0.02)
                amplitudes[i] = 0.4

        return deviations, amplitudes

    def generate_comparison_set(
        self,
        num_bars: int = 8,
    ) -> dict[str, SyntheticGroove]:
        """
        Generate a set of all pattern types for comparison.

        Returns
        -------
        dict
            Pattern name -> SyntheticGroove
        """
        patterns = [
            'perfect', 'swing', 'rushed', 'dragged', 'accent_downbeat',
            'random', 'compensatory', 'pocket', 'push_pull', 'fatback'
        ]

        return {
            pattern: self.generate(pattern, num_bars=num_bars)
            for pattern in patterns
        }
