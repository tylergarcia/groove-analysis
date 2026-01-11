"""
Onset Detection Module

Extracts drum hit timing and amplitude from audio files using librosa.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class OnsetDetectionResult:
    """Results from onset detection."""

    onset_times: np.ndarray  # Time in seconds
    onset_strengths: np.ndarray  # Normalized 0-1
    onset_amplitudes: np.ndarray  # RMS amplitude at each onset
    sample_rate: int
    hop_length: int
    detection_params: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.onset_times)

    def to_csv(self, path: Path) -> None:
        """Export onset data to CSV."""
        import pandas as pd
        df = pd.DataFrame({
            'timestamp_s': self.onset_times,
            'amplitude': self.onset_amplitudes,
            'onset_strength': self.onset_strengths,
        })
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(self)} onsets to {path}")

    def save_params(self, path: Path) -> None:
        """Save detection parameters to JSON."""
        with open(path, 'w') as f:
            json.dump(self.detection_params, f, indent=2)


class OnsetDetector:
    """
    Extract drum hit timing and amplitude from audio files.

    Uses librosa's onset detection with configurable parameters.
    Optimized for percussive/drum content.

    Parameters
    ----------
    hop_length : int, default=256
        Analysis hop size in samples (affects temporal resolution).
        256 @ 44100 Hz = ~5.8ms resolution
    onset_threshold : float, default=0.1
        Minimum onset strength (normalized 0-1) to consider as hit
    backtrack : bool, default=True
        Whether to backtrack onset times to local energy minimum
    """

    def __init__(
        self,
        hop_length: int = 256,
        onset_threshold: float = 0.1,
        backtrack: bool = True,
    ):
        self.hop_length = hop_length
        self.onset_threshold = onset_threshold
        self.backtrack = backtrack

    def detect_onsets(
        self,
        audio_path: str | Path,
        sr: int = 44100,
    ) -> OnsetDetectionResult:
        """
        Detect onsets in an audio file.

        Parameters
        ----------
        audio_path : str or Path
            Path to WAV file
        sr : int, default=44100
            Sample rate for loading audio

        Returns
        -------
        OnsetDetectionResult
            Contains onset times, strengths, and amplitudes
        """
        audio_path = Path(audio_path)
        logger.info(f"Loading audio from {audio_path}")

        # Load audio
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        duration = len(y) / sr
        logger.info(f"Loaded {duration:.2f}s of audio at {sr} Hz")

        # Compute onset envelope (optimized for percussion)
        onset_env = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=self.hop_length,
            aggregate=np.median,  # More robust for drums
        )

        # Normalize onset envelope
        onset_env_norm = onset_env / (onset_env.max() + 1e-8)

        # Detect onset frames
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length,
            backtrack=self.backtrack,
            units='frames',
        )

        # Convert to times
        onset_times = librosa.frames_to_time(
            onset_frames,
            sr=sr,
            hop_length=self.hop_length,
        )

        # Get onset strengths at detected positions
        onset_strengths = onset_env_norm[onset_frames]

        # Filter by threshold
        mask = onset_strengths >= self.onset_threshold
        onset_times = onset_times[mask]
        onset_frames = onset_frames[mask]
        onset_strengths = onset_strengths[mask]

        # Calculate amplitude at each onset using RMS in a window around onset
        onset_amplitudes = self._calculate_onset_amplitudes(
            y, onset_times, sr, window_ms=20
        )

        # Normalize amplitudes to 0-1
        onset_amplitudes = onset_amplitudes / (onset_amplitudes.max() + 1e-8)

        logger.info(f"Detected {len(onset_times)} onsets")

        return OnsetDetectionResult(
            onset_times=onset_times,
            onset_strengths=onset_strengths,
            onset_amplitudes=onset_amplitudes,
            sample_rate=sr,
            hop_length=self.hop_length,
            detection_params={
                'audio_file': str(audio_path),
                'sample_rate': sr,
                'hop_length': self.hop_length,
                'onset_threshold': self.onset_threshold,
                'backtrack': self.backtrack,
                'duration_s': duration,
                'num_onsets': len(onset_times),
            }
        )

    def _calculate_onset_amplitudes(
        self,
        y: np.ndarray,
        onset_times: np.ndarray,
        sr: int,
        window_ms: float = 20,
    ) -> np.ndarray:
        """Calculate RMS amplitude in a window after each onset."""
        window_samples = int(sr * window_ms / 1000)
        amplitudes = np.zeros(len(onset_times))

        for i, t in enumerate(onset_times):
            start = int(t * sr)
            end = min(start + window_samples, len(y))
            if start < len(y):
                amplitudes[i] = np.sqrt(np.mean(y[start:end] ** 2))

        return amplitudes

    def visualize_detection(
        self,
        audio_path: str | Path,
        result: OnsetDetectionResult,
        output_path: Optional[Path] = None,
        figsize: tuple = (14, 6),
    ) -> plt.Figure:
        """
        Overlay detected onsets on waveform for visual verification.

        Parameters
        ----------
        audio_path : str or Path
            Path to audio file
        result : OnsetDetectionResult
            Detection results to visualize
        output_path : Path, optional
            If provided, save figure to this path
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.Figure
        """
        y, sr = librosa.load(audio_path, sr=result.sample_rate, mono=True)
        times = np.arange(len(y)) / sr

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Waveform with onset markers
        ax1 = axes[0]
        ax1.plot(times, y, color='steelblue', alpha=0.7, linewidth=0.5)
        for t in result.onset_times:
            ax1.axvline(t, color='red', alpha=0.7, linewidth=0.8)
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Waveform with {len(result)} Detected Onsets')
        ax1.set_xlim(0, times[-1])

        # Onset strength envelope
        ax2 = axes[1]
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=self.hop_length
        )
        onset_times_env = librosa.times_like(onset_env, sr=sr, hop_length=self.hop_length)
        ax2.plot(onset_times_env, onset_env, color='green', alpha=0.8)
        ax2.axhline(
            self.onset_threshold * onset_env.max(),
            color='orange',
            linestyle='--',
            label=f'Threshold ({self.onset_threshold})'
        )
        for t in result.onset_times:
            ax2.axvline(t, color='red', alpha=0.5, linewidth=0.8)
        ax2.set_ylabel('Onset Strength')
        ax2.set_xlabel('Time (s)')
        ax2.legend(loc='upper right')

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved detection visualization to {output_path}")

        return fig
