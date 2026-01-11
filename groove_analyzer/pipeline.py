"""
Main Analysis Pipeline

Orchestrates the full analysis workflow from audio files or synthetic data
to visualizations and reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .onset_detection import OnsetDetector, OnsetDetectionResult
from .quantization import GridQuantizer, QuantizationResult
from .vector_representation import GrooveVectorizer, VectorRepresentations
from .pattern_analysis import PatternAnalyzer, AnalysisResults
from .visualization import GrooveVisualizer
from .synthetic import SyntheticGrooveGenerator, SyntheticGroove

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the analysis pipeline."""

    # Onset detection
    hop_length: int = 256
    onset_threshold: float = 0.1

    # Quantization
    tempo_bpm: float = 120.0
    grid_subdivision: int = 16
    first_downbeat: float = 0.0

    # Analysis
    run_pca: bool = True
    run_clustering: bool = True
    run_significance: bool = True
    n_clusters: list = None

    # Visualization
    output_formats: list = None

    # Output
    save_intermediate: bool = True

    def __post_init__(self):
        if self.n_clusters is None:
            self.n_clusters = [3, 4, 5]
        if self.output_formats is None:
            self.output_formats = ['png']

    @classmethod
    def from_yaml(cls, path: Path) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        import yaml
        with open(path) as f:
            config_dict = yaml.safe_load(f)

        # Flatten nested config
        flat = {}
        for section in ['onset_detection', 'quantization', 'analysis',
                       'visualization', 'output']:
            if section in config_dict:
                flat.update(config_dict[section])

        return cls(**{k: v for k, v in flat.items() if k in cls.__dataclass_fields__})


class GrooveAnalyzer:
    """
    Main analysis pipeline for drum groove analysis.

    Orchestrates the full workflow:
    1. Onset detection (from audio) or synthetic data generation
    2. Grid quantization
    3. Vector representation
    4. Pattern analysis
    5. Visualization
    6. Report generation

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration
    output_dir : Path
        Directory for outputs
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        output_dir: Optional[Path] = None,
    ):
        self.config = config or PipelineConfig()
        self.output_dir = Path(output_dir) if output_dir else Path('results')

        # Pipeline state
        self.onset_result: Optional[OnsetDetectionResult] = None
        self.quant_result: Optional[QuantizationResult] = None
        self.representations: Optional[VectorRepresentations] = None
        self.analysis_results: Optional[AnalysisResults] = None

        # Create output directories
        self._setup_output_dirs()

    def _setup_output_dirs(self) -> None:
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'raw_data').mkdir(exist_ok=True)
        (self.output_dir / 'representations').mkdir(exist_ok=True)
        (self.output_dir / 'analysis').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)

    def analyze_audio(
        self,
        audio_path: Union[str, Path],
        tempo_bpm: Optional[float] = None,
    ) -> AnalysisResults:
        """
        Run full analysis pipeline on audio file.

        Parameters
        ----------
        audio_path : str or Path
            Path to WAV file
        tempo_bpm : float, optional
            Override tempo from config

        Returns
        -------
        AnalysisResults
        """
        audio_path = Path(audio_path)
        tempo = tempo_bpm or self.config.tempo_bpm

        logger.info(f"Analyzing {audio_path.name} at {tempo} BPM")

        # 1. Onset detection
        detector = OnsetDetector(
            hop_length=self.config.hop_length,
            onset_threshold=self.config.onset_threshold,
        )
        self.onset_result = detector.detect_onsets(audio_path)

        # Save onset visualization
        detector.visualize_detection(
            audio_path, self.onset_result,
            output_path=self.output_dir / 'raw_data' / 'detection_visual.png'
        )

        # 2. Quantize to grid
        quantizer = GridQuantizer(
            tempo_bpm=tempo,
            grid_subdivision=self.config.grid_subdivision,
            first_downbeat=self.config.first_downbeat,
        )
        self.quant_result = quantizer.quantize(
            self.onset_result.onset_times,
            self.onset_result.onset_amplitudes,
        )

        # Continue with common pipeline
        return self._run_analysis_pipeline()

    def analyze_synthetic(
        self,
        pattern_type: str = 'swing',
        num_bars: int = 8,
        tempo_bpm: Optional[float] = None,
        noise_level: float = 0.02,
        random_seed: Optional[int] = None,
    ) -> AnalysisResults:
        """
        Run analysis pipeline on synthetic data.

        Parameters
        ----------
        pattern_type : str
            Type of synthetic pattern to generate
        num_bars : int
            Number of bars
        tempo_bpm : float, optional
            Override tempo from config
        noise_level : float
            Amount of random noise to add
        random_seed : int, optional
            Seed for reproducibility

        Returns
        -------
        AnalysisResults
        """
        tempo = tempo_bpm or self.config.tempo_bpm

        logger.info(f"Generating synthetic '{pattern_type}' pattern")

        # Generate synthetic data
        generator = SyntheticGrooveGenerator(
            tempo_bpm=tempo,
            grid_subdivision=self.config.grid_subdivision,
            random_seed=random_seed,
        )

        synthetic = generator.generate(
            pattern_type=pattern_type,
            num_bars=num_bars,
            noise_level=noise_level,
        )

        # Create a mock quantization result
        self.quant_result = QuantizationResult(
            grid_positions=synthetic.grid_positions,
            actual_times=synthetic.onset_times,
            deviations_ms=synthetic.deviations_ms,
            deviations_normalized=synthetic.deviations_normalized,
            amplitudes_raw=synthetic.amplitudes,
            amplitudes_normalized=synthetic.amplitudes,
            tempo_bpm=tempo,
            grid_subdivision=self.config.grid_subdivision,
            grid_interval_ms=synthetic.grid_interval_ms,
        )

        # Continue with common pipeline
        return self._run_analysis_pipeline()

    def _run_analysis_pipeline(self) -> AnalysisResults:
        """Run analysis from quantized data onwards."""

        # 3. Create vector representations
        vectorizer = GrooveVectorizer(normalize_deviation=True)
        self.representations = vectorizer.create_representations(
            deviations=self.quant_result.deviations_normalized,
            amplitudes=self.quant_result.amplitudes_normalized,
            grid_interval_ms=self.quant_result.grid_interval_ms,
            grid_positions=self.quant_result.grid_positions,
            grid_subdivision=self.config.grid_subdivision,
        )

        # 4. Pattern analysis
        analyzer = PatternAnalyzer(
            hit_matrix=self.representations.hit_matrix,
            complex_sequence=self.representations.complex_sequence,
        )
        self.analysis_results = analyzer.run_full_analysis(
            run_pca=self.config.run_pca,
            run_clustering=self.config.run_clustering,
            run_significance=self.config.run_significance,
            n_clusters=self.config.n_clusters,
        )

        # Detect swing if grid positions available
        swing_result = analyzer.detect_swing(self.quant_result.grid_positions)
        if swing_result:
            self.analysis_results.pattern_matching = {'swing': swing_result}

        # 5. Save intermediate data
        if self.config.save_intermediate:
            self._save_intermediate_data()

        # 6. Generate visualizations
        self._generate_visualizations()

        # 7. Save analysis results
        self.analysis_results.to_json(
            self.output_dir / 'analysis' / 'analysis_results.json'
        )

        # 8. Save metadata
        self._save_metadata()

        logger.info(f"Analysis complete. Results saved to {self.output_dir}")
        return self.analysis_results

    def _save_intermediate_data(self) -> None:
        """Save intermediate data for reproducibility."""
        raw_dir = self.output_dir / 'raw_data'
        rep_dir = self.output_dir / 'representations'

        # Quantized data
        self.quant_result.to_csv(raw_dir / 'quantized.csv')
        self.quant_result.save_params(raw_dir / 'grid_params.json')

        # Vector representations
        self.representations.save_numpy(rep_dir)
        self.representations.save_hdf5(rep_dir / 'representations.h5')

        # Statistics
        vectorizer = GrooveVectorizer()
        vectorizer.save_statistics(
            self.representations,
            rep_dir / 'vector_stats.json'
        )

    def _generate_visualizations(self) -> None:
        """Generate all visualizations."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        visualizer = GrooveVisualizer(self.representations)
        visualizer.save_all_visualizations(
            output_dir=self.output_dir / 'visualizations',
            formats=self.config.output_formats,
        )
        plt.close('all')

    def _save_metadata(self) -> None:
        """Save processing metadata."""
        import sys

        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'tempo_bpm': self.config.tempo_bpm,
                'grid_subdivision': self.config.grid_subdivision,
                'hop_length': self.config.hop_length,
                'onset_threshold': self.config.onset_threshold,
            },
            'data_summary': {
                'n_hits': len(self.quant_result.grid_positions),
                'duration_bars': int(
                    self.quant_result.grid_positions.max() /
                    (self.config.grid_subdivision * 4 / 4) + 1
                ),
            },
            'software_versions': {
                'python': sys.version,
            },
        }

        with open(self.output_dir / 'metadata' / 'processing_log.json', 'w') as f:
            json.dump(metadata, f, indent=2)


def run_synthetic_comparison(
    output_dir: Path,
    patterns: list = None,
    num_bars: int = 8,
    tempo_bpm: float = 120,
) -> dict:
    """
    Run analysis on multiple synthetic patterns for comparison.

    Parameters
    ----------
    output_dir : Path
        Base output directory
    patterns : list
        Pattern types to analyze
    num_bars : int
        Number of bars per pattern
    tempo_bpm : float
        Tempo for all patterns

    Returns
    -------
    dict
        Pattern name -> AnalysisResults
    """
    if patterns is None:
        patterns = [
            'perfect', 'swing', 'compensatory', 'pocket', 'fatback', 'random'
        ]

    output_dir = Path(output_dir)
    results = {}

    for pattern in patterns:
        pattern_dir = output_dir / pattern
        config = PipelineConfig(tempo_bpm=tempo_bpm)

        analyzer = GrooveAnalyzer(config=config, output_dir=pattern_dir)
        results[pattern] = analyzer.analyze_synthetic(
            pattern_type=pattern,
            num_bars=num_bars,
            random_seed=42,
        )

        logger.info(f"Completed analysis of '{pattern}' pattern")

    return results
