"""
Groove Analyzer: Vector Representation of Drum Timing-Amplitude Patterns

A mathematical framework for analyzing drum groove by representing each drum hit
as a 2D vector combining timing deviation and amplitude. This enables pattern
discovery through linear algebra, complex analysis, and phase space methods.
"""

from .onset_detection import OnsetDetector
from .quantization import GridQuantizer
from .vector_representation import GrooveVectorizer
from .pattern_analysis import PatternAnalyzer
from .visualization import GrooveVisualizer
from .synthetic import SyntheticGrooveGenerator
from .pipeline import GrooveAnalyzer, PipelineConfig, run_synthetic_comparison

__version__ = "0.1.0"
__all__ = [
    "OnsetDetector",
    "GridQuantizer",
    "GrooveVectorizer",
    "PatternAnalyzer",
    "GrooveVisualizer",
    "SyntheticGrooveGenerator",
    "GrooveAnalyzer",
    "PipelineConfig",
    "run_synthetic_comparison",
]
