#!/usr/bin/env python3
"""
Drum Groove Analysis CLI

Analyze drum performances to discover patterns in timing-amplitude space.

Usage:
    python analyze_groove.py --input drum.wav --tempo 120 --output results/
    python analyze_groove.py --synthetic swing --output results/
    python analyze_groove.py --synthetic-comparison --output comparison/
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze drum groove timing-amplitude patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Analyze a WAV file:
    python analyze_groove.py --input recording.wav --tempo 120

  Generate and analyze synthetic data:
    python analyze_groove.py --synthetic swing --bars 16

  Compare multiple synthetic patterns:
    python analyze_groove.py --synthetic-comparison

  Use a config file:
    python analyze_groove.py --input recording.wav --config config.yaml
"""
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i',
        type=Path,
        help='Path to WAV file for analysis'
    )
    input_group.add_argument(
        '--synthetic', '-s',
        type=str,
        choices=['perfect', 'swing', 'rushed', 'dragged', 'accent_downbeat',
                'random', 'compensatory', 'pocket', 'push_pull', 'fatback'],
        help='Generate and analyze synthetic pattern'
    )
    input_group.add_argument(
        '--synthetic-comparison',
        action='store_true',
        help='Compare multiple synthetic patterns'
    )

    # Output
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('results'),
        help='Output directory (default: results/)'
    )

    # Tempo and timing
    parser.add_argument(
        '--tempo', '-t',
        type=float,
        default=120.0,
        help='Tempo in BPM (default: 120)'
    )
    parser.add_argument(
        '--subdivision',
        type=int,
        default=16,
        choices=[4, 8, 16, 32],
        help='Grid subdivision (default: 16 for 16th notes)'
    )
    parser.add_argument(
        '--first-downbeat',
        type=float,
        default=0.0,
        help='Time of first downbeat in seconds (default: 0)'
    )
    parser.add_argument(
        '--highpass',
        type=float,
        default=None,
        help='High-pass filter cutoff in Hz (e.g., 600 to isolate hi-hat from kick)'
    )

    # Synthetic options
    parser.add_argument(
        '--bars',
        type=int,
        default=8,
        help='Number of bars for synthetic data (default: 8)'
    )
    parser.add_argument(
        '--noise',
        type=float,
        default=0.02,
        help='Noise level for synthetic data (default: 0.02)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    # Analysis options
    parser.add_argument(
        '--skip-pca',
        action='store_true',
        help='Skip PCA analysis'
    )
    parser.add_argument(
        '--skip-clustering',
        action='store_true',
        help='Skip clustering analysis'
    )
    parser.add_argument(
        '--skip-significance',
        action='store_true',
        help='Skip significance tests'
    )

    # Output options
    parser.add_argument(
        '--formats',
        nargs='+',
        default=['png'],
        choices=['png', 'svg', 'pdf'],
        help='Output formats for visualizations (default: png)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to YAML config file'
    )

    args = parser.parse_args()

    # Import here to avoid slow startup
    from groove_analyzer import GrooveAnalyzer, PipelineConfig, run_synthetic_comparison

    # Build configuration
    if args.config and args.config.exists():
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig(
            tempo_bpm=args.tempo,
            grid_subdivision=args.subdivision,
            first_downbeat=args.first_downbeat,
            highpass_freq=args.highpass,
            run_pca=not args.skip_pca,
            run_clustering=not args.skip_clustering,
            run_significance=not args.skip_significance,
            output_formats=args.formats,
        )

    # Run analysis
    if args.synthetic_comparison:
        logger.info("Running synthetic pattern comparison...")
        results = run_synthetic_comparison(
            output_dir=args.output,
            num_bars=args.bars,
            tempo_bpm=args.tempo,
        )
        logger.info(f"Comparison complete. Results in {args.output}")

    elif args.synthetic:
        logger.info(f"Analyzing synthetic '{args.synthetic}' pattern...")
        analyzer = GrooveAnalyzer(config=config, output_dir=args.output)
        results = analyzer.analyze_synthetic(
            pattern_type=args.synthetic,
            num_bars=args.bars,
            noise_level=args.noise,
            random_seed=args.seed,
        )
        _print_summary(results)

    else:
        if not args.input.exists():
            parser.error(f"Input file not found: {args.input}")

        logger.info(f"Analyzing {args.input}...")
        analyzer = GrooveAnalyzer(config=config, output_dir=args.output)
        results = analyzer.analyze_audio(args.input, tempo_bpm=args.tempo)
        _print_summary(results, analyzer.quant_result)


def _print_summary(results, quant_result=None):
    """Print a summary of analysis results."""
    stats = results.statistics

    # Detection noise floor (empirically determined from Logic metronome click at exact 128 BPM)
    # Based on interval std, not grid deviation (immune to tempo mismatch)
    NOISE_FLOOR_MS = 0.02

    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)

    print(f"\nTotal hits analyzed: {stats['n_hits']}")

    # Show global offset correction and tempo-relative stats if available
    if quant_result is not None:
        tempo = quant_result.tempo_bpm
        quarter_note_ms = 60000 / tempo  # ms per quarter note

        print(f"\nGlobal Offset: {quant_result.global_offset_ms:.2f}ms")

        # Calculate timing stats in ms and as % of quarter note
        std_ms = quant_result.deviations_corrected_ms.std()
        std_pct = (std_ms / quarter_note_ms) * 100

        # Estimate "true" groove variation by subtracting noise floor (in quadrature)
        if std_ms > NOISE_FLOOR_MS:
            true_std_ms = (std_ms**2 - NOISE_FLOOR_MS**2)**0.5
        else:
            true_std_ms = 0.0
        true_std_pct = (true_std_ms / quarter_note_ms) * 100

        # Interval-based analysis (immune to tempo mismatch)
        import numpy as np
        intervals_ms = np.diff(quant_result.actual_times) * 1000
        interval_std_ms = intervals_ms.std() if len(intervals_ms) > 1 else 0
        interval_std_pct = (interval_std_ms / quarter_note_ms) * 100

        # Estimate true groove from intervals
        if interval_std_ms > NOISE_FLOOR_MS:
            true_interval_std = (interval_std_ms**2 - NOISE_FLOOR_MS**2)**0.5
        else:
            true_interval_std = 0.0
        true_interval_pct = (true_interval_std / quarter_note_ms) * 100

        print(f"\nTiming Variation (interval-based):")
        print(f"  Interval std:   {interval_std_ms:.2f}ms ({interval_std_pct:.2f}% of quarter note)")
        print(f"  Noise floor:    {NOISE_FLOOR_MS:.2f}ms (detection precision)")
        print(f"  True groove:    {true_interval_std:.2f}ms ({true_interval_pct:.2f}% of quarter note)")
    else:
        print("\nTiming Deviations (offset-corrected):")
        t = stats['timing']
        print(f"  Mean: {t['mean']:.4f} (normalized)")
        print(f"  Std:  {t['std']:.4f}")
        print(f"  Range: [{t['min']:.4f}, {t['max']:.4f}]")

    print("\nAmplitude:")
    a = stats['amplitude']
    print(f"  Mean: {a['mean']:.4f}")
    print(f"  Std:  {a['std']:.4f}")

    if 'joint' in stats:
        print(f"\nTiming-Amplitude Correlation: {stats['joint']['correlation']:.4f}")

    # Autocorrelation interpretation
    acf = results.autocorrelation
    if 'interpretation' in acf:
        interp = acf['interpretation']
        print(f"\nTiming Pattern: {interp['timing_pattern']}")
        print(f"  Lag-1 autocorrelation: {interp['timing_lag1']:.4f}")

    # Swing detection
    if results.pattern_matching and 'swing' in results.pattern_matching:
        swing = results.pattern_matching['swing']
        print(f"\nSwing Analysis:")
        print(f"  Type: {swing['swing_type']}")
        print(f"  Offset: {swing['swing_offset_normalized']:.4f}")
        print(f"  Significant: {swing['significant']}")

    # Significance
    if results.significance_tests:
        sig = results.significance_tests
        print("\nSignificance Tests:")
        if 'timing_amplitude_correlation' in sig:
            tac = sig['timing_amplitude_correlation']
            print(f"  Timing-amplitude correlation p={tac['p_value']:.4f}")
        if 'timing_autocorrelation' in sig:
            ta = sig['timing_autocorrelation']
            print(f"  Timing autocorrelation p={ta['p_value']:.4f} ({ta['interpretation']})")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
