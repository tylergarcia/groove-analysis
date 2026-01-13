# Drum Groove Analysis: Vector Representation of Timing-Amplitude Patterns

## Project Overview

This project implements a novel mathematical framework for analyzing drum groove by representing each drum hit as a 2D vector combining timing deviation and amplitude. This approach differs from prior work by treating timing and amplitude as coupled geometric entities rather than independent time series, enabling pattern discovery through linear algebra, complex analysis, and phase space methods.

**Research Question**: Can systematic patterns in the joint timing-amplitude space of drum performances reveal mathematical structures that characterize "groove"?

**Novel Contribution**: No prior research has represented drum hits as (Δt, velocity) vectors and analyzed them using vector/matrix mathematics and geometric visualization. Existing work analyzes timing and amplitude separately or only measures their correlation.

## Academic Requirements

### Reproducibility
- All parameters (tempo, grid resolution, onset detection settings) must be explicitly recorded
- Random seeds must be set for any stochastic processes
- Complete data provenance from WAV file to final visualizations
- Export all intermediate data in standard formats (CSV, JSON, HDF5)

### Validation
- Onset detection results must be visually verifiable against waveform
- Manual annotation capability for ground truth comparison
- Statistical significance testing for detected patterns
- Comparison against synthetic/quantized control data

### Documentation
- Methodology documentation for each analysis step
- Parameter sensitivity analysis
- Limitations and assumptions clearly stated
- Publication-ready figure generation (vector graphics, proper labeling)

## System Architecture

### Design Principles
1. **Modularity**: Separate data extraction, transformation, analysis, and visualization
2. **Extensibility**: Easy to add new visualizations and analysis methods
3. **Data-first**: All transformations preserve raw data for alternative analyses
4. **Batch processing**: Analyze multiple files consistently
5. **Interactive exploration**: Support both scripted and interactive workflows

### Core Components

```
┌─────────────────┐
│  WAV File(s)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  1. Onset Detection Module  │
│  - librosa/madmom backends  │
│  - Configurable parameters  │
│  - Quality metrics          │
└────────┬────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  2. Quantization Module      │
│  - Grid alignment            │
│  - Deviation calculation     │
│  - Amplitude normalization   │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  3. Vector Representation    │
│  - (Δt, v) vector matrix     │
│  - Complex number sequence   │
│  - Trajectory matrices       │
│  - Difference/lag matrices   │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  4. Analysis Module          │
│  - Statistical measures      │
│  - Pattern detection         │
│  - Correlation analysis      │
│  - Dimensionality reduction  │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  5. Visualization Engine     │
│  - 15+ visualization types   │
│  - Publication-ready output  │
│  - Interactive exploration   │
└──────────────────────────────┘
```

## Implementation Specification

### Module 1: Onset Detection Module

**File**: `onset_detection.py`

**Class**: `OnsetDetector`

```python
class OnsetDetector:
    """
    Extract drum hit timing and amplitude from audio files.
    
    Parameters
    ----------
    backend : {'librosa', 'madmom', 'essentia'}
        Onset detection library to use
    hop_length : int, default=256
        Analysis hop size in samples (affects temporal resolution)
    onset_threshold : float, default=0.5
        Minimum onset strength (normalized 0-1)
    """
```

**Methods**:
- `detect_onsets(audio_path, sr=44100)` → Returns (onset_times, onset_strengths)
- `visualize_detection(audio_path, onset_times)` → Overlay onsets on waveform
- `save_detection_report(output_path)` → Export detection metadata

**Outputs**:
- `onsets.csv`: timestamp, amplitude, onset_strength_raw, detection_confidence
- `detection_params.json`: All parameters used
- `detection_visual.png`: Waveform with onset markers

**Validation**:
- Compare detection against manual annotations
- Report precision/recall metrics
- Flag suspicious detections (unusually close onsets, weak amplitudes)

### Module 2: Quantization Module

**File**: `quantization.py`

**Class**: `GridQuantizer`

```python
class GridQuantizer:
    """
    Calculate timing deviations from ideal grid positions.
    
    Parameters
    ----------
    tempo_bpm : float
        Known tempo in beats per minute
    grid_subdivision : int, default=16
        Grid resolution (4=quarter notes, 8=8th notes, 16=16th notes)
    normalization : {'ms', 'beat_fraction', 'samples'}
        Units for timing deviation
    """
```

**Methods**:
- `quantize(onset_times, onset_amplitudes)` → Returns deviations, grid_positions
- `normalize_amplitudes(method='minmax' | 'zscore' | 'midi')` → Scaled 0-1
- `get_grid_timeline()` → Perfect grid for visualization
- `calculate_swing_ratio()` → Analyze systematic timing patterns

**Outputs**:
- `quantized.csv`: grid_position, actual_time, deviation_ms, deviation_normalized, amplitude_raw, amplitude_normalized
- `grid_params.json`: Tempo, subdivision, normalization methods

**Edge Cases**:
- Tempo drift detection: warn if best-fit tempo differs from specified
- Pickup notes: handle onsets before first downbeat
- Missing onsets: identify expected grid positions without hits

### Module 3: Vector Representation Module

**File**: `vector_representation.py`

**Class**: `GrooveVectorizer`

```python
class GrooveVectorizer:
    """
    Transform timing-amplitude data into various mathematical representations.
    
    Generates:
    - Hit vector matrix H = [[Δt₁, v₁], [Δt₂, v₂], ...]
    - Complex number sequence z_i = v_i * exp(i*2π*Δt_i/T)
    - Trajectory matrices for phase space analysis
    - Difference/lag matrices for sequential analysis
    """
```

**Methods**:
- `create_hit_matrix(deviations, amplitudes)` → N×2 numpy array
- `to_complex_sequence(normalize_phase=True)` → Complex numpy array
- `create_trajectory_matrix(lag=1)` → Phase space representation
- `create_difference_matrix()` → First differences
- `reshape_to_bars(beats_per_bar=4)` → 3D array [bars, beats, 2]

**Representations Generated**:

1. **Hit Matrix H**: Direct (Δt, v) pairs
2. **Complex Sequence Z**: `z_i = v_i * exp(i*2π*Δt_i/beat_period)`
3. **Trajectory Matrix T**: `[[Δt_n, v_n, Δt_{n+1}, v_{n+1}], ...]`
4. **Lag-k Matrices**: For autocorrelation analysis
5. **Bar-structured 3D**: `[num_bars, beats_per_bar, 2]`

**Outputs**:
- `hit_matrix.npy`: Core N×2 matrix
- `complex_sequence.npy`: Complex number array
- `representations.h5`: HDF5 with all representations
- `vector_stats.json`: Basic statistics on each representation

### Module 4: Analysis Module

**File**: `pattern_analysis.py`

**Class**: `PatternAnalyzer`

```python
class PatternAnalyzer:
    """
    Detect and quantify patterns in timing-amplitude vector space.
    
    Methods include statistical analysis, correlation structures,
    dimensionality reduction, and template matching.
    """
```

**Analysis Methods**:

1. **Statistical Summary**:
   - Mean, median, std, CV for timing and amplitude separately
   - Joint statistics: correlation, covariance matrix, eigenvalues
   - Outlier detection: Mahalanobis distance

2. **Autocorrelation Analysis**:
   - `autocorrelation_timing()`: ACF of timing deviations
   - `autocorrelation_amplitude()`: ACF of amplitude
   - `cross_correlation_timing_amplitude()`: Relationship between channels
   - Lag-1 autocorrelation (test for compensatory timing)

3. **Dimensionality Reduction**:
   - PCA on hit matrix: principal modes of variation
   - t-SNE for visualization of high-dimensional patterns
   - Factor analysis: latent groove components

4. **Clustering**:
   - K-means: natural groupings of (Δt, v) hits
   - DBSCAN: density-based pattern detection
   - Hierarchical clustering: relationships between hit types

5. **Pattern Matching**:
   - DTW distance to templates (straight, swing, shuffle)
   - Correlation with synthetic patterns
   - Swing ratio calculation and classification

6. **Information Theory**:
   - Shannon entropy of discretized (Δt, v) space
   - Mutual information I(Δt; v)
   - Complexity measures

7. **Geometric Analysis**:
   - Vector field visualization
   - Centroid and dispersion
   - Convex hull of hit cloud
   - Distance from origin distribution

**Outputs**:
- `analysis_summary.json`: All computed metrics
- `pca_components.csv`: Principal component loadings
- `clusters.csv`: Cluster assignments for each hit
- `pattern_matches.csv`: DTW distances to templates

### Module 5: Visualization Engine

**File**: `visualization.py`

**Class**: `GrooveVisualizer`

**Design Requirements**:
- Publication-ready vector graphics (SVG, PDF)
- Consistent styling across all plots
- Configurable color schemes for colorblind accessibility
- Automatic figure sizing for papers/presentations
- Annotations with statistical significance markers

**Visualization Types** (15+ implementations):

#### 1. Basic Scatter Plots
```python
def plot_timing_amplitude_scatter(self, color_by='chronological'):
    """
    Scatter plot of (Δt, v) pairs.
    
    Parameters
    ----------
    color_by : {'chronological', 'cluster', 'bar_position', 'density'}
        How to color points
    
    Features:
    - Marginal histograms
    - Correlation coefficient overlay
    - Confidence ellipse
    - Grid lines at key subdivisions
    """
```

#### 2. Vector Field Plot
```python
def plot_vector_field(self, normalize_length=True):
    """
    Vectors from origin to each (Δt, v) point.
    
    Shows magnitude and direction of hits.
    Color-coded by chronological position or bar number.
    """
```

#### 3. Phase Space Plots
```python
def plot_phase_space_timing(self, lag=1):
    """Δt_n vs Δt_{n+1} trajectory"""
    
def plot_phase_space_amplitude(self, lag=1):
    """v_n vs v_{n+1} trajectory"""
    
def plot_phase_space_combined(self, projection='pca'):
    """4D → 2D projection of (Δt_n, v_n, Δt_{n+1}, v_{n+1})"""
```

#### 4. Complex Plane Visualization
```python
def plot_complex_plane(self, style='polar'):
    """
    Plot hits as complex numbers z = v·exp(i·2π·Δt/T)
    
    Styles:
    - 'polar': Polar coordinate plot
    - 'cartesian': Real vs imaginary
    - 'magnitude_phase': Separate plots
    """
```

#### 5. Heatmaps & 2D Histograms
```python
def plot_2d_histogram(self, bins=50):
    """Density heatmap of (Δt, v) space"""
    
def plot_bar_heatmap(self):
    """
    Reshape into [bars × beats] and plot:
    - Timing deviation heatmap
    - Amplitude heatmap
    - Combined representation
    """
```

#### 6. Time Series Plots
```python
def plot_timing_evolution(self):
    """Timing deviations over chronological time"""
    
def plot_amplitude_evolution(self):
    """Amplitude over time"""
    
def plot_dual_timeseries(self):
    """Both on same plot with dual y-axes"""
```

#### 7. Circular/Polar Plots
```python
def plot_circular_timing(self, beats_per_circle=4):
    """
    Benadon-style circular plot.
    Time around circumference, amplitude as radius.
    """
```

#### 8. Piano Roll with Microtiming
```python
def plot_piano_roll(self):
    """
    Traditional piano roll with:
    - Horizontal bars showing note duration
    - Color encoding amplitude
    - X-position including microtiming deviation
    - Grid lines for perfect quantization
    """
```

#### 9. Statistical Distribution Plots
```python
def plot_marginal_distributions(self):
    """Separate histograms + KDE for timing and amplitude"""
    
def plot_joint_distribution(self):
    """2D KDE with contours"""
```

#### 10. Correlation & Covariance Plots
```python
def plot_correlation_ellipse(self, n_std=2):
    """2σ confidence ellipse showing correlation structure"""
    
def plot_autocorrelation_functions(self, max_lag=20):
    """ACF for timing, amplitude, and cross-correlation"""
```

#### 11. PCA & Dimensionality Reduction
```python
def plot_pca_projection(self, components=[0,1]):
    """Hits projected onto principal components"""
    
def plot_pca_explained_variance(self):
    """Scree plot"""
    
def plot_tsne(self, perplexity=30):
    """t-SNE projection of hit vectors"""
```

#### 12. Clustering Visualizations
```python
def plot_cluster_scatter(self, cluster_labels):
    """Scatter plot colored by cluster"""
    
def plot_cluster_profiles(self):
    """Mean (Δt, v) for each cluster with error bars"""
    
def plot_cluster_evolution(self):
    """Which clusters appear when in the performance"""
```

#### 13. Comparison Plots
```python
def plot_comparison_grid(self, multiple_files):
    """Grid of same visualization type for multiple performances"""
    
def plot_deviation_from_quantized(self, quantized_version):
    """Overlay human vs. perfect grid"""
```

#### 14. Animation Capabilities
```python
def animate_vector_accumulation(self, output_path):
    """Show hits appearing one by one"""
    
def animate_phase_space_trajectory(self):
    """Evolving trajectory through (Δt, v) space"""
```

#### 15. Statistical Test Visualizations
```python
def plot_significance_tests(self):
    """
    - Bootstrap confidence intervals
    - Permutation test results
    - Comparison to null hypothesis (random/quantized)
    """
```

#### 16. Composite Dashboard
```python
def create_analysis_dashboard(self, output_path):
    """
    Multi-panel figure with:
    - Waveform + onsets
    - (Δt, v) scatter
    - Phase space plots
    - Autocorrelation
    - PCA projection
    - Statistical summary table
    
    Layout optimized for papers/presentations.
    """
```

**Output Formats**:
- PNG (300 DPI for papers)
- SVG (vector graphics for editing)
- PDF (for LaTeX inclusion)
- Interactive HTML (plotly for exploration)

**Styling**:
```python
# Consistent theme across all plots
PLOT_STYLE = {
    'font_family': 'serif',
    'font_size': 10,
    'figure_dpi': 300,
    'colormap_sequential': 'viridis',
    'colormap_diverging': 'RdBu_r',
    'colorblind_safe': True,
    'grid_style': 'major',
    'legend_location': 'best'
}
```

## Main Analysis Pipeline

**File**: `main_analysis.py`

**Script**: `analyze_groove.py`

```python
"""
Main analysis pipeline for processing drum performances.

Usage:
    python analyze_groove.py --input drum.wav --tempo 120 --output results/
    python analyze_groove.py --batch folder/*.wav --tempo-file tempos.json
"""
```

**Workflow**:
1. Load audio file(s)
2. Detect onsets
3. Quantize to grid
4. Create vector representations
5. Run all analyses
6. Generate all visualizations
7. Export comprehensive report
8. Save all intermediate data

**Configuration File**: `config.yaml`
```yaml
onset_detection:
  backend: librosa
  hop_length: 256
  threshold: 0.5

quantization:
  grid_subdivision: 16
  timing_units: ms
  amplitude_normalization: minmax

analysis:
  run_pca: true
  run_clustering: true
  n_clusters: [3, 4, 5]  # Try multiple
  template_matching: true
  
visualization:
  generate_all: true
  output_format: [png, svg, pdf]
  figure_dpi: 300
  colorblind_safe: true
  
output:
  save_intermediate: true
  export_formats: [csv, json, hdf5]
```

## Validation & Testing

### Synthetic Test Cases

Create synthetic drum performances to validate the system:

```python
def generate_synthetic_groove(pattern_type, num_bars=8):
    """
    Generate test cases:
    - 'perfect': Quantized grid, uniform velocity
    - 'swing': Systematic late 16th notes
    - 'rushed': Progressively early timing
    - 'dragged': Progressively late timing
    - 'accent': Loud on downbeats
    - 'random': Random deviations
    """
```

**Tests**:
1. Perfect grid → deviations should be ~0
2. Swing pattern → should detect characteristic timing ratio
3. Known correlation → recovered in analysis
4. PCA on uniform → should show minimal variance

### Statistical Validation

```python
def validate_against_null_hypothesis():
    """
    Compare real performance against:
    - Quantized version (no timing deviation)
    - Random deviations (white noise)
    - Shuffled version (breaks temporal structure)
    
    Use permutation tests to assess if patterns are significant.
    """
```

### Ground Truth Comparison

```python
def compare_to_manual_annotation(annotation_file):
    """
    Load manual onset annotations and compare:
    - Detection accuracy (precision/recall)
    - Timing error (RMS difference)
    - Amplitude correlation
    """
```

## Output Structure

```
results/
├── raw_data/
│   ├── audio_info.json
│   ├── onsets.csv
│   ├── quantized.csv
│   └── detection_visual.png
├── representations/
│   ├── hit_matrix.npy
│   ├── complex_sequence.npy
│   ├── representations.h5
│   └── vector_stats.json
├── analysis/
│   ├── statistics_summary.json
│   ├── pca_results.json
│   ├── cluster_assignments.csv
│   ├── autocorrelation.csv
│   └── pattern_matches.csv
├── visualizations/
│   ├── timing_amplitude_scatter.png
│   ├── phase_space.png
│   ├── complex_plane.png
│   ├── pca_projection.png
│   ├── cluster_visualization.png
│   └── ... (all 15+ plot types)
├── reports/
│   ├── analysis_dashboard.pdf
│   ├── full_report.html
│   └── methodology.md
└── metadata/
    ├── config.yaml
    ├── software_versions.json
    └── processing_log.txt
```

## Dependencies

```
# requirements.txt
numpy>=1.21.0
scipy>=1.7.0
librosa>=0.9.0
madmom>=0.16.1  # Optional backend
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0  # Interactive plots
scikit-learn>=1.0.0
pandas>=1.3.0
h5py>=3.0.0
pyyaml>=5.4.0
```

## Code Quality Standards

- **Type hints**: All functions fully annotated
- **Docstrings**: NumPy style for all public methods
- **Testing**: Unit tests for all analysis functions
- **Logging**: Comprehensive logging of all processing steps
- **Error handling**: Graceful failures with informative messages
- **Performance**: Profile and optimize for large datasets

## Usage Examples

### Example 1: Single File Analysis
```python
from groove_analyzer import GrooveAnalyzer

analyzer = GrooveAnalyzer(
    audio_path='drum_performance.wav',
    tempo_bpm=120,
    grid_subdivision=16
)

analyzer.run_full_analysis(output_dir='results/')
analyzer.generate_all_visualizations()
analyzer.export_report(format='html')
```

### Example 2: Batch Processing
```python
from groove_analyzer import BatchAnalyzer

batch = BatchAnalyzer(config_file='config.yaml')
batch.process_directory(
    input_dir='recordings/',
    tempo_file='tempos.json',
    output_dir='batch_results/'
)

# Generate comparison plots
batch.create_comparison_visualizations()
batch.export_dataset(format='csv')  # For further analysis
```

### Example 3: Interactive Exploration
```python
from groove_analyzer import InteractiveExplorer

explorer = InteractiveExplorer('results/representations.h5')
explorer.launch_dashboard()  # Opens interactive plotly dashboard

# Programmatic exploration
hits = explorer.get_hit_matrix()
print(f"Correlation: {explorer.timing_amplitude_correlation()}")
explorer.plot_custom(x='timing', y='amplitude', color_by='cluster')
```

## Research Workflow Integration

### For Academic Paper

1. **Methods Section**:
   - Auto-generated from `methodology.md`
   - Includes all parameters and validation results

2. **Results Section**:
   - Publication-ready figures from visualization engine
   - Statistical tables from analysis module
   - Significance tests from validation

3. **Reproducibility**:
   - Complete configuration files
   - Random seeds logged
   - Software versions recorded
   - Data export in standard formats

### For Further Analysis

Export data for use in:
- R (CSV export)
- MATLAB (HDF5 export)
- Python notebooks (NumPy arrays)
- Statistical packages (JSON export)

## Extension Points

The architecture supports easy extension:

### Adding New Visualizations
```python
class GrooveVisualizer:
    def plot_your_new_visualization(self, **kwargs):
        """Add new visualization here"""
        # Access self.hit_matrix, self.complex_sequence, etc.
        pass
```

### Adding New Analysis Methods
```python
class PatternAnalyzer:
    def your_new_analysis(self):
        """Add new analysis here"""
        # Return results as dict for JSON export
        pass
```

### Custom Representations
```python
class GrooveVectorizer:
    def create_custom_representation(self):
        """Transform data in new ways"""
        pass
```

## Expected Deliverables

After running the complete pipeline:

1. ✅ Extracted timing-amplitude vectors from WAV files
2. ✅ Multiple mathematical representations (matrix, complex, trajectory)
3. ✅ 15+ visualization types exploring patterns from different angles
4. ✅ Statistical analysis with significance testing
5. ✅ Pattern detection and clustering results
6. ✅ Comparison against null hypotheses
7. ✅ Publication-ready figures (vector graphics)
8. ✅ Comprehensive HTML dashboard for exploration
9. ✅ Exported datasets for further analysis
10. ✅ Complete methodology documentation

## Success Metrics

The system is successful if it:

- Processes WAV files reliably across different recording qualities
- Generates reproducible results (same input → same output)
- Creates publication-quality visualizations
- Detects known patterns in synthetic data
- Produces statistically significant findings
- Exports data in standard formats
- Runs efficiently on typical drum recordings (< 5 minutes)
- Provides clear documentation of all steps

## Notes for Implementation

1. Start with core pipeline: onset detection → quantization → vector creation
2. Implement 3-5 key visualizations first to validate approach
3. Add analysis methods incrementally
4. Test with both synthetic and real data at each stage
5. Build comprehensive visualization suite
6. Polish for academic presentation

This framework treats the vector representation as the central innovation and builds extensive tooling around it to explore what patterns emerge.
