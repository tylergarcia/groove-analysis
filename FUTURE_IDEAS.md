# Future Ideas

This document tracks ideas for future visualizations, analyses, and features.

## Implemented (v0.1)

### Visualizations
1. **Timing-Amplitude Scatter** - Core (Δt, v) space with marginal distributions and correlation ellipse
2. **Phase Space (Timing)** - Δt_n vs Δt_{n+1} to detect compensatory timing patterns
3. **Complex Plane** - z = v·exp(i·2π·Δt/T) for phase coherence analysis
4. **Vector Field** - Arrows from origin showing hit magnitude/direction
5. **Bar Heatmap** - Position-specific patterns across bars (timing + amplitude)

### Analyses
- Basic statistics (moments, correlation, covariance)
- Autocorrelation (timing and amplitude)
- PCA on hit matrix
- K-means clustering
- Swing detection
- Significance tests (permutation tests for correlation and autocorrelation)

---

## Future Visualizations

### Phase Space Extensions
- [ ] **Amplitude Phase Space**: v_n vs v_{n+1} (do loud hits follow loud hits?)
- [ ] **Cross Phase Space**: Δt_n vs v_{n+1} (does timing affect next amplitude?)
- [ ] **4D Phase Space Projection**: Full (Δt_n, v_n, Δt_{n+1}, v_{n+1}) with PCA projection

### Circular/Polar Plots
- [ ] **Benadon Circular Plot**: Time around circumference, amplitude as radius
- [ ] **Rose Diagram**: Angular histogram of hit directions in (Δt, v) space
- [ ] **Phase Histogram**: Circular histogram of timing phases

### Time Series
- [ ] **Timing Evolution**: Deviation over chronological time with trend line
- [ ] **Amplitude Evolution**: Amplitude over time
- [ ] **Dual Time Series**: Both on same plot with dual y-axes
- [ ] **Rolling Statistics**: Moving window mean/std of timing

### Distribution Plots
- [ ] **Joint KDE**: 2D kernel density with contours
- [ ] **QQ Plots**: Compare timing/amplitude to normal distribution
- [ ] **Cumulative Distribution**: ECDF of deviations

### Comparison Plots
- [ ] **Multi-Performance Grid**: Same visualization across multiple takes
- [ ] **Human vs Quantized Overlay**: Show deviation from perfect grid
- [ ] **Before/After Comparison**: Compare two sections of same performance

### Piano Roll Style
- [ ] **Microtiming Piano Roll**: Traditional view with deviation encoded in position
- [ ] **Velocity-Colored Piano Roll**: Color encodes amplitude

### Animation
- [ ] **Vector Accumulation**: Show hits appearing one by one
- [ ] **Phase Space Trajectory**: Animated path through state space
- [ ] **Complex Plane Spiral**: Animated trajectory in complex representation

### Statistical Visualization
- [ ] **Bootstrap Confidence Intervals**: Visualize uncertainty
- [ ] **Permutation Test Distribution**: Show null vs observed
- [ ] **PCA Biplot**: Loadings and scores together
- [ ] **Cluster Dendogram**: Hierarchical clustering tree

---

## Future Analyses

### Pattern Detection
- [ ] **Template Matching**: DTW distance to canonical groove patterns
- [ ] **Motif Discovery**: Find recurring timing-amplitude patterns
- [ ] **Change Point Detection**: Find where groove character changes
- [ ] **Periodicity Analysis**: FFT of timing deviation sequence

### Information Theory
- [ ] **Shannon Entropy**: Of discretized (Δt, v) space
- [ ] **Mutual Information**: I(Δt; v) - how much timing tells about amplitude
- [ ] **Transfer Entropy**: Directional information flow over time
- [ ] **Complexity Measures**: Lempel-Ziv, sample entropy

### Advanced Statistics
- [ ] **Gaussian Mixture Models**: Probabilistic clustering
- [ ] **Hidden Markov Models**: Discover latent groove states
- [ ] **Bayesian Inference**: Posterior over groove parameters
- [ ] **Copula Analysis**: Dependence structure beyond correlation

### Geometric Analysis
- [ ] **Convex Hull**: Boundary of hit cloud
- [ ] **Alpha Shapes**: Concave boundaries at different scales
- [ ] **Persistent Homology**: Topological features of hit cloud
- [ ] **Procrustes Analysis**: Compare groove shapes across performances

### Comparison Methods
- [ ] **Groove Fingerprinting**: Unique identifier for each drummer
- [ ] **Style Classification**: Train classifier on known styles
- [ ] **Anomaly Detection**: Find unusual passages

---

## Future Features

### Input/Output
- [ ] **MIDI Input**: Analyze MIDI files directly
- [ ] **Multi-Track Analysis**: Separate analysis per instrument
- [ ] **Real-Time Analysis**: Stream audio analysis
- [ ] **Database Integration**: Store and query historical analyses

### Interactivity
- [ ] **Plotly Dashboard**: Interactive web-based exploration
- [ ] **Jupyter Widgets**: Interactive parameter exploration
- [ ] **Manual Annotation Tool**: Correct onset detection errors
- [ ] **A/B Comparison Tool**: Compare two performances interactively

### Batch Processing
- [ ] **Folder Processing**: Analyze entire dataset
- [ ] **Parallel Processing**: Speed up large batches
- [ ] **Progress Reporting**: Detailed progress for long jobs

### Reporting
- [ ] **LaTeX Report Generation**: Auto-generate methods section
- [ ] **HTML Interactive Report**: Web-based exploration
- [ ] **Statistical Summary Tables**: Publication-ready tables

### Alternative Backends
- [ ] **Madmom Backend**: Alternative onset detection
- [ ] **Essentia Backend**: Another alternative
- [ ] **Custom Models**: Train ML models for onset detection

---

## Research Questions to Explore

1. **Compensatory Timing**: Is there significant negative lag-1 autocorrelation? Do drummers correct timing errors on the next hit?

2. **Timing-Amplitude Coupling**: Is there a consistent relationship? (e.g., louder = later, or louder = more precise)

3. **Position-Specific Patterns**: Are certain beat positions consistently early/late or loud/soft?

4. **Groove Fingerprints**: Can we identify drummers by their timing-amplitude distributions?

5. **Style Signatures**: Do jazz, rock, funk, etc. have characteristic patterns in this space?

6. **Fatigue Effects**: Does groove precision degrade over the course of a performance?

7. **Tension/Release**: Do timing deviations follow musical phrase structure?

8. **Attractor States**: Are there discrete "groove states" that performances settle into?

9. **Complexity Measures**: Which grooves are more "complex" by information-theoretic measures?

10. **Cross-Cultural Patterns**: Do different musical traditions show different geometric structures?

---

## Notes

- Prioritize visualizations that reveal **mathematically novel** structures
- Focus on patterns that are **not visible** in traditional time-series analysis
- Ensure all analyses have **clear null hypotheses** and significance tests
- Consider **cognitive/perceptual** implications of discovered patterns
