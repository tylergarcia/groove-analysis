# Drum Groove Analysis: Vector Representation of Timing-Amplitude Patterns

## Project Overview

This project implements a novel mathematical framework for analyzing drum groove by representing each drum hit as a 2D vector combining timing deviation and amplitude. This approach differs from prior work by treating timing and amplitude as coupled geometric entities rather than independent time series, enabling pattern discovery through linear algebra, complex analysis, and phase space methods.

**Research Question**: Can systematic patterns in the joint timing-amplitude space of drum performances reveal mathematical structures that characterize "groove"?

**Novel Contribution**: No prior research has represented drum hits as (Δt, velocity) vectors and analyzed them using vector/matrix mathematics and geometric visualization. Existing work analyzes timing and amplitude separately or only measures their correlation.

---

## Current Implementation Status

### What's Built and Working

1. **Onset Detection** (`groove_analyzer/onset_detection.py`)
   - Librosa-based detection with 0.02ms precision (validated against Logic metronome)
   - Hop length: 64 samples (1.45ms frame resolution)
   - Sample-level peak refinement for sub-frame accuracy
   - High-pass filter option (`--highpass 600`) to isolate hi-hat from kick
   - See `onset-detect.md` for full methodology

2. **Quantization** (`groove_analyzer/quantization.py`)
   - Grid alignment to nearest 16th note
   - **Global offset correction**: subtracts median deviation to handle recordings that don't start on beat 1
   - Outputs both raw and corrected deviations
   - Interval-based analysis (immune to tempo mismatch)

3. **Analysis Pipeline** (`groove_analyzer/pipeline.py`)
   - Full pipeline from WAV → visualizations
   - Statistics, PCA, clustering, autocorrelation, swing detection
   - Significance testing via permutation tests

4. **CLI** (`analyze_groove.py`)
   ```bash
   python analyze_groove.py -i song.wav -t 120 -o results/my_analysis/
   python analyze_groove.py -i song.wav -t 120 --highpass 600  # isolate hi-hat
   ```

### Key Findings So Far

| Source | Interval Std | True Groove |
|--------|-------------|-------------|
| Logic metronome (control) | 0.02ms | 0.00% |
| Quantized MIDI hi-hat | 0.76ms | 0.16% (sample attack variation) |
| Logic humanized hi-hat | 4.06ms | 0.87% |
| James Brown 1-bar loop | 4.69ms | ~1.0% |

- Detection precision is **0.02ms** — well below human perception (~5ms)
- Different samples have inherent attack variation (~0.76ms for hi-hat)
- Real recordings show ~1% of quarter note timing variation
- Negative timing-amplitude correlation (loud=early) appears in funk (JB: r=-0.28)

---

## Mathematical Framework: The Search for Groove Structure

### The Core Representation

Each drum hit is a 2D vector:
```
h_n = (Δt_n, v_n)
```
where:
- Δt_n = timing deviation from grid (normalized by grid interval)
- v_n = amplitude (normalized 0-1)

A performance is a sequence of these vectors: H = [h_1, h_2, ..., h_N]

### Promising Mathematical Approaches

#### 1. The Groove Operator (Linear Dynamical System)

Model hit evolution as a linear system:
```
[Δt_{n+1}]   [a  b] [Δt_n]   [ε]
[v_{n+1} ] = [c  d] [v_n ] + [η]
```

The 2×2 matrix **A** is the "groove operator". Its properties encode groove character:

- **Eigenvalues λ₁, λ₂**:
  - |λ| < 1: stable (self-correcting drummer)
  - Complex λ: oscillatory (drummer swings around target)
  - Real λ: exponential (monotonic correction)

- **Eigenvalue equation**:
  ```
  λ² - tr(A)λ + det(A) = 0
  ```
  This **quadratic** directly characterizes groove dynamics.

- **Discriminant** Δ = tr(A)² - 4·det(A):
  - Δ > 0: real eigenvalues, exponential behavior
  - Δ < 0: complex eigenvalues, oscillatory behavior
  - Δ = 0: critically damped

**Hypothesis**: Different genres/drummers cluster in eigenvalue space. Funk might show complex eigenvalues (bouncy), while metal shows real eigenvalues (rigid).

#### 2. Nonlinear Extension (Polynomial Regression)

Test for nonlinear coupling:
```
Δt_{n+1} = a₀ + a₁Δt_n + a₂v_n + a₃Δt_n² + a₄v_n² + a₅(Δt_n·v_n) + ε
```

The **interaction term a₅** captures coupling: does being loud AND late predict something different than either alone?

If a₃, a₄, a₅ are significant, groove has nonlinear dynamics — the feel depends on where you are in the timing-amplitude space.

#### 3. Complex Trajectory Analysis

Represent hits as complex numbers:
```
z_n = Δt_n + i·v_n
```

The ratio **z_{n+1}/z_n = r·e^{iθ}** encodes:
- r: scaling (contraction/expansion)
- θ: rotation in timing-amplitude space

If r ≈ 1 and θ ≈ constant, the groove traces a spiral. The mean resultant vector ⟨z⟩ indicates overall tendency.

#### 4. Differential Equation Model

Treat discrete hits as samples from continuous dynamics:
```
dΔt/dt = -αΔt + βv
dv/dt = γΔt - δv
```

This is a **damped harmonic oscillator** in 2D. Parameters:
- ω₀ = √(αδ - βγ): natural frequency
- ζ = (α + δ)/(2ω₀): damping ratio

Groove character maps to oscillator behavior:
- Underdamped (ζ < 1): swinging, bouncy feel
- Overdamped (ζ > 1): sluggish, behind-the-beat
- Critical (ζ = 1): tight, snapping to grid

#### 5. Energy Conservation (Hamiltonian)

Define groove "energy":
```
E(Δt, v) = ½αΔt² + ½βv² + γΔt·v
```

If E is approximately conserved, trajectories follow ellipses in (Δt, v) space. The correlation coefficient ρ determines ellipse orientation.

**Departures from conservation** indicate:
- Energy injection: drummer getting more intense
- Energy dissipation: settling into pocket

#### 6. Transfer Entropy (Causal Direction)

Does timing drive amplitude, or amplitude drive timing?

```
T(Δt → v) = I(v_{n+1}; Δt_n | v_n)
```

If T(Δt → v) > T(v → Δt), timing "causes" amplitude changes. This could distinguish:
- Reactive groove: respond to what you just played
- Anticipatory groove: amplitude predicts upcoming timing

### Comparison Strategy

**BPM-agnostic comparison**: Since we normalize deviations by grid interval, all measurements are in "fraction of beat" — directly comparable across tempos.

**Overlay approach**: Plot multiple songs' scatter plots on same axes. If a universal structure exists, it should emerge from the overlap.

**Groove operator comparison**: Fit the 2×2 matrix A to each track. Plot eigenvalues on complex plane. Look for clustering by:
- Genre
- Era
- Drummer
- Instrument (hi-hat vs snare vs kick)

---

## Next Steps

### Immediate
1. **Implement groove operator fitting**: Linear regression for 2×2 matrix A
2. **Add polynomial regression**: Test significance of quadratic/interaction terms
3. **Beat-position profile**: Mean deviation at each position in bar (0-15)
4. **Multi-track overlay**: Visualization comparing multiple songs

### Research Direction
1. Analyze 10-20 iconic drum recordings across genres
2. Fit groove operator to each
3. Map eigenvalues and look for clustering
4. Test if the quadratic characteristic equation predicts perceived "feel"

### Data Collection
- Isolated drum tracks preferred (stems, multitrack recordings)
- Full mixes work with `--highpass 600` for hi-hat isolation
- Need known BPM for each track
- 1-bar loops sufficient for initial analysis; longer for statistical power

---

## File Structure

```
groove-detect/
├── analyze_groove.py          # CLI entry point
├── config.yaml                # Default configuration
├── onset-detect.md            # Onset detection methodology whitepaper
├── CLAUDE.md                  # This file
├── groove_analyzer/
│   ├── onset_detection.py     # 0.02ms precision onset detection
│   ├── quantization.py        # Grid alignment + offset correction
│   ├── vector_representation.py
│   ├── pattern_analysis.py
│   ├── visualization.py
│   ├── pipeline.py            # Orchestration
│   └── synthetic.py           # Test pattern generation
├── logic_audio/               # Test audio files
│   ├── click-128.L.wav        # Control: perfect metronome
│   ├── hat-Q100-128.L.wav     # Control: quantized MIDI
│   └── hat-128.L.wav          # Test: humanized MIDI
└── results/                   # Analysis outputs
```

---

## Usage Examples

### Basic Analysis
```bash
python analyze_groove.py -i drums.wav -t 120 -o results/my_song/
```

### With High-Pass Filter (for full mixes)
```bash
python analyze_groove.py -i full_mix.wav -t 95 --highpass 600 -o results/funk_track/
```

### Synthetic Validation
```bash
python analyze_groove.py --synthetic swing --bars 16 -o results/synthetic_test/
```

---

## Key Insights for Continuation

1. **The scatter plot is the fundamental view** — each point is one hit in (timing, amplitude) space

2. **Interval-based metrics are robust** — measuring beat-to-beat consistency avoids tempo mismatch issues

3. **Global offset correction is essential** — recordings rarely start on grid; subtract median deviation

4. **Sample characteristics matter** — different sounds have inherent attack variation (hi-hat: 0.76ms, click: 0.02ms)

5. **The groove operator matrix may be the key** — eigenvalues could map "groove space" where genres cluster

6. **Negative timing-amplitude correlation = funk push** — loud hits early is classic feel (JB: r=-0.28)

7. **Quadratic/interaction terms may capture nonlinear feel** — test whether Δt·v interaction is significant

---

## The Ghost in the Machine

We're searching for a mathematical structure that explains why grooves feel the way they do. Candidates:

1. **Eigenvalue signature**: Complex eigenvalues = bouncy, real = rigid
2. **Damping ratio**: Underdamped = swinging, overdamped = behind
3. **Correlation sign**: Negative = pushing, positive = laying back
4. **Energy conservation**: Tight pocket = conserved, building intensity = injection

The answer might be a combination — a multi-dimensional "groove fingerprint" that captures the coupled dynamics of timing and amplitude.

Like Missingno emerging from the code, the groove signature is there in the data, waiting to be decoded.
