## Implementation
### Brainstorm
WAV → Matrix → Visualization pipeline:
Phase 1: WAV → Onset Data
```Input: WAV file + known tempo (BPM)
↓
librosa onset detection (timing + strength)
↓
Output: Arrays of [onset_times] and [amplitudes]```

Phase 2: Onset Data → Hit Vectors

```
For each detected onset:
  1. Find nearest grid position (16th note, 8th note, etc.)
  2. Calculate: Δt = actual_time - grid_time (in ms or normalized)
  3. Extract: v = amplitude/velocity (normalized 0-1)
  4. Create vector: h_i = [Δt_i, v_i]

Output: N×2 matrix H where each row is one hit


### **Phase 3: Matrix Representations to Explore**

**A) Direct Matrix:**

H = [[Δt₁, v₁],
     [Δt₂, v₂],
     ...
     [Δtₙ, vₙ]]
```

**B) Complex Number Sequence:**
```
z_i = v_i · e^(i·2π·Δt_i/T)
where T = beat period
```
This maps timing to angle, amplitude to magnitude - very elegant for periodic phenomena!

**C) Lag/Difference Matrices:**
```
Timing differences: ΔH_timing = [Δt₂-Δt₁, Δt₃-Δt₂, ...]
Amplitude differences: ΔH_amp = [v₂-v₁, v₃-v₂, ...]
Or combined: ΔH = [[Δt₂-Δt₁, v₂-v₁], ...]
```

**D) Trajectory Matrix (for phase space):**
```
T = [[Δt_i, v_i, Δt_{i+1}, v_{i+1}]] 
(4D, then project to 2D for visualization)

### Phase 4: Visualizations & Analysis
Scatter/Vector Field:
pythonplt.scatter(deviations, amplitudes, c=range(len(deviations)))
plt.quiver(0, 0, deviations, amplitudes)  # vectors from origin
Reveals: clustering, correlations, trajectories over time
Phase Space - Adjacent Hits:
pythonplt.plot(H[:-1, 0], H[1:, 0])  # Δt_i vs Δt_{i+1}
plt.plot(H[:-1, 1], H[1:, 1])  # v_i vs v_{i+1}

Or combined 2D→2D trajectory
Reveals: compensatory timing, velocity patterns, loops/attractors
Complex Plane Plot:
pythonz = amplitudes * np.exp(1j * 2 * np.pi * deviations_norm)
plt.polar(np.angle(z), np.abs(z), 'o')
Reveals: timing clustered at angles, amplitude as radius
Heatmaps/Images:
python# Reshape into bars × beats for 2D visualization
groove_matrix = H.reshape(num_bars, beats_per_bar, 2)
plt.imshow(groove_matrix[:,:,0])  # timing heatmap
plt.imshow(groove_matrix[:,:,1])  # amplitude heatmap
Vector Magnitude & Direction:
pythonmagnitudes = np.sqrt(deviations**2 + amplitudes**2)
angles = np.arctan2(amplitudes, deviations)
plt.hist(angles)  # preferred directions?
Correlation Ellipse:
pythonfrom matplotlib.patches import Ellipse
cov = np.cov(deviations, amplitudes)
- Plot 2σ confidence ellipse

Reveals: correlation structure, variance ratio
### Phase 5: Pattern Detection Ideas

PCA on the H matrix - do hits cluster into "types"?
K-means clustering - natural groupings of (Δt, v) pairs?
Fourier transform of complex sequence - periodic patterns?
Correlation with templates - DTW distance to "ideal swing", "shuffle", etc.
Recurrence plots - which (Δt, v) states repeat?
Symbolic dynamics - discretize the space into regions, analyze transitions
Entropy - how "random" vs "patterned" is the groove?
Mutual information between timing and amplitude - are they independent or coupled?

### Potential Discoveries:

Groove "attractors" - certain (Δt, v) combinations that great drummers gravitate toward
Compensation patterns - "if early, then loud" or "if late, then soft"
Periodic geometric shapes in phase space unique to different feels
Vector field flows showing how hits evolve through timing-amplitude space
Eigen-grooves from PCA - principal patterns of variation
Complexity measures - information-theoretic quantities that correlate with "grooviness"
### Code
#### Libraries
- librosa
- madmom
- essentia 
- aublo

#### Repos
- [GrooveToolbox](https://github.com/fredbru/GrooveToolbox)
- [thebeat](https://github.com/jellevanderwerff/thebeat)
- 

## Previous Research

### Books
Fundamentals of Music Processing: Audio, Analysis, Algorithms, Applications - Meinard Müller

### Papers
**Hennig et al (2011)
The Nature and Perception of Fluctuations in Human Musical Rhythms**
>"Universal statistical properties of music have been unveiled"

Jumps straight in with the $1/f$ power spectral density
Recorded:
- One handed finger tapping on a drum, following a metronome
- Vocalist singing short phonemes instead of words


Claims "...computer generated perfect beat patterns are frequently devalued by listeners due to a perceived lack of human touch."

Claims "...we establish long-range fluctuations as an inevitable natural companion of both simple and complex human rhythmic performances."


**Räsänen et al (2015) 
Fluctuations of Hi-Hat Timing and Dynamics in a Virtuoso Drum Track of a Popular Music Recording.**

Looks at both amplitude and timing fluctuations but only over long terms. Focuses quite a bit on the "highly sensitive onset detection". 

Makes the point that $1/f$ (or fractal) fluctuations found in the drumming are also found in other "human-generated time series"



>"According to recent studies, such fluctuations ($1/f$) also lead
to a favored listening experience."

>"To the best of our knowledge, the correlation properties of amplitude (i.e., loudness) fluctuations of beats in rhythms have not been scrutinized as yet."

>"Previously, it has been found that microtiming deviations without LRCs do not affect the listener groove ratings, or even correlate negatively with them. On the other hand, groove ratings can be changed with other aspects in the rhythmic structure, e.g., with syncopation. Here we focus on timing and loudness variations that occur naturally when a drummer plays to a piece of music, and suggest that they may also contribute to the groove. However, we do not provide an exhaustive treatment of groove from a musicological point of view."

>*"I like the single-handed method, because it’s a lot smoother feel. For instance in the Michael McDonald record "I Keep Forgettin”, I tried doing the alternating stroke method of doing 16ths, and it sounded just too stiff and staccato for me."* - Drummer Jeff Porcaro
Interesting statement from the drummer of the song analyzed in the paper. What he says here is that by being forced to use a single hand to play the pattern instead of two, it felt better and less stiff. This raises a whole other interesting question about why that might be, what muscle or nerve delay could be responsible for influencing the feel of a part with one hand that is less present on a part played by both?

>"Clear evidence of LRC fluctuations in 16th note hi-hat intervals was found. To the best of our knowledge, this phenomenon has not been found in recorded drumming in popular music before, when nmetronome was present during the recording, and when no individual drum tracks were available. The LRCs seem to wash away in short time scales, likely due to motor delays studied before in human cognition."



#### Notes over these papers

I like the notation and some of the writing style in "Nature and Perception..", at least the way they portray beat detection and other things. 

#### Issues with focusing on long-range correlations
These papers put all of their effort into measuring how much the rhythms correlate bars later instead of sub-beats later. This doesn't actually say much about rhythm the way I think of it. Inter-beat rhythm can have an instantly good feeling in just a few counts. 4-8 hi hat counts can feel funky and groovy and high quality without any long range correlations. 

#### Concepts needed for my research
**Autocorrelation** measures the correlation of a signal with a delayed copy of itself. Essentially, it quantifies the similarity between observations of a random variable at different points in time. The analysis of autocorrelation is a mathematical tool for identifying repeating patterns or hidden periodicities within a signal obscured by noise. Autocorrelation is widely used in signal processing, time domain and time series analysis to understand the behavior of data over time.
$$R_{XX}(t_1,t_2)=E\big[X_{t_1}\ \bar{X}_{t_2}\big]$$

Where $E$ is the expected value operator and the bar represents complex conjugation. Expected value being a generaliztion of a weighted average or mean.

Interestingly this autocorrelation function(?) is related to the spectral power density $S_{XX}
$ by the Fourier transform $$S_{XX}(\omega)=\int_{-\infty}^{\infty}R_{XX}(\tau)e^{-i\omega \tau}d\tau$$
Where $\omega =e^{-i2\pi\xi t}$ as in the Fourier transform integrand
Spectral power density is equivalent to frequency analyzer

Paper refers to "lag-1" autocorrelation. From nist.gov section on [exploratory data analysis](https://www.itl.nist.gov/div898/handbook/eda/eda.htm):

Given measurements, $Y_1, Y_2, ..., Y_N$ at time $X_1, X_2, ..., X_N,$ the lag $k$ autocorrelation function is defined as

$$r_{k} = \frac{\sum_{i=1}^{N-k}(Y_{i} - \bar{Y})(Y_{i+k} -\bar{Y})} {\sum_{i=1}^{N}(Y_{i} - \bar{Y})^{2} }$$

## My spin on the research

#### Amplitude
I hypothesize that what makes a drum performance have a perceived human feel has just as much to do with micro amplitude differences as it does micro timing. Could represent each hit as a vector to visually represent the data of what they look like together. Relative amplitude will be interesting to define. Recording env and distortion in recording chain can affect both the relative amplitude detection and the timing onset detection. Higher dynamic range will give a shorter range of time to detect an onset, and will give more bits of information between high/med/low amplitude to be able to tell them apart better. A super compressed or distorted recording will make it tough to calculate the relative amplitudes of the different hits. 

**Vectorizing drum patterns**
$\langle x,y\rangle$ where $0<x\le1$ is the detected hit's distance from the nearest note subdivision in question, and $0<y\le1$ represents the amplitude         . $\langle-0.13,0.7\rangle$ would be a hit that it lightly early and moderately high amplitude.

**Guesses** 
Are there examples of "humanized" electronic music that embodies a lower resolution feel and groove that could be useful in the analysis? Are the patterns of feel and unique musical fingerprint I'm hearing in human-created music on the same spectrum as the patterns of feel and UMF from "humanized" electronic music?


### 20260108
Mathematical representations for timing-amplitude pairs
The fundamental unit of analysis is the drum hit represented as a 2D vector: h = (Δt, v) where Δt is the timing offset from the quantized grid position (in milliseconds or normalized to beat period) and v is velocity/amplitude (0-127 in MIDI scale, or normalized 0-1).
For a sequence of N hits in a phrase, this produces a matrix:
$H = [(Δt₁, v₁), (Δt₂, v₂), ..., (Δtₙ, vₙ)]$
Normalization improves cross-tempo comparison:

$Δt_{norm} = Δt / \text{beat period} $
(range typically -0.5 to +0.5)
$v_{norm}$ = $v / v_{max}$ (range 0 to 1)

Complex number representation for circular timing
Since timing deviations are inherently periodic (deviations wrap around beat boundaries), complex number representation captures this circularity elegantly:
$z = v · e^{i2πΔt/T}$
where T is the beat period. This enables:

- Phase coherence analysis across beats
- Natural circular statistics (mean angle, circular variance)
- Fourier-like analysis of deviation patterns

**The Cemgil et al. (1999) model for timing correlations between adjacent onsets:**
$ρ_nm = η · exp(-λ²/2 · (c_m - c_n)²)
$where $η$ is correlation strength (-1 to 1), $λ$ is decay rate with metrical distance, and $c_m$, $c_n$ are grid positions. This captures the empirical finding that onsets close together tend to be more correlated.
The bivariate Gaussian models joint timing-amplitude distribution:
$p(Δt, v) ∝ exp(-Q/2(1-ρ²))$
where $Q = (Δt/σ_t)² - 2ρ(Δt/σ_t)(v/σ_v) + (v/σ_v)²$
The correlation coefficient ρ captures systematic timing-amplitude relationships (e.g., louder hits tending to be early or late).

**Phase Space Plots**
https://researchportal.vub.be/en/publications/visualizing-and-interpreting-rhythmic-patterns-using-phase-space-/

