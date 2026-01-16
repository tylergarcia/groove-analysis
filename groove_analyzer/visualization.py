"""
Visualization Engine

Publication-ready visualizations for exploring timing-amplitude patterns.

Implements key visualizations chosen for their ability to reveal
different mathematical structures in groove data:

1. Timing-Amplitude Scatter - Core (Δt, v) space with correlation structure
2. Lag Plot - Temporal dependencies (Δt_n vs Δt_{n+1})
3. Cumulative Drift - Running sum of deviations showing oscillatory correction
4. Complex Plane - Phase coherence and circular patterns
5. Vector Field - Geometric patterns from origin
6. Bar Heatmap - Position-specific patterns across bars
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import seaborn as sns

from .vector_representation import VectorRepresentations

logger = logging.getLogger(__name__)

# Publication-ready style settings
STYLE = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
}


class GrooveVisualizer:
    """
    Create publication-ready visualizations of groove patterns.

    Each visualization is designed to reveal specific mathematical
    structures in the timing-amplitude data.

    Parameters
    ----------
    representations : VectorRepresentations
        The vector representations to visualize
    style : str, default='publication'
        Style preset: 'publication', 'presentation', 'exploration'
    colormap : str, default='viridis'
        Default colormap for sequential data
    """

    # Default axis limits for timing (in ms) - grows if data exceeds
    DEFAULT_TIMING_LIMIT_MS = 25.0

    def __init__(
        self,
        representations: VectorRepresentations,
        style: Literal['publication', 'presentation', 'exploration'] = 'publication',
        colormap: str = 'viridis',
    ):
        self.reps = representations
        self.hit_matrix = representations.hit_matrix
        self.complex_seq = representations.complex_sequence
        self.colormap = colormap
        self.grid_interval_ms = representations.grid_interval_ms

        # Apply style
        plt.rcParams.update(STYLE)
        if style == 'presentation':
            plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16})
        elif style == 'exploration':
            plt.rcParams.update({'figure.dpi': 100})

    def _get_timing_ms(self) -> np.ndarray:
        """Convert normalized timing deviations to milliseconds."""
        return self.hit_matrix[:, 0] * self.grid_interval_ms

    def _get_timing_axis_limits(self, timing_ms: np.ndarray) -> tuple:
        """Get axis limits for timing, using default unless data exceeds it."""
        data_max = max(abs(timing_ms.min()), abs(timing_ms.max())) if len(timing_ms) > 0 else 0
        limit = max(self.DEFAULT_TIMING_LIMIT_MS, data_max * 1.1)
        return (-limit, limit)

    def plot_timing_amplitude_scatter(
        self,
        ax: Optional[plt.Axes] = None,
        color_by: Literal['chronological', 'density', 'amplitude'] = 'chronological',
        show_marginals: bool = False,
        show_ellipse: bool = True,
        show_correlation: bool = True,
        figsize: tuple = (8, 8),
    ) -> plt.Figure:
        """
        Scatter plot of (Δt, v) pairs - the core visualization.

        This is the fundamental view of the timing-amplitude space.
        Each point represents one drum hit.

        Features:
        - Marginal histograms show distributions
        - Correlation ellipse shows covariance structure
        - Color encodes chronological position or density

        Parameters
        ----------
        color_by : str
            How to color points:
            - 'chronological': Color by position in sequence
            - 'density': Color by local density
            - 'amplitude': Use amplitude for color
        show_marginals : bool
            Whether to show marginal histograms
        show_ellipse : bool
            Whether to show 2σ confidence ellipse
        show_correlation : bool
            Whether to annotate with correlation coefficient

        Returns
        -------
        matplotlib.Figure
        """
        deviations = self.hit_matrix[:, 0]
        amplitudes = self.hit_matrix[:, 1]

        if show_marginals:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(
                2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                hspace=0.05, wspace=0.05
            )
            ax_main = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
            ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)
        else:
            if ax is None:
                fig, ax_main = plt.subplots(figsize=figsize)
            else:
                ax_main = ax
                fig = ax.figure

        # Determine colors
        if color_by == 'chronological':
            colors = np.arange(len(deviations))
            cmap = self.colormap
            label = 'Hit number'
        elif color_by == 'amplitude':
            colors = amplitudes
            cmap = 'Reds'
            label = 'Amplitude'
        else:  # density - will be handled by hexbin or kde
            colors = 'steelblue'
            cmap = None
            label = None

        # Main scatter
        if color_by != 'density':
            scatter = ax_main.scatter(
                deviations, amplitudes,
                c=colors, cmap=cmap, alpha=0.7, s=30, edgecolors='none'
            )
            if cmap:
                cbar = plt.colorbar(scatter, ax=ax_main, shrink=0.8, pad=0.02)
                cbar.set_label(label)
        else:
            ax_main.hexbin(
                deviations, amplitudes,
                gridsize=30, cmap='Blues', mincnt=1
            )

        # Correlation ellipse
        if show_ellipse and len(deviations) > 2:
            self._add_confidence_ellipse(ax_main, deviations, amplitudes)

        # Correlation annotation
        if show_correlation and len(deviations) > 2:
            corr = np.corrcoef(deviations, amplitudes)[0, 1]
            ax_main.annotate(
                f'r = {corr:.3f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

        ax_main.set_xlabel('Timing Deviation (normalized)')
        ax_main.set_ylabel('Amplitude (normalized)')
        ax_main.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax_main.axhline(0.5, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)

        # Marginal histograms
        if show_marginals:
            ax_histx.hist(deviations, bins=30, color='steelblue', alpha=0.7, edgecolor='none')
            ax_histx.set_ylabel('Count')
            ax_histx.tick_params(labelbottom=False)

            ax_histy.hist(amplitudes, bins=30, orientation='horizontal',
                         color='steelblue', alpha=0.7, edgecolor='none')
            ax_histy.set_xlabel('Count')
            ax_histy.tick_params(labelleft=False)

            ax_main.set_title('Timing-Amplitude Space', y=-0.12)
        else:
            ax_main.set_title('Timing-Amplitude Space')

        plt.tight_layout()
        return fig

    def _add_confidence_ellipse(
        self,
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        n_std: float = 2.0,
        **kwargs,
    ) -> None:
        """Add confidence ellipse to axes."""
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms

        cov = np.cov(x, y)
        pearson = cov[0, 1] / (np.sqrt(cov[0, 0] * cov[1, 1]) + 1e-8)

        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)

        ellipse = Ellipse(
            (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
            facecolor='none', edgecolor='red', linestyle='--',
            linewidth=1.5, alpha=0.8, **kwargs
        )

        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_x, mean_y = np.mean(x), np.mean(y)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)

    def plot_lag_timing(
        self,
        ax: Optional[plt.Axes] = None,
        lag: int = 1,
        show_diagonal: bool = True,
        show_trajectory: bool = True,
        figsize: tuple = (7, 7),
    ) -> plt.Figure:
        """
        Lag plot: Δt_n vs Δt_{n+lag}.

        This reveals temporal dependencies in timing:
        - Points on diagonal → consistent drift (early stays early)
        - Points in opposite quadrants → compensatory timing (early → late)
        - Slope < 1 → regression to mean (self-correcting)

        Negative lag-1 autocorrelation (points in quadrants 2 and 4)
        suggests the drummer compensates for timing errors.

        Parameters
        ----------
        lag : int
            Number of hits to skip (1 = consecutive pairs)
        show_diagonal : bool
            Whether to show y=x diagonal reference
        show_trajectory : bool
            Whether to connect points with lines

        Returns
        -------
        matplotlib.Figure
        """
        deviations = self.hit_matrix[:, 0]

        if len(deviations) <= lag:
            raise ValueError(f"Not enough data points for lag {lag}")

        x = deviations[:-lag]
        y = deviations[lag:]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Color by position
        colors = np.arange(len(x))

        # Trajectory lines
        if show_trajectory:
            points = np.column_stack([x, y]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(
                segments, cmap=self.colormap, alpha=0.4, linewidth=0.5
            )
            lc.set_array(colors[:-1])
            ax.add_collection(lc)

        # Scatter points
        scatter = ax.scatter(x, y, c=colors, cmap=self.colormap, s=25, alpha=0.8)

        # Reference lines
        if show_diagonal:
            lim = max(abs(x.min()), abs(x.max()), abs(y.min()), abs(y.max())) * 1.1
            ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, linewidth=1)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)

        ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)

        # Calculate and display lag-k autocorrelation
        autocorr = np.corrcoef(x, y)[0, 1]
        ax.annotate(
            f'r(lag={lag}) = {autocorr:.3f}',
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        ax.set_xlabel(f'Δt(n)')
        ax.set_ylabel(f'Δt(n+{lag})')
        ax.set_title(f'Lag Plot: Timing (lag={lag})')
        ax.set_aspect('equal')

        plt.colorbar(scatter, ax=ax, label='Hit number', shrink=0.8)
        plt.tight_layout()
        return fig

    def plot_cumulative_drift(
        self,
        ax: Optional[plt.Axes] = None,
        show_bar_boundaries: bool = True,
        show_zero_line: bool = True,
        figsize: tuple = (12, 5),
    ) -> plt.Figure:
        """
        Plot cumulative timing drift over the performance.

        The cumulative sum of timing deviations shows how far "ahead" or "behind"
        the drummer is relative to the grid at any point. If the drummer self-corrects,
        this will oscillate around zero rather than drifting away.

        Key insight: Even if individual deviations are small, their cumulative effect
        reveals whether timing errors compound (random walk) or cancel out (oscillatory).

        A groove that "breathes" will show wave-like patterns in cumulative drift,
        often with periodicity matching the bar length.

        Parameters
        ----------
        show_bar_boundaries : bool
            Draw vertical lines at bar boundaries (requires bar_tensor)
        show_zero_line : bool
            Draw horizontal line at zero drift

        Returns
        -------
        matplotlib.Figure
        """
        timing_norm = self.hit_matrix[:, 0]  # normalized deviations
        timing_ms = timing_norm * self.grid_interval_ms  # convert to ms
        n_hits = len(timing_ms)

        # Cumulative drift
        cumulative_ms = np.cumsum(timing_ms)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Plot cumulative drift
        hit_indices = np.arange(n_hits)
        ax.plot(hit_indices, cumulative_ms, 'b-', linewidth=1.5, alpha=0.8, label='Cumulative drift')
        ax.scatter(hit_indices, cumulative_ms, c=hit_indices, cmap=self.colormap, s=20, alpha=0.7, zorder=5)

        # Zero reference line
        if show_zero_line:
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # Bar boundaries
        if show_bar_boundaries and self.reps.bar_tensor is not None:
            positions_per_bar = self.reps.bar_tensor.shape[1]
            n_bars = n_hits // positions_per_bar
            for bar_idx in range(1, n_bars + 1):
                bar_start = bar_idx * positions_per_bar
                if bar_start < n_hits:
                    ax.axvline(bar_start, color='red', linestyle=':', alpha=0.4, linewidth=1)
            # Add bar labels
            ax.text(0.02, 0.98, f'Red lines = bar boundaries ({positions_per_bar} hits/bar)',
                   transform=ax.transAxes, fontsize=8, va='top', color='red', alpha=0.7)

        # Calculate drift statistics
        final_drift = cumulative_ms[-1]
        max_drift = np.max(np.abs(cumulative_ms))
        drift_std = np.std(cumulative_ms)

        # Check for oscillatory behavior: does it cross zero multiple times?
        zero_crossings = np.sum(np.diff(np.sign(cumulative_ms)) != 0)
        crossings_per_bar = zero_crossings / (n_hits / 16) if n_hits >= 16 else 0

        stats_text = (f'Final drift: {final_drift:.1f}ms\n'
                     f'Max |drift|: {max_drift:.1f}ms\n'
                     f'Drift σ: {drift_std:.1f}ms\n'
                     f'Zero crossings: {zero_crossings}')

        ax.annotate(stats_text, xy=(0.98, 0.98), xycoords='axes fraction',
                   fontsize=9, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Hit number')
        ax.set_ylabel('Cumulative drift (ms)')
        ax.set_title('Cumulative Timing Drift')
        ax.set_xlim(-0.5, n_hits - 0.5)

        plt.tight_layout()
        return fig

    def plot_drift_periodicity(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: tuple = (10, 5),
    ) -> plt.Figure:
        """
        Analyze periodicity in cumulative drift using autocorrelation.

        If the drummer self-corrects at the bar level, we expect to see
        peaks in autocorrelation at bar-length intervals.

        Also fits a sinusoid to estimate the dominant oscillation period.

        Returns
        -------
        matplotlib.Figure
        """
        timing_norm = self.hit_matrix[:, 0]
        timing_ms = timing_norm * self.grid_interval_ms
        cumulative_ms = np.cumsum(timing_ms)
        n_hits = len(cumulative_ms)

        if n_hits < 8:
            logger.warning("Not enough hits for periodicity analysis")
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Compute autocorrelation of cumulative drift
        cumulative_centered = cumulative_ms - np.mean(cumulative_ms)
        autocorr = np.correlate(cumulative_centered, cumulative_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only
        autocorr = autocorr / autocorr[0]  # Normalize

        max_lag = min(len(autocorr), n_hits // 2)
        lags = np.arange(max_lag)
        autocorr = autocorr[:max_lag]

        # Plot autocorrelation
        ax.bar(lags, autocorr, color='steelblue', alpha=0.7, width=0.8)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)

        # Mark bar-length lags
        if self.reps.bar_tensor is not None:
            positions_per_bar = self.reps.bar_tensor.shape[1]
            for bar_mult in range(1, max_lag // positions_per_bar + 1):
                bar_lag = bar_mult * positions_per_bar
                if bar_lag < max_lag:
                    ax.axvline(bar_lag, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
                    ax.text(bar_lag, ax.get_ylim()[1] * 0.9, f'{bar_mult} bar',
                           fontsize=8, ha='center', color='red')

        # Find peaks in autocorrelation (excluding lag 0)
        if len(autocorr) > 3:
            # Simple peak detection: points higher than neighbors
            peaks = []
            for i in range(2, len(autocorr) - 1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.1:
                    peaks.append((i, autocorr[i]))

            if peaks:
                dominant_period = peaks[0][0]
                ax.annotate(f'Dominant period: {dominant_period} hits',
                           xy=(0.98, 0.98), xycoords='axes fraction',
                           fontsize=10, ha='right', va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Lag (hits)')
        ax.set_ylabel('Autocorrelation')
        ax.set_title('Cumulative Drift Periodicity')
        ax.set_xlim(-0.5, max_lag - 0.5)

        plt.tight_layout()
        return fig

    def plot_complex_plane(
        self,
        ax: Optional[plt.Axes] = None,
        style: Literal['scatter', 'trajectory', 'polar'] = 'scatter',
        show_mean_vector: bool = True,
        figsize: tuple = (8, 8),
    ) -> plt.Figure:
        """
        Plot hits as complex numbers: z = amplitude × exp(i × 2π × deviation).

        This novel representation maps:
        - Timing deviation → phase angle
        - Amplitude → magnitude

        Reveals:
        - Phase coherence (how tight the angle distribution is)
        - Mean resultant vector (overall groove tendency)
        - Circular patterns in timing

        A tight cluster indicates consistent timing-amplitude coupling.
        Spread around the circle indicates varied timing.

        Parameters
        ----------
        style : str
            - 'scatter': Points on complex plane
            - 'trajectory': Connected path through complex plane
            - 'polar': Polar coordinate visualization
        show_mean_vector : bool
            Whether to show the mean resultant vector

        Returns
        -------
        matplotlib.Figure
        """
        z = self.complex_seq
        n_points = len(z)

        if style == 'polar':
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})

            # Convert to polar
            r = np.abs(z)
            theta = np.angle(z)

            colors = np.arange(n_points)
            scatter = ax.scatter(theta, r, c=colors, cmap=self.colormap, s=30, alpha=0.7)

            if show_mean_vector:
                mean_z = np.mean(z)
                mean_r = np.abs(mean_z)
                mean_theta = np.angle(mean_z)
                ax.annotate(
                    '', xy=(mean_theta, mean_r), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2)
                )

            ax.set_title('Complex Plane (Polar)\nAngle = Timing, Radius = Amplitude')
            plt.colorbar(scatter, ax=ax, label='Hit number', shrink=0.8, pad=0.1)

        else:  # scatter or trajectory
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure

            x = z.real
            y = z.imag
            colors = np.arange(n_points)

            if style == 'trajectory':
                points = np.column_stack([x, y]).reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=self.colormap, alpha=0.5, linewidth=0.8)
                lc.set_array(colors[:-1])
                ax.add_collection(lc)

            scatter = ax.scatter(x, y, c=colors, cmap=self.colormap, s=30, alpha=0.7)

            # Unit circle for reference
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.2, linewidth=1)

            # Mean vector
            if show_mean_vector:
                mean_z = np.mean(z)
                ax.annotate(
                    '', xy=(mean_z.real, mean_z.imag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2)
                )

                # Annotate mean resultant length
                mrl = np.abs(mean_z)
                ax.annotate(
                    f'MRL = {mrl:.3f}',
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=10, ha='left', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )

            ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
            ax.axvline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
            ax.set_xlabel('Real (amplitude × cos(timing))')
            ax.set_ylabel('Imaginary (amplitude × sin(timing))')
            ax.set_title('Complex Plane Representation')
            ax.set_aspect('equal')

            plt.colorbar(scatter, ax=ax, label='Hit number', shrink=0.8)

        plt.tight_layout()
        return fig

    def plot_vector_field(
        self,
        ax: Optional[plt.Axes] = None,
        normalize_length: bool = False,
        show_centroid: bool = True,
        figsize: tuple = (8, 8),
    ) -> plt.Figure:
        """
        Vector field plot: arrows from origin to each (Δt, v) point.

        Shows the magnitude and direction of each hit in timing-amplitude space.

        Patterns to look for:
        - Clustered directions → consistent groove character
        - Spread directions → variable playing
        - Systematic rotation → evolving feel

        Parameters
        ----------
        normalize_length : bool
            If True, all vectors have same length (direction only)
        show_centroid : bool
            Whether to highlight the centroid (mean) vector

        Returns
        -------
        matplotlib.Figure
        """
        deviations = self.hit_matrix[:, 0]
        amplitudes = self.hit_matrix[:, 1]
        n_points = len(deviations)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        colors = np.arange(n_points)
        cmap = plt.get_cmap(self.colormap)
        norm = Normalize(vmin=0, vmax=n_points-1)

        # Draw vectors
        for i in range(n_points):
            dx, dy = deviations[i], amplitudes[i]

            if normalize_length:
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx, dy = dx/length * 0.1, dy/length * 0.1

            ax.annotate(
                '', xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle='->', color=cmap(norm(i)),
                    alpha=0.6, lw=0.8
                )
            )

        # Scatter for visibility
        scatter = ax.scatter(
            deviations, amplitudes,
            c=colors, cmap=self.colormap, s=20, alpha=0.8, zorder=5
        )

        # Centroid
        if show_centroid:
            cx, cy = np.mean(deviations), np.mean(amplitudes)
            ax.plot(cx, cy, 'r*', markersize=15, zorder=10, label='Centroid')
            ax.annotate(
                '', xy=(cx, cy), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5)
            )

        # Origin marker
        ax.plot(0, 0, 'ko', markersize=8, zorder=10)

        ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('Timing Deviation (normalized)')
        ax.set_ylabel('Amplitude (normalized)')
        ax.set_title('Vector Field: Hit Directions from Origin')

        if show_centroid:
            ax.legend(loc='upper right')

        plt.colorbar(scatter, ax=ax, label='Hit number', shrink=0.8)
        plt.tight_layout()
        return fig

    def plot_bar_heatmap(
        self,
        figsize: tuple = (12, 6),
        aspect: str = 'auto',
    ) -> Optional[plt.Figure]:
        """
        Heatmap of timing/amplitude patterns across bars.

        Reshapes data into [bars × positions] to reveal:
        - Position-specific patterns (always late on beat 2)
        - Cross-bar consistency (same pattern each bar)
        - Evolution over time (gradual drift)

        Returns
        -------
        matplotlib.Figure or None if bar tensor not available
        """
        if self.reps.bar_tensor is None:
            logger.warning("Bar tensor not available - need grid positions")
            return None

        bar_tensor = self.reps.bar_tensor
        n_bars, positions_per_bar, _ = bar_tensor.shape

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Timing deviations
        ax1 = axes[0]
        dev_data = bar_tensor[:, :, 0]
        im1 = ax1.imshow(
            dev_data, aspect=aspect, cmap='RdBu_r',
            vmin=-0.3, vmax=0.3, interpolation='nearest'
        )
        ax1.set_xlabel('Position in Bar')
        ax1.set_ylabel('Bar Number')
        ax1.set_title('Timing Deviations')
        plt.colorbar(im1, ax=ax1, shrink=0.8, label='Deviation (normalized)')

        # Mark beat positions
        grids_per_beat = positions_per_bar // 4
        for beat in range(1, 4):
            ax1.axvline(beat * grids_per_beat - 0.5, color='white', linewidth=0.5)

        # Amplitudes
        ax2 = axes[1]
        amp_data = bar_tensor[:, :, 1]
        im2 = ax2.imshow(
            amp_data, aspect=aspect, cmap='Oranges',
            vmin=0, vmax=1, interpolation='nearest'
        )
        ax2.set_xlabel('Position in Bar')
        ax2.set_ylabel('Bar Number')
        ax2.set_title('Amplitudes')
        plt.colorbar(im2, ax=ax2, shrink=0.8, label='Amplitude (normalized)')

        for beat in range(1, 4):
            ax2.axvline(beat * grids_per_beat - 0.5, color='white', linewidth=0.5)

        plt.suptitle('Bar-Structured Patterns', y=1.02)
        plt.tight_layout()
        return fig

    def create_analysis_dashboard(
        self,
        output_path: Optional[Path] = None,
        title: str = 'Groove Analysis',
        figsize: tuple = (16, 12),
    ) -> plt.Figure:
        """
        Create a multi-panel dashboard with all key visualizations.

        Layout:
        - Top left: Timing-amplitude scatter
        - Top right: Lag plot (timing lag-1)
        - Bottom left: Complex plane
        - Bottom right: Vector field

        Parameters
        ----------
        output_path : Path, optional
            If provided, save figure to this path
        title : str
            Main title for the dashboard
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.Figure
        """
        fig = plt.figure(figsize=figsize)

        # Create 2x2 grid
        gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)

        # Timing-amplitude scatter (with marginals would be complex, so simplified)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_scatter_simple(ax1)

        # Lag plot
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_lag_simple(ax2)

        # Complex plane
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_complex_simple(ax3)

        # Vector field
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_vector_field_simple(ax4)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved dashboard to {output_path}")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _plot_scatter_simple(self, ax: plt.Axes) -> None:
        """Simplified scatter for dashboard."""
        timing_ms = self._get_timing_ms()
        amplitudes = self.hit_matrix[:, 1]
        colors = np.arange(len(timing_ms))

        ax.scatter(timing_ms, amplitudes, c=colors, cmap=self.colormap, s=20, alpha=0.7)

        if len(timing_ms) > 2:
            corr = np.corrcoef(timing_ms, amplitudes)[0, 1]
            ax.annotate(f'r = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        xlim = self._get_timing_axis_limits(timing_ms)
        ax.set_xlim(xlim)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Timing Deviation (ms)')
        ax.set_ylabel('Amplitude (normalized)')
        ax.set_title('Timing-Amplitude Space')

    def _plot_lag_simple(self, ax: plt.Axes) -> None:
        """Simplified lag plot for dashboard."""
        timing_ms = self._get_timing_ms()
        if len(timing_ms) <= 1:
            return

        x = timing_ms[:-1]
        y = timing_ms[1:]
        colors = np.arange(len(x))

        ax.scatter(x, y, c=colors, cmap=self.colormap, s=20, alpha=0.7)

        # Use fixed limits based on default, or grow if needed
        xlim = self._get_timing_axis_limits(timing_ms)
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], 'k--', alpha=0.3, linewidth=1)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)

        autocorr = np.corrcoef(x, y)[0, 1]
        ax.annotate(f'r(lag=1) = {autocorr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_xlabel('Δt(n) (ms)')
        ax.set_ylabel('Δt(n+1) (ms)')
        ax.set_title('Lag Plot: Timing')
        ax.set_aspect('equal')

    def _plot_complex_simple(self, ax: plt.Axes) -> None:
        """Simplified complex plane for dashboard."""
        z = self.complex_seq
        x = z.real
        y = z.imag
        colors = np.arange(len(z))

        ax.scatter(x, y, c=colors, cmap=self.colormap, s=20, alpha=0.7)

        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.2, linewidth=1)

        mean_z = np.mean(z)
        ax.annotate('', xy=(mean_z.real, mean_z.imag), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))

        mrl = np.abs(mean_z)
        ax.annotate(f'MRL = {mrl:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title('Complex Plane')
        ax.set_aspect('equal')

    def _plot_vector_field_simple(self, ax: plt.Axes) -> None:
        """Simplified vector field for dashboard."""
        timing_ms = self._get_timing_ms()
        amplitudes = self.hit_matrix[:, 1]
        n = len(timing_ms)
        colors = np.arange(n)

        ax.scatter(timing_ms, amplitudes, c=colors, cmap=self.colormap, s=20, alpha=0.7)

        cx, cy = np.mean(timing_ms), np.mean(amplitudes)
        ax.plot(cx, cy, 'r*', markersize=12)
        ax.annotate('', xy=(cx, cy), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))

        ax.plot(0, 0, 'ko', markersize=6)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        xlim = self._get_timing_axis_limits(timing_ms)
        ax.set_xlim(xlim)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Timing Deviation (ms)')
        ax.set_ylabel('Amplitude (normalized)')
        ax.set_title('Vector Field')

    def plot_bar_trajectory(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: tuple = (10, 8),
        bar_index: Optional[int] = None,
    ) -> Optional[plt.Figure]:
        """
        Plot the trajectory through timing-amplitude space for one or more bars.

        Each bar is a path connecting 16 points (for 16th note grid).
        The SHAPE of this path may fingerprint the groove.

        Parameters
        ----------
        bar_index : int, optional
            Specific bar to plot. If None, plots all bars with different colors.

        Returns
        -------
        matplotlib.Figure or None
        """
        if self.reps.bar_tensor is None:
            logger.warning("Bar tensor not available - need grid positions for trajectory")
            return None

        bar_tensor = self.reps.bar_tensor  # shape: (n_bars, positions_per_bar, 2)
        n_bars, positions_per_bar, _ = bar_tensor.shape

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        cmap = plt.get_cmap('tab10')

        if bar_index is not None:
            # Plot single bar
            bars_to_plot = [bar_index]
        else:
            # Plot all bars
            bars_to_plot = range(n_bars)

        for idx, bar_idx in enumerate(bars_to_plot):
            bar_data = bar_tensor[bar_idx]  # (positions_per_bar, 2)
            timing = bar_data[:, 0]
            amplitude = bar_data[:, 1]

            # Filter out NaN positions (missing hits)
            valid_mask = ~(np.isnan(timing) | np.isnan(amplitude))
            if valid_mask.sum() < 2:
                continue

            valid_positions = np.where(valid_mask)[0]
            valid_timing = timing[valid_mask]
            valid_amplitude = amplitude[valid_mask]

            color = cmap(idx % 10)

            # Plot trajectory with arrows
            ax.plot(valid_timing, valid_amplitude, '-', color=color, alpha=0.6,
                   linewidth=1.5, label=f'Bar {bar_idx + 1}')

            # Mark each position with its beat number
            for i, pos in enumerate(valid_positions):
                ax.scatter(valid_timing[i], valid_amplitude[i], c=[color], s=50,
                          zorder=5, edgecolors='black', linewidth=0.5)
                # Annotate with position number
                ax.annotate(str(pos), (valid_timing[i], valid_amplitude[i]),
                           textcoords='offset points', xytext=(3, 3),
                           fontsize=7, alpha=0.7)

            # Draw arrows to show direction
            for i in range(len(valid_timing) - 1):
                dx = valid_timing[i+1] - valid_timing[i]
                dy = valid_amplitude[i+1] - valid_amplitude[i]
                ax.annotate('', xy=(valid_timing[i+1], valid_amplitude[i+1]),
                           xytext=(valid_timing[i], valid_amplitude[i]),
                           arrowprops=dict(arrowstyle='->', color=color, alpha=0.4, lw=1))

        # Calculate path statistics for annotation
        if n_bars > 0:
            # Use first valid bar for stats
            bar_data = bar_tensor[0]
            valid_mask = ~(np.isnan(bar_data[:, 0]) | np.isnan(bar_data[:, 1]))
            if valid_mask.sum() >= 2:
                valid_data = bar_data[valid_mask]
                path_length = np.sum(np.sqrt(np.diff(valid_data[:, 0])**2 +
                                            np.diff(valid_data[:, 1])**2))
                net_displacement = np.sqrt((valid_data[-1, 0] - valid_data[0, 0])**2 +
                                          (valid_data[-1, 1] - valid_data[0, 1])**2)

                ax.annotate(f'Path length: {path_length:.3f}\nNet displacement: {net_displacement:.3f}',
                           xy=(0.95, 0.95), xycoords='axes fraction',
                           fontsize=9, ha='right', va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_xlabel('Timing Deviation (normalized)')
        ax.set_ylabel('Amplitude (normalized)')
        ax.set_title('Bar Trajectory through Timing-Amplitude Space')

        if len(bars_to_plot) <= 10:
            ax.legend(loc='lower right', fontsize=8)

        plt.tight_layout()
        return fig

    def plot_difference_vectors(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: tuple = (10, 8),
        show_field: bool = True,
    ) -> plt.Figure:
        """
        Plot the difference vectors between consecutive hits.

        Δhₙ = hₙ₊₁ - hₙ = (Δtₙ₊₁ - Δtₙ, vₙ₊₁ - vₙ)

        This shows HOW the drummer moves through timing-amplitude space,
        regardless of where they are.

        Parameters
        ----------
        show_field : bool
            If True, shows a binned vector field summary

        Returns
        -------
        matplotlib.Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Compute difference vectors
        diff_timing = np.diff(self.hit_matrix[:, 0])
        diff_amplitude = np.diff(self.hit_matrix[:, 1])

        n_diffs = len(diff_timing)
        colors = np.arange(n_diffs)

        # Scatter plot of difference vectors
        scatter = ax.scatter(diff_timing, diff_amplitude, c=colors, cmap=self.colormap,
                            s=40, alpha=0.7, edgecolors='black', linewidth=0.3)

        # Draw arrows from origin to each difference vector
        for i in range(min(n_diffs, 50)):  # Limit arrows for readability
            ax.annotate('', xy=(diff_timing[i], diff_amplitude[i]), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color=plt.get_cmap(self.colormap)(i/n_diffs),
                                      alpha=0.3, lw=0.5))

        # Statistics
        mean_dt = np.mean(diff_timing)
        mean_dv = np.mean(diff_amplitude)
        std_dt = np.std(diff_timing)
        std_dv = np.std(diff_amplitude)

        # Mean difference vector
        ax.annotate('', xy=(mean_dt, mean_dv), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
        ax.plot(mean_dt, mean_dv, 'r*', markersize=15, label='Mean Δh', zorder=10)

        # Annotate statistics
        magnitude_mean = np.mean(np.sqrt(diff_timing**2 + diff_amplitude**2))
        angle_mean = np.mean(np.arctan2(diff_amplitude, diff_timing))

        stats_text = (f'Mean Δt: {mean_dt:.4f} (σ={std_dt:.4f})\n'
                     f'Mean Δv: {mean_dv:.4f} (σ={std_dv:.4f})\n'
                     f'Mean |Δh|: {magnitude_mean:.4f}\n'
                     f'Mean angle: {np.degrees(angle_mean):.1f}°')
        ax.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=9, ha='left', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('Δ(Timing Deviation) — timing change')
        ax.set_ylabel('Δ(Amplitude) — amplitude change')
        ax.set_title('Difference Vectors: How Drummer Moves Through Space')
        ax.legend(loc='lower right')

        plt.colorbar(scatter, ax=ax, label='Transition number', shrink=0.8)
        plt.tight_layout()
        return fig

    def save_all_visualizations(
        self,
        output_dir: Path,
        prefix: str = '',
        formats: list = ['png', 'svg'],
        title: Optional[str] = None,
    ) -> list:
        """
        Generate and save all visualizations.

        Parameters
        ----------
        output_dir : Path
            Directory to save figures
        prefix : str
            Prefix for filenames
        formats : list
            Output formats (e.g., ['png', 'svg', 'pdf'])
        title : str, optional
            Title for the dashboard (e.g., source filename)

        Returns
        -------
        list
            Paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        # Use title for dashboard, default to 'Groove Analysis'
        dashboard_title = title if title else 'Groove Analysis'

        visualizations = [
            ('timing_amplitude_scatter', self.plot_timing_amplitude_scatter),
            ('lag_timing', self.plot_lag_timing),
            ('cumulative_drift', self.plot_cumulative_drift),
            ('drift_periodicity', self.plot_drift_periodicity),
            ('complex_plane', lambda: self.plot_complex_plane(style='scatter')),
            ('vector_field', self.plot_vector_field),
            ('difference_vectors', self.plot_difference_vectors),
            ('dashboard', lambda: self.create_analysis_dashboard(title=dashboard_title)),
        ]

        # Add bar-based visualizations if available
        if self.reps.bar_tensor is not None:
            visualizations.append(('bar_heatmap', self.plot_bar_heatmap))
            visualizations.append(('bar_trajectory', self.plot_bar_trajectory))

        for name, plot_func in visualizations:
            try:
                fig = plot_func()
                if fig is None:
                    continue

                for fmt in formats:
                    filename = f"{prefix}{name}.{fmt}" if prefix else f"{name}.{fmt}"
                    filepath = output_dir / filename
                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
                    saved_paths.append(filepath)

                plt.close(fig)
                logger.info(f"Saved {name}")

            except Exception as e:
                logger.error(f"Failed to create {name}: {e}")

        return saved_paths
