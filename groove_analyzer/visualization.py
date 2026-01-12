"""
Visualization Engine

Publication-ready visualizations for exploring timing-amplitude patterns.

Implements the 5 key visualizations chosen for their ability to reveal
different mathematical structures in groove data:

1. Timing-Amplitude Scatter - Core (Δt, v) space with correlation structure
2. Phase Space Plot - Temporal dependencies (Δt_n vs Δt_{n+1})
3. Complex Plane - Phase coherence and circular patterns
4. Vector Field - Geometric patterns from origin
5. Bar Heatmap - Position-specific patterns across bars
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

        # Apply style
        plt.rcParams.update(STYLE)
        if style == 'presentation':
            plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16})
        elif style == 'exploration':
            plt.rcParams.update({'figure.dpi': 100})

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

    def plot_phase_space_timing(
        self,
        ax: Optional[plt.Axes] = None,
        lag: int = 1,
        show_diagonal: bool = True,
        show_trajectory: bool = True,
        figsize: tuple = (7, 7),
    ) -> plt.Figure:
        """
        Phase space plot: Δt_n vs Δt_{n+lag}.

        This reveals temporal dependencies in timing:
        - Points on diagonal → consistent timing
        - Points in opposite quadrants → compensatory timing
        - Clusters → discrete timing states

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
        ax.set_title(f'Phase Space: Timing Lag-{lag}')
        ax.set_aspect('equal')

        plt.colorbar(scatter, ax=ax, label='Hit number', shrink=0.8)
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
        - Top right: Phase space (timing lag-1)
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

        # Phase space
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_phase_space_simple(ax2)

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
        deviations = self.hit_matrix[:, 0]
        amplitudes = self.hit_matrix[:, 1]
        colors = np.arange(len(deviations))

        ax.scatter(deviations, amplitudes, c=colors, cmap=self.colormap, s=20, alpha=0.7)

        if len(deviations) > 2:
            corr = np.corrcoef(deviations, amplitudes)[0, 1]
            ax.annotate(f'r = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.set_xlabel('Timing Deviation')
        ax.set_ylabel('Amplitude')
        ax.set_title('Timing-Amplitude Space')

    def _plot_phase_space_simple(self, ax: plt.Axes) -> None:
        """Simplified phase space for dashboard."""
        deviations = self.hit_matrix[:, 0]
        if len(deviations) <= 1:
            return

        x = deviations[:-1]
        y = deviations[1:]
        colors = np.arange(len(x))

        ax.scatter(x, y, c=colors, cmap=self.colormap, s=20, alpha=0.7)

        lim = max(abs(x.min()), abs(x.max()), abs(y.min()), abs(y.max())) * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, linewidth=1)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)

        autocorr = np.corrcoef(x, y)[0, 1]
        ax.annotate(f'r(lag=1) = {autocorr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_xlabel('Δt(n)')
        ax.set_ylabel('Δt(n+1)')
        ax.set_title('Phase Space: Timing')
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
        deviations = self.hit_matrix[:, 0]
        amplitudes = self.hit_matrix[:, 1]
        n = len(deviations)
        colors = np.arange(n)

        ax.scatter(deviations, amplitudes, c=colors, cmap=self.colormap, s=20, alpha=0.7)

        cx, cy = np.mean(deviations), np.mean(amplitudes)
        ax.plot(cx, cy, 'r*', markersize=12)
        ax.annotate('', xy=(cx, cy), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))

        ax.plot(0, 0, 'ko', markersize=6)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('Timing Deviation')
        ax.set_ylabel('Amplitude')
        ax.set_title('Vector Field')

    def save_all_visualizations(
        self,
        output_dir: Path,
        prefix: str = '',
        formats: list = ['png', 'svg'],
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

        Returns
        -------
        list
            Paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        visualizations = [
            ('timing_amplitude_scatter', self.plot_timing_amplitude_scatter),
            ('phase_space_timing', self.plot_phase_space_timing),
            ('complex_plane', lambda: self.plot_complex_plane(style='scatter')),
            ('vector_field', self.plot_vector_field),
            ('dashboard', self.create_analysis_dashboard),
        ]

        # Add bar heatmap if available
        if self.reps.bar_tensor is not None:
            visualizations.append(('bar_heatmap', self.plot_bar_heatmap))

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
