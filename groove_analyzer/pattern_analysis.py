"""
Pattern Analysis Module

Detects and quantifies patterns in timing-amplitude vector space.
Includes statistical analysis, correlation structures, dimensionality
reduction, and pattern matching.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

import numpy as np
from scipy import stats
from scipy.spatial.distance import mahalanobis

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResults:
    """Container for all analysis results."""

    # Basic statistics
    statistics: dict

    # Autocorrelation results
    autocorrelation: dict

    # PCA results
    pca: Optional[dict] = None

    # Clustering results
    clustering: Optional[dict] = None

    # Pattern matching results
    pattern_matching: Optional[dict] = None

    # Significance tests
    significance_tests: Optional[dict] = None

    def to_json(self, path: Path) -> None:
        """Save all results to JSON."""
        results = {
            'statistics': self.statistics,
            'autocorrelation': self.autocorrelation,
        }
        if self.pca:
            results['pca'] = self.pca
        if self.clustering:
            results['clustering'] = self.clustering
        if self.pattern_matching:
            results['pattern_matching'] = self.pattern_matching
        if self.significance_tests:
            results['significance_tests'] = self.significance_tests

        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved analysis results to {path}")


class PatternAnalyzer:
    """
    Detect and quantify patterns in timing-amplitude vector space.

    Provides statistical analysis, autocorrelation, PCA, clustering,
    and significance testing for the groove vectors.

    Parameters
    ----------
    hit_matrix : np.ndarray
        N x 2 matrix of (timing_deviation, amplitude) pairs
    complex_sequence : np.ndarray, optional
        Complex number representation
    """

    def __init__(
        self,
        hit_matrix: np.ndarray,
        complex_sequence: Optional[np.ndarray] = None,
    ):
        self.hit_matrix = hit_matrix
        self.complex_sequence = complex_sequence
        self.n_hits = len(hit_matrix)
        self.deviations = hit_matrix[:, 0]
        self.amplitudes = hit_matrix[:, 1]

    def run_full_analysis(
        self,
        run_pca: bool = True,
        run_clustering: bool = True,
        run_significance: bool = True,
        n_clusters: list = [3, 4, 5],
    ) -> AnalysisResults:
        """
        Run complete analysis pipeline.

        Parameters
        ----------
        run_pca : bool
            Whether to run PCA
        run_clustering : bool
            Whether to run clustering
        run_significance : bool
            Whether to run significance tests
        n_clusters : list
            Number of clusters to try

        Returns
        -------
        AnalysisResults
        """
        logger.info(f"Running analysis on {self.n_hits} hits")

        # Basic statistics
        stats_results = self.compute_statistics()

        # Autocorrelation
        autocorr_results = self.compute_autocorrelation()

        # PCA
        pca_results = None
        if run_pca and self.n_hits > 3:
            pca_results = self.compute_pca()

        # Clustering
        cluster_results = None
        if run_clustering and self.n_hits > 10:
            cluster_results = self.compute_clustering(n_clusters)

        # Significance tests
        sig_results = None
        if run_significance and self.n_hits > 20:
            sig_results = self.compute_significance_tests()

        return AnalysisResults(
            statistics=stats_results,
            autocorrelation=autocorr_results,
            pca=pca_results,
            clustering=cluster_results,
            significance_tests=sig_results,
        )

    def compute_statistics(self) -> dict:
        """
        Compute comprehensive statistics on timing-amplitude data.

        Returns
        -------
        dict
            Statistics including moments, correlations, and geometric properties
        """
        dev = self.deviations
        amp = self.amplitudes

        # Basic moments
        stats_dict = {
            'n_hits': self.n_hits,
            'timing': {
                'mean': float(np.mean(dev)),
                'std': float(np.std(dev)),
                'var': float(np.var(dev)),
                'skewness': float(stats.skew(dev)) if len(dev) > 2 else 0,
                'kurtosis': float(stats.kurtosis(dev)) if len(dev) > 3 else 0,
                'min': float(np.min(dev)),
                'max': float(np.max(dev)),
                'median': float(np.median(dev)),
                'iqr': float(np.percentile(dev, 75) - np.percentile(dev, 25)),
            },
            'amplitude': {
                'mean': float(np.mean(amp)),
                'std': float(np.std(amp)),
                'var': float(np.var(amp)),
                'skewness': float(stats.skew(amp)) if len(amp) > 2 else 0,
                'kurtosis': float(stats.kurtosis(amp)) if len(amp) > 3 else 0,
                'min': float(np.min(amp)),
                'max': float(np.max(amp)),
                'median': float(np.median(amp)),
                'iqr': float(np.percentile(amp, 75) - np.percentile(amp, 25)),
            },
        }

        # Joint statistics
        if self.n_hits > 2:
            corr = np.corrcoef(dev, amp)[0, 1]
            cov_matrix = np.cov(self.hit_matrix.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            stats_dict['joint'] = {
                'correlation': float(corr) if not np.isnan(corr) else 0,
                'covariance_matrix': cov_matrix.tolist(),
                'eigenvalues': eigenvalues.tolist(),
                'eigenvectors': eigenvectors.tolist(),
                'condition_number': float(eigenvalues.max() / (eigenvalues.min() + 1e-10)),
            }

            # Geometric properties
            magnitudes = np.sqrt(dev**2 + amp**2)
            angles = np.arctan2(amp, dev)

            stats_dict['geometry'] = {
                'centroid': [float(np.mean(dev)), float(np.mean(amp))],
                'magnitude_mean': float(np.mean(magnitudes)),
                'magnitude_std': float(np.std(magnitudes)),
                'angle_mean': float(np.mean(angles)),
                'angle_std': float(np.std(angles)),
                'angle_circular_mean': float(np.angle(np.mean(np.exp(1j * angles)))),
            }

            # Outlier detection using Mahalanobis distance
            try:
                cov_inv = np.linalg.inv(cov_matrix)
                mean_vec = np.mean(self.hit_matrix, axis=0)
                mahal_dists = np.array([
                    mahalanobis(hit, mean_vec, cov_inv)
                    for hit in self.hit_matrix
                ])
                outlier_threshold = 3.0
                n_outliers = np.sum(mahal_dists > outlier_threshold)

                stats_dict['outliers'] = {
                    'mahalanobis_threshold': outlier_threshold,
                    'n_outliers': int(n_outliers),
                    'outlier_fraction': float(n_outliers / self.n_hits),
                    'max_mahalanobis': float(np.max(mahal_dists)),
                }
            except np.linalg.LinAlgError:
                pass

        # Complex sequence statistics
        if self.complex_sequence is not None:
            z = self.complex_sequence
            mean_z = np.mean(z)
            mrl = np.abs(mean_z)

            stats_dict['complex'] = {
                'mean_resultant_length': float(mrl),
                'mean_resultant_phase': float(np.angle(mean_z)),
                'phase_coherence': float(mrl / (np.mean(np.abs(z)) + 1e-10)),
                'circular_variance': float(1 - mrl / (np.mean(np.abs(z)) + 1e-10)),
                'total_power': float(np.sum(np.abs(z)**2)),
            }

        return stats_dict

    def compute_autocorrelation(self, max_lag: int = 20) -> dict:
        """
        Compute autocorrelation functions for timing and amplitude.

        Key insights:
        - Negative lag-1 autocorrelation in timing → compensatory timing
        - Positive autocorrelation → momentum/inertia
        - Periodic peaks → cyclic patterns

        Parameters
        ----------
        max_lag : int
            Maximum lag to compute

        Returns
        -------
        dict
            Autocorrelation values and interpretations
        """
        max_lag = min(max_lag, self.n_hits // 2)

        results = {
            'max_lag': max_lag,
            'timing_acf': [],
            'amplitude_acf': [],
            'cross_correlation': [],
        }

        # Compute ACF for timing
        dev_centered = self.deviations - np.mean(self.deviations)
        var_dev = np.var(self.deviations)

        for lag in range(max_lag + 1):
            if var_dev > 1e-10:
                acf = np.mean(dev_centered[:-lag if lag > 0 else None] *
                             dev_centered[lag:]) / var_dev
            else:
                acf = 0
            results['timing_acf'].append(float(acf))

        # Compute ACF for amplitude
        amp_centered = self.amplitudes - np.mean(self.amplitudes)
        var_amp = np.var(self.amplitudes)

        for lag in range(max_lag + 1):
            if var_amp > 1e-10:
                acf = np.mean(amp_centered[:-lag if lag > 0 else None] *
                             amp_centered[lag:]) / var_amp
            else:
                acf = 0
            results['amplitude_acf'].append(float(acf))

        # Cross-correlation (does timing predict amplitude?)
        if var_dev > 1e-10 and var_amp > 1e-10:
            for lag in range(-max_lag, max_lag + 1):
                if lag >= 0:
                    cc = np.mean(dev_centered[:-lag if lag > 0 else None] *
                                amp_centered[lag:])
                else:
                    cc = np.mean(dev_centered[-lag:] *
                                amp_centered[:lag if lag < 0 else None])
                cc = cc / (np.sqrt(var_dev * var_amp))
                results['cross_correlation'].append({
                    'lag': lag,
                    'value': float(cc)
                })

        # Interpret key autocorrelations
        if len(results['timing_acf']) > 1:
            acf1 = results['timing_acf'][1]
            results['interpretation'] = {
                'timing_lag1': float(acf1),
                'timing_pattern': 'compensatory' if acf1 < -0.2 else
                                 'momentum' if acf1 > 0.2 else 'independent',
            }

            # Check for periodicity
            acf_array = np.array(results['timing_acf'][1:])
            peaks = np.where((acf_array[1:-1] > acf_array[:-2]) &
                           (acf_array[1:-1] > acf_array[2:]))[0] + 1
            if len(peaks) > 0:
                results['interpretation']['periodic_lag'] = int(peaks[0] + 1)

        return results

    def compute_pca(self) -> dict:
        """
        Principal Component Analysis on hit matrix.

        Reveals the main modes of variation in the timing-amplitude space.

        Returns
        -------
        dict
            PCA results including explained variance and loadings
        """
        # Center the data
        centered = self.hit_matrix - np.mean(self.hit_matrix, axis=0)

        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # Explained variance
        explained_var = (S ** 2) / (self.n_hits - 1)
        total_var = explained_var.sum()
        explained_var_ratio = explained_var / total_var

        # Principal components (loadings)
        components = Vt

        # Project data onto PCs
        projections = centered @ Vt.T

        results = {
            'explained_variance': explained_var.tolist(),
            'explained_variance_ratio': explained_var_ratio.tolist(),
            'cumulative_variance_ratio': np.cumsum(explained_var_ratio).tolist(),
            'components': components.tolist(),
            'component_interpretation': [],
        }

        # Interpret components
        for i, comp in enumerate(components):
            timing_loading = comp[0]
            amplitude_loading = comp[1]

            if abs(timing_loading) > abs(amplitude_loading) * 2:
                interpretation = 'timing-dominated'
            elif abs(amplitude_loading) > abs(timing_loading) * 2:
                interpretation = 'amplitude-dominated'
            elif timing_loading * amplitude_loading > 0:
                interpretation = 'positive coupling (loud=late or soft=early)'
            else:
                interpretation = 'negative coupling (loud=early or soft=late)'

            results['component_interpretation'].append({
                'pc': i + 1,
                'timing_loading': float(timing_loading),
                'amplitude_loading': float(amplitude_loading),
                'interpretation': interpretation,
                'variance_explained': float(explained_var_ratio[i]),
            })

        return results

    def compute_clustering(
        self,
        n_clusters_list: list = [3, 4, 5],
    ) -> dict:
        """
        Cluster analysis to find natural groupings of hits.

        Uses K-means and evaluates multiple cluster numbers.

        Parameters
        ----------
        n_clusters_list : list
            Numbers of clusters to try

        Returns
        -------
        dict
            Clustering results and quality metrics
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, calinski_harabasz_score

        results = {
            'models': [],
            'best_n_clusters': None,
            'best_silhouette': -1,
        }

        for n_clusters in n_clusters_list:
            if n_clusters >= self.n_hits:
                continue

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.hit_matrix)

            # Quality metrics
            silhouette = silhouette_score(self.hit_matrix, labels)
            calinski = calinski_harabasz_score(self.hit_matrix, labels)

            # Cluster profiles
            cluster_profiles = []
            for i in range(n_clusters):
                mask = labels == i
                cluster_hits = self.hit_matrix[mask]
                cluster_profiles.append({
                    'cluster_id': i,
                    'n_hits': int(mask.sum()),
                    'centroid_timing': float(np.mean(cluster_hits[:, 0])),
                    'centroid_amplitude': float(np.mean(cluster_hits[:, 1])),
                    'std_timing': float(np.std(cluster_hits[:, 0])),
                    'std_amplitude': float(np.std(cluster_hits[:, 1])),
                })

            model_result = {
                'n_clusters': n_clusters,
                'silhouette_score': float(silhouette),
                'calinski_harabasz_score': float(calinski),
                'inertia': float(kmeans.inertia_),
                'cluster_profiles': cluster_profiles,
                'labels': labels.tolist(),
            }

            results['models'].append(model_result)

            if silhouette > results['best_silhouette']:
                results['best_silhouette'] = float(silhouette)
                results['best_n_clusters'] = n_clusters

        return results

    def compute_significance_tests(
        self,
        n_permutations: int = 1000,
    ) -> dict:
        """
        Statistical significance tests comparing to null hypotheses.

        Tests:
        1. Is timing-amplitude correlation significant?
        2. Is timing autocorrelation significant?
        3. Is the pattern different from random?

        Parameters
        ----------
        n_permutations : int
            Number of permutations for permutation tests

        Returns
        -------
        dict
            Test results with p-values
        """
        results = {}

        # Test 1: Timing-amplitude correlation
        observed_corr = np.corrcoef(self.deviations, self.amplitudes)[0, 1]
        if np.isnan(observed_corr):
            observed_corr = 0

        null_corrs = []
        for _ in range(n_permutations):
            shuffled_amp = np.random.permutation(self.amplitudes)
            null_corr = np.corrcoef(self.deviations, shuffled_amp)[0, 1]
            if not np.isnan(null_corr):
                null_corrs.append(null_corr)

        if null_corrs:
            p_value = np.mean(np.abs(null_corrs) >= np.abs(observed_corr))
            results['timing_amplitude_correlation'] = {
                'observed': float(observed_corr),
                'null_mean': float(np.mean(null_corrs)),
                'null_std': float(np.std(null_corrs)),
                'p_value': float(p_value),
                'significant_05': p_value < 0.05,
                'significant_01': p_value < 0.01,
            }

        # Test 2: Timing lag-1 autocorrelation
        if self.n_hits > 2:
            observed_acf1 = np.corrcoef(self.deviations[:-1], self.deviations[1:])[0, 1]
            if np.isnan(observed_acf1):
                observed_acf1 = 0

            null_acf1s = []
            for _ in range(n_permutations):
                shuffled = np.random.permutation(self.deviations)
                null_acf1 = np.corrcoef(shuffled[:-1], shuffled[1:])[0, 1]
                if not np.isnan(null_acf1):
                    null_acf1s.append(null_acf1)

            if null_acf1s:
                p_value = np.mean(np.abs(null_acf1s) >= np.abs(observed_acf1))
                results['timing_autocorrelation'] = {
                    'observed': float(observed_acf1),
                    'null_mean': float(np.mean(null_acf1s)),
                    'null_std': float(np.std(null_acf1s)),
                    'p_value': float(p_value),
                    'significant_05': p_value < 0.05,
                    'interpretation': 'compensatory' if observed_acf1 < -0.1 else
                                     'momentum' if observed_acf1 > 0.1 else 'none',
                }

        # Test 3: Normality tests
        if self.n_hits >= 8:
            _, timing_norm_p = stats.shapiro(self.deviations[:min(5000, self.n_hits)])
            _, amp_norm_p = stats.shapiro(self.amplitudes[:min(5000, self.n_hits)])

            results['normality'] = {
                'timing_shapiro_p': float(timing_norm_p),
                'timing_is_normal': timing_norm_p > 0.05,
                'amplitude_shapiro_p': float(amp_norm_p),
                'amplitude_is_normal': amp_norm_p > 0.05,
            }

        # Test 4: Stationarity (is groove consistent or drifting?)
        if self.n_hits >= 20:
            half = self.n_hits // 2
            first_half = self.deviations[:half]
            second_half = self.deviations[half:]

            _, drift_p = stats.ttest_ind(first_half, second_half)
            results['stationarity'] = {
                'first_half_mean': float(np.mean(first_half)),
                'second_half_mean': float(np.mean(second_half)),
                'drift_p_value': float(drift_p),
                'is_stationary': drift_p > 0.05,
            }

        return results

    def detect_swing(self, grid_positions: np.ndarray) -> Optional[dict]:
        """
        Detect and quantify swing feel.

        Compares on-beat vs off-beat timing to identify swing patterns.

        Parameters
        ----------
        grid_positions : np.ndarray
            Grid position indices

        Returns
        -------
        dict or None
            Swing analysis results
        """
        if len(grid_positions) != len(self.deviations):
            return None

        # Separate on-beats and off-beats (for 16th note grid)
        on_mask = grid_positions % 2 == 0
        off_mask = grid_positions % 2 == 1

        if on_mask.sum() < 3 or off_mask.sum() < 3:
            return None

        on_dev = self.deviations[on_mask]
        off_dev = self.deviations[off_mask]

        # Calculate swing offset
        swing_offset = np.mean(off_dev) - np.mean(on_dev)

        # Statistical test
        _, p_value = stats.ttest_ind(on_dev, off_dev)

        # Classify swing type
        if swing_offset > 0.15:
            swing_type = 'heavy_swing'
        elif swing_offset > 0.08:
            swing_type = 'medium_swing'
        elif swing_offset > 0.03:
            swing_type = 'light_swing'
        elif swing_offset < -0.03:
            swing_type = 'pushed'
        else:
            swing_type = 'straight'

        return {
            'swing_offset_normalized': float(swing_offset),
            'on_beat_mean': float(np.mean(on_dev)),
            'off_beat_mean': float(np.mean(off_dev)),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'swing_type': swing_type,
            'n_on_beats': int(on_mask.sum()),
            'n_off_beats': int(off_mask.sum()),
        }
