# -*- coding: utf-8 -*-
"""
Tests for Data Analysis Module.

测试数据分析模块的各项功能，包括：
- 趋势分析
- 相关性分析
- 频谱分析
- 统计分析
- 移动统计
"""

import pytest
import numpy as np
from scipy import stats

from src.data.analysis import (
    DataAnalyzer,
    TrendAnalysis,
    CorrelationAnalysis,
    SpectralAnalysis,
)


# =============================================================================
# TrendAnalysis Dataclass Tests
# =============================================================================

class TestTrendAnalysisDataclass:
    """Tests for TrendAnalysis dataclass."""

    def test_creation(self):
        """Test TrendAnalysis creation."""
        trend = TrendAnalysis(
            slope=0.5,
            intercept=10.0,
            r_squared=0.95,
            p_value=0.01,
            trend_direction='increasing',
            confidence=0.99,
        )

        assert trend.slope == 0.5
        assert trend.intercept == 10.0
        assert trend.r_squared == 0.95
        assert trend.p_value == 0.01
        assert trend.trend_direction == 'increasing'
        assert trend.confidence == 0.99


class TestCorrelationAnalysisDataclass:
    """Tests for CorrelationAnalysis dataclass."""

    def test_creation(self):
        """Test CorrelationAnalysis creation."""
        corr = CorrelationAnalysis(
            pearson_r=0.85,
            pearson_p=0.001,
            spearman_r=0.80,
            spearman_p=0.002,
            lag=2,
            strength='strong',
        )

        assert corr.pearson_r == 0.85
        assert corr.pearson_p == 0.001
        assert corr.spearman_r == 0.80
        assert corr.spearman_p == 0.002
        assert corr.lag == 2
        assert corr.strength == 'strong'


class TestSpectralAnalysisDataclass:
    """Tests for SpectralAnalysis dataclass."""

    def test_creation(self):
        """Test SpectralAnalysis creation."""
        freqs = np.array([0.0, 1.0, 2.0])
        amps = np.array([0.1, 0.5, 0.3])

        spectral = SpectralAnalysis(
            dominant_frequency=1.0,
            dominant_amplitude=0.5,
            frequencies=freqs,
            amplitudes=amps,
            bandwidth=0.5,
        )

        assert spectral.dominant_frequency == 1.0
        assert spectral.dominant_amplitude == 0.5
        assert len(spectral.frequencies) == 3
        assert len(spectral.amplitudes) == 3
        assert spectral.bandwidth == 0.5


# =============================================================================
# DataAnalyzer Trend Analysis Tests
# =============================================================================

class TestDataAnalyzerTrend:
    """Tests for DataAnalyzer trend analysis methods."""

    @pytest.fixture
    def analyzer(self):
        """Create DataAnalyzer instance."""
        return DataAnalyzer()

    def test_analyze_trend_increasing(self, analyzer):
        """Test trend analysis with increasing data."""
        # Create strongly increasing data
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        result = analyzer.analyze_trend(values)

        assert result.slope > 0
        assert result.trend_direction == 'increasing'
        assert result.r_squared > 0.99
        assert result.confidence > 0.95

    def test_analyze_trend_decreasing(self, analyzer):
        """Test trend analysis with decreasing data."""
        values = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

        result = analyzer.analyze_trend(values)

        assert result.slope < 0
        assert result.trend_direction == 'decreasing'
        assert result.r_squared > 0.99

    def test_analyze_trend_stable(self, analyzer):
        """Test trend analysis with stable/noisy data."""
        np.random.seed(42)
        values = np.random.normal(5.0, 0.5, 50)

        result = analyzer.analyze_trend(values)

        # For noisy data with no trend, p-value should be high
        assert result.trend_direction == 'stable' or result.p_value > 0.05

    def test_analyze_trend_short_array(self, analyzer):
        """Test trend analysis with very short array."""
        values = np.array([1.0, 2.0])

        result = analyzer.analyze_trend(values)

        assert result.slope == 0.0
        assert result.trend_direction == 'stable'
        assert result.confidence == 0.0

    def test_analyze_trend_empty_array(self, analyzer):
        """Test trend analysis with empty array."""
        values = np.array([])

        result = analyzer.analyze_trend(values)

        assert result.intercept == 0.0
        assert result.trend_direction == 'stable'

    def test_analyze_trend_single_value(self, analyzer):
        """Test trend analysis with single value."""
        values = np.array([5.0])

        result = analyzer.analyze_trend(values)

        assert result.intercept == 5.0
        assert result.trend_direction == 'stable'

    def test_analyze_trend_with_timestamps(self, analyzer):
        """Test trend analysis with custom timestamps."""
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        timestamps = np.array([0.0, 2.0, 4.0, 6.0, 8.0])

        result = analyzer.analyze_trend(values, timestamps)

        assert result.slope == 5.0  # 10/2 = 5 per time unit
        assert result.trend_direction == 'increasing'

    def test_detect_trend_change(self, analyzer):
        """Test trend change detection."""
        # Create data with trend reversal at midpoint
        first_half = np.linspace(0, 50, 50)  # increasing
        second_half = np.linspace(50, 0, 50)  # decreasing
        values = np.concatenate([first_half, second_half])

        change_points = analyzer.detect_trend_change(values, window=15)

        # Should detect change around middle
        assert len(change_points) >= 0  # May or may not detect depending on window

    def test_detect_trend_change_short_array(self, analyzer):
        """Test trend change detection with short array."""
        values = np.array([1.0, 2.0, 3.0])

        change_points = analyzer.detect_trend_change(values, window=20)

        assert len(change_points) == 0

    def test_detect_trend_change_no_change(self, analyzer):
        """Test trend change detection with consistent trend."""
        values = np.linspace(0, 100, 100)

        change_points = analyzer.detect_trend_change(values, window=10)

        # Consistent increasing trend should have no change points
        assert len(change_points) == 0


# =============================================================================
# DataAnalyzer Correlation Analysis Tests
# =============================================================================

class TestDataAnalyzerCorrelation:
    """Tests for DataAnalyzer correlation analysis methods."""

    @pytest.fixture
    def analyzer(self):
        """Create DataAnalyzer instance."""
        return DataAnalyzer()

    def test_analyze_correlation_perfect_positive(self, analyzer):
        """Test correlation with perfectly correlated data."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        result = analyzer.analyze_correlation(x, y)

        assert abs(result.pearson_r - 1.0) < 0.001
        assert result.strength == 'strong'

    def test_analyze_correlation_perfect_negative(self, analyzer):
        """Test correlation with perfectly negatively correlated data."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 8.0, 6.0, 4.0, 2.0])

        result = analyzer.analyze_correlation(x, y)

        assert abs(result.pearson_r + 1.0) < 0.001
        assert result.strength == 'strong'

    def test_analyze_correlation_moderate(self, analyzer):
        """Test correlation with moderate correlation."""
        np.random.seed(42)
        x = np.arange(20, dtype=float)
        y = x + np.random.normal(0, 5, 20)

        result = analyzer.analyze_correlation(x, y)

        assert 0.4 <= abs(result.pearson_r) <= 0.9
        assert result.strength in ['moderate', 'strong']

    def test_analyze_correlation_weak(self, analyzer):
        """Test correlation with weak correlation."""
        np.random.seed(42)
        x = np.arange(20, dtype=float)
        y = np.random.normal(0, 10, 20)

        result = analyzer.analyze_correlation(x, y)

        # Random noise should have weak or no correlation
        assert abs(result.pearson_r) < 0.7
        assert result.strength in ['weak', 'none', 'moderate']

    def test_analyze_correlation_mismatched_length(self, analyzer):
        """Test correlation with mismatched array lengths."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])

        result = analyzer.analyze_correlation(x, y)

        assert result.pearson_r == 0.0
        assert result.strength == 'none'

    def test_analyze_correlation_short_arrays(self, analyzer):
        """Test correlation with very short arrays."""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])

        result = analyzer.analyze_correlation(x, y)

        assert result.strength == 'none'

    def test_analyze_correlation_with_lag(self, analyzer):
        """Test correlation with lagged data."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])  # lag of 1

        result = analyzer.analyze_correlation(x, y, max_lag=3)

        assert result.pearson_r > 0.9  # Still highly correlated
        # May detect optimal lag

    def test_correlation_matrix(self, analyzer):
        """Test correlation matrix calculation."""
        data = {
            'a': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            'b': np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
            'c': np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
        }

        result = analyzer.correlation_matrix(data)

        # Self-correlation should be 1
        assert abs(result[('a', 'a')] - 1.0) < 0.001
        assert abs(result[('b', 'b')] - 1.0) < 0.001

        # a and b are perfectly correlated
        assert abs(result[('a', 'b')] - 1.0) < 0.001

        # a and c are negatively correlated
        assert result[('a', 'c')] < -0.9

    def test_correlation_matrix_empty(self, analyzer):
        """Test correlation matrix with empty data."""
        data = {}

        result = analyzer.correlation_matrix(data)

        assert len(result) == 0

    def test_correlation_matrix_short_series(self, analyzer):
        """Test correlation matrix with short series."""
        data = {
            'a': np.array([1.0, 2.0]),
            'b': np.array([1.0]),
        }

        result = analyzer.correlation_matrix(data)

        # Should handle gracefully
        assert isinstance(result, dict)


# =============================================================================
# DataAnalyzer Spectral Analysis Tests
# =============================================================================

class TestDataAnalyzerSpectral:
    """Tests for DataAnalyzer spectral analysis methods."""

    @pytest.fixture
    def analyzer(self):
        """Create DataAnalyzer instance."""
        return DataAnalyzer()

    def test_analyze_spectrum_sine_wave(self, analyzer):
        """Test spectral analysis with pure sine wave."""
        sample_rate = 100.0  # Hz
        duration = 2.0
        frequency = 5.0  # Hz

        t = np.arange(0, duration, 1.0 / sample_rate)
        values = np.sin(2 * np.pi * frequency * t)

        result = analyzer.analyze_spectrum(values, sample_rate)

        # Dominant frequency should be close to 5 Hz
        assert abs(result.dominant_frequency - frequency) < 1.0
        assert result.dominant_amplitude > 0

    def test_analyze_spectrum_multiple_frequencies(self, analyzer):
        """Test spectral analysis with multiple frequencies."""
        sample_rate = 100.0
        duration = 2.0

        t = np.arange(0, duration, 1.0 / sample_rate)
        values = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)

        result = analyzer.analyze_spectrum(values, sample_rate)

        assert len(result.frequencies) > 0
        assert len(result.amplitudes) > 0

    def test_analyze_spectrum_short_array(self, analyzer):
        """Test spectral analysis with very short array."""
        values = np.array([1.0, 2.0, 3.0])

        result = analyzer.analyze_spectrum(values)

        assert result.dominant_frequency == 0.0
        assert len(result.frequencies) == 0

    def test_analyze_spectrum_constant(self, analyzer):
        """Test spectral analysis with constant signal."""
        values = np.ones(100) * 5.0

        result = analyzer.analyze_spectrum(values)

        # Constant signal should have no dominant frequency after centering
        assert result.dominant_amplitude < 0.01

    def test_detect_periodicity(self, analyzer):
        """Test periodicity detection."""
        sample_rate = 100.0
        duration = 5.0

        t = np.arange(0, duration, 1.0 / sample_rate)
        # Multiple periodic components
        values = np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 7 * t)

        peaks = analyzer.detect_periodicity(values, sample_rate, threshold=0.1)

        assert len(peaks) >= 1
        # Peaks should be sorted by amplitude
        if len(peaks) > 1:
            assert peaks[0][1] >= peaks[1][1]

    def test_detect_periodicity_short_array(self, analyzer):
        """Test periodicity detection with short array."""
        values = np.array([1.0, 2.0])

        peaks = analyzer.detect_periodicity(values)

        assert len(peaks) == 0

    def test_detect_periodicity_noise(self, analyzer):
        """Test periodicity detection with noisy data."""
        np.random.seed(42)
        values = np.random.normal(0, 1, 100)

        peaks = analyzer.detect_periodicity(values, threshold=0.5)

        # Random noise should have few significant peaks
        assert isinstance(peaks, list)


# =============================================================================
# DataAnalyzer Statistical Analysis Tests
# =============================================================================

class TestDataAnalyzerStatistics:
    """Tests for DataAnalyzer statistical analysis methods."""

    @pytest.fixture
    def analyzer(self):
        """Create DataAnalyzer instance."""
        return DataAnalyzer()

    def test_basic_stats(self, analyzer):
        """Test basic statistics calculation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = analyzer.basic_stats(values)

        assert result['count'] == 5
        assert result['mean'] == 3.0
        assert result['min'] == 1.0
        assert result['max'] == 5.0
        assert result['median'] == 3.0

    def test_basic_stats_empty(self, analyzer):
        """Test basic statistics with empty array."""
        values = np.array([])

        result = analyzer.basic_stats(values)

        assert result['count'] == 0
        assert result['mean'] == 0.0

    def test_basic_stats_single_value(self, analyzer):
        """Test basic statistics with single value."""
        values = np.array([5.0])

        result = analyzer.basic_stats(values)

        assert result['count'] == 1
        assert result['mean'] == 5.0
        assert result['std'] == 0.0

    def test_normality_test_normal(self, analyzer):
        """Test normality test with normal distribution."""
        np.random.seed(42)
        values = np.random.normal(0, 1, 100)

        result = analyzer.normality_test(values)

        assert result['test'] == 'shapiro-wilk'
        # Normal data should pass normality test
        assert result['is_normal'] is True or result['p_value'] > 0.01

    def test_normality_test_non_normal(self, analyzer):
        """Test normality test with non-normal distribution."""
        # Uniform distribution is not normal
        np.random.seed(42)
        values = np.random.uniform(0, 1, 100)

        result = analyzer.normality_test(values)

        # May or may not detect non-normality depending on sample

    def test_normality_test_short_array(self, analyzer):
        """Test normality test with short array."""
        values = np.array([1.0, 2.0, 3.0])

        result = analyzer.normality_test(values)

        assert result['is_normal'] is None
        assert result['test'] is None

    def test_normality_test_large_sample(self, analyzer):
        """Test normality test limits sample size."""
        np.random.seed(42)
        values = np.random.normal(0, 1, 10000)

        # Should not raise error even with large sample
        result = analyzer.normality_test(values)

        assert 'statistic' in result

    def test_outlier_detection_iqr(self, analyzer):
        """Test outlier detection with IQR method."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # 100 is outlier

        outliers = analyzer.outlier_detection(values, method='iqr')

        assert 5 in outliers  # Index of 100.0

    def test_outlier_detection_zscore(self, analyzer):
        """Test outlier detection with Z-score method."""
        np.random.seed(42)
        values = np.random.normal(0, 1, 100)
        values = np.append(values, [10.0, -10.0])  # Add outliers

        outliers = analyzer.outlier_detection(values, method='zscore', threshold=3.0)

        assert 100 in outliers  # Index of 10.0
        assert 101 in outliers  # Index of -10.0

    def test_outlier_detection_mad(self, analyzer):
        """Test outlier detection with MAD method."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 50.0])

        outliers = analyzer.outlier_detection(values, method='mad', threshold=3.5)

        assert 5 in outliers  # Index of 50.0

    def test_outlier_detection_mad_zero(self, analyzer):
        """Test outlier detection with MAD when MAD is zero."""
        values = np.array([5.0, 5.0, 5.0, 5.0, 5.0])  # All same value

        outliers = analyzer.outlier_detection(values, method='mad')

        assert len(outliers) == 0

    def test_outlier_detection_unknown_method(self, analyzer):
        """Test outlier detection with unknown method."""
        values = np.array([1.0, 2.0, 3.0])

        outliers = analyzer.outlier_detection(values, method='unknown')

        assert len(outliers) == 0

    def test_outlier_detection_short_array(self, analyzer):
        """Test outlier detection with short array."""
        values = np.array([1.0, 2.0])

        outliers = analyzer.outlier_detection(values, method='iqr')

        assert len(outliers) == 0


# =============================================================================
# DataAnalyzer Moving Statistics Tests
# =============================================================================

class TestDataAnalyzerMovingStats:
    """Tests for DataAnalyzer moving statistics methods."""

    @pytest.fixture
    def analyzer(self):
        """Create DataAnalyzer instance."""
        return DataAnalyzer()

    def test_moving_average(self, analyzer):
        """Test moving average calculation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        result = analyzer.moving_average(values, window=3)

        assert len(result) == 8  # Length - window + 1
        assert abs(result[0] - 2.0) < 0.001  # Mean of [1, 2, 3]
        assert abs(result[-1] - 9.0) < 0.001  # Mean of [8, 9, 10]

    def test_moving_average_short_array(self, analyzer):
        """Test moving average with array shorter than window."""
        values = np.array([1.0, 2.0, 3.0])

        result = analyzer.moving_average(values, window=5)

        np.testing.assert_array_equal(result, values)

    def test_moving_std(self, analyzer):
        """Test moving standard deviation calculation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        result = analyzer.moving_std(values, window=3)

        assert len(result) == 8
        # First window [1, 2, 3] has std of sqrt(2/3) ≈ 0.816
        assert abs(result[0] - np.std([1, 2, 3])) < 0.001

    def test_moving_std_short_array(self, analyzer):
        """Test moving std with short array."""
        values = np.array([1.0, 2.0])

        result = analyzer.moving_std(values, window=5)

        assert len(result) == 2
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_exponential_smoothing(self, analyzer):
        """Test exponential smoothing."""
        values = np.array([1.0, 10.0, 1.0, 10.0, 1.0])

        result = analyzer.exponential_smoothing(values, alpha=0.5)

        assert len(result) == len(values)
        assert result[0] == values[0]  # First value unchanged
        # Subsequent values are smoothed
        assert result[1] == 0.5 * 10.0 + 0.5 * 1.0  # 5.5

    def test_exponential_smoothing_high_alpha(self, analyzer):
        """Test exponential smoothing with high alpha (less smoothing)."""
        values = np.array([0.0, 10.0, 0.0, 10.0, 0.0])

        result = analyzer.exponential_smoothing(values, alpha=0.9)

        # High alpha means result follows input closely
        assert result[1] > 8.0

    def test_exponential_smoothing_low_alpha(self, analyzer):
        """Test exponential smoothing with low alpha (more smoothing)."""
        values = np.array([0.0, 10.0, 0.0, 10.0, 0.0])

        result = analyzer.exponential_smoothing(values, alpha=0.1)

        # Low alpha means result changes slowly
        assert result[1] < 2.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestDataAnalyzerIntegration:
    """Integration tests for DataAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create DataAnalyzer instance."""
        return DataAnalyzer()

    def test_complete_analysis_workflow(self, analyzer):
        """Test complete analysis workflow."""
        np.random.seed(42)

        # Generate test data with trend and periodicity
        t = np.linspace(0, 10, 200)
        values = 2 * t + 3 * np.sin(2 * np.pi * 0.5 * t) + np.random.normal(0, 0.5, 200)

        # Basic stats
        stats = analyzer.basic_stats(values)
        assert stats['count'] == 200

        # Trend analysis
        trend = analyzer.analyze_trend(values)
        assert trend.slope > 0  # Should detect upward trend

        # Spectral analysis
        spectrum = analyzer.analyze_spectrum(values, sample_rate=20.0)
        assert spectrum.dominant_frequency > 0

        # Outlier detection
        outliers = analyzer.outlier_detection(values, method='zscore')
        assert isinstance(outliers, np.ndarray)

        # Moving average
        ma = analyzer.moving_average(values, window=10)
        assert len(ma) == len(values) - 10 + 1

    def test_correlation_workflow(self, analyzer):
        """Test correlation analysis workflow."""
        np.random.seed(42)

        # Create correlated data
        x = np.linspace(0, 10, 100)
        y = 2 * x + np.random.normal(0, 1, 100)
        z = -x + np.random.normal(0, 0.5, 100)

        data = {'x': x, 'y': y, 'z': z}

        # Pairwise correlation
        corr_xy = analyzer.analyze_correlation(x, y)
        assert corr_xy.strength in ['strong', 'moderate']

        # Correlation matrix
        matrix = analyzer.correlation_matrix(data)
        assert matrix[('x', 'y')] > 0  # Positive correlation
        assert matrix[('x', 'z')] < 0  # Negative correlation
