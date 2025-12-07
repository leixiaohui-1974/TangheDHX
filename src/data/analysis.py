# -*- coding: utf-8 -*-
"""
Data Analysis Module.

数据分析模块，提供：
- 趋势分析
- 相关性分析
- 统计分析
- 频谱分析
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy import stats, signal

logger = logging.getLogger(__name__)


@dataclass
class TrendAnalysis:
    """趋势分析结果."""
    slope: float           # 斜率
    intercept: float       # 截距
    r_squared: float       # R²
    p_value: float         # p值
    trend_direction: str   # 'increasing', 'decreasing', 'stable'
    confidence: float      # 置信度


@dataclass
class CorrelationAnalysis:
    """相关性分析结果."""
    pearson_r: float       # Pearson相关系数
    pearson_p: float       # p值
    spearman_r: float      # Spearman相关系数
    spearman_p: float      # p值
    lag: int               # 滞后
    strength: str          # 'strong', 'moderate', 'weak', 'none'


@dataclass
class SpectralAnalysis:
    """频谱分析结果."""
    dominant_frequency: float    # 主频率
    dominant_amplitude: float    # 主幅值
    frequencies: np.ndarray      # 频率数组
    amplitudes: np.ndarray       # 幅值数组
    bandwidth: float             # 带宽


class DataAnalyzer:
    """
    数据分析器.

    提供各种数据分析功能。
    """

    def __init__(self):
        pass

    # -------------------------------------------------------------------------
    # Trend Analysis
    # -------------------------------------------------------------------------

    def analyze_trend(
        self,
        values: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> TrendAnalysis:
        """
        分析趋势.

        Args:
            values: 数据值数组
            timestamps: 时间戳数组 (可选)

        Returns:
            TrendAnalysis 结果
        """
        if len(values) < 3:
            return TrendAnalysis(
                slope=0.0,
                intercept=float(np.mean(values)) if len(values) > 0 else 0.0,
                r_squared=0.0,
                p_value=1.0,
                trend_direction='stable',
                confidence=0.0,
            )

        # 使用索引作为x轴
        x = timestamps if timestamps is not None else np.arange(len(values))

        # 线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

        # 判断趋势方向
        if p_value < 0.05:  # 显著
            if slope > 0:
                direction = 'increasing'
            else:
                direction = 'decreasing'
        else:
            direction = 'stable'

        confidence = 1.0 - p_value

        return TrendAnalysis(
            slope=float(slope),
            intercept=float(intercept),
            r_squared=float(r_value ** 2),
            p_value=float(p_value),
            trend_direction=direction,
            confidence=confidence,
        )

    def detect_trend_change(
        self,
        values: np.ndarray,
        window: int = 20,
    ) -> List[int]:
        """
        检测趋势变化点.

        Args:
            values: 数据值数组
            window: 窗口大小

        Returns:
            变化点索引列表
        """
        if len(values) < window * 2:
            return []

        change_points = []

        for i in range(window, len(values) - window):
            before = values[i-window:i]
            after = values[i:i+window]

            # 比较前后趋势
            trend_before = self.analyze_trend(before)
            trend_after = self.analyze_trend(after)

            # 检测方向变化
            if trend_before.trend_direction != trend_after.trend_direction:
                if (trend_before.confidence > 0.7 and trend_after.confidence > 0.7):
                    change_points.append(i)

        return change_points

    # -------------------------------------------------------------------------
    # Correlation Analysis
    # -------------------------------------------------------------------------

    def analyze_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int = 10,
    ) -> CorrelationAnalysis:
        """
        分析相关性.

        Args:
            x: 第一个数据序列
            y: 第二个数据序列
            max_lag: 最大滞后

        Returns:
            CorrelationAnalysis 结果
        """
        if len(x) != len(y) or len(x) < 3:
            return CorrelationAnalysis(
                pearson_r=0.0,
                pearson_p=1.0,
                spearman_r=0.0,
                spearman_p=1.0,
                lag=0,
                strength='none',
            )

        # Pearson相关
        pearson_r, pearson_p = stats.pearsonr(x, y)

        # Spearman相关
        spearman_r, spearman_p = stats.spearmanr(x, y)

        # 交叉相关 (找最佳滞后)
        best_lag = 0
        best_corr = abs(pearson_r)

        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                continue
            if lag > 0:
                x_shifted = x[lag:]
                y_shifted = y[:-lag]
            else:
                x_shifted = x[:lag]
                y_shifted = y[-lag:]

            if len(x_shifted) > 2:
                corr, _ = stats.pearsonr(x_shifted, y_shifted)
                if abs(corr) > best_corr:
                    best_corr = abs(corr)
                    best_lag = lag

        # 判断强度
        abs_r = abs(pearson_r)
        if abs_r >= 0.7:
            strength = 'strong'
        elif abs_r >= 0.4:
            strength = 'moderate'
        elif abs_r >= 0.2:
            strength = 'weak'
        else:
            strength = 'none'

        return CorrelationAnalysis(
            pearson_r=float(pearson_r),
            pearson_p=float(pearson_p),
            spearman_r=float(spearman_r),
            spearman_p=float(spearman_p),
            lag=best_lag,
            strength=strength,
        )

    def correlation_matrix(
        self,
        data: Dict[str, np.ndarray],
    ) -> Dict[Tuple[str, str], float]:
        """
        计算相关矩阵.

        Args:
            data: {变量名: 数据数组} 字典

        Returns:
            {(变量1, 变量2): 相关系数} 字典
        """
        names = list(data.keys())
        result = {}

        for i, name1 in enumerate(names):
            for name2 in names[i:]:
                x = data[name1]
                y = data[name2]

                min_len = min(len(x), len(y))
                if min_len > 2:
                    corr, _ = stats.pearsonr(x[:min_len], y[:min_len])
                    result[(name1, name2)] = float(corr)
                    result[(name2, name1)] = float(corr)

        return result

    # -------------------------------------------------------------------------
    # Spectral Analysis
    # -------------------------------------------------------------------------

    def analyze_spectrum(
        self,
        values: np.ndarray,
        sample_rate: float = 10.0,
    ) -> SpectralAnalysis:
        """
        频谱分析.

        Args:
            values: 数据值数组
            sample_rate: 采样率 [Hz]

        Returns:
            SpectralAnalysis 结果
        """
        if len(values) < 4:
            return SpectralAnalysis(
                dominant_frequency=0.0,
                dominant_amplitude=0.0,
                frequencies=np.array([]),
                amplitudes=np.array([]),
                bandwidth=0.0,
            )

        # 去均值
        values_centered = values - np.mean(values)

        # FFT
        n = len(values_centered)
        fft_result = np.fft.rfft(values_centered)
        frequencies = np.fft.rfftfreq(n, 1.0 / sample_rate)
        amplitudes = np.abs(fft_result) * 2 / n

        # 找主频
        if len(amplitudes) > 1:
            # 排除DC分量
            idx = np.argmax(amplitudes[1:]) + 1
            dominant_freq = frequencies[idx]
            dominant_amp = amplitudes[idx]
        else:
            dominant_freq = 0.0
            dominant_amp = 0.0

        # 计算带宽 (-3dB)
        if dominant_amp > 0:
            threshold = dominant_amp / np.sqrt(2)
            above_threshold = amplitudes >= threshold
            if np.any(above_threshold):
                indices = np.where(above_threshold)[0]
                bandwidth = frequencies[indices[-1]] - frequencies[indices[0]]
            else:
                bandwidth = 0.0
        else:
            bandwidth = 0.0

        return SpectralAnalysis(
            dominant_frequency=float(dominant_freq),
            dominant_amplitude=float(dominant_amp),
            frequencies=frequencies,
            amplitudes=amplitudes,
            bandwidth=float(bandwidth),
        )

    def detect_periodicity(
        self,
        values: np.ndarray,
        sample_rate: float = 10.0,
        threshold: float = 0.1,
    ) -> List[Tuple[float, float]]:
        """
        检测周期性成分.

        Args:
            values: 数据值数组
            sample_rate: 采样率
            threshold: 相对幅值阈值

        Returns:
            [(频率, 幅值), ...] 列表
        """
        spectrum = self.analyze_spectrum(values, sample_rate)

        if len(spectrum.amplitudes) == 0:
            return []

        # 找峰值
        peaks = []
        max_amp = np.max(spectrum.amplitudes)

        for i in range(1, len(spectrum.amplitudes) - 1):
            amp = spectrum.amplitudes[i]
            if amp > spectrum.amplitudes[i-1] and amp > spectrum.amplitudes[i+1]:
                if amp > max_amp * threshold:
                    peaks.append((
                        float(spectrum.frequencies[i]),
                        float(amp),
                    ))

        return sorted(peaks, key=lambda x: x[1], reverse=True)

    # -------------------------------------------------------------------------
    # Statistical Analysis
    # -------------------------------------------------------------------------

    def basic_stats(self, values: np.ndarray) -> Dict[str, float]:
        """基本统计."""
        if len(values) == 0:
            return {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'q25': 0.0,
                'q75': 0.0,
            }

        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
        }

    def normality_test(self, values: np.ndarray) -> Dict[str, Any]:
        """正态性检验."""
        if len(values) < 8:
            return {'is_normal': None, 'p_value': None, 'test': None}

        # Shapiro-Wilk test
        stat, p_value = stats.shapiro(values[:5000])  # 限制样本量

        return {
            'is_normal': p_value > 0.05,
            'p_value': float(p_value),
            'statistic': float(stat),
            'test': 'shapiro-wilk',
        }

    def outlier_detection(
        self,
        values: np.ndarray,
        method: str = 'iqr',
        threshold: float = 1.5,
    ) -> np.ndarray:
        """
        异常值检测.

        Args:
            values: 数据值数组
            method: 方法 ('iqr', 'zscore', 'mad')
            threshold: 阈值

        Returns:
            异常值索引数组
        """
        if len(values) < 4:
            return np.array([])

        if method == 'iqr':
            q25, q75 = np.percentile(values, [25, 75])
            iqr = q75 - q25
            lower = q25 - threshold * iqr
            upper = q75 + threshold * iqr
            outliers = np.where((values < lower) | (values > upper))[0]

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(values))
            outliers = np.where(z_scores > threshold)[0]

        elif method == 'mad':
            # Median Absolute Deviation
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            if mad > 0:
                modified_z = 0.6745 * (values - median) / mad
                outliers = np.where(np.abs(modified_z) > threshold)[0]
            else:
                outliers = np.array([])

        else:
            outliers = np.array([])

        return outliers

    # -------------------------------------------------------------------------
    # Moving Statistics
    # -------------------------------------------------------------------------

    def moving_average(
        self,
        values: np.ndarray,
        window: int = 10,
    ) -> np.ndarray:
        """移动平均."""
        if len(values) < window:
            return values.copy()

        return np.convolve(values, np.ones(window)/window, mode='valid')

    def moving_std(
        self,
        values: np.ndarray,
        window: int = 10,
    ) -> np.ndarray:
        """移动标准差."""
        if len(values) < window:
            return np.zeros_like(values)

        result = np.zeros(len(values) - window + 1)
        for i in range(len(result)):
            result[i] = np.std(values[i:i+window])

        return result

    def exponential_smoothing(
        self,
        values: np.ndarray,
        alpha: float = 0.3,
    ) -> np.ndarray:
        """指数平滑."""
        result = np.zeros_like(values, dtype=float)
        result[0] = values[0]

        for i in range(1, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]

        return result
