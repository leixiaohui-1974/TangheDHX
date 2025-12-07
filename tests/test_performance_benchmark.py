# -*- coding: utf-8 -*-
"""
Performance Benchmark Tests.

性能基准测试模块，验证实时控制性能：
- 控制器计算延迟
- 物理模型步进性能
- 数据存储吞吐量
- 场景检测延迟
- 端到端响应时间
"""

import pytest
import time
import statistics
import numpy as np
from typing import List, Tuple

from src.simulation.physics import TangheSiphonModel
from src.control.mpc import SpectralMPC
from src.control.pid import MultiChannelPID, HybridController
from src.control.integrated_controller import IntegratedController, ScenarioType
from src.control.scenario_advanced import ScenarioDetector
from src.data.storage import TimeSeriesStorage
from src.data.analysis import DataAnalyzer


# =============================================================================
# Benchmark Configuration
# =============================================================================

# Real-time control requirement: < 10ms per control cycle
REALTIME_THRESHOLD_MS = 10.0

# Number of iterations for benchmark
BENCHMARK_ITERATIONS = 100


def measure_execution_time(func, iterations: int = BENCHMARK_ITERATIONS) -> Tuple[float, float, float]:
    """
    Measure execution time statistics.

    Args:
        func: Function to measure
        iterations: Number of iterations

    Returns:
        (mean_ms, std_ms, max_ms)
    """
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return (
        statistics.mean(times),
        statistics.stdev(times) if len(times) > 1 else 0.0,
        max(times)
    )


# =============================================================================
# Physics Model Performance Tests
# =============================================================================

class TestPhysicsModelPerformance:
    """Performance tests for physics simulation."""

    @pytest.fixture
    def model(self):
        """Create physics model."""
        return TangheSiphonModel()

    def test_step_performance(self, model):
        """Test physics step execution time."""
        openings = [0.5, 0.5, 0.5]
        dt = 0.1

        def step_func():
            model.step(openings, dt)

        mean_ms, std_ms, max_ms = measure_execution_time(step_func)

        print(f"\nPhysics step: mean={mean_ms:.3f}ms, std={std_ms:.3f}ms, max={max_ms:.3f}ms")

        # Physics step should be very fast (< 1ms)
        assert mean_ms < 1.0, f"Physics step too slow: {mean_ms:.3f}ms"
        assert max_ms < 5.0, f"Physics step max too high: {max_ms:.3f}ms"

    def test_state_query_performance(self, model):
        """Test state query performance."""
        def query_func():
            model.get_state()

        mean_ms, std_ms, max_ms = measure_execution_time(query_func)

        print(f"\nState query: mean={mean_ms:.3f}ms, std={std_ms:.3f}ms, max={max_ms:.3f}ms")

        assert mean_ms < 0.5, f"State query too slow: {mean_ms:.3f}ms"

    def test_reset_performance(self, model):
        """Test model reset performance."""
        def reset_func():
            model.reset()

        mean_ms, std_ms, max_ms = measure_execution_time(reset_func)

        print(f"\nModel reset: mean={mean_ms:.3f}ms, std={std_ms:.3f}ms, max={max_ms:.3f}ms")

        assert mean_ms < 1.0, f"Model reset too slow: {mean_ms:.3f}ms"


# =============================================================================
# Controller Performance Tests
# =============================================================================

class TestMPCPerformance:
    """Performance tests for MPC controller."""

    @pytest.fixture
    def mpc(self):
        """Create MPC controller."""
        model = TangheSiphonModel()
        return SpectralMPC(model)

    def test_mpc_computation(self, mpc):
        """Test MPC computation time."""
        target_flow = 150.0
        current_openings = [0.5, 0.5, 0.5]
        head_diff = 3.0

        def compute_func():
            mpc.get_target_openings(target_flow, current_openings, head_diff)

        mean_ms, std_ms, max_ms = measure_execution_time(compute_func, iterations=50)

        print(f"\nMPC computation: mean={mean_ms:.3f}ms, std={std_ms:.3f}ms, max={max_ms:.3f}ms")

        # MPC should complete within real-time budget
        assert mean_ms < REALTIME_THRESHOLD_MS, f"MPC too slow: {mean_ms:.3f}ms"

    def test_mpc_with_constraints(self, mpc):
        """Test MPC with various constraint scenarios."""
        scenarios = [
            (50.0, [0.2, 0.2, 0.2], 2.0),   # Low flow
            (150.0, [0.5, 0.5, 0.5], 3.0),  # Medium flow
            (350.0, [0.8, 0.8, 0.8], 5.0),  # High flow
        ]

        for target, openings, head in scenarios:
            def compute_func():
                mpc.get_target_openings(target, openings, head)

            mean_ms, _, max_ms = measure_execution_time(compute_func, iterations=30)
            print(f"  MPC (target={target}): mean={mean_ms:.3f}ms, max={max_ms:.3f}ms")

            assert mean_ms < REALTIME_THRESHOLD_MS


class TestPIDPerformance:
    """Performance tests for PID controller."""

    @pytest.fixture
    def pid(self):
        """Create PID controller."""
        return MultiChannelPID(num_channels=3)

    def test_pid_update(self, pid):
        """Test PID update time."""
        setpoints = [0.5, 0.5, 0.5]
        measurements = [0.48, 0.52, 0.50]
        dt = 0.1

        def update_func():
            pid.update(setpoints, measurements, dt)

        mean_ms, std_ms, max_ms = measure_execution_time(update_func)

        print(f"\nPID update: mean={mean_ms:.3f}ms, std={std_ms:.3f}ms, max={max_ms:.3f}ms")

        # PID should be very fast (< 0.1ms)
        assert mean_ms < 0.5, f"PID too slow: {mean_ms:.3f}ms"


class TestIntegratedControllerPerformance:
    """Performance tests for integrated controller."""

    @pytest.fixture
    def controller(self):
        """Create integrated controller."""
        model = TangheSiphonModel()
        return IntegratedController(model)

    def test_full_update_cycle(self, controller):
        """Test full controller update cycle."""
        controller.set_target_flow(150.0)
        dt = 0.1

        def update_func():
            controller.update(dt)

        mean_ms, std_ms, max_ms = measure_execution_time(update_func, iterations=50)

        print(f"\nIntegrated controller update: mean={mean_ms:.3f}ms, std={std_ms:.3f}ms, max={max_ms:.3f}ms")

        # Full control cycle must meet real-time requirement
        assert mean_ms < REALTIME_THRESHOLD_MS, f"Controller too slow: {mean_ms:.3f}ms"
        assert max_ms < REALTIME_THRESHOLD_MS * 2, f"Controller max too high: {max_ms:.3f}ms"

    def test_scenario_transitions(self, controller):
        """Test performance during scenario transitions."""
        scenarios = [
            ScenarioType.NORMAL_LOW_FLOW,
            ScenarioType.RESONANCE_CROSSING,
            ScenarioType.NORMAL_HIGH_FLOW,
            ScenarioType.EMERGENCY_GATE_FAULT,
        ]

        times = []
        for scenario in scenarios:
            controller.force_scenario(scenario)
            controller.set_target_flow(150.0)

            start = time.perf_counter()
            controller.update(0.1)
            end = time.perf_counter()

            times.append((end - start) * 1000)

        print(f"\nScenario transition times: {[f'{t:.3f}ms' for t in times]}")

        for t in times:
            assert t < REALTIME_THRESHOLD_MS * 1.5


# =============================================================================
# Scenario Detection Performance Tests
# =============================================================================

class TestScenarioDetectorPerformance:
    """Performance tests for scenario detection."""

    @pytest.fixture
    def detector(self):
        """Create scenario detector."""
        model = TangheSiphonModel()
        return ScenarioDetector(model)

    def test_detection_performance(self, detector):
        """Test scenario detection time."""
        def detect_func():
            detector.detect()

        mean_ms, std_ms, max_ms = measure_execution_time(detect_func)

        print(f"\nScenario detection: mean={mean_ms:.3f}ms, std={std_ms:.3f}ms, max={max_ms:.3f}ms")

        # Detection should be fast (< 1ms)
        assert mean_ms < 2.0, f"Detection too slow: {mean_ms:.3f}ms"


# =============================================================================
# Data Storage Performance Tests
# =============================================================================

class TestStoragePerformance:
    """Performance tests for data storage."""

    @pytest.fixture
    def storage(self):
        """Create storage instance."""
        return TimeSeriesStorage()

    def test_single_write_performance(self, storage):
        """Test single write performance."""
        storage.create_series('test')
        i = [0]

        def write_func():
            storage.write('test', float(i[0]), float(i[0]))
            i[0] += 1

        mean_ms, std_ms, max_ms = measure_execution_time(write_func, iterations=1000)

        print(f"\nStorage write: mean={mean_ms:.3f}ms, std={std_ms:.3f}ms, max={max_ms:.3f}ms")

        # Write should be very fast (< 0.1ms)
        assert mean_ms < 0.5, f"Write too slow: {mean_ms:.3f}ms"

    def test_batch_write_performance(self, storage):
        """Test batch write performance."""
        def batch_func():
            data = {
                'flow': 100.0,
                'pressure': 5.0,
                'temp': 25.0,
                'velocity': 2.5,
                'vibration': 10.0,
            }
            storage.write_dict(data)

        mean_ms, std_ms, max_ms = measure_execution_time(batch_func, iterations=100)

        print(f"\nBatch write (5 series): mean={mean_ms:.3f}ms, std={std_ms:.3f}ms, max={max_ms:.3f}ms")

        assert mean_ms < 1.0, f"Batch write too slow: {mean_ms:.3f}ms"

    def test_read_performance(self, storage):
        """Test read performance."""
        # Pre-populate data
        for i in range(1000):
            storage.write('test', float(i), float(i))

        def read_func():
            storage.read('test', 0.0, 1000.0, limit=100)

        mean_ms, std_ms, max_ms = measure_execution_time(read_func)

        print(f"\nStorage read (100 points): mean={mean_ms:.3f}ms, std={std_ms:.3f}ms, max={max_ms:.3f}ms")

        assert mean_ms < 2.0, f"Read too slow: {mean_ms:.3f}ms"

    def test_throughput(self, storage):
        """Test write throughput."""
        storage.create_series('throughput')

        iterations = 10000
        start = time.perf_counter()

        for i in range(iterations):
            storage.write('throughput', float(i), float(i))

        elapsed = time.perf_counter() - start
        throughput = iterations / elapsed

        print(f"\nWrite throughput: {throughput:.0f} points/sec")

        # Should handle at least 10000 writes/sec
        assert throughput > 10000, f"Throughput too low: {throughput:.0f}/sec"


# =============================================================================
# Data Analysis Performance Tests
# =============================================================================

class TestAnalysisPerformance:
    """Performance tests for data analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return DataAnalyzer()

    def test_basic_stats_performance(self, analyzer):
        """Test basic stats calculation performance."""
        data = np.random.randn(1000)

        def stats_func():
            analyzer.basic_stats(data)

        mean_ms, std_ms, max_ms = measure_execution_time(stats_func)

        print(f"\nBasic stats (1000 points): mean={mean_ms:.3f}ms, std={std_ms:.3f}ms, max={max_ms:.3f}ms")

        assert mean_ms < 1.0

    def test_trend_analysis_performance(self, analyzer):
        """Test trend analysis performance."""
        data = np.random.randn(500)

        def trend_func():
            analyzer.analyze_trend(data)

        mean_ms, std_ms, max_ms = measure_execution_time(trend_func)

        print(f"\nTrend analysis (500 points): mean={mean_ms:.3f}ms, std={std_ms:.3f}ms, max={max_ms:.3f}ms")

        assert mean_ms < 5.0

    def test_spectral_analysis_performance(self, analyzer):
        """Test spectral analysis performance."""
        data = np.random.randn(1000)

        def spectral_func():
            analyzer.analyze_spectrum(data, sample_rate=100.0)

        mean_ms, std_ms, max_ms = measure_execution_time(spectral_func)

        print(f"\nSpectral analysis (1000 points): mean={mean_ms:.3f}ms, std={std_ms:.3f}ms, max={max_ms:.3f}ms")

        assert mean_ms < 10.0


# =============================================================================
# End-to-End Performance Tests
# =============================================================================

class TestEndToEndPerformance:
    """End-to-end performance tests."""

    def test_full_control_loop(self):
        """Test complete control loop performance."""
        # Setup
        model = TangheSiphonModel()
        controller = IntegratedController(model)
        storage = TimeSeriesStorage()

        storage.create_series('flow')
        storage.create_series('target')

        controller.set_target_flow(150.0)

        def control_loop():
            # 1. Run controller
            diagnostics = controller.update(0.1)

            # 2. Get state
            state = model.get_state()

            # 3. Store data
            ts = time.time()
            storage.write('flow', state['total_flow'], ts)
            storage.write('target', 150.0, ts)

            return diagnostics

        mean_ms, std_ms, max_ms = measure_execution_time(control_loop, iterations=50)

        print(f"\nFull control loop: mean={mean_ms:.3f}ms, std={std_ms:.3f}ms, max={max_ms:.3f}ms")

        # Complete loop must meet real-time requirement
        assert mean_ms < REALTIME_THRESHOLD_MS, f"Control loop too slow: {mean_ms:.3f}ms"

    def test_sustained_operation(self):
        """Test sustained operation performance."""
        model = TangheSiphonModel()
        controller = IntegratedController(model)

        controller.set_target_flow(150.0)

        # Run for simulated 10 seconds (100 steps at dt=0.1)
        times = []
        for _ in range(100):
            start = time.perf_counter()
            controller.update(0.1)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        mean_ms = statistics.mean(times)
        max_ms = max(times)
        p99_ms = sorted(times)[98]  # 99th percentile

        print(f"\nSustained operation (100 steps):")
        print(f"  Mean: {mean_ms:.3f}ms")
        print(f"  Max: {max_ms:.3f}ms")
        print(f"  P99: {p99_ms:.3f}ms")

        assert mean_ms < REALTIME_THRESHOLD_MS
        assert p99_ms < REALTIME_THRESHOLD_MS * 1.5

    def test_memory_stability(self):
        """Test memory doesn't grow during operation."""
        import sys

        model = TangheSiphonModel()
        controller = IntegratedController(model)
        storage = TimeSeriesStorage(max_points_per_series=1000)

        controller.set_target_flow(150.0)

        # Get initial memory estimate
        initial_size = sys.getsizeof(storage._series)

        # Run many iterations
        for i in range(5000):
            controller.update(0.1)
            state = model.get_state()
            storage.write('flow', state['total_flow'], float(i))

        # Check memory hasn't grown significantly
        final_size = sys.getsizeof(storage._series)

        # Storage should be bounded by max_points
        points = storage.get_series('flow')
        assert len(points.points) <= 1000

        print(f"\nMemory: initial={initial_size}, final={final_size}, points={len(points.points)}")


# =============================================================================
# Benchmark Summary
# =============================================================================

class TestBenchmarkSummary:
    """Generate benchmark summary."""

    def test_generate_summary(self):
        """Generate comprehensive benchmark summary."""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Real-time threshold: {REALTIME_THRESHOLD_MS}ms per control cycle")
        print(f"Benchmark iterations: {BENCHMARK_ITERATIONS}")
        print("=" * 60)

        results = {}

        # Physics
        model = TangheSiphonModel()
        mean, _, max_t = measure_execution_time(lambda: model.step([0.5]*3, 0.1), 50)
        results['Physics step'] = mean
        print(f"Physics step:        {mean:.3f}ms (max: {max_t:.3f}ms)")

        # MPC
        mpc = SpectralMPC(model)
        mean, _, max_t = measure_execution_time(
            lambda: mpc.get_target_openings(150.0, [0.5]*3, 3.0), 30
        )
        results['MPC computation'] = mean
        print(f"MPC computation:     {mean:.3f}ms (max: {max_t:.3f}ms)")

        # Integrated Controller
        controller = IntegratedController(model)
        controller.set_target_flow(150.0)
        mean, _, max_t = measure_execution_time(lambda: controller.update(0.1), 50)
        results['Controller update'] = mean
        print(f"Controller update:   {mean:.3f}ms (max: {max_t:.3f}ms)")

        # Storage
        storage = TimeSeriesStorage()
        storage.create_series('test')
        i = [0]
        mean, _, max_t = measure_execution_time(
            lambda: storage.write('test', float(i[0]), float(i[0])) or (i.__setitem__(0, i[0]+1)),
            500
        )
        results['Storage write'] = mean
        print(f"Storage write:       {mean:.3f}ms (max: {max_t:.3f}ms)")

        print("=" * 60)

        # Check all meet real-time requirement
        all_pass = all(t < REALTIME_THRESHOLD_MS for t in results.values())
        status = "PASS" if all_pass else "FAIL"
        print(f"Overall status: {status}")
        print("=" * 60)

        assert all_pass, "Some benchmarks exceeded real-time threshold"
