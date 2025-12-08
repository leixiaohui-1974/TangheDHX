"""
Unit tests for Automated Optimization Module.

Tests:
- Parameter bounds and specs
- Optimization algorithms
- Controller optimizer
- Calibration optimizer
- Auto tuner
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from src.optimization.optimizer import (
    # Enums
    OptimizationAlgorithm,
    OptimizationStatus,
    OptimizationObjective,
    # Data classes
    ParameterBounds,
    ParameterSpec,
    OptimizationConfig,
    OptimizationResult,
    OptimizationHistory,
    # Classes
    ObjectiveFunction,
    SimpleObjective,
    GradientOptimizer,
    EvolutionaryOptimizer,
    BayesianOptimizer,
    GridSearchOptimizer,
    ControllerOptimizer,
    CalibrationOptimizer,
    AutoTuner
)


# =============================================================================
# Enum Tests
# =============================================================================

class TestEnums:
    """Test enumeration values."""

    def test_algorithm_values(self):
        """Test OptimizationAlgorithm enum values."""
        assert OptimizationAlgorithm.GRADIENT_DESCENT is not None
        assert OptimizationAlgorithm.EVOLUTIONARY is not None
        assert OptimizationAlgorithm.BAYESIAN is not None
        assert OptimizationAlgorithm.GRID_SEARCH is not None

    def test_status_values(self):
        """Test OptimizationStatus enum values."""
        assert OptimizationStatus.IDLE is not None
        assert OptimizationStatus.RUNNING is not None
        assert OptimizationStatus.CONVERGED is not None
        assert OptimizationStatus.FAILED is not None

    def test_objective_values(self):
        """Test OptimizationObjective enum values."""
        assert OptimizationObjective.MINIMIZE is not None
        assert OptimizationObjective.MAXIMIZE is not None


# =============================================================================
# Parameter Bounds Tests
# =============================================================================

class TestParameterBounds:
    """Test ParameterBounds class."""

    def test_creation(self):
        """Test bounds creation."""
        bounds = ParameterBounds(0.0, 10.0)
        assert bounds.lower == 0.0
        assert bounds.upper == 10.0

    def test_invalid_bounds(self):
        """Test that invalid bounds raise error."""
        with pytest.raises(ValueError):
            ParameterBounds(10.0, 5.0)

    def test_equal_bounds_invalid(self):
        """Test that equal bounds raise error."""
        with pytest.raises(ValueError):
            ParameterBounds(5.0, 5.0)

    def test_clip_within(self):
        """Test clipping value within bounds."""
        bounds = ParameterBounds(0.0, 10.0)
        assert bounds.clip(5.0) == 5.0

    def test_clip_below(self):
        """Test clipping value below lower bound."""
        bounds = ParameterBounds(0.0, 10.0)
        assert bounds.clip(-5.0) == 0.0

    def test_clip_above(self):
        """Test clipping value above upper bound."""
        bounds = ParameterBounds(0.0, 10.0)
        assert bounds.clip(15.0) == 10.0

    def test_contains_true(self):
        """Test contains with value in bounds."""
        bounds = ParameterBounds(0.0, 10.0)
        assert bounds.contains(5.0) is True

    def test_contains_false(self):
        """Test contains with value out of bounds."""
        bounds = ParameterBounds(0.0, 10.0)
        assert bounds.contains(15.0) is False

    def test_sample(self):
        """Test uniform sampling."""
        bounds = ParameterBounds(0.0, 10.0)
        rng = np.random.default_rng(42)
        samples = [bounds.sample(rng) for _ in range(100)]

        assert all(bounds.contains(s) for s in samples)
        assert min(samples) >= 0.0
        assert max(samples) <= 10.0


# =============================================================================
# Parameter Spec Tests
# =============================================================================

class TestParameterSpec:
    """Test ParameterSpec class."""

    def test_creation(self):
        """Test spec creation."""
        spec = ParameterSpec(
            name='kp',
            bounds=ParameterBounds(0.1, 10.0),
            initial=1.0
        )
        assert spec.name == 'kp'
        assert spec.initial == 1.0

    def test_default_initial(self):
        """Test default initial value is midpoint."""
        spec = ParameterSpec(
            name='kp',
            bounds=ParameterBounds(0.0, 10.0)
        )
        assert spec.initial == 5.0

    def test_default_step_size(self):
        """Test default step size."""
        spec = ParameterSpec(
            name='kp',
            bounds=ParameterBounds(0.0, 10.0)
        )
        assert spec.step_size == 0.1  # (10-0)/100

    def test_integer_flag(self):
        """Test integer parameter flag."""
        spec = ParameterSpec(
            name='n_layers',
            bounds=ParameterBounds(1.0, 10.0),
            is_integer=True
        )
        assert spec.is_integer is True


# =============================================================================
# Optimization Config Tests
# =============================================================================

class TestOptimizationConfig:
    """Test OptimizationConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OptimizationConfig()
        assert config.max_iterations == 100
        assert config.tolerance == 1e-6
        assert config.population_size == 20
        assert config.early_stopping is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = OptimizationConfig(
            max_iterations=50,
            tolerance=1e-4,
            seed=42
        )
        assert config.max_iterations == 50
        assert config.seed == 42


# =============================================================================
# Optimization Result Tests
# =============================================================================

class TestOptimizationResult:
    """Test OptimizationResult class."""

    def test_creation(self):
        """Test result creation."""
        result = OptimizationResult(
            result_id='test_001',
            algorithm=OptimizationAlgorithm.BAYESIAN,
            status=OptimizationStatus.CONVERGED,
            best_params={'x': 1.5},
            best_value=0.001,
            iterations=50,
            evaluations=100,
            elapsed_seconds=5.0
        )
        assert result.result_id == 'test_001'
        assert result.best_value == 0.001

    def test_convergence_history(self):
        """Test convergence history."""
        result = OptimizationResult(
            result_id='test',
            algorithm=OptimizationAlgorithm.GRADIENT_DESCENT,
            status=OptimizationStatus.CONVERGED,
            best_params={},
            best_value=0.0,
            iterations=10,
            evaluations=10,
            elapsed_seconds=1.0,
            convergence_history=[1.0, 0.5, 0.2, 0.1, 0.05]
        )
        assert len(result.convergence_history) == 5


# =============================================================================
# Optimization History Tests
# =============================================================================

class TestOptimizationHistory:
    """Test OptimizationHistory class."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results."""
        return [
            OptimizationResult(
                result_id=f'r{i}',
                algorithm=OptimizationAlgorithm.BAYESIAN,
                status=OptimizationStatus.CONVERGED,
                best_params={'x': float(i)},
                best_value=float(i),
                iterations=10,
                evaluations=20,
                elapsed_seconds=1.0
            )
            for i in range(5)
        ]

    def test_add(self, sample_results):
        """Test adding results."""
        history = OptimizationHistory()
        for r in sample_results:
            history.add(r)
        assert len(history.runs) == 5

    def test_get_best_minimize(self, sample_results):
        """Test getting best result for minimization."""
        history = OptimizationHistory()
        for r in sample_results:
            history.add(r)

        best = history.get_best(OptimizationObjective.MINIMIZE)
        assert best.best_value == 0.0

    def test_get_best_maximize(self, sample_results):
        """Test getting best result for maximization."""
        history = OptimizationHistory()
        for r in sample_results:
            history.add(r)

        best = history.get_best(OptimizationObjective.MAXIMIZE)
        assert best.best_value == 4.0

    def test_get_recent(self, sample_results):
        """Test getting recent results."""
        history = OptimizationHistory()
        for r in sample_results:
            history.add(r)

        recent = history.get_recent(3)
        assert len(recent) == 3
        assert recent[0].result_id == 'r4'  # Most recent first

    def test_get_statistics(self, sample_results):
        """Test statistics computation."""
        history = OptimizationHistory()
        for r in sample_results:
            history.add(r)

        stats = history.get_statistics()
        assert stats['count'] == 5
        assert stats['best_value'] == 0.0
        assert stats['worst_value'] == 4.0
        assert stats['mean_value'] == 2.0

    def test_empty_history(self):
        """Test empty history."""
        history = OptimizationHistory()
        assert history.get_best() is None
        assert history.get_statistics()['count'] == 0


# =============================================================================
# Objective Function Tests
# =============================================================================

class TestObjectiveFunction:
    """Test ObjectiveFunction classes."""

    def test_simple_objective(self):
        """Test SimpleObjective wrapper."""
        def rosenbrock(params):
            x = params.get('x', 0)
            y = params.get('y', 0)
            return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

        obj = SimpleObjective(rosenbrock)
        value = obj.evaluate({'x': 1.0, 'y': 1.0})
        assert value == pytest.approx(0.0, abs=1e-10)

    def test_numerical_gradient(self):
        """Test numerical gradient computation."""
        def quadratic(params):
            return params['x'] ** 2

        obj = SimpleObjective(quadratic)
        grad = obj.gradient({'x': 2.0})
        assert grad['x'] == pytest.approx(4.0, rel=1e-4)


# =============================================================================
# Gradient Optimizer Tests
# =============================================================================

class TestGradientOptimizer:
    """Test GradientOptimizer class."""

    @pytest.fixture
    def quadratic_problem(self):
        """Create simple quadratic optimization problem."""
        params = [
            ParameterSpec('x', ParameterBounds(-10.0, 10.0), initial=5.0)
        ]

        def objective(p):
            return (p['x'] - 2.0) ** 2

        return params, SimpleObjective(objective)

    def test_initialization(self, quadratic_problem):
        """Test optimizer initialization."""
        params, obj = quadratic_problem
        optimizer = GradientOptimizer(params, obj)
        assert optimizer.algorithm == OptimizationAlgorithm.GRADIENT_DESCENT

    def test_optimize_quadratic(self, quadratic_problem):
        """Test optimization of simple quadratic."""
        params, obj = quadratic_problem
        config = OptimizationConfig(
            max_iterations=100,
            learning_rate=0.1,
            tolerance=1e-4
        )
        optimizer = GradientOptimizer(params, obj, config)

        result = optimizer.optimize()

        assert result.status in [OptimizationStatus.CONVERGED, OptimizationStatus.MAX_ITERATIONS]
        assert result.best_params['x'] == pytest.approx(2.0, abs=0.1)
        assert result.best_value < 0.01

    def test_convergence_history(self, quadratic_problem):
        """Test that convergence history is recorded."""
        params, obj = quadratic_problem
        config = OptimizationConfig(max_iterations=20)
        optimizer = GradientOptimizer(params, obj, config)

        result = optimizer.optimize()

        assert len(result.convergence_history) > 0
        # Should be decreasing trend
        assert result.convergence_history[-1] <= result.convergence_history[0]


# =============================================================================
# Evolutionary Optimizer Tests
# =============================================================================

class TestEvolutionaryOptimizer:
    """Test EvolutionaryOptimizer class."""

    @pytest.fixture
    def sphere_problem(self):
        """Create sphere optimization problem."""
        params = [
            ParameterSpec('x', ParameterBounds(-5.0, 5.0)),
            ParameterSpec('y', ParameterBounds(-5.0, 5.0))
        ]

        def objective(p):
            return p['x'] ** 2 + p['y'] ** 2

        return params, SimpleObjective(objective)

    def test_initialization(self, sphere_problem):
        """Test optimizer initialization."""
        params, obj = sphere_problem
        optimizer = EvolutionaryOptimizer(params, obj)
        assert optimizer.algorithm == OptimizationAlgorithm.EVOLUTIONARY

    def test_optimize_sphere(self, sphere_problem):
        """Test optimization of sphere function."""
        params, obj = sphere_problem
        config = OptimizationConfig(
            max_iterations=50,
            population_size=20,
            seed=42
        )
        optimizer = EvolutionaryOptimizer(params, obj, config)

        result = optimizer.optimize()

        assert result.status in [OptimizationStatus.CONVERGED, OptimizationStatus.MAX_ITERATIONS]
        assert abs(result.best_params['x']) < 1.0
        assert abs(result.best_params['y']) < 1.0

    def test_population_evaluation(self, sphere_problem):
        """Test that entire population is evaluated."""
        params, obj = sphere_problem
        config = OptimizationConfig(
            max_iterations=5,
            population_size=10
        )
        optimizer = EvolutionaryOptimizer(params, obj, config)

        result = optimizer.optimize()

        # At least population_size * iterations evaluations
        assert result.evaluations >= 50


# =============================================================================
# Bayesian Optimizer Tests
# =============================================================================

class TestBayesianOptimizer:
    """Test BayesianOptimizer class."""

    @pytest.fixture
    def simple_problem(self):
        """Create simple optimization problem."""
        params = [
            ParameterSpec('x', ParameterBounds(0.0, 10.0))
        ]

        def objective(p):
            x = p['x']
            return (x - 3) ** 2 + 1  # Minimum at x=3

        return params, SimpleObjective(objective)

    def test_initialization(self, simple_problem):
        """Test optimizer initialization."""
        params, obj = simple_problem
        optimizer = BayesianOptimizer(params, obj)
        assert optimizer.algorithm == OptimizationAlgorithm.BAYESIAN

    def test_optimize(self, simple_problem):
        """Test Bayesian optimization."""
        params, obj = simple_problem
        config = OptimizationConfig(
            max_iterations=30,
            seed=42
        )
        optimizer = BayesianOptimizer(params, obj, config)

        result = optimizer.optimize()

        assert result.status in [OptimizationStatus.CONVERGED, OptimizationStatus.MAX_ITERATIONS]
        # Should find approximate minimum
        assert 1.0 < result.best_params['x'] < 5.0

    def test_exploration_exploitation(self, simple_problem):
        """Test that optimizer balances exploration and exploitation."""
        params, obj = simple_problem
        config = OptimizationConfig(max_iterations=20, seed=42)
        optimizer = BayesianOptimizer(params, obj, config)

        result = optimizer.optimize()

        # Parameter history should show variety (exploration)
        x_values = [p['x'] for p in result.parameter_history]
        assert max(x_values) - min(x_values) > 2.0  # Some spread


# =============================================================================
# Grid Search Optimizer Tests
# =============================================================================

class TestGridSearchOptimizer:
    """Test GridSearchOptimizer class."""

    @pytest.fixture
    def simple_problem(self):
        """Create simple problem."""
        params = [
            ParameterSpec('x', ParameterBounds(0.0, 4.0))
        ]

        def objective(p):
            return (p['x'] - 2) ** 2

        return params, SimpleObjective(objective)

    def test_initialization(self, simple_problem):
        """Test optimizer initialization."""
        params, obj = simple_problem
        optimizer = GridSearchOptimizer(params, obj, grid_points=5)
        assert optimizer.algorithm == OptimizationAlgorithm.GRID_SEARCH

    def test_grid_coverage(self, simple_problem):
        """Test that grid covers parameter space."""
        params, obj = simple_problem
        config = OptimizationConfig(max_iterations=10)
        optimizer = GridSearchOptimizer(params, obj, config, grid_points=5)

        result = optimizer.optimize()

        # Should evaluate exactly grid_points values
        assert result.evaluations == 5

    def test_find_minimum(self, simple_problem):
        """Test finding minimum on grid."""
        params, obj = simple_problem
        config = OptimizationConfig(max_iterations=100)
        optimizer = GridSearchOptimizer(params, obj, config, grid_points=11)

        result = optimizer.optimize()

        # Best should be close to x=2
        assert result.best_params['x'] == pytest.approx(2.0, abs=0.5)


# =============================================================================
# Controller Optimizer Tests
# =============================================================================

class TestControllerOptimizer:
    """Test ControllerOptimizer class."""

    @pytest.fixture
    def mock_controller(self):
        """Create mock controller."""
        controller = Mock()
        controller.get_gains.return_value = (1.0, 0.1, 0.01)
        controller.compute.return_value = 0.5
        return controller

    @pytest.fixture
    def mock_simulator(self):
        """Create mock simulator."""
        simulator = Mock()
        simulator.get_state.return_value = {'total_flow': 150.0}
        return simulator

    def test_initialization(self, mock_controller, mock_simulator):
        """Test optimizer initialization."""
        optimizer = ControllerOptimizer(mock_controller, mock_simulator)
        assert optimizer is not None

    def test_optimize_pid(self, mock_controller, mock_simulator):
        """Test PID optimization."""
        config = OptimizationConfig(max_iterations=10)
        optimizer = ControllerOptimizer(mock_controller, mock_simulator, config)

        result = optimizer.optimize_pid(
            algorithm=OptimizationAlgorithm.GRID_SEARCH,
            kp_bounds=(0.5, 2.0),
            ki_bounds=(0.0, 1.0),
            kd_bounds=(0.0, 0.5),
            simulation_time=10.0
        )

        assert result is not None
        assert 'kp' in result.best_params
        assert 'ki' in result.best_params
        assert 'kd' in result.best_params

    def test_optimize_mpc(self, mock_controller, mock_simulator):
        """Test MPC optimization."""
        mock_controller.set_weights = Mock()

        config = OptimizationConfig(max_iterations=10)
        optimizer = ControllerOptimizer(mock_controller, mock_simulator, config)

        result = optimizer.optimize_mpc(
            algorithm=OptimizationAlgorithm.GRID_SEARCH,
            q_bounds=(0.5, 5.0),
            r_bounds=(0.1, 1.0),
            simulation_time=10.0
        )

        assert result is not None
        assert 'q' in result.best_params
        assert 'r' in result.best_params

    def test_history_tracking(self, mock_controller, mock_simulator):
        """Test optimization history tracking."""
        config = OptimizationConfig(max_iterations=5)
        optimizer = ControllerOptimizer(mock_controller, mock_simulator, config)

        optimizer.optimize_pid(algorithm=OptimizationAlgorithm.GRID_SEARCH)
        optimizer.optimize_pid(algorithm=OptimizationAlgorithm.GRID_SEARCH)

        history = optimizer.get_history()
        assert len(history.runs) == 2


# =============================================================================
# Calibration Optimizer Tests
# =============================================================================

class TestCalibrationOptimizer:
    """Test CalibrationOptimizer class."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.simulate.return_value = np.array([1.0, 2.0, 3.0])
        return model

    def test_initialization(self, mock_model):
        """Test optimizer initialization."""
        optimizer = CalibrationOptimizer(mock_model)
        assert optimizer is not None

    def test_set_reference_data(self, mock_model):
        """Test setting reference data."""
        optimizer = CalibrationOptimizer(mock_model)
        data = {'flow': np.array([1.0, 2.0, 3.0])}
        optimizer.set_reference_data(data)
        assert optimizer._reference_data == data

    def test_optimize(self, mock_model):
        """Test calibration optimization."""
        reference = {'output': np.array([1.0, 2.0, 3.0])}
        config = OptimizationConfig(max_iterations=10)
        optimizer = CalibrationOptimizer(mock_model, reference, config)

        params = [
            ParameterSpec('param1', ParameterBounds(0.0, 10.0))
        ]

        result = optimizer.optimize(params, OptimizationAlgorithm.GRID_SEARCH)

        assert result is not None
        assert 'param1' in result.best_params

    def test_history_tracking(self, mock_model):
        """Test history tracking."""
        config = OptimizationConfig(max_iterations=5)
        optimizer = CalibrationOptimizer(mock_model, config=config)

        params = [ParameterSpec('p', ParameterBounds(0.0, 1.0))]
        optimizer.optimize(params, OptimizationAlgorithm.GRID_SEARCH)

        history = optimizer.get_history()
        assert len(history.runs) == 1


# =============================================================================
# Auto Tuner Tests
# =============================================================================

class TestAutoTuner:
    """Test AutoTuner class."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.get_state.return_value = {'total_flow': 150.0}
        model.step.return_value = None
        return model

    @pytest.fixture
    def mock_controller(self):
        """Create mock controller."""
        controller = Mock()
        controller.get_gains.return_value = (1.0, 0.1, 0.01)
        controller.compute.return_value = 0.5
        return controller

    @pytest.fixture
    def tuner(self, mock_model, mock_controller):
        """Create auto tuner."""
        config = OptimizationConfig(max_iterations=5)
        return AutoTuner(mock_model, mock_controller, config)

    def test_initialization(self, tuner):
        """Test tuner initialization."""
        assert tuner is not None
        assert tuner.is_running is False

    def test_tune_pid(self, tuner):
        """Test PID tuning."""
        result = tuner.tune_pid(algorithm=OptimizationAlgorithm.GRID_SEARCH)

        assert result is not None
        assert tuner.is_running is False

    def test_tune_mpc(self, tuner, mock_controller):
        """Test MPC tuning."""
        mock_controller.set_weights = Mock()
        result = tuner.tune_mpc(algorithm=OptimizationAlgorithm.GRID_SEARCH)

        assert result is not None

    def test_custom_optimize(self, tuner):
        """Test custom optimization."""
        params = [ParameterSpec('x', ParameterBounds(0.0, 10.0))]

        def objective(p):
            return (p['x'] - 5) ** 2

        result = tuner.custom_optimize(
            params,
            objective,
            algorithm=OptimizationAlgorithm.GRID_SEARCH,
            task_name='custom_test'
        )

        assert result is not None

    def test_calibrate(self, tuner):
        """Test calibration."""
        params = [ParameterSpec('c', ParameterBounds(0.0, 1.0))]
        reference = {'output': np.array([1.0, 2.0, 3.0])}

        result = tuner.calibrate(
            params,
            reference,
            algorithm=OptimizationAlgorithm.GRID_SEARCH
        )

        assert result is not None

    def test_get_status(self, tuner):
        """Test status retrieval."""
        status = tuner.get_status()

        assert 'running' in status
        assert 'current_task' in status
        assert 'history' in status

    def test_concurrent_prevention(self, tuner):
        """Test that concurrent optimization is prevented."""
        # Start a long-running optimization in a thread
        import threading

        def long_optimize():
            time.sleep(0.5)

        # Manually set running state
        tuner._running = True
        tuner._current_task = 'test'

        # Try to start another optimization
        result = tuner.tune_pid(algorithm=OptimizationAlgorithm.GRID_SEARCH)

        # Should return None due to running check
        assert result is None

        tuner._running = False

    def test_history_accumulation(self, tuner, mock_controller):
        """Test that history accumulates across runs."""
        mock_controller.set_weights = Mock()

        tuner.tune_pid(algorithm=OptimizationAlgorithm.GRID_SEARCH)
        tuner.tune_mpc(algorithm=OptimizationAlgorithm.GRID_SEARCH)

        history = tuner.get_history()
        assert len(history.runs) == 2


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for optimization module."""

    def test_rosenbrock_optimization(self):
        """Test optimizing Rosenbrock function."""
        params = [
            ParameterSpec('x', ParameterBounds(-5.0, 5.0), initial=0.0),
            ParameterSpec('y', ParameterBounds(-5.0, 5.0), initial=0.0)
        ]

        def rosenbrock(p):
            x, y = p['x'], p['y']
            return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

        config = OptimizationConfig(max_iterations=100, seed=42)
        optimizer = EvolutionaryOptimizer(params, SimpleObjective(rosenbrock), config)

        result = optimizer.optimize()

        # Should get reasonably close to (1, 1)
        assert result.best_value < 10.0

    def test_multi_algorithm_comparison(self):
        """Test comparing multiple algorithms."""
        params = [ParameterSpec('x', ParameterBounds(-10.0, 10.0))]

        def objective(p):
            return (p['x'] - 3) ** 2

        config = OptimizationConfig(max_iterations=50, seed=42)

        results = {}
        for algo, OptimizerClass in [
            (OptimizationAlgorithm.GRADIENT_DESCENT, GradientOptimizer),
            (OptimizationAlgorithm.EVOLUTIONARY, EvolutionaryOptimizer),
            (OptimizationAlgorithm.BAYESIAN, BayesianOptimizer),
        ]:
            optimizer = OptimizerClass(params, SimpleObjective(objective), config)
            result = optimizer.optimize()
            results[algo] = result

        # All should find approximate minimum
        for algo, result in results.items():
            assert abs(result.best_params['x'] - 3.0) < 2.0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_parameter(self):
        """Test optimization with single parameter."""
        params = [ParameterSpec('x', ParameterBounds(0.0, 10.0))]

        def objective(p):
            return p['x'] ** 2

        config = OptimizationConfig(max_iterations=20)
        optimizer = GradientOptimizer(params, SimpleObjective(objective), config)

        result = optimizer.optimize()
        assert result.best_params['x'] < 1.0

    def test_integer_parameter(self):
        """Test optimization with integer parameter."""
        params = [
            ParameterSpec('n', ParameterBounds(1.0, 10.0), is_integer=True)
        ]

        def objective(p):
            return abs(p['n'] - 5)

        config = OptimizationConfig(max_iterations=20)
        optimizer = GridSearchOptimizer(params, SimpleObjective(objective), config, grid_points=10)

        result = optimizer.optimize()
        # Best value should be integer
        assert result.best_params['n'] == int(result.best_params['n'])

    def test_timeout(self):
        """Test timeout handling."""
        params = [ParameterSpec('x', ParameterBounds(0.0, 10.0))]

        call_count = [0]

        def slow_objective(p):
            call_count[0] += 1
            time.sleep(0.1)
            return p['x'] ** 2

        config = OptimizationConfig(
            max_iterations=1000,
            timeout_seconds=0.5
        )
        optimizer = GradientOptimizer(params, SimpleObjective(slow_objective), config)

        result = optimizer.optimize()
        assert result.status == OptimizationStatus.TIMEOUT
        assert call_count[0] < 100  # Should stop early

    def test_cancel_optimization(self):
        """Test cancellation during optimization."""
        params = [ParameterSpec('x', ParameterBounds(0.0, 10.0))]

        call_count = [0]

        def objective(p):
            call_count[0] += 1
            return p['x'] ** 2

        config = OptimizationConfig(max_iterations=1000)
        optimizer = EvolutionaryOptimizer(params, SimpleObjective(objective), config)

        # Start optimization in thread and cancel quickly
        import threading

        result_holder = [None]

        def run_opt():
            result_holder[0] = optimizer.optimize()

        thread = threading.Thread(target=run_opt)
        thread.start()
        time.sleep(0.01)  # Let it start
        optimizer.cancel()
        thread.join(timeout=5.0)

        result = result_holder[0]
        # Either cancelled or finished quickly before cancellation took effect
        assert result is not None
        assert result.iterations < 1000  # Stopped early

    def test_bounds_clipping(self):
        """Test that parameters are clipped to bounds."""
        params = [ParameterSpec('x', ParameterBounds(0.0, 1.0), initial=0.5)]

        def objective(p):
            # This would push x negative without clipping
            return -p['x']

        config = OptimizationConfig(max_iterations=50, learning_rate=0.5)
        optimizer = GradientOptimizer(params, SimpleObjective(objective), config)

        result = optimizer.optimize()

        # x should be clipped to upper bound
        assert 0.0 <= result.best_params['x'] <= 1.0
