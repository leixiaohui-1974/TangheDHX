"""
Automated Optimization Engine.

This module provides comprehensive optimization capabilities for:
- Controller parameter tuning (PID, MPC)
- Model calibration
- System performance optimization
"""

import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class OptimizationAlgorithm(Enum):
    """Available optimization algorithms."""
    GRADIENT_DESCENT = auto()
    NELDER_MEAD = auto()
    BFGS = auto()
    POWELL = auto()
    EVOLUTIONARY = auto()
    DIFFERENTIAL_EVOLUTION = auto()
    PARTICLE_SWARM = auto()
    BAYESIAN = auto()
    GRID_SEARCH = auto()
    RANDOM_SEARCH = auto()


class OptimizationStatus(Enum):
    """Optimization status."""
    IDLE = auto()
    RUNNING = auto()
    CONVERGED = auto()
    MAX_ITERATIONS = auto()
    TIMEOUT = auto()
    FAILED = auto()
    CANCELLED = auto()


class OptimizationObjective(Enum):
    """Optimization objective type."""
    MINIMIZE = auto()
    MAXIMIZE = auto()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ParameterBounds:
    """Parameter bounds for optimization."""
    lower: float
    upper: float

    def __post_init__(self):
        if self.lower >= self.upper:
            raise ValueError(f"Lower bound ({self.lower}) must be less than upper ({self.upper})")

    def clip(self, value: float) -> float:
        """Clip value to bounds."""
        return max(self.lower, min(self.upper, value))

    def contains(self, value: float) -> bool:
        """Check if value is within bounds."""
        return self.lower <= value <= self.upper

    def sample(self, rng: Optional[np.random.Generator] = None) -> float:
        """Sample uniformly from bounds."""
        rng = rng or np.random.default_rng()
        return rng.uniform(self.lower, self.upper)


@dataclass
class ParameterSpec:
    """Specification for an optimizable parameter."""
    name: str
    bounds: ParameterBounds
    initial: Optional[float] = None
    step_size: Optional[float] = None
    is_integer: bool = False
    description: str = ""

    def __post_init__(self):
        if self.initial is None:
            self.initial = (self.bounds.lower + self.bounds.upper) / 2
        if self.step_size is None:
            self.step_size = (self.bounds.upper - self.bounds.lower) / 100


@dataclass
class OptimizationConfig:
    """Configuration for optimization run."""
    max_iterations: int = 100
    tolerance: float = 1e-6
    timeout_seconds: float = 3600.0
    population_size: int = 20
    learning_rate: float = 0.01
    momentum: float = 0.9
    patience: int = 10
    n_jobs: int = 1
    seed: Optional[int] = None
    verbose: bool = False
    early_stopping: bool = True


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    result_id: str
    algorithm: OptimizationAlgorithm
    status: OptimizationStatus
    best_params: Dict[str, float]
    best_value: float
    iterations: int
    evaluations: int
    elapsed_seconds: float
    convergence_history: List[float] = field(default_factory=list)
    parameter_history: List[Dict[str, float]] = field(default_factory=list)
    final_message: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationHistory:
    """History of optimization runs."""
    runs: List[OptimizationResult] = field(default_factory=list)

    def add(self, result: OptimizationResult) -> None:
        """Add a result to history."""
        self.runs.append(result)

    def get_best(self, objective: OptimizationObjective = OptimizationObjective.MINIMIZE) -> Optional[OptimizationResult]:
        """Get best result from history."""
        if not self.runs:
            return None
        if objective == OptimizationObjective.MINIMIZE:
            return min(self.runs, key=lambda r: r.best_value)
        else:
            return max(self.runs, key=lambda r: r.best_value)

    def get_recent(self, n: int = 10) -> List[OptimizationResult]:
        """Get n most recent results."""
        return list(reversed(self.runs[-n:]))

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about optimization history."""
        if not self.runs:
            return {'count': 0}

        values = [r.best_value for r in self.runs]
        return {
            'count': len(self.runs),
            'best_value': min(values),
            'worst_value': max(values),
            'mean_value': np.mean(values),
            'std_value': np.std(values),
            'total_iterations': sum(r.iterations for r in self.runs),
            'total_evaluations': sum(r.evaluations for r in self.runs),
            'total_time_seconds': sum(r.elapsed_seconds for r in self.runs),
            'by_status': {
                status.name: sum(1 for r in self.runs if r.status == status)
                for status in OptimizationStatus
            }
        }


# =============================================================================
# Objective Function Interface
# =============================================================================

class ObjectiveFunction(ABC):
    """Abstract base class for objective functions."""

    @abstractmethod
    def evaluate(self, params: Dict[str, float]) -> float:
        """Evaluate the objective function."""
        pass

    def gradient(self, params: Dict[str, float], epsilon: float = 1e-6) -> Dict[str, float]:
        """Compute numerical gradient."""
        grad = {}
        base_value = self.evaluate(params)
        for name, value in params.items():
            params_plus = params.copy()
            params_plus[name] = value + epsilon
            grad[name] = (self.evaluate(params_plus) - base_value) / epsilon
        return grad


class SimpleObjective(ObjectiveFunction):
    """Simple objective function wrapper."""

    def __init__(self, func: Callable[[Dict[str, float]], float]):
        self._func = func

    def evaluate(self, params: Dict[str, float]) -> float:
        return self._func(params)


# =============================================================================
# Base Optimizer
# =============================================================================

class BaseOptimizer(ABC):
    """Abstract base class for optimizers."""

    def __init__(
        self,
        parameters: List[ParameterSpec],
        objective: ObjectiveFunction,
        config: Optional[OptimizationConfig] = None
    ):
        self._parameters = {p.name: p for p in parameters}
        self._param_names = [p.name for p in parameters]
        self._objective = objective
        self._config = config or OptimizationConfig()
        self._status = OptimizationStatus.IDLE
        self._best_params: Dict[str, float] = {}
        self._best_value: float = float('inf')
        self._iterations = 0
        self._evaluations = 0
        self._history: List[float] = []
        self._param_history: List[Dict[str, float]] = []
        self._cancelled = False
        self._rng = np.random.default_rng(self._config.seed)

    @property
    @abstractmethod
    def algorithm(self) -> OptimizationAlgorithm:
        """Get algorithm type."""
        pass

    @abstractmethod
    def _optimize(self) -> None:
        """Internal optimization implementation."""
        pass

    def optimize(self) -> OptimizationResult:
        """Run optimization."""
        self._status = OptimizationStatus.RUNNING
        self._cancelled = False
        self._iterations = 0
        self._evaluations = 0
        self._history = []
        self._param_history = []
        self._best_value = float('inf')
        self._best_params = {
            name: spec.initial for name, spec in self._parameters.items()
        }

        start_time = time.time()

        try:
            self._optimize()
        except Exception as e:
            self._status = OptimizationStatus.FAILED
            logger.error(f"Optimization failed: {e}")
            return self._create_result(start_time, str(e))

        elapsed = time.time() - start_time

        if self._cancelled:
            self._status = OptimizationStatus.CANCELLED
        elif elapsed >= self._config.timeout_seconds:
            self._status = OptimizationStatus.TIMEOUT
        elif self._iterations >= self._config.max_iterations:
            self._status = OptimizationStatus.MAX_ITERATIONS

        return self._create_result(start_time)

    def cancel(self) -> None:
        """Cancel optimization."""
        self._cancelled = True

    def _evaluate(self, params: Dict[str, float]) -> float:
        """Evaluate objective and track."""
        # Clip to bounds
        clipped = {
            name: self._parameters[name].bounds.clip(value)
            for name, value in params.items()
        }

        # Round integers
        for name, spec in self._parameters.items():
            if spec.is_integer:
                clipped[name] = round(clipped[name])

        value = self._objective.evaluate(clipped)
        self._evaluations += 1

        if value < self._best_value:
            self._best_value = value
            self._best_params = clipped.copy()

        return value

    def _create_result(self, start_time: float, message: str = "") -> OptimizationResult:
        """Create optimization result."""
        return OptimizationResult(
            result_id=str(uuid.uuid4())[:8],
            algorithm=self.algorithm,
            status=self._status,
            best_params=self._best_params,
            best_value=self._best_value,
            iterations=self._iterations,
            evaluations=self._evaluations,
            elapsed_seconds=time.time() - start_time,
            convergence_history=self._history.copy(),
            parameter_history=self._param_history.copy(),
            final_message=message
        )

    def _check_convergence(self, window: int = 10) -> bool:
        """Check if optimization has converged."""
        if len(self._history) < window:
            return False
        recent = self._history[-window:]
        return max(recent) - min(recent) < self._config.tolerance

    def _check_stopping(self, start_time: float) -> bool:
        """Check stopping conditions."""
        if self._cancelled:
            return True
        if self._iterations >= self._config.max_iterations:
            return True
        if time.time() - start_time >= self._config.timeout_seconds:
            return True
        if self._config.early_stopping and self._check_convergence():
            self._status = OptimizationStatus.CONVERGED
            return True
        return False


# =============================================================================
# Gradient-Based Optimizer
# =============================================================================

class GradientOptimizer(BaseOptimizer):
    """Gradient descent optimizer with momentum."""

    @property
    def algorithm(self) -> OptimizationAlgorithm:
        return OptimizationAlgorithm.GRADIENT_DESCENT

    def _optimize(self) -> None:
        """Run gradient descent."""
        start_time = time.time()

        # Initialize
        params = {name: spec.initial for name, spec in self._parameters.items()}
        velocity = {name: 0.0 for name in self._param_names}

        while not self._check_stopping(start_time):
            # Compute gradient
            grad = self._objective.gradient(params)

            # Update with momentum
            for name in self._param_names:
                velocity[name] = (
                    self._config.momentum * velocity[name] -
                    self._config.learning_rate * grad[name]
                )
                params[name] += velocity[name]
                params[name] = self._parameters[name].bounds.clip(params[name])

            # Evaluate
            value = self._evaluate(params)
            self._history.append(value)
            self._param_history.append(params.copy())
            self._iterations += 1

            if self._config.verbose and self._iterations % 10 == 0:
                logger.info(f"Iteration {self._iterations}: value={value:.6f}")


# =============================================================================
# Evolutionary Optimizer
# =============================================================================

class EvolutionaryOptimizer(BaseOptimizer):
    """Evolutionary/genetic algorithm optimizer."""

    @property
    def algorithm(self) -> OptimizationAlgorithm:
        return OptimizationAlgorithm.EVOLUTIONARY

    def _optimize(self) -> None:
        """Run evolutionary optimization."""
        start_time = time.time()
        pop_size = self._config.population_size

        # Initialize population
        population = []
        for _ in range(pop_size):
            individual = {
                name: spec.bounds.sample(self._rng)
                for name, spec in self._parameters.items()
            }
            population.append(individual)

        # Evaluate initial population
        fitness = [self._evaluate(ind) for ind in population]

        while not self._check_stopping(start_time):
            # Selection (tournament)
            selected = []
            for _ in range(pop_size):
                i, j = self._rng.choice(pop_size, 2, replace=False)
                if fitness[i] < fitness[j]:
                    selected.append(population[i].copy())
                else:
                    selected.append(population[j].copy())

            # Crossover
            children = []
            for i in range(0, pop_size - 1, 2):
                p1, p2 = selected[i], selected[i + 1]
                c1, c2 = self._crossover(p1, p2)
                children.extend([c1, c2])
            if len(children) < pop_size:
                children.append(selected[-1].copy())

            # Mutation
            for child in children:
                self._mutate(child)

            # Evaluate children
            child_fitness = [self._evaluate(c) for c in children]

            # Elitism: keep best from previous generation
            best_idx = np.argmin(fitness)
            worst_child_idx = np.argmax(child_fitness)
            if fitness[best_idx] < child_fitness[worst_child_idx]:
                children[worst_child_idx] = population[best_idx].copy()
                child_fitness[worst_child_idx] = fitness[best_idx]

            population = children
            fitness = child_fitness

            best_gen_value = min(fitness)
            self._history.append(best_gen_value)
            self._param_history.append(population[np.argmin(fitness)].copy())
            self._iterations += 1

            if self._config.verbose and self._iterations % 5 == 0:
                logger.info(f"Generation {self._iterations}: best={best_gen_value:.6f}")

    def _crossover(
        self,
        p1: Dict[str, float],
        p2: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Perform crossover."""
        c1, c2 = {}, {}
        for name in self._param_names:
            if self._rng.random() < 0.5:
                c1[name] = p1[name]
                c2[name] = p2[name]
            else:
                c1[name] = p2[name]
                c2[name] = p1[name]
        return c1, c2

    def _mutate(self, individual: Dict[str, float], rate: float = 0.1) -> None:
        """Mutate individual in place."""
        for name, spec in self._parameters.items():
            if self._rng.random() < rate:
                delta = self._rng.normal(0, spec.step_size * 3)
                individual[name] = spec.bounds.clip(individual[name] + delta)


# =============================================================================
# Bayesian Optimizer
# =============================================================================

class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization with Gaussian process surrogate."""

    @property
    def algorithm(self) -> OptimizationAlgorithm:
        return OptimizationAlgorithm.BAYESIAN

    def _optimize(self) -> None:
        """Run Bayesian optimization."""
        start_time = time.time()

        # Initial random samples
        n_initial = min(10, self._config.max_iterations // 2)
        X = []
        y = []

        for _ in range(n_initial):
            params = {
                name: spec.bounds.sample(self._rng)
                for name, spec in self._parameters.items()
            }
            value = self._evaluate(params)
            X.append([params[name] for name in self._param_names])
            y.append(value)
            self._history.append(value)
            self._param_history.append(params)
            self._iterations += 1

            if self._check_stopping(start_time):
                return

        X = np.array(X)
        y = np.array(y)

        # Bayesian optimization loop
        while not self._check_stopping(start_time):
            # Fit simple surrogate (polynomial regression as approximation)
            # In production, use proper GP implementation
            next_params = self._acquisition(X, y)

            # Evaluate
            value = self._evaluate(next_params)

            # Update data
            X = np.vstack([X, [next_params[name] for name in self._param_names]])
            y = np.append(y, value)

            self._history.append(value)
            self._param_history.append(next_params)
            self._iterations += 1

            if self._config.verbose and self._iterations % 5 == 0:
                logger.info(f"Iteration {self._iterations}: value={value:.6f}")

    def _acquisition(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Acquisition function to select next point.
        Uses Expected Improvement approximation.
        """
        best_y = np.min(y)

        # Generate candidates
        n_candidates = 1000
        candidates = []
        for _ in range(n_candidates):
            params = {
                name: spec.bounds.sample(self._rng)
                for name, spec in self._parameters.items()
            }
            candidates.append(params)

        # Estimate mean and variance using nearest neighbors
        best_score = -float('inf')
        best_candidate = candidates[0]

        for candidate in candidates:
            x = np.array([candidate[name] for name in self._param_names])

            # Distance to observed points
            dists = np.linalg.norm(X - x, axis=1)
            weights = np.exp(-dists)
            weights /= weights.sum() + 1e-10

            # Estimate mean and std
            mean = np.dot(weights, y)
            std = np.sqrt(np.dot(weights, (y - mean) ** 2)) + 1e-6

            # Expected improvement
            z = (best_y - mean) / std
            ei = std * (z * self._norm_cdf(z) + self._norm_pdf(z))

            if ei > best_score:
                best_score = ei
                best_candidate = candidate

        return best_candidate

    def _norm_cdf(self, x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + np.tanh(x * 0.7978845608))

    def _norm_pdf(self, x: float) -> float:
        """Standard normal PDF."""
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


# =============================================================================
# Grid Search Optimizer
# =============================================================================

class GridSearchOptimizer(BaseOptimizer):
    """Grid search optimizer."""

    def __init__(
        self,
        parameters: List[ParameterSpec],
        objective: ObjectiveFunction,
        config: Optional[OptimizationConfig] = None,
        grid_points: int = 10
    ):
        super().__init__(parameters, objective, config)
        self._grid_points = grid_points

    @property
    def algorithm(self) -> OptimizationAlgorithm:
        return OptimizationAlgorithm.GRID_SEARCH

    def _optimize(self) -> None:
        """Run grid search."""
        start_time = time.time()

        # Create grid
        grids = {}
        for name, spec in self._parameters.items():
            grids[name] = np.linspace(
                spec.bounds.lower,
                spec.bounds.upper,
                self._grid_points
            )

        # Generate all combinations
        from itertools import product
        param_names = list(grids.keys())
        grid_values = [grids[name] for name in param_names]

        for values in product(*grid_values):
            if self._check_stopping(start_time):
                break

            params = dict(zip(param_names, values))
            value = self._evaluate(params)
            self._history.append(value)
            self._param_history.append(params)
            self._iterations += 1

            if self._config.verbose and self._iterations % 100 == 0:
                logger.info(f"Evaluated {self._iterations} points, best={self._best_value:.6f}")


# =============================================================================
# Controller Optimizer
# =============================================================================

class ControllerOptimizer:
    """
    Optimizer for controller parameters.

    Supports PID and MPC controller tuning.
    """

    def __init__(
        self,
        controller: Any,
        simulator: Any,
        config: Optional[OptimizationConfig] = None
    ):
        """
        Initialize controller optimizer.

        Args:
            controller: Controller to optimize
            simulator: Simulation model for evaluation
            config: Optimization configuration
        """
        self._controller = controller
        self._simulator = simulator
        self._config = config or OptimizationConfig()
        self._history = OptimizationHistory()

        logger.info("ControllerOptimizer initialized")

    def optimize_pid(
        self,
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.BAYESIAN,
        kp_bounds: Tuple[float, float] = (0.1, 10.0),
        ki_bounds: Tuple[float, float] = (0.0, 5.0),
        kd_bounds: Tuple[float, float] = (0.0, 2.0),
        simulation_time: float = 100.0,
        target_flow: float = 150.0
    ) -> OptimizationResult:
        """
        Optimize PID gains.

        Args:
            algorithm: Optimization algorithm to use
            kp_bounds: Bounds for proportional gain
            ki_bounds: Bounds for integral gain
            kd_bounds: Bounds for derivative gain
            simulation_time: Time to simulate for evaluation
            target_flow: Target flow for evaluation

        Returns:
            Optimization result
        """
        parameters = [
            ParameterSpec('kp', ParameterBounds(*kp_bounds), description="Proportional gain"),
            ParameterSpec('ki', ParameterBounds(*ki_bounds), description="Integral gain"),
            ParameterSpec('kd', ParameterBounds(*kd_bounds), description="Derivative gain"),
        ]

        def objective(params: Dict[str, float]) -> float:
            return self._evaluate_pid(params, simulation_time, target_flow)

        optimizer = self._create_optimizer(algorithm, parameters, SimpleObjective(objective))
        result = optimizer.optimize()
        self._history.add(result)

        logger.info(
            f"PID optimization complete: Kp={result.best_params.get('kp', 0):.4f}, "
            f"Ki={result.best_params.get('ki', 0):.4f}, "
            f"Kd={result.best_params.get('kd', 0):.4f}, "
            f"cost={result.best_value:.6f}"
        )

        return result

    def _evaluate_pid(
        self,
        params: Dict[str, float],
        sim_time: float,
        target: float
    ) -> float:
        """Evaluate PID parameters."""
        # Save current gains
        original_gains = None
        if hasattr(self._controller, 'get_gains'):
            original_gains = self._controller.get_gains()

        try:
            # Set new gains
            if hasattr(self._controller, 'set_gains'):
                self._controller.set_gains(
                    params.get('kp', 1.0),
                    params.get('ki', 0.0),
                    params.get('kd', 0.0)
                )

            # Run simulation
            cost = self._run_simulation(sim_time, target)
            return cost

        finally:
            # Restore original gains
            if original_gains is not None and hasattr(self._controller, 'set_gains'):
                self._controller.set_gains(*original_gains)

    def optimize_mpc(
        self,
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.BAYESIAN,
        q_bounds: Tuple[float, float] = (0.1, 100.0),
        r_bounds: Tuple[float, float] = (0.01, 10.0),
        simulation_time: float = 100.0,
        target_flow: float = 150.0
    ) -> OptimizationResult:
        """
        Optimize MPC weights.

        Args:
            algorithm: Optimization algorithm to use
            q_bounds: Bounds for state weight
            r_bounds: Bounds for control weight
            simulation_time: Time to simulate for evaluation
            target_flow: Target flow for evaluation

        Returns:
            Optimization result
        """
        parameters = [
            ParameterSpec('q', ParameterBounds(*q_bounds), description="State weight"),
            ParameterSpec('r', ParameterBounds(*r_bounds), description="Control weight"),
        ]

        def objective(params: Dict[str, float]) -> float:
            return self._evaluate_mpc(params, simulation_time, target_flow)

        optimizer = self._create_optimizer(algorithm, parameters, SimpleObjective(objective))
        result = optimizer.optimize()
        self._history.add(result)

        logger.info(
            f"MPC optimization complete: Q={result.best_params.get('q', 0):.4f}, "
            f"R={result.best_params.get('r', 0):.4f}, "
            f"cost={result.best_value:.6f}"
        )

        return result

    def _evaluate_mpc(
        self,
        params: Dict[str, float],
        sim_time: float,
        target: float
    ) -> float:
        """Evaluate MPC parameters."""
        try:
            # Set new weights if possible
            if hasattr(self._controller, 'set_weights'):
                self._controller.set_weights(
                    params.get('q', 1.0),
                    params.get('r', 0.1)
                )

            # Run simulation
            cost = self._run_simulation(sim_time, target)
            return cost

        except Exception as e:
            logger.warning(f"MPC evaluation failed: {e}")
            return float('inf')

    def _run_simulation(self, sim_time: float, target: float) -> float:
        """Run simulation and compute cost."""
        dt = 0.1
        steps = int(sim_time / dt)

        errors = []
        control_efforts = []

        # Reset simulator state if possible
        if hasattr(self._simulator, 'reset'):
            self._simulator.reset()

        prev_control = 0.0

        for _ in range(steps):
            # Get current state
            state = self._simulator.get_state() if hasattr(self._simulator, 'get_state') else {}
            current_flow = state.get('total_flow', target)

            # Compute error
            error = target - current_flow
            errors.append(error ** 2)

            # Get control action
            if hasattr(self._controller, 'compute'):
                control = self._controller.compute(error, dt)
            elif hasattr(self._controller, 'step'):
                control = self._controller.step(state, target, dt)
            else:
                control = 0.0

            # Track control effort
            control_efforts.append((control - prev_control) ** 2)
            prev_control = control

            # Step simulator
            if hasattr(self._simulator, 'step'):
                self._simulator.step(dt)

        # Cost: weighted sum of tracking error and control effort
        ise = np.mean(errors)  # Integral squared error
        control_cost = np.mean(control_efforts) * 0.1

        return ise + control_cost

    def _create_optimizer(
        self,
        algorithm: OptimizationAlgorithm,
        parameters: List[ParameterSpec],
        objective: ObjectiveFunction
    ) -> BaseOptimizer:
        """Create optimizer based on algorithm type."""
        if algorithm == OptimizationAlgorithm.GRADIENT_DESCENT:
            return GradientOptimizer(parameters, objective, self._config)
        elif algorithm == OptimizationAlgorithm.EVOLUTIONARY:
            return EvolutionaryOptimizer(parameters, objective, self._config)
        elif algorithm == OptimizationAlgorithm.BAYESIAN:
            return BayesianOptimizer(parameters, objective, self._config)
        elif algorithm == OptimizationAlgorithm.GRID_SEARCH:
            return GridSearchOptimizer(parameters, objective, self._config)
        else:
            return BayesianOptimizer(parameters, objective, self._config)

    def get_history(self) -> OptimizationHistory:
        """Get optimization history."""
        return self._history


# =============================================================================
# Calibration Optimizer
# =============================================================================

class CalibrationOptimizer:
    """
    Optimizer for model calibration parameters.
    """

    def __init__(
        self,
        model: Any,
        reference_data: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[OptimizationConfig] = None
    ):
        """
        Initialize calibration optimizer.

        Args:
            model: Model to calibrate
            reference_data: Reference data for calibration
            config: Optimization configuration
        """
        self._model = model
        self._reference_data = reference_data or {}
        self._config = config or OptimizationConfig()
        self._history = OptimizationHistory()

        logger.info("CalibrationOptimizer initialized")

    def set_reference_data(self, data: Dict[str, np.ndarray]) -> None:
        """Set reference data for calibration."""
        self._reference_data = data

    def optimize(
        self,
        parameters: List[ParameterSpec],
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.BAYESIAN
    ) -> OptimizationResult:
        """
        Optimize calibration parameters.

        Args:
            parameters: Parameters to optimize
            algorithm: Optimization algorithm

        Returns:
            Optimization result
        """
        def objective(params: Dict[str, float]) -> float:
            return self._evaluate_calibration(params)

        optimizer = self._create_optimizer(algorithm, parameters, SimpleObjective(objective))
        result = optimizer.optimize()
        self._history.add(result)

        logger.info(f"Calibration optimization complete: cost={result.best_value:.6f}")
        return result

    def _evaluate_calibration(self, params: Dict[str, float]) -> float:
        """Evaluate calibration parameters against reference data."""
        if not self._reference_data:
            return 0.0

        # Apply parameters to model
        for name, value in params.items():
            if hasattr(self._model, name):
                setattr(self._model, name, value)
            elif hasattr(self._model, 'set_parameter'):
                self._model.set_parameter(name, value)

        # Run model and compare to reference
        total_error = 0.0
        count = 0

        for var_name, ref_values in self._reference_data.items():
            if hasattr(self._model, 'simulate'):
                model_values = self._model.simulate(len(ref_values))
            else:
                model_values = np.zeros_like(ref_values)

            # RMSE
            mse = np.mean((ref_values - model_values) ** 2)
            total_error += mse
            count += 1

        return total_error / max(count, 1)

    def _create_optimizer(
        self,
        algorithm: OptimizationAlgorithm,
        parameters: List[ParameterSpec],
        objective: ObjectiveFunction
    ) -> BaseOptimizer:
        """Create optimizer based on algorithm type."""
        if algorithm == OptimizationAlgorithm.EVOLUTIONARY:
            return EvolutionaryOptimizer(parameters, objective, self._config)
        elif algorithm == OptimizationAlgorithm.BAYESIAN:
            return BayesianOptimizer(parameters, objective, self._config)
        elif algorithm == OptimizationAlgorithm.GRID_SEARCH:
            return GridSearchOptimizer(parameters, objective, self._config)
        else:
            return BayesianOptimizer(parameters, objective, self._config)

    def get_history(self) -> OptimizationHistory:
        """Get optimization history."""
        return self._history


# =============================================================================
# Auto Tuner
# =============================================================================

class AutoTuner:
    """
    Automatic parameter tuning system.

    Provides unified interface for various optimization tasks.
    """

    def __init__(
        self,
        model: Any,
        controller: Optional[Any] = None,
        config: Optional[OptimizationConfig] = None
    ):
        """
        Initialize auto tuner.

        Args:
            model: Physics model
            controller: Optional controller to tune
            config: Optimization configuration
        """
        self._model = model
        self._controller = controller
        self._config = config or OptimizationConfig()
        self._history = OptimizationHistory()
        self._running = False
        self._current_task: Optional[str] = None
        self._lock = threading.Lock()

        # Sub-optimizers
        if controller is not None:
            self._controller_opt = ControllerOptimizer(controller, model, config)
        else:
            self._controller_opt = None

        self._calibration_opt = CalibrationOptimizer(model, config=config)

        logger.info("AutoTuner initialized")

    @property
    def is_running(self) -> bool:
        """Check if optimization is running."""
        return self._running

    @property
    def current_task(self) -> Optional[str]:
        """Get current optimization task."""
        return self._current_task

    def tune_pid(
        self,
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.BAYESIAN,
        **kwargs
    ) -> Optional[OptimizationResult]:
        """Tune PID controller parameters."""
        if self._controller_opt is None:
            logger.warning("No controller configured for tuning")
            return None

        with self._lock:
            if self._running:
                logger.warning("Optimization already running")
                return None
            self._running = True
            self._current_task = "pid_tuning"

        try:
            result = self._controller_opt.optimize_pid(algorithm, **kwargs)
            self._history.add(result)
            return result
        finally:
            with self._lock:
                self._running = False
                self._current_task = None

    def tune_mpc(
        self,
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.BAYESIAN,
        **kwargs
    ) -> Optional[OptimizationResult]:
        """Tune MPC controller parameters."""
        if self._controller_opt is None:
            logger.warning("No controller configured for tuning")
            return None

        with self._lock:
            if self._running:
                logger.warning("Optimization already running")
                return None
            self._running = True
            self._current_task = "mpc_tuning"

        try:
            result = self._controller_opt.optimize_mpc(algorithm, **kwargs)
            self._history.add(result)
            return result
        finally:
            with self._lock:
                self._running = False
                self._current_task = None

    def calibrate(
        self,
        parameters: List[ParameterSpec],
        reference_data: Dict[str, np.ndarray],
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.BAYESIAN
    ) -> Optional[OptimizationResult]:
        """Calibrate model parameters."""
        with self._lock:
            if self._running:
                logger.warning("Optimization already running")
                return None
            self._running = True
            self._current_task = "calibration"

        try:
            self._calibration_opt.set_reference_data(reference_data)
            result = self._calibration_opt.optimize(parameters, algorithm)
            self._history.add(result)
            return result
        finally:
            with self._lock:
                self._running = False
                self._current_task = None

    def custom_optimize(
        self,
        parameters: List[ParameterSpec],
        objective: Callable[[Dict[str, float]], float],
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.BAYESIAN,
        task_name: str = "custom"
    ) -> Optional[OptimizationResult]:
        """Run custom optimization."""
        with self._lock:
            if self._running:
                logger.warning("Optimization already running")
                return None
            self._running = True
            self._current_task = task_name

        try:
            if algorithm == OptimizationAlgorithm.EVOLUTIONARY:
                optimizer = EvolutionaryOptimizer(parameters, SimpleObjective(objective), self._config)
            elif algorithm == OptimizationAlgorithm.BAYESIAN:
                optimizer = BayesianOptimizer(parameters, SimpleObjective(objective), self._config)
            elif algorithm == OptimizationAlgorithm.GRADIENT_DESCENT:
                optimizer = GradientOptimizer(parameters, SimpleObjective(objective), self._config)
            else:
                optimizer = BayesianOptimizer(parameters, SimpleObjective(objective), self._config)

            result = optimizer.optimize()
            self._history.add(result)
            return result
        finally:
            with self._lock:
                self._running = False
                self._current_task = None

    def get_history(self) -> OptimizationHistory:
        """Get optimization history."""
        return self._history

    def get_status(self) -> Dict[str, Any]:
        """Get tuner status."""
        return {
            'running': self._running,
            'current_task': self._current_task,
            'history': self._history.get_statistics()
        }

    def get_best_result(self, task: Optional[str] = None) -> Optional[OptimizationResult]:
        """Get best result, optionally filtered by task."""
        runs = self._history.runs
        if task:
            # Filter by task name in final_message or algorithm
            runs = [r for r in runs if task.lower() in r.final_message.lower()]
        if not runs:
            return self._history.get_best()
        return min(runs, key=lambda r: r.best_value)
