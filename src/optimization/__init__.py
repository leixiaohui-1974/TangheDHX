"""
Automated Optimization Module.

Provides:
- Multiple optimization algorithms (gradient-based, evolutionary, Bayesian)
- Controller parameter optimization (PID gains, MPC weights)
- Model calibration parameter optimization
- Optimization history tracking and analysis
"""

from src.optimization.optimizer import (
    OptimizationAlgorithm,
    OptimizationStatus,
    OptimizationObjective,
    ParameterSpec,
    ParameterBounds,
    OptimizationConfig,
    OptimizationResult,
    OptimizationHistory,
    ObjectiveFunction,
    GradientOptimizer,
    EvolutionaryOptimizer,
    BayesianOptimizer,
    GridSearchOptimizer,
    ControllerOptimizer,
    CalibrationOptimizer,
    AutoTuner
)

__all__ = [
    'OptimizationAlgorithm',
    'OptimizationStatus',
    'OptimizationObjective',
    'ParameterSpec',
    'ParameterBounds',
    'OptimizationConfig',
    'OptimizationResult',
    'OptimizationHistory',
    'ObjectiveFunction',
    'GradientOptimizer',
    'EvolutionaryOptimizer',
    'BayesianOptimizer',
    'GridSearchOptimizer',
    'ControllerOptimizer',
    'CalibrationOptimizer',
    'AutoTuner'
]
