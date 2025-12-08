"""
Simulation Scenario Generator Module.

Provides:
- Scenario templates for common situations
- Parameter randomization and variation
- Scenario sequencing and scheduling
- Automated test scenario generation
"""

from src.scenarios.generator import (
    ScenarioTemplate,
    ScenarioCategory,
    ScenarioSeverity,
    ParameterRange,
    ParameterDistribution,
    ScenarioEvent,
    ScenarioTimeline,
    ScenarioGenerator,
    ScenarioLibrary,
    ScenarioExecutor,
    ScenarioResult,
    AutomatedTestRunner
)

__all__ = [
    'ScenarioTemplate',
    'ScenarioCategory',
    'ScenarioSeverity',
    'ParameterRange',
    'ParameterDistribution',
    'ScenarioEvent',
    'ScenarioTimeline',
    'ScenarioGenerator',
    'ScenarioLibrary',
    'ScenarioExecutor',
    'ScenarioResult',
    'AutomatedTestRunner'
]
