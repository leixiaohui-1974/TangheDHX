# -*- coding: utf-8 -*-
"""
Scenario Manager.

This module orchestrates simulation scenarios for testing and
demonstration purposes.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ScenarioManager:
    """
    Orchestrates simulation scenarios.

    Manages scenario state, initial conditions, and time-based events
    such as fault injection.

    Attributes:
        model: Reference to the physics model
        active_scenario: Currently active scenario ID
        scenario_start_time: Time when scenario was started
    """

    # Scenario definitions
    SCENARIOS = {
        'S1.1': 'Normal Operation',
        'S2.1': 'Resonance Zone Crossing',
        'S4.1': 'Trash Rack Blockage',
        'S4.2': 'Gate Stuck',
    }

    def __init__(self, model: 'TangheSiphonModel') -> None:
        """
        Initialize the scenario manager.

        Args:
            model: Reference to the physics model
        """
        self.model = model
        self.active_scenario: Optional[str] = None
        self.scenario_start_time: float = 0.0

        logger.debug("ScenarioManager initialized")

    def set_scenario(self, scenario_id: str) -> None:
        """
        Set and initialize a scenario.

        Args:
            scenario_id: Scenario identifier (e.g., 'S1.1', 'S4.2')

        Raises:
            ValueError: If scenario_id is unknown
        """
        if scenario_id not in self.SCENARIOS:
            valid = ', '.join(self.SCENARIOS.keys())
            raise ValueError(
                f"Unknown scenario '{scenario_id}'. Valid scenarios: {valid}"
            )

        self.active_scenario = scenario_id
        self.scenario_start_time = self.model.time

        logger.info(
            "Scenario %s (%s) started at t=%.2fs",
            scenario_id,
            self.SCENARIOS[scenario_id],
            self.model.time
        )

        # Reset faults
        self.model.gate_stuck = [False] * self.model.num_gates

        # Apply scenario initial conditions
        self._apply_initial_conditions(scenario_id)

    def _apply_initial_conditions(self, scenario_id: str) -> None:
        """Apply initial conditions for the specified scenario."""
        if scenario_id == 'S1.1':
            # Normal Operation
            self.model.head_upstream = 10.0

        elif scenario_id == 'S2.1':
            # Resonance Zone Crossing
            self.model.head_upstream = 10.0

        elif scenario_id == 'S4.1':
            # Trash Rack Blockage (reduced head)
            self.model.head_upstream = 8.5
            logger.info("S4.1: Head reduced to 8.5m (blockage)")

        elif scenario_id == 'S4.2':
            # Gate Stuck (fault injected after delay in update())
            pass

    def update(self) -> None:
        """
        Update scenario state based on elapsed time.

        This method handles time-based events such as fault injection.
        """
        if self.active_scenario is None:
            return

        elapsed = self.model.time - self.scenario_start_time

        if self.active_scenario == 'S4.2':
            # At T+10s, Gate 2 gets stuck
            if elapsed > 10.0 and not self.model.gate_stuck[1]:
                logger.warning("S4.2: Injecting fault - Gate 2 stuck at t=%.2fs", self.model.time)
                self.model.inject_fault(1, 'stuck')

    def get_elapsed_time(self) -> float:
        """Get elapsed time since scenario started."""
        return self.model.time - self.scenario_start_time

    def reset(self) -> None:
        """Reset scenario manager state."""
        self.active_scenario = None
        self.scenario_start_time = 0.0
        logger.debug("ScenarioManager reset")
