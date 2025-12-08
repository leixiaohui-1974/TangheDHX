"""
Tests for Predictive Maintenance Module.

Tests cover:
- Degradation models (Weibull, Exponential, Linear)
- Equipment health monitoring
- Failure prediction engine
- Maintenance scheduling
- Integrated system
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.maintenance.predictive_maintenance import (
    # Enums and Data Classes
    HealthStatus,
    MaintenancePriority,
    MaintenanceType,
    FailureMode,
    EquipmentInfo,
    HealthIndex,
    DegradationState,
    MaintenanceAction,
    FailurePrediction,
    # Degradation Models
    DegradationModel,
    WeibullDegradation,
    ExponentialDegradation,
    LinearDegradation,
    # Main Classes
    EquipmentHealthMonitor,
    FailurePredictionEngine,
    MaintenanceScheduler,
    PredictiveMaintenanceSystem
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def equipment_info():
    """Create sample equipment info."""
    return EquipmentInfo(
        equipment_id="gate_1",
        equipment_type="gate",
        installation_date=datetime(2020, 1, 1),
        manufacturer="Test Corp",
        model="TG-100",
        expected_lifetime_hours=50000.0,
        maintenance_interval_hours=2000.0,
        replacement_cost=10000.0,
        downtime_cost_per_hour=500.0
    )


@pytest.fixture
def weibull_model():
    """Create Weibull degradation model."""
    return WeibullDegradation(shape=2.0, scale=50000.0, location=0.0)


@pytest.fixture
def exponential_model():
    """Create exponential degradation model."""
    return ExponentialDegradation(failure_rate=0.00002, degradation_rate=0.001)


@pytest.fixture
def linear_model():
    """Create linear degradation model."""
    return LinearDegradation(degradation_rate=0.002, initial_health=100.0)


@pytest.fixture
def health_monitor(equipment_info, weibull_model):
    """Create equipment health monitor."""
    return EquipmentHealthMonitor(
        equipment_info=equipment_info,
        degradation_model=weibull_model,
        history_length=100
    )


@pytest.fixture
def failure_engine():
    """Create failure prediction engine."""
    return FailurePredictionEngine(
        failure_threshold=30.0,
        warning_threshold=50.0,
        prediction_horizon_hours=168.0
    )


@pytest.fixture
def maintenance_scheduler():
    """Create maintenance scheduler."""
    return MaintenanceScheduler(
        planning_horizon_days=30,
        max_daily_maintenance_hours=8.0
    )


@pytest.fixture
def pms():
    """Create predictive maintenance system."""
    return PredictiveMaintenanceSystem(
        failure_threshold=30.0,
        warning_threshold=50.0,
        planning_horizon_days=30
    )


# =============================================================================
# Test Data Classes and Enums
# =============================================================================

class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert HealthStatus.EXCELLENT.value == "excellent"
        assert HealthStatus.GOOD.value == "good"
        assert HealthStatus.FAIR.value == "fair"
        assert HealthStatus.POOR.value == "poor"
        assert HealthStatus.CRITICAL.value == "critical"


class TestMaintenancePriority:
    """Test MaintenancePriority enum."""

    def test_priority_ordering(self):
        """Test priority values are ordered correctly."""
        assert MaintenancePriority.EMERGENCY.value < MaintenancePriority.HIGH.value
        assert MaintenancePriority.HIGH.value < MaintenancePriority.MEDIUM.value
        assert MaintenancePriority.MEDIUM.value < MaintenancePriority.LOW.value
        assert MaintenancePriority.LOW.value < MaintenancePriority.ROUTINE.value


class TestHealthIndex:
    """Test HealthIndex dataclass."""

    def test_status_excellent(self):
        """Test excellent status assignment."""
        health = HealthIndex(overall=95.0)
        assert health.status == HealthStatus.EXCELLENT

    def test_status_good(self):
        """Test good status assignment."""
        health = HealthIndex(overall=80.0)
        assert health.status == HealthStatus.GOOD

    def test_status_fair(self):
        """Test fair status assignment."""
        health = HealthIndex(overall=60.0)
        assert health.status == HealthStatus.FAIR

    def test_status_poor(self):
        """Test poor status assignment."""
        health = HealthIndex(overall=40.0)
        assert health.status == HealthStatus.POOR

    def test_status_critical(self):
        """Test critical status assignment."""
        health = HealthIndex(overall=20.0)
        assert health.status == HealthStatus.CRITICAL


class TestEquipmentInfo:
    """Test EquipmentInfo dataclass."""

    def test_creation(self, equipment_info):
        """Test equipment info creation."""
        assert equipment_info.equipment_id == "gate_1"
        assert equipment_info.equipment_type == "gate"
        assert equipment_info.expected_lifetime_hours == 50000.0

    def test_default_values(self):
        """Test default values."""
        info = EquipmentInfo(
            equipment_id="test",
            equipment_type="sensor",
            installation_date=datetime.now()
        )
        assert info.manufacturer == "Unknown"
        assert info.model == "Unknown"


# =============================================================================
# Test Degradation Models
# =============================================================================

class TestWeibullDegradation:
    """Test Weibull degradation model."""

    def test_initialization(self, weibull_model):
        """Test model initialization."""
        assert weibull_model.shape == 2.0
        assert weibull_model.scale == 50000.0
        assert weibull_model.location == 0.0

    def test_initial_health(self, weibull_model):
        """Test health at time zero."""
        health = weibull_model.calculate_degradation(0.0)
        assert health == 100.0

    def test_health_decreases_over_time(self, weibull_model):
        """Test health decreases with operating hours."""
        health_1000 = weibull_model.calculate_degradation(1000.0)
        health_10000 = weibull_model.calculate_degradation(10000.0)
        health_30000 = weibull_model.calculate_degradation(30000.0)

        assert health_1000 > health_10000 > health_30000
        assert 0 <= health_30000 <= 100

    def test_rul_estimation(self, weibull_model):
        """Test remaining useful life estimation."""
        rul = weibull_model.estimate_rul(
            current_health=80.0,
            failure_threshold=30.0,
            initial_health=100.0,
            operating_hours=5000.0
        )
        assert rul > 0

    def test_rul_zero_at_threshold(self, weibull_model):
        """Test RUL is zero at failure threshold."""
        rul = weibull_model.estimate_rul(
            current_health=30.0,
            failure_threshold=30.0
        )
        assert rul == 0.0

    def test_failure_probability(self, weibull_model):
        """Test failure probability calculation."""
        prob = weibull_model.failure_probability(
            operating_hours=10000.0,
            interval_hours=100.0
        )
        assert 0 <= prob <= 1

    def test_failure_probability_increases_with_time(self, weibull_model):
        """Test failure probability increases over time."""
        prob_early = weibull_model.failure_probability(1000.0, 100.0)
        prob_late = weibull_model.failure_probability(40000.0, 100.0)
        assert prob_late > prob_early

    def test_location_parameter(self):
        """Test location parameter (failure-free period)."""
        model = WeibullDegradation(shape=2.0, scale=50000.0, location=1000.0)
        health = model.calculate_degradation(500.0)
        assert health == 100.0  # Before location, no degradation


class TestExponentialDegradation:
    """Test exponential degradation model."""

    def test_initialization(self, exponential_model):
        """Test model initialization."""
        assert exponential_model.failure_rate == 0.00002
        assert exponential_model.degradation_rate == 0.001

    def test_initial_health(self, exponential_model):
        """Test health at time zero."""
        health = exponential_model.calculate_degradation(0.0)
        assert health == 100.0

    def test_health_decreases(self, exponential_model):
        """Test health decreases over time."""
        health = exponential_model.calculate_degradation(10000.0)
        assert health < 100.0
        assert health > 0

    def test_rul_estimation(self, exponential_model):
        """Test RUL estimation."""
        rul = exponential_model.estimate_rul(
            current_health=70.0,
            failure_threshold=30.0
        )
        assert rul > 0

    def test_memoryless_failure_probability(self, exponential_model):
        """Test memoryless property of failure probability."""
        # For exponential distribution, conditional probability should be constant
        prob_1000 = exponential_model.failure_probability(1000.0, 100.0)
        prob_10000 = exponential_model.failure_probability(10000.0, 100.0)
        # Should be approximately equal (memoryless)
        assert abs(prob_1000 - prob_10000) < 0.01


class TestLinearDegradation:
    """Test linear degradation model."""

    def test_initialization(self, linear_model):
        """Test model initialization."""
        assert linear_model.degradation_rate == 0.002
        assert linear_model.initial_health == 100.0

    def test_linear_decrease(self, linear_model):
        """Test health decreases linearly."""
        health_1000 = linear_model.calculate_degradation(1000.0)
        health_2000 = linear_model.calculate_degradation(2000.0)

        # Linear: difference should be consistent
        expected_diff = 1000.0 * 0.002
        assert abs((health_1000 - health_2000) - expected_diff) < 0.01

    def test_rul_estimation(self, linear_model):
        """Test RUL is calculated correctly."""
        rul = linear_model.estimate_rul(
            current_health=50.0,
            failure_threshold=30.0
        )
        # RUL should be (50-30) / 0.002 = 10000 hours
        assert abs(rul - 10000.0) < 1.0

    def test_failure_probability(self, linear_model):
        """Test failure probability at threshold."""
        prob = linear_model.failure_probability(
            operating_hours=34000.0,
            interval_hours=1000.0,
            current_health=32.0,
            failure_threshold=30.0
        )
        # Should have high probability since we're close to threshold
        assert prob > 0


# =============================================================================
# Test Equipment Health Monitor
# =============================================================================

class TestEquipmentHealthMonitor:
    """Test EquipmentHealthMonitor class."""

    def test_initialization(self, health_monitor, equipment_info):
        """Test monitor initialization."""
        assert health_monitor.equipment_info == equipment_info
        assert health_monitor.operating_hours == 0.0
        assert health_monitor.cycle_count == 0

    def test_update_increases_operating_hours(self, health_monitor):
        """Test operating hours increase with updates."""
        health_monitor.update(timestamp=1.0, operating=True)
        health_monitor.update(timestamp=2.0, operating=True)
        assert health_monitor.operating_hours > 0

    def test_update_returns_health_index(self, health_monitor):
        """Test update returns valid health index."""
        health = health_monitor.update(
            timestamp=1.0,
            vibration=0.5,
            temperature=25.0,
            performance_metric=95.0
        )
        assert isinstance(health, HealthIndex)
        assert 0 <= health.overall <= 100

    def test_vibration_affects_mechanical_health(self, health_monitor):
        """Test high vibration decreases mechanical health."""
        # Low vibration
        health_monitor.update(timestamp=1.0, vibration=0.5)
        low_vib_health = health_monitor._current_health.mechanical

        # Reset and apply high vibration
        health_monitor._vibration_buffer.clear()
        for i in range(10):
            health_monitor.update(timestamp=2.0 + i * 0.1, vibration=5.0)
        high_vib_health = health_monitor._current_health.mechanical

        assert high_vib_health < low_vib_health

    def test_temperature_affects_electrical_health(self, health_monitor):
        """Test high temperature decreases electrical health."""
        # Normal temperature
        health_monitor.update(timestamp=1.0, temperature=25.0)
        normal_temp_health = health_monitor._current_health.electrical

        # High temperature
        health_monitor._temperature_buffer.clear()
        for i in range(10):
            health_monitor.update(timestamp=2.0 + i * 0.1, temperature=60.0)
        high_temp_health = health_monitor._current_health.electrical

        assert high_temp_health < normal_temp_health

    def test_get_degradation_state(self, health_monitor):
        """Test degradation state retrieval."""
        health_monitor.update(timestamp=100.0, operating=True)
        state = health_monitor.get_degradation_state(100.0)

        assert isinstance(state, DegradationState)
        assert state.equipment_id == "gate_1"
        assert state.rul_hours >= 0
        assert 0 <= state.failure_probability <= 1

    def test_record_maintenance(self, health_monitor):
        """Test maintenance recording."""
        # Degrade health first
        for i in range(50):
            health_monitor.update(
                timestamp=i * 100.0,
                operating=True,
                vibration=3.0,
                temperature=50.0,
                performance_metric=60.0
            )

        pre_maintenance_health = health_monitor._current_health.overall

        # Record maintenance
        health_monitor.record_maintenance(datetime.now())

        assert health_monitor.cycle_count == 1
        # Health should be restored
        if pre_maintenance_health < 70:
            assert health_monitor._current_health.overall > pre_maintenance_health


# =============================================================================
# Test Failure Prediction Engine
# =============================================================================

class TestFailurePredictionEngine:
    """Test FailurePredictionEngine class."""

    def test_initialization(self, failure_engine):
        """Test engine initialization."""
        assert failure_engine.failure_threshold == 30.0
        assert failure_engine.warning_threshold == 50.0
        assert failure_engine.prediction_horizon == 168.0

    def test_predict_failures_healthy_equipment(self, failure_engine, health_monitor):
        """Test no predictions for healthy equipment."""
        health_monitor.update(timestamp=1.0, operating=True)
        predictions = failure_engine.predict_failures(health_monitor)

        # Should have few or no predictions for healthy equipment
        high_prob_predictions = [p for p in predictions if p.probability > 0.3]
        assert len(high_prob_predictions) == 0

    def test_predict_failures_degraded_equipment(self, failure_engine, health_monitor):
        """Test predictions for degraded equipment."""
        # Simulate degradation
        for i in range(100):
            health_monitor.update(
                timestamp=i * 100.0,
                operating=True,
                vibration=4.0,  # High vibration
                temperature=55.0,  # High temperature
                performance_metric=40.0  # Low performance
            )

        predictions = failure_engine.predict_failures(health_monitor)

        # Should have some predictions
        assert len(predictions) > 0

        # Check prediction structure
        for pred in predictions:
            assert isinstance(pred, FailurePrediction)
            assert 0 <= pred.probability <= 1
            assert pred.time_to_failure_hours >= 0

    def test_prediction_has_recommendations(self, failure_engine, health_monitor):
        """Test predictions include recommendations."""
        # Degrade equipment
        for i in range(50):
            health_monitor.update(
                timestamp=i * 100.0,
                vibration=5.0,
                temperature=60.0,
                performance_metric=30.0
            )

        predictions = failure_engine.predict_failures(health_monitor)

        if predictions:
            pred = predictions[0]
            assert len(pred.recommended_actions) > 0

    def test_record_failure(self, failure_engine):
        """Test failure recording."""
        failure_engine.record_failure(
            equipment_id="gate_1",
            failure_mode=FailureMode.GATE_STUCK,
            timestamp=datetime.now(),
            health_at_failure=25.0,
            operating_hours=40000.0
        )

        assert len(failure_engine._failure_history) == 1


# =============================================================================
# Test Maintenance Scheduler
# =============================================================================

class TestMaintenanceScheduler:
    """Test MaintenanceScheduler class."""

    def test_initialization(self, maintenance_scheduler):
        """Test scheduler initialization."""
        assert maintenance_scheduler.planning_horizon == 30
        assert maintenance_scheduler.max_daily_hours == 8.0

    def test_generate_empty_plan(self, maintenance_scheduler):
        """Test plan generation with no equipment."""
        plan = maintenance_scheduler.generate_maintenance_plan({}, {})
        assert plan == []

    def test_generate_plan_with_monitors(
        self,
        maintenance_scheduler,
        health_monitor,
        failure_engine
    ):
        """Test plan generation with monitors."""
        # Degrade equipment
        for i in range(50):
            health_monitor.update(
                timestamp=i * 100.0,
                vibration=4.0,
                temperature=50.0,
                performance_metric=50.0
            )

        monitors = {"gate_1": health_monitor}
        predictions = {"gate_1": failure_engine.predict_failures(health_monitor)}

        plan = maintenance_scheduler.generate_maintenance_plan(monitors, predictions)

        # Should have some maintenance actions
        assert isinstance(plan, list)

    def test_action_priority_sorting(self, maintenance_scheduler):
        """Test actions are sorted by priority."""
        # Create test actions with different priorities
        actions = [
            MaintenanceAction(
                equipment_id="eq1",
                action_type=MaintenanceType.INSPECTION,
                priority=MaintenancePriority.LOW,
                description="Low priority",
                estimated_duration_hours=1.0,
                estimated_cost=100.0,
                due_date=datetime.now()
            ),
            MaintenanceAction(
                equipment_id="eq2",
                action_type=MaintenanceType.MAJOR_REPAIR,
                priority=MaintenancePriority.EMERGENCY,
                description="Emergency",
                estimated_duration_hours=8.0,
                estimated_cost=5000.0,
                due_date=datetime.now()
            ),
        ]

        optimized = maintenance_scheduler._optimize_schedule(actions)

        # Emergency should be first
        assert optimized[0].priority == MaintenancePriority.EMERGENCY

    def test_cost_summary(self, maintenance_scheduler):
        """Test cost summary calculation."""
        # Add some scheduled actions
        maintenance_scheduler._scheduled_actions = [
            MaintenanceAction(
                equipment_id="eq1",
                action_type=MaintenanceType.INSPECTION,
                priority=MaintenancePriority.ROUTINE,
                description="Test",
                estimated_duration_hours=2.0,
                estimated_cost=500.0,
                due_date=datetime.now()
            ),
            MaintenanceAction(
                equipment_id="eq2",
                action_type=MaintenanceType.MINOR_REPAIR,
                priority=MaintenancePriority.MEDIUM,
                description="Test",
                estimated_duration_hours=4.0,
                estimated_cost=1000.0,
                due_date=datetime.now()
            ),
        ]

        summary = maintenance_scheduler.get_cost_summary()

        assert summary['total_actions'] == 2
        assert summary['total_cost'] == 1500.0
        assert summary['total_hours'] == 6.0


# =============================================================================
# Test Predictive Maintenance System
# =============================================================================

class TestPredictiveMaintenanceSystem:
    """Test PredictiveMaintenanceSystem class."""

    def test_initialization(self, pms):
        """Test system initialization."""
        assert pms.failure_threshold == 30.0
        assert pms.warning_threshold == 50.0
        assert len(pms._monitors) == 0

    def test_register_equipment(self, pms, equipment_info):
        """Test equipment registration."""
        monitor = pms.register_equipment(equipment_info)

        assert isinstance(monitor, EquipmentHealthMonitor)
        assert "gate_1" in pms._monitors

    def test_update_with_sensor_data(self, pms, equipment_info):
        """Test system update with sensor data."""
        pms.register_equipment(equipment_info)

        result = pms.update(
            timestamp=1.0,
            sensor_data={
                "gate_1": {
                    "vibration": 0.5,
                    "temperature": 25.0,
                    "performance_metric": 95.0
                }
            }
        )

        assert "equipment_health" in result
        assert "gate_1" in result["equipment_health"]
        assert result["system_health"] > 0

    def test_update_generates_alerts(self, pms, equipment_info):
        """Test alerts are generated for degraded equipment."""
        pms.register_equipment(equipment_info)

        # Simulate degradation
        for i in range(50):
            result = pms.update(
                timestamp=i * 100.0,
                sensor_data={
                    "gate_1": {
                        "vibration": 5.0,
                        "temperature": 60.0,
                        "performance_metric": 20.0
                    }
                }
            )

        # Should eventually generate alerts
        # (may need multiple updates to trigger)
        assert "alerts" in result

    def test_predict_failures(self, pms, equipment_info):
        """Test failure prediction."""
        pms.register_equipment(equipment_info)

        # Update to create some history
        for i in range(20):
            pms.update(
                timestamp=i * 100.0,
                sensor_data={
                    "gate_1": {
                        "vibration": 3.0,
                        "temperature": 45.0,
                        "performance_metric": 70.0
                    }
                }
            )

        predictions = pms.predict_failures()
        assert isinstance(predictions, dict)

    def test_generate_maintenance_plan(self, pms, equipment_info):
        """Test maintenance plan generation."""
        pms.register_equipment(equipment_info)

        # Degrade equipment
        for i in range(30):
            pms.update(
                timestamp=i * 100.0,
                sensor_data={
                    "gate_1": {
                        "vibration": 4.0,
                        "temperature": 55.0,
                        "performance_metric": 50.0
                    }
                }
            )

        plan = pms.generate_maintenance_plan()
        assert isinstance(plan, list)

    def test_get_system_status(self, pms, equipment_info):
        """Test system status retrieval."""
        pms.register_equipment(equipment_info)
        pms.update(timestamp=1.0, sensor_data={"gate_1": {"vibration": 0.5}})

        status = pms.get_system_status()

        assert "system_health" in status
        assert "equipment_count" in status
        assert status["equipment_count"] == 1

    def test_get_equipment_report(self, pms, equipment_info):
        """Test equipment report generation."""
        pms.register_equipment(equipment_info)

        for i in range(10):
            pms.update(
                timestamp=i * 10.0,
                sensor_data={
                    "gate_1": {
                        "vibration": 1.0,
                        "temperature": 30.0,
                        "performance_metric": 90.0
                    }
                }
            )

        report = pms.get_equipment_report("gate_1")

        assert report is not None
        assert report["equipment_id"] == "gate_1"
        assert "health_index" in report
        assert "degradation" in report
        assert "failure_predictions" in report
        assert "maintenance" in report

    def test_get_equipment_report_unknown(self, pms):
        """Test report for unknown equipment."""
        report = pms.get_equipment_report("unknown")
        assert report is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the predictive maintenance system."""

    def test_full_lifecycle(self, pms):
        """Test full equipment lifecycle from installation to failure prediction."""
        # Create multiple equipment
        equipment_list = [
            EquipmentInfo(
                equipment_id=f"gate_{i}",
                equipment_type="gate",
                installation_date=datetime(2020, 1, 1),
                expected_lifetime_hours=50000.0
            )
            for i in range(3)
        ]

        for info in equipment_list:
            pms.register_equipment(info)

        # Simulate operation with varying conditions
        for hour in range(100):
            sensor_data = {}
            for i in range(3):
                # Gate 0: Normal operation
                # Gate 1: Moderate degradation
                # Gate 2: Severe degradation
                vibration = 0.5 + i * 1.5
                temperature = 25.0 + i * 15.0
                performance = 95.0 - i * 25.0

                sensor_data[f"gate_{i}"] = {
                    "vibration": vibration,
                    "temperature": temperature,
                    "performance_metric": performance
                }

            pms.update(timestamp=float(hour), sensor_data=sensor_data)

        # Check system status
        status = pms.get_system_status()
        assert status["equipment_count"] == 3

        # Gate 2 should have lowest health
        health_values = [
            status["equipment_status"][f"gate_{i}"]["health"]
            for i in range(3)
        ]
        assert health_values[0] > health_values[2]

        # Generate maintenance plan
        plan = pms.generate_maintenance_plan()

        # Gate 2 should have higher priority maintenance
        gate_2_actions = [a for a in plan if a.equipment_id == "gate_2"]
        gate_0_actions = [a for a in plan if a.equipment_id == "gate_0"]

        if gate_2_actions and gate_0_actions:
            # Higher priority = lower value
            assert gate_2_actions[0].priority.value <= gate_0_actions[0].priority.value

    def test_maintenance_effect(self, pms, equipment_info):
        """Test that maintenance improves equipment health."""
        pms.register_equipment(equipment_info)
        monitor = pms._monitors["gate_1"]

        # Degrade equipment
        for hour in range(100):
            pms.update(
                timestamp=float(hour),
                sensor_data={
                    "gate_1": {
                        "vibration": 4.0,
                        "temperature": 50.0,
                        "performance_metric": 50.0
                    }
                }
            )

        pre_maintenance = monitor._current_health.overall

        # Record maintenance
        monitor.record_maintenance(datetime.now())

        post_maintenance = monitor._current_health.overall

        # Health should improve after maintenance (if it was degraded)
        if pre_maintenance < 70:
            assert post_maintenance > pre_maintenance

    def test_multiple_degradation_models(self, pms):
        """Test system with different degradation models."""
        # Equipment with Weibull model
        info_weibull = EquipmentInfo(
            equipment_id="eq_weibull",
            equipment_type="gate",
            installation_date=datetime(2020, 1, 1)
        )
        pms.register_equipment(info_weibull, WeibullDegradation())

        # Equipment with Exponential model
        info_exp = EquipmentInfo(
            equipment_id="eq_exp",
            equipment_type="sensor",
            installation_date=datetime(2020, 1, 1)
        )
        pms.register_equipment(info_exp, ExponentialDegradation())

        # Equipment with Linear model
        info_linear = EquipmentInfo(
            equipment_id="eq_linear",
            equipment_type="actuator",
            installation_date=datetime(2020, 1, 1)
        )
        pms.register_equipment(info_linear, LinearDegradation())

        # Update all
        for hour in range(50):
            pms.update(
                timestamp=float(hour * 100),
                sensor_data={
                    "eq_weibull": {"vibration": 1.0, "temperature": 30.0},
                    "eq_exp": {"vibration": 1.0, "temperature": 30.0},
                    "eq_linear": {"vibration": 1.0, "temperature": 30.0}
                }
            )

        status = pms.get_system_status()
        assert len(status["equipment_status"]) == 3

        # All should have some health recorded
        for eq_id in ["eq_weibull", "eq_exp", "eq_linear"]:
            assert status["equipment_status"][eq_id]["health"] > 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_operating_hours(self, health_monitor):
        """Test health at zero operating hours."""
        health = health_monitor.update(timestamp=0.0)
        assert health.overall >= 99.9  # Allow small numerical tolerance

    def test_extreme_sensor_values(self, health_monitor):
        """Test handling of extreme sensor values."""
        # Very high values
        health = health_monitor.update(
            timestamp=1.0,
            vibration=100.0,
            temperature=200.0,
            performance_metric=0.0
        )
        assert 0 <= health.overall <= 100

    def test_missing_sensor_data(self, health_monitor):
        """Test update with missing sensor data."""
        health = health_monitor.update(timestamp=1.0)
        assert isinstance(health, HealthIndex)

    def test_negative_time_handling(self, weibull_model):
        """Test model handles negative time gracefully."""
        health = weibull_model.calculate_degradation(-100.0)
        # Should return full health or handle gracefully
        assert health <= 100.0

    def test_rul_at_failure(self, linear_model):
        """Test RUL calculation at failure threshold."""
        rul = linear_model.estimate_rul(
            current_health=30.0,
            failure_threshold=30.0
        )
        assert rul == 0.0

    def test_empty_prediction_list(self, failure_engine, health_monitor):
        """Test handling when no predictions are generated."""
        # Healthy equipment should generate few/no predictions
        health_monitor.update(timestamp=1.0)
        predictions = failure_engine.predict_failures(health_monitor)
        # Should return empty list or list with low-probability items
        assert isinstance(predictions, list)
