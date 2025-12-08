# -*- coding: utf-8 -*-
"""
Data Governance Module for Digital Twin System.

This module provides comprehensive data governance including:
- Data quality assessment and scoring
- Data validation and cleaning
- Missing data handling
- Outlier detection and treatment
- Data lineage tracking
- Compliance and audit logging
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable, Union
from collections import deque
from enum import Enum
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class DataQualityDimension(Enum):
    """Data quality dimensions."""
    COMPLETENESS = "completeness"  # No missing values
    ACCURACY = "accuracy"          # Values are correct
    CONSISTENCY = "consistency"    # Values are consistent
    TIMELINESS = "timeliness"     # Data is current
    VALIDITY = "validity"         # Values meet defined rules
    UNIQUENESS = "uniqueness"     # No duplicates


class DataStatus(Enum):
    """Data processing status."""
    RAW = "raw"
    VALIDATED = "validated"
    CLEANED = "cleaned"
    REJECTED = "rejected"
    IMPUTED = "imputed"


class SeverityLevel(Enum):
    """Issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DataQualityScore:
    """Data quality assessment score."""
    overall: float = 1.0
    completeness: float = 1.0
    accuracy: float = 1.0
    consistency: float = 1.0
    timeliness: float = 1.0
    validity: float = 1.0
    uniqueness: float = 1.0
    issues: List[str] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class DataIssue:
    """Data quality issue record."""
    dimension: DataQualityDimension
    severity: SeverityLevel
    field_name: str
    description: str
    value: Any
    suggested_action: str
    timestamp: float = 0.0


@dataclass
class DataLineage:
    """Data lineage record."""
    source: str
    transformations: List[str] = field(default_factory=list)
    created_at: float = 0.0
    modified_at: float = 0.0
    version: int = 1
    quality_scores: List[DataQualityScore] = field(default_factory=list)


@dataclass
class ValidationRule:
    """Data validation rule definition."""
    name: str
    field: str
    condition: Callable[[Any], bool]
    severity: SeverityLevel = SeverityLevel.ERROR
    message: str = ""
    auto_fix: Optional[Callable[[Any], Any]] = None


class DataValidator:
    """
    Data validation engine.

    Validates data against defined rules and constraints.
    """

    def __init__(self):
        self._rules: Dict[str, List[ValidationRule]] = {}
        self._validation_history: List[Dict[str, Any]] = []

        # Initialize default rules
        self._init_default_rules()

        logger.debug("DataValidator initialized")

    def _init_default_rules(self) -> None:
        """Initialize default validation rules for common fields."""
        # Flow rate rules
        self.add_rule(ValidationRule(
            name="flow_rate_range",
            field="flow_rate",
            condition=lambda x: 0 <= x <= 200,
            severity=SeverityLevel.ERROR,
            message="Flow rate out of valid range [0, 200] m³/s",
            auto_fix=lambda x: np.clip(x, 0, 200)
        ))

        # Gate opening rules
        self.add_rule(ValidationRule(
            name="gate_opening_range",
            field="gate_opening",
            condition=lambda x: 0 <= x <= 5,
            severity=SeverityLevel.ERROR,
            message="Gate opening out of valid range [0, 5] m",
            auto_fix=lambda x: np.clip(x, 0, 5)
        ))

        # Water level rules
        self.add_rule(ValidationRule(
            name="water_level_range",
            field="water_level",
            condition=lambda x: 0 <= x <= 50,
            severity=SeverityLevel.ERROR,
            message="Water level out of valid range [0, 50] m",
            auto_fix=lambda x: np.clip(x, 0, 50)
        ))

        # Velocity rules
        self.add_rule(ValidationRule(
            name="velocity_range",
            field="velocity",
            condition=lambda x: 0 <= x <= 10,
            severity=SeverityLevel.ERROR,
            message="Velocity out of valid range [0, 10] m/s",
            auto_fix=lambda x: np.clip(x, 0, 10)
        ))

        # Vibration rules
        self.add_rule(ValidationRule(
            name="vibration_range",
            field="vibration",
            condition=lambda x: 0 <= x <= 5,
            severity=SeverityLevel.WARNING,
            message="Vibration out of expected range [0, 5] g"
        ))

        # Temperature rules
        self.add_rule(ValidationRule(
            name="temperature_range",
            field="temperature",
            condition=lambda x: -10 <= x <= 50,
            severity=SeverityLevel.WARNING,
            message="Temperature out of expected range [-10, 50] °C"
        ))

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        if rule.field not in self._rules:
            self._rules[rule.field] = []
        self._rules[rule.field].append(rule)
        logger.debug("Added validation rule: %s for field %s", rule.name, rule.field)

    def validate(
        self,
        data: Dict[str, Any],
        timestamp: float = 0.0
    ) -> Tuple[bool, List[DataIssue], Dict[str, Any]]:
        """
        Validate data against all applicable rules.

        Args:
            data: Data dictionary to validate
            timestamp: Timestamp for the validation

        Returns:
            Tuple of (is_valid, issues, corrected_data)
        """
        issues: List[DataIssue] = []
        corrected_data = data.copy()
        is_valid = True

        for field_name, value in data.items():
            if field_name not in self._rules:
                continue

            for rule in self._rules[field_name]:
                try:
                    if not rule.condition(value):
                        issue = DataIssue(
                            dimension=DataQualityDimension.VALIDITY,
                            severity=rule.severity,
                            field_name=field_name,
                            description=rule.message,
                            value=value,
                            suggested_action="Apply auto-fix" if rule.auto_fix else "Manual review",
                            timestamp=timestamp
                        )
                        issues.append(issue)

                        if rule.severity in [SeverityLevel.ERROR, SeverityLevel.CRITICAL]:
                            is_valid = False

                        # Apply auto-fix if available
                        if rule.auto_fix:
                            corrected_data[field_name] = rule.auto_fix(value)

                except Exception as e:
                    logger.warning("Rule %s failed on field %s: %s",
                                 rule.name, field_name, str(e))

        # Record validation
        self._validation_history.append({
            'timestamp': timestamp,
            'is_valid': is_valid,
            'issue_count': len(issues),
            'fields_validated': len(data)
        })

        return is_valid, issues, corrected_data

    def validate_batch(
        self,
        data_batch: List[Dict[str, Any]],
        timestamps: Optional[List[float]] = None
    ) -> Tuple[List[bool], List[List[DataIssue]], List[Dict[str, Any]]]:
        """Validate a batch of data records."""
        if timestamps is None:
            timestamps = [0.0] * len(data_batch)

        all_valid = []
        all_issues = []
        all_corrected = []

        for data, ts in zip(data_batch, timestamps):
            is_valid, issues, corrected = self.validate(data, ts)
            all_valid.append(is_valid)
            all_issues.append(issues)
            all_corrected.append(corrected)

        return all_valid, all_issues, all_corrected

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        if not self._validation_history:
            return {}

        total = len(self._validation_history)
        valid_count = sum(1 for v in self._validation_history if v['is_valid'])

        return {
            'total_validations': total,
            'valid_count': valid_count,
            'invalid_count': total - valid_count,
            'validation_rate': valid_count / total if total > 0 else 0.0,
            'total_issues': sum(v['issue_count'] for v in self._validation_history)
        }


class DataCleaner:
    """
    Data cleaning and preprocessing engine.

    Handles:
    - Missing value imputation
    - Outlier detection and treatment
    - Noise filtering
    - Data smoothing
    """

    def __init__(self, window_size: int = 10):
        self._window_size = window_size
        self._history: Dict[str, deque] = {}
        self._cleaning_log: List[Dict[str, Any]] = []

        logger.debug("DataCleaner initialized with window_size=%d", window_size)

    def clean(
        self,
        data: Dict[str, Any],
        timestamp: float = 0.0,
        methods: Optional[Dict[str, str]] = None
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Clean data using specified methods.

        Args:
            data: Data dictionary to clean
            timestamp: Data timestamp
            methods: Dict of field -> cleaning method

        Returns:
            Tuple of (cleaned_data, applied_methods)
        """
        if methods is None:
            methods = {}

        cleaned_data = data.copy()
        applied_methods = []

        for field_name, value in data.items():
            # Update history
            if field_name not in self._history:
                self._history[field_name] = deque(maxlen=self._window_size)

            method = methods.get(field_name, 'auto')

            # Handle missing/invalid values
            if value is None or (isinstance(value, float) and np.isnan(value)):
                cleaned_value = self._impute_missing(field_name, method)
                cleaned_data[field_name] = cleaned_value
                applied_methods.append(f"{field_name}: imputed")

            else:
                # Check for outliers
                is_outlier, cleaned_value = self._detect_and_handle_outlier(
                    field_name, value, method
                )

                if is_outlier:
                    cleaned_data[field_name] = cleaned_value
                    applied_methods.append(f"{field_name}: outlier_treated")

                # Apply smoothing if requested
                if method in ['smooth', 'filter']:
                    cleaned_value = self._apply_smoothing(field_name, cleaned_value)
                    cleaned_data[field_name] = cleaned_value
                    applied_methods.append(f"{field_name}: smoothed")

            # Update history with cleaned value
            self._history[field_name].append(cleaned_data[field_name])

        # Log cleaning operation
        self._cleaning_log.append({
            'timestamp': timestamp,
            'original_fields': len(data),
            'methods_applied': len(applied_methods)
        })

        return cleaned_data, applied_methods

    def _impute_missing(self, field_name: str, method: str) -> float:
        """Impute missing value."""
        history = list(self._history.get(field_name, []))

        if not history:
            return 0.0  # Default value

        if method in ['auto', 'mean']:
            return float(np.mean(history))
        elif method == 'median':
            return float(np.median(history))
        elif method == 'last':
            return history[-1]
        elif method == 'interpolate':
            # Linear interpolation using last two values
            if len(history) >= 2:
                return 2 * history[-1] - history[-2]
            return history[-1]
        else:
            return float(np.mean(history))

    def _detect_and_handle_outlier(
        self,
        field_name: str,
        value: float,
        method: str
    ) -> Tuple[bool, float]:
        """Detect and handle outlier."""
        history = list(self._history.get(field_name, []))

        if len(history) < 3:
            return False, value

        mean = np.mean(history)
        std = np.std(history)

        if std < 1e-10:
            return False, value

        z_score = abs(value - mean) / std

        # Outlier detection threshold
        threshold = 3.0

        if z_score > threshold:
            # Handle outlier based on method
            if method in ['auto', 'clip']:
                # Clip to 3 sigma range
                return True, np.clip(value, mean - 3 * std, mean + 3 * std)
            elif method == 'replace':
                return True, mean
            elif method == 'median':
                return True, float(np.median(history))
            else:
                return True, mean

        return False, value

    def _apply_smoothing(self, field_name: str, value: float) -> float:
        """Apply exponential smoothing."""
        history = list(self._history.get(field_name, []))

        if not history:
            return value

        alpha = 0.3  # Smoothing factor
        return alpha * value + (1 - alpha) * history[-1]

    def get_cleaning_stats(self) -> Dict[str, Any]:
        """Get cleaning statistics."""
        if not self._cleaning_log:
            return {}

        return {
            'total_operations': len(self._cleaning_log),
            'total_methods_applied': sum(c['methods_applied'] for c in self._cleaning_log),
            'fields_with_history': len(self._history)
        }


class DataQualityAssessor:
    """
    Comprehensive data quality assessment engine.

    Assesses data quality across multiple dimensions and
    provides actionable quality scores.
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self._thresholds = thresholds or {
            'completeness': 0.95,
            'accuracy': 0.90,
            'consistency': 0.95,
            'timeliness': 0.90,
            'validity': 0.95,
            'uniqueness': 0.99
        }

        self._assessment_history: List[DataQualityScore] = []
        self._reference_values: Dict[str, Any] = {}

        logger.debug("DataQualityAssessor initialized")

    def assess(
        self,
        data: Dict[str, Any],
        reference: Optional[Dict[str, Any]] = None,
        timestamp: float = 0.0,
        max_age: float = 60.0  # seconds
    ) -> DataQualityScore:
        """
        Assess data quality across all dimensions.

        Args:
            data: Data to assess
            reference: Reference/expected values for accuracy check
            timestamp: Data timestamp
            max_age: Maximum acceptable data age for timeliness

        Returns:
            DataQualityScore with all dimension scores
        """
        issues = []

        # Completeness: Check for missing values
        completeness = self._assess_completeness(data)
        if completeness < self._thresholds['completeness']:
            issues.append(f"Completeness below threshold: {completeness:.2%}")

        # Accuracy: Compare with reference if available
        accuracy = self._assess_accuracy(data, reference)
        if accuracy < self._thresholds['accuracy']:
            issues.append(f"Accuracy below threshold: {accuracy:.2%}")

        # Consistency: Check internal consistency
        consistency = self._assess_consistency(data)
        if consistency < self._thresholds['consistency']:
            issues.append(f"Consistency below threshold: {consistency:.2%}")

        # Timeliness: Check data age
        current_time = timestamp  # Assuming current simulation time
        data_time = data.get('timestamp', timestamp)
        age = current_time - data_time
        timeliness = max(0.0, 1.0 - age / max_age) if max_age > 0 else 1.0
        if timeliness < self._thresholds['timeliness']:
            issues.append(f"Data age exceeds threshold: {age:.1f}s")

        # Validity: Check value ranges
        validity = self._assess_validity(data)
        if validity < self._thresholds['validity']:
            issues.append(f"Validity below threshold: {validity:.2%}")

        # Uniqueness: Check for duplicates (simplified)
        uniqueness = 1.0  # Assume unique for single records

        # Calculate overall score (weighted average)
        weights = {
            'completeness': 0.20,
            'accuracy': 0.25,
            'consistency': 0.20,
            'timeliness': 0.15,
            'validity': 0.15,
            'uniqueness': 0.05
        }

        overall = (
            completeness * weights['completeness'] +
            accuracy * weights['accuracy'] +
            consistency * weights['consistency'] +
            timeliness * weights['timeliness'] +
            validity * weights['validity'] +
            uniqueness * weights['uniqueness']
        )

        score = DataQualityScore(
            overall=overall,
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            validity=validity,
            uniqueness=uniqueness,
            issues=issues,
            timestamp=timestamp
        )

        self._assessment_history.append(score)

        return score

    def _assess_completeness(self, data: Dict[str, Any]) -> float:
        """Assess data completeness."""
        if not data:
            return 0.0

        total_fields = len(data)
        missing_count = sum(
            1 for v in data.values()
            if v is None or (isinstance(v, float) and np.isnan(v))
        )

        return 1.0 - (missing_count / total_fields)

    def _assess_accuracy(
        self,
        data: Dict[str, Any],
        reference: Optional[Dict[str, Any]]
    ) -> float:
        """Assess data accuracy against reference."""
        if reference is None:
            # Use stored reference values or assume accurate
            reference = self._reference_values

        if not reference:
            return 1.0

        accurate_count = 0
        compared_count = 0

        for field, value in data.items():
            if field in reference and value is not None:
                ref_value = reference[field]
                compared_count += 1

                # Calculate relative error
                if isinstance(value, (int, float)) and isinstance(ref_value, (int, float)):
                    if ref_value != 0:
                        rel_error = abs(value - ref_value) / abs(ref_value)
                    else:
                        rel_error = abs(value - ref_value)

                    if rel_error < 0.1:  # 10% accuracy threshold
                        accurate_count += 1
                elif value == ref_value:
                    accurate_count += 1

        return accurate_count / compared_count if compared_count > 0 else 1.0

    def _assess_consistency(self, data: Dict[str, Any]) -> float:
        """Assess internal data consistency."""
        consistency_checks = 0
        passed_checks = 0

        # Check flow rate vs velocity consistency
        if 'flow_rate' in data and 'velocity' in data:
            flow = data['flow_rate']
            velocity = data['velocity']
            # Skip if either is None
            if flow is not None and velocity is not None:
                consistency_checks += 1
                # Simplified check: both should be positive or both zero
                if (flow >= 0 and velocity >= 0) or (flow == 0 and velocity == 0):
                    passed_checks += 1

        # Check upstream vs downstream level
        if 'water_level_upstream' in data and 'water_level_downstream' in data:
            consistency_checks += 1
            # Upstream should generally be >= downstream
            if data['water_level_upstream'] >= data['water_level_downstream'] - 0.5:
                passed_checks += 1

        # Check total flow vs sum of individual flows
        if 'total_flow' in data and 'flow_rates' in data:
            consistency_checks += 1
            if isinstance(data['flow_rates'], list):
                sum_flows = sum(data['flow_rates'])
                if abs(data['total_flow'] - sum_flows) < 0.1:
                    passed_checks += 1

        return passed_checks / consistency_checks if consistency_checks > 0 else 1.0

    def _assess_validity(self, data: Dict[str, Any]) -> float:
        """Assess data validity against expected ranges."""
        # Define expected ranges for common fields
        valid_ranges = {
            'flow_rate': (0, 200),
            'velocity': (0, 10),
            'water_level': (0, 50),
            'gate_opening': (0, 5),
            'vibration': (0, 5),
            'temperature': (-20, 60),
            'pressure': (0, 20)
        }

        valid_count = 0
        checked_count = 0

        for field, value in data.items():
            # Match field to known ranges
            matched_range = None
            for range_field, range_values in valid_ranges.items():
                if range_field in field.lower():
                    matched_range = range_values
                    break

            if matched_range and isinstance(value, (int, float)):
                checked_count += 1
                if matched_range[0] <= value <= matched_range[1]:
                    valid_count += 1

        return valid_count / checked_count if checked_count > 0 else 1.0

    def set_reference(self, reference: Dict[str, Any]) -> None:
        """Set reference values for accuracy assessment."""
        self._reference_values = reference

    def get_quality_trend(self, count: int = 100) -> List[float]:
        """Get recent quality score trend."""
        recent = self._assessment_history[-count:]
        return [s.overall for s in recent]

    def get_dimension_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each quality dimension."""
        if not self._assessment_history:
            return {}

        dimensions = ['completeness', 'accuracy', 'consistency',
                     'timeliness', 'validity', 'uniqueness']

        stats = {}
        for dim in dimensions:
            values = [getattr(s, dim) for s in self._assessment_history]
            stats[dim] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'current': values[-1] if values else 0.0
            }

        return stats


class DataGovernanceEngine:
    """
    Unified data governance engine.

    Integrates validation, cleaning, quality assessment,
    and audit logging into a single framework.
    """

    def __init__(self):
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.assessor = DataQualityAssessor()

        # Lineage tracking
        self._lineage_records: Dict[str, DataLineage] = {}

        # Audit log
        self._audit_log: List[Dict[str, Any]] = []

        # Processing statistics
        self._stats = {
            'total_records': 0,
            'validated': 0,
            'cleaned': 0,
            'rejected': 0
        }

        logger.info("DataGovernanceEngine initialized")

    def process(
        self,
        data: Dict[str, Any],
        source: str = "simulation",
        timestamp: float = 0.0,
        auto_clean: bool = True,
        reject_invalid: bool = False
    ) -> Tuple[Dict[str, Any], DataQualityScore, DataStatus]:
        """
        Process data through the governance pipeline.

        Args:
            data: Raw data to process
            source: Data source identifier
            timestamp: Data timestamp
            auto_clean: Whether to automatically clean data
            reject_invalid: Whether to reject invalid data

        Returns:
            Tuple of (processed_data, quality_score, status)
        """
        self._stats['total_records'] += 1

        # Step 1: Validation
        is_valid, issues, validated_data = self.validator.validate(data, timestamp)
        self._stats['validated'] += 1

        if not is_valid and reject_invalid:
            self._stats['rejected'] += 1
            self._log_audit('REJECTED', source, data, issues)
            return data, DataQualityScore(overall=0.0), DataStatus.REJECTED

        # Step 2: Cleaning (if enabled and needed)
        processed_data = validated_data
        status = DataStatus.VALIDATED

        if auto_clean and issues:
            processed_data, applied_methods = self.cleaner.clean(validated_data, timestamp)
            status = DataStatus.CLEANED
            self._stats['cleaned'] += 1

        # Step 3: Quality Assessment
        quality_score = self.assessor.assess(processed_data, None, timestamp)

        # Step 4: Update lineage
        self._update_lineage(source, processed_data, quality_score, timestamp)

        # Step 5: Audit logging
        self._log_audit('PROCESSED', source, processed_data, issues, quality_score)

        return processed_data, quality_score, status

    def _update_lineage(
        self,
        source: str,
        data: Dict[str, Any],
        quality_score: DataQualityScore,
        timestamp: float
    ) -> None:
        """Update data lineage record."""
        if source not in self._lineage_records:
            self._lineage_records[source] = DataLineage(
                source=source,
                created_at=timestamp
            )

        lineage = self._lineage_records[source]
        lineage.modified_at = timestamp
        lineage.version += 1
        lineage.quality_scores.append(quality_score)

        # Keep only recent quality scores
        if len(lineage.quality_scores) > 1000:
            lineage.quality_scores = lineage.quality_scores[-1000:]

    def _log_audit(
        self,
        action: str,
        source: str,
        data: Dict[str, Any],
        issues: Optional[List[DataIssue]] = None,
        quality_score: Optional[DataQualityScore] = None
    ) -> None:
        """Add entry to audit log."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'source': source,
            'field_count': len(data),
            'issue_count': len(issues) if issues else 0,
            'quality_score': quality_score.overall if quality_score else None
        }
        self._audit_log.append(entry)

        # Keep log size manageable
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-10000:]

    def get_governance_report(self) -> Dict[str, Any]:
        """Generate comprehensive governance report."""
        return {
            'statistics': self._stats.copy(),
            'validation_stats': self.validator.get_validation_stats(),
            'cleaning_stats': self.cleaner.get_cleaning_stats(),
            'quality_stats': self.assessor.get_dimension_stats(),
            'lineage_sources': list(self._lineage_records.keys()),
            'audit_log_size': len(self._audit_log),
            'recent_audit': self._audit_log[-10:] if self._audit_log else []
        }

    def get_lineage(self, source: str) -> Optional[DataLineage]:
        """Get lineage record for a data source."""
        return self._lineage_records.get(source)

    def export_audit_log(self, count: int = 100) -> List[Dict[str, Any]]:
        """Export recent audit log entries."""
        return self._audit_log[-count:]

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self._stats = {
            'total_records': 0,
            'validated': 0,
            'cleaned': 0,
            'rejected': 0
        }
        logger.info("Governance statistics reset")
