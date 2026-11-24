import logging

import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


class DataDriftDetector:

    def __init__(
        self,
        threshold=0.05,
        effect_size_threshold=0.1
    ):
        self.threshold = threshold
        self.effect_size_threshold = effect_size_threshold

    def _benjamini_hochberg(self, p_values):
        """
        Apply Benjamini-Hochberg correction
        to control the false discovery rate.
        """

        total = len(p_values)

        if total == 0:
            return []

        sorted_indices = sorted(
            range(total),
            key=lambda i: p_values[i]
        )

        adjusted = [1.0] * total
        minimum = 1.0

        for rank_index in range(total - 1, -1, -1):

            original_index = sorted_indices[rank_index]
            rank = rank_index + 1

            corrected = (
                p_values[original_index] * total / rank
            )

            minimum = min(minimum, corrected)

            adjusted[original_index] = min(
                minimum,
                1.0
            )

        return adjusted

    def detect(self, reference_data, current_data):

        if reference_data.empty or current_data.empty:
            raise ValueError("Reference and current datasets cannot be empty.")

        common_features = [
            feature
            for feature in reference_data.columns
            if feature in current_data.columns
        ]

        if not common_features:
            raise ValueError("No common features found.")

        raw_results = []

        for feature in common_features:

            reference = pd.to_numeric(
                reference_data[feature],
                errors="coerce"
            ).dropna()

            current = pd.to_numeric(
                current_data[feature],
                errors="coerce"
            ).dropna()

            if len(reference) < 2 or len(current) < 2:
                logger.warning(
                    "Skipping feature %s due to insufficient data.",
                    feature
                )
                continue

            statistic, p_value = ks_2samp(
                reference,
                current
            )

            raw_results.append({
                "feature": str(feature),
                "statistic": float(statistic),
                "p_value": float(p_value),
                "reference_samples": int(len(reference)),
                "current_samples": int(len(current))
            })

        if not raw_results:
            raise ValueError(
                "No features had sufficient valid samples."
            )

        # Multiple-testing correction
        adjusted_p_values = self._benjamini_hochberg(
            [result["p_value"] for result in raw_results]
        )

        results = []

        for result, adjusted_p in zip(
            raw_results,
            adjusted_p_values
        ):

            effect_size = result["statistic"]

            statistically_significant = (
                adjusted_p < self.threshold
            )

            meaningful_drift = (
                effect_size >= self.effect_size_threshold
            )

            drift_detected = (
                statistically_significant
                and meaningful_drift
            )

            # Severity based on KS effect size
            if not drift_detected:
                severity = "none"

            elif effect_size < 0.2:
                severity = "low"

            elif effect_size < 0.4:
                severity = "moderate"

            else:
                severity = "high"

            results.append({
                **result,
                "adjusted_p_value": float(adjusted_p),
                "effect_size": float(effect_size),
                "statistically_significant": bool(
                    statistically_significant
                ),
                "drift_detected": bool(drift_detected),
                "severity": severity
            })

        drifted_features = [
            result
            for result in results
            if result["drift_detected"]
        ]

        drift_percentage = (
            len(drifted_features) / len(results) * 100
        )

        if drift_percentage == 0:
            overall_severity = "none"

        elif drift_percentage < 20:
            overall_severity = "low"

        elif drift_percentage < 50:
            overall_severity = "moderate"

        else:
            overall_severity = "high"

        return {
            "total_features": len(results),
            "drifted_features": len(drifted_features),
            "drift_percentage": round(
                drift_percentage,
                2
            ),
            "drift_detected": len(drifted_features) > 0,
            "overall_severity": overall_severity,
            "significance_threshold": self.threshold,
            "effect_size_threshold": self.effect_size_threshold,
            "correction_method": "Benjamini-Hochberg",
            "feature_results": results
        }