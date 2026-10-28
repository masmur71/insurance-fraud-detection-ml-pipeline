"""Bounded, per-process telemetry without storing claim payloads or identifiers."""
import logging
from collections import Counter
from threading import Lock

LOGGER = logging.getLogger("claim_fraud.predictions")


class Monitor:
    def __init__(self, reference_review_fraction=0.05):
        self.lock = Lock()
        self.requests = Counter()
        self.request_seconds = 0.0
        self.predictions = 0
        self.flagged = 0
        self.score_sum = 0.0
        self.histogram = [0] * 10
        self.reference_review_fraction = reference_review_fraction

    def observe_request(self, route, status, seconds):
        with self.lock:
            self.requests[(route, status)] += 1
            self.request_seconds += seconds

    def observe_prediction(self, score, flagged, version):
        with self.lock:
            self.predictions += 1
            self.flagged += int(flagged)
            self.score_sum += score
            self.histogram[min(int(score * 10), 9)] += 1
        LOGGER.info("prediction model=%s score=%.6f review=%s", version, score, flagged)

    def snapshot(self):
        with self.lock:
            rate = self.flagged / self.predictions if self.predictions else None
            return {"scope": "process_lifetime", "predictions": self.predictions,
                    "review_fraction": rate,
                    "mean_score": self.score_sum / self.predictions if self.predictions else None,
                    "score_bins": list(self.histogram), "score_bin_width": 0.1,
                    "reference_review_fraction": self.reference_review_fraction,
                    "review_fraction_shift": (rate - self.reference_review_fraction
                                               if rate is not None else None),
                    "review_volume_alert": self.predictions >= 100 and rate > 0.075}

    def prometheus(self):
        with self.lock:
            rows = ["# TYPE claim_api_requests_total counter"]
            rows += [f'claim_api_requests_total{{route="{route}",status="{status}"}} {count}'
                     for (route, status), count in sorted(self.requests.items())]
            rows += ["# TYPE claim_api_request_duration_seconds_sum counter",
                     f"claim_api_request_duration_seconds_sum {self.request_seconds:.6f}",
                     "# TYPE claim_api_predictions_total counter",
                     f"claim_api_predictions_total {self.predictions}",
                     "# TYPE claim_api_review_total counter",
                     f"claim_api_review_total {self.flagged}"]
            return "\n".join(rows) + "\n"
