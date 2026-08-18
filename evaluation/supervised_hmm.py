"""Supervised Gaussian HMM (Wang & Hirsa).

Slim extract of the regime-identification and supervised HMM utilities from
HMMGAN/src/trainer/trainer_hmmgan.py. No torch / GAN dependencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import SpectralClustering

def _coerce_series(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Expected at least one return observation.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Return observations must be finite.")
    return values


def _ewm_volatility(returns: np.ndarray, span: int) -> np.ndarray:
    """Ground-truth feature: exponentially weighted moving volatility."""
    volatility = (
        pd.Series(_coerce_series(returns))
        .ewm(span=max(2, int(span)), adjust=False)
        .var(bias=False)
        .pow(0.5)
        .bfill()
        .fillna(0.0)
    )
    return volatility.to_numpy(dtype=float)


def _segment_sse(values: np.ndarray) -> float:
    centered = values - float(np.mean(values))
    return float(np.dot(centered, centered))


def _detect_change_points(
    values: np.ndarray,
    min_segment_length: int,
    penalty: float | None = None,
    max_segments: int = 20,
) -> list[int]:
    """Binary segmentation approximation of the change-point step."""
    values = _coerce_series(values)
    min_segment_length = max(2, int(min_segment_length))
    if penalty is None:
        penalty = float(np.var(values) * np.log(len(values) + 1.0))

    segments = [(0, len(values))]
    while len(segments) < int(max_segments):
        best = None
        best_gain = float(penalty)
        for segment_index, (start, end) in enumerate(segments):
            if end - start < 2 * min_segment_length:
                continue
            total = _segment_sse(values[start:end])
            for split in range(start + min_segment_length, end - min_segment_length + 1):
                gain = (
                    total
                    - _segment_sse(values[start:split])
                    - _segment_sse(values[split:end])
                )
                if gain > best_gain:
                    best_gain = float(gain)
                    best = (segment_index, start, split, end)
        if best is None:
            break
        segment_index, start, split, end = best
        segments.pop(segment_index)
        segments.insert(segment_index, (start, split))
        segments.insert(segment_index + 1, (split, end))
    return sorted(end for _, end in segments[:-1])


def identify_regimes(
    standardized_returns: np.ndarray,
    n_regimes: int = 5,
    volatility_window: int = 20,
    min_segment_length: int = 32,
    changepoint_penalty: float | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, dict]:
    """Five-regime ground truth from EWM volatility segmentation.

    This method computes EWM volatility, detects change points, and clusters the
    resulting contiguous segments with spectral clustering. Labels are remapped
    so 0 is the lowest-volatility regime and ``n_regimes - 1`` the highest.
    """
    returns = _coerce_series(standardized_returns)
    volatility = _ewm_volatility(returns, volatility_window)
    change_points = _detect_change_points(
        volatility,
        min_segment_length=min_segment_length,
        penalty=changepoint_penalty,
        max_segments=max(n_regimes * 4, n_regimes),
    )

    boundaries = [0, *change_points, len(returns)]
    segment_rows = []
    features = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment_returns = returns[start:end]
        segment_volatility = volatility[start:end]
        row = {
            "start": int(start),
            "end": int(end),
            "length": int(end - start),
            "mean_return": float(np.mean(segment_returns)),
            "variance_return": float(np.var(segment_returns)),
            "mean_abs_return": float(np.mean(np.abs(segment_returns))),
            "mean_volatility": float(np.mean(segment_volatility)),
            "std_volatility": float(np.std(segment_volatility)),
            "max_volatility": float(np.max(segment_volatility)),
        }
        segment_rows.append(row)
        features.append(
            [
                row["mean_return"],
                row["variance_return"],
                row["mean_abs_return"],
                row["mean_volatility"],
                row["std_volatility"],
                row["max_volatility"],
                np.log1p(row["length"]),
            ]
        )

    if len(segment_rows) < n_regimes:
        raise ValueError(
            f"Need at least {n_regimes} change-point segments; found {len(segment_rows)}."
        )

    features = np.asarray(features, dtype=float)
    scale = np.std(features, axis=0, keepdims=True)
    features = (features - np.mean(features, axis=0, keepdims=True)) / np.where(
        scale == 0.0, 1.0, scale
    )
    clustering = SpectralClustering(
        n_clusters=int(n_regimes),
        affinity="rbf",
        assign_labels="kmeans",
        random_state=int(random_state),
    )
    raw_segment_labels = clustering.fit_predict(features)
    cluster_volatility = np.array(
        [
            np.mean(
                [
                    row["mean_volatility"]
                    for row, raw_label in zip(segment_rows, raw_segment_labels)
                    if raw_label == cluster_id
                ]
            )
            for cluster_id in range(n_regimes)
        ],
        dtype=float,
    )
    order = np.argsort(cluster_volatility)
    mapping = {int(old): int(new) for new, old in enumerate(order)}
    segment_labels = np.array([mapping[int(label)] for label in raw_segment_labels], dtype=int)

    labels = np.empty(len(returns), dtype=int)
    for row, label in zip(segment_rows, segment_labels):
        labels[row["start"] : row["end"]] = label

    diagnostics = {
        "method": "ewmv_changepoint_spectral",
        "volatility": volatility,
        "change_points": change_points,
        "segment_table": [
            {**row, "raw_cluster": int(raw), "regime_label": int(label)}
            for row, raw, label in zip(segment_rows, raw_segment_labels, segment_labels)
        ],
        "regime_mean_volatility": cluster_volatility[order],
    }
    return labels, diagnostics


def assign_regimes_by_centers(
    standardized_returns: np.ndarray,
    regime_mean_volatility: np.ndarray,
    volatility_window: int = 20,
) -> np.ndarray:
    """Assign new observations to the nearest regime volatility center."""
    volatility = _ewm_volatility(standardized_returns, volatility_window)
    centers = np.asarray(regime_mean_volatility, dtype=float).reshape(1, -1)
    distances = np.abs(volatility[:, np.newaxis] - centers)
    return np.argmin(distances, axis=1).astype(int)


class SupervisedGaussianHMM:
    """Supervised Gaussian HMM following Section 6.3 of Wang and Hirsa.

    Training conditions on both observed state labels and observed returns:
    transitions use the posterior mean under a Dirichlet(1/K) prior,
    and each state's Gaussian emission is MAP-estimated under Normal(0,1)
    mean and Half-Cauchy(1) scale priors. Inference uses forward-backward;
    forecasts use the final filtered state only.
    """

    def __init__(
        self,
        n_components: int,
        transition_prior: float | None = None,
        min_variance: float = 1e-8,
    ):
        self.n_components = int(n_components)
        self.transition_prior = (
            1.0 / self.n_components if transition_prior is None else float(transition_prior)
        )
        self.min_variance = float(min_variance)

    def fit(self, observed_returns: np.ndarray, regime_labels: np.ndarray):
        observations = _coerce_series(observed_returns)
        labels = np.asarray(regime_labels, dtype=int).reshape(-1)
        if len(observations) != len(labels):
            raise ValueError("observed_returns and regime_labels must have equal length.")
        if np.any((labels < 0) | (labels >= self.n_components)):
            raise ValueError("regime_labels contain a state outside the configured range.")

        counts = np.full(
            (self.n_components, self.n_components),
            self.transition_prior,
            dtype=float,
        )
        for left, right in zip(labels[:-1], labels[1:]):
            counts[int(left), int(right)] += 1.0
        self.transmat_ = counts / counts.sum(axis=1, keepdims=True)

        # Initialize the hidden sequence in Regime 0.
        self.startprob_ = np.zeros(self.n_components, dtype=float)
        self.startprob_[0] = 1.0

        global_mean = float(np.mean(observations))
        global_variance = float(max(np.var(observations), self.min_variance))
        self.means_ = np.full(self.n_components, global_mean, dtype=float)
        self.variances_ = np.full(self.n_components, global_variance, dtype=float)
        for state in range(self.n_components):
            values = observations[labels == state]
            if len(values) == 0:
                continue

            initial_std = float(
                np.sqrt(
                    max(
                        np.var(values, ddof=1) if len(values) > 1 else global_variance,
                        self.min_variance,
                    )
                )
            )
            initial = np.array([float(np.mean(values)), np.log(initial_std)], dtype=float)

            def negative_log_posterior(parameters: np.ndarray, values=values) -> float:
                mean, log_scale = parameters
                scale_squared = np.exp(2.0 * log_scale)
                residual_sum = float(np.sum((values - mean) ** 2))
                return float(
                    len(values) * log_scale
                    + 0.5 * residual_sum / scale_squared
                    + 0.5 * mean**2
                    + np.log1p(scale_squared)
                )

            fitted = minimize(
                negative_log_posterior,
                initial,
                method="L-BFGS-B",
                bounds=[(None, None), (-12.0, 6.0)],
            )
            if fitted.success and np.all(np.isfinite(fitted.x)):
                self.means_[state] = float(fitted.x[0])
                self.variances_[state] = float(
                    max(np.exp(2.0 * fitted.x[1]), self.min_variance)
                )
            else:
                self.means_[state] = float(np.mean(values))
                self.variances_[state] = float(initial_std**2)

        self.training_labels_ = labels.copy()
        return self

    def _emission_probabilities(self, observations: np.ndarray) -> np.ndarray:
        observations = _coerce_series(observations)
        variance = np.clip(self.variances_, self.min_variance, None)
        residual = observations[:, np.newaxis] - self.means_[np.newaxis, :]
        density = np.exp(-0.5 * residual**2 / variance[np.newaxis, :])
        density /= np.sqrt(2.0 * np.pi * variance[np.newaxis, :])
        return np.clip(density, 1e-300, None)

    def forward_backward(
        self, observed_returns: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return filtered probabilities, backward messages, and smoothed posteriors."""
        emissions = self._emission_probabilities(observed_returns)
        n_steps = len(emissions)
        filtered = np.zeros_like(emissions)
        scales = np.ones(n_steps, dtype=float)

        filtered[0] = self.startprob_ * emissions[0]
        scales[0] = max(float(filtered[0].sum()), 1e-300)
        filtered[0] /= scales[0]
        for t in range(1, n_steps):
            filtered[t] = (filtered[t - 1] @ self.transmat_) * emissions[t]
            scales[t] = max(float(filtered[t].sum()), 1e-300)
            filtered[t] /= scales[t]

        backward = np.ones_like(emissions)
        for t in range(n_steps - 2, -1, -1):
            backward[t] = self.transmat_ @ (emissions[t + 1] * backward[t + 1])
            backward[t] /= max(float(backward[t].sum()), 1e-300)

        posterior = filtered * backward
        posterior /= np.clip(posterior.sum(axis=1, keepdims=True), 1e-300, None)
        return filtered, backward, posterior

    def predict_proba(self, observed_returns: np.ndarray, smoothed: bool = True) -> np.ndarray:
        filtered, _, posterior = self.forward_backward(observed_returns)
        return posterior if smoothed else filtered

    def predict(self, observed_returns: np.ndarray, smoothed: bool = True) -> np.ndarray:
        return np.argmax(self.predict_proba(observed_returns, smoothed=smoothed), axis=1)

    def forecast_occupancy(
        self,
        observed_returns: np.ndarray,
        horizon: int = 60,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Causal forecast from the final filtered state, not a smoothed future state."""
        filtered = self.predict_proba(observed_returns, smoothed=False)
        current = filtered[-1].copy()
        probabilities = []
        for _ in range(max(1, int(horizon))):
            current = current @ self.transmat_
            probabilities.append(current.copy())
        probabilities = np.asarray(probabilities, dtype=float)
        occupancy = probabilities.mean(axis=0)
        return probabilities, occupancy, int(np.argmax(occupancy))


def fit_supervised_hmm(
    observed_returns: np.ndarray,
    regime_labels: np.ndarray,
    n_regimes: int = 5,
) -> SupervisedGaussianHMM:
    """Fit the supervised state-conditioned Gaussian HMM."""
    return SupervisedGaussianHMM(n_components=n_regimes).fit(
        observed_returns=observed_returns,
        regime_labels=regime_labels,
    )
