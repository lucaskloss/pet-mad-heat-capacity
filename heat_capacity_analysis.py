"""Heat-capacity estimators and correlation analysis."""

import numpy as np

K_B_AU = 3.166811563e-6


def autocorrelation_time(values: np.ndarray) -> float:
    """Return a conservative integrated autocorrelation time in samples."""
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        return 1.0
    centered = x - np.mean(x)
    variance = np.dot(centered, centered)
    if variance <= 0.0:
        return 1.0
    ac = np.correlate(centered, centered, mode="full")[x.size - 1:] / variance
    first_negative = np.flatnonzero(ac[1:] <= 0.0)
    end = first_negative[0] + 1 if first_negative.size else ac.size
    positive = ac[1:end]
    positive = positive[positive > 0.0]
    return max(1.0, float(1.0 + 2.0 * np.sum(positive)))


def _standard_error(values: np.ndarray) -> tuple[float, float]:
    """Return standard error and effective sample count for correlated data."""
    values = np.asarray(values, dtype=float)
    tau = autocorrelation_time(values)
    n_eff = max(1.0, values.size / tau)
    if values.size < 2:
        return 0.0, n_eff
    return float(np.std(values, ddof=1) / np.sqrt(n_eff)), n_eff


def heat_capacity_from_scaledcoords(eps_v: np.ndarray, eps_v_prime: np.ndarray, *, temperature: float,
                                    nmolecules: int, skip: int = 20) -> dict[str, float]:
    """Evaluate Yamamoto's scaled-coordinate estimator in units of kB."""
    eps_v = np.asarray(eps_v, dtype=float).reshape(-1)
    eps_v_prime = np.asarray(eps_v_prime, dtype=float).reshape(-1)
    if eps_v.size != eps_v_prime.size:
        raise ValueError("eps_v and eps_v_prime must contain the same samples")
    if not 0 <= skip < eps_v.size:
        raise ValueError(f"skip={skip} must be between 0 and {eps_v.size - 1}")
    if nmolecules <= 0 or temperature <= 0:
        raise ValueError("nmolecules and temperature must be positive")
    eps_v = eps_v[skip:]
    eps_v_prime = eps_v_prime[skip:]
    beta = 1.0 / (K_B_AU * temperature)
    delta_eps_v_squared = (eps_v - np.mean(eps_v)) ** 2
    cv = K_B_AU * beta**2 * (np.mean(delta_eps_v_squared) - np.mean(eps_v_prime))
    err_delta, n_eff_delta = _standard_error(delta_eps_v_squared)
    err_prime, n_eff_prime = _standard_error(eps_v_prime)
    cv_error = K_B_AU * beta**2 * np.sqrt(err_delta**2 + err_prime**2)
    return {"cv_au": float(cv), "cv_error_au": float(cv_error),
            "cv_per_molecule_kb": float(cv / nmolecules / K_B_AU),
            "cv_error_per_molecule_kb": float(cv_error / nmolecules / K_B_AU),
            "tau_delta_eps_v": autocorrelation_time(delta_eps_v_squared),
            "tau_eps_v_prime": autocorrelation_time(eps_v_prime),
            "effective_samples_delta_eps_v": n_eff_delta,
            "effective_samples_eps_v_prime": n_eff_prime, "samples": float(eps_v.size)}
