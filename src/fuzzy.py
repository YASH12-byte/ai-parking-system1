import numpy as np


def compute_slot_score(vehicle_size: float, distance_to_gate: float, user_priority: float) -> float:
    """Simple soft-computing/fuzzy-like scoring.

    - Smaller distance is better.
    - Vehicle size closer to 1.0 is better; penalize extremes.
    - Higher user priority is better.
    Returns score in [0, 1].
    """
    # Normalize distance with soft cap
    dist_norm = 1.0 - (np.tanh(distance_to_gate / 50.0))  # ~1 near gate, ~0 far

    # Size preference: ideal is 1.0, penalize deviation with Gaussian
    size_penalty = np.exp(-((vehicle_size - 1.0) ** 2) / (2 * 0.4 ** 2))  # sigma=0.4

    # Combine with weights (soft computing style)
    score = 0.45 * dist_norm + 0.25 * size_penalty + 0.30 * user_priority
    score = float(np.clip(score, 0.0, 1.0))
    return score


