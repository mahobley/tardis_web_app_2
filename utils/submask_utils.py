from __future__ import annotations

from typing import TypeVar

import numpy as np

LabelT = TypeVar("LabelT")

from typing import TypeVar
import numpy as np

LabelT = TypeVar("LabelT")

from typing import TypeVar
import numpy as np

LabelT = TypeVar("LabelT")

# need to put this in and rerun trainings # MAH 2026-05-02 13:05:00
def filter_out_submasks(
    masks: list[np.ndarray],
    labels: list[LabelT] | None = None,
    confidences: list[float] | None = None,
    threshold: float = 0.5,
    remove_super: bool = False,
    verbose: bool = False,
) -> tuple[list[np.ndarray], list[LabelT] | None, list[bool]]:
    """Remove masks that are mostly contained within another mask.

    Always returns:
        filtered_masks, filtered_labels, filtered_bool

    where:
        filtered_bool[i] is True iff masks[i] was kept.

    Notes:
    - If labels is None, filtered_labels is None.
    - If two masks mostly contain each other, they are treated as near-duplicates,
    and the higher-confidence mask is kept.
    - If confidences is None, ties are resolved by keeping the earlier mask.
    """

    n = len(masks)

    if labels is not None and len(labels) != n:
        raise ValueError(
            f"Expected equal lengths for masks and labels, got {n} and {len(labels)}."
        )

    if confidences is not None and len(confidences) != n:
        raise ValueError(
            f"Expected equal lengths for masks and confidences, got {n} and {len(confidences)}."
        )

    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}.")

    if confidences is None:
        # Deterministic fallback: all masks have equal confidence.
        conf = [1.0] * n
    else:
        conf = list(confidences)

    bool_masks = [np.asarray(mask, dtype=bool) for mask in masks]
    areas = np.array([mask.sum() for mask in bool_masks], dtype=float)

    bad_indices: set[int] = set()

    def priority(idx: int) -> tuple[float, int]:
        # Higher confidence wins.
        # For equal confidence, lower original index wins.
        return conf[idx], -idx

    def remove(idx: int, reason: str) -> None:
        if idx not in bad_indices:
            bad_indices.add(idx)
            if verbose:
                print(reason)

    for i in range(n):
        if areas[i] == 0:
            remove(i, f"Removing mask {i} because it has zero area.")
            continue

        for j in range(i + 1, n):
            if areas[j] == 0:
                remove(j, f"Removing mask {j} because it has zero area.")
                continue

            intersection = np.logical_and(bool_masks[i], bool_masks[j]).sum()

            if intersection == 0:
                continue

            i_contained_in_j = intersection / areas[i]
            j_contained_in_i = intersection / areas[j]

            i_in_j = i_contained_in_j > threshold
            j_in_i = j_contained_in_i > threshold

            if verbose:
                print(
                    f"{i=}, {j=}, "
                    f"intersection={intersection}, "
                    f"area_i={areas[i]}, area_j={areas[j]}, "
                    f"i_contained_in_j={i_contained_in_j:.3f}, "
                    f"j_contained_in_i={j_contained_in_i:.3f}, "
                    f"conf_i={conf[i]:.3f}, conf_j={conf[j]:.3f}"
                )

            # Near-duplicate: both masks mostly contain each other.
            # Keep higher confidence.
            if i_in_j and j_in_i:
                loser = j if priority(i) >= priority(j) else i
                winner = i if loser == j else j

                remove(
                    loser,
                    (
                        f"Removing mask {loser} as near-duplicate of mask {winner}; "
                        f"conf_loser={conf[loser]:.3f}, "
                        f"conf_winner={conf[winner]:.3f}, "
                        f"i_contained_in_j={i_contained_in_j:.3f}, "
                        f"j_contained_in_i={j_contained_in_i:.3f}"
                    ),
                )

            # i is mostly inside j.
            elif i_in_j:
                if remove_super:
                    loser = j
                    reason = (
                        f"Removing mask {j} because it is a supermask of mask {i}; "
                        f"ioa={i_contained_in_j:.3f}"
                    )
                else:
                    loser = i
                    reason = (
                        f"Removing mask {i} because it is a submask of mask {j}; "
                        f"ioa={i_contained_in_j:.3f}"
                    )

                remove(loser, reason)

            # j is mostly inside i.
            elif j_in_i:
                if remove_super:
                    loser = i
                    reason = (
                        f"Removing mask {i} because it is a supermask of mask {j}; "
                        f"ioa={j_contained_in_i:.3f}"
                    )
                else:
                    loser = j
                    reason = (
                        f"Removing mask {j} because it is a submask of mask {i}; "
                        f"ioa={j_contained_in_i:.3f}"
                    )

                remove(loser, reason)

    kept_indices = [i for i in range(n) if i not in bad_indices]

    filtered_masks = [masks[i] for i in kept_indices]
    filtered_labels = (
        [labels[i] for i in kept_indices] if labels is not None else None
    )
    filtered_bool = [i not in bad_indices for i in range(n)]

    return filtered_masks, filtered_labels, filtered_bool