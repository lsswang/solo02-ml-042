"""Global subtitle segmentation that consumes joint word-boundary tables.

Unlike the previous mean-boundary/greedy implementation, this module treats
each supplied table as the joint distribution of the true start of word ``i``
and the true end of word ``j``.  Coverage of a cue therefore uses the joint
survival mass ``sum_{S >= s, E <= e} P(S, E)``, and cue cost uses the joint
second moments.  No independence assumption is made and no acoustic model is
consulted.

The optimisation is an exact dynamic program over the last covered word and
the end frame of the previous cue.  It explores all legal word groupings and
all integer display intervals at once.
"""

from __future__ import annotations

import math
from array import array

from common import (
    InputError,
    is_real,
    require_int,
    require_mapping,
    require_real,
    require_sequence,
    require_text,
    scalar_len,
)

PROB_SUM_TOL = 1e-12
COVERAGE_TOL = 1e-10
MAX_T = 48
MAX_WORDS = 48  # an external caller must still supply at least one table per group
MAX_CHARS = 40


def _parse_group_key(key, word_count):
    if isinstance(key, str):
        parts = key.split(":")
        if len(parts) != 2:
            raise InputError(f"invalid interval table key {key!r}")
        for part in parts:
            if not part.isdigit():
                raise InputError(f"invalid interval table key {key!r}")
        first = int(parts[0])
        last = int(parts[1])
    elif isinstance(key, (tuple, list)) and len(key) == 2:
        first = require_int(key[0], "interval table key start", low=0)
        last = require_int(key[1], "interval table key end", low=0)
    else:
        raise InputError(f"invalid interval table key {key!r}")
    if not (0 <= first <= last < word_count):
        raise InputError(f"interval table key {key!r} is outside the word range")
    return first, last


def _row_fields(row, label):
    if isinstance(row, dict):
        for field in ("start", "end", "probability"):
            if field not in row:
                raise InputError(f"{label} is missing {field!r}")
        return row["start"], row["end"], row["probability"]
    if isinstance(row, (list, tuple)):
        if len(row) != 3:
            raise InputError(f"{label} must contain (start, end, probability)")
        return row[0], row[1], row[2]
    raise InputError(f"{label} must be a mapping or a 3-item sequence")


def _validate_policy(policy):
    policy_map = require_mapping(policy, "policy")
    required = (
        "T",
        "max_chars",
        "min_duration",
        "max_duration",
        "chars_per_frame",
        "coverage",
        "cue_penalty",
    )
    for field in required:
        if field not in policy_map:
            raise InputError(f"policy is missing {field!r}")
    frame_count = require_int(policy_map["T"], "policy.T", low=1, high=MAX_T)
    max_chars = require_int(policy_map["max_chars"], "policy.max_chars", low=1, high=MAX_CHARS)
    min_duration = require_int(
        policy_map["min_duration"], "policy.min_duration", low=1, high=frame_count
    )
    max_duration = require_int(
        policy_map["max_duration"], "policy.max_duration", low=1, high=frame_count
    )
    if min_duration > max_duration:
        raise InputError("policy.min_duration must not exceed policy.max_duration")
    chars_per_frame = require_real(
        policy_map["chars_per_frame"],
        "policy.chars_per_frame",
        low=0.5,
        high=20.0,
    )
    coverage = require_real(
        policy_map["coverage"], "policy.coverage", low=0.5, high=1.0
    )
    cue_penalty = require_real(
        policy_map["cue_penalty"], "policy.cue_penalty", low=0.0, high=20.0
    )
    return {
        "T": frame_count,
        "max_chars": max_chars,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "chars_per_frame": chars_per_frame,
        "coverage": coverage,
        "cue_penalty": cue_penalty,
    }


def _validate_words(words):
    raw = require_sequence(words, "words")
    if not 1 <= len(raw) <= MAX_WORDS:
        raise InputError(f"words must contain 1..{MAX_WORDS} words")
    parsed = [
        require_text(word, f"words[{index}]") for index, word in enumerate(raw)
    ]
    widths = [scalar_len(word) for word in parsed]
    return parsed, widths


def _iter_table_rows(rows, label):
    """Yield ``(start, end, probability)`` from a supported table spelling.

    Accepted forms:

    * list of ``{"start": ..., "end": ..., "probability": ...}`` mappings;
    * list of ``(start, end, probability)`` triples;
    * mapping ``{(start, end): probability, ...}``.
    """

    if isinstance(rows, dict):
        for boundaries, probability in rows.items():
            if not isinstance(boundaries, (tuple, list)) or len(boundaries) != 2:
                raise InputError(
                    f"{label} keys must be (start, end) pairs, got {boundaries!r}"
                )
            yield boundaries[0], boundaries[1], probability
        return
    if not isinstance(rows, (list, tuple)):
        raise InputError(f"{label} must be a list, tuple, or mapping")
    for row_index, row in enumerate(rows):
        start, end, probability = _row_fields(row, f"{label}[{row_index}]")
        yield start, end, probability


def _validate_tables(interval_tables, word_count, frame_count):
    tables_map = require_mapping(interval_tables, "interval_tables")
    parsed = {}
    for key, rows in tables_map.items():
        first, last = _parse_group_key(key, word_count)
        label = f"interval_tables[{key!r}]"
        entries = {}
        raw_total = []
        for row_index, (start_raw, end_raw, probability_raw) in enumerate(
            _iter_table_rows(rows, label)
        ):
            row_label = f"{label}[{row_index}]"
            start = require_int(start_raw, f"{row_label}.start", low=0, high=frame_count)
            end = require_int(end_raw, f"{row_label}.end", low=0, high=frame_count)
            if start >= end:
                raise InputError(f"{row_label} must satisfy start < end")
            if not is_real(probability_raw):
                raise InputError(f"{row_label}.probability must be a real number")
            probability = float(probability_raw)
            if not math.isfinite(probability) or probability < 0.0:
                raise InputError(
                    f"{row_label}.probability must be finite and non-negative"
                )
            if (start, end) in entries:
                # A distribution is a function of the boundary pair; repeated
                # rows are added rather than silently overwriting each other.
                entries[(start, end)] += probability
            else:
                entries[(start, end)] = probability
            raw_total.append(probability)
        if not entries:
            raise InputError(f"{label} must contain at least one probability entry")
        total = math.fsum(raw_total)
        if total <= 0.0:
            raise InputError(f"{label} has no positive probability")
        if abs(total - 1.0) > PROB_SUM_TOL:
            raise InputError(
                f"{label} probabilities sum to {total!r}, not 1 within "
                f"{PROB_SUM_TOL}"
            )
        parsed[(first, last)] = {
            boundaries: probability / total for boundaries, probability in entries.items()
        }

    for first in range(word_count):
        for last in range(first, word_count):
            if (first, last) not in parsed:
                raise InputError(
                    f"interval_tables is missing the table for word group {first}:{last}"
                )
    return parsed


class _Group:
    __slots__ = (
        "first",
        "last",
        "expected_start",
        "expected_end",
        "second_start",
        "second_end",
        "coverage",
        "stride",
        "joined_width",
        "one_line_width",
        "two_line_split",
        "two_line_widths",
    )

    def __init__(self, first, last, entries, frame_count):
        self.first = first
        self.last = last
        stride = frame_count + 1
        # math.fsum keeps the stored-value sums stable.
        self.expected_start = math.fsum(
            probability * start for (start, _end), probability in entries.items()
        )
        self.expected_end = math.fsum(
            probability * end for (_start, end), probability in entries.items()
        )
        self.second_start = math.fsum(
            probability * start * start for (start, _end), probability in entries.items()
        )
        self.second_end = math.fsum(
            probability * end * end for (_start, end), probability in entries.items()
        )

        self.stride = stride
        # prefix over end for each start, then suffix over start.
        prefix = [0.0] * (stride * stride)
        for (start, end), probability in entries.items():
            prefix[start * stride + end] += probability
        for start in range(stride):
            running = 0.0
            base = start * stride
            for end in range(stride):
                running += prefix[base + end]
                prefix[base + end] = running
        coverage = array("d", [0.0]) * (stride * stride)
        for start in range(stride - 1, -1, -1):
            base = start * stride
            next_base = (start + 1) * stride if start + 1 < stride else None
            for end in range(stride):
                value = prefix[base + end]
                if next_base is not None:
                    value += coverage[next_base + end]
                coverage[base + end] = value
        self.coverage = coverage

        self.one_line_width = None
        self.two_line_split = None
        self.two_line_widths = None

    def set_layouts(self, words, widths, max_chars):
        total_chars = sum(widths[self.first:self.last + 1])
        spaces = self.last - self.first
        one_width = total_chars + spaces
        self.joined_width = one_width
        if one_width <= max_chars:
            self.one_line_width = one_width

        # A two-line layout needs at least two words and a split where both
        # non-empty sides fit.  Choose the most balanced valid split.
        if self.last > self.first:
            best = None
            best_key = None
            left_chars = 0
            for split in range(self.first, self.last):
                left_chars += widths[split]
                right_chars = total_chars - left_chars
                left_width = left_chars + (split - self.first)
                right_width = right_chars + (self.last - split - 1)
                if left_width <= max_chars and right_width <= max_chars:
                    key = (max(left_width, right_width), left_width, split)
                    if best_key is None or key < best_key:
                        best_key = key
                        best = (split, left_width, right_width)
            if best is not None:
                self.two_line_split = best[0]
                self.two_line_widths = (best[1], best[2])

    def feasible(self, start, end, policy):
        duration = end - start
        if duration < policy["min_duration"] or duration > policy["max_duration"]:
            return False
        if self.coverage[start * self.stride + end] < policy["coverage"] - COVERAGE_TOL:
            return False
        # The reading-speed rule always uses the group joined on one line,
        # including its spaces.  A line break may improve the layout but must
        # never reduce the amount of text the viewer has to read.
        if self.joined_width > policy["chars_per_frame"] * duration:
            return False
        if self.one_line_width is not None:
            return True
        return self.two_line_split is not None

    def error_terms(self, start, end):
        start_term = (
            start * start
            - 2.0 * start * self.expected_start
            + self.second_start
        )
        end_term = (
            end * end
            - 2.0 * end * self.expected_end
            + self.second_end
        )
        return start_term, end_term


def _choose_layout(group, start, end, policy, words, words_slice):
    duration = end - start
    if group.one_line_width is not None:
        return [" ".join(words_slice)]
    if group.two_line_split is not None:
        split = group.two_line_split
        left = " ".join(words[group.first:split + 1])
        right = " ".join(words[split + 1:group.last + 1])
        return [left, right]
    # Should not be reached because feasibility was checked first.
    return None


def plan(words, interval_tables, policy):
    """Find the globally cheapest legal subtitle plan.

    ``interval_tables`` maps ``"i:j"`` (or ``(i, j)``) to a distribution over
    integer ``(start, end)`` boundaries of the consecutive word group ``i..j``.
    Every group table is used directly; they are never multiplied together.
    """

    words_list, word_widths = _validate_words(words)
    policy_map = _validate_policy(policy)
    frame_count = policy_map["T"]
    word_count = len(words_list)
    entries_by_group = _validate_tables(interval_tables, word_count, frame_count)

    groups = {}
    for first in range(word_count):
        for last in range(first, word_count):
            group = _Group(first, last, entries_by_group[(first, last)], frame_count)
            group.set_layouts(
                words_list,
                word_widths,
                policy_map["max_chars"],
            )
            groups[(first, last)] = group

    infinity = float("inf")
    # dp[j][e]: cheapest way to cover words 0..j-1 with the last cue ending at e.
    dp = [[infinity] * (frame_count + 1) for _ in range(word_count + 1)]
    dp[0][0] = 0.0
    # back[j][e] = (previous covered-word count, cue start, previous cue end)
    back = [[None] * (frame_count + 1) for _ in range(word_count + 1)]
    back[0][0] = (None, None, None)

    for covered in range(word_count):
        # A cue may start anywhere at or after the previous cue's end.
        prefix_best = [infinity] * (frame_count + 1)
        prefix_arg = [-1] * (frame_count + 1)
        best_value = infinity
        best_arg = -1
        for start in range(frame_count + 1):
            value = dp[covered][start]
            if value < best_value:
                best_value = value
                best_arg = start
            prefix_best[start] = best_value
            prefix_arg[start] = best_arg
        if best_value == infinity:
            continue

        for last in range(covered, word_count):
            group = groups[(covered, last)]
            # s is the cue start; e the exclusive cue end.
            for start in range(frame_count):
                base = prefix_best[start]
                if base == infinity:
                    continue
                previous_end = prefix_arg[start]
                lowest_end = start + policy_map["min_duration"]
                highest_end = min(
                    frame_count, start + policy_map["max_duration"]
                )
                for end in range(lowest_end, highest_end + 1):
                    if not group.feasible(start, end, policy_map):
                        continue
                    start_term, end_term = group.error_terms(start, end)
                    cost = start_term + end_term + policy_map["cue_penalty"]
                    total = base + cost
                    if total < dp[last + 1][end]:
                        dp[last + 1][end] = total
                        back[last + 1][end] = (covered, start, previous_end)

    final_end = -1
    final_cost = infinity
    for end in range(frame_count + 1):
        if dp[word_count][end] < final_cost:
            final_cost = dp[word_count][end]
            final_end = end
    if final_end < 0 or final_cost == infinity:
        return {
            "status": "infeasible",
            "cues": [],
            "total_cost": infinity,
            "objective": infinity,
        }

    cues_reversed = []
    covered = word_count
    end = final_end
    while covered > 0:
        previous_covered, start, previous_end = back[covered][end]
        if previous_covered is None:
            raise RuntimeError("internal error: subtitle backtracking failed")
        group = groups[(previous_covered, covered - 1)]
        lines = _choose_layout(
            group,
            start,
            end,
            policy_map,
            words_list,
            words_list[previous_covered:covered],
        )
        if lines is None:
            raise RuntimeError("internal error: chosen cue has no layout")
        start_term, end_term = group.error_terms(start, end)
        coverage = group.coverage[start * group.stride + end]
        cues_reversed.append(
            {
                "first": previous_covered,
                "last": covered - 1,
                "words": words_list[previous_covered:covered],
                "start": start,
                "end": end,
                "duration": end - start,
                "lines": lines,
                "text": "\n".join(lines),
                "plain_text": " ".join(words_list[previous_covered:covered]),
                "start_error": start_term,
                "end_error": end_term,
                "expected_error": start_term + end_term,
                "cue_penalty": policy_map["cue_penalty"],
                "cost": start_term + end_term + policy_map["cue_penalty"],
                "coverage": coverage,
            }
        )
        covered = previous_covered
        end = previous_end

    cues = list(reversed(cues_reversed))
    # Recompute the objective from the returned cues as an independent sanity
    # check; the DP value and this sum may differ only by roundoff.
    recomputed = math.fsum(cue["cost"] for cue in cues)
    return {
        "status": "ok",
        "cues": cues,
        "total_cost": final_cost,
        "objective": final_cost,
        "recomputed_cost": recomputed,
    }
