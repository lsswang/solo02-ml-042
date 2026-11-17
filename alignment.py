"""Pronunciation-uncertain forced alignment.

The model is a small weighted finite-state transducer: for every word one of
its pronunciation candidates is chosen, the chosen phone strings are
concatenated, and the T frames are partitioned into one non-empty contiguous
block per phone.  A path's mass is

    product(pronunciation prior) * product(acoustic probability per frame).

There are no duration or transition penalties.  ``log_partition`` is the log
sum of the masses of *all* such paths, not the Viterbi score.  Boundary
posteriors are obtained by forward/backward summation over word segments; the
returned interval tables are true joint distributions over (start of word i,
end of word j), not products of one-boundary marginals.

Everything is accumulated in log space, so legal paths whose mass is far below
the smallest normal double still have a finite log mass and are never mistaken
for "no path".
"""

from __future__ import annotations

import math

from common import (
    NEG_INF,
    InputError,
    is_real,
    logadd,
    require_mapping,
    require_sequence,
    require_text,
)

MAX_WORDS = 8
MAX_FRAMES = 48
MIN_PHONES = 2
MAX_PHONES = 8
MAX_CANDIDATES = 3
MAX_CANDIDATE_PHONES = 4

PROB_SUM_TOL = 1e-12
MIN_POSITIVE = 1e-12


def _logsumexp(values):
    total = NEG_INF
    for value in values:
        total = logadd(total, value)
    return total


def _validate_emissions(emissions):
    frames = require_sequence(emissions, "emissions")
    if not 1 <= len(frames) <= MAX_FRAMES:
        raise InputError(f"emissions must contain 1..{MAX_FRAMES} frames")

    phone_set = None
    log_emit = None
    for t, frame in enumerate(frames):
        row = require_mapping(frame, f"emissions[{t}]")
        keys = tuple(row.keys())
        for key in keys:
            if not isinstance(key, str) or key == "":
                raise InputError(
                    f"emissions[{t}] has a non-string or empty phone symbol"
                )
            require_text(key, f"emissions[{t}] phone symbol")
        if phone_set is None:
            phone_set = tuple(sorted(keys))
            if not MIN_PHONES <= len(phone_set) <= MAX_PHONES:
                raise InputError(
                    f"the phone table must contain {MIN_PHONES}..{MAX_PHONES} phones"
                )
            log_emit = {phone: [] for phone in phone_set}
        elif set(keys) != set(phone_set):
            raise InputError(
                f"emissions[{t}] must contain exactly the same phones as frame 0"
            )

        values = {}
        for phone in phone_set:
            if phone not in row:
                raise InputError(f"emissions[{t}] is missing phone {phone!r}")
            value = row[phone]
            if not is_real(value):
                raise InputError(f"emissions[{t}][{phone!r}] must be a real number")
            value = float(value)
            if not math.isfinite(value):
                raise InputError(f"emissions[{t}][{phone!r}] must be finite")
            if value < 0.0:
                raise InputError(f"emissions[{t}][{phone!r}] must be non-negative")
            if 0.0 < value < MIN_POSITIVE:
                raise InputError(
                    f"positive emission probabilities must be at least {MIN_POSITIVE}"
                )
            values[phone] = value
        total = math.fsum(values.values())
        if total <= 0.0:
            raise InputError(f"emissions[{t}] has no positive probability")
        if abs(total - 1.0) > PROB_SUM_TOL:
            raise InputError(
                f"emissions[{t}] probabilities sum to {total!r}, not 1 within "
                f"{PROB_SUM_TOL}"
            )
        # Normalise by the actual row sum, as required by the contract.
        for phone in phone_set:
            probability = values[phone] / total
            log_emit[phone].append(
                math.log(probability) if probability > 0.0 else NEG_INF
            )
    return phone_set, log_emit, len(frames)


def _validate_lexicon(words, lexicon, phone_set):
    lexicon_map = require_mapping(lexicon, "lexicon")
    phone_lookup = set(phone_set)
    parsed = []
    for word_index, word in enumerate(words):
        if word not in lexicon_map:
            raise InputError(f"lexicon has no entry for word {word!r}")
        candidates = require_sequence(
            lexicon_map[word], f"lexicon[{word!r}] pronunciation candidates"
        )
        if not 1 <= len(candidates) <= MAX_CANDIDATES:
            raise InputError(
                f"word {word!r} must have 1..{MAX_CANDIDATES} pronunciation candidates"
            )
        seen_ids = []
        word_candidates = []
        prior_total = 0.0
        for candidate_index, candidate in enumerate(candidates):
            label = f"lexicon[{word!r}][{candidate_index}]"
            candidate_map = require_mapping(candidate, label)
            for required in ("id", "phones", "prior"):
                if required not in candidate_map:
                    raise InputError(f"{label} is missing {required!r}")
            candidate_id = candidate_map["id"]
            if candidate_id is None:
                raise InputError(f"{label}.id must not be None")
            try:
                hash(candidate_id)
            except TypeError as exc:
                raise InputError(f"{label}.id must be hashable") from exc
            if candidate_id in seen_ids:
                raise InputError(
                    f"duplicate pronunciation id {candidate_id!r} for word {word!r}"
                )
            seen_ids.append(candidate_id)

            phones_raw = require_sequence(candidate_map["phones"], f"{label}.phones")
            if not 1 <= len(phones_raw) <= MAX_CANDIDATE_PHONES:
                raise InputError(
                    f"{label}.phones must contain 1..{MAX_CANDIDATE_PHONES} phones"
                )
            phones = tuple(
                require_text(phone, f"{label}.phones[{phone_index}]")
                for phone_index, phone in enumerate(phones_raw)
            )
            for phone in phones:
                if phone not in phone_lookup:
                    raise InputError(
                        f"{label} contains unknown phone {phone!r}"
                    )

            prior = candidate_map["prior"]
            if not is_real(prior):
                raise InputError(f"{label}.prior must be a real number")
            prior = float(prior)
            if not math.isfinite(prior) or prior <= 0.0:
                raise InputError(f"{label}.prior must be finite and strictly positive")
            prior_total += prior
            word_candidates.append(
                {
                    "id": candidate_id,
                    "phones": phones,
                    "prior": prior,
                }
            )
        if abs(prior_total - 1.0) > PROB_SUM_TOL:
            raise InputError(
                f"prior for word {word!r} sums to {prior_total!r}, not 1 within "
                f"{PROB_SUM_TOL}"
            )
        for candidate in word_candidates:
            candidate["log_prior"] = math.log(candidate["prior"] / prior_total)
        parsed.append(word_candidates)
    return parsed


def _segment_tables(phones, log_emit, frame_count):
    """Log mass and log Viterbi tables for one candidate.

    ``sum_table[s][e]`` is the log sum over every valid segmentation of frames
    ``[s, e)`` into the candidate's phones (each at least one frame), excluding
    the pronunciation prior.  ``max_table`` is the analogous maximum.
    """

    phone_count = len(phones)
    sum_table = [[NEG_INF] * (frame_count + 1) for _ in range(frame_count + 1)]
    max_table = [[NEG_INF] * (frame_count + 1) for _ in range(frame_count + 1)]

    for start in range(frame_count - phone_count + 1):
        sum_prev = [NEG_INF] * (frame_count + 1)
        max_prev = [NEG_INF] * (frame_count + 1)
        sum_prev[start] = 0.0
        max_prev[start] = 0.0
        for phone_index in range(1, phone_count + 1):
            emission = log_emit[phones[phone_index - 1]]
            # Remaining phones need at least one frame each.
            last_end = frame_count - (phone_count - phone_index)
            sum_cur = [NEG_INF] * (frame_count + 1)
            max_cur = [NEG_INF] * (frame_count + 1)
            for previous_end in range(start + phone_index - 1, last_end):
                base_sum = sum_prev[previous_end]
                base_max = max_prev[previous_end]
                if base_sum == NEG_INF and base_max == NEG_INF:
                    continue
                segment_sum = 0.0
                for end in range(previous_end + 1, last_end + 1):
                    segment_sum += emission[end - 1]
                    if base_sum != NEG_INF:
                        value = base_sum + segment_sum
                        sum_cur[end] = logadd(sum_cur[end], value)
                    if base_max != NEG_INF:
                        value = base_max + segment_sum
                        if value > max_cur[end]:
                            max_cur[end] = value
            sum_prev, max_prev = sum_cur, max_cur
        for end in range(start + phone_count, frame_count + 1):
            sum_table[start][end] = sum_prev[end]
            max_table[start][end] = max_prev[end]
    return sum_table, max_table


def _reconstruct_phone_intervals(phones, log_emit, start, end):
    """Viterbi segmentation of ``[start, end)`` for a known candidate."""

    phone_count = len(phones)
    dp = [[NEG_INF] * (end + 1) for _ in range(phone_count + 1)]
    back = [[None] * (end + 1) for _ in range(phone_count + 1)]
    dp[0][start] = 0.0
    for phone_index in range(1, phone_count + 1):
        emission = log_emit[phones[phone_index - 1]]
        last_end = end - (phone_count - phone_index)
        for previous_end in range(start + phone_index - 1, last_end):
            if dp[phone_index - 1][previous_end] == NEG_INF:
                continue
            segment_sum = 0.0
            for finish in range(previous_end + 1, last_end + 1):
                segment_sum += emission[finish - 1]
                value = dp[phone_index - 1][previous_end] + segment_sum
                if value > dp[phone_index][finish]:
                    dp[phone_index][finish] = value
                    back[phone_index][finish] = previous_end
    if dp[phone_count][end] == NEG_INF:
        raise RuntimeError("internal error: chosen segmentation is not reachable")
    intervals = []
    finish = end
    for phone_index in range(phone_count, 0, -1):
        previous_end = back[phone_index][finish]
        intervals.append((previous_end, finish))
        finish = previous_end
    intervals.reverse()
    return intervals


def _empty_path():
    return {
        "log_score": NEG_INF,
        "words": [],
        "word_intervals": [],
        "phone_intervals": [],
        "pronunciation_ids": [],
        "pronunciations": [],
        "states": [],
    }


def align(words, emissions, lexicon):
    """Align ``words`` to frames under a pronunciation lexicon.

    Returns a dictionary with ``status``, ``log_partition``, the best ``path``
    (including ``log_score``), and ``interval_tables``.  On ``no_path`` the
    status is ``"no_path"``, ``log_partition`` is ``-inf``, the path is empty,
    and ``interval_tables`` is ``None``.
    """

    words_raw = require_sequence(words, "words")
    if not 1 <= len(words_raw) <= MAX_WORDS:
        raise InputError(f"words must contain 1..{MAX_WORDS} words")
    words_list = [
        require_text(word, f"words[{index}]") for index, word in enumerate(words_raw)
    ]

    phone_set, log_emit, frame_count = _validate_emissions(emissions)
    candidates_by_word = _validate_lexicon(words_list, lexicon, phone_set)

    word_count = len(words_list)

    # Per-word, per-candidate segment tables (log sum / log max).
    candidate_tables = []
    word_sum = []
    word_max = []
    for candidates in candidates_by_word:
        candidate_entries = []
        sum_table = [[NEG_INF] * (frame_count + 1) for _ in range(frame_count + 1)]
        max_table = [[NEG_INF] * (frame_count + 1) for _ in range(frame_count + 1)]
        for candidate in candidates:
            raw_sum, raw_max = _segment_tables(
                candidate["phones"], log_emit, frame_count
            )
            log_prior = candidate["log_prior"]
            candidate_entries.append(
                {
                    "candidate": candidate,
                    "sum": raw_sum,
                    "max": raw_max,
                }
            )
            for start in range(frame_count + 1):
                for end in range(start + 1, frame_count + 1):
                    value_sum = raw_sum[start][end]
                    if value_sum != NEG_INF:
                        total = log_prior + value_sum
                        sum_table[start][end] = logadd(sum_table[start][end], total)
                    value_max = raw_max[start][end]
                    if value_max != NEG_INF:
                        total = log_prior + value_max
                        if total > max_table[start][end]:
                            max_table[start][end] = total
        candidate_tables.append(candidate_entries)
        word_sum.append(sum_table)
        word_max.append(max_table)

    # Forward / backward sums over word boundaries.
    forward = [[NEG_INF] * (frame_count + 1) for _ in range(word_count + 1)]
    forward[0][0] = 0.0
    for word_index in range(word_count):
        table = word_sum[word_index]
        for start in range(frame_count + 1):
            base = forward[word_index][start]
            if base == NEG_INF:
                continue
            for end in range(start + 1, frame_count + 1):
                value = table[start][end]
                if value != NEG_INF:
                    forward[word_index + 1][end] = logadd(
                        forward[word_index + 1][end], base + value
                    )

    backward = [[NEG_INF] * (frame_count + 1) for _ in range(word_count + 1)]
    backward[word_count][frame_count] = 0.0
    for word_index in range(word_count - 1, -1, -1):
        table = word_sum[word_index]
        for start in range(frame_count + 1):
            total = NEG_INF
            for end in range(start + 1, frame_count + 1):
                value = table[start][end]
                if value != NEG_INF and backward[word_index + 1][end] != NEG_INF:
                    total = logadd(total, value + backward[word_index + 1][end])
            backward[word_index][start] = total

    log_partition = forward[word_count][frame_count]

    if log_partition == NEG_INF:
        return {
            "status": "no_path",
            "log_partition": NEG_INF,
            "path": _empty_path(),
            "best_path": _empty_path(),
            "interval_tables": None,
            "score": NEG_INF,
            "log_score": NEG_INF,
            "boundaries": [],
            "states": [],
        }

    # Viterbi over words and boundaries.
    best = [NEG_INF] * (frame_count + 1)
    best[0] = 0.0
    back_words = [[None] * (frame_count + 1) for _ in range(word_count)]
    for word_index in range(word_count):
        table = word_max[word_index]
        candidates = candidate_tables[word_index]
        current = [NEG_INF] * (frame_count + 1)
        current_back = [None] * (frame_count + 1)
        for start in range(frame_count + 1):
            base = best[start]
            if base == NEG_INF:
                continue
            for end in range(start + 1, frame_count + 1):
                for candidate_index, entry in enumerate(candidates):
                    value = entry["max"][start][end]
                    if value == NEG_INF:
                        continue
                    total = base + entry["candidate"]["log_prior"] + value
                    if total > current[end]:
                        current[end] = total
                        current_back[end] = (start, candidate_index)
        best = current
        back_words[word_index] = current_back

    best_score = best[frame_count]
    if best_score == NEG_INF:
        # Mathematically this cannot happen if the log partition is finite.
        return {
            "status": "no_path",
            "log_partition": NEG_INF,
            "path": _empty_path(),
            "best_path": _empty_path(),
            "interval_tables": None,
            "score": NEG_INF,
            "log_score": NEG_INF,
            "boundaries": [],
            "states": [],
        }

    choices = []
    end = frame_count
    for word_index in range(word_count - 1, -1, -1):
        entry = back_words[word_index][end]
        if entry is None:
            raise RuntimeError("internal error: Viterbi backtracking failed")
        start, candidate_index = entry
        choices.append((word_index, start, end, candidate_index))
        end = start
    if end != 0:
        raise RuntimeError("internal error: Viterbi path does not start at frame 0")
    choices.reverse()

    word_reports = []
    global_phone_intervals = []
    pronunciation_ids = []
    pronunciations = []
    states = [None] * frame_count
    recomputed_score = 0.0
    global_phone_index = 0
    boundaries = [0]
    for word_index, start, end, candidate_index in choices:
        entry = candidate_tables[word_index][candidate_index]
        candidate = entry["candidate"]
        phones = candidate["phones"]
        intervals = _reconstruct_phone_intervals(phones, log_emit, start, end)
        recomputed_score += candidate["log_prior"]
        phone_reports = []
        for phone, (phone_start, phone_end) in zip(phones, intervals):
            phone_log = sum(log_emit[phone][frame] for frame in range(phone_start, phone_end))
            recomputed_score += phone_log
            phone_reports.append(
                {
                    "phone": phone,
                    "start": phone_start,
                    "end": phone_end,
                    "state": global_phone_index,
                }
            )
            global_phone_intervals.append(
                {"phone": phone, "start": phone_start, "end": phone_end,
                 "state": global_phone_index}
            )
            for frame in range(phone_start, phone_end):
                states[frame] = global_phone_index
            global_phone_index += 1
        word_reports.append(
            {
                "index": word_index,
                "word": words_list[word_index],
                "start": start,
                "end": end,
                "pronunciation_id": candidate["id"],
                "phones": list(phones),
                "phone_intervals": phone_reports,
            }
        )
        pronunciation_ids.append(candidate["id"])
        pronunciations.append(list(phones))
        boundaries.append(end)

    # The path score from the Viterbi recursion and the score obtained from the
    # reconstructed intervals should agree up to floating-point roundoff.  Use
    # the recursion's value so "best" is exactly what was optimised.
    path = {
        "log_score": best_score,
        "words": word_reports,
        "word_intervals": [[item["start"], item["end"]] for item in word_reports],
        "phone_intervals": global_phone_intervals,
        "pronunciation_ids": pronunciation_ids,
        "pronunciations": pronunciations,
        "states": states,
        "recomputed_log_score": recomputed_score,
    }

    # Joint boundary tables for every consecutive word group i..j.
    interval_tables = {}
    for first in range(word_count):
        middle = [row[:] for row in word_sum[first]]
        for last in range(first, word_count):
            if last > first:
                previous = middle
                current = [
                    [NEG_INF] * (frame_count + 1) for _ in range(frame_count + 1)
                ]
                table = word_sum[last]
                for start in range(frame_count + 1):
                    row = previous[start]
                    for split in range(start + 1, frame_count + 1):
                        left = row[split]
                        if left == NEG_INF:
                            continue
                        segment_row = table[split]
                        for end in range(split + 1, frame_count + 1):
                            right = segment_row[end]
                            if right != NEG_INF:
                                current[start][end] = logadd(
                                    current[start][end], left + right
                                )
                middle = current

            joint = {}
            for start in range(frame_count + 1):
                prefix = forward[first][start]
                if prefix == NEG_INF:
                    continue
                segment_row = middle[start]
                for end in range(start + 1, frame_count + 1):
                    segment = segment_row[end]
                    if segment == NEG_INF:
                        continue
                    suffix = backward[last + 1][end]
                    if suffix == NEG_INF:
                        continue
                    joint[(start, end)] = prefix + segment + suffix
            normaliser = _logsumexp(joint.values())
            rows = []
            for start in range(frame_count + 1):
                for end in range(start + 1, frame_count + 1):
                    value = joint.get((start, end))
                    probability = 0.0 if value is None else math.exp(value - normaliser)
                    rows.append(
                        {"start": start, "end": end, "probability": probability}
                    )
            interval_tables[f"{first}:{last}"] = rows

    return {
        "status": "ok",
        "log_partition": log_partition,
        "path": path,
        "best_path": path,
        "interval_tables": interval_tables,
        "score": best_score,
        "log_score": best_score,
        "boundaries": boundaries,
        "states": states,
    }
