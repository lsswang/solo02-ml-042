"""Independent brute-force oracles used only by the test-suite.

These implementations deliberately share no dynamic-programming machinery with
``alignment`` or ``subtitles``.  They enumerate every legal path / cue plan so
the fast solvers can be checked against the mathematical definition rather than
against themselves.
"""

from __future__ import annotations

import itertools
import math

from common import scalar_len


def _enumerate_candidate_segmentations(frame_count, phone_count):
    """All compositions of ``frame_count`` into ``phone_count`` positive parts."""

    def rec(remaining_frames, remaining_phones, prefix):
        if remaining_phones == 1:
            if remaining_frames >= 1:
                yield tuple(prefix + [remaining_frames])
            return
        for first in range(1, remaining_frames - remaining_phones + 2):
            yield from rec(
                remaining_frames - first, remaining_phones - 1, prefix + [first]
            )

    yield from rec(frame_count, phone_count, [])


def brute_force_align(words, emissions, lexicon):
    """Enumerate every pronunciation choice and every frame partition."""

    phones = sorted(emissions[0].keys())
    frames = len(emissions)
    # Normalise rows exactly as the specification requires.
    row_sums = [sum(row[p] for p in phones) for row in emissions]
    log_emit = [
        {
            p: (math.log(emissions[t][p] / row_sums[t]) if emissions[t][p] > 0 else -math.inf)
            for p in phones
        }
        for t in range(frames)
    ]

    levelled = []
    for word in words:
        options = []
        for candidate in lexicon[word]:
            prior_total = sum(c["prior"] for c in lexicon[word])
            options.append(
                (
                    candidate["id"],
                    list(candidate["phones"]),
                    math.log(candidate["prior"] / prior_total),
                )
            )
        levelled.append(options)

    paths = []
    # A path is fully determined by one candidate per word plus a segmentation
    # per candidate; enumerate those directly.
    candidate_choices = list(itertools.product(*[range(len(o)) for o in levelled]))
    for choice in candidate_choices:
        selected = [levelled[w][choice[w]] for w in range(len(words))]
        phone_total = sum(len(item[1]) for item in selected)
        if phone_total > frames:
            continue
        for segmentation in _enumerate_candidate_segmentations(frames, phone_total):
            log_score = 0.0
            ok = True
            frame = 0
            intervals = []
            word_intervals = []
            cursor = 0
            for word_index, (cand_id, cand_phones, log_prior) in enumerate(selected):
                log_score += log_prior
                word_start = frame
                for phone, length in zip(
                    cand_phones,
                    segmentation[cursor:cursor + len(cand_phones)],
                ):
                    for t in range(frame, frame + length):
                        log_score += log_emit[t][phone]
                    intervals.append((phone, frame, frame + length))
                    frame += length
                cursor += len(cand_phones)
                word_intervals.append((word_start, frame))
            if frame != frames or not ok:
                continue
            paths.append(
                {
                    "ids": tuple(item[0] for item in selected),
                    "log_score": log_score,
                    "intervals": intervals,
                    "word_intervals": word_intervals,
                }
            )

    if not paths:
        return {"status": "no_path", "paths": []}
    finite_paths = [p for p in paths if math.isfinite(p["log_score"])]
    if not finite_paths:
        return {"status": "no_path", "paths": []}
    log_partition = math.log(
        math.fsum(math.exp(p["log_score"]) for p in finite_paths)
    )
    best = max(finite_paths, key=lambda p: p["log_score"])
    return {
        "status": "ok",
        "paths": paths,
        "log_partition": log_partition,
        "best": best,
    }


def brute_force_tables(result, word_count, frame_count):
    """Joint (start of i, end of j) probabilities from the enumerated paths."""

    paths = result["paths"]
    log_partition = result["log_partition"]
    tables = {}
    for first in range(word_count):
        for last in range(first, word_count):
            joint = {}
            for path in paths:
                if not math.isfinite(path["log_score"]):
                    continue
                start = path["word_intervals"][first][0]
                end = path["word_intervals"][last][1]
                mass = path["log_score"] - log_partition
                joint[(start, end)] = joint.get((start, end), 0.0) + math.exp(mass)
            tables[f"{first}:{last}"] = joint
    return tables


def brute_force_plan(words, tables, policy):
    """Enumerate group partitions, display intervals, and line layouts."""

    frame_count = policy["T"]
    word_count = len(words)
    cpf = policy["chars_per_frame"]

    def distributions(first, last):
        return tables[f"{first}:{last}"]

    def layouts(first, last):
        group_words = words[first:last + 1]
        options = []
        one = " ".join(group_words)
        if sum(scalar_len(word) for word in group_words) + len(group_words) - 1 <= policy["max_chars"]:
            options.append((True, [one]))
        for split in range(1, len(group_words)):
            left = " ".join(group_words[:split])
            right = " ".join(group_words[split:])
            if (
                scalar_len(left) <= policy["max_chars"]
                and scalar_len(right) <= policy["max_chars"]
            ):
                options.append((False, [left, right]))
        return options

    def cue_options(first, last):
        dist = distributions(first, last)
        start_expect = sum(s * p for (s, e), p in dist.items())
        end_expect = sum(e * p for (s, e), p in dist.items())
        start_second = sum(s * s * p for (s, e), p in dist.items())
        end_second = sum(e * e * p for (s, e), p in dist.items())
        layout_options = layouts(first, last)
        options = []
        for start in range(frame_count + 1):
            for end in range(start + 1, frame_count + 1):
                duration = end - start
                if not (policy["min_duration"] <= duration <= policy["max_duration"]):
                    continue
                coverage = sum(
                    p for (s, e), p in dist.items() if s >= start and e <= end
                )
                if coverage < policy["coverage"] - 1e-10:
                    continue
                valid_layouts = []
                joined_width = sum(
                    scalar_len(word) for word in words[first:last + 1]
                ) + (last - first)
                if joined_width > cpf * duration:
                    continue
                for single_line, lines in layout_options:
                    valid_layouts.append(lines)
                if not valid_layouts:
                    continue
                cost = (
                    start * start - 2 * start * start_expect + start_second
                    + end * end - 2 * end * end_expect + end_second
                    + policy["cue_penalty"]
                )
                for lines in valid_layouts:
                    options.append((start, end, cost, lines))
        return options

    best = None
    # Enumerate all compositions of the word sequence into consecutive cues.
    def partitions(total):
        if total == 0:
            yield []
            return
        for last in range(total - 1, total):
            pass
        for split in range(1, total + 1):
            for rest in partitions(total - split):
                yield [split] + rest

    for lengths in partitions(word_count):
        first = 0
        group_ranges = []
        for length in lengths:
            group_ranges.append((first, first + length - 1))
            first += length
        per_cue = [cue_options(a, b) for a, b in group_ranges]
        if any(not options for options in per_cue):
            continue
        for combination in itertools.product(*per_cue):
            previous_end = 0
            total = 0.0
            feasible = True
            for start, end, cost, _lines in combination:
                if start < previous_end:
                    feasible = False
                    break
                previous_end = end
                total += cost
            if not feasible:
                continue
            if best is None or total < best[0]:
                best = (total, group_ranges, combination)
    if best is None:
        return {"status": "infeasible"}
    total, group_ranges, combination = best
    cues = []
    for (first, last), (start, end, cost, lines) in zip(group_ranges, combination):
        cues.append(
            {
                "first": first,
                "last": last,
                "start": start,
                "end": end,
                "lines": lines,
                "cost": cost,
            }
        )
    return {"status": "ok", "total_cost": total, "cues": cues}


def verify_plan_constraints(words, tables, policy, cues):
    """Check every hard constraint with tolerant but independent arithmetic."""

    frame_count = policy["T"]
    if sum(cue["end"] - cue["start"] for cue in cues) < 0:
        return False
    cursor = 0
    previous_end = 0
    for cue in cues:
        first, last = cue["first"], cue["last"]
        if first != cursor or last < first:
            return False
        cursor = last + 1
        start, end = cue["start"], cue["end"]
        if not (0 <= start < end <= frame_count):
            return False
        if start < previous_end:
            return False
        previous_end = end
        duration = end - start
        if not (policy["min_duration"] <= duration <= policy["max_duration"]):
            return False
        group_words = words[first:last + 1]
        lines = cue["lines"]
        if len(lines) not in (1, 2):
            return False
        if any(not line for line in lines):
            return False
        words_in_lines = [line.split(" ") for line in lines]
        # The concatenation must equal the group's words; no word may be cut.
        if sum(words_in_lines, []) != group_words:
            return False
        if any(scalar_len(line) > policy["max_chars"] for line in lines):
            return False
        if len(lines) == 1:
            width = sum(scalar_len(word) for word in group_words) + len(group_words) - 1
        else:
            # The reading-speed value is always the group on one line.
            width = sum(scalar_len(word) for word in group_words) + len(group_words) - 1
        if width > policy["chars_per_frame"] * duration:
            return False
        dist = tables[f"{first}:{last}"]
        coverage = math.fsum(
            p for (s, e), p in dist.items() if s >= start and e <= end
        )
        if coverage < policy["coverage"] - 1e-10:
            return False
    return cursor == len(words)
