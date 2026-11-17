"""Demonstrations for pronunciation-uncertain alignment and joint subtitle timing.

Run ``python3 demo.py``.  Everything is synthetic: the acoustic probabilities
and the (optional) hand-written boundary tables below are the whole input.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from alignment import align
from subtitles import plan

FIXTURES = Path(__file__).parent / "fixtures" / "cases.json"

NORMAL_EMISSIONS = [
    {"k": 0.9, "g": 0.1},
    {"k": 0.1, "g": 0.9},
    {"k": 0.1, "g": 0.9},
]


def _slim(result, table_preview=False):
    """Small JSON-friendly view of an alignment result."""

    if result["status"] != "ok":
        return {"status": result["status"], "log_partition": result["log_partition"]}
    path = result["path"]
    view = {
        "status": "ok",
        "log_partition": round(result["log_partition"], 12),
        "best": {
            "log_score": round(path["log_score"], 12),
            "pronunciation_ids": path["pronunciation_ids"],
            "word_intervals": path["word_intervals"],
            "phone_intervals": [
                [item["phone"], item["start"], item["end"]]
                for item in path["phone_intervals"]
            ],
        },
    }
    if table_preview:
        view["joint_tables"] = {
            key: sorted(
                [
                    [row["start"], row["end"], round(row["probability"], 9)]
                    for row in rows
                    if row["probability"] > 0.0
                ]
            )
            for key, rows in result["interval_tables"].items()
        }
    return view


def _slim_plan(result):
    if result["status"] != "ok":
        return {"status": result["status"]}
    return {
        "status": "ok",
        "total_cost": round(result["total_cost"], 12),
        "cues": [
            {
                "words": cue["words"],
                "text": cue["text"],
                "start": cue["start"],
                "end": cue["end"],
                "coverage": round(cue["coverage"], 12),
                "cost": round(cue["cost"], 12),
                "lines": cue["lines"],
            }
            for cue in result["cues"]
        ],
    }


def ordinary_alignment(case):
    # Original single-pronunciation demo: both words have exactly one
    # candidate in this lexicon, so the result is the familiar Viterbi path.
    result = align(["cat", "go"], NORMAL_EMISSIONS, case["lexicon"])
    return {
        "alignment": _slim(result),
        "subtitles": _slim_plan(
            plan(
                ["cat", "go"],
                {
                    "0:0": [{"start": 0, "end": 1, "probability": 1.0}],
                    "1:1": [{"start": 1, "end": 3, "probability": 1.0}],
                    "0:1": [{"start": 0, "end": 3, "probability": 1.0}],
                },
                {
                    "T": 3,
                    "max_chars": 7,
                    "min_duration": 1,
                    "max_duration": 3,
                    "chars_per_frame": 4.0,
                    "coverage": 1.0,
                    "cue_penalty": 0.1,
                },
            )
        ),
    }


def pronunciation_uncertainty(case):
    # "red" has one-phone and two-phone pronunciations.  The two acoustic
    # versions change which one wins and move the joint boundary tables.
    out = {}
    for name in ("before", "after"):
        result = align(case["words"], case[name], case["lexicon"])
        out[name] = _slim(result, table_preview=True)
    return out


def external_tables(case):
    # Hand-written group distributions with no acoustic model behind them.
    # Each single-word table is genuinely spread out, but the group tables are
    # correlated, so only the joint table makes a tight cue possible.
    tables = {
        "0:0": [{"start": 0, "end": 2, "probability": 0.5},
                {"start": 1, "end": 3, "probability": 0.5}],
        "1:1": [{"start": 3, "end": 5, "probability": 0.5},
                {"start": 4, "end": 6, "probability": 0.5}],
        "2:2": [{"start": 6, "end": 8, "probability": 0.5},
                {"start": 7, "end": 9, "probability": 0.5}],
        "0:1": [{"start": 0, "end": 4, "probability": 0.6},
                {"start": 1, "end": 5, "probability": 0.4}],
        "1:2": [{"start": 4, "end": 8, "probability": 0.6},
                {"start": 5, "end": 9, "probability": 0.4}],
        "0:2": [{"start": 0, "end": 9, "probability": 1.0}],
    }
    policy = {
        "T": 9,
        "max_chars": 6,
        "min_duration": 2,
        "max_duration": 6,
        "chars_per_frame": 1.5,
        "coverage": 0.7,
        "cue_penalty": 0.5,
    }
    return _slim_plan(plan(["ten", "green", "vine"], tables, policy))


def combined(case):
    # The end-to-end path: align -> interval_tables -> plan, using both
    # acoustic variants and the coupling policy from the fixture.
    out = {}
    for name in ("before", "after"):
        aligned = align(case["words"], case[name], case["lexicon"])
        planned = plan(case["words"], aligned["interval_tables"], case["coupling_policy"])
        out[name] = {
            "log_partition": round(aligned["log_partition"], 12),
            "pronunciation_ids": aligned["path"]["pronunciation_ids"],
            "plan": _slim_plan(planned),
        }
    return out


def stale_table_control(case):
    # Feed the *current* tables to the planner but also evaluate the plan that
    # the old tables would have produced under the current tables.  The latter
    # violates the current coverage requirement, showing the planner really
    # consumes the table it is given.
    old = align(case["words"], case["map_counterexample"], case["lexicon"])
    new = align(case["words"], case["before"], case["lexicon"])

    def coverage(table, first, last, start, end):
        return math.fsum(
            row["probability"]
            for row in table[f"{first}:{last}"]
            if row["start"] >= start and row["end"] <= end
        )

    current = plan(case["words"], new["interval_tables"], case["coupling_policy"])
    stale = plan(case["words"], old["interval_tables"], case["coupling_policy"])

    def report(result, table):
        return [
            {
                "range": f"{cue['first']}:{cue['last']}",
                "start": cue["start"],
                "end": cue["end"],
                "coverage_under_table": round(
                    coverage(table, cue["first"], cue["last"], cue["start"], cue["end"]),
                    12,
                ),
            }
            for cue in result["cues"]
        ]

    return {
        "current_plan": report(current, new["interval_tables"]),
        "stale_plan_checked_against_current": report(
            stale, new["interval_tables"]
        ),
    }


def infeasible_demo(case):
    policy = dict(case["coupling_policy"])
    policy["max_duration"] = 2
    policy["min_duration"] = 1
    # With three words, each cue at most 2 frames and no overlap, the third
    # word cannot be covered by the end of the clip.
    aligned = align(case["words"], case["before"], case["lexicon"])
    return _slim_plan(plan(case["words"], aligned["interval_tables"], policy))


def main():
    case = json.loads(FIXTURES.read_text())
    output = {
        "ordinary": ordinary_alignment(case),
        "pronunciation_uncertainty": pronunciation_uncertainty(case),
        "external_tables": external_tables(case),
        "combined": combined(case),
        "stale_table_control": stale_table_control(case),
        "infeasible": infeasible_demo(case),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
