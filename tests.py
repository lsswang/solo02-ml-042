"""Tests: brute-force oracles, validation, edge cases, and the demos.

Run with ``python3 -m unittest -v`` (or ``python3 tests.py``).
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from alignment import align
from common import InputError
from oracle import (
    brute_force_align,
    brute_force_plan,
    brute_force_tables,
    verify_plan_constraints,
)
from subtitles import plan

FIXTURES = Path(__file__).parent / "fixtures" / "cases.json"


def make_emissions(frames, phones, values):
    """Build row-normalised emission dictionaries from a flat list."""

    emissions = []
    for t in range(frames):
        row = {}
        for p_index, phone in enumerate(phones):
            row[phone] = values[t * len(phones) + p_index]
        emissions.append(row)
    return emissions


def normalise(rows):
    out = []
    for row in rows:
        total = sum(row.values())
        out.append({key: value / total for key, value in row.items()})
    return out


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        case = json.loads(FIXTURES.read_text())
        cls.case = case

    def check_case(self, words, emissions, lexicon, tolerance=1e-9):
        result = align(words, emissions, lexicon)
        oracle = brute_force_align(words, emissions, lexicon)
        self.assertEqual(result["status"], oracle["status"])
        if oracle["status"] == "no_path":
            self.assertEqual(result["log_partition"], float("-inf"))
            self.assertEqual(result["path"]["words"], [])
            self.assertIsNone(result["interval_tables"])
            return result, oracle
        self.assertAlmostEqual(result["log_partition"], oracle["log_partition"], delta=tolerance)
        self.assertAlmostEqual(
            result["path"]["log_score"], oracle["best"]["log_score"], delta=tolerance
        )
        self.assertAlmostEqual(
            result["path"]["recomputed_log_score"],
            result["path"]["log_score"],
            delta=tolerance,
        )
        oracle_tables = brute_force_tables(oracle, len(words), len(emissions))
        for key, rows in result["interval_tables"].items():
            got = {(row["start"], row["end"]): row["probability"] for row in rows}
            total = math.fsum(got.values())
            self.assertAlmostEqual(total, 1.0, delta=tolerance)
            for boundaries, expected in oracle_tables[key].items():
                self.assertAlmostEqual(
                    got.get(boundaries, 0.0), expected, delta=tolerance
                )
            # Every legal (start, end) pair is present, including zero mass.
            for start in range(len(emissions) + 1):
                for end in range(start + 1, len(emissions) + 1):
                    self.assertIn((start, end), got)
        return result, oracle

    def test_fixture_before(self):
        self.check_case(self.case["words"], self.case["before"], self.case["lexicon"])

    def test_fixture_after(self):
        result, _ = self.check_case(
            self.case["words"], self.case["after"], self.case["lexicon"]
        )
        # The second acoustics favour the two-phone pronunciation of "red".
        self.assertEqual(result["path"]["pronunciation_ids"][0], "red:1")

    def test_fixture_map_counterexample(self):
        self.check_case(
            self.case["words"], self.case["map_counterexample"], self.case["lexicon"]
        )

    def test_repeated_phone_needs_separate_states(self):
        emissions = normalise(
            [{"a": 0.6, "b": 0.2, "c": 0.2}] * 4
        )
        lexicon = {
            "x": [{"id": "x:rep", "phones": ["a", "a"], "prior": 1.0}],
            "y": [{"id": "y:0", "phones": ["b", "c"], "prior": 1.0}],
        }
        result, oracle = self.check_case(["x", "y"], emissions, lexicon)
        self.assertEqual(result["path"]["words"][0]["phone_intervals"][0]["phone"], "a")
        # Two occurrences of "a" must own two disjoint non-empty intervals.
        intervals = result["path"]["words"][0]["phone_intervals"]
        self.assertEqual([(iv["start"], iv["end"]) for iv in intervals], [(0, 1), (1, 2)])

    def test_prior_competition_and_partition(self):
        emissions = normalise(
            [
                {"a": 0.5, "b": 0.5},
                {"a": 0.5, "b": 0.5},
                {"a": 0.5, "b": 0.5},
            ]
        )
        lexicon = {
            "w": [
                {"id": "w:a", "phones": ["a"], "prior": 0.25},
                {"id": "w:ab", "phones": ["a", "b"], "prior": 0.75},
            ]
        }
        result, oracle = self.check_case(["w"], emissions, lexicon)
        self.assertGreaterEqual(
            result["path"]["log_score"], oracle["best"]["log_score"] - 1e-9
        )

    def test_zero_evidence_paths_and_underflow_stability(self):
        # The only viable path has mass 1e-12 * 1e-12 per frame, i.e. far below
        # the smallest normal double; log-space summation must keep it alive.
        tiny = 1e-12
        rest = 0.6
        emissions = normalise(
            [{"a": tiny, "b": rest}, {"a": tiny, "b": rest}, {"a": tiny, "b": rest}]
        )
        lexicon = {"w": [{"id": "w:a", "phones": ["a"], "prior": 1.0}]}
        result, oracle = self.check_case(["w"], emissions, lexicon)
        self.assertNotEqual(result["status"], "no_path")

    def test_no_path_frame_deficit(self):
        emissions = normalise([{"a": 0.5, "b": 0.5}] * 2)
        lexicon = {"w": [{"id": "w:0", "phones": ["a", "b", "a"], "prior": 1.0}]}
        result, oracle = self.check_case(["w"], emissions, lexicon)
        self.assertEqual(result["status"], "no_path")
        self.assertEqual(result["log_partition"], float("-inf"))
        self.assertIsNone(result["interval_tables"])

    def test_all_zero_for_required_phone_is_no_path(self):
        emissions = normalise(
            [{"a": 0.0, "b": 1.0}, {"a": 0.0, "b": 1.0}, {"a": 0.0, "b": 1.0}]
        )
        lexicon = {"w": [{"id": "w:0", "phones": ["a"], "prior": 1.0}]}
        result, oracle = self.check_case(["w"], emissions, lexicon)
        self.assertEqual(result["status"], "no_path")

    def test_input_errors(self):
        emissions = normalise([{"a": 0.5, "b": 0.5}] * 3)
        good = {"w": [{"id": "w:0", "phones": ["a"], "prior": 1.0}]}

        with self.assertRaises(InputError):
            align([], emissions, good)  # no words
        with self.assertRaises(InputError):
            align(["w"], [], good)  # no frames
        with self.assertRaises(InputError):
            align(["w"], normalise([{"a": 1.0}] * 3), good)  # one phone only
        with self.assertRaises(InputError):
            align(["w"], normalise([{"a": 0.5, "b": 0.5}] * 3), {})  # missing word
        with self.assertRaises(InputError):
            align(
                ["w"],
                emissions,
                {"w": [{"id": "w:0", "phones": ["z"], "prior": 1.0}]},
            )  # unknown phone
        with self.assertRaises(InputError):
            align(
                ["w"],
                emissions,
                {"w": [{"id": "w:0", "phones": [], "prior": 1.0}]},
            )  # empty phones
        with self.assertRaises(InputError):
            align(
                ["w"],
                emissions,
                {
                    "w": [
                        {"id": "same", "phones": ["a"], "prior": 0.5},
                        {"id": "same", "phones": ["b"], "prior": 0.5},
                    ]
                },
            )  # duplicate id
        with self.assertRaises(InputError):
            align(
                ["w"],
                emissions,
                {"w": [{"id": "w:0", "phones": ["a"], "prior": 0.5}]},
            )  # priors do not sum to one
        with self.assertRaises(InputError):
            align(
                ["w"],
                emissions,
                {"w": [{"id": "w:0", "phones": ["a"], "prior": 0.0}]},
            )  # non-positive prior
        bad_rows = [{"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.4}, {"a": 0.5, "b": 0.5}]
        with self.assertRaises(InputError):
            align(["w"], bad_rows, good)  # row sum not one
        with self.assertRaises(InputError):
            align(
                ["w"],
                [{"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5000001}],
                good,
            )  # row sum off by 1e-7
        with self.assertRaises(InputError):
            align(
                ["w"],
                [{"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.4999999999}],
                good,
            )
        with self.assertRaises(InputError):
            align(
                ["w"],
                normalise([{"a": -0.1, "b": 0.7}] * 3),
                good,
            )  # negative probability
        with self.assertRaises(InputError):
            align(
                ["w"],
                [{"a": 0.0, "b": 0.0}] * 3,
                good,
            )  # no positive mass
        with self.assertRaises(InputError):
            align(
                ["w"],
                normalise([{"a": 1e-15, "b": 1.0}] * 3),
                good,
            )  # positive below 1e-12
        with self.assertRaises(InputError):
            align(
                ["w", "w"],
                emissions,
                {
                    "w": [
                        {"id": "w:0", "phones": ["a"], "prior": 1.0},
                        {"id": "w:0", "phones": ["b"], "prior": 0.0},
                    ]
                },
            )
        # Row sums are normalised by the actual sum; 0.999999999999 fits.
        tolerated = [{"a": 0.4999999999995, "b": 0.5}] * 3
        self.assertEqual(align(["w"], tolerated, good)["status"], "ok")

    def test_unpaired_surrogate_word_is_input_error(self):
        emissions = normalise([{"a": 0.5, "b": 0.5}] * 3)
        bad = "\ud800"
        with self.assertRaises(InputError):
            align([bad], emissions, {bad: [{"id": "w:0", "phones": ["a"], "prior": 1.0}]})


class SubtitleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        case = json.loads(FIXTURES.read_text())
        cls.case = case
        cls.alignment_result = align(
            case["words"], case["map_counterexample"], case["lexicon"]
        )
        # Independent oracle needs plain dictionaries.
        matrix = {
            key: {(row["start"], row["end"]): row["probability"] for row in rows}
            for key, rows in cls.alignment_result["interval_tables"].items()
        }
        cls.matrix = matrix

    def check_plan(self, words, matrix, policy, tolerance=1e-8):
        result = plan(words, matrix, policy)
        oracle = brute_force_plan(words, matrix, policy)
        self.assertEqual(result["status"], oracle["status"])
        if oracle["status"] == "infeasible":
            self.assertEqual(result["cues"], [])
            return result, oracle
        self.assertTrue(
            verify_plan_constraints(words, matrix, policy, result["cues"])
        )
        self.assertAlmostEqual(
            result["total_cost"], oracle["total_cost"], delta=tolerance
        )
        self.assertAlmostEqual(
            result["recomputed_cost"], result["total_cost"], delta=tolerance
        )
        for cue in result["cues"]:
            self.assertAlmostEqual(cue["cost"], (
                cue["start"] ** 2 - 2.0 * cue["start"] * sum(
                    p * s for (s, e), p in matrix[f"{cue['first']}:{cue['last']}"].items()
                ) + sum(p * s * s for (s, e), p in matrix[f"{cue['first']}:{cue['last']}"].items())
                + cue["end"] ** 2 - 2.0 * cue["end"] * sum(
                    p * e for (s, e), p in matrix[f"{cue['first']}:{cue['last']}"].items()
                ) + sum(p * e * e for (s, e), p in matrix[f"{cue['first']}:{cue['last']}"].items())
                + policy["cue_penalty"]), delta=tolerance
            )
        return result, oracle

    def test_fixture_planner(self):
        self.check_plan(self.case["words"], self.matrix, self.case["coupling_policy"])

    def test_two_line_wrapping(self):
        words = ["alpha", "beta", "gamma"]
        tables = {
            "0:0": {(2, 3): 1.0},
            "1:1": {(3, 7): 1.0},
            "2:2": {(7, 10): 1.0},
            "0:1": {(2, 7): 1.0},
            "1:2": {(3, 10): 1.0},
            "0:2": {(2, 10): 1.0},
        }
        policy = {
            "T": 10,
            "max_chars": 8,
            "min_duration": 1,
            "max_duration": 6,
            "chars_per_frame": 2.0,
            "coverage": 0.7,
            "cue_penalty": 0.0,
        }
        result, oracle = self.check_plan(words, tables, policy)
        self.assertEqual(result["status"], "ok")
        # "alpha beta" has 10 scalars and cannot fit on one 8-char line.
        cue = [c for c in result["cues"] if c["first"] == 0][0]
        self.assertEqual(cue["lines"], ["alpha", "beta"])
        self.assertEqual(cue["start"], 2)
        self.assertEqual(cue["end"], 7)

    def test_coverage_uses_joint_not_marginals(self):
        # Joint table over (S, E): (0,2) with 0.2, (1,5) with 0.35, (2,4) with
        # 0.45.  For the interval [1,4) the joint survival mass is only 0.45
        # (the (2,4) atom), while the product of the marginal probabilities
        # P(S>=1)*P(E<=4) = 0.8 * 0.65 = 0.52 would wrongly look sufficient.
        table = {(0, 2): 0.2, (1, 5): 0.35, (2, 4): 0.45}
        marginals_start = {0: 0.2, 1: 0.35, 2: 0.45}
        marginals_end = {2: 0.2, 4: 0.45, 5: 0.35}
        product = (
            sum(p for s, p in marginals_start.items() if s >= 1)
            * sum(p for e, p in marginals_end.items() if e <= 4)
        )
        joint = sum(p for (s, e), p in table.items() if s >= 1 and e <= 4)
        self.assertGreater(product, 0.5)
        self.assertLess(joint, 0.5)

        policy = {
            "T": 5,
            "max_chars": 1,
            "min_duration": 1,
            "max_duration": 3,
            "chars_per_frame": 4.0,
            "coverage": 0.5,
            "cue_penalty": 0.0,
        }
        result, oracle = self.check_plan(["w"], {"0:0": table}, policy)
        # No display interval of at most 3 frames has joint coverage >= 0.5.
        self.assertEqual(result["status"], "infeasible")

        # A correlated table with the same marginals used to be confused with
        # its marginal product; here the atom (0,2)+(2,4) yields coverage 1.0.
        correlated = {"0:0": {(0, 2): 0.4, (2, 4): 0.6}}
        res, _ = self.check_plan(["w"], correlated, policy)
        self.assertEqual(res["status"], "ok")

    def test_adjacent_cues_must_not_overlap(self):
        words = ["ab", "cd", "ef"]
        tables = {
            "0:0": {(0, 2): 1.0},
            "1:1": {(0, 2): 1.0},
            "2:2": {(0, 1): 1.0},
            "0:1": {(0, 2): 1.0},
            "1:2": {(0, 1): 1.0},
            "0:2": {(0, 2): 1.0},
        }
        policy = {
            "T": 2,
            "max_chars": 2,
            "min_duration": 1,
            "max_duration": 2,
            "chars_per_frame": 4.0,
            "coverage": 1.0,
            "cue_penalty": 0.0,
        }
        # Three 2-char words cannot share one two-line cue (max_chars=2), and
        # each of words 0 and 1 must span [0,2); the third cue then has to
        # overlap, so no legal plan exists.
        result, oracle = self.check_plan(words, tables, policy)
        self.assertEqual(result["status"], "infeasible")

    def test_gap_between_cues_is_allowed(self):
        words = ["a", "b"]
        tables = {
            "0:0": {(0, 1): 1.0},
            "1:1": {(2, 3): 1.0},
            "0:1": {(0, 1): 0.0, (0, 3): 1.0, (2, 3): 0.0},
        }
        policy = {
            "T": 4,
            "max_chars": 1,
            "min_duration": 1,
            "max_duration": 2,
            "chars_per_frame": 4.0,
            "coverage": 1.0,
            "cue_penalty": 0.0,
        }
        result, oracle = self.check_plan(words, tables, policy)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([(c["start"], c["end"]) for c in result["cues"]], [(0, 1), (2, 3)])

    def test_reading_speed_forces_wider_display(self):
        words = ["abcdefghij"]
        narrow = {"0:0": {(0, 1): 1.0}}
        policy = {
            "T": 3,
            "max_chars": 40,
            "min_duration": 1,
            "max_duration": 3,
            "chars_per_frame": 4.0,
            "coverage": 1.0,
            "cue_penalty": 0.0,
        }
        # 10 chars <= 4.0 * 3 frames.
        self.assertEqual(plan(words, narrow, policy)["status"], "ok")
        slow = dict(policy, chars_per_frame=0.5)
        # 10 chars at 0.5 chars/frame needs 20 frames, more than T=3.
        self.assertEqual(plan(words, narrow, slow)["status"], "infeasible")

    def test_reading_speed_infeasible(self):
        words = ["abcdefghij"]
        tables = {"0:0": {(0, 1): 1.0}}
        policy = {
            "T": 2,
            "max_chars": 40,
            "min_duration": 1,
            "max_duration": 2,
            "chars_per_frame": 0.5,
            "coverage": 1.0,
            "cue_penalty": 0.0,
        }
        # 10 chars > 0.5 * 2 frames.
        result, oracle = self.check_plan(words, tables, policy)
        self.assertEqual(result["status"], "infeasible")

    def test_word_width_counts_unicode_scalars_and_spaces(self):
        words = ["\U0001F600", "\U0001F600"]  # 2 emoji, 2 scalars total + 1 space
        tables = {
            "0:0": {(0, 1): 1.0},
            "1:1": {(1, 2): 1.0},
            "0:1": {(0, 2): 1.0},
        }
        policy = {
            "T": 2,
            "max_chars": 2,
            "min_duration": 1,
            "max_duration": 2,
            "chars_per_frame": 1.5,
            "coverage": 1.0,
            "cue_penalty": 0.0,
        }
        result, oracle = self.check_plan(words, tables, policy)
        self.assertEqual(result["status"], "ok")
        # "😀 😀" is 3 scalars, wider than max_chars=2, so two lines are used.
        self.assertEqual(result["cues"][0]["lines"], ["\U0001F600", "\U0001F600"])

    def test_external_tables_need_no_acoustics(self):
        # A hand-written distribution unrelated to any acoustic model.
        words = ["x", "y"]
        tables = {
            "0:0": {(0, 2): 0.7, (1, 3): 0.3},
            "1:1": {(2, 4): 0.6, (3, 5): 0.4},
            "0:1": {(0, 2): 0.0, (0, 4): 0.5, (2, 4): 0.2, (1, 5): 0.3},
        }
        policy = {
            "T": 5,
            "max_chars": 3,
            "min_duration": 2,
            "max_duration": 5,
            "chars_per_frame": 0.5,
            "coverage": 0.8,
            "cue_penalty": 0.2,
        }
        self.check_plan(words, tables, policy)

    def test_infeasible_is_not_unresolved(self):
        words = ["a", "b"]
        tables = {
            "0:0": {(0, 1): 1.0},
            "1:1": {(1, 2): 1.0},
            "0:1": {(0, 2): 1.0},
        }
        policy = {
            "T": 2,
            "max_chars": 1,
            "min_duration": 1,
            "max_duration": 1,
            "chars_per_frame": 0.5,
            "coverage": 1.0,
            "cue_penalty": 0.0,
        }
        # Two cues each need one frame but both boundaries locked at 0:1/1:2;
        # with no overlap allowed the plan is mathematically impossible.
        result, _ = self.check_plan(words, tables, policy)
        self.assertEqual(result["status"], "infeasible")
        self.assertEqual(result["cues"], [])

    def test_input_errors(self):
        words = ["a", "b"]
        good = {
            "0:0": {(0, 1): 1.0},
            "1:1": {(1, 2): 1.0},
            "0:1": {(0, 2): 1.0},
        }
        policy = {
            "T": 2,
            "max_chars": 2,
            "min_duration": 1,
            "max_duration": 2,
            "chars_per_frame": 1.0,
            "coverage": 1.0,
            "cue_penalty": 0.0,
        }
        with self.assertRaises(InputError):
            plan([], good, policy)
        with self.assertRaises(InputError):
            plan(["a"], {"0:0": {(0, 1): 1.0}, "0:1": {(0, 1): 1.0}}, policy)  # missing 1:1
        with self.assertRaises(InputError):
            plan(words, dict(good, **{"0:0": {(0, 0): 1.0}}), policy)  # start >= end
        with self.assertRaises(InputError):
            plan(words, dict(good, **{"0:0": {(0, 3): 1.0}}), policy)  # out of range
        with self.assertRaises(InputError):
            plan(words, dict(good, **{"0:0": {(0, 1): 1.0, (1, 2): 0.1}}), policy)
        with self.assertRaises(InputError):
            plan(words, dict(good, **{"0:0": {(0, 1): -0.1, (1, 2): 1.1}}), policy)
        with self.assertRaises(InputError):
            plan(words, dict(good, **{"0:0": {}}), policy)
        for field, bad in [
            ("T", 0),
            ("T", 49),
            ("max_chars", 0),
            ("min_duration", 0),
            ("max_duration", 3),
            ("min_duration", 2),
        ]:
            broken = dict(policy, **{field: bad})
            if field == "min_duration":
                broken["max_duration"] = bad - 1
            with self.assertRaises(InputError):
                plan(words, good, broken)
        with self.assertRaises(InputError):
            plan(words, good, dict(policy, chars_per_frame=0.4))
        with self.assertRaises(InputError):
            plan(words, good, dict(policy, chars_per_frame=25.0))
        with self.assertRaises(InputError):
            plan(words, good, dict(policy, coverage=0.4))
        with self.assertRaises(InputError):
            plan(words, good, dict(policy, cue_penalty=-1.0))
        with self.assertRaises(InputError):
            plan(words, good, dict(policy, cue_penalty=21.0))
        with self.assertRaises(InputError):
            plan(words, good, {k: v for k, v in policy.items() if k != "T"})
        with self.assertRaises(InputError):
            plan(words, {"0:0": "nonsense", "1:1": {(1, 2): 1.0}, "0:1": {(0, 2): 1.0}}, policy)
        with self.assertRaises(InputError):
            plan(["\ud800", "b"], good, policy)
        # Duplicate rows are added, not silently dropped.
        duplicated = {
            "0:0": {(0, 1): 0.6, (0, 2): 0.2, (1, 2): 0.2},
            "1:1": {(1, 2): 1.0},
            "0:1": {(0, 2): 1.0},
        }
        doubled = {
            "0:0": [(0, 1, 0.3), (0, 2, 0.2), (1, 2, 0.2), (0, 1, 0.3)],
            "1:1": {(1, 2): 1.0},
            "0:1": {(0, 2): 1.0},
        }
        self.assertEqual(
            plan(words, duplicated, policy)["status"],
            plan(words, doubled, policy)["status"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
