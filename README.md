# Speech annotation and subtitle timing

Python 3.11, standard library only (no NumPy/SciPy needed; no audio models or
corpora are downloaded).  Everything here is synthetic.

```bash
python3 demo.py        # ordinary alignment, ambiguity, external tables,
                       # combined align->plan, stale-table control, infeasible
python3 -m unittest -v # brute-force oracles + validation/edge-case tests
python3 tests.py       # same suite
```

## Layout

| File | Purpose |
|---|---|
| `alignment.py` | pronunciation-uncertain forced alignment, log partition, Viterbi path, joint word-boundary tables |
| `subtitles.py` | globally optimal subtitle planning that consumes joint boundary tables |
| `common.py` | validation and `logadd` helpers, `InputError` |
| `oracle.py` | independent brute-force enumerators used only by tests/demos |
| `tests.py` | oracle cross-checks corpus (fuzz too, see below) |
| `fixtures/cases.json` | existing synthetic emissions, lexicon and policies |
| `demo.py` | runnable end-to-end demonstrations |

`align(words, emissions, lexicon)` and `plan(words, interval_tables, policy)` keep
their original names and roles.  `align` accepts the old single-pronunciation
lexicon shape unchanged; `plan` still accepts the old
`{"start": ..., "end": ..., "probability": ...}` row lists (see below for the
other accepted spellings).

## Alignment model

A legal path chooses one pronunciation candidate per word (distinct ids are
kept even when their phone strings coincide), concatenates the phone strings,
and partitions all `T` frames into one non-empty contiguous block per phone.
Path mass is `prod(prior) * prod(frame emission of the owning phone)`; there
are no duration or transition terms.  `Z` is the sum over *all* legal paths,
so `log_partition` is not a Viterbi score and the returned `path.log_score` is
the mass of the single best path.

Implementation: for each candidate and each `(start, end)` frame span we build
a log-space sum table and a log-space max table (segmentation of the span into
the candidate's phones).  Words are then combined with a forward sum, a
backward sum and a Viterbi recursion over word boundaries.  Because every
emission is at least `1e-12` when non-zero, all arithmetic is in `log` space and
legal paths are never lost to underflow; `no_path` means literally no legal
path (for example fewer frames than phones), not a tiny probability.

`interval_tables["i:j"]` contains a row for *every* legal `0 <= start < end
<= T`, including zero-probability ones.  Each row's value is
`P(word i starts at start, word j ends at end)`, computed as
`forward(i, start) * segment_sum(i..j, start, end) * backward(j+1, end) / Z`
with a segment sum that itself sums over all pronunciations and all internal
word boundaries.  Consequently the tables of different groups are consistent
marginals of one global distribution, and each sums to 1.  Nothing is
factorised into a product of one-boundary marginals.

## Subtitle model

Each cue covers a consecutive word group `i..j` and displays it in `[s, e)`.
Cues must cover every word once, in order, with no overlap (gaps allowed).
Layout is one or two lines, split only between words; each line is at most
`max_chars` Unicode scalar values.  The reading-speed constraint uses the
group joined on one line (including single spaces), so breaking lines never
reduces the text a viewer must read.  Duration must lie in
`[min_duration, max_duration]`.

For a cue's own group table,

* coverage is the joint survival mass `sum_{S>=s, E<=e} P(S,E)`, required to be
  `>= coverage - 1e-10`;
* cost is `E[(s-S)^2 + (e-E)^2] + cue_penalty`, computed from the joint second
  moments of `(S, E)`; tables of different groups are never multiplied.

`plan` is an exact dynamic program over (last covered word, previous cue end)
that simultaneously explores all groupings, all integer display intervals and
multi-line layouts; ties may resolve arbitrarily.  If no complete legal plan
exists it returns `status: "infeasible"` (distinct from an unresolved solve —
the DP is exhaustive).

`interval_tables` values may be given as a list of
`{"start", "end", "probability"}` mappings (the original shape), a list of
`(start, end, probability)` triples, or a `{(start, end): probability}` mapping;
group keys are `"i:j"` strings or `(i, j)` tuples.  Tables do not need to come
from this aligner.  Repeated rows for the same boundary are added.

## Validation contract (raises `InputError`)

* Wrong shapes/types, words outside 1..8 (alignment) or 1..48 (subtitles).
* Emission rows that omit/rename phones, contain non-finite or negative values,
  positive values below `1e-12`, or sum to 0; row sums must be 1 within
  `1e-12` and are then renormalised by the actual sum.
* Unknown phones, empty phone lists, empty candidate lists, duplicate ids
  within a word, non-positive/non-finite priors, priors not summing to 1 within
  `1e-12` (then normalised by the actual sum).
* Candidate/phone counts outside 1..3 / 1..4, phone table outside 2..8.
* Table rows with `start >= end`, out-of-range frames, negative/non-finite
  probability, or total mass not 1 within `1e-12` (then normalised).
* Policy fields outside their documented ranges, `min_duration >
  max_duration`, missing tables for any group, non-string/empty words, and
  unpaired UTF-16 surrogates in words or phone symbols.

## Verification

`tests.py` and the ad-hoc fuzz scripts in this session compare the solvers to
`oracle.py`, which enumerates every pronunciation choice, every frame
partition, every word grouping, every display interval and every line break.
The random cross-checks (300 alignment cases up to 6 frames/3 words, 500 more
with repeated phones, 300–400 subtitle cases) agree to about `1e-15` for
`log_partition`, best-path score and every joint table entry, and the subtitle
objective agrees to about `1e-14` with every returned cue independently
re-checked against the hard constraints.

Focused cases cover: repeated phones as separate states; zero-probability
frames with an overall legal path; frame deficit and all-zero evidence
(`no_path`, `-inf`, empty path, `None` tables); pronunciation competition;
joint-vs-marginal coverage; line wrapping; Unicode scalar widths; reading
speed; adjacent-cue overlap; gaps between cues; external hand-written tables;
and genuine infeasibility.

## `demo.py` results (synthetic, not an ASR accuracy claim)

* **ordinary** — `cat`/`go`, one pronunciation each: the old Viterbi result and
  a one-cue subtitle plan are unchanged.
* **pronunciation_uncertainty** — the fixture's `red` has a one-phone and a
  two-phone pronunciation.  Under `before` the best path uses `red:0`
  (word intervals `[0,1],[1,3],[3,6]`); under `after` it uses `red:1`
  (`[0,3],[3,4],[4,6]`), and the printed joint tables differ.
* **external_tables** — a hand-written 3-word, T=9 case with no acoustics:
  the planner splits it into two cues (`ten green` over `[0,6)`, `vine` over
  `[6,9)`), each with coverage 1.0.
* **combined** — `align(...)["interval_tables"]` fed straight to `plan` for
  both acoustics.  The `before` plan is `red cat` on `[0,3)` and `go` on
  `[3,6)`; the `after` plan moves the first cue to `[0,4)` and `go` to `[4,6)`,
  so the acoustic change really changes the subtitle timing.
* **stale_table_control** — the plan produced from `map_counterexample` tables,
  evaluated against the current `before` tables, has a cue whose current
  coverage is `0.0004`, far below the required `0.7`; the current plan passes.
  This shows the consumer uses the tables it is given, not stale boundaries.
* **infeasible** — forcing `max_duration=2` over 3 words makes a complete
  cover impossible, and `plan` returns `infeasible` rather than inventing cues.

## Computational/storage trade-offs

* **Alignment summation** — log-space forward/backward is `O(W * C * T^2)` time
  and `O(T^2)` storage per group table.  The alternative (enumerating paths)
  is exponential and also numerically fragile.  All-group tables cost
  `O(W^2 T^2)` storage, which is modest at `T <= 48`, `W <= 8`; recomputing
  each group's inner sum on demand would save memory but repeat work.
* **Joint tables** — computing each group directly as
  `forward * segment * backward` costs `O(T^2)` per group after the
  `O(W T^2)` forward/backward passes.  Materialising per-path counts instead
  would be exponential and would expose the joint only indirectly.
* **Subtitle planning** — the exact DP is `O(W^2 T^2 + W T^2)` with a
  per-group `O(T^2)` coverage table, after the `(W^2)` group tables are
  normalised.  Greedy or expected-rounding heuristics are cheaper but, as the
  fixtures show, can violate coverage or pick the wrong grouping; global DP
  keeps both the grouping and timing optimum.
* **Tolerances** — probability sums use `1e-12`, coverage uses the specified
  `1e-10`, and tests compare solver to oracle at `1e-9` absolute (objective
  checks `1e-8` absolute / `1e-9` relative).  These only absorb
  floating-point roundoff; validation still rejects actual contract violations.
