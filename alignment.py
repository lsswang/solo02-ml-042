"""A small existing single-pronunciation Viterbi aligner."""
import math


def align(words, emissions, lexicon):
    phones = []
    ends = []
    for word in words:
        phones.extend(lexicon[word][0]['phones'])
        ends.append(len(phones) - 1)
    t_count, p_count = len(emissions), len(phones)
    dp = {(0, 0): (math.log(emissions[0][phones[0]]), [0])}
    for t in range(1, t_count):
        for s in range(p_count):
            candidates = []
            for previous in (s, s - 1):
                if (t - 1, previous) in dp:
                    value, path = dp[t - 1, previous]
                    candidates.append((value + math.log(emissions[t][phones[s]]), path + [s]))
            if candidates:
                dp[t, s] = max(candidates, key=lambda item: item[0])
    if (t_count - 1, p_count - 1) not in dp:
        return {'status': 'no_path'}
    score, states = dp[t_count - 1, p_count - 1]
    boundaries = [0]
    for last_state in ends:
        boundaries.append(max(t for t, state in enumerate(states) if state <= last_state) + 1)
    return {'status': 'ok', 'score': score, 'boundaries': boundaries, 'states': states}
