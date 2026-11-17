"""Existing point-timing subtitle segmentation for a small editing tool."""


def plan(words, interval_tables, policy):
    means = {}
    for key, rows in interval_tables.items():
        means[key] = (sum(row['start'] * row['probability'] for row in rows),
                      sum(row['end'] * row['probability'] for row in rows))
    cues = []
    first = 0
    previous_end = 0
    while first < len(words):
        last = first
        while last + 1 < len(words) and len(' '.join(words[first:last + 2])) <= policy['max_chars']:
            last += 1
        start, end = means[f'{first}:{last}']
        start = max(previous_end, round(start))
        end = max(start + 1, round(end))
        cues.append({'first': first, 'last': last, 'start': start, 'end': end,
                     'text': ' '.join(words[first:last + 1])})
        first = last + 1
        previous_end = end
    return {'status': 'ok', 'cues': cues}
