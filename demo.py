import json
from pathlib import Path
from alignment import align
from subtitles import plan

case = json.loads((Path(__file__).parent / 'fixtures/cases.json').read_text())
normal = align(['cat', 'go'], [{'k':0.9,'g':0.1},{'k':0.1,'g':0.9},{'k':0.1,'g':0.9}], case['lexicon'])
tables = {'0:0':[{'start':0,'end':1,'probability':1.0}],
          '1:1':[{'start':1,'end':3,'probability':1.0}],
          '0:1':[{'start':0,'end':3,'probability':1.0}]}
print(json.dumps({'alignment':normal,'subtitles':plan(['cat','go'],tables,case['policy'])}, ensure_ascii=False))
