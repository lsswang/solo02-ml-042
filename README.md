# Speech annotation and subtitle timing

Python 3.11, standard library. Run `python3 demo.py` for the existing ordinary path. `alignment.align` currently uses the first dictionary pronunciation and returns a Viterbi path. `subtitles.plan` currently replaces interval distributions with their mean boundaries and greedily groups words. The JSON fixture contains short synthetic acoustic probabilities and a pronunciation lexicon; no audio/model download is required.
