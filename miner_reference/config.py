"""Committed, replay-safe configuration knobs (part of the admitted source)."""

# Few-shot library mode appended to the generation instructions:
#   "full"    - patterns.md + all examples in library/index.json
#   "compact" - patterns.md only (ablation: cheaper input, no full examples)
#   "off"     - neither
LIBRARY_MODE = "full"

# Extra judge views: one additional render per valid candidate with side / back / top
# cameras on a white background (the organizer judge uses white-background views and
# checks side/back consistency). Costs renders only, no model calls.
EXTRA_VIEWS = True

# Self-consistent selection: run the judge twice with the candidate labels in reversed
# order (request ids 13 / 14). Agreement wins; disagreement keeps the incumbent
# (round 1: the first candidate). Costs +2 model calls per prompt.
JUDGE_SWAP = False

# Model used for the two refinements (experiment: gpt-6-astra vs gpt-6.1-sol).
REFINE_MODEL = "gpt-6-astra"
