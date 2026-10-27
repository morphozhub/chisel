"""Committed, replay-safe configuration knobs (part of the admitted source)."""

# Few-shot library mode appended to the generation instructions:
#   "full"    - patterns.md + all examples in library/index.json
#   "compact" - patterns.md only (ablation: cheaper input, no full examples)
#   "off"     - neither
LIBRARY_MODE = "full"
