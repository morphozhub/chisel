# SN17 Competition 3 miner (closed-source model phase)

Entry point: `main.py` → `generate(image, seed, witness, renderer)`.
Based on the organizer's `miner-reference` (404-Repo/404-gen-subnet, branch
`feat/witness-public-release` @ 980eb3ef49975668fb080d6904cb6e35eb5358b1) with
reliability changes in `miner_reference/pipeline.py`:

- provider quota / rate-limit / auth / 5xx errors abort the prompt (`ProviderUnavailable`)
  instead of silently producing `None`, so the prompt stays recoverable;
- deterministic fallbacks when a selector call fails: first valid candidate
  (round 1) or the round-1 winner (final).

All model calls go through the supplied witness client; rendering through the
supplied renderer. No network, no randomness beyond the supplied seed.
