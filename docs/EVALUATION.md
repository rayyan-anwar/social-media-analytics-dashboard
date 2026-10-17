# Evaluation

Author: Rayyan Anwar

Real platform data was not available (no network, platform terms), so the evaluation uses a seeded generator with **planted** signals and checks that the analytics recover them. `python scripts/evaluate.py` writes `results/RESULTS.md`.

## Planted-signal recovery (20 seeds, 3000 posts over 60 days; committed run)
| Check | Result |
|---|---:|
| Spike day detected | 1.00 |
| False spike days per dataset | 0.15 |
| Trend direction correct | 1.00 |
| Best hour ranked first / in top 3 | 1.00 / 1.00 |
| Highest-engagement hashtag correct | 1.00 |
| Lexicon sentiment accuracy vs generator labels | 0.971 ± 0.003 |

## Input validation
With 50 bad timestamps, 70 negative/non-numeric counts and 30 duplicate ids injected, the report matched the injected counts exactly.

## Runtime
Load + validate + analyse: 10,000 rows in about 0.1 s and 50,000 rows in about 0.25 s on the 1-core sandbox.

## Caveats
- The planted effects are strong (4x spike, 2x best hour, 2.5x hashtag, 80% growth), so near-perfect recovery shows the code works, **not** that it will find subtle effects in real data.
- The sentiment generator and lexicon were written by the same author; 97% is a sanity check, not accuracy on real posts (sarcasm, emojis, slang and other languages are not handled).
- A mid-development test caught a real bug (valid rows with mixed timestamp formats were dropped); it is fixed and covered by a test.
