# Secure Hachi-Co presets

Written by `parameters.sage` (Hachi-Co section); do not edit by hand. The same data, with every derived
quantity, is in `secure.json`, which `hachi-co-params` embeds and re-derives in its tests.

- Command: `sage parameters.sage --hachi-co-only` (run from the repository root; SageMath version 10.9, Release Date: 2026-05-04, started 2026-10-17 14:05).
- Lattice estimator: https://github.com/malb/lattice-estimator, revision `53da5982597709ba0fdf94ea37a84d822310fd84` (an existing checkout given by LATTICE_ESTIMATOR_DIR).
- Cascade: `6_Collaborative_Hachi.tex:265-281` of the draft at commit `625ced1`, with the tight tau and the rules of `docs/m0-spec.md`.
- Fixed choices: lambda = 115, k = 4 (kappa = 2), b = 16, tight tau, the l1-ball challenge set with the smallest omega such that 2^(r+1)/|C| <= 2^-lambda, n_D = 1, t = floor((n - 1)/2).
- n_A and q: MSIS for A (rank n_A d, width delta 2^m, infinity bound 2 omega_bar beta_bar = 8 omega b^tau, 6_Collaborative_Hachi.tex:563) in the lowest delta-band (b^(delta-1) < q <= b^delta) of admissible q < 2^63 that admits a secure n_A: the smallest secure n_A in that band, and the smallest prime = 5 mod 8 of the band above the Setup bounds with that MSIS margin (bisection on log2 q to 1/256 bit, the final prime checked again). The last column lists the higher bands where fewer rows of A would do; each band adds a digit to delta, which costs more in mu and N. n_B: the smallest height with MSIS for B secure (bound 2 gamma_bar = 2b); m_rho: the smallest value with MLWE_{q, d, n_D, m_rho - n_D, b} secure (5_Building_Blocks.tex:193).
- MSIS: first the trivial linear-algebra attack, rank log2(q/(2B + 1)) >= lambda, then `SIS.lattice` (infinity norm) with zeta = 0 on m' columns, minimised over m' (a grid of 1.25 to 64 times the rank, refined by ternary search), because the estimator's own search over ignored columns misses interior optima and returns infinite costs for wide matrices. MLWE: `LWE.estimate` without arora-gb and bkw (only n_D d samples per secret), and also without dual_hybrid when n_D d < (m_rho - n_D) d, because its MATZOV cost model assumes n samples and uses m only to cap the block size; secret and error uniform on [-8, 7].
- Bits are log2 of the estimated cost; 'inf' means that no number of columns gave a finite cost (every evaluation infinite or beyond 2^1000). Estimates, not proofs; see the notes in `parameters.sage`.

| preset | l | n | d | omega | q | log2 q | delta | tau | r | m | n_A | n_B | m_rho | mu | l'' | N | log2 beta | MSIS A trivial / lattice | MSIS B | MLWE (attack) | M^-n | fewer rows of A in higher bands (delta: n_A) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| secure-l16-n2-a8 | 16 | 2 | 256 | 21 | 34801213478789 | 44.984 | 12 | 9 | 6 | 5 | 7 | 1 | 5 | 9803 | 23 | 98304 | 32.98 | 1060.7 / 115.0 | 498.7 | 145.6 (bdd) | 0.607 | 14: 6; 16: 5 |
| secure-l16-n2-a10 | 16 | 2 | 1024 | 15 | 17171531776141 | 43.965 | 11 | 9 | 5 | 4 | 2 | 1 | 3 | 2713 | 23 | 180224 | 32.37 | 119.2 / 132.0 | inf | 201.6 (bdd) | 0.607 | none |
| secure-l16-n4-a8 | 16 | 4 | 256 | 21 | 384156492036173 | 48.449 | 13 | 10 | 6 | 6 | 8 | 1 | 5 | 16297 | 23 | 212992 | 35.09 | 115.4 / 120.7 | 538.6 | 132.9 (bdd) | 0.607 | 14: 7; 16: 6 |
| secure-l16-n4-a10 | 16 | 4 | 1024 | 15 | 17171531776141 | 43.965 | 11 | 9 | 5 | 5 | 2 | 1 | 3 | 4369 | 24 | 360448 | 34.37 | 119.2 / 132.0 | inf | 201.6 (bdd) | 0.607 | none |
| secure-l20-n2-a8 | 20 | 2 | 256 | 22 | 403122206336093 | 48.518 | 13 | 10 | 8 | 7 | 8 | 1 | 5 | 46837 | 25 | 425984 | 37.16 | 120.4 / 120.7 | 539.3 | 132.6 (bdd) | 0.607 | 14: 7; 16: 6 |
| secure-l20-n2-a10 | 20 | 2 | 1024 | 15 | 274744508417461 | 47.965 | 12 | 10 | 7 | 6 | 2 | 1 | 3 | 12367 | 25 | 786432 | 36.49 | 119.2 / 121.9 | inf | 181.6 (bdd) | 0.607 | none |
| secure-l20-n4-a8 | 20 | 4 | 256 | 22 | 15395032342003157 | 53.773 | 14 | 11 | 8 | 8 | 8 | 1 | 5 | 72205 | 26 | 917504 | 39.27 | 2690.8 / 115.1 | 600.1 | 117.0 (bdd) | 0.607 | 16: 7 |
| secure-l20-n4-a10 | 20 | 4 | 1024 | 15 | 274744508417461 | 47.965 | 12 | 10 | 7 | 7 | 2 | 1 | 3 | 20125 | 26 | 1572864 | 38.49 | 119.2 / 121.9 | inf | 181.6 (bdd) | 0.607 | none |
| secure-l24-n2-a8 | 24 | 2 | 256 | 22 | 15395032342003157 | 53.773 | 14 | 11 | 10 | 9 | 8 | 1 | 5 | 208135 | 27 | 1835008 | 41.27 | 2690.8 / 115.1 | 600.1 | 117.0 (bdd) | 0.607 | 16: 7 |
| secure-l24-n2-a10 | 24 | 2 | 1024 | 16 | 8327002901668021 | 52.887 | 14 | 11 | 9 | 8 | 2 | 1 | 3 | 61019 | 27 | 3670016 | 40.81 | 1816.0 / 115.1 | inf | 161.6 (bdd) | 0.607 | none |
| secure-l24-n4-a8 | 24 | 4 | 256 | 22 | 102705810279674029 | 56.511 | 15 | 12 | 10 | 10 | 9 | 1 | 6 | 338545 | 28 | 3932160 | 43.37 | 119.5 / 118.6 | 631.8 | 165.7 (bdd) | 0.607 | 16: 8 |
| secure-l24-n4-a10 | 24 | 4 | 1024 | 16 | 8327002901668021 | 52.887 | 14 | 11 | 9 | 9 | 2 | 1 | 3 | 100533 | 28 | 7340032 | 42.81 | 1816.0 / 115.1 | inf | 161.6 (bdd) | 0.607 | none |
| secure-l28-n2-a8 | 28 | 2 | 256 | 22 | 102705810279674029 | 56.511 | 15 | 12 | 12 | 11 | 9 | 1 | 6 | 983353 | 29 | 7864320 | 45.37 | 119.5 / 118.6 | 631.8 | 165.7 (bdd) | 0.607 | 16: 8 |
| secure-l28-n2-a10 | 28 | 2 | 1024 | 16 | 74035283097514237 | 56.039 | 15 | 12 | 11 | 10 | 3 | 1 | 3 | 307327 | 30 | 15728640 | 44.91 | 120.0 / 162.5 | inf | 150.9 (bdd) | 0.607 | 16: 2 |
| secure-l28-n4-a8 | 28 | 4 | 256 | 22 | 6062548914780259333 | 62.395 | 16 | 13 | 12 | 12 | 9 | 1 | 6 | 1507993 | 30 | 16777216 | 47.46 | 4458.7 / 115.1 | 699.9 | 146.7 (bdd) | 0.607 | none |
| secure-l28-n4-a10 | 28 | 4 | 1024 | 16 | 1183762967321726413 | 60.038 | 16 | 13 | 11 | 11 | 3 | 1 | 3 | 557325 | 31 | 33554432 | 47.00 | 117.0 / 152.1 | inf | 139.3 (bdd) | 0.607 | none |
| secure-l30-n2-a8 | 30 | 2 | 256 | 23 | 6618259129420972141 | 62.521 | 16 | 13 | 13 | 12 | 9 | 1 | 6 | 2163021 | 31 | 16777216 | 47.52 | 4602.5 / 115.1 | 701.3 | 146.3 (bdd) | 0.607 | none |
| secure-l30-n2-a10 | 30 | 2 | 1024 | 16 | 1183762967321726413 | 60.038 | 16 | 13 | 12 | 11 | 3 | 1 | 3 | 688263 | 31 | 33554432 | 47.00 | 117.0 / 152.1 | inf | 139.3 (bdd) | 0.607 | none |
| secure-l30-n4-a8 | 30 | 4 | 256 | 23 | 6618259129420972141 | 62.521 | 16 | 13 | 13 | 13 | 9 | 1 | 6 | 3015321 | 31 | 33554432 | 49.52 | 4602.5 / 115.1 | 701.3 | 146.3 (bdd) | 0.607 | none |
| secure-l30-n4-a10 | 30 | 4 | 1024 | 16 | 1183762967321726413 | 60.038 | 16 | 13 | 12 | 12 | 3 | 1 | 3 | 1114381 | 32 | 67108864 | 49.00 | 117.0 / 152.1 | inf | 139.3 (bdd) | 0.607 | none |
| secure-l16-n8-a8 | 16 | 8 | 256 | 21 | 384156492036173 | 48.449 | 13 | 10 | 7 | 6 | 8 | 1 | 5 | 24273 | 24 | 212992 | 37.09 | 115.4 / 120.7 | 538.6 | 132.9 (bdd) | 0.607 | 14: 7; 16: 6 |
| secure-l16-n8-a10 | 16 | 8 | 1024 | 15 | 274744508417461 | 47.965 | 12 | 10 | 6 | 5 | 2 | 1 | 3 | 6457 | 24 | 393216 | 36.49 | 119.2 / 121.9 | inf | 181.6 (bdd) | 0.607 | none |
| secure-l20-n8-a8 | 20 | 8 | 256 | 22 | 15395032342003157 | 53.773 | 14 | 11 | 9 | 8 | 8 | 1 | 5 | 104985 | 26 | 917504 | 41.27 | 2690.8 / 115.1 | 600.1 | 117.0 (bdd) | 0.607 | 16: 7 |
| secure-l20-n8-a10 | 20 | 8 | 1024 | 15 | 4334287545157813 | 51.945 | 13 | 11 | 8 | 7 | 3 | 1 | 3 | 32057 | 26 | 1703936 | 40.61 | 116.2 / 174.9 | inf | 165.1 (bdd) | 0.607 | 14: 2 |
| secure-l24-n8-a8 | 24 | 8 | 256 | 22 | 102705810279674029 | 56.511 | 15 | 12 | 11 | 10 | 9 | 1 | 6 | 492769 | 28 | 3932160 | 45.37 | 119.5 / 118.6 | 631.8 | 165.7 (bdd) | 0.607 | 16: 8 |
| secure-l24-n8-a10 | 24 | 8 | 1024 | 16 | 74035283097514237 | 56.039 | 15 | 12 | 10 | 9 | 3 | 1 | 3 | 154105 | 29 | 7864320 | 44.91 | 120.0 / 162.5 | inf | 150.9 (bdd) | 0.607 | 16: 2 |
