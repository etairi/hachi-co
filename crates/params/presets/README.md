# Secure Hachi-Co presets

Written by `parameters.sage` (Hachi-Co section); do not edit by hand. The same data, with every derived
quantity, is in `secure.json`, which `hachi-co-params` embeds and re-derives in its tests.

- Command: `sage parameters.sage --hachi-co-only` (run from the repository root; SageMath version 10.9, Release Date: 2026-05-04, started 2026-11-15 20:38).
- Lattice estimator: https://github.com/malb/lattice-estimator, revision `53da5982597709ba0fdf94ea37a84d822310fd84` (an existing checkout given by LATTICE_ESTIMATOR_DIR).
- Cascade: `6_Collaborative_Hachi.tex:267-285` of the draft at commit `8834c9f`, with beta_1 = 2^(r - ceil(log n)) omega floor(b/2) as the draft's commit `2ee6507` revises line 279 (the bound of `5_Building_Blocks.tex`:141), gamma_bar = b - 1 as the draft's commit `28721d6` sets it, the least tau whose balanced digits represent beta (the Setup's rule since 2026-11-10), the L = lambda masks per prover in mu and in the width of D and no bound 1 + 2^(l+3) n omega^2 b on q (the parameter table of the draft's commit `1096ea2`), and the rules of `docs/m0-spec.md`.
- Fixed choices: lambda = 115, k = 4 (kappa = 2), b = 16, tight tau, the l1-ball challenge set with the smallest omega such that 2 L 2^r/|C| <= 2^-lambda, L = lambda, n_D = 1, t = n - 1.
- n_A and q: MSIS for A (rank n_A d, width delta 2^m, infinity bound 2 omega_bar beta_bar = 8 omega b^tau, 6_Collaborative_Hachi.tex:581) in the lowest delta-band (b^(delta-1) < q <= b^delta) of admissible q < 2^63 that admits a secure n_A: the smallest secure n_A in that band, and the smallest prime = 5 mod 8 of the band above the Setup bounds with that MSIS margin (bisection on log2 q to 1/256 bit, the final prime checked again). The last column lists the higher bands where fewer rows of A would do; each band adds a digit to delta, which costs more in mu and N. n_B: the smallest height with MSIS for B secure (bound 2 gamma_bar = 2(b - 1)); m_v and m_u: the smallest values with MLWE_{q, d, n_D, m_v - n_D, b} and MLWE_{q, d, n_B, m_u - n_B, b} secure (t-hiding.tex:20-26, 5_Building_Blocks.tex:207).
- MSIS: first the trivial linear-algebra attack, rank log2(q/(2B + 1)) >= lambda, then `SIS.lattice` (infinity norm) with zeta = 0 on m' columns, minimised over m' (a grid of 1.25 to 64 times the rank, refined by ternary search), because the estimator's own search over ignored columns misses interior optima and returns infinite costs for wide matrices. MLWE: `LWE.estimate` without arora-gb and bkw (only n_D d samples per secret), and also without dual_hybrid when n d < (m - n) d, because its MATZOV cost model assumes n samples and uses m only to cap the block size; secret and error uniform on [-8, 7].
- Bits are log2 of the estimated cost; 'inf' means that no number of columns gave a finite cost (every evaluation infinite or beyond 2^1000). Estimates, not proofs; see the notes in `parameters.sage`.

| preset | l | n | d | omega | q | log2 q | delta | tau | r | m | n_A | n_B | m_u | m_v | mu | l'' | N | log2 beta | MSIS A trivial / lattice | MSIS B | MLWE (attack) | M^-n | fewer rows of A in higher bands (delta: n_A) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| secure-l16-n2-a8 | 16 | 2 | 256 | 23 | 41160292688429 | 45.226 | 12 | 9 | 6 | 5 | 7 | 1 | 5 | 5 | 31702 | 24 | 98304 | 32.11 | 1259.3 / 115.0 | 512.2 | 144.6 (bdd) | 0.607 | 14: 6; 16: 5 |
| secure-l16-n2-a10 | 16 | 2 | 1024 | 16 | 18321384349621 | 44.059 | 12 | 9 | 5 | 4 | 2 | 1 | 3 | 3 | 11174 | 25 | 196608 | 31.58 | 120.0 / 131.7 | inf | 201.0 (bdd) | 0.607 | none |
| secure-l16-n4-a8 | 16 | 4 | 256 | 23 | 41160292688429 | 45.226 | 12 | 9 | 6 | 6 | 7 | 1 | 5 | 5 | 57258 | 25 | 196608 | 34.11 | 1259.3 / 115.0 | 512.2 | 144.6 (bdd) | 0.607 | 14: 6; 16: 5 |
| secure-l16-n4-a10 | 16 | 4 | 1024 | 16 | 18321384349621 | 44.059 | 12 | 9 | 5 | 5 | 2 | 1 | 3 | 3 | 21194 | 26 | 393216 | 33.58 | 120.0 / 131.7 | inf | 201.0 (bdd) | 0.607 | none |
| secure-l20-n2-a8 | 20 | 2 | 256 | 23 | 421135043797661 | 48.581 | 13 | 10 | 8 | 7 | 8 | 1 | 5 | 5 | 73524 | 26 | 425984 | 36.22 | 118.2 / 120.4 | 551.4 | 132.4 (bdd) | 0.607 | 14: 7; 16: 6 |
| secure-l20-n2-a10 | 20 | 2 | 1024 | 16 | 293142149592181 | 48.059 | 13 | 10 | 7 | 6 | 2 | 1 | 3 | 3 | 22296 | 26 | 851968 | 35.70 | 120.0 / 121.7 | inf | 181.1 (bdd) | 0.607 | none |
| secure-l20-n4-a8 | 20 | 4 | 256 | 23 | 421135043797661 | 48.581 | 13 | 10 | 8 | 8 | 8 | 1 | 5 | 5 | 117094 | 26 | 851968 | 38.22 | 118.2 / 120.4 | 551.4 | 132.4 (bdd) | 0.607 | 14: 7; 16: 6 |
| secure-l20-n4-a10 | 20 | 4 | 1024 | 16 | 293142149592181 | 48.059 | 13 | 10 | 7 | 7 | 2 | 1 | 3 | 3 | 39598 | 27 | 1703936 | 37.70 | 120.0 / 121.7 | inf | 181.1 (bdd) | 0.607 | none |
| secure-l22-n2-a8 | 22 | 2 | 256 | 23 | 421135043797661 | 48.581 | 13 | 10 | 9 | 8 | 8 | 1 | 5 | 5 | 120116 | 26 | 851968 | 38.22 | 118.2 / 120.4 | 551.4 | 132.4 (bdd) | 0.607 | 14: 7; 16: 6 |
| secure-l22-n2-a10 | 22 | 2 | 1024 | 16 | 293142149592181 | 48.059 | 13 | 10 | 8 | 7 | 2 | 1 | 3 | 3 | 35608 | 27 | 1703936 | 37.70 | 120.0 / 121.7 | inf | 181.1 (bdd) | 0.607 | none |
| secure-l22-n4-a8 | 22 | 4 | 256 | 23 | 16790798540447021 | 53.899 | 14 | 11 | 9 | 9 | 8 | 1 | 5 | 5 | 201362 | 27 | 1835008 | 40.33 | 2815.9 / 115.1 | 614.0 | 116.7 (bdd) | 0.607 | 16: 7 |
| secure-l22-n4-a10 | 22 | 4 | 1024 | 16 | 8327002901668021 | 52.887 | 14 | 11 | 8 | 8 | 2 | 1 | 3 | 3 | 69522 | 28 | 3670016 | 39.81 | 1816.0 / 115.1 | inf | 161.6 (bdd) | 0.607 | none |
| secure-l24-n2-a8 | 24 | 2 | 256 | 24 | 18211211930825437 | 54.016 | 14 | 11 | 10 | 9 | 8 | 1 | 5 | 5 | 236874 | 27 | 1835008 | 40.39 | 2930.1 / 115.1 | 615.4 | 116.4 (bdd) | 0.607 | 16: 7 |
| secure-l24-n2-a10 | 24 | 2 | 1024 | 17 | 9351599833054789 | 53.054 | 14 | 11 | 9 | 8 | 2 | 1 | 3 | 3 | 70602 | 28 | 3670016 | 39.89 | 1979.7 / 115.1 | inf | 161.0 (bdd) | 0.607 | none |
| secure-l24-n4-a8 | 24 | 4 | 256 | 24 | 18211211930825437 | 54.016 | 14 | 11 | 10 | 10 | 8 | 1 | 5 | 5 | 344722 | 28 | 3670016 | 42.39 | 2930.1 / 115.1 | 615.4 | 116.4 (bdd) | 0.607 | 16: 7 |
| secure-l24-n4-a10 | 24 | 4 | 1024 | 17 | 9351599833054789 | 53.054 | 14 | 11 | 9 | 9 | 2 | 1 | 3 | 3 | 119698 | 28 | 7340032 | 41.89 | 1979.7 / 115.1 | inf | 161.0 (bdd) | 0.607 | none |
| secure-l28-n2-a8 | 28 | 2 | 256 | 24 | 111899985626772997 | 56.635 | 15 | 12 | 12 | 11 | 9 | 1 | 6 | 6 | 1017566 | 29 | 7864320 | 44.49 | 115.3 / 118.3 | 646.2 | 165.3 (bdd) | 0.607 | 16: 8 |
| secure-l28-n2-a10 | 28 | 2 | 1024 | 17 | 78615930945368149 | 56.126 | 15 | 12 | 11 | 10 | 3 | 1 | 3 | 3 | 321014 | 30 | 15728640 | 43.99 | 117.4 / 162.2 | inf | 150.7 (bdd) | 0.607 | 16: 2 |
| secure-l28-n4-a8 | 28 | 4 | 256 | 24 | 111899985626772997 | 56.635 | 15 | 12 | 12 | 12 | 9 | 1 | 6 | 6 | 1420730 | 30 | 15728640 | 46.49 | 115.3 / 118.3 | 646.2 | 165.3 (bdd) | 0.607 | 16: 8 |
| secure-l28-n4-a10 | 28 | 4 | 1024 | 17 | 78615930945368149 | 56.126 | 15 | 12 | 11 | 11 | 3 | 1 | 3 | 3 | 519146 | 30 | 31457280 | 45.99 | 117.4 / 162.2 | inf | 150.7 (bdd) | 0.607 | 16: 2 |
| secure-l30-n2-a8 | 30 | 2 | 256 | 24 | 111899985626772997 | 56.635 | 15 | 12 | 13 | 12 | 9 | 1 | 6 | 6 | 2000606 | 30 | 15728640 | 46.49 | 115.3 / 118.3 | 646.2 | 165.3 (bdd) | 0.607 | 16: 8 |
| secure-l30-n2-a10 | 30 | 2 | 1024 | 17 | 78615930945368149 | 56.126 | 15 | 12 | 12 | 11 | 3 | 1 | 3 | 3 | 628214 | 31 | 31457280 | 45.99 | 117.4 / 162.2 | inf | 150.7 (bdd) | 0.607 | 16: 2 |
| secure-l30-n4-a8 | 30 | 4 | 256 | 24 | 7194089776299060413 | 62.642 | 16 | 13 | 13 | 13 | 9 | 1 | 6 | 6 | 3088306 | 31 | 33554432 | 48.58 | 4738.3 / 115.1 | 717.4 | 146.0 (bdd) | 0.607 | none |
| secure-l30-n4-a10 | 30 | 4 | 1024 | 17 | 1259260689129766237 | 60.127 | 16 | 13 | 12 | 12 | 3 | 1 | 3 | 3 | 1143578 | 32 | 67108864 | 48.09 | 122.3 / 152.1 | inf | 139.1 (bdd) | 0.607 | none |
| secure-l16-n8-a8 | 16 | 8 | 256 | 23 | 421135043797661 | 48.581 | 13 | 10 | 7 | 6 | 8 | 1 | 5 | 5 | 131018 | 26 | 212992 | 36.22 | 118.2 / 120.4 | 551.4 | 132.4 (bdd) | 0.607 | 14: 7; 16: 6 |
| secure-l16-n8-a10 | 16 | 8 | 1024 | 16 | 293142149592181 | 48.059 | 13 | 10 | 6 | 5 | 2 | 1 | 3 | 3 | 42586 | 27 | 425984 | 35.70 | 120.0 / 121.7 | inf | 181.1 (bdd) | 0.607 | none |
| secure-l20-n8-a8 | 20 | 8 | 256 | 23 | 16790798540447021 | 53.899 | 14 | 11 | 9 | 8 | 8 | 1 | 5 | 5 | 219938 | 27 | 917504 | 40.33 | 2815.9 / 115.1 | 614.0 | 116.7 (bdd) | 0.607 | 16: 7 |
| secure-l20-n8-a10 | 20 | 8 | 1024 | 16 | 8327002901668021 | 52.887 | 14 | 11 | 8 | 7 | 2 | 1 | 3 | 3 | 69154 | 28 | 1835008 | 39.81 | 1816.0 / 115.1 | inf | 161.6 (bdd) | 0.607 | none |
| secure-l22-n8-a8 | 22 | 8 | 256 | 24 | 18211211930825437 | 54.016 | 14 | 11 | 10 | 9 | 8 | 1 | 5 | 5 | 323874 | 28 | 1835008 | 42.39 | 2930.1 / 115.1 | 615.4 | 116.4 (bdd) | 0.607 | 16: 7 |
| secure-l22-n8-a10 | 22 | 8 | 1024 | 17 | 9351599833054789 | 53.054 | 14 | 11 | 9 | 8 | 2 | 1 | 3 | 3 | 99618 | 28 | 3670016 | 41.89 | 1979.7 / 115.1 | inf | 161.0 (bdd) | 0.607 | none |
| secure-l24-n8-a8 | 24 | 8 | 256 | 24 | 111899985626772997 | 56.635 | 15 | 12 | 11 | 10 | 9 | 1 | 6 | 6 | 629618 | 29 | 3932160 | 44.49 | 115.3 / 118.3 | 646.2 | 165.3 (bdd) | 0.607 | 16: 8 |
| secure-l24-n8-a10 | 24 | 8 | 1024 | 17 | 78615930945368149 | 56.126 | 15 | 12 | 10 | 9 | 3 | 1 | 3 | 3 | 208850 | 29 | 7864320 | 43.99 | 117.4 / 162.2 | inf | 150.7 (bdd) | 0.607 | 16: 2 |
