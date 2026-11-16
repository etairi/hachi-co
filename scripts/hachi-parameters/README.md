# Hachi's parameters for the secure presets

`crates/params/presets/hachi.json` gives, for each secure preset of Hachi-Co's main family (`secure.json`, 115 bits),
one choice of Hachi's decomposition base b, challenge radius ω and matrix height n_A (n_B = n_D = 1), and where one
exists the masks of the share polynomial and the blinders: the MLWE rank ν and the height of level 0, shared by the
three objects. `hachi100.json` does the same for the family `secure100` (`secure100.json`, 100 bits), which serves the
comparison with PPP26 (ePrint 2026/729), whose implementation targets about 100 bits. Each family's base PCS is chosen
at the family's λ: the Module-SIS column, the knowledge-error target and the λ of the masks. It is an estimate of the
Module-SIS instances that Hachi's iterations use and of the MLWE instance of its commitments, not a proof. The masks
are chosen for the draft's three commitments per prover, not for the merged option.

Steps, from the repository root, for a family `F` at λ bits (`secure` and 115, or `secure100` and 100). The Sage
scripts expect the lattice estimator at revision `53da5982597709ba0fdf94ea37a84d822310fd84`, the revision of the
preset files, checked out as `lattice-estimator` next to them (copy them next to a checkout to run them elsewhere).

1. `thresholds.txt`: the largest log₂ β with infinity-norm Module-SIS at each target of its header (115 and 128
   bits), per modulus and rank. About 7 hours on one core:

   ```text
   sage scripts/hachi-parameters/msis_thresholds.sage > scripts/hachi-parameters/thresholds.txt
   ```

   The moduli are those of `secure.json`. Pass new ones as arguments, each as its own word, and `targets=` for other
   targets. The rows of 107304362334462661, the modulus of `secure-l30-n2-a8` since the revised β₁ of 2026-11-02, took
   37 minutes:

   ```text
   sage msis_thresholds.sage 107304362334462661 ranks=256,512,1024,1280,1536,1792,2048,3072,4096 mlwe=256,512,1024,2048,3072
   ```

   Since then no preset uses 6062548914780259333, whose rows stay. The family `secure100` has its own moduli and
   target, in `thresholds-100.txt`:

   ```text
   sage msis_thresholds.sage <the moduli of secure100.json> targets=100 ranks=256,512,1024,1280,1536,1792,2048,3072,4096 mlwe=none > thresholds-100.txt
   ```

   On 2026-11-16 the presets were regenerated with Hachi-Co's knowledge-error rule 2L·2^r/|𝒞| ≤ 2^−λ (the forking of
   Π_B capped at |𝒞|), which moved q at every preset but `secure100-l24-n8-a10`. Of the 24 moduli of the two
   families, 22 had no rows (8327002901668021, now the modulus of a preset of `secure.json`, and 65326880641120541,
   that of `secure100-l24-n8-a10`, had them). The rows of the 22 have only the ranks that `choose.py` reads (n_A·d for
   n_A ≤ 8 at d = 256, n_A ≤ 6 at d = 1024) and the family's target, about 25 minutes per modulus: the 11 of
   `secure.json` in `thresholds-115-2026-11-16.txt` (115 bits, its own header, as `thresholds.txt` has two columns),
   and the 11 of `secure100.json` at the end of `thresholds-100.txt`. Three moduli of `secure.json` were run first
   with the ranks of the command above and both targets, and the file keeps their 115 column.

   ```text
   sage msis_thresholds.sage <modulus> ranks=256,512,768,1024,1280,1536,1792,2048 mlwe=none targets=115    # d = 256
   sage msis_thresholds.sage <modulus> ranks=1024,2048,3072,4096,5120,6144 mlwe=none targets=115           # d = 1024
   ```

2. `mlwe_ranks-<λ>.txt`: the least ν per modulus, ring degree and base at λ bits, for uniform digits as secret and
   error. Some minutes on one core (6 for the 26 presets of `secure.json`). `mlwe_ranks-128.txt` is the 128-bit file
   that the main family used until 2026-11-12:

   ```text
   sage mlwe_ranks.sage crates/params/presets/secure.json target=115 > scripts/hachi-parameters/mlwe_ranks-115.txt
   sage mlwe_ranks.sage crates/params/presets/secure100.json target=100 > scripts/hachi-parameters/mlwe_ranks-100.txt
   ```

3. The schedules and proof sizes of the candidates, for the family's presets as they are in its file, with masks at
   the family's λ. The candidates are b ∈ {4, 8, 16}, every ω from 8 to 32, n_A up to 8 rows (`-a8`) or 6 (`-a10`)
   and, with masks, the height n_A0 of level 0 from n_A to 12. About 10 minutes per family:

   ```text
   HACHI_PROBE_FAMILY=F cargo test --release -p hachi-co --test hachi_parameters hachi_layouts -- --ignored --nocapture \
     | grep '^{' > /tmp/layouts-F.jsonl
   ```

4. The choice: per preset, among the candidates where the share polynomial and both blinders reach the target, the one
   with the fewest proof words, a masked share polynomial first (`choose.py` documents the instances):

   ```text
   python3 scripts/hachi-parameters/choose.py /tmp/layouts-secure.jsonl --target 115 \
     --thresholds scripts/hachi-parameters/thresholds.txt \
     --thresholds scripts/hachi-parameters/thresholds-115-2026-11-16.txt > crates/params/presets/hachi.json
   python3 scripts/hachi-parameters/choose.py /tmp/layouts-secure100.jsonl --target 100 \
     --thresholds scripts/hachi-parameters/thresholds-100.txt > crates/params/presets/hachi100.json
   ```

   The files of 2026-11-16 were chosen this way for the regenerated presets, with the layouts of the code that
   commits every later witness as Section 3.1 commits a polynomial over the field (proof format 8). Every preset of
   both families has masks.

   The test `every_hachi_file_targets_its_family` checks that each file was chosen at its family's λ.
