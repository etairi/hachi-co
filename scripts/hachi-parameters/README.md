# Hachi's parameters for the secure presets

`crates/params/presets/hachi.json` gives, for each secure preset of Hachi-Co, one choice of Hachi's decomposition
base b, challenge radius ω and matrix height n_A (n_B = n_D = 1), without hiding. It is an estimate of the binding of
the Module-SIS instances that Hachi's iterations use, not a proof, and it does not cover hiding.

Steps, from the repository root:

1. `thresholds.txt`: the largest log₂ β with infinity-norm Module-SIS at 115 and 128 bits, per modulus and rank.
   About 7 hours on one core with the lattice estimator at revision `53da5982597709ba0fdf94ea37a84d822310fd84`, the
   revision of `secure.json`, checked out as `scripts/hachi-parameters/lattice-estimator`:

   ```text
   sage scripts/hachi-parameters/msis_thresholds.sage > scripts/hachi-parameters/thresholds.txt
   ```

   The moduli are those of `secure.json`; pass new ones as arguments, each as its own word.

2. The schedules and proof sizes of the candidates, for the secure presets as they are in `secure.json`:

   ```text
   cargo test --release -p hachi-co --test hachi_parameters hachi_layouts -- --ignored --nocapture \
     | grep '^{' > /tmp/layouts.jsonl
   ```

3. The choice: per preset, among the candidates where the share polynomial and both blinders reach the target, the one
   with the fewest proof words (`choose.py` documents the instances):

   ```text
   python3 scripts/hachi-parameters/choose.py /tmp/layouts.jsonl > crates/params/presets/hachi.json
   ```
