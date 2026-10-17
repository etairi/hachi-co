# Hachi-Co Sage reference oracle (milestone 0)

A SageMath/numpy reference implementation of the Hachi-Co protocol at the toy presets of `docs/m0-spec.md`, for
n ∈ {1, 2, 3}. It is a test oracle and fixture generator for the Rust milestone-1 code. It is not secure and
not optimized. Everything follows the draft at `/Users/etairi/research/collaborative-hatchi` (commit `625ced1`).
Line references use the short names `4_P` (4_preliminaries.tex), `5_BB` (5_Building_Blocks.tex),
`6_CH` (6_Collaborative_Hachi.tex) and `F2R` (field-to-ring-rok.tex).

Implemented: Setup, Commit, Open (6_CH:263-406), PConv and Πconv (F2R:36-169), the ring-to-linear step with
masks, ρ, rejection sampling, aborts and retries (5_BB:98-193), and the collaborative sumcheck with its verifier
(5_BB:195-335). Hachi as the base PCS is deferred: its openings are checked by evaluating the committed
polynomials directly.

## Running

```
sage -python ref/run_oracle.py                 # self-tests, toy-n1..n3, negative tests, writes ref/fixtures/*.json
sage -python ref/run_oracle.py --n 3           # one preset
sage -python ref/run_oracle.py --no-write      # do not touch the fixtures
sage -python ref/run_oracle.py --sweep 4       # plus 4 more honest runs per preset (other seeds)
sage -python ref/run_oracle.py --seed-index 0  # use seed index 0 instead of the selection rule below
```

The script works from any directory and exits with status 0 only if every check passes. Measured on the
development machine with Sage 10.9, Python 3.14.3 and numpy 2.4.4:

| preset | one honest run (Commit to final verifier check) | negative tests | fixture |
|---|---|---|---|
| toy-n1 | 0.7 s | 2.7 s | 210 KB |
| toy-n2 | 1.2 s | 8.2 s | 392 KB |
| toy-n3 | 3.2 s | 21 s | 609 KB |

The full default run takes about 45 s, including the self-tests and the runs tried by the seed selection. Two
runs produce byte-identical fixtures.

## Files

| file | content |
|---|---|
| `run_oracle.py` | driver: self-tests, instances, negative tests, fixtures |
| `hachico_ref/subfield.py` | R_q^H, Tr_H, ψ, ψ⁻¹, E, monomials (main session; read-only here) |
| `hachico_ref/params.py` | Setup cascade 6_CH:265-281, τ rule, q rule, toy table check |
| `hachico_ref/prg.py` | SHAKE128 counter-mode PRG, uniform/box/S_β samplers, exact ℓ₁-ball sampler |
| `hachico_ref/ring.py` | R_q on numpy int64 (with an overflow guard), G, G⁻¹, J, J⁻¹ |
| `hachico_ref/fqk.py` | vectorized F_{q^k} in Hachi's basis (product table taken from `subfield.py`), eq, MLE, Lagrange |
| `hachico_ref/context.py` | per-instance context; a Sage (FLINT) R_q used only to check relations |
| `hachico_ref/commit.py` | Setup matrices, input, AS.Share, PConv, Commit, Open |
| `hachico_ref/conv.py` | Πconv: x′, b, a, v, d, ⟨u⟩, F_AH(E), the check y = E(u) |
| `hachico_ref/r2l.py` | ring-to-linear: z′ layout, partition, M, masks, ρ, F_AH(D), Rej, retries, ẑ; verifier coins |
| `hachico_ref/sumcheck.py` | quotient shares, ⟨w̃⟩ tables, blinders, M̃*, provers' rounds, verifier, ideal PCS |
| `hachico_ref/oracle.py` | one honest run with every relation check; the verifier's public recomputation |
| `hachico_ref/negative.py` | tampered transcripts and witnesses, Open negatives, Rej boundary, abort rate |
| `hachico_ref/selftest.py` | arithmetic self-tests against Sage, brute force and `subfield.py` |
| `hachico_ref/fixtures.py` | fixture writer (format below) |
| `fixtures/toy-n{1,2,3}.json` | the fixtures |

`fixtures/gen_math_fixtures.py` belongs to the math agent (fixtures for `crates/math`); the oracle does not use it.

## Conventions

- **Indexing.** All positions are 0-indexed. The draft numbers z′, y, the rows i′ of M̃ and the hypercube
  coordinates from 1; its position u is our u − 1. Provers are numbered 1..n.
- **Bit strings.** nat(b₁..b_s) = Σ_j b_j 2^{s−j}, with b₁ the most significant bit (m0-spec). This replaces
  the draft's nat (4_P:7: 1-indexed, b₁ least significant). Coefficient vectors and monomial vectors are ordered
  by nat; X₁ goes with the most significant bit.
- **Field.** F_{q^k} = R_q^H in Hachi's basis (Eq. (7)); an element is (a₀..a_{k−1}). Arithmetic is that of
  `subfield.py`: the vectorized product uses the table `basis_a · basis_b` computed with `Subfield.fmul`.
- **Gadgets.** Digits lie in [⌈−b/2⌉, ⌈b/2⌉−1]. The layout is element-major: entry i's δ digits sit at
  iδ..iδ+δ−1, and digit e has weight b^e. J is the same with τ digits; τ follows the tight rule. G⁻¹(x)
  decomposes rep(x) = x if x ≤ top, else x − q, where top = (⌈b/2⌉−1)(b^δ−1)/(b−1). J⁻¹ decomposes the
  integer itself and fails outside [⌈−b/2⌉S_τ, (⌈b/2⌉−1)S_τ].
- **Blocks and owners.** f′ has 2^r blocks of 2^m ring elements; f′[i·2^m + j] is entry j of block i. Block i
  belongs to prover (i >> (r − ⌈log n⌉)) + 1 if that number is ≤ n. Otherwise the block is all-zero and belongs
  to nobody: for n = 3, blocks 6 and 7.
- **PConv.** f̂[nat(p)·2^ℓ + nat(q)] = ⟨f⟩_{nat(p)+1}[nat(q)] for nat(p) < n, else 0. Then
  f′[t] = ψ(f̂[t·d/k .. t·d/k + d/k − 1]).
- **Πconv.** x′ = 1^{⌈log n⌉} ‖ x[0 : ℓ−α+κ], b = monomials(x′[0:r]), a = monomials(x′[r:ℓ′]),
  v = ψ(monomials(x[ℓ−α+κ:])), and d[nat(i)·2^m + nat(j)] = b_i a_j.
- **z′ layout.** z′ = [ŵ (2^rδ) | ŵ₀ (nδ) | t̂₀ (n n_A δ) | ρ (n m_ρ) | t̂ (2^r n_A δ) | ẑ (2^m δ τ) | ⟨u⟩ (1)].
  Entry (jδ+e)τ+t of ẑ is digit t of z[jδ+e].
- **Rows of M.** D (n_D), B (n_B), bᵀG (1), the c row (1), the A rows (n_A); y = (v, u_com, 0, 0, 0).
- **Shares.** ⟨z′⟩_p equals z′ on S_p and 0 elsewhere, except position μ−1 (⟨u⟩), which holds ⟨u⟩_p = dᵀ{f′}_p.
  Prover 1 holds y and ẑ. S_p consists of p's blocks of ŵ and t̂, its own ŵ₀, t̂₀ and ρ, and, for p = 1, ẑ.
- **Hypercube.** V = (U, L), with |U| = ⌈log(μ+n′)⌉ and |L| = log 2d. A point has index u0·2d + l0, and V₁ is
  the most significant bit. Round r binds V_r (the table update is T′[x] = T[x] + a_r(T[x+h] − T[x])).
- **eq.** eq(τ₀, (U, L, bit(i))) pairs τ₀[0:ℓ″] with (U, L) and τ₀[ℓ″:] with the ⌈log n⌉ bits of i − 1.
  eq(τ₁, bit(i′)) is entry i′ − 1 of the eq table of τ₁.
- **Round messages.** H_{0,r}(t) for t = 1..d₀+1 and H_{1,r}(t) for t = 1..d₁+1 (the F_AH(I) outputs). V
  interpolates H(0) and H(a_r).
- **Blinders.** g(V) = s₀ + Σ_{j=1}^{ℓ″} Σ_{k=1}^{D} s[j−1][k−1] V_j^k, with D = d₀ for g₀ and D = d₁ for g₁.

## Choices where the draft is silent or inconsistent

1. **Bit order and 1-indexing (4_P:7).** We use the m0-spec nat everywhere, as described above.
2. **Representative for G⁻¹ (m0-spec "Digit sets"; 4_P:96).** The spec decomposes "the unique representative
   of x in the window". When b^δ > q the representative is not unique: at the toy presets b^δ ≈ 2q, so most
   residues have two. We take x if x ≤ top and x − q otherwise, the rule of the legacy `to_window`
   (crates/hachi/src/arithmetic/utils.rs:100). The math crate must use the same rule, or the fixtures' s, t̂,
   ŵ, t̂₀ and ŵ₀ will differ. Separately, the spec writes the window as [−⌈b/2⌉S, (⌈b/2⌉−1)S]. With the digit set
   [⌈−b/2⌉, ⌈b/2⌉−1] the lower end is ⌈−b/2⌉S = −⌊b/2⌋S. The two agree for even b, including b = 4.
3. **PConv runs once.** Both Commit (6_CH:310) and Πconv (F2R:124-126) call PConv. We run it in Commit and
   reuse f′ from the states. PConv is deterministic, so the result is the same.
4. **u stays shared.** F2R:100 and F2R:166 list u as public, but only E(u) is opened (F2R:160-164), and the
   author comment at F2R:100 asks for u to stay secret-shared. Πconv outputs ⟨u⟩, not u, and ⟨u⟩ is the last
   entry of z′ (5_BB:171).
5. **Aborts and retries (6_CH:417, 5_BB:129).** An abort ends Eval with (⊥, ⊥), and the caller restarts it.
   Πconv is deterministic, so the oracle re-runs only the ring-to-linear step, with fresh masks, ρ and c
   (attempt index in the PRG domain). Every attempt is recorded.
6. **S_i when n is not a power of two (5_BB:197).** The draft calls (S_1..S_n) a partition of [μ−1]. For n = 3
   the positions of the unowned zero blocks (ŵ and t̂ of blocks 6 and 7) belong to no prover. We leave them in
   no S_i, so F_{0,i} forces every prover's share there to be zero. The S_i together with these positions form
   the partition.
7. **Variable order in ⟨w̃⟩_i and M̃_α (5_BB:207, 225, 254).** The order is (I′, U, L), with U before L. The
   order within each block follows nat, most significant bit first, and the sumcheck binds V₁ = U₁ first.
8. **e_i and the zero-check term of F₀ (5_BB:262, 269-270).** We read the formula literally. S_i ⊆ [μ−1], so
   ⟨u⟩ (draft position μ) and the quotient rows (μ+1..μ+n′) lie in no S_i, and on them the bracket
   1 − e_i − Σ_{u=μ}^{μ+n′} χ_{bit(u)} is 0. These rows are therefore neither range-checked nor zero-checked per
   prover: they are ordinary additive shares. Only their sums are constrained, by the E rows, the bᵀG row and
   the third term of M̃. A test confirms that compensated changes to the ⟨u⟩ or quotient shares are accepted.
   The padding rows (u > μ+n′) are zero-checked for every prover.
9. **The E rows (5_BB:259, 264, 266).** Row n′+j of M̃ (1-indexed j ∈ [d]) has E_{j,ℓ} at u = μ for ℓ ∈ [d],
   and pairs with y_{j−1}. Here E_{j,ℓ} = `E_matrix[j−1][ℓ−1]`, where `E_matrix[i][j]` is coefficient i of
   E(X^j). The draft's index ranges are consistent once translated. Only the d low coefficients of ⟨u⟩ enter
   these rows; the third term covers the high ones.
10. **First term of M̃_α (5_BB:254-255).** For u ≤ μ it runs over all ℓ ∈ [2d], as written.
11. **Evaluation nodes (5_BB:290-294).** [d₀+1] is read as {1, …, d₀+1}; H(0) comes from interpolation.
12. **The y in the claimed sum a (5_BB:264).** It is the public claim y embedded in R_q^H, which equals E(u)
    after Πconv's check.
13. **Base PCS (6_CH:289-292, 5_BB:220-230, 314-326).** The PCS is not modelled. Commitments are omitted, and
    an opening is accepted iff the claimed value equals the committed polynomial evaluated directly (`IdealPCS`).
    ⟨w̃⟩_p(a) is evaluated as Σ_x eq(a, x)·W_p[x], a different path from the provers' folding.
14. **Verifier coins.** They are interactive public coins derived from the instance seed with domain
    separation; Fiat–Shamir is not modelled. They never depend on the messages, so a tampered transcript can be
    re-verified with the same coins.
15. **Challenge distribution (5_BB:122, unspecified).** c is uniform on C = {c : ‖c‖₁ ≤ ω}, drawn with the
    exact sampler of the m0-spec: a weighted support size, then a subset, signs and a composition.
16. **Randomness and matrices.** All randomness comes from a SHAKE128 counter-mode PRG (`prg.py`). Entry (i, j)
    of A, B and D comes from its own stream `Prg(setup_seed, "setup/A", i·cols + j)`, and so on. The Rust side
    reads the matrices and all sampled values from the fixtures and does not re-derive them.
17. **Fold and ẑ (5_BB:128-133).** Σ c_{i′} s_{i′} is computed over Z[X]/(X^d+1) from centered digits. Rej
    accepts iff ‖z_i‖∞ ≤ γ − β₁ (4_P:119), and ẑ = J⁻¹(z) of the integer z.
18. **Open's slack space (6_CH:394).** c ∈ Z is tested as ‖c_i‖₁ ≤ ω̄ and gcd(c_i, X^d+1) = 1. Open is run
    with c* = (1, …, 1).
19. **Fixture seed.** The fixture uses the smallest seed index whose run contains an aborted attempt, so that
    the fixtures cover the abort-and-retry path. This is test coverage, not selection of results: every run
    tried on the way is fully checked, printed and listed under `seeds.selection.tried`. All of them passed.
    Fixture seeds: toy-n1 index 3 (four aborts), toy-n2 index 1 (four aborts), toy-n3 index 1 (one abort).

### Observations about the draft (no effect on correctness)

- **d₀ is larger than needed (6_CH:281).** d₀ = 2b + n′ + 1, but the individual degree of F_{0,i} is 2b + 1: 1
  from eq, 1 from e_i and 2b − 1 from the range product. The sum Σ_{u} χ_{bit(u)} is still multilinear. This is
  derived, and also measured: in every round of all three presets the unblinded round polynomials have degree
  ≤ 9 = 2b + 1 in the U rounds and 8 in the L rounds, while F₁ has degree 2 ≤ d₁ = 3 (fixture key
  `sumcheck.round_degrees_unblinded`). With g₀ of degree 2b + 1, each round message would be 2b + 2 field
  elements instead of 2b + n′ + 2.
- **6_CH:313** says "the i-th prover holds ⟨f′⟩_i = f′_i" for i ∈ {0,1}^r, which mixes block and prover
  indices. We follow 6_CH:406: block i belongs to the prover named by its top ⌈log n⌉ bits.
- **5_BB:134** says "slices of v, u, y". The y vector holds v and the commitment u; the claimed evaluation y
  enters only through a (5_BB:264).

## Fixture format (`fixtures/toy-n{1,2,3}.json`, format `hachi-co-ref-fixture/v1`)

**Encodings.**
- A residue is an integer in [0, q).
- A ring element is a list of d residues (index j ↔ X^j). Short integer objects (masks r_i, ρ_i, challenges c,
  z_i, z) are lists of signed integers.
- An F_{q^k} element is a list of k residues (Hachi-basis coordinates).
- A quotient share is a list of 2d residues.
- Nested lists put the outermost index first.
- Large derived objects are given as SHA-256 digests of their residues written as little-endian u64 in
  row-major order of the stated shape: M (n′ × μ × d), the ⟨w̃⟩_p tables (2^{ℓ″}), the M* table (2^{ℓ″} × k) and
  the eq(τ₁) table (2^{⌈log ν⌉} × k).

**Top-level keys.**

- `params`: every derived parameter, plus ln M, the acceptance probability per prover and the abort probability.
- `seeds`: the instance seed, the setup seed and the seed-selection record.
- `conventions`: the conventions above, as text.
- `setup`: A, B, D.
- `input`: f, the initial shares f_i, x, y = f(x) (computed with `subfield.py`), and y as a ring element.
- `commit`:
  - `reshare[i][j]` = ⟨f_i⟩_j, and the summed shares ⟨f⟩_j;
  - f̂ and f′;
  - the owner of each block, and per owned block: s, t and t̂;
  - the full t̂ and u.
- `open`: the bit and each sub-condition.
- `conv`: x′, b, a, v, d, the shares ⟨u⟩_p, the sum u formed inside F_AH(E), E(u), the matrix of E, and V's
  check.
- `ring_to_linear`:
  - w and ŵ;
  - per attempt: for each prover the mask, ρ, t₀, t̂₀, w₀, ŵ₀, ‖fold‖∞, ‖z_i‖∞ and accept; then v, c,
    abort, and the aborting provers;
  - the accepted attempt, the z_i sent to prover 1, z, ‖z‖∞ and the norm check;
  - the layout and row indices, the partition (as [start, end) ranges), the unowned positions;
  - z′, y, and the digest of M.
- `sumcheck`:
  - the quotient shares ⟨r⟩_p and their sum (low half);
  - the blinder coefficients, G₀,p and G₁,p;
  - the coins α, ρ, τ₀, τ₁ and a₁..a_ℓ″, and the claimed sum a;
  - per round, H₀ and H₁ (public), with the per-prover F_AH(I) inputs;
  - the final openings ⟨w̃⟩_p(a), g₀,p(a), g₁,p(a), and M̃*(a);
  - the degree audit and the table digests.
- `verifier`: every named check and the accept bit.
- `relations`: every oracle-level relation check (list below).
- `negative_tests`: the results of all negative tests, with the names of the checks that failed.

## Checks

**Relations (oracle level), all passing for n = 1, 2, 3.**
- Open(pp, st, u, f, c*) = 1 and each of its conditions, checked in Sage arithmetic.
- E·cf(u) = cf(E(u)); E(u) ∈ R_q^H; y = E(u), with y = f(x) computed independently by `subfield.py`.
- c ∈ C; ‖z‖∞ ≤ β at prover 1; J ẑ = z; z = Σ c_{i′}s_{i′} + Σ r_p over Z.
- M z′ = y over R_q, checked in Sage (FLINT), independently of the numpy ring code.
- The meaning of each row block (5_BB:182), checked on the unpacked values.
- (X^d+1)·r = Mz′ − y over Z_q[X], with the high half of Σ⟨r⟩_p equal to 0, checked in Sage.
- Share 1 of the quotient equals the literal ½(1−X^d)(…) mod X^{2d}+1, checked in Sage.
- P_p(x) = 0 at every hypercube point for every prover (range and zero checks).
- Σ_x Σ_p ⟨w̃⟩_p(x) M*(x) = a.
- The degrees of the round polynomials are at most d₀ and d₁.
- V's recomputed M equals the provers' M.
- The MLE of the provers' M* table at a equals V's factorized M̃*(a).
- V's a equals the provers' a.

**Verifier.** The check y = E(u), both round checks in every round, 3n PCS openings, and both final checks:
36, 39 and 44 checks for n = 1, 2, 3. All pass on honest runs, and on every run tried with other seeds
(`--sweep`).

**Negative tests.** Each passes iff exactly the expected checks fail.
- **Round messages.** H(1) += 1 at rounds 1, mid and last makes that round's check fail. Adding 5X(X−1)
  leaves the tampered round passing and makes the next round, or the final check, fail.
- **Final claims.** ⟨w̃⟩_p(a) += 1 fails the PCS opening and both final checks. g₀,p(a) or g₁,p(a) += 1 fails
  its PCS opening and its final check. G₀,p += 1 fails round 1 of H₀.
- **Public values.** The claimed y + 1 fails Πconv's check and round 1 of H₁. A tampered E(u) fails Πconv's
  check. A tampered v fails round 1 of H₁.
- **Witnesses.**
  - An out-of-range ẑ digit with J ẑ unchanged, so that M z′ = y still holds: only H₀ fails.
  - An in-range ŵ digit changed: only H₁ fails.
  - Prover 2 writes outside its slice with the sum unchanged: only H₀ fails.
  - A quotient whose sum has a nonzero coefficient of X^{d+1}: only H₁ fails.
  - Compensated changes to the ⟨u⟩ shares or to the quotient shares: accepted.
- **Open.** A changed digit of s, a changed f, or c outside Z each gives 0.
- **Rej.** Exhaustively at (N, γ, β₁) = (2, 4, 2), with the oracle's own Rej: for every v with ‖v‖∞ ≤ β₁ each
  z of the box is output by exactly one mask; for all 24 v with ‖v‖∞ = β₁+1 the accepted set differs. At the
  toy parameters, exactly: with v₁ = β₁+1 the first coordinate reaches 2γ−2β₁ of the 2γ−2β₁+1 box values, so
  Rej(v) is not distributed as SimRej.
- **Abort rate.** 10⁴ attempts with numpy PCG64 masks (seed 12345) through the oracle's Rej fall within 3σ of
  the exact 1 − M^{−n} (≈ 0.3935) for each preset.

**Self-tests.**
- The toy table and q rule are re-derived.
- `subfield.self_test` passes at the three moduli.
- The vectorized field product agrees with `Subfield.fmul`; associativity, distributivity and the Frobenius
  order hold.
- Ring products and mat-vec agree with Sage, including all-(q−1) operands.
- G and J round trips pass at the window boundaries; one step beyond fails.
- The ℓ₁-ball count agrees with brute force, and the sampler passes χ² over C(4,2).
- The mask and S_b samplers pass χ².
- eq, MLE and Lagrange interpolation are correct.
- The blinder's closed forms agree with brute force at ℓ″ = 4.

## Not covered

- Hachi as the base PCS, hiding commitments (A′·r_i) and Fiat–Shamir are not modelled. F_AH is ideal: the
  shares are summed.
- Only the toy presets were run. The numpy code asserts int64 headroom: d(q−1)² < 2⁶² for R_q, and q < 2³¹ for
  F_{q^k}.
- The oracle checks completeness and that the listed tampering is rejected. It proves neither soundness nor
  zero-knowledge.
- The PRG is the oracle's own, so the Rust side must inject the recorded randomness to compare runs.
