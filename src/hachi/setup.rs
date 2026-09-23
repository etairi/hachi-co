//! # Public parameters (paper, Sections 4.1 and 4.4, Fig. 8)
//!
//! Chooses the parameters of one round of the scheme for a polynomial in $\ell$ variables: the ring
//! $\mathbf R_q = \mathbb Z_q\[X\]/(X^d+1)$ with $d = 2^\alpha$, the folding split
//! $\ell = r + m + \alpha$ ($2^r$ chunks of $2^m$ ring elements), the decomposition base $b$ with
//! its digit counts $\delta = \lceil \log_b q \rceil$ (full residues) and $\tau$ (the response
//! $\mathbf z$), the challenge sparsity $\omega$, and the dimensions and seeds of the commitment
//! matrices $\mathbf A \in \mathbf R_q^{n \times \delta 2^m}$, $\mathbf B \in \mathbf R_q^{n \times n\delta 2^r}$
//! and $\mathbf D \in \mathbf R_q^{n \times \delta 2^r}$ (Section 4.1 and Eq. (16)).
//!
//! Everything except $\ell$ and the `decompose` flag is hard-coded in the constants of this module;
//! [`Setup::setup`] only derives the dependent quantities. Names follow the crate-level notation
//! table: `k` is the challenge sparsity $\omega$ (not the extension degree), `delta_z` and `z_bound`
//! are $\tau$ and $\beta$.

use crate::hachi::Hachi;

// Default parameters

/// The prime modulus $q = 4294967197 = 2^{32} - 99$, with $q \equiv 5 \pmod 8$ as the paper
/// requires (Section 2.1, Lemma 3, and the subfields of Section 3). The base field of
/// [`crate::arithmetic::ExtField`] hard-codes the same value, so changing it here alone is not
/// enough.
pub const Q: u64 = 4294967197;
/// $\alpha = \log_2 d$. The ring dimension is fixed to $d = 2^{10} = 1024$ (a note in the source
/// says it should be chosen adaptively).
const MAX_ALPHA: usize = 10; // Maximum ring dimension is 2^10

/// The decomposition base $b = 16$; digits are balanced, in $\[-8, 7\]$.
const DECOMP_BASE: u64 = 16;
/// $\delta = \lceil \log_b q \rceil = 8$: digits of a full residue ($16^7 \lt q \lt 16^8$).
const DECOMP_DELTA: usize = 8;

/// $\tau = 4$: digits of a coefficient of the response $\mathbf z$ (the paper's
/// $\tau = \lceil \log_b \beta \rceil$, Section 4.2).
const Z_DECOMP_DELTA: usize = 4;
// require z in range [-8 . (16^3 + 16^2 + 16 + 1) ... 7 . (16^3 + 16^2 + 16 + 1)] = [-34952 ... 30583]
/// $\beta = 30583$: the bound on $\lVert \mathbf z \rVert_\infty$ that the prover asserts after
/// folding. It is the top of the balanced window of $\tau = 4$ base-16 digits,
/// $\[-8 \cdot 4369, 7 \cdot 4369\] = \[-34952, 30583\]$ with $4369 = (16^4-1)/15$, so that
/// $\mathbf z$ decomposes into $\tau$ digits without any residue-to-window mapping. The bound is
/// heuristic: the worst case $2^r \omega \cdot b/2 = 2^r \cdot 128$ (the paper's naive bound is
/// $2^r \omega b$, Section 4.4) exceeds it as soon as $r \ge 8$, i.e. $\ell \ge 25$ (since $r = \ell - \lfloor(\ell-\alpha)/2\rfloor - \alpha$), and the prover
/// then panics (the abort of Fig. 3).
const Z_BOUND: u64 = 30583; 

// TODO: K should be calculated based on alpha (this is correct for alpha=10).
/// $\omega = 16$: number of nonzero coefficients, each $\pm 1$, of a challenge $c \in \mathcal C$,
/// so $\lVert c \rVert_1 = \omega$. For $d = 1024$ the challenge set has
/// $\binom{1024}{16} \cdot 2^{16} \approx 2^{131.6}$ elements. The value is for $\alpha = 10$; the
/// source notes it should be derived from $\alpha$.
const K: usize = 16;

/// The public parameters of one round, shared by prover and verifier.
///
/// Produced by [`Setup::setup`] and never modified afterwards. Sizes are in ring elements unless
/// stated otherwise; residues are stored as `u64` in $\[0, q)$.
pub struct Parameters {
    /// $\ell$: number of variables of the committed multilinear polynomial, which has $2^\ell$
    /// coefficients in $\mathbb Z_q$. Requires $\ell \ge \alpha$ (the split below underflows otherwise).
    pub l: usize,
    /// $q$: the prime modulus [`Q`], $q \equiv 5 \pmod 8$.
    pub q: u64,
    /// $d = 2^\alpha$: dimension of $\mathbf R_q = \mathbb Z_q\[X\]/(X^d+1)$, fixed to $1024$;
    /// `d.log()` gives $\alpha$.
    pub d: usize,
    /// $m$: each chunk $\mathbf f_i$ has $2^m$ ring elements, $m = \lfloor (\ell - \alpha)/2 \rfloor$.
    pub m: usize,
    /// $r$: the witness is read as $2^r$ chunks, $r = \ell - m - \alpha$, so $r = m$ or $r = m+1$
    /// (the paper takes $m = r = (\ell-\alpha)/2$, Section 4.4).
    pub r: usize,

    /// $\omega$: number of nonzero $\pm 1$ coefficients of each challenge $c_i$ (see [`K`]). This is
    /// the challenge sparsity, not the extension degree of $\mathbb F_{q^4}$.
    pub k: usize,

    // Decomposition
    /// Whether [`crate::hachi::commit`] gadget-decomposes each chunk,
    /// $\mathbf s_i = \mathbf G^{-1}(\mathbf f_i)$ (paper, Eq. (13)), before the Ajtai commitment.
    /// `true` for the original witness; `false` for the next witness $(\mathbf z^{\prime}, \mathbf r)$, whose
    /// entries are already digits and are committed directly (Section 4.5, "avoiding
    /// re-decomposition"). Determines [`Parameters::width_a`]. The prover requires `true`.
    pub decomp_witness: bool,
    /// $\beta$: bound on $\lVert \mathbf z \rVert_\infty$, the largest absolute coefficient of the
    /// folded response (not its $\ell_1$ norm), asserted by the prover; see [`Z_BOUND`].
    pub z_bound: u64,
    /// $b$: the decomposition base ([`DECOMP_BASE`]); digits lie in $\[-b/2, b/2-1\]$.
    pub b: u64,
    /// $\delta = \lceil \log_b q \rceil$: digits of a residue, i.e. the expansion factor of
    /// $\mathbf G^{-1}$ on arbitrary elements ([`DECOMP_DELTA`]).
    pub delta: usize,
    /// $\tau$: digits of a coefficient of $\mathbf z$, the expansion factor of
    /// $\hat{\mathbf z} = \mathbf J^{-1}(\mathbf z)$ ([`Z_DECOMP_DELTA`]).
    pub delta_z: usize,
    
    // Dimensions of commitment matrices.
    /// $n = n_A = n_B = n_D$: height of the commitment matrices, $2^{10-\alpha} = 1$, chosen so that
    /// the Module-SIS rank $n d$ is $2^{10}$.
    pub n: usize,
    /// Width of $\mathbf A$: $\delta 2^m$ when [`Parameters::decomp_witness`] is set, else $2^m$.
    pub width_a: usize,
    /// Width of $\mathbf B$: $n \delta 2^r$ (one $\hat{\mathbf t}\relax_i \in \mathbf R_q^{n\delta}$ per chunk).
    pub width_b: usize,
    /// Width of $\mathbf D$: $\delta 2^r$ (one $\hat w_i \in \mathbf R_q^{\delta}$ per chunk).
    pub width_d: usize,
    /// `true` iff the three widths coincide; then one matrix, sampled from [`Parameters::a_seed`],
    /// serves as $\mathbf A$, $\mathbf B$ and $\mathbf D$ (paper, Section 5.4). With $n = 1$ and
    /// decomposition this holds iff $m = r$, i.e. $\ell - \alpha$ even; it never holds for the
    /// next-witness parameters.
    pub reuse_mats: bool,

    // Commitment matrix seeds.
    /// ChaCha12 seed of $\mathbf A$, fixed to `[1u8; 32]`. The matrices are regenerated from their
    /// seeds by commit, prove and verify (the cost is part of the benchmarks, Section 5.4).
    pub a_seed: [u8; 32],
    /// Seed of $\mathbf B$: `[2u8; 32]`, or [`Parameters::a_seed`] when [`Parameters::reuse_mats`].
    pub b_seed: [u8; 32],
    /// Seed of $\mathbf D$: `[3u8; 32]`, or [`Parameters::a_seed`] when [`Parameters::reuse_mats`].
    pub d_seed: [u8; 32],
}

/// Parameter generation.
pub trait Setup  {
    /// Public parameters for a multilinear polynomial in `l` $= \ell$ variables. `decompose` sets
    /// [`Parameters::decomp_witness`]: `true` for the original witness, `false` for the next
    /// witness (see [`crate::hachi::prove`]). Deterministic: the same `(l, decompose)` always
    /// yields the same parameters, matrix seeds included.
    fn setup(l: usize, decompose: bool) -> Parameters;
}

/// Derivation of the parameters from the constants of this module; the formulas are given in the
/// field documentation of [`Parameters`].
impl Setup for Hachi {
    fn setup(l: usize, decompose: bool) -> Parameters {
        // Modulus
        let q = Q;

        // Ring dimension. TODO: should be chosen adaptively.
        let alpha = MAX_ALPHA;
        let d: usize = 1 << alpha; 

        // folding parameters - set (nearly) equal: m = floor((l - alpha)/2), r = l - m - alpha
        let m = (l - alpha) / 2;
        let r = l - m - alpha;

        // number of non-zero coefficients in challenges
        let k = K;

        // decomposition
        let decomp_witness = decompose;
        let z_bound = Z_BOUND;
        let b = DECOMP_BASE;
        let delta = DECOMP_DELTA;
        let delta_z = Z_DECOMP_DELTA;

        // height of matrices - we require height * ring dimension at least 2^10
        let n = 1 << (10-alpha);

        // width of matrices
        let width_a = if decomp_witness {(1 << m) * delta } else { 1 << m };
        let width_b = n * (1 << r) * delta;
        let width_d = (1 << r) * delta;
        let reuse_mats = width_a == width_b && width_b == width_d;

        // seeds. TODO: make non-deterministic.
        let a_seed = [1u8; 32];
        let b_seed = if reuse_mats { a_seed } else { [2u8; 32] };
        let d_seed = if reuse_mats { a_seed } else { [3u8; 32] };

        Parameters { l, q, d, m, r, k, decomp_witness, z_bound, b, delta, delta_z, n, width_a, width_b, width_d, reuse_mats, a_seed, b_seed, d_seed }
    }
}
