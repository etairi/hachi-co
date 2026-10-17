//! Integer and field helpers shared by the arithmetic layer and the protocol: multilinear
//! monomials over $\mathbb Z_q$, power and gadget vectors, the balanced-digit window and the
//! per-coefficient digit split behind the gadget decomposition $\mathbf G^{-1}$ (paper,
//! Section 2.1), rejection sampling of uniform residues and of elements of $\mathbb F_{q^4}$,
//! and the lift of integers into the sumcheck field with the mixed
//! $\mathbb Z_q \times \mathbb F_{q^4}$ multiplication of the paper's Section 5.4.
//!
//! Integer conventions ([`CoeffType`] is `u64`): a *residue* is an element of $\mathbb Z_q$
//! stored in $\[0, q)$; a *short integer* (a digit, or a coefficient of the response
//! $\mathbf{z}$) is a signed value stored in two's complement, so that $-a$ is stored as
//! $2^{64} - a$. The two representations are interchangeable only in [`lift_int`], which reads
//! its argument as a signed `i64`; every other function states which one it expects.

use ark_ff::{AdditiveGroup, Field};
use rand::Rng;

use crate::arithmetic::{CoeffType, ExtField, field::{Fq, Fq2, Fq4}};

/// The multilinear monomial $\prod_{t < \mathrm{len}} x_t^{i_t} \bmod q$, where $i_t$ is bit $t$
/// of `i`: bit $t$ of the index pairs with `x[t]`, so the least significant bit selects `x[0]`.
/// For an evaluation point $x$ this is the factor by which the $i$-th coefficient of a
/// multilinear polynomial in coefficient form contributes to its value at $x$; the verifier uses
/// it to build the vector $\mathbf v_x$ of the reduction check
/// $y = \langle \mathrm{cf}(Y), \mathbf v_x \rangle$ (see [`crate::hachi::verify`]) and `main`
/// uses it for the reference evaluation.
///
/// Preconditions: every `x[t]` is a residue in $\[0, q)$ and $q < 2^{32}$, so that the product
/// of two residues fits in a `u64` before reduction. Only the low `len` bits of `i` are used.
/// $O(\mathrm{len})$ multiplications.
pub fn multi_lin_coeff_int(x: &[u64], i: usize, len: usize, q: u64) -> u64 {
    let mut c = 1;

    for j in 0..len {
        if ((i >> j) & 1) == 1 {
            c = (c * x[j]) % q;
        }
    }

    c
}

/// The powers $(1, x, x^2, \dots, x^{l-1})$ of a field element. Used for the table of
/// $\tilde{\alpha}(\ell) = \alpha^\ell$, $\ell < d$, of the ring-switching point $\alpha$ (paper,
/// Section 4.3).
pub fn powers<F: Field>(x: F, l: usize) -> Vec<F> {
    let mut powers = vec![F::ONE; l];

    for i in 1..l {
        powers[i] = x * powers[i-1];
    }

    powers
}

/// The gadget row $\mathbf g^T = (1, b, b^2, \dots, b^{l-1})$ of the gadget matrix
/// $\mathbf G_{b,n} = \mathbf I_n \otimes \mathbf g^T$ (paper, Section 2.1), as plain `u64`
/// integers. The caller must ensure $b^{l-1} < 2^{64}$: the products are unchecked, so an
/// overflow panics in debug builds and wraps in release builds.
pub fn gadget(b: u64, l: usize) -> Vec<u64> {
    let mut gadget = vec![1u64; l];

    for i in 1..l {
        gadget[i] = b * gadget[i-1];
    }

    gadget
}

/// Largest non-negative integer representable with `delta` balanced base-`b` digits, i.e.
/// with digits in $\[-b/2, b/2 - 1\]$:
/// $$\mathrm{top}(b, \delta) = \left(\frac{b}{2} - 1\right) \cdot \frac{b^\delta - 1}{b - 1}
///   = \left(\frac{b}{2} - 1\right) \sum_{i < \delta} b^i.$$
/// The set of all such integers, the *window*, is the block of $b^\delta$ consecutive integers
/// $\[-\frac{b}{2} \cdot \frac{b^\delta - 1}{b - 1}, \mathrm{top}(b, \delta)\]$. Requires $b$
/// even and $b \ge 2$ (the decomposition base is a power of two); panics if $b^\delta$
/// overflows a `u64`. For the default parameters $b = 16$, $\delta = 8$ the value is
/// $7 \cdot (16^8 - 1)/15 = 2004318071$.
pub fn window_top(b: u64, delta: usize) -> u64 {
    let b_pow = b.checked_pow(delta as u32).expect("b^delta overflows u64");
    (b / 2 - 1) * ((b_pow - 1) / (b - 1))
}

/// Map a residue $x \in \[0, q)$ to its representative in the balanced-digit window of `delta`
/// base-`b` digits (see [`window_top`]), returned as a wrapping two's-complement `u64`: $x$
/// itself if $x \le \mathrm{top}(b, \delta)$, and the negative integer $x - q$ (stored as
/// $2^{64} + x - q$) otherwise.
///
/// This is the first step of the gadget decomposition $\mathbf G^{-1}$ of a $\mathbb Z_q$
/// coefficient (paper, Section 2.1): the representative is then split into digits by
/// [`b_decomp`], and $\mathbf{G} \cdot \mathbf G^{-1}(x) \equiv x \pmod q$ holds because the
/// representative is congruent to $x$ and lies inside the window, where the digit recomposition
/// is exact over the integers.
///
/// Requires $b^\delta \ge q$ (checked in debug builds), so that the window, a block of
/// $b^\delta$ consecutive integers, contains a representative of every residue; this holds
/// whenever $\delta \ge \lceil \log_b q \rceil$. When $b^\delta > q$ a few residues have two
/// representatives in the window; this rule takes the non-negative one, which is the one of
/// smaller absolute value whenever $\mathrm{top}(b, \delta) \le q/2$ (true for the default
/// parameters). Panics if `x` is not a residue in $\[0, q)$.
pub fn to_window(x: CoeffType, q: u64, b: u64, delta: usize) -> CoeffType {
    assert!(x < q, "to_window expects a residue in [0, q), got {}", x);
    debug_assert!(b.checked_pow(delta as u32).expect("b^delta overflows u64") >= q, "b^delta < q: window too small");
    if x > window_top(b, delta) { x.wrapping_sub(q) } else { x }
}

/// Integer base-2 logarithm rounded up: `x.log()` is the smallest $k \ge 0$ with $2^k \ge x$,
/// i.e. $\lceil \log_2 x \rceil$ for $x \ge 1$, and $0$ for $x \in \lbrace 0, 1 \rbrace$.
///
/// Used for the bit length of the modulus (`q.log()`, the `logq` of [`rand_int`]), for the
/// number of variables of a table of length $2^\nu$ (`len.log()` $= \nu$ when `len` is a power
/// of two) and for $\alpha = \log_2 d$. For a length that is not a power of two the result is
/// the rounded-up exponent, not the floor.
pub trait Logarithm {
    /// The smallest $k$ with $2^k \ge$ `self` (see [`Logarithm`]).
    fn log(&self) -> usize;
}

/// $\lceil \log_2 x \rceil$ for `usize`; the search stops at $k = 64$.
impl Logarithm for usize {
    fn log(&self) -> usize {
        let mut logx = 0;
        while logx < 64 && *self > (1 << logx) { logx += 1; }
        logx
    }
}

/// $\lceil \log_2 x \rceil$ for `u64`; the search stops at $k = 64$.
impl Logarithm for u64 {
    fn log(&self) -> usize {
        let mut logx = 0;
        while logx < 64 && *self > (1 << logx) { logx += 1; }
        logx
    }
}

/// $\lceil \log_2 x \rceil$ for `u32`; the search stops at $k = 32$, so `u32::MAX.log() == 32`.
impl Logarithm for u32 {
    fn log(&self) -> usize {
        let mut logx = 0;
        while logx < 32 && *self > (1 << logx) { logx += 1; }
        logx
    }
}

/// Sample a uniform residue in $\[0, q)$ by rejection sampling: draw 64 random bits, keep the
/// low `logq` bits (mask $2^{\mathrm{logq}} - 1$) and accept the result if it is below $q$,
/// otherwise draw again.
///
/// Uniformity over $\[0, q)$ requires $2^{\mathrm{logq}} \ge q$, i.e. `logq >= q.log()` (see
/// [`Logarithm`]); with `logq = q.log()` the acceptance probability $q / 2^{\mathrm{logq}}$
/// exceeds $1/2$, so fewer than two draws are needed on average (for the default
/// $q = 2^{32} - 99$ it is essentially one). If `logq` is too small the output is uniform on
/// $\[0, 2^{\mathrm{logq}})$ only. Requires `logq <= 63`, as the code assumes a modulus of at
/// most 63 bits. This is the sampler behind [`rand_field`], the random commitment matrices and
/// the witness generator.
pub fn rand_int(q: CoeffType, logq: usize, rng: &mut impl Rng) ->  CoeffType {
    // assume q is at most 63 bits.
    let mask = (1 << logq) - 1;
    let mut rnd = q;

    // perform rejection sampling to produce output in range
    while rnd >= q {
        // generate random bits and mask to higher bits to get pow random bits
        // all 0 <= i < q produced with equal likliehood
        rnd = rng.next_u64() & mask;
    }

    rnd
}

/// Sample an element of $\mathbb F_{q^4}$ ([`ExtField`]) whose four $\mathbb Z_q$ coordinates
/// are drawn independently by [`rand_int`] with `logq = q.log()`: the draws $(a, b, c, d)$
/// become $(a + b u) + (c + d u) v$ in the tower of [`crate::arithmetic::field`]. The result is
/// uniform over the field when `q` is the field's modulus. This is how the challenges $\alpha$,
/// $\tau_0$, $\tau_1$ and the sumcheck challenges are derived from a Fiat-Shamir-seeded
/// `ChaCha12Rng` (see [`crate::arithmetic::fs`]).
pub fn rand_field(q: CoeffType, rng: &mut impl Rng) -> ExtField {
    let logq = q.log();

    let a = rand_int(q, logq, rng);
    let b = rand_int(q, logq, rng);
    let c = rand_int(q, logq, rng);
    let d = rand_int(q, logq, rng);

    Fq4::new(Fq2::new(Fq::from(a), Fq::from(b)), Fq2::new(Fq::from(c), Fq::from(d)))
}

/// Balanced base-$b$ digit decomposition of one short integer, $b = 2^{\mathrm{logb}}$: writes
/// into `out[0..delta]` the digits $a_0, \dots, a_{\delta - 1} \in \[-b/2, b/2 - 1\]$ (as
/// wrapping two's-complement `u64`) with
/// $$\sum_{i < \delta} a_i b^i \equiv x \pmod{b^\delta},$$
/// where `x` is read as a wrapping-signed integer. Digit $i$ is the low `logb` bits of the
/// running value, moved to the negative side when it is at least $b/2$; the running value is
/// then reduced by the digit and shifted down, so a borrow propagates to the next digit.
///
/// The congruence is an equality over the integers exactly when $x$ lies in the window of
/// [`window_top`], which [`to_window`] guarantees for residues; this is the per-coefficient
/// step of $\mathbf G^{-1}$, with the digit range $\[\lceil -b/2 \rceil, \lceil b/2 \rceil - 1\]$
/// of the paper's Section 2.1. The base is passed as its logarithm `logb`, unlike
/// [`to_window`], which takes $b$ itself. Requires $\delta \cdot \mathrm{logb} \le 64$: the
/// running value is shifted with a logical (unsigned) shift, so the digits of a negative input
/// are only correct while they are read from bits that the sign extension has not reached.
pub fn b_decomp(x: CoeffType, logb: usize, delta: usize, out: &mut [CoeffType]) {
    let mut cur = x;
    let base = 1 << logb;
    let mask = (1 << logb) - 1;
    let half_base = (1 << logb) >> 1;

    for i in 0..delta {
        let r = cur & mask;
        let a = if r >= half_base { r.wrapping_sub(base) } else { r };
        cur = (cur.wrapping_sub(a)) >> logb;
        out[i] = a;
    }
}

/// Lift an integer coefficient into $\mathbb F_{q^4}$ as the base-field element $x \bmod q$
/// (the coordinate $c_{0,0}$; all other coordinates zero).
///
/// The `u64` is reinterpreted as a two's-complement `i64` before reduction, so both integer
/// representations of this crate map to the intended element: a residue $x \in \[0, q)$ lifts
/// to $x$ (any value below $2^{63}$ does, and $q < 2^{32}$), and a wrapping-signed short
/// integer $-a$, stored as $2^{64} - a$, lifts to $-a \bmod q$. This is how the digits of the
/// next witness $\tilde{w}$ enter the sumcheck tables (paper, Section 4.3, Eq. (21)) and how the
/// interpolation nodes of [`crate::arithmetic::sumcheck::Univariate::eval`] are formed.
pub fn lift_int(x: CoeffType) -> ExtField {
    Fq4::new(Fq2::new(Fq::from(x as i64), Fq::from(0)), Fq2::ZERO)
}

/// Multiply $f \in \mathbb F_{q^4}$ by an integer: $f \cdot (x \bmod q)$, using the base-field
/// scalar multiplication of ark-ff (four $\mathbb Z_q$ multiplications instead of a full
/// $\mathbb F_{q^4}$ product; this is the mixed multiplication of the paper's Section 5.4).
///
/// Unlike [`lift_int`], `x` is reduced as an *unsigned* integer: a wrapping-signed negative
/// value $2^{64} - a$ would be read as $(2^{64} - a) \bmod q \ne -a \bmod q$. Callers pass
/// residues in $\[0, q)$ (polynomial coefficients evaluated at $\alpha$) or small non-negative
/// integers (the interpolation nodes $0, \dots, D$ of the sumcheck prover).
pub fn mul_int_field(x: CoeffType, f: ExtField) -> ExtField {
    f.mul_by_base_prime_field(&Fq::from(x))
}

/// The single-coordinate equality polynomial $\mathrm{eq}(a, b) = ab + (1 - a)(1 - b)$ of the
/// paper's Definition 2, so that $\mathrm{eq}(\tau, x) = \prod_j \mathrm{eq}(\tau_j, x_j)$
/// equals $1$ at $x = \tau$ and $0$ at every other point of the hypercube.
pub fn eq(a: ExtField, b: ExtField) -> ExtField {
    a * b + (ExtField::ONE - a) * (ExtField::ONE - b)
}

/// $\mathrm{eq}(a, \mathrm{bin})$ for a boolean second argument: $a$ if `bin == 1` and $1 - a$
/// otherwise. `bin` must be $0$ or $1$ (callers pass a single extracted bit, `(i >> j) & 1`);
/// any other value is treated as $0$.
pub fn eq_bin(a: ExtField, bin: usize) -> ExtField {
    if bin == 1 { a }
    else { ExtField::ONE - a }
}
