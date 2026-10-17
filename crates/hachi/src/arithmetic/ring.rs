//! Multiplication over the ring $\mathbf R_q = \mathbb Z_q\[X\]/(X^d + 1)$ and over the
//! polynomial ring $\mathbb Z_q\[X\]$, built on the 64-bit native NTT of the tfhe-ntt crate,
//! together with the direct products by sparse challenges (paper, Section 5.4, "Polynomial
//! Arithmetic" and "Optimized Sparse Polynomial Multiplication").
//!
//! ## Why a native NTT
//!
//! The modulus must satisfy $q \equiv 5 \pmod 8$ (paper, Section 2.1, Lemma 3, and the
//! field-to-ring transformation of Section 3), so it cannot be NTT-friendly: $q - 1$ is
//! divisible by $4$ but not by $8$, hence $\mathbb Z_q$ has no primitive $2d$-th root of unity
//! for $d \ge 4$ and no negacyclic NTT modulo $q$ exists. A [`Ring`] therefore computes products
//! *over the integers* and reduces modulo $q$ afterwards. tfhe-ntt's `native64` plan reduces
//! every `u64` coefficient modulo each of several NTT-friendly primes $p_i$, runs one negacyclic
//! NTT of length $N$ per prime, multiplies pointwise (for a matrix-vector product it also
//! accumulates a whole row in the NTT domain), inverts, and reconstructs the centered CRT
//! representative modulo $P = \prod_i p_i$, returned modulo $2^{64}$. Since $P \approx 2^{150}$
//! exceeds every integer that the inputs used here can produce, the value returned is the exact
//! product in $(\mathbb Z/2^{64}\mathbb Z)\[X\]/(X^N + 1)$, i.e. the product under wrapping
//! 64-bit arithmetic (paper, Section 5.4). Two feature paths select the plan:
//!
//! * default (`Plan32`, [`NttType`](crate::arithmetic::NttType) = `u32`): five primes below
//!   $2^{30}$, `0x3F5A0001`, `0x3F5D0001`, `0x3F760001`, `0x3F820001`, `0x3FAC0001`,
//!   $P \approx 2^{149.9}$; AVX2 or AVX-512 kernels are chosen at run time;
//! * feature `nightly` (`Plan52`, [`NttType`](crate::arithmetic::NttType) = `u64`): three primes
//!   below $2^{50}$, `0x3FFFFFE770001`, `0x3FFFFFEB90001`, `0x3FFFFFEC80001`, $P \approx 2^{150}$,
//!   with AVX-512 IFMA kernels; the plan needs the `tfhe-ntt/avx512` feature and IFMA support at
//!   run time, and [`Ring::init`] panics if `Plan52::try_new` finds neither.
//!
//! Both plans need $N$ to be a power of two (at least $32$ for `Plan32`, $16$ for `Plan52`) with
//! $2N \mid p_i - 1$ for every prime, which allows $N \le 2^{15}$.
//!
//! ## When the reduction modulo $q$ is exact
//!
//! The scheme only ever multiplies a full ring element (residues in $\[0, q)$) by a *short* one,
//! a gadget-decomposed vector or the response $\mathbf z$, stored as wrapping two's-complement
//! `u64` (paper, Section 5.4). The inverse transform (`_inv`) reads each reconstructed
//! coefficient as a signed `i64` and reduces it modulo $q$; this is exact if and only if the
//! integer coefficient of the product, with the short operand read as signed, lies in
//! $\[-2^{63}, 2^{63})$. For a matrix-vector product ([`Ring::mat_mul_vec`]) with $w$ columns,
//! ring dimension $d$, matrix entries below $q$ and a vector of infinity norm $S$, the
//! accumulated coefficient is bounded by $w d (q - 1) S$, so with $d = 2^{10}$ and $q \lt 2^{32}$
//! it suffices that $w S \lt 2^{21}$. The paper's Section 5.4 states the single-product version
//! of this ("no more than 22 bits" for the short operand); accumulating the $w$ columns of a
//! matrix tightens it by $\log_2 w$. For base-16 digits ($S = 8$) the condition holds for widths
//! up to $2^{18}$; e.g. $w = 2^{13}$ gives $2^{58}$. For the response $\mathbf z$ with
//! $\ell = 30$ ($m = r = 10$, $w = 2^m \delta = 2^{13}$, $S$ up to `params.z_bound`
//! $\approx 2^{15}$) the worst case is about $2^{70}$ and exceeds the bound; the prototype does
//! not check it, and for random data the sums are far smaller because the coefficients of
//! $\mathbf z$ have mixed signs, but exactness is not guaranteed in that case. The CRT bound is
//! never the limiting one: with all inputs below $2^{64}$ and the matrix entries below $2^{32}$,
//! the unsigned accumulated integer is below $w N 2^{96} \lt P/2$ whenever $w N \lt 2^{52}$.
//!
//! ## Cyclotomic and full mode
//!
//! A `Ring` is created in one of two modes ([`Ring::init`]):
//!
//! * `cyclotomic = true`: $N = d$; inputs and outputs are slices of length $d$ and every product
//!   is reduced modulo $X^d + 1$, i.e. taken in $\mathbf R_q$. Used for the commitments and for
//!   folding the witness.
//! * `cyclotomic = false` (*full mode*): $N = 2d$; inputs are slices of length $d$, zero-padded
//!   to $2d$ before the forward transform, and outputs are slices of length $2d$ holding the
//!   product in $\mathbb Z_q\[X\]$. Both factors have degree below $d$, so the product has degree
//!   at most $2d - 2$ and the reduction modulo $X^{2d} + 1$ never takes effect (paper,
//!   Section 5.4). The prover uses this mode to lift the verification equations of Eq. (20) from
//!   $\mathbf R_q$ to $\mathbb Z_q\[X\]$ (ring switching, Section 4.3), then splits every product
//!   into quotient and remainder modulo $X^d + 1$ with
//!   [`crate::arithmetic::poly::Poly::cyclotomic_div`].
//!
//! The scalar and sparse-challenge products ([`Ring::int_mul_poly`], [`Ring::chal_mul_poly`],
//! [`Ring::chal_mul_small_poly`]) do not use the NTT: a challenge with $k$ nonzero $\pm 1$
//! coefficients is multiplied in as $k$ signed, shifted copies of the other operand, with the
//! sign flip of $X^d = -1$ in cyclotomic mode (paper, Section 5.4).
//!
//! ## Summary
//!
//! | Method | Computes | Arithmetic | Effect on `out` |
//! |---|---|---|---|
//! | [`int_mul_poly`](Ring::int_mul_poly) | $a \cdot p$, $a \in \mathbb Z_q$ | `u128`, reduced | residues, added |
//! | [`chal_mul_poly`](Ring::chal_mul_poly) | $c \cdot p$ | shifts, reduced | residues, added |
//! | [`chal_mul_small_poly`](Ring::chal_mul_small_poly) | $c \cdot s$, $s$ short | shifts, wrapping | signed, added |
//! | [`int_vec_mul_poly_vec`](Ring::int_vec_mul_poly_vec) | $\sum_i a_i p_i$ | `u128`, reduced | residues, added |
//! | [`chal_mul_poly_vec`](Ring::chal_mul_poly_vec) | $c \cdot s_i$ for all $i$ | shifts, wrapping | signed, added |
//! | [`chal_vec_mul_poly_vec`](Ring::chal_vec_mul_poly_vec) | $\sum_i c_i p_i$ | shifts, reduced | residues, added |
//! | [`mat_fwd_ntt`](Ring::mat_fwd_ntt) | NTT of every entry of $\mathbf M$ | tfhe-ntt | (NTT form) |
//! | [`mat_mul_vec`](Ring::mat_mul_vec) | $\mathbf M \mathbf v$ | tfhe-ntt, signed reduction | residues, **overwritten** |

#[cfg(not(feature = "nightly"))]
use tfhe_ntt::native64::Plan32 as Native;

#[cfg(feature = "nightly")]
use tfhe_ntt::native64::Plan52 as Native;

use crate::arithmetic::{poly_chal::PChal, poly_mat::PMat, poly_mat_ntt::PMatNtt, poly_vec::PVec};

// #[cfg(feature = "nightly")]
// use crate::arithmetic::PVecNtt;

/// Arithmetic over $\mathbf R_q = \mathbb Z_q\[X\]/(X^d + 1)$ (cyclotomic mode) or over
/// $\mathbb Z_q\[X\]$ (full mode) for one modulus $q$ and one ring dimension $d$; see the module
/// documentation for the NTT strategy and the correctness conditions.
///
/// In cyclotomic mode every polynomial argument, input or output, is a slice of length $d$. In
/// full mode inputs are slices of length $d$ and outputs slices of length $2d$. These lengths are
/// asserted. An instance holds the tfhe-ntt plan (twiddle factors) for its transform length, so
/// it should be created once and reused; it is immutable and every method takes `&self`.
pub struct Ring {
    /// The tfhe-ntt negacyclic plan of length `output_dim` (`Plan32`, or `Plan52` with `nightly`).
    native: Native,
    /// The modulus $q$; coefficients are reduced into $\[0, q)$ after every NTT product.
    q: u64,
    /// Length $d$ of every input polynomial, the ring dimension.
    input_dim: usize,
    /// Transform length $N$ and length of every output polynomial: $d$ in cyclotomic mode, $2d$ in full mode.
    output_dim: usize,
    /// `true` for $\mathbf R_q$ (reduction modulo $X^d + 1$), `false` for products in $\mathbb Z_q\[X\]$.
    cyclotomic: bool
}

/// Construction, and the products that do not use the NTT.
impl Ring {
    /// Create the ring for modulus `q` and dimension `d` in cyclotomic mode (`cyclotomic = true`:
    /// $\mathbf R_q$, outputs of length $d$) or full mode (`cyclotomic = false`: $\mathbb Z_q\[X\]$,
    /// outputs of length $2d$). Builds the tfhe-ntt plan of length $N = d$ or $2d$, which must be
    /// a power of two between $32$ ($16$ with `nightly`) and $2^{15}$; panics otherwise, and
    /// with `nightly` also when AVX-512 IFMA is unavailable. Requires $q \lt 2^{63}$ for the
    /// signed reduction of `_inv`; the scheme uses $q \lt 2^{32}$.
    pub fn init(q: u64, d: usize, cyclotomic: bool) -> Self {
        let output_dim = if cyclotomic { d } else {2 * d};
        let native = Native::try_new(output_dim).unwrap();
        return Self { native, q, input_dim: d, output_dim, cyclotomic };
    }
    
    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Scalar product $\mathrm{out} \leftarrow \mathrm{out} + a \cdot p$ over $\mathbb Z_q$,
    /// coefficient by coefficient: `out[j] = (out[j] + a * poly[j]) mod q` for $j \lt d$, computed
    /// in `u128` so that no overflow is possible for any `u64` inputs, which are read as unsigned
    /// integers (a wrapping-signed input is *not* read as negative). `poly` has length $d$ and
    /// `out` the output length of the ring; in full mode the upper $d$ entries of `out` are left
    /// untouched. The touched entries end up as residues in $\[0, q)$. Cost: $d$ `u128`
    /// divisions. The prover uses it for $w_i = \mathbf a^T \mathbf f_i$ and
    /// $Y = \sum_i b_i w_i$ (paper, Section 4.2, the definition of $\mathbf w$ and Eq. (17)).
    pub fn int_mul_poly(&self, a: u64, poly: &[u64], out: &mut [u64]) {
        assert_eq!(self.input_dim, poly.len());
        assert_eq!(self.output_dim, out.len());

        let a = a as u128;
        
        for i in 0..self.input_dim {
            // expand to 128 bits to avoid overflow.
            out[i] = ((out[i] as u128 + a * poly[i] as u128) % self.q as u128) as u64;
        }
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Sparse product $\mathrm{out} \leftarrow \mathrm{out} + c \cdot p$ over $\mathbb Z_q$ for a
    /// challenge $c = \sum_{i \lt k} \varepsilon_i X^{e_i}$, $\varepsilon_i \in \lbrace -1, 1 \rbrace$
    /// ([`PChal`]): for each nonzero coefficient, the copy of $p$ shifted by $e_i$ is added to or
    /// subtracted from `out`. In cyclotomic mode a monomial $X^{e_i + j}$ with $e_i + j \ge d$
    /// wraps to $-X^{e_i + j - d}$; in full mode it lands at index $e_i + j \le 2d - 2$.
    ///
    /// Preconditions (asserted): `poly` has length $d$ with residues in $\[0, q)$, and `out` has
    /// the output length with residues in $\[0, q)$; the result is again fully reduced. Cost:
    /// $k d$ modular additions and no multiplication (paper, Section 5.4). The prover uses it in
    /// full mode for $\sum_i c_i \mathbf t_i$ when lifting Eq. (19)-(20) to $\mathbb Z_q\[X\]$.
    pub fn chal_mul_poly(&self, chal: &PChal, poly: &[u64], out: &mut [u64]) {
        assert_eq!(self.input_dim, poly.len());
        assert_eq!(self.output_dim, out.len());

        // iterate over non-zero coefficients of challenge
        for i in 0..chal.k() {
            let (exp, sign) = chal.get(i);
            
            // iterate over all coefficients of the polynomial
            for j in 0..self.input_dim {
                assert!(poly[j] < self.q);

                // get the exponent of the current product
                let mut exp = exp + j;
                let mut sign = sign;

                // cyclotomic reduction
                if self.cyclotomic && exp >= self.input_dim {
                    exp = exp - self.input_dim;
                    sign = !sign;
                }

                assert!(out[exp] < self.q);

                // multiply coefficients
                if sign {
                    out[exp] = out[exp] + poly[j];

                    if out[exp] >= self.q {
                        out[exp] = out[exp].wrapping_sub(self.q);
                    }
                }
                else {
                    if out[exp] >= poly[j] {
                        out[exp] = out[exp].wrapping_sub(poly[j]);
                    }
                    else {
                        out[exp] = (out[exp] + self.q).wrapping_sub(poly[j]);
                    }
                }
            }
        }
    }
    
    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Sparse product $\mathrm{out} \leftarrow \mathrm{out} + c \cdot s$ over $\mathbb Z$ for a
    /// short polynomial $s$: the same $k$ shifted additions and subtractions as
    /// [`chal_mul_poly`](Ring::chal_mul_poly), but in wrapping two's-complement arithmetic and
    /// without any reduction modulo $q$. Both `poly` and `out` are read and written as
    /// wrapping-signed integers. The result is the exact integer product as long as every
    /// coefficient of `out` stays in $\[-2^{63}, 2^{63})$, and it is the centered representative
    /// of $c \cdot s \bmod q$ as long as it stays below $q/2$ in absolute value, which is what the
    /// callers rely on: for a gadget-decomposed $s$,
    /// $\lVert c s \rVert_\infty \le \lVert c \rVert_1 \lVert s \rVert_\infty = k \lVert s \rVert_\infty$
    /// (paper, Section 2.1, Lemma 2). Lengths as in [`chal_mul_poly`](Ring::chal_mul_poly). This
    /// is the product used to fold the witness into $\mathbf z = \sum_i c_i \mathbf s_i$ (paper,
    /// Section 4.2).
    pub fn chal_mul_small_poly(&self, chal: &PChal, poly: &[u64], out: &mut [u64]) {
        assert_eq!(self.input_dim, poly.len());
        assert_eq!(self.output_dim, out.len());

        // iterate over non-zero coefficients of challenge
        for i in 0..chal.k() {
            let (exp, sign) = chal.get(i);

            // iterate over all coefficients of the polynomial
            for j in 0..self.input_dim {
                // get the exponent of the current product
                let mut exp = exp + j;
                let mut sign = sign;

                // cyclotomic reduction
                if self.cyclotomic && exp >= self.input_dim {
                    exp = exp - self.input_dim;
                    sign = !sign;
                }

                // multiply coefficients
                if sign {
                    out[exp] = out[exp].wrapping_add(poly[j]);
                }
                else {
                    out[exp] = out[exp].wrapping_sub(poly[j]);
                }
            }
        }
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Inner product of a vector of scalars with a vector of polynomials,
    /// $\mathrm{out}\[\mathrm{index}\] \leftarrow \mathrm{out}\[\mathrm{index}\] + \sum_i a_i \cdot p_i$
    /// over $\mathbb Z_q$, accumulated with [`int_mul_poly`](Ring::int_mul_poly) into the single
    /// element `index` of `out`. `int_vec` and `p_vec` have the same length (asserted); the
    /// elements of `p_vec` have length $d$ and those of `out` the output length. Inputs are read
    /// as unsigned integers and the result is a residue vector. Cost: $d$ `u128` divisions per
    /// element of `p_vec`. The prover uses it for $w_i = \mathbf a^T \mathbf f_i$, the inner
    /// product of a chunk of the witness with the evaluation vector $\mathbf a$ (paper,
    /// Section 4.2).
    pub fn int_vec_mul_poly_vec(&self, int_vec: &Vec<u64>, p_vec: &PVec, out: &mut PVec, index: usize) {
        assert_eq!(int_vec.len(), p_vec.length());

        for i in 0..p_vec.length() {
            self.int_mul_poly(int_vec[i], p_vec.element(i), out.mut_element(index));
        }
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Elementwise sparse product of a challenge with a vector of short polynomials,
    /// $\mathrm{out}\[\mathrm{index} + i\] \leftarrow \mathrm{out}\[\mathrm{index} + i\] + c \cdot s_i$
    /// for every $i$, in wrapping two's-complement arithmetic without reduction
    /// ([`chal_mul_small_poly`](Ring::chal_mul_small_poly)); the correctness conditions of that
    /// method apply to each entry. `poly_vec` and `out` must have the same length (asserted), so
    /// any `index > 0` makes the last elements fall outside `out` and the call panic: in practice
    /// `index` is always `0`. Calling it once per challenge $c_i$ with the decomposed chunk
    /// $\mathbf s_i$ and the same `out` accumulates the folded witness
    /// $\mathbf z = \sum_{i \le 2^r} c_i \mathbf s_i$ of the paper's Section 4.2 (Fig. 3).
    pub fn chal_mul_poly_vec(&self, chal: &PChal, poly_vec: &PVec, out: &mut PVec, index: usize) {
        assert_eq!(poly_vec.length(), out.length());

        for i in 0..poly_vec.length() {
            self.chal_mul_small_poly(chal, poly_vec.element(i), out.mut_element(index + i));
        }
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Inner product of a vector of challenges with a vector of polynomials,
    /// $\mathrm{out}\[\mathrm{index}\] \leftarrow \mathrm{out}\[\mathrm{index}\] + \sum_i c_i \cdot p_i$
    /// over $\mathbb Z_q$, accumulated with [`chal_mul_poly`](Ring::chal_mul_poly) into the single
    /// element `index` of `out`: `chals` and `poly` have the same length (asserted), the elements
    /// of `poly` are residue vectors of length $d$, and the result is fully reduced. Cost: $k d$
    /// additions per element. The prover uses it in full mode for $\sum_i c_i w_i \in \mathbb Z_q\[X\]$,
    /// the right-hand side of the relation $\mathbf a^T \mathbf G \mathbf z = \sum_i c_i w_i$ of
    /// the paper's Eq. (18), lifted to $\mathbb Z_q\[X\]$.
    pub fn chal_vec_mul_poly_vec(&self, chals: &Vec<PChal>, poly: &PVec, out: &mut PVec, index: usize) {
        assert_eq!(chals.len(), poly.length());

        for i in 0..poly.length() {
            self.chal_mul_poly(&chals[i], poly.element(i), out.mut_element(index));
        }
    }
}

#[cfg(not(feature = "nightly"))]
/// NTT-based products on the default `Plan32` path (five primes, `u32` residues). The `nightly`
/// block is the same code for `Plan52` with three residue vectors instead of five.
impl Ring {
    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Forward transform of one polynomial: `p` (length $d$, any `u64` coefficients; tfhe-ntt
    /// reduces them modulo each prime, so residues and wrapping-signed values are both fine) is
    /// written as its five NTT-domain residue vectors `p0..p4` of length $N$ (the output length,
    /// asserted). In full mode `p` is first zero-padded to $2d$. Cost: five NTTs of length $N$.
    fn _fwd(&self, p: &[u64], p0: &mut [u32], p1: &mut [u32], p2: &mut [u32], p3: &mut [u32], p4: &mut [u32]) {
        assert_eq!(self.input_dim, p.len());
        assert_eq!(self.output_dim, p0.len());
        assert_eq!(self.output_dim, p1.len());
        assert_eq!(self.output_dim, p2.len());
        assert_eq!(self.output_dim, p3.len());
        assert_eq!(self.output_dim, p4.len());

        if self.cyclotomic {
            self.native.fwd(p, p0, p1, p2, p3, p4);
        }
        else {
            // if non-cyclotomic, pad p
            let mut p_pad = vec![0u64; self.output_dim];
            
            for i in 0..self.input_dim {
                p_pad[i] = p[i];
            }

            self.native.fwd(&p_pad, p0, p1, p2, p3, p4);
        }
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Inverse transform of five NTT-domain residue vectors into the coefficient vector `p` of
    /// length $N$: each residue vector is first multiplied by $N^{-1} \bmod p_i$ (`normalize`,
    /// since tfhe-ntt's `inv` is unnormalized), then inverse-transformed and CRT-reconstructed
    /// into the exact integer result modulo $2^{64}$; finally every coefficient is read as a
    /// signed `i64` and reduced to a residue in $\[0, q)$. The reduction is exact iff the exact
    /// integer lies in $\[-2^{63}, 2^{63})$ (module documentation). The residue vectors are
    /// modified in place. Cost: five inverse NTTs of length $N$ plus $N$ signed divisions.
    fn _inv(&self, p: &mut [u64], p0: &mut [u32], p1: &mut [u32], p2: &mut [u32], p3: &mut [u32], p4: &mut [u32]) {
        assert_eq!(self.output_dim, p.len());
        assert_eq!(self.output_dim, p0.len());
        assert_eq!(self.output_dim, p1.len());
        assert_eq!(self.output_dim, p2.len());
        assert_eq!(self.output_dim, p3.len());
        assert_eq!(self.output_dim, p4.len());
        
        self.native.ntt_0().normalize(p0);
        self.native.ntt_1().normalize(p1);
        self.native.ntt_2().normalize(p2);
        self.native.ntt_3().normalize(p3);
        self.native.ntt_4().normalize(p4);

        self.native.inv(p, p0, p1, p2, p3, p4);

        for j in 0..self.output_dim {
            let tmp = p[j] as i64 % self.q as i64;
            p[j] = if tmp >= 0 { tmp as u64 } else { (tmp + self.q as i64) as u64 };
        }
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Pointwise multiply-accumulate in the NTT domain, $s_i \leftarrow s_i + l_i \odot r_i \bmod p_i$
    /// for each of the five primes; all slices have length $N$. This is what lets a whole row of
    /// a matrix-vector product be summed before a single inverse transform.
    fn _mul_acc(&self, 
        l0: &[u32], l1: &[u32], l2: &[u32], l3: &[u32], l4: &[u32],
        r0: &[u32], r1: &[u32], r2: &[u32], r3: &[u32], r4: &[u32],
        s0: &mut [u32], s1: &mut [u32], s2: &mut [u32], s3: &mut [u32], s4: &mut [u32],
    ) {
        self.native.ntt_0().mul_accumulate(s0, l0, r0);
        self.native.ntt_1().mul_accumulate(s1, l1, r1);
        self.native.ntt_2().mul_accumulate(s2, l2, r2);
        self.native.ntt_3().mul_accumulate(s3, l3, r3);
        self.native.ntt_4().mul_accumulate(s4, l4, r4);
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Transform every entry of `mat` into NTT form, entry by entry with `_fwd`; `mat_ntt` must
    /// have the same height and width (asserted) and transform length $N$. The public matrices
    /// $\mathbf A$, $\mathbf B$, $\mathbf D$ are transformed once and reused for every product
    /// (paper, Section 5.4, keeping re-used values in NTT form). Cost: five NTTs per entry.
    pub fn mat_fwd_ntt(&self, mat: &PMat, mat_ntt: &mut PMatNtt) {
        assert_eq!(mat.height(), mat_ntt.height());
        assert_eq!(mat.width(), mat_ntt.width());

        for i in 0..mat.height() {
            for j in 0..mat.width() {
                let (p0, p1, p2, p3, p4) = mat_ntt.mut_element(i, j);
                self._fwd(mat.element(i, j), p0, p1, p2, p3, p4);
            }
        }
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Matrix-vector product $\mathbf M \mathbf v$ with $\mathbf M$ in NTT form and $\mathbf v$ in
    /// coefficient form: for every row, `out[index + row]` is **overwritten** with
    /// $\sum_{\mathrm{col}} \mathbf M\[\mathrm{row}, \mathrm{col}\] \cdot \mathbf v\[\mathrm{col}\]$,
    /// reduced modulo $X^d + 1$ in cyclotomic mode or taken in $\mathbb Z_q\[X\]$ (length $2d$)
    /// in full mode, as residues in $\[0, q)$.
    ///
    /// Each entry of `vec` is transformed once, on the fly, and its products with the whole column
    /// are accumulated in the NTT domain (`_mul_acc`), so the cost is `width` forward and
    /// `height` inverse transforms plus `height * width` pointwise products; transforming the
    /// vector up front was measured to be slower, presumably for cache reasons. `mat.width()`
    /// must equal `vec.length()` (asserted); the elements of `vec` have length $d$, those of
    /// `out` the output length, and `out` must have at least `index + height` elements.
    ///
    /// `vec` may hold residues or wrapping-signed short integers; the result is exact iff the
    /// integer coefficients of the accumulated products lie in $\[-2^{63}, 2^{63})$, i.e. for a
    /// vector of infinity norm $S$ and width $w$ whenever $w d (q - 1) S \lt 2^{63}$ (module
    /// documentation). This computes the Ajtai commitments $\mathbf t_i = \mathbf A \mathbf s_i$,
    /// the outer commitments $\mathbf u = \mathbf B \hat{\mathbf t}$ and $\mathbf v = \mathbf D \hat{\mathbf w}$
    /// (paper, Eq. (13)-(14), (16)) and, in full mode, the products $\mathbf A \mathbf z$,
    /// $\mathbf B \hat{\mathbf t}$ and $\mathbf D \hat{\mathbf w}$ of Eq. (19)-(20) lifted to
    /// $\mathbb Z_q\[X\]$.
    pub fn mat_mul_vec(&self, mat: &PMatNtt, vec: &PVec, out: &mut PVec, index: usize) {
        use crate::arithmetic::poly_vec_ntt::PVecNtt;

        assert_eq!(mat.width(), vec.length());

        // Vectors to store forward NTT of elements as required.
        // Only perform forward NTT on elements as required (rather than the whole vector at the start).
        // Much quicker (possibly to do with memory access to larger vector).
        let mut r0 = vec![0u32; self.output_dim];
        let mut r1 = vec![0u32; self.output_dim];
        let mut r2 = vec![0u32; self.output_dim];
        let mut r3 = vec![0u32; self.output_dim];
        let mut r4 = vec![0u32; self.output_dim];

        // Vector to store product in NTT domain
        let mut out_ntt = PVecNtt::zero(mat.height(), self.output_dim);

        // Iterate over columns.
        for col in 0..mat.width() {
            // Perform forward NTT on corresponding element of the vector.
            self._fwd(vec.element(col), &mut r0, &mut r1, &mut r2, &mut r3, &mut r4);

            // Iterate over over rows.
            for row in 0..mat.height() {
                // Get corresponding element of the matrix.
                let (l0, l1, l2, l3, l4) = mat.element(row, col);

                // Get corresponding element of the sum.
                let (s0, s1, s2, s3, s4) = out_ntt.mut_element(row);

                // Pointwise multiplication accumulated in the sum.
                self._mul_acc(
                    l0, l1, l2, l3, l4, 
                    &r0, &r1, &r2, &r3, &r4, 
                    s0, s1, s2, s3, s4
                );
            }
        }

        // Inverse NTT on the result.
        for i in 0..out_ntt.length() {
            let (p0, p1, p2, p3, p4) = out_ntt.mut_element(i);
            self._inv(out.mut_element(index + i), p0, p1, p2, p3, p4);
        }
    }
}

#[cfg(feature = "nightly")]
/// NTT-based products on the `Plan52` path (three primes, `u64` residues, AVX-512 IFMA). Each
/// method is the `Plan32` one with three residue vectors instead of five; see that block for
/// the full documentation.
impl Ring {    
    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Forward transform of one polynomial `p` (length $d$, any `u64` coefficients) into its three
    /// NTT-domain residue vectors of length $N$; zero-pads to $2d$ in full mode.
    fn _fwd(&self, p: &[u64], p0: &mut [u64], p1: &mut [u64], p2: &mut [u64]) {
        assert_eq!(self.input_dim, p.len());
        assert_eq!(self.output_dim, p0.len());
        assert_eq!(self.output_dim, p1.len());
        assert_eq!(self.output_dim, p2.len());

        if self.cyclotomic {
            self.native.fwd(p, p0, p1, p2);
        }
        else {
            // if non-cyclotomic, pad p
            let mut p_pad = vec![0u64; self.output_dim];
            
            for i in 0..self.input_dim {
                p_pad[i] = p[i];
            }

            self.native.fwd(&p_pad, p0, p1, p2);
        }
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Inverse transform of three residue vectors into `p` (length $N$): normalize by $N^{-1}$,
    /// inverse NTT, CRT reconstruction modulo $2^{64}$, then signed reduction into $\[0, q)$
    /// (exact iff the integer result lies in $\[-2^{63}, 2^{63})$).
    fn _inv(&self, p: &mut [u64], p0: &mut [u64], p1: &mut [u64], p2: &mut [u64]) {
        assert_eq!(self.output_dim, p.len());
        assert_eq!(self.output_dim, p0.len());
        assert_eq!(self.output_dim, p1.len());
        assert_eq!(self.output_dim, p2.len());
        
        self.native.ntt_0().normalize(p0);
        self.native.ntt_1().normalize(p1);
        self.native.ntt_2().normalize(p2);

        self.native.inv(p, p0, p1, p2);

        for j in 0..self.output_dim {
            let tmp = p[j] as i64 % self.q as i64;
            p[j] = if tmp >= 0 { tmp as u64 } else { (tmp + self.q as i64) as u64 };
        }
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Pointwise multiply-accumulate $s_i \leftarrow s_i + l_i \odot r_i \bmod p_i$ for each of
    /// the three primes.
    fn _mul_acc(&self, 
        l0: &[u64], l1: &[u64], l2: &[u64],
        r0: &[u64], r1: &[u64], r2: &[u64],
        s0: &mut [u64], s1: &mut [u64], s2: &mut [u64],
    ) {
        self.native.ntt_0().mul_accumulate(s0, l0, r0);
        self.native.ntt_1().mul_accumulate(s1, l1, r1);
        self.native.ntt_2().mul_accumulate(s2, l2, r2);
    }

    /// Transform every entry of `mat` into NTT form (`mat_ntt` of the same shape, asserted).
    pub fn mat_fwd_ntt(&self, mat: &PMat, mat_ntt: &mut PMatNtt) {
        assert_eq!(mat.height(), mat_ntt.height());
        assert_eq!(mat.width(), mat_ntt.width());

        for i in 0..mat.height() {
            for j in 0..mat.width() {
                let (p0, p1, p2) = mat_ntt.mut_element(i, j);
                self._fwd(mat.element(i, j), p0, p1, p2);
            }
        }
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Matrix-vector product $\mathbf M \mathbf v$, **overwriting** `out[index + row]` for every
    /// row with the reduced product; same preconditions and exactness condition as the `Plan32`
    /// version.
    pub fn mat_mul_vec(&self, mat: &PMatNtt, vec: &PVec, out: &mut PVec, index: usize) {
        use crate::arithmetic::poly_vec_ntt::PVecNtt;

        assert_eq!(mat.width(), vec.length());

        // Vectors to store forward NTT of elements as required.
        // Only perform forward NTT on elements as required (rather than the whole vector at the start).
        // Much quicker (possibly to do with memory access to larger vector).
        let mut r0 = vec![0u64; self.output_dim];
        let mut r1 = vec![0u64; self.output_dim];
        let mut r2 = vec![0u64; self.output_dim];

        // Vector to store product in NTT domain
        let mut out_ntt = PVecNtt::zero(mat.height(), self.output_dim);

        // Iterate over columns.
        for col in 0..mat.width() {
            // Perform forward NTT on corresponding element of the vector.
            self._fwd(vec.element(col), &mut r0, &mut r1, &mut r2);

            // Iterate over over rows.
            for row in 0..mat.height() {
                // Get corresponding element of the matrix.
                let (l0, l1, l2) = mat.element(row, col);

                // Get corresponding element of the sum.
                let (s0, s1, s2) = out_ntt.mut_element(row);

                // Pointwise multiplication accumulated in the sum.
                self._mul_acc(l0, l1, l2, &r0, &r1, &r2, s0, s1, s2);
            }
        }

        // Inverse NTT on the result.
        for i in 0..out_ntt.length() {
            let (p0, p1, p2) = out_ntt.mut_element(i);
            self._inv(out.mut_element(index + i), p0, p1, p2);
        }
    }
}

#[cfg(test)]
/// Tests for Ring.
mod test_ring {
    use rand::rng;

    use super::*;
    // use default modulus for all tests.
    use crate::hachi::setup::Q;
    use crate::arithmetic::{poly::Poly, poly_vec::PVec, poly_vec_ntt::PVecNtt};
    use crate::arithmetic::utils::{rand_int, Logarithm};

    const P0: u64 = 0b0011_1111_0101_1010_0000_0000_0000_0001;
    const P1: u64 = 0b0011_1111_0101_1101_0000_0000_0000_0001;
    const P2: u64 = 0b0011_1111_0111_0110_0000_0000_0000_0001;

    #[cfg(not(feature = "nightly"))]
    const P3: u64 = 0b0011_1111_1000_0010_0000_0000_0000_0001;

    #[cfg(not(feature = "nightly"))]
    const P4: u64 = 0b0011_1111_1010_1100_0000_0000_0000_0001;

    /// add mod q
    fn add_64(a: u64, b: u64, q: u64) -> u64 {
        ((a % q) + (b % q)) % q
    }

    /// add mod q
    fn add_32(a: u32, b: u32, q: u64) -> u32 {
        add_64(a as u64, b as u64, q) as u32
    }

    /// subtract mod q
    fn sub_64(a: u64, b: u64, q: u64) -> u64 {
        let a = a % q;
        let b = b % q;

        if a >= b {
            a.wrapping_sub(b)
        }
        else {
            (a + q).wrapping_sub(b)
        }
    }

    /// subtract mod q
    fn sub_32(a: u32, b: u32, q: u64) -> u32 {
        sub_64(a as u64, b as u64, q) as u32
    }

    /// multiply mod q
    fn mul_64(a: u64, b: u64, q: u64) -> u64 {
        let a = (a % q) as u128;
        let b = (b % q) as u128;
        ((a * b) % q as u128) as u64
    }

    /// multiply mod q
    fn mul_32(a: u32, b: u32, q: u64) -> u32 {
        mul_64(a as u64, b as u64, q) as u32
    }

    /// schoolbook polynomial multiplication
    fn poly_mul(p1: &[u64], p2: &[u64], p_out: &mut [u64], q: u64) {
        let d = p1.len();
        assert_eq!(d, p2.len());
        assert_eq!(2*d, p_out.len());

        for i in 0..d {
            for j in 0..d {
                p_out[i + j] = add_64(p_out[i + j], mul_64(p1[i], p2[j], q), q);
            }
        }
    }

    /// schoolbook polynomial multiplication
    fn poly_mul_cyclotomic(p1: &[u64], p2: &[u64], p_out: &mut [u64], q: u64) {
        let d = p1.len();
        assert_eq!(d, p2.len());
        assert_eq!(d, p_out.len());

        let mut prod = vec![0; 2 * d];

        for i in 0..d {
            for j in 0..d {
                prod[i + j] = add_64(prod[i + j], mul_64(p1[i], p2[j], q), q);
            }
        }

        for i in 0..d {
            p_out[i] = sub_64(prod[i], prod[d + i], q);
        }
    }

    #[test]
    fn test_test_methods() {
        let a = Q - 10224;
        let b = 10224;
        let c = 24243;

        assert_eq!(0, add_64(a, b, Q));
        assert_eq!(b + c, add_64(b, c, Q));
        assert_eq!(c.wrapping_sub(b), add_64(a, c ,Q));

        assert_eq!(a, sub_64(0, b, Q));
        assert_eq!(c - b, sub_64(c, b, Q));
        assert_eq!(a.wrapping_sub(c), sub_64(a, c, Q));
        assert_eq!((b + Q).wrapping_sub(c), sub_64(b, c, Q));
        
        assert_eq!((a * b) % Q, mul_64(a, b, Q));
        assert_eq!(b * c, mul_64(b, c, Q));        
        assert_eq!(((a as u128 * a as u128) % Q as u128) as u64, mul_64(a, a, Q));    

        let a = a as u32;
        let b = b as u32;
        let c = c as u32;

        assert_eq!(0, add_32(a, b, Q));
        assert_eq!(b + c, add_32(b, c, Q));
        assert_eq!(c.wrapping_sub(b), add_32(a, c, Q));

        assert_eq!(a, sub_32(0, b, Q));
        assert_eq!(c - b, sub_32(c, b, Q));
        assert_eq!(a.wrapping_sub(c), sub_32(a, c, Q));
        assert_eq!((Q as u32).wrapping_sub(c.wrapping_sub(b)), sub_32(b, c, Q));

        assert_eq!(((a  as u64 * b as u64) % Q) as u32, mul_32(a, b, Q));
        assert_eq!(b * c, mul_32(b, c, Q));        
        assert_eq!(((a as u128 * a as u128) % Q as u128) as u32, mul_32(a, a, Q));

        let p1_coeffs: [i64; 4] = [-1, -2, 5432, 43];
        let p2_coeffs: [i64; 4] = [6, 602, -5, 4];
        let expected_coeffs: [i64; 8] = [-6, -614, 31393, 3270328, -1282, 21513, 172, 0];

        let mut p1 = [0u64; 4];
        let mut p2 = [0u64; 4];
        let mut expected = [0u64; 8];

        for i in 0..4 {
            if p1_coeffs[i] < 0 {
                p1[i] = (p1_coeffs[i] + Q as i64) as u64;
            }
            else {
                p1[i] = p1_coeffs[i] as u64;
            }

            if p2_coeffs[i] < 0 {
                p2[i] = (p2_coeffs[i] + Q as i64) as u64;
            }
            else {
                p2[i] = p2_coeffs[i] as u64;
            }
        }

        for i in 0..8 {
            if expected_coeffs[i] < 0 {
                expected[i] = (expected_coeffs[i] + Q as i64) as u64;
            }
            else {
                expected[i] = expected_coeffs[i] as u64;
            }
        }

        let mut actual = [0u64; 8];
        poly_mul(&p1, &p2, &mut actual, Q);
        assert_eq!(expected, actual);

        let mut expected_cyclotomic = [0u64; 4];
        let mut _quo = [0u64; 4];
        expected.as_slice().cyclotomic_div(Q, &mut _quo, &mut expected_cyclotomic);

        let mut actual = [0u64; 4];
        poly_mul_cyclotomic(&p1, &p2, &mut actual, Q);
        assert_eq!(expected_cyclotomic, actual);
    }

    #[cfg(not(feature = "nightly"))]
    #[test]
    fn test_fwd_inv() {
        let d = 512;
        let ring = Ring::init(Q, d, true);

        // do forward NTT
        let p = PVec::rand(1, d, Q, [1; 32]);
        let mut ntt = PVecNtt::zero(1, d);
        let (a0, a1, a2, a3, a4) = ntt.mut_element(0);
        ring._fwd(p.element(0), a0, a1, a2, a3, a4);
        

        // do inverse NTT
        let mut p_out = PVec::zero(1, d);
        ring._inv(p_out.mut_element(0), a0, a1, a2, a3, a4);

        // check we get the same thing back
        assert_eq!(p.element(0), p_out.element(0));
    }

    #[cfg(not(feature = "nightly"))]
    #[test]
    fn test_fwd_sum_inv() {
        let d = 512;
        let ring = Ring::init(Q, d, true);

        // generate many polynomials and sum them in coefficient form
        let p_vec = PVec::rand(1024, d, Q, [1; 32]);
        let mut sum_actual = vec![0u64; d];

        for i in 0..p_vec.length() {
            let p = p_vec.element(i);

            for j in 0..d {
                sum_actual[j] = add_64(sum_actual[j], p[j], Q);
            }
        }

        // sum them over NTT form
        let mut p_vec_ntt = PVecNtt::zero(1024, d);
        let mut s0 = vec![0u32; d];
        let mut s1 = vec![0u32; d];
        let mut s2 = vec![0u32; d];
        let mut s3 = vec![0u32; d];
        let mut s4 = vec![0u32; d];
        
        for i in 0..p_vec.length() {
            let p = p_vec.element(i);

            // do forward NTT
            let (a0, a1, a2, a3, a4) = p_vec_ntt.mut_element(i);
            ring._fwd(p, a0, a1, a2, a3, a4);

            // sum over NTT domains
            for j in 0..d {
                s0[j] = add_32(s0[j], a0[j], P0);
                s1[j] = add_32(s1[j], a1[j], P1);
                s2[j] = add_32(s2[j], a2[j], P2);
                s3[j] = add_32(s3[j], a3[j], P3);
                s4[j] = add_32(s4[j], a4[j], P4);
            }
        }

        // do inverse NTT on sum
        let mut sum_actual = vec![0u64; d];
        ring._inv(&mut sum_actual, &mut s0, &mut s1, &mut s2, &mut s3, &mut s4);

        // check we get the same thing back
        assert_eq!(sum_actual, sum_actual);
    }

    #[cfg(not(feature = "nightly"))]
    #[test]
    fn test_fwd_mul_inv() {
        // Note: multiplication in our ring is only guaranteed if each coefficient of the product does not exceed
        // 63 bits.
        let d = 32;
        let ring = Ring::init(Q, d, true);

        // generate arbitrary element and small element (e.g. as in commitment matrix * decomposed witness)
        let p_vec_1 = PVec::rand(1, d, Q, [1; 32]);
        let p_vec_2 = PVec::rand(1, d, 16, [1; 32]);

        let mut mul_expected = vec![0u64; d];
        poly_mul_cyclotomic(p_vec_1.element(0), p_vec_2.element(0), &mut mul_expected, Q);

        // do forward NTT on both elements
        let mut a0 = vec![0u32; d];
        let mut a1 = vec![0u32; d];
        let mut a2 = vec![0u32; d];
        let mut a3 = vec![0u32; d];
        let mut a4 = vec![0u32; d];
        ring._fwd(p_vec_1.element(0), &mut a0, &mut a1, &mut a2, &mut a3, &mut a4);

        let mut b0 = vec![0u32; d];
        let mut b1 = vec![0u32; d];
        let mut b2 = vec![0u32; d];
        let mut b3 = vec![0u32; d];
        let mut b4 = vec![0u32; d];       
        ring._fwd(p_vec_2.element(0), &mut b0, &mut b1, &mut b2, &mut b3, &mut b4);

        // pointwise multiply
        let mut m0 = vec![0u32; d];
        let mut m1 = vec![0u32; d];
        let mut m2 = vec![0u32; d];
        let mut m3 = vec![0u32; d];
        let mut m4 = vec![0u32; d];

        for j in 0..d {
            m0[j] = mul_32(a0[j], b0[j], P0);
            m1[j] = mul_32(a1[j], b1[j], P1);
            m2[j] = mul_32(a2[j], b2[j], P2);
            m3[j] = mul_32(a3[j], b3[j], P3);
            m4[j] = mul_32(a4[j], b4[j], P4);
        }

        // do inverse NTT and pointwise product
        let mut mul_actual = vec![0u64; d];
        ring._inv(&mut mul_actual, &mut m0, &mut m1, &mut m2, &mut m3, &mut m4);

        assert_eq!(mul_expected, mul_actual);
    }

    #[cfg(not(feature = "nightly"))]
    #[test]
    fn test_mul_acc() {
        let d = 32;
        let ring = Ring::init(Q, d, true);
        let length = 100;

        // generate some random NTT domain polynomials
        let mut rng = rng();
        let mut p_vec_ntt_1 = PVecNtt::zero(length, d);
        let mut p_vec_ntt_2 = PVecNtt::zero(length, d);

        for i in 0..length {
            let (a0, a1, a2, a3, a4) = p_vec_ntt_1.mut_element(i);
            let (b0, b1, b2, b3, b4) = p_vec_ntt_2.mut_element(i);

            for j in 0..d {
                a0[j] = rand_int(P4, P4.log(), &mut rng) as u32;
                a1[j] = rand_int(P4, P4.log(), &mut rng) as u32;
                a2[j] = rand_int(P4, P4.log(), &mut rng) as u32;
                a3[j] = rand_int(P4, P4.log(), &mut rng) as u32;
                a4[j] = rand_int(P4, P4.log(), &mut rng) as u32;

                b0[j] = rand_int(P4, P4.log(), &mut rng) as u32;
                b1[j] = rand_int(P4, P4.log(), &mut rng) as u32;
                b2[j] = rand_int(P4, P4.log(), &mut rng) as u32;
                b3[j] = rand_int(P4, P4.log(), &mut rng) as u32;
                b4[j] = rand_int(P4, P4.log(), &mut rng) as u32;
            }
        }

        // do pointwise product and sum
        let mut s0 = vec![0u32; d];
        let mut s1 = vec![0u32; d];
        let mut s2 = vec![0u32; d];
        let mut s3 = vec![0u32; d];
        let mut s4 = vec![0u32; d];

        let mut p0 = vec![0u32; d];
        let mut p1 = vec![0u32; d];
        let mut p2 = vec![0u32; d];
        let mut p3 = vec![0u32; d];
        let mut p4 = vec![0u32; d];

        for i in 0..p_vec_ntt_1.length() {
            let (a0, a1, a2, a3, a4) = p_vec_ntt_1.element(i);
            let (b0, b1, b2, b3, b4) = p_vec_ntt_2.element(i);

            // pointwise multiply and sum
            for j in 0..d {
                s0[j] = add_32(s0[j], mul_32(a0[j], b0[j], P0), P0);
                s1[j] = add_32(s1[j], mul_32(a1[j], b1[j], P1), P1);
                s2[j] = add_32(s2[j], mul_32(a2[j], b2[j], P2), P2);
                s3[j] = add_32(s3[j], mul_32(a3[j], b3[j], P3), P3);
                s4[j] = add_32(s4[j], mul_32(a4[j], b4[j], P4), P4);
            }

            // use method
            ring._mul_acc(
                a0, a1, a2, a3, a4, 
                b0, b1, b2, b3, b4, 
                &mut p0, &mut p1, &mut p2, &mut p3, &mut p4);
        }

        assert_eq!(s0, p0);
        assert_eq!(s1, p1);
        assert_eq!(s2, p2);
        assert_eq!(s3, p3);
        assert_eq!(s4, p4);
    }

   #[cfg(feature = "nightly")]
    #[test]
    fn test_fwd_inv() {
        let d = 512;
        let ring = Ring::init(Q, d, true);

        // do forward NTT
        let p = PVec::rand(1, d, Q, [1; 32]);
        let mut ntt = PVecNtt::zero(1, d);
        let (a0, a1, a2) = ntt.mut_element(0);
        ring._fwd(p.element(0), a0, a1, a2);
        

        // do inverse NTT
        let mut p_out = PVec::zero(1, d);
        ring._inv(p_out.mut_element(0), a0, a1, a2);

        // check we get the same thing back
        assert_eq!(p.element(0), p_out.element(0));
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_fwd_sum_inv() {
        let d = 512;
        let ring = Ring::init(Q, d, true);

        // generate many polynomials and sum them in coefficient form
        let p_vec = PVec::rand(1024, d, Q, [1; 32]);
        let mut sum_actual = vec![0u64; d];

        for i in 0..p_vec.length() {
            let p = p_vec.element(i);

            for j in 0..d {
                sum_actual[j] = add_64(sum_actual[j], p[j], Q);
            }
        }

        // sum them over NTT form
        let mut p_vec_ntt = PVecNtt::zero(1024, d);
        let mut s0 = vec![0u64; d];
        let mut s1 = vec![0u64; d];
        let mut s2 = vec![0u64; d];
        
        for i in 0..p_vec.length() {
            let p = p_vec.element(i);

            // do forward NTT
            let (a0, a1, a2) = p_vec_ntt.mut_element(i);
            ring._fwd(p, a0, a1, a2);

            // sum over NTT domains
            for j in 0..d {
                s0[j] = add_64(s0[j], a0[j], P0);
                s1[j] = add_64(s1[j], a1[j], P1);
                s2[j] = add_64(s2[j], a2[j], P2);
            }
        }

        // do inverse NTT on sum
        let mut sum_actual = vec![0u64; d];
        ring._inv(&mut sum_actual, &mut s0, &mut s1, &mut s2);

        // check we get the same thing back
        assert_eq!(sum_actual, sum_actual);
    }

    #[test]
    fn test_int_mul_poly() {
        let d = 1024;
        let a = 42314u64;
        let p_vec = PVec::rand(1, d, Q, [1u8; 32]);
        let p = p_vec.element(0);

        // pad a to be a full polynomial
        let mut a_poly = vec![0u64; d];
        a_poly[0] = a;

        // schoolbook multiplication
        let mut expected = vec![0u64; d * 2];
        poly_mul(&a_poly, &p, &mut expected, Q);

        // int * mul
        let ring = Ring::init(Q, d, false);
        let mut actual = vec![0u64; d * 2];
        ring.int_mul_poly(a, &p, &mut actual);
        assert_eq!(expected, actual);

        // int * mul with cyclotomic recuction - should not change
        let ring = Ring::init(Q, d, true);
        let mut actual = vec![0u64; d];
        ring.int_mul_poly(a, &p, &mut actual);
        assert_eq!(expected[0..d], actual);
    }

    #[test]
    fn test_chal_mul_poly_exact_cancellation() {
        // a -1 challenge coefficient subtracting a value equal to the accumulator must leave 0, not q
        let d = 32;
        let chal = PChal::from_entries(&[(0, false)]);
        let mut poly = vec![0u64; d];
        poly[0] = 5;
        let mut out = vec![0u64; 2 * d];
        out[0] = 5;

        let ring = Ring::init(Q, d, false);
        ring.chal_mul_poly(&chal, &poly, &mut out);

        assert_eq!(0, out[0]);
        assert!(out.iter().all(|&c| c < Q));
    }

    #[test]
    fn test_chal_mul_poly() {
        let d = 32;
        let chal_vec = PChal::rand_vec(1, d, 16, [1u8; 32]);
        let p_vec = PVec::rand(1, d, Q, [1u8; 32]);
        let p2 = p_vec.element(0);

        // convert sparse poly into full poly
        let mut p1 = vec![0u64; d];

        for i in 0..chal_vec[0].k() {
            let (exp, sign) = chal_vec[0].get(i);
            
            if sign {
                p1[exp] = 1;
            }
            else {
                p1[exp] = Q.wrapping_sub(1); // -1 mod q;
            }
        }

        // schoolbook multiplication
        let mut expected = vec![0u64; d * 2];
        poly_mul(&p1, &p2, &mut expected, Q);

        // chal * mul
        let ring = Ring::init(Q, d, false);
        let mut actual = vec![0u64; d * 2];
        ring.chal_mul_poly(&chal_vec[0], &p2, &mut actual);

        assert_eq!(expected, actual);

        // schoolbook multiplication with cyclotomic reduction
        let mut expected = vec![0u64; d];
        poly_mul_cyclotomic(&p1, &p2, &mut expected, Q);

        // chal * mul with cyclotomic recuction
        let ring = Ring::init(Q, d, true);
        let mut actual = vec![0u64; d];
        ring.chal_mul_poly(&chal_vec[0], &p2, &mut actual);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_chal_mul_small_poly() {
        let d = 32;
        let chal_vec = PChal::rand_vec(1, d, 16, [1u8; 32]);

        // sample a polynomial with coefficients at most 15
        let p_vec = PVec::rand(1, d, 16, [1u8; 32]);
        let p2 = p_vec.element(0);

        // convert sparse poly into full poly
        let mut p1 = vec![0u64; d];

        for i in 0..chal_vec[0].k() {
            let (exp, sign) = chal_vec[0].get(i);
            
            if sign {
                p1[exp] = 1;
            }
            else {
                p1[exp] = Q.wrapping_sub(1); // -1 mod q;
            }
        }

        // schoolbook multiplication
        let mut expected = vec![0u64; d * 2];
        poly_mul(&p1, &p2, &mut expected, Q);

        // adjust to balance the coefficients. I.e. coefficients in range q/2...q -> -q/2...0
        for i in 0..expected.len() {
            if expected[i] > (Q >> 1) {
                expected[i] = expected[i].wrapping_sub(Q);
            }
        }

        // chal * mul
        let ring = Ring::init(Q, d, false);
        let mut actual = vec![0u64; d * 2];
        ring.chal_mul_small_poly(&chal_vec[0], &p2, &mut actual);

        assert_eq!(expected, actual);

        // schoolbook multiplication with cyclotomic reduction
        let mut expected = vec![0u64; d];
        poly_mul_cyclotomic(&p1, &p2, &mut expected, Q);

        // adjust to balance the coefficients. I.e. coefficients in range q/2...q -> -q/2...0
        for i in 0..expected.len() {
            if expected[i] > (Q >> 1) {
                expected[i] = expected[i].wrapping_sub(Q);
            }
        }

        // chal * mul with cyclotomic recuction
        let ring = Ring::init(Q, d, true);
        let mut actual = vec![0u64; d];
        ring.chal_mul_small_poly(&chal_vec[0], &p2, &mut actual);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_int_vec_mul_poly_vec() {
        let n = 1024;
        let d = 128;

        // create a vector of random integers
        let mut int_vec = vec![0u64; n];
        let mut rng = rng();

        for i in 0..n {
            int_vec[i] = rand_int(Q, Q.log(), &mut rng);
        }

        // sample a random poly vector
        let p_vec = PVec::rand(n, d, Q, [1; 32]);

        // perform inner product manually
        let mut prod_expected = vec![0u64; 2 * d];

        for i in 0..n {
            // pad the integer to be a full ring element
            let mut p1 = vec![0u64; d];
            p1[0] = int_vec[i];

            // get the next poly
            let p2 = p_vec.element(i);

            // multiply and add to inner product
            let mut prod = vec![0u64; 2 * d];
            poly_mul(&p1, p2, &mut prod, Q);

            for i in 0..d {
                prod_expected[i] = add_64(prod_expected[i], prod[i], Q);
            }
        }

        // perform inner product with non-cyclotomic ring
        let ring = Ring::init(Q, d, false);
        let mut prod_actual = PVec::zero(1, 2 * d);
        ring.int_vec_mul_poly_vec(&int_vec, &p_vec, &mut prod_actual, 0);
        assert_eq!(prod_expected, prod_actual.element(0));

        // perform inner product with cyclotomic ring
        let ring = Ring::init(Q, d, true);
        let mut prod_actual = PVec::zero(1, d);
        ring.int_vec_mul_poly_vec(&int_vec, &p_vec, &mut prod_actual, 0);
        assert_eq!(&prod_expected[0..d], prod_actual.element(0));

    }

    #[test]
    fn test_chal_mul_poly_vec() {
        let n = 128;
        let d = 256;

        // sample a random challenge
        let chal = PChal::rand(d, 16, &mut rng());

        // convert sparse poly into full poly
        let mut p1 = vec![0u64; d];

        for i in 0..chal.k() {
            let (exp, sign) = chal.get(i);
            
            if sign {
                p1[exp] = 1;
            }
            else {
                p1[exp] = Q.wrapping_sub(1); // -1 mod q;
            }
        }
        
        // sample a random poly vec with small coefficients
        let p_vec = PVec::rand(n, d, 16, [1u8; 32]);
        
        // manually compute chal*vec
        let mut prod_expected = PVec::zero(n, 2 * d);

        for i in 0..n {
            let out = prod_expected.mut_element(i);
            poly_mul(&p1, p_vec.element(i), out, Q);

            // adjust to balance coefficients
            for i in 0..2 * d {
                if out[i] > (Q >> 1) {
                    out[i] = out[i].wrapping_sub(Q);
                }
            }
        }

        // compute using ring
        let ring = Ring::init(Q, d, false);
        let mut prod_actual = PVec::zero(n, 2 * d);
        ring.chal_mul_poly_vec(&chal, &p_vec, &mut prod_actual, 0);

        assert_eq!(prod_expected.slice(), prod_actual.slice());
    }

    #[test]
    fn test_chal_mul_poly_vec_cyclotomic() {
        let n = 128;
        let d = 256;

        // sample a random challenge
        let chal = PChal::rand(d, 16, &mut rng());

        // convert sparse poly into full poly
        let mut p1 = vec![0u64; d];

        for i in 0..chal.k() {
            let (exp, sign) = chal.get(i);
            
            if sign {
                p1[exp] = 1;
            }
            else {
                p1[exp] = Q.wrapping_sub(1); // -1 mod q;
            }
        }
        
        // sample a random poly vec with small coefficients
        let p_vec = PVec::rand(n, d, 16, [1u8; 32]);
        
        // manually compute chal*vec
        let mut prod_expected = PVec::zero(n, d);

        for i in 0..n {
            let out = prod_expected.mut_element(i);
            poly_mul_cyclotomic(&p1, p_vec.element(i), out, Q);
        
            // adjust to balance coefficients
            for i in 0..d {
                if out[i] > (Q >> 1) {
                    out[i] = out[i].wrapping_sub(Q);
                }
            }
        }

        // compute using ring
        let ring = Ring::init(Q, d, true);
        let mut prod_actual = PVec::zero(n, d);
        ring.chal_mul_poly_vec(&chal, &p_vec, &mut prod_actual, 0);

        assert_eq!(prod_expected.slice(), prod_actual.slice());
    }

    #[test]
    fn test_chal_vec_mul_poly_vec() {
        let n = 1024;
        let d = 64;

        // sample a random challenge vector
        let chal_vec = PChal::rand_vec(n, d, 16, [1u8; 32]);

        // sample a random poly vec
        let p_vec = PVec::rand(n, d, Q, [1u8; 32]);
        
        // do inner product manually
        let mut prod_expected = vec![0u64; 2 * d];

        for i in 0..n {
            // convert sparse poly into full poly
            let mut p1 = vec![0u64; d];

            for j in 0..chal_vec[i].k() {
                let (exp, sign) = chal_vec[i].get(j);
                
                if sign {
                    p1[exp] = 1;
                }
                else {
                    p1[exp] = Q.wrapping_sub(1); // -1 mod q;
                }
            }

            let mut prod = vec![0u64; 2 * d];
            poly_mul(&p1, p_vec.element(i), &mut prod, Q);

            for j in 0..2 * d {
                prod_expected[j] = add_64(prod_expected[j], prod[j], Q)
            }            
        }

        // do inner product with ring
        let ring = Ring::init(Q, d, false);
        let mut prod_actual = PVec::zero(1, 2 * d);
        ring.chal_vec_mul_poly_vec(&chal_vec, &p_vec, &mut prod_actual, 0);

        assert_eq!(prod_expected, prod_actual.element(0));
    }

    #[test]
    fn test_chal_vec_mul_poly_vec_cyclotomic() {
        let n = 1024;
        let d = 64;

        // sample a random challenge vector
        let chal_vec = PChal::rand_vec(n, d, 16, [1u8; 32]);

        // sample a random poly vec
        let p_vec = PVec::rand(n, d, Q, [1u8; 32]);
        
        // do inner product manually
        let mut prod_expected = vec![0u64; d];

        for i in 0..n {
            // convert sparse poly into full poly
            let mut p1 = vec![0u64; d];

            for j in 0..chal_vec[i].k() {
                let (exp, sign) = chal_vec[i].get(j);
                
                if sign {
                    p1[exp] = 1;
                }
                else {
                    p1[exp] = Q.wrapping_sub(1); // -1 mod q;
                }
            }

            let mut prod = vec![0u64; d];
            poly_mul_cyclotomic(&p1, p_vec.element(i), &mut prod, Q);

            for j in 0..d {
                prod_expected[j] = add_64(prod_expected[j], prod[j], Q)
            }            
        }

        // do inner product with ring
        let ring = Ring::init(Q, d, true);
        let mut prod_actual = PVec::zero(1, d);
        ring.chal_vec_mul_poly_vec(&chal_vec, &p_vec, &mut prod_actual, 0);

        assert_eq!(prod_expected, prod_actual.element(0));
    }

    #[test]
    fn test_mat_fwd_ntt() {
        let d = 1024;
        let ring = Ring::init(Q, d, true);

        // sample a random matrix
        let (h, w) = (4, 32);
        let mat = PMat::rand(h, w, d, Q, [1; 32]);

        // perform forward NTT
        let mut mat_ntt = PMatNtt::zero(h, w, d);
        ring.mat_fwd_ntt(&mat, &mut mat_ntt);

        // check each element was processed correctly
        for i in 0..h {
            for j in 0..w {
                #[cfg(not(feature = "nightly"))]
                {
                let mut p0 = vec![0u32; d];
                let mut p1 = vec![0u32; d];
                let mut p2 = vec![0u32; d];
                let mut p3 = vec![0u32; d];
                let mut p4 = vec![0u32; d];
                ring.native.fwd(mat.element(i, j), &mut p0, &mut p1, &mut p2, &mut p3, &mut p4);
                
                let (s0, s1, s2, s3, s4) = mat_ntt.element(i, j);
                assert_eq!(p0, s0);
                assert_eq!(p1, s1);
                assert_eq!(p2, s2);
                assert_eq!(p3, s3);
                assert_eq!(p4, s4);
                }

                #[cfg(feature = "nightly")]
                {
                let mut p0 = vec![0u64; d];
                let mut p1 = vec![0u64; d];
                let mut p2 = vec![0u64; d];
                ring.native.fwd(mat.element(i, j), &mut p0, &mut p1, &mut p2);
                
                let (s0, s1, s2) = mat_ntt.element(i, j);
                assert_eq!(p0, s0);
                assert_eq!(p1, s1);
                assert_eq!(p2, s2);
                }
            }
        }
    }

    #[test]
    fn test_mat_mul_vec() {
        let d = 64;
        let ring = Ring::init(Q, d, false);

        // sample a random matrix
        let (h, w) = (8, 8);
        let mat = PMat::rand(h, w, d, Q, [1; 32]);
        
        // sample a vector with small coefficients
        let vec = PVec::rand(w, d, 16, [2; 32]);
        
        // manually perform matrix * vector
        let mut prod_expected = PVec::zero(h, 2 * d);

        for row in 0..h {
            let p = prod_expected.mut_element(row);

            for col in 0..w {
                let m = mat.element(row, col);
                let v = vec.element(col);

                let mut prod = vec![0u64; 2 * d];
                poly_mul(m, v, &mut prod, Q);

                for k in 0..2 * d {
                    p[k] = add_64(p[k], prod[k], Q);
                }
            }
        }

        // forward NTT on the matrix
        let mut mat_ntt = PMatNtt::zero(h, w, 2 * d);
        ring.mat_fwd_ntt(&mat, &mut mat_ntt);

        // perform product with the ring
        let mut prod_actual = PVec::zero(h, 2 * d);
        ring.mat_mul_vec(&mat_ntt, &vec, &mut prod_actual, 0);

        // check equal
        assert_eq!(prod_expected.slice(), prod_actual.slice());
    }

    #[test]
    fn test_mat_mul_vec_cyclotomic() {
        let d = 512;
        let ring = Ring::init(Q, d, true);

        // sample a random matrix
        let (h, w) = (4, 48);
        let mat = PMat::rand(h, w, d, Q, [1; 32]);
        
        // sample a vector with small coefficients
        let vec = PVec::rand(w, d, 16, [2; 32]);
        
        // manually perform matrix * vector
        let mut prod_expected = PVec::zero(h, d);

        for row in 0..h {
            let p = prod_expected.mut_element(row);

            for col in 0..w {
                let m = mat.element(row, col);
                let v = vec.element(col);
                
                let mut prod = vec![0u64; d];
                poly_mul_cyclotomic(m, v, &mut prod, Q);

                for k in 0..d {
                    p[k] = add_64(p[k], prod[k], Q);
                }
            }
        }

        // forward NTT on the matrix
        let mut mat_ntt = PMatNtt::zero(h, w, d);
        ring.mat_fwd_ntt(&mat, &mut mat_ntt);

        // perform product with the ring
        let mut prod_actual = PVec::zero(h, d);
        ring.mat_mul_vec(&mat_ntt, &vec, &mut prod_actual, 0);

        // check equal
        assert_eq!(prod_expected.slice(), prod_actual.slice());
    }

    #[test]
    fn test_mat_mul_vec_negative() {
        let d = 1024;
        let ring = Ring::init(Q, d, false);

        // sample a random matrix
        let (h, w) = (4, 32);
        let mat = PMat::rand(h, w, d, Q, [1; 32]);
        
        // sample a vector with small-ish coefficients
        let mut vec = PVec::rand(w, d, 1 << 16, [2; 32]);

        // adjust the coefficients so they're balanced (using wrappping arithmetic)
        let v = vec.mut_slice();

        for i in 0..v.len() {
            v[i] = v[i].wrapping_sub(1 << 15);
        }
        
        // manually perform matrix * vector
        let mut prod_expected = PVec::zero(h, 2 * d);

        for row in 0..h {
            let p = prod_expected.mut_element(row);

            for col in 0..w {
                let m = mat.element(row, col);
                let v = vec.element(col);
                let mut v_pos = vec![0u64; v.len()];

                // adjust the coefficients of v so that they are between 0..7 and q-8..q-1
                for i in 0..v.len() {
                    if (v[i] as i64) < 0 {
                        v_pos[i] = (Q as i64 + v[i] as i64) as u64;
                    }
                    else {
                        v_pos[i] = v[i];
                    }
                }
                
                let mut prod = vec![0u64; 2 * d];
                poly_mul(m, &v_pos, &mut prod, Q);

                for k in 0..2 * d {
                    p[k] = add_64(p[k], prod[k], Q);
                }
            }
        }

        // forward NTT on the matrix
        let mut mat_ntt = PMatNtt::zero(h, w, 2 * d);
        ring.mat_fwd_ntt(&mat, &mut mat_ntt);

        // perform product with the ring
        let mut prod_actual = PVec::zero(h, 2 * d);
        ring.mat_mul_vec(&mat_ntt, &vec, &mut prod_actual, 0);

        // check equal
        assert_eq!(prod_expected.slice(), prod_actual.slice());
    }

    #[test]
    fn test_mat_mul_vec_cyclotomic_negative() {
        let d = 512;
        let ring = Ring::init(Q, d, true);

        // sample a random matrix
        let (h, w) = (4, 48);
        let mat = PMat::rand(h, w, d, Q, [1; 32]);
        
        // sample a vector with small coefficients
        let mut vec = PVec::rand(w, d, 16, [2; 32]);

        // adjust the coefficients so they're between -8..7 instead of 0..15 (using wrappping arithmetic)
        let v = vec.mut_slice();

        for i in 0..v.len() {
            v[i] = v[i].wrapping_sub(8);
        }
        
        // manually perform matrix * vector
        let mut prod_expected = PVec::zero(h, d);

        for row in 0..h {
            let p = prod_expected.mut_element(row);

            for col in 0..w {
                let m = mat.element(row, col);
                let v = vec.element(col);
                let mut v_pos = vec![0u64; v.len()];

                // adjust the coefficients of v so that they are between 0..7 and q-8..q-1
                for i in 0..v.len() {
                    if (v[i] as i64) < 0 {
                        v_pos[i] = (Q as i64 + v[i] as i64) as u64;
                    }
                    else {
                        v_pos[i] = v[i];
                    }
                }
                
                let mut prod = vec![0u64; d];
                poly_mul_cyclotomic(m, &v_pos, &mut prod, Q);

                for k in 0..d {
                    p[k] = add_64(p[k], prod[k], Q);
                }
            }
        }

        // forward NTT on the matrix
        let mut mat_ntt = PMatNtt::zero(h, w, d);
        ring.mat_fwd_ntt(&mat, &mut mat_ntt);

        // perform product with the ring
        let mut prod_actual = PVec::zero(h, d);
        ring.mat_mul_vec(&mat_ntt, &vec, &mut prod_actual, 0);

        // check equal
        assert_eq!(prod_expected.slice(), prod_actual.slice());
    }
}
