//! The prover's evaluation proof before LaBRADOR (`polcom_eval`).
//!
//! For the point $x \in \mathbb Z_q$, with $\bar x = x^{64}$:
//!
//! - $w_i = \sum_{j < m} \bar x^j s_{im+j} \in \mathcal R_q$ for each column $i$ (the
//!   reference's $w^\top = a^\top [\mathbf s_0 | \cdots | \mathbf s_{n-1}]$), in digits
//!   $\hat w$, entry $i f_u + j$ digit $j$ of $w_i$, and $u_2 = D \hat w$;
//! - $y = f(x) = \sum_i \bar x^{mi} \mathrm{ct}(\sigma_{-1}(a) w_i)$ for $a = \sum_{l < 64} x^l
//!   X^l$, since $\mathrm{ct}(\sigma_{-1}(a) w) = \sum_l x^l w_l$;
//! - challenges $c_0, \dots, c_{n-1}$ from the transcript of the parameters, the seed of the
//!   public matrices, $u_1$, $x$, $y$ and $u_2$ ([`transcript`]);
//! - the amortized opening $z = \sum_i c_i \hat{\mathbf s}_i \in \mathbb Z\[X\]/(X^{64} + 1)^{mf}$,
//!   exactly, in two digits $z = z_0 + 2^{b_u} z_1$;
//! - the witness $(z_0, z_1, \hat t, \hat w)$ and its squared norm $\beta^2$, which the proof
//!   states and the reduced relation takes as its bound (as the reference does).
//!
//! **Deviations.** The reference computes $w_i$ and $z$ in its transform domain from the
//! transforms of $\hat{\mathbf s}_i$ that it keeps from the commitment (1.6 GB for degree
//! $2^{25}$); here $w_i$ is computed from the coefficients and $z$ by sparse products of the
//! challenges, without transforms. The reference decomposes $w_i$ from a representative in
//! $[0, q)$, so its top digit is not centred; here every decomposition starts from the centred
//! representative, as the reference does for $t_i$ and $z$. The reference orders $\hat w$ by
//! digit, then column ($jn + i$); here by column, then digit ($i f_u + j$), the order of $\hat t$
//! within a row, so that the amortization's coefficients $\gamma$ serve both. The reference
//! hashes the low 32 bits of $x$ and $y$ (and, at `LOGQ` other than 32, a buffer whose length
//! differs from what it writes); here $x$ and $y$ are absorbed whole into a framed transcript.
use crate::challenge::{Challenge, challenges};
use crate::commit::{Commitment, Opening, check_poly, digit_bound, to_polys};
use crate::digits::Digits;
use crate::encode::{pack, packed_len, unpack};
use crate::error::{Error, Result};
use crate::params::{N, Params};
use crate::pcs::PublicParams;
use hachi_co_math::Poly;
use hachi_co_transcript::Transcript;
use rayon::prelude::*;

/// The domain of the transcript.
pub const TRANSCRIPT_DOMAIN: &[u8] = b"hachi-co-greyhound/v1";

/// The evaluation proof's own messages: $u_2 \in \mathcal R_q^{\kappa_1}$ and the bound
/// $\beta^2$ of the reduced relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrontProof {
    /// The $\kappa_1$ ring elements of $u_2 = D \hat w$.
    pub u2: Vec<Poly>,
    /// The squared norm of the witness, the bound of the reduced relation.
    pub norm_sq: u128,
}

impl FrontProof {
    /// $u_2$ [`crate::encode::pack`]ed, then $\beta^2$ as `LE128`.
    pub fn to_bytes(&self, params: &Params) -> Vec<u8> {
        let mut out = Vec::new();
        pack(
            self.u2.iter().flat_map(|p| p.0.iter().copied()),
            params.residue_bits(),
            &mut out,
        );
        out.extend_from_slice(&self.norm_sq.to_le_bytes());
        out
    }

    /// The length of [`FrontProof::to_bytes`].
    pub fn encoded_len(params: &Params) -> usize {
        packed_len(params.kappa1 * N, params.residue_bits()) + 16
    }

    /// The proof of [`FrontProof::to_bytes`].
    ///
    /// # Errors
    /// [`Error::Encoding`] for a wrong length or a residue of $q$ or more.
    pub fn from_bytes(params: &Params, bytes: &[u8]) -> Result<Self> {
        if bytes.len() != Self::encoded_len(params) {
            return Err(Error::Encoding);
        }
        let (residues, norm) = bytes.split_at(bytes.len() - 16);
        let values = unpack(residues, params.residue_bits(), params.kappa1 * N, params.q)?;
        Ok(Self {
            u2: values
                .as_chunks::<N>()
                .0
                .iter()
                .map(|c| Poly(c.to_vec()))
                .collect(),
            norm_sq: u128::from_le_bytes(norm.try_into().expect("16 bytes")),
        })
    }
}

/// The witness of the reduced relation: $z_0, z_1$ ($mf$ polynomials each), $\hat t$ ($\kappa
/// f_u n$) and $\hat w$ ($f_u n$), each as its centred coefficients, polynomial by polynomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrontWitness {
    /// $z_0$, $z_1$, $\hat t$ and $\hat w$.
    pub vectors: [Vec<i64>; 4],
}

impl FrontWitness {
    /// $\sum_i \Vert v_i \Vert^2$.
    pub fn norm_sq(&self) -> u128 {
        self.vectors
            .iter()
            .flatten()
            .map(|x| {
                let a = u128::from(x.unsigned_abs());
                a * a
            })
            .sum()
    }
}

/// Everything the prover's front end produces.
#[derive(Clone, Debug)]
pub struct FrontEnd {
    /// $y = f(x)$.
    pub y: u64,
    /// $u_2$ and $\beta^2$.
    pub proof: FrontProof,
    /// The witness of the reduced relation.
    pub witness: FrontWitness,
    /// The challenges $c_0, \dots, c_{n-1}$.
    pub challenges: Vec<Challenge>,
    /// The transcript after $\beta^2$, whose digest binds the LaBRADOR proof to it.
    pub transcript: Transcript,
}

/// The transcript up to the challenges: the parameters, the seed, $u_1$, $x$, $y$ and $u_2$,
/// then the challenge seed and the challenges.
pub fn transcript(
    pp: &PublicParams,
    com: &Commitment,
    x: u64,
    y: u64,
    u2: &[Poly],
) -> (Transcript, Vec<Challenge>) {
    let p = pp.params();
    let mut t = Transcript::new(TRANSCRIPT_DOMAIN);
    t.append_message(b"params", &p.to_bytes());
    t.append_message(b"seed", pp.seed());
    t.append_message(b"u1", &com.to_bytes(p));
    t.append_u64(b"x", x);
    t.append_u64(b"y", y);
    let mut bytes = Vec::new();
    pack(
        u2.iter().flat_map(|p| p.0.iter().copied()),
        p.residue_bits(),
        &mut bytes,
    );
    t.append_message(b"u2", &bytes);
    let seed = t.challenge_seed(b"challenges");
    let cs = challenges(&seed, p.n);
    (t, cs)
}

/// The transcript's last message, $\beta^2$, before the LaBRADOR proof.
pub fn absorb_norm(t: &mut Transcript, norm_sq: u128) {
    t.append_message(b"norm", &norm_sq.to_le_bytes());
}

/// How many products of residues below $q$ a `u128` sum takes before it could overflow.
fn lazy_terms(q: u64) -> usize {
    let square = u128::from(q - 1) * u128::from(q - 1);
    ((u128::MAX / square.max(1)).min(1 << 30) as usize).max(1)
}

/// $w_i$ for each column: coefficient-wise $\sum_j \bar x^j s_{im+j} \bmod q$, summed in `u128`
/// and reduced whenever the sum could overflow.
fn evaluations(pp: &PublicParams, poly: &[u64], xpow: &[u64]) -> Vec<Vec<u64>> {
    let p = pp.params();
    let every = lazy_terms(p.q);
    let zq = pp.ring().zq();
    (0..p.n)
        .into_par_iter()
        .map(|i| {
            let mut acc = [0u128; N];
            let mut out = [0u64; N];
            let mut pending = 0;
            for (j, xj) in xpow.iter().enumerate() {
                let idx = i * p.m + j;
                if idx >= p.len {
                    break;
                }
                let s = &poly[idx * N..(idx + 1) * N];
                for (a, c) in acc.iter_mut().zip(s) {
                    *a += u128::from(*xj) * u128::from(*c);
                }
                pending += 1;
                if pending == every {
                    for (a, o) in acc.iter_mut().zip(out.iter_mut()) {
                        *o = zq.add(*o, zq.reduce_u128(*a));
                        *a = 0;
                    }
                    pending = 0;
                }
            }
            for (a, o) in acc.iter().zip(out.iter_mut()) {
                *o = zq.add(*o, zq.reduce_u128(*a));
            }
            out.to_vec()
        })
        .collect()
}

/// Rows of the polynomial's columns that one task of [`amortize`] takes.
const AMORTIZE_ROWS: usize = 32;

/// `to += w from` for a weight $w \in \{\pm 1, \pm 2\}$, in loops that vectorize.
#[inline(always)]
fn add_scaled(to: &mut [i32], from: &[i32], w: i32) {
    debug_assert_eq!(to.len(), from.len());
    match w {
        1 => to.iter_mut().zip(from).for_each(|(a, s)| *a += *s),
        -1 => to.iter_mut().zip(from).for_each(|(a, s)| *a -= *s),
        _ => to.iter_mut().zip(from).for_each(|(a, s)| *a += w * *s),
    }
}

/// The amortized opening $z = \sum_i c_i \hat{\mathbf s}_i$ over the integers, entry $km + j$ at
/// `z[(k m + j) * 64..]`, in `i32` when $n \Vert c \Vert_1 \max |\hat s|$ fits it (else
/// [`amortize_reference`]). A task takes [`AMORTIZE_ROWS`] rows $j$ of every digit and holds them
/// transposed, coefficient by coefficient, so that a term $v X^p$ of a challenge adds the block
/// of coefficient rows $0, \dots, 63 - p$ of the digits to rows $p, \dots, 63$ of the sums and
/// subtracts the other block from rows $0, \dots, p - 1$: two long contiguous additions per term
/// for all the task's rows at once.
fn amortize(pp: &PublicParams, poly: &[u64], cs: &[Challenge]) -> Result<Vec<i64>> {
    let p = pp.params();
    let l1 = cs.iter().map(Challenge::l1_norm).max().unwrap_or(0);
    let bound = u128::from(l1) * p.n as u128 * u128::from(digit_bound(p.q, p.b, p.f));
    if bound >= i32::MAX as u128 {
        return amortize_reference(pp, p, poly, cs);
    }
    let zq = pp.ring().zq();
    let (m, f) = (p.m, p.f);
    let digits = Digits::new(p.b, f);
    const R: usize = AMORTIZE_ROWS;
    let terms: Vec<Vec<(usize, i32)>> = cs
        .iter()
        .map(|c| {
            c.coeffs
                .iter()
                .enumerate()
                .filter(|(_, v)| **v != 0)
                .map(|(pos, v)| (pos, i32::from(*v)))
                .collect()
        })
        .collect();
    let chunks: Vec<Vec<i32>> = (0..m.div_ceil(R))
        .into_par_iter()
        .map(|chunk| {
            let j0 = chunk * R;
            let rows = R.min(m - j0);
            // Digit k, coefficient c, row jj at (k 64 + c) R + jj.
            let mut acc = vec![0i32; f * N * R];
            let mut digit_rows = vec![0i32; f * N * R];
            for (i, column_terms) in terms.iter().enumerate() {
                for jj in 0..R {
                    let idx = i * m + j0 + jj;
                    if jj < rows && idx < p.len {
                        for (c, x) in poly[idx * N..(idx + 1) * N].iter().enumerate() {
                            let centred = zq.to_centered(*x);
                            for k in 0..f {
                                digit_rows[(k * N + c) * R + jj] = digits.digit(centred, k) as i32;
                            }
                        }
                    } else {
                        for kc in 0..f * N {
                            digit_rows[kc * R + jj] = 0;
                        }
                    }
                }
                for &(pos, v) in column_terms {
                    for k in 0..f {
                        // X^pos moves coefficient rows 0..64-pos up to pos..64 and the others,
                        // negated, down to 0..pos: two contiguous blocks.
                        let acc_k = &mut acc[k * N * R..(k + 1) * N * R];
                        let src_k = &digit_rows[k * N * R..(k + 1) * N * R];
                        let (low, high) = acc_k.split_at_mut(pos * R);
                        let (stays, wraps) = src_k.split_at((N - pos) * R);
                        add_scaled(high, stays, v);
                        add_scaled(low, wraps, -v);
                    }
                }
            }
            acc
        })
        .collect();
    let mut z = vec![0i64; p.z_len() * N];
    for (chunk, acc) in chunks.iter().enumerate() {
        let j0 = chunk * R;
        for k in 0..f {
            for jj in 0..R.min(m - j0) {
                let to = (k * m + j0 + jj) * N;
                for c in 0..N {
                    z[to + c] = i64::from(acc[(k * N + c) * R + jj]);
                }
            }
        }
    }
    Ok(z)
}

/// [`amortize`] in `i64`, one sparse product per polynomial (the test oracle, and the path for
/// sums beyond `i32`).
fn amortize_reference(
    pp: &PublicParams,
    p: &Params,
    poly: &[u64],
    cs: &[Challenge],
) -> Result<Vec<i64>> {
    let ring = pp.ring();
    let zq = ring.zq();
    let (m, f) = (p.m, p.f);
    let digits = Digits::new(p.b, f);
    let sparse: Vec<_> = cs
        .iter()
        .map(|c| ring.sparse(&c.to_i64()))
        .collect::<core::result::Result<_, _>>()?;
    const ROWS: usize = 8;
    let chunks: Vec<Vec<i64>> = (0..m.div_ceil(ROWS))
        .into_par_iter()
        .map(|chunk| {
            let rows = ROWS.min(m - chunk * ROWS);
            // Digit k of row j of the chunk at (k rows + j) 64.
            let mut acc = vec![0i64; f * rows * N];
            let (mut centred, mut digit) = ([0i64; N], [0i64; N]);
            for (i, c) in sparse.iter().enumerate() {
                for j in 0..rows {
                    let idx = i * m + chunk * ROWS + j;
                    if idx >= p.len {
                        break;
                    }
                    for (o, x) in centred.iter_mut().zip(&poly[idx * N..(idx + 1) * N]) {
                        *o = zq.to_centered(*x);
                    }
                    for k in 0..f {
                        for (o, x) in digit.iter_mut().zip(&centred) {
                            *o = digits.digit(*x, k);
                        }
                        let at = (k * rows + j) * N;
                        ring.add_sparse_product_signed(&mut acc[at..at + N], c, &digit)?;
                    }
                }
            }
            Ok(acc)
        })
        .collect::<Result<_>>()?;
    let mut z = vec![0i64; p.z_len() * N];
    for (chunk, acc) in chunks.iter().enumerate() {
        let rows = ROWS.min(m - chunk * ROWS);
        for k in 0..f {
            for j in 0..rows {
                let from = (k * rows + j) * N;
                let to = (k * m + chunk * ROWS + j) * N;
                z[to..to + N].copy_from_slice(&acc[from..from + N]);
            }
        }
    }
    Ok(z)
}

/// The prover's front end for the point `x` (a residue below $q$): $y$, $u_2$, the challenges,
/// the witness and its norm (module documentation).
///
/// # Errors
/// [`Error::Shape`] for a polynomial or point out of range; arithmetic errors.
pub fn prove_front(
    pp: &PublicParams,
    com: &Commitment,
    opening: &Opening,
    poly: &[u64],
    x: u64,
) -> Result<FrontEnd> {
    let p = pp.params();
    check_poly(p, poly)?;
    if x >= p.q || opening.t_hat.len() != p.t_len() * N || com.u1.len() != p.kappa1 {
        return Err(Error::Shape(
            "point, opening or commitment of the wrong size",
        ));
    }
    let ring = pp.ring();
    let zq = ring.zq();
    let xbar = zq.pow(x, N as u64);
    let mut xpow = Vec::with_capacity(p.m);
    let mut acc = 1u64;
    for _ in 0..p.m {
        xpow.push(acc);
        acc = zq.mul(acc, xbar);
    }
    let w = evaluations(pp, poly, &xpow);
    // y = sum_i xbar^(m i) sum_l x^l w_i[l].
    let mut xl = [0u64; N];
    let mut acc = 1u64;
    for v in xl.iter_mut() {
        *v = acc;
        acc = zq.mul(acc, x);
    }
    let xbar_m = zq.pow(xbar, p.m as u64);
    let mut y = 0u64;
    let mut scale = 1u64;
    for wi in &w {
        let ct = wi
            .iter()
            .zip(&xl)
            .fold(0u64, |s, (a, b)| zq.add(s, zq.mul(*a, *b)));
        y = zq.add(y, zq.mul(scale, ct));
        scale = zq.mul(scale, xbar_m);
    }
    // w-hat: digit j of w_i at i f_u + j.
    let (fu, bu) = (p.fu, p.bu);
    let wdigits = Digits::new(bu, fu);
    let mut w_hat = vec![0i64; p.w_len() * N];
    for (i, wi) in w.iter().enumerate() {
        for (c, v) in wi.iter().enumerate() {
            let centred = zq.to_centered(*v);
            for j in 0..fu {
                w_hat[(i * fu + j) * N + c] = wdigits.digit(centred, j);
            }
        }
    }
    let u2 = ring.mat_vec_short(
        &pp.key().d,
        &to_polys(ring, &w_hat),
        digit_bound(p.q, bu, fu),
    )?;
    let (mut transcript, cs) = transcript(pp, com, x, y, &u2);
    // z = z0 + 2^bu z1.
    let z = amortize(pp, poly, &cs)?;
    let zdigits = Digits::new(bu, 2);
    let z0: Vec<i64> = z.par_iter().map(|v| zdigits.digit(*v, 0)).collect();
    let z1: Vec<i64> = z.par_iter().map(|v| zdigits.digit(*v, 1)).collect();
    let witness = FrontWitness {
        vectors: [z0, z1, opening.t_hat.clone(), w_hat],
    };
    let norm_sq = witness.norm_sq();
    absorb_norm(&mut transcript, norm_sq);
    Ok(FrontEnd {
        y,
        proof: FrontProof { u2, norm_sq },
        witness,
        challenges: cs,
        transcript,
    })
}

/// $f(x)$ by Horner's rule, for tests and callers: the polynomial of [`crate::commit::commit`].
///
/// # Errors
/// [`Error::Shape`] for a coefficient or point of $q$ or more.
pub fn evaluate(q: u64, poly: &[u64], x: u64) -> Result<u64> {
    let zq = hachi_co_math::Zq::new(q)?;
    if x >= q || poly.iter().any(|c| *c >= q) {
        return Err(Error::Shape(
            "coefficients and point must be residues below q",
        ));
    }
    Ok(poly
        .iter()
        .rev()
        .fold(0u64, |acc, c| zq.add(zq.mul(acc, x), *c)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Security;
    use hachi_co_sampling::ShakePrg;
    use rand::Rng;

    /// The transposed `i32` sums are the sparse products, for columns of any length (a partial
    /// last column, $m$ not a multiple of the task's rows) and both moduli.
    #[test]
    fn transposed_amortization_is_the_sparse_products() {
        for (q, len) in [
            (4294967197u64, 300usize),
            (4294967197, 1),
            (4294967197, 4096 + 17),
            (93266870910952069, 700),
        ] {
            let pp = PublicParams::new(len, q, Security::labrador(), [5u8; 32]).unwrap();
            let p = *pp.params();
            let mut prg = ShakePrg::new(&[6u8; 32], b"poly", len as u64);
            let poly: Vec<u64> = (0..len * N).map(|_| prg.next_u64() % q).collect();
            let cs: Vec<Challenge> = (0..p.n as u64)
                .map(|i| Challenge::sample(&[7; 32], i))
                .collect();
            let fast = amortize(&pp, &poly, &cs).unwrap();
            let reference = amortize_reference(&pp, &p, &poly, &cs).unwrap();
            assert_eq!(fast, reference, "q {q} len {len} m {}", p.m);
        }
    }
}
