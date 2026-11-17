//! Commitments (`polcom_commit`).
//!
//! The polynomial's $\ell$ ring elements $s_0, \dots, s_{\ell-1}$ fill $n$ columns of $m$ (the
//! last zero-padded): column $i$ is $\mathbf s_i = (s_{im}, \dots, s_{im+m-1})$. Each column is
//! decomposed into $f$ digits of base $2^b$, $\hat{\mathbf s}_i = G^{-1}(\mathbf s_i) \in
//! \mathcal R_q^{mf}$ with entry $km + j$ digit $k$ of $s_{im+j}$ (the reference's layout), its
//! inner commitment is $t_i = A \hat{\mathbf s}_i \in \mathcal R_q^\kappa$, decomposed into $f_u$
//! digits of base $2^{b_u}$, and the commitment is $u_1 = B \hat t$ for $\hat t$ the digits of all
//! inner commitments, entry $(kn + i) f_u + j$ digit $j$ of row $k$ of $t_i$. **Deviation:** the
//! reference orders $\hat t$ by column, then digit, then row ($(i f_u + j) \kappa + k$); the
//! order here keeps the entries of each row $k$ together, so that its inner-commitment equation
//! reads one contiguous range of the witness. The two differ by a permutation of the columns of
//! $B$, which is uniform either way.
use crate::digits::Digits;
use crate::encode::{pack, packed_len, unpack};
use crate::error::{Error, Result};
use crate::params::{N, Params};
use crate::pcs::{ProverKey, PublicParams};
use hachi_co_math::{Poly, Ring, rns::NADDS};
use rayon::prelude::*;

/// Columns committed together by one sequential product (they share the reads of $A$).
const GROUP: usize = 8;

/// A commitment: $u_1 \in \mathcal R_q^{\kappa_1}$.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commitment {
    /// The $\kappa_1$ ring elements of $u_1$.
    pub u1: Vec<Poly>,
}

impl Commitment {
    /// The residues of $u_1$, [`crate::encode::pack`]ed.
    pub fn to_bytes(&self, params: &Params) -> Vec<u8> {
        let mut out = Vec::new();
        pack(
            self.u1.iter().flat_map(|p| p.0.iter().copied()),
            params.residue_bits(),
            &mut out,
        );
        out
    }

    /// The commitment of [`Commitment::to_bytes`].
    ///
    /// # Errors
    /// [`Error::Encoding`] for a wrong length or a residue of $q$ or more.
    pub fn from_bytes(params: &Params, bytes: &[u8]) -> Result<Self> {
        let values = unpack(bytes, params.residue_bits(), params.kappa1 * N, params.q)?;
        Ok(Self {
            u1: values
                .as_chunks::<N>()
                .0
                .iter()
                .map(|c| Poly(c.to_vec()))
                .collect(),
        })
    }

    /// The length of [`Commitment::to_bytes`].
    pub fn encoded_len(params: &Params) -> usize {
        packed_len(params.kappa1 * N, params.residue_bits())
    }
}

/// What the prover keeps of a commitment: the digits $\hat t$ of the inner commitments.
#[derive(Clone, Debug)]
pub struct Opening {
    /// $\hat t$, $\kappa f_u n$ polynomials of centred digits, coefficient by coefficient.
    pub(crate) t_hat: Vec<i64>,
}

/// The centred coefficients of a polynomial of $\ell$ ring elements, checked.
pub(crate) fn check_poly(params: &Params, poly: &[u64]) -> Result<()> {
    if poly.len() != params.len * N {
        return Err(Error::Shape("the polynomial must have 64 len coefficients"));
    }
    if poly.iter().any(|x| *x >= params.q) {
        return Err(Error::Shape("coefficients must be residues below q"));
    }
    Ok(())
}

/// The largest digit of the decomposition of a centred residue.
pub(crate) fn digit_bound(q: u64, d: u32, parts: usize) -> u64 {
    Digits::new(d, parts).bound((q - 1) / 2)
}

/// `out` = digits `digit` of the centred coefficients of ring element `idx` of `poly`, or zero
/// past its end.
#[inline]
pub(crate) fn fill_digit(
    ring: &Ring,
    poly: &[u64],
    idx: usize,
    digits: &Digits,
    digit: usize,
    out: &mut [i64],
) {
    if (idx + 1) * N <= poly.len() {
        let zq = ring.zq();
        for (o, x) in out.iter_mut().zip(&poly[idx * N..(idx + 1) * N]) {
            *o = digits.digit(zq.to_centered(*x), digit);
        }
    } else {
        out.fill(0);
    }
}

/// Commit to `poly`, the $64\ell$ coefficients of a polynomial over $\mathbb Z_q$ (coefficient
/// $64i + j$ is coefficient $j$ of ring element $s_i$), residues in $[0, q)$.
///
/// # Errors
/// [`Error::Shape`] for a wrong length or a coefficient of $q$ or more; arithmetic errors.
pub fn commit(pp: &PublicParams, pk: &ProverKey, poly: &[u64]) -> Result<(Commitment, Opening)> {
    let p = pp.params();
    check_poly(p, poly)?;
    let ring = pp.ring();
    let zq = ring.zq();
    let (m, n, mf) = (p.m, p.n, p.z_len());
    let digits = Digits::new(p.b, p.f);
    let pieces: Vec<core::ops::Range<usize>> = (0..mf)
        .step_by(NADDS)
        .map(|s| s..(s + NADDS).min(mf))
        .collect();
    let groups: Vec<core::ops::Range<usize>> = (0..n)
        .step_by(GROUP)
        .map(|s| s..(s + GROUP).min(n))
        .collect();
    // Inner commitments t_i = A s_i, GROUP columns per sequential product.
    let t: Vec<Vec<Poly>> = groups
        .par_iter()
        .map(|cols| {
            ring.mat_vec_prepared_pieces_many(
                &pk.a,
                cols.len(),
                pieces.iter().cloned(),
                |v, piece, out| {
                    let col = cols.start + v;
                    for (slot, e) in out.as_chunks_mut::<N>().0.iter_mut().zip(piece) {
                        fill_digit(ring, poly, col * m + e % m, &digits, e / m, slot);
                    }
                    Ok(())
                },
            )
        })
        .collect::<core::result::Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();
    // Their digits, row k of column i at (k n + i) f_u.
    let (fu, kappa) = (p.fu, p.kappa);
    let tdigits = Digits::new(p.bu, fu);
    let mut t_hat = vec![0i64; p.t_len() * N];
    t_hat
        .par_chunks_mut(n * fu * N)
        .enumerate()
        .for_each(|(k, row)| {
            for (i, ti) in t.iter().enumerate() {
                for (c, x) in ti[k].0.iter().enumerate() {
                    let centred = zq.to_centered(*x);
                    for j in 0..fu {
                        row[(i * fu + j) * N + c] = tdigits.digit(centred, j);
                    }
                }
            }
        });
    debug_assert_eq!(t.len(), n);
    debug_assert!(t.iter().all(|ti| ti.len() == kappa));
    let u1 = ring.mat_vec_short(
        &pp.key().b,
        &to_polys(ring, &t_hat),
        digit_bound(p.q, p.bu, fu),
    )?;
    Ok((Commitment { u1 }, Opening { t_hat }))
}

/// Centred integer coefficients as ring elements.
pub(crate) fn to_polys(ring: &Ring, coeffs: &[i64]) -> Vec<Poly> {
    let zq = ring.zq();
    coeffs
        .par_chunks(N)
        .map(|c| Poly(c.iter().map(|x| zq.from_i64(*x)).collect()))
        .collect()
}
