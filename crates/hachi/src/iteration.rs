//! One iteration (paper, Fig. 3 and Section 4.3), and the reduction of its output claim to the
//! next iteration's claim (Section 4.5).
//!
//! **Claim.** For a committed witness of $2^r$ chunks $\mathbf f_i$ of $2^m$ ring elements:
//! $\sum_i b_i \cdot \mathbf a^\top \mathbf f_i = T$ ([`Claim`]).
//!
//! **Iteration.**
//! 1. $w_i = \mathbf a^\top \mathbf f_i$, $\hat{\mathbf w} = \mathbf G^{-1}(\mathbf w)$, send
//!    $\mathbf v = \mathbf D \hat{\mathbf w}$.
//! 2. Receive $\mathbf c \in \mathcal C^{2^r}$; $\mathbf z = \sum_i c_i \mathbf s_i$ (norm at most
//!    $\beta$), $\hat{\mathbf z} = \mathbf J^{-1}(\mathbf z)$.
//! 3. Ring switching: the quotients of the relation's rows ([`crate::relation`]) and the next
//!    witness table $\tilde w$ of $2^L$ digits. Commit to it packed (below), and send the
//!    commitment $\mathbf u'$.
//! 4. Receive $\alpha, \tau_0, \tau_1$; the two sumchecks ([`crate::sumcheck`]) reduce everything
//!    to $\tilde w(\mathbf a) = y'$, which the verifier checks against both final claims.
//!
//! **Masked level** ([`Hiding`](crate::params::Hiding), `hachi-pcs.tex`, steps 2(a) to 2(c) of
//! `Eval`). In a masked layout, steps 1 and 2 become attempts. An attempt draws
//! masks $\mathbf s_0, \boldsymbol\varrho_0 \leftarrow D_{\mathfrak s}$, sets $w_0 = \mathbf a^\top
//! \mathbf G \mathbf s_0$ and $\mathbf t_0 = \mathbf A \mathbf s_0 + \mathbf A'
//! \boldsymbol\varrho_0$, computes $\mathbf v = \mathbf D(\hat{\mathbf w}, \hat w_0, \hat{\mathbf
//! t}_0)$ and derives $\mathbf c$ from it. It then computes the folds with the masks and passes
//! them through $\mathsf{Rej}$. A rejected attempt leaves no trace: it runs on a copy of the
//! transcript, and only the accepted attempt's $\mathbf v$ enters the proof.
//!
//! **Reduction to the next claim** (Section 4.5, with $\kappa = \log_2 K$). Write a table index as
//! $J \Vert i$, $i$ the last $\kappa$ bits, and $\mathbf a = (\mathbf a_0, \mathbf a_1)$
//! accordingly. The packed next witness is $\hat w_J = \sum_i \tilde w_{J \Vert i} Z_i \in
//! \mathbb F_{q^K}$, with $Z_i$ the basis of Hachi's Eq. (7) (coordinates of `Ext`); $d/K$
//! consecutive $\hat w_J$ go into one ring element by $\psi$, and these form the next message
//! chunks (`pack`). In hypercube form $\hat w$ is `packed_hypercube`.
//! 1. The prover sends $K - 1$ of the partial evaluations $y_i = \sum_J \tilde w_{J \Vert i}
//!    \widetilde{eq}(J, \mathbf a_0)$, and the verifier derives the other from $y' = \sum_i
//!    \widetilde{eq}(i, \mathbf a_1) y_i$ (Eq. (24) and footnote 10; [`partials::complete`]).
//! 2. A batching sumcheck over the $L - \kappa$ variables of $\hat w$ ([`crate::partials`],
//!    `reduce_prove`) binds the $y_i$ to the committed next witness by ring-switching, and
//!    leaves $\hat w(\mathbf r) =$ `packed_eval` at its point $\mathbf r$.
//!    **This deviates from Section 4.5**, which proves only $(K/d)\,\mathrm{Tr}\_H(p') = \sum_i
//!    y_i Z_i$ (step 4 below at $\mathbf a_0$). That single $\mathbb F_{q^K}$ equation does not
//!    determine the $y_i$: a prover could shift them along its kernel and, with the l-trick on
//!    the last sumcheck round, move $y'$ with them (`crate::partials` and the reduction tests).
//!    The batching binds every $y_i$ instead: it proves a claim on each row of their coordinates.
//! 3. Split $\mathbf r = (\mathbf a_r, \mathbf a_m, \mathbf a_e)$ along (chunk, element,
//!    position in $\psi$). Then $b'_i = \widetilde{eq}(i, \mathbf a_r)$ and $a'_j =
//!    \widetilde{eq}(j, \mathbf a_m) \cdot \sigma_{-1}(\psi(\widetilde{eq}(\cdot, \mathbf a_e)))$
//!    (`next_weights`).
//! 4. The prover sends $p' = \sum_i b'_i \mathbf a'^\top \mathbf f'_i$. By Theorem 2,
//!    $(K/d)\,\mathrm{Tr}\_H(p') = \hat w(\mathbf r) =$ `packed_eval`, which the verifier checks
//!    with `Subfield::e_map`. The next claim is $(\mathbf b', \mathbf a', p')$ at $\mathbf r$.

use crate::commit::{Committed, Ctx, commit};
use crate::error::VerifyError;
use crate::fs::Fs;
use crate::mle::{eq_eval, eq_table, mle_eval, powers};
use crate::params::IterParams;
use crate::partials::{self, RoundValues};
use crate::relation::{Masked, Public, mstar, next_table, quotients};
use crate::sumcheck::{self, RoundMessage, range_poly};
use crate::{Error, Result};
use hachi_co_math::decompose::{digit_bound, digit_range, g_apply, g_inv, j_inv};
use hachi_co_math::{Ext, Poly, Ring, Sparse};
use hachi_co_sampling::ShakePrg;
use hachi_co_sampling::gauss::{WideGaussian, bernoulli_exp};
use rayon::prelude::*;

const LABEL_V: &[u8] = b"hachi/iter/v";
const LABEL_C: &[u8] = b"hachi/iter/c";
const LABEL_U_NEXT: &[u8] = b"hachi/iter/u-next";
const LABEL_COINS: &[u8] = b"hachi/iter/coins";
const LABEL_FINAL: &[u8] = b"hachi/iter/final";
const LABEL_PARTIALS: &[u8] = b"hachi/iter/partials";
const LABEL_TARGET: &[u8] = b"hachi/iter/next-target";

/// The PRG domain of the masks of an attempt: the stream of ring element $e$ of attempt $j$ has
/// index $2^{32} j + e$, so that the elements can be drawn in parallel.
pub const DOMAIN_MASK: &[u8] = b"hachi/prove/mask";
/// The PRG domain of the decision of $\mathsf{Rej}$ in attempt $j$ (index $j$).
pub const DOMAIN_REJ: &[u8] = b"hachi/prove/rej";

/// A matrix of ring elements, row after row.
type Matrix = Vec<Vec<Poly>>;

/// $\mathbf A_1'$ at a masked level, else an empty matrix.
fn rand_matrix<const K: usize>(ctx: &Ctx<K>, params: &IterParams) -> Result<Matrix> {
    if params.layout.masked {
        ctx.rand_matrix()
    } else {
        Ok(Vec::new())
    }
}

/// The randomness of the commitment of a witness in `layout`: per chunk of a masked layout,
/// $\nu + n_A$ ring elements of uniform digits from the stream `(seed, domain, 0)`, chunk after
/// chunk. No chunk otherwise.
///
/// # Errors
/// Sampling errors.
pub fn commitment_randomness<const K: usize>(
    ctx: &Ctx<K>,
    layout: &crate::params::Layout,
    seed: &[u8; 32],
    domain: &[u8],
) -> Result<Vec<Vec<Poly>>> {
    if !layout.masked {
        return Ok(Vec::new());
    }
    let mut prg = ShakePrg::new(seed, domain, 0);
    let len = ctx.rand_len()?;
    (0..layout.chunks())
        .map(|_| ctx.short(&mut prg, len))
        .collect()
}

/// The claim $\sum_i b_i \cdot \mathbf a^\top \mathbf f_i = T$ on a committed witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    /// $\mathbf b \in \mathcal R_q^{2^r}$.
    pub b: Vec<Poly>,
    /// $\mathbf a$, one entry per ring element of a chunk: $\mathbf a \in \mathcal R_q^{2^m}$, or
    /// $(\mathbf a, x^*)$ at a masked level.
    pub a: Vec<Poly>,
    /// $T \in \mathcal R_q$.
    pub target: Poly,
}

/// The messages of one iteration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IterationProof<const K: usize> {
    /// $\mathbf v = \mathbf D \hat{\mathbf w}$, or $\mathbf D (\hat{\mathbf w}, \hat w_0,
    /// \hat{\mathbf t}_0)$ at a masked level ($n_D$ ring elements).
    pub v: Vec<Poly>,
    /// The commitment $\mathbf u'$ to the next witness ($n_B$ ring elements).
    pub next_commitment: Vec<Poly>,
    /// The sumcheck rounds ($L$).
    pub rounds: Vec<RoundMessage<K>>,
    /// $y' = \tilde w(\mathbf a)$.
    pub final_eval: Ext<K>,
    /// The $K - 1$ partial evaluations $y_i$ that the verifier does not derive from $y'$
    /// ([`partials::complete`]).
    pub partials: Vec<Ext<K>>,
    /// The rounds of the batching sumcheck that binds the partial evaluations to the packed
    /// next witness ([`crate::partials`], the reduction's deviation from Section 4.5):
    /// $L - \kappa$ of them.
    pub batch_rounds: Vec<RoundValues<K>>,
    /// $\hat w(\mathbf r)$, the packed next witness at the batching sumcheck's point.
    pub packed_eval: Ext<K>,
    /// $p'$, the next claim's value (at the batching sumcheck's point).
    pub next_target: Poly,
}

/// The digit set $D$ of base $b$ as residues.
pub fn digit_residues<const K: usize>(ctx: &Ctx<K>) -> Vec<u64> {
    let (lo, hi) = digit_range(ctx.choices.b);
    (lo..=hi).map(|c| ctx.ring.zq().from_i128(c)).collect()
}

/// $\lceil \log_2 \rceil$ of the number of rows.
fn log_rows(rows: usize) -> usize {
    rows.next_power_of_two().trailing_zeros() as usize
}

/// The basis element $Z_i$ of $\mathbb F_{q^K}$ (unit coordinate vector).
#[cfg(test)]
pub(crate) fn basis<const K: usize>(i: usize) -> Ext<K> {
    let mut e = Ext([0u64; K]);
    e.0[i] = 1;
    e
}

/// The next claim's $\mathbf b'$ and $\mathbf a'$ from the batching sumcheck's point
/// $\mathbf r$ (module documentation, step 3), for the next layout's $r'$ and $m'$. $\mathbf r$
/// has $L - \kappa$ coordinates, split into $(\mathbf a_r, \mathbf a_m, \mathbf a_e)$ along
/// (chunk, element, position in $\psi$).
fn next_weights<const K: usize>(
    ctx: &Ctx<K>,
    params: &IterParams,
    r_point: &[Ext<K>],
) -> Result<(Vec<Poly>, Vec<Poly>)> {
    let field = ctx.field();
    let (r, m) = (params.next.r, params.next.m);
    let (a_r, rest) = r_point.split_at(r);
    let (a_m, a_e) = rest.split_at(m);
    let b: Vec<Poly> = eq_table(field, a_r)
        .iter()
        .map(|e| ctx.sub.embed(e))
        .collect();
    let fe = eq_table(field, a_e);
    let a_e_ring = ctx.ring.automorphism(&ctx.sub.psi(&fe)?, -1)?;
    let a = eq_table(field, a_m)
        .iter()
        .map(|g| ctx.ring.mul(&ctx.sub.embed(g), &a_e_ring))
        .collect::<hachi_co_math::Result<Vec<_>>>()?;
    Ok((b, a))
}

/// $\sum_i b_i \cdot \mathbf a^\top \mathbf f_i$.
pub fn evaluate_claim<const K: usize>(
    ctx: &Ctx<K>,
    b: &[Poly],
    a: &[Poly],
    message: &[Vec<Poly>],
) -> Result<Poly> {
    let ring = &ctx.ring;
    let mut acc = ring.zero();
    for (bi, fi) in b.iter().zip(message) {
        acc = ring.add(&acc, &ring.mul(bi, &ring.inner(a, fi)?)?)?;
    }
    Ok(acc)
}

/// The next message chunks: the table packed into $\mathbb F_{q^K}$ by $Z_i$, then into ring
/// elements by $\psi$, then cut into $2^{r'}$ chunks of $2^{m'}$ elements.
fn pack<const K: usize>(
    ctx: &Ctx<K>,
    params: &IterParams,
    table: &[u64],
) -> Result<Vec<Vec<Poly>>> {
    let exts: Vec<Ext<K>> = table.as_chunks::<K>().0.iter().map(|c| Ext(*c)).collect();
    let elements = exts
        .chunks_exact(ctx.d_over_k())
        .map(|block| ctx.sub.psi(block))
        .collect::<hachi_co_math::Result<Vec<_>>>()?;
    let len = params.next.chunk_len();
    if elements.len() != params.next.chunks() * len {
        return Err(Error::Shape(
            "the packed witness does not fit the next layout".into(),
        ));
    }
    Ok(elements.chunks_exact(len).map(<[Poly]>::to_vec).collect())
}

/// The partial evaluations $y_i = \sum_J \tilde w_{J \Vert i} \widetilde{eq}(J, \mathbf a_0)$.
fn partials<const K: usize>(ctx: &Ctx<K>, table: &[u64], a0: &[Ext<K>]) -> Vec<Ext<K>> {
    let field = ctx.field();
    let eq = eq_table(field, a0);
    (0..K)
        .map(|i| {
            eq.iter().enumerate().fold(field.zero(), |acc, (j, e)| {
                field.add(&acc, &field.scale(e, table[j * K + i]))
            })
        })
        .collect()
}

/// $\sum_i \widetilde{eq}(i, \mathbf a_1) y_i = \tilde w(\mathbf a_0, \mathbf a_1)$ (paper, Eq.
/// (24)): the value the partial evaluations `ys` must reconstruct.
fn combined_value<const K: usize>(ctx: &Ctx<K>, ys: &[Ext<K>], a1: &[Ext<K>]) -> Ext<K> {
    let field = ctx.field();
    eq_table(field, a1)
        .iter()
        .zip(ys)
        .fold(field.zero(), |acc, (e, y)| {
            field.add(&acc, &field.mul(e, y))
        })
}

/// The packed next witness $\hat w_J = \sum_i \tilde w_{J \Vert i} Z_i$ on the hypercube, from
/// the table of residues: $Z_i$ puts $\tilde w_{J \Vert i}$ in coordinate $i$, so $\hat w_J$ is
/// the $K$-tuple `table[JK .. JK + K]`.
fn packed_hypercube<const K: usize>(table: &[u64]) -> Vec<Ext<K>> {
    table.as_chunks::<K>().0.iter().map(|c| Ext(*c)).collect()
}

/// $(K/d)\,\mathrm{Tr}\_H(p)$ as a field element.
fn scaled_trace<const K: usize>(ctx: &Ctx<K>, p: &Poly) -> Result<Ext<K>> {
    let one = ctx.ring.monomial(0);
    Ok(ctx.sub.extract(&ctx.sub.e_map(p, &one)?)?)
}

/// $\sum_i c_i \mathbf x_i$ over the chunks' vectors `chunk(i)` of `width` ring elements, in
/// $\mathcal R_q$.
fn fold_chunks<'a, const K: usize, F>(
    ctx: &Ctx<K>,
    sparse: &[Sparse],
    width: usize,
    chunk: F,
) -> Result<Vec<Poly>>
where
    F: Fn(usize) -> Result<std::borrow::Cow<'a, [Poly]>> + Sync,
{
    let ring = &ctx.ring;
    let zeros = || vec![ring.zero(); width];
    sparse
        .par_iter()
        .enumerate()
        .try_fold(zeros, |mut acc, (i, ci)| -> Result<Vec<Poly>> {
            let x = chunk(i)?;
            for (zx, sx) in acc.iter_mut().zip(x.iter()) {
                ring.add_sparse_product(zx, ci, sx)?;
            }
            Ok(acc)
        })
        .try_reduce(zeros, |mut acc, other| {
            for (x, y) in acc.iter_mut().zip(&other) {
                *x = ring.add(x, y)?;
            }
            Ok(acc)
        })
}

/// The challenges as ring elements and as sparse vectors.
fn folding_challenges<const K: usize>(
    ctx: &Ctx<K>,
    chunks: usize,
    fs: &mut Fs,
) -> Result<(Vec<Poly>, Vec<Sparse>)> {
    let ring = &ctx.ring;
    let coeffs = fs.challenges(LABEL_C, chunks, ctx.d(), ctx.choices.omega)?;
    let c = coeffs
        .iter()
        .map(|ci| ring.from_i64(ci))
        .collect::<hachi_co_math::Result<_>>()?;
    let sparse = coeffs
        .iter()
        .map(|ci| ring.sparse(ci))
        .collect::<hachi_co_math::Result<_>>()?;
    Ok((c, sparse))
}

/// The folds of an iteration: the first message, the challenges, and the rest of $\mathbf z'$.
struct Fold {
    v: Vec<Poly>,
    c: Vec<Poly>,
    z: Vec<Poly>,
    /// The masked parts, at a masked level.
    masks: Option<Masks>,
}

/// What a masked level adds to the witness ([`Masked`]).
struct Masks {
    w0: Poly,
    w0_hat: Vec<Poly>,
    t0: Vec<Poly>,
    t0_hat: Vec<Poly>,
    z_r: Vec<Poly>,
}

impl Masks {
    fn parts(&self) -> Masked<'_> {
        Masked {
            w0: &self.w0,
            w0_hat: &self.w0_hat,
            t0: &self.t0,
            t0_hat: &self.t0_hat,
            z_r: &self.z_r,
        }
    }
}

/// $\mathsf{Rej}(\mathbf v, D_{\mathfrak s}^N, D_{\mathfrak s}^N, M; \mathbf r)$: whether $\mathbf z
/// = \mathbf v + \mathbf r$ is accepted, with probability $\min\lbrace D(\mathbf z) / (M
/// D(\mathbf r)), 1 \rbrace = \min\lbrace \exp(-\gamma), 1 \rbrace$, where
/// $\gamma = (2 \langle \mathbf v, \mathbf r \rangle + \lVert \mathbf v \rVert^2) / (2 \mathfrak
/// s^2) + \ln M$. Everything is an exact ratio of integers, and the bit comes from
/// [`bernoulli_exp`].
fn rej<const K: usize>(
    ctx: &Ctx<K>,
    hiding: &crate::params::Hiding,
    sigma: u64,
    v: &[Poly],
    r: &[Poly],
    prg: &mut ShakePrg,
) -> Result<bool> {
    let ring: &Ring = &ctx.ring;
    let overflow = || Error::Parameter("the rejection test does not fit 128 bits".into());
    let (inner, norm) = v
        .par_iter()
        .zip(r)
        .map(|(x, y)| -> Result<(i128, i128)> {
            let (x, y) = (ring.to_centered(x)?, ring.to_centered(y)?);
            Ok(x.iter().zip(&y).fold((0, 0), |(i, n), (a, b)| {
                let a = i128::from(*a);
                (i + a * i128::from(*b), n + a * a)
            }))
        })
        .try_reduce(|| (0, 0), |a, b| Ok((a.0 + b.0, a.1 + b.1)))?;
    let (log_num, log_den) = hiding.log_rejection();
    let sigma2 = u128::from(sigma) * u128::from(sigma);
    let den = sigma2.checked_mul(log_den).ok_or_else(overflow)?;
    // gamma = (2 <v, r> + |v|^2) phi^2 / den + log_num sigma^2 / den, as log_den = 2 phi^2.
    let phi2 = i128::try_from(log_den / 2).map_err(|_| overflow())?;
    let top = 2i128
        .checked_mul(inner)
        .and_then(|x| x.checked_add(norm))
        .and_then(|x| x.checked_mul(phi2))
        .and_then(|x| {
            let shift = i128::try_from(sigma2.checked_mul(log_num)?).ok()?;
            x.checked_add(shift)
        })
        .ok_or_else(overflow)?;
    match u128::try_from(top) {
        Ok(gamma) => Ok(bernoulli_exp(prg, gamma, den)?),
        Err(_) => Ok(true),
    }
}

/// `count` ring elements of mask, coefficients from `gauss`, each from its own stream of `seed`.
fn draw_masks<const K: usize>(
    ctx: &Ctx<K>,
    gauss: &WideGaussian,
    seed: &[u8; 32],
    attempt: u32,
    first: usize,
    count: usize,
) -> Result<Vec<Poly>> {
    (first..first + count)
        .into_par_iter()
        .map(|element| {
            let index = (u64::from(attempt) << 32) | element as u64;
            let mut prg = ShakePrg::new(seed, DOMAIN_MASK, index);
            let mut x = vec![0i64; ctx.d()];
            gauss.fill(&mut prg, &mut x)?;
            Ok(ctx.ring.from_i64(&x)?)
        })
        .collect()
}

/// Steps 2(a) to 2(c) of `Eval` at a masked level (module documentation). The transcript `fs` gains
/// the accepted attempt's $\mathbf v$ and $\mathbf c$ only.
///
/// # Errors
/// [`Error::Protocol`] if the fold's bound fails (a witness outside the layout's bound) or if no
/// attempt is accepted within [`Hiding::attempts`](crate::params::Hiding::attempts). Arithmetic
/// and sampling errors.
#[allow(clippy::too_many_arguments)] // the parts of the statement and of the witness
fn masked_fold<const K: usize>(
    ctx: &Ctx<K>,
    index: usize,
    params: &IterParams,
    witness: &Committed,
    claim: &Claim,
    w_hat: &[Poly],
    d_mat: &Matrix,
    a_mat: &Matrix,
    a_rand: &Matrix,
    fs: &mut Fs,
    seed: &[u8; 32],
) -> Result<Fold> {
    let (ring, b, delta) = (&ctx.ring, ctx.choices.b, params.delta);
    let hiding = ctx
        .hiding()
        .ok_or_else(|| Error::Parameter("a masked level without hiding choices".into()))?;
    let layout = &params.layout;
    let width = layout.width(delta);
    let rand_len = ctx.rand_len()?;
    let gauss = WideGaussian::new(params.sigma)?;
    let a_prepared = ring.prepare(a_mat)?;
    let n_d = ctx.choices.n_d;
    for attempt in 0..hiding.attempts() {
        // Step 2(a).
        let s0 = draw_masks(ctx, &gauss, seed, attempt, 0, width)?;
        let r0 = draw_masks(ctx, &gauss, seed, attempt, width, rand_len)?;
        let w0 = ring.inner(&claim.a, &g_apply(ring, b, delta, &s0)?)?;
        let mut t0 = ring.mat_vec_prepared(&a_prepared, &s0)?;
        let term = ctx.rand_term(a_rand, &r0)?;
        for (x, y) in t0.iter_mut().zip(&term) {
            *x = ring.add(x, y)?;
        }
        let w0_hat = g_inv(ring, b, delta, core::slice::from_ref(&w0))?;
        let t0_hat = g_inv(ring, b, delta, &t0)?;
        let mut columns = w_hat.to_vec();
        columns.extend_from_slice(&w0_hat);
        columns.extend_from_slice(&t0_hat);
        let v = ring.mat_vec_short(d_mat, &columns, digit_bound(b))?;
        debug_assert_eq!(v.len(), n_d);
        let mut attempt_fs = fs.clone();
        attempt_fs.polys(LABEL_V, &v);
        // Step 2(b).
        let (c, sparse) = folding_challenges(ctx, layout.chunks(), &mut attempt_fs)?;
        // Step 2(c).
        let mut v_all = fold_chunks(ctx, &sparse, width, |i| {
            Committed::digits(ctx, layout, delta, &witness.message[i])
        })?;
        let v_r = fold_chunks(ctx, &sparse, rand_len, |i| {
            Ok(std::borrow::Cow::Borrowed(&witness.r[i][..]))
        })?;
        v_all.extend(v_r);
        let mut r_all = s0;
        r_all.extend(r0);
        let mut prg = ShakePrg::new(seed, DOMAIN_REJ, u64::from(attempt));
        let accepted = rej(ctx, &hiding, params.sigma, &v_all, &r_all, &mut prg)?;
        let z_all = v_all
            .iter()
            .zip(&r_all)
            .map(|(x, y)| ring.add(x, y))
            .collect::<hachi_co_math::Result<Vec<_>>>()?;
        let mut within = true;
        for x in &z_all {
            within &= ring.inf_norm(x)? <= params.beta;
        }
        if accepted && within {
            *fs = attempt_fs;
            let (z, z_r) = z_all.split_at(width);
            return Ok(Fold {
                v,
                c,
                z: z.to_vec(),
                masks: Some(Masks {
                    w0,
                    w0_hat,
                    t0,
                    t0_hat,
                    z_r: z_r.to_vec(),
                }),
            });
        }
    }
    Err(Error::Protocol(format!(
        "no fold was accepted in {} attempts at level {index}",
        hiding.attempts()
    )))
}

/// Steps 1 and 2 of an iteration without masks (module documentation).
fn plain_fold<const K: usize>(
    ctx: &Ctx<K>,
    params: &IterParams,
    witness: &Committed,
    w_hat: &[Poly],
    d_mat: &Matrix,
    fs: &mut Fs,
) -> Result<Fold> {
    let (ring, delta) = (&ctx.ring, params.delta);
    let layout = &params.layout;
    let v = ring.mat_vec_short(d_mat, w_hat, digit_bound(ctx.choices.b))?;
    fs.polys(LABEL_V, &v);
    let (c, sparse) = folding_challenges(ctx, layout.chunks(), fs)?;
    let z = fold_chunks(ctx, &sparse, layout.width(delta), |i| {
        Committed::digits(ctx, layout, delta, &witness.message[i])
    })?;
    for zx in &z {
        if ring.inf_norm(zx)? > params.beta {
            return Err(Error::Protocol(format!(
                "the fold exceeds its bound beta = {}",
                params.beta
            )));
        }
    }
    Ok(Fold {
        v,
        c,
        z,
        masks: None,
    })
}

/// The prover of one iteration (`index` numbers the iterations from 0 and selects the matrices).
/// `seed` is the opening's secret seed, which only a masked level uses. Returns the messages, the
/// next committed witness and the next claim.
///
/// # Errors
/// [`Error::Protocol`] if an honest-prover invariant fails (a bug or a witness outside the
/// layout's bound); arithmetic and sampling errors.
pub fn prove<const K: usize>(
    ctx: &Ctx<K>,
    index: usize,
    params: &IterParams,
    witness: &Committed,
    claim: &Claim,
    fs: &mut Fs,
    seed: &[u8; 32],
) -> Result<(IterationProof<K>, Committed, Claim)> {
    let (ring, field, d) = (&ctx.ring, ctx.field(), ctx.d());
    let (b, delta, tau, n_a) = (ctx.choices.b, params.delta, params.tau, params.n_a);
    let layout = &params.layout;
    if witness.layout != *layout {
        return Err(Error::Shape(
            "the witness does not have the iteration's layout".into(),
        ));
    }
    // Step 1.
    let a_row = ring.prepare(core::slice::from_ref(&claim.a))?;
    let w: Vec<Poly> = witness
        .message
        .par_iter()
        .map(|fi| {
            ring.mat_vec_prepared(&a_row, fi)?
                .pop()
                .ok_or_else(|| Error::Shape("a^T f_i is one row".into()))
        })
        .collect::<Result<_>>()?;
    let w_hat = g_inv(ring, b, delta, &w)?;
    let d_mat = ctx.matrix(b"D", index, ctx.choices.n_d, params.d_cols())?;
    let width = layout.width(delta);
    let a_mat = ctx.matrix(b"A", index, n_a, width)?;
    let a_rand = rand_matrix(ctx, params)?;
    // Step 2.
    let fold = if layout.masked {
        masked_fold(
            ctx, index, params, witness, claim, &w_hat, &d_mat, &a_mat, &a_rand, fs, seed,
        )?
    } else {
        plain_fold(ctx, params, witness, &w_hat, &d_mat, fs)?
    };
    let Fold { v, c, z, masks } = fold;
    let z_hat = j_inv(ring, b, tau, &z)?;
    // Step 3.
    let b_mat = ctx.matrix(b"B", index, ctx.choices.n_b, layout.chunks() * n_a * delta)?;
    let public = Public {
        params,
        b: &claim.b,
        a: &claim.a,
        target: &claim.target,
        c: &c,
        d_mat: &d_mat,
        b_mat: &b_mat,
        a_mat: &a_mat,
        a_rand: &a_rand,
        v: &v,
        u: &witness.u,
    };
    let parts = masks.as_ref().map(Masks::parts);
    let q = quotients(
        ctx,
        &public,
        &w,
        &w_hat,
        &witness.t,
        &witness.t_hat,
        &z,
        parts.as_ref(),
    )?;
    let mut z_prime = w_hat;
    if let Some(m) = &masks {
        z_prime.extend_from_slice(&m.w0_hat);
        z_prime.extend_from_slice(&m.t0_hat);
    }
    z_prime.extend(witness.t_hat.iter().cloned());
    z_prime.extend(z_hat);
    if let Some(m) = &masks {
        z_prime.extend(j_inv(ring, b, tau, &m.z_r)?);
    }
    let table = next_table(ctx, params, &z_prime, &q)?;
    let next = commit(
        ctx,
        index + 1,
        params.next,
        pack(ctx, params, &table)?,
        Vec::new(),
    )?;
    fs.polys(LABEL_U_NEXT, &next.u);
    // Step 4.
    let lr = log_rows(params.rows);
    let coins = fs.field_coins::<K>(LABEL_COINS, ctx.q, 1 + params.table_vars + lr)?;
    let (alpha, tau0, tau1) = (
        coins[0],
        &coins[1..=params.table_vars],
        &coins[1 + params.table_vars..],
    );
    let alpha_pows = powers(field, &alpha, d + 1);
    let (m, _sum) = mstar(ctx, &public, &eq_table(field, tau1), &alpha_pows)?;
    let g: Vec<Ext<K>> = m
        .iter()
        .flat_map(|mu| alpha_pows[..d].iter().map(move |ap| field.mul(mu, ap)))
        .collect();
    let digits = digit_residues(ctx);
    let (rounds, point, final_eval) =
        sumcheck::prove(field, &digits, &table, eq_table(field, tau0), g, fs)?;
    fs.exts(LABEL_FINAL, &[final_eval]);
    let (ys, batch_rounds, packed_eval, claim) =
        reduce_prove(ctx, params, &table, &point, final_eval, &next, fs)?;
    let proof = IterationProof {
        v,
        next_commitment: next.u.clone(),
        rounds,
        final_eval,
        partials: ys,
        batch_rounds,
        packed_eval,
        next_target: claim.target.clone(),
    };
    Ok((proof, next, claim))
}

/// The transcript labels of the batching sumcheck that binds an iteration's partial evaluations
/// to its committed next witness ([`crate::partials`]).
const BATCH_LABELS: crate::partials::Labels = crate::partials::Labels {
    eta: b"hachi/iter/batch-eta",
    round_poly: b"hachi/iter/batch-h",
    challenge: b"hachi/iter/batch-round",
    eval: b"hachi/iter/batch-eval",
};

/// The reduction of the sumcheck's output to the next claim (paper, Section 4.5, with the
/// deviation of [`crate::partials`]). The sumcheck leaves $\tilde w(\mathbf a) =$ `final_eval` at
/// the point $\mathbf a =$ `point`; write $\mathbf a = (\mathbf a_0, \mathbf a_1)$ with
/// $\mathbf a_1$ the last $\kappa$ coordinates. The prover sends the partial evaluations
/// $y_i = \sum_J \tilde w_{J \Vert i} \widetilde{eq}(J, \mathbf a_0)$, all but the one the
/// verifier derives from $\sum_i \widetilde{eq}(i, \mathbf a_1) y_i =$ `final_eval`; a batching
/// sumcheck over the packed table $\hat w_J = \sum_i \tilde w_{J \Vert i} Z_i$
/// ([`crate::partials`], hypercube form) binds the $y_i$ to it and leaves $\hat w(\mathbf r) =$
/// `packed_eval` at its point $\mathbf r$. The next claim $(\mathbf b', \mathbf a', p')$ is built
/// at $\mathbf r$, with $p' = \sum_i b'_i \mathbf a'^\top \mathbf f'_i$ and the trace check
/// $(K/d)\,\mathrm{Tr}\_H(p') =$ `packed_eval`. Returns the $K - 1$ partial evaluations sent, the
/// batching rounds, `packed_eval` and the next claim.
///
/// # Errors
/// [`Error::Protocol`] if an honest-prover invariant fails (`final_eval`, the batching's first
/// round, or the trace); arithmetic errors.
fn reduce_prove<const K: usize>(
    ctx: &Ctx<K>,
    params: &IterParams,
    table: &[u64],
    point: &[Ext<K>],
    final_eval: Ext<K>,
    next: &Committed,
    fs: &mut Fs,
) -> Result<(Vec<Ext<K>>, Vec<RoundValues<K>>, Ext<K>, Claim)> {
    let field = ctx.field();
    let split = point.len() - ctx.kappa();
    let (a0, a1) = (&point[..split], &point[split..]);
    let ys = partials(ctx, table, a0);
    if combined_value(ctx, &ys, a1) != final_eval {
        return Err(Error::Protocol(
            "the partial evaluations do not give the sumcheck's value".into(),
        ));
    }
    let sent = partials::omit(field, &eq_table(field, a1), &ys)?;
    fs.exts(LABEL_PARTIALS, &sent);
    let packed_table = packed_hypercube(table);
    let (batch_rounds, r, packed_eval) =
        partials::prove_table(field, packed_table, a0, &ys, &BATCH_LABELS, fs)?;
    let (b_next, a_next) = next_weights(ctx, params, &r)?;
    let target = evaluate_claim(ctx, &b_next, &a_next, &next.message)?;
    fs.polys(LABEL_TARGET, core::slice::from_ref(&target));
    if scaled_trace(ctx, &target)? != packed_eval {
        return Err(Error::Protocol(
            "the trace of the next claim's value disagrees with the packed table".into(),
        ));
    }
    Ok((
        sent,
        batch_rounds,
        packed_eval,
        Claim {
            b: b_next,
            a: a_next,
            target,
        },
    ))
}

/// Checks that `polys` are `count` ring elements of degree $d$ with canonical coefficients.
pub fn check_polys<const K: usize>(
    ctx: &Ctx<K>,
    what: &str,
    polys: &[Poly],
    count: usize,
) -> core::result::Result<(), VerifyError> {
    if polys.len() != count || polys.iter().any(|p| p.0.len() != ctx.d()) {
        return Err(VerifyError::Malformed(format!(
            "{what}: {count} ring elements of {} coefficients expected",
            ctx.d()
        )));
    }
    if polys.iter().flat_map(|p| p.0.iter()).any(|x| *x >= ctx.q) {
        return Err(VerifyError::Malformed(format!(
            "{what}: a coefficient is not below q"
        )));
    }
    Ok(())
}

/// Checks that field elements are canonical.
pub fn check_exts<const K: usize>(
    ctx: &Ctx<K>,
    what: &str,
    exts: &[Ext<K>],
    count: usize,
) -> core::result::Result<(), VerifyError> {
    if exts.len() != count || exts.iter().flat_map(|e| e.0).any(|x| x >= ctx.q) {
        return Err(VerifyError::Malformed(format!(
            "{what}: {count} canonical field elements expected"
        )));
    }
    Ok(())
}

/// The verifier of one iteration on the commitment `commitment` and the claim `claim`. Returns
/// the next claim; the next commitment is `proof.next_commitment`.
///
/// # Errors
/// [`VerifyError`] for a malformed or rejected proof.
pub fn verify<const K: usize>(
    ctx: &Ctx<K>,
    index: usize,
    params: &IterParams,
    commitment: &[Poly],
    claim: &Claim,
    proof: &IterationProof<K>,
    fs: &mut Fs,
) -> core::result::Result<Claim, VerifyError> {
    check_polys(ctx, "v", &proof.v, ctx.choices.n_d)?;
    check_polys(
        ctx,
        "next commitment",
        &proof.next_commitment,
        ctx.choices.n_b,
    )?;
    check_polys(
        ctx,
        "next target",
        core::slice::from_ref(&proof.next_target),
        1,
    )?;
    check_exts(
        ctx,
        "final evaluation",
        core::slice::from_ref(&proof.final_eval),
        1,
    )?;
    check_exts(ctx, "partial evaluations", &proof.partials, K - 1)?;
    check_exts(
        ctx,
        "packed evaluation",
        core::slice::from_ref(&proof.packed_eval),
        1,
    )?;
    for (i, round) in proof.rounds.iter().enumerate() {
        let what = format!("round {}", i + 1);
        check_exts(ctx, &what, &round.h0, round.h0.len())?;
        check_exts(ctx, &what, &round.h_alpha, round.h_alpha.len())?;
    }
    for (i, round) in proof.batch_rounds.iter().enumerate() {
        check_exts(ctx, &format!("batching round {}", i + 1), round, 2)?;
    }
    let (point, y) = verify_sumcheck(ctx, index, params, commitment, claim, proof, fs)?;
    reduce_verify(ctx, params, proof, &point, y, fs)
}

/// The range and linear sumchecks of one iteration (paper, Section 4.3): absorbs $\mathbf v$,
/// draws the challenges, absorbs $\mathbf u'$, draws the coins, and runs the two sumchecks with
/// common challenges. Returns their point $\mathbf a$ and $y = \tilde w(\mathbf a) =$
/// `proof.final_eval`, which it checks against both final claims and then absorbs. Shared by
/// [`verify`] and the reduction tests.
///
/// # Errors
/// [`VerifyError::Check`] if a sumcheck's final value is wrong; arithmetic errors.
fn verify_sumcheck<const K: usize>(
    ctx: &Ctx<K>,
    index: usize,
    params: &IterParams,
    commitment: &[Poly],
    claim: &Claim,
    proof: &IterationProof<K>,
    fs: &mut Fs,
) -> core::result::Result<(Vec<Ext<K>>, Ext<K>), VerifyError> {
    let (ring, field, d) = (&ctx.ring, ctx.field(), ctx.d());
    let layout = &params.layout;
    let (delta, n_a) = (params.delta, params.n_a);
    fs.polys(LABEL_V, &proof.v);
    let c: Vec<Poly> = fs
        .challenges(LABEL_C, layout.chunks(), d, ctx.choices.omega)?
        .iter()
        .map(|ci| ring.from_i64(ci))
        .collect::<hachi_co_math::Result<_>>()?;
    fs.polys(LABEL_U_NEXT, &proof.next_commitment);
    let lr = log_rows(params.rows);
    let coins = fs.field_coins::<K>(LABEL_COINS, ctx.q, 1 + params.table_vars + lr)?;
    let (alpha, tau0, tau1) = (
        coins[0],
        &coins[1..=params.table_vars],
        &coins[1 + params.table_vars..],
    );
    let d_mat = ctx.matrix(b"D", index, ctx.choices.n_d, params.d_cols())?;
    let b_mat = ctx.matrix(b"B", index, ctx.choices.n_b, layout.chunks() * n_a * delta)?;
    let a_mat = ctx.matrix(b"A", index, n_a, layout.width(delta))?;
    let a_rand = rand_matrix(ctx, params)?;
    let public = Public {
        params,
        b: &claim.b,
        a: &claim.a,
        target: &claim.target,
        c: &c,
        d_mat: &d_mat,
        b_mat: &b_mat,
        a_mat: &a_mat,
        a_rand: &a_rand,
        v: &proof.v,
        u: commitment,
    };
    let alpha_pows = powers(field, &alpha, d + 1);
    let (m, sum) = mstar(ctx, &public, &eq_table(field, tau1), &alpha_pows)?;
    let digits = digit_residues(ctx);
    let reduced = sumcheck::verify(
        field,
        digits.len(),
        &proof.rounds,
        params.table_vars,
        sum,
        fs,
    )?;
    let point = reduced.point;
    let y = proof.final_eval;
    let expected0 = field.mul(
        &eq_eval(field, tau0, &point),
        &range_poly(field, &digits, &y),
    );
    if reduced.claim0 != expected0 {
        return Err(VerifyError::Check(
            "the range sumcheck's final value".into(),
        ));
    }
    let split_l = params.table_vars - ctx.alpha();
    let g_at = field.mul(
        &mle_eval(field, &m, &point[..split_l]),
        &mle_eval(field, &alpha_pows[..d], &point[split_l..]),
    );
    if reduced.claim_alpha != field.mul(&y, &g_at) {
        return Err(VerifyError::Check(
            "the linear sumcheck's final value".into(),
        ));
    }
    fs.exts(LABEL_FINAL, &[y]);
    Ok((point, y))
}

/// The verifier's reduction to the next claim (the mirror of [`reduce_prove`], paper Section
/// 4.5 with the deviation of [`crate::partials`]): derives the partial evaluation left out from
/// $y =$ `final_eval`, runs the batching sumcheck's verifier to bind them to the packed next
/// witness, and checks the next claim's trace against `proof.packed_eval` at the batching point.
///
/// # Errors
/// [`VerifyError::Check`] if the batching sumcheck or the trace fail.
fn reduce_verify<const K: usize>(
    ctx: &Ctx<K>,
    params: &IterParams,
    proof: &IterationProof<K>,
    point: &[Ext<K>],
    y: Ext<K>,
    fs: &mut Fs,
) -> core::result::Result<Claim, VerifyError> {
    let field = ctx.field();
    let split = point.len() - ctx.kappa();
    let (a0, a1) = (&point[..split], &point[split..]);
    let ys = partials::complete(field, &eq_table(field, a1), &y, &proof.partials)?;
    fs.exts(LABEL_PARTIALS, &proof.partials);
    let r = partials::verify(
        field,
        a0,
        &ys,
        &proof.batch_rounds,
        &proof.packed_eval,
        &BATCH_LABELS,
        fs,
    )?;
    let (b_next, a_next) = next_weights(ctx, params, &r)?;
    fs.polys(LABEL_TARGET, core::slice::from_ref(&proof.next_target));
    if scaled_trace(ctx, &proof.next_target)? != proof.packed_eval {
        return Err(VerifyError::Check(
            "the trace of the next claim's value".into(),
        ));
    }
    Ok(Claim {
        b: b_next,
        a: a_next,
        target: proof.next_target.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::{Ctx, commit};
    use crate::params::{HachiChoices, Layout, Schedule};
    use hachi_co_sampling::ShakePrg;
    use hachi_co_sampling::uniform::uniform_mod_q;

    const K: usize = 4;
    /// The modulus of Hachi-Co's toy-n1 preset (24 bits).
    const Q: u64 = 8_388_637;
    const CONTEXT: [u8; 32] = [3u8; 32];
    const SEED: [u8; 32] = [1u8; 32];
    /// The verifier's message when the batching sumcheck ends on a false claim.
    const BATCH_CHECK: &str = "the batching sumcheck's final value";

    /// `n` pseudorandom ring elements of degree `d`.
    fn polys(tag: &str, n: usize, d: usize) -> Vec<Poly> {
        uniform_mod_q(&mut ShakePrg::new(&[5u8; 32], tag.as_bytes(), 0), Q, n * d)
            .unwrap()
            .chunks_exact(d)
            .map(|c| Poly(c.to_vec()))
            .collect()
    }

    /// An honest one-iteration instance (no hiding): the context, the iteration's parameters, a
    /// committed first witness and a valid claim on it.
    fn honest_instance() -> (Ctx<K>, IterParams, Committed, Claim) {
        let choices = HachiChoices {
            fixed_iterations: Some(1),
            ..HachiChoices::toy(16)
        };
        let ctx = Ctx::<K>::new(Q, choices.clone()).unwrap();
        let d = ctx.d();
        let layout = Layout {
            r: 2,
            m: 2,
            decomposed: true,
            beta_s: 2,
            masked: false,
            extra: 0,
        };
        let params = Schedule::fixed(&choices, Q, layout, 1).unwrap().iterations[0];
        let chunks = layout.chunks();
        let message: Vec<Vec<Poly>> = (0..chunks)
            .map(|i| polys(&format!("f{i}"), layout.chunk_len(), d))
            .collect();
        let witness = commit(&ctx, 0, layout, message.clone(), Vec::new()).unwrap();
        let a = polys("a", layout.chunk_len(), d);
        let b = polys("b", chunks, d);
        let target = evaluate_claim(&ctx, &b, &a, &message).unwrap();
        (ctx, params, witness, Claim { b, a, target })
    }

    /// The acceptance rate of [`rej`] on masks of $D_{\mathfrak s}$, over `trials` attempts, and the
    /// mean of $\mathbf z_0^2 / \mathfrak s^2$ over the accepted ones.
    fn rejection_rate(v: &[Poly], ctx: &Ctx<K>, sigma: u64, trials: u32) -> (f64, f64) {
        let hiding = ctx.hiding().unwrap();
        let gauss = WideGaussian::new(sigma).unwrap();
        let (mut accepted, mut sum) = (0u32, 0f64);
        let mut seeds = ShakePrg::new(&SEED, b"test/rej", 0);
        for _ in 0..trials {
            let r: Vec<Poly> = (0..v.len())
                .map(|_| {
                    let mut x = vec![0i64; ctx.d()];
                    gauss.fill(&mut seeds, &mut x).unwrap();
                    ctx.ring.from_i64(&x).unwrap()
                })
                .collect();
            if rej(ctx, &hiding, sigma, v, &r, &mut seeds).unwrap() {
                accepted += 1;
                let z = ctx.ring.add(&v[0], &r[0]).unwrap();
                let z0 = ctx.ring.to_centered(&z).unwrap()[0] as f64 / sigma as f64;
                sum += z0 * z0;
            }
        }
        (
            f64::from(accepted) / f64::from(trials),
            sum / f64::from(accepted),
        )
    }

    #[test]
    fn rejection_keeps_one_in_m_and_hides_the_fold() {
        // Lemma 21: every attempt is accepted with probability in [1 / M-bar, 1 / M], and an accepted
        // z follows D_s whatever v is. With v = 0 the rate is 1 / M exactly. A v at the lemma's bound
        // has the same rate and the same second moment, within 5 standard errors.
        let choices = HachiChoices::toy_hiding(16);
        let ctx = Ctx::<K>::new(Q, choices).unwrap();
        let hiding = ctx.hiding().unwrap();
        let (num, den) = hiding.log_rejection();
        let m = (num as f64 / den as f64).exp();
        let elements = 20usize;
        let trials = 40_000u32;
        let norm = 3 * (elements * ctx.d()) as u64;
        let sigma = (u64::from(hiding.phi) * norm).div_ceil(16) * 16;
        let zero = vec![ctx.ring.zero(); elements];
        // A v of alternating signs, with ||v||_2 about sigma / (2 phi).
        let large: Vec<Poly> = (0..elements)
            .map(|i| {
                let c: Vec<i64> = (0..ctx.d())
                    .map(|j| if (i + j) % 2 == 0 { 3 } else { -3 } * (elements * ctx.d()) as i64 / 40)
                    .collect();
                ctx.ring.from_i64(&c).unwrap()
            })
            .collect();
        let se_rate = ((1.0 / m) * (1.0 - 1.0 / m) / f64::from(trials)).sqrt();
        for (name, v) in [("zero", &zero), ("large", &large)] {
            let (rate, second) = rejection_rate(v, &ctx, sigma, trials);
            assert!(
                (rate - 1.0 / m).abs() < 5.0 * se_rate,
                "{name}: rate {rate}, 1/M {}",
                1.0 / m
            );
            let se = (2.0 / (f64::from(trials) / m)).sqrt();
            assert!(
                (second - 1.0).abs() < 5.0 * se,
                "{name}: E[z^2] / s^2 = {second}"
            );
        }
    }

    #[test]
    fn honest_iterations_prove_and_verify() {
        let (ctx, params, witness, claim) = honest_instance();
        let mut fs_p = Fs::new(&CONTEXT);
        let (proof, _next, _next_claim) =
            prove(&ctx, 0, &params, &witness, &claim, &mut fs_p, &SEED).unwrap();
        assert_eq!(proof.batch_rounds.len(), params.table_vars - ctx.kappa());
        let mut fs_v = Fs::new(&CONTEXT);
        verify(&ctx, 0, &params, &witness.u, &claim, &proof, &mut fs_v).unwrap();
        assert_eq!(fs_p.digest(), fs_v.digest(), "the transcripts agree");
    }

    #[test]
    fn partials_shifted_along_the_reduction_kernel_are_rejected() {
        // The Section 4.5 forgery that the pre-fix reduction accepted: shift the K partial
        // evaluations by a delta in the joint kernel of the two old checks -- sum_i delta_i Z_i
        // = 0 (the packed value f''(a0) at the sumcheck point) and sum_i eq(i, a1) delta_i = 0
        // (the value, Eq. (24)) -- keeping the honest batching rounds, packed value and next
        // target. The old verifier checked only those two combinations and accepted. The fixed
        // verifier runs the batching sumcheck on the shifted partials, whose rows (hence the
        // running claim) differ, and rejects.
        let (ctx, params, witness, claim) = honest_instance();
        let field = ctx.field();
        let (proof, _next, _next_claim) = prove(
            &ctx,
            0,
            &params,
            &witness,
            &claim,
            &mut Fs::new(&CONTEXT),
            &SEED,
        )
        .unwrap();
        // The sumcheck point, as the verifier derives it; a1 = its last kappa coordinates.
        let (point, _y) = verify_sumcheck(
            &ctx,
            0,
            &params,
            &witness.u,
            &claim,
            &proof,
            &mut Fs::new(&CONTEXT),
        )
        .unwrap();
        let a1 = &point[point.len() - ctx.kappa()..];
        let eq = eq_table(field, a1);
        // delta = (d0, d1, 1, 0) solving sum_i delta_i Z_i = 0 and sum_i eq(i,a1) delta_i = 0.
        let (e1, e2) = (basis::<K>(1), basis::<K>(2));
        let det = field.sub(&eq[1], &field.mul(&eq[0], &e1));
        assert!(
            !field.is_zero(&det),
            "degenerate point; choose other inputs"
        );
        let inv = field.inv(&det).unwrap();
        let d1 = field.mul(&field.sub(&field.mul(&eq[0], &e2), &eq[2]), &inv);
        let d0 = field.neg(&field.add(&field.mul(&d1, &e1), &e2));
        let delta = [d0, d1, field.one(), field.zero()];
        assert!(delta.iter().any(|x| !field.is_zero(x)), "delta is zero");
        // The packed value f''(a0) = sum_i p_i Z_i.
        let packed = |p: &[Ext<K>]| {
            (0..K).fold(field.zero(), |acc, i| {
                field.add(&acc, &field.mul(&p[i], &basis::<K>(i)))
            })
        };
        // The proof sends all but the first (eq[0] != 0 here); the verifier derives that one
        // from the value, which follows the shift because sum_i eq(i, a1) delta_i = 0.
        assert!(!field.is_zero(&eq[0]), "degenerate point");
        let mut forged = proof.clone();
        for (y, d) in forged.partials.iter_mut().zip(&delta[1..]) {
            *y = field.add(y, d);
        }
        assert_ne!(forged.partials, proof.partials, "the shift is nontrivial");
        let all = |p: &IterationProof<K>| {
            partials::complete(field, &eq, &p.final_eval, &p.partials).unwrap()
        };
        let (honest_ys, forged_ys) = (all(&proof), all(&forged));
        for i in 0..K {
            assert_eq!(forged_ys[i], field.add(&honest_ys[i], &delta[i]), "{i}");
        }
        // The old reduction's two checks are blind to the shift: the value Eq. (24) and the
        // packed value f''(a0) are both unchanged.
        assert_eq!(
            combined_value(&ctx, &forged_ys, a1),
            proof.final_eval,
            "the value Eq. (24) is preserved"
        );
        assert_eq!(
            packed(&forged_ys),
            packed(&honest_ys),
            "the packed value is preserved"
        );
        // The fixed verifier rejects: the batching sumcheck binds every partial evaluation.
        let verdict = verify(
            &ctx,
            0,
            &params,
            &witness.u,
            &claim,
            &forged,
            &mut Fs::new(&CONTEXT),
        );
        assert!(
            matches!(&verdict, Err(VerifyError::Check(m)) if m == BATCH_CHECK),
            "the shifted partials were accepted or rejected elsewhere: {verdict:?}"
        );
        // A shift of a single partial evaluation: the derived one moves with it, and the
        // batching sumcheck rejects the rows.
        let mut one = proof.clone();
        one.partials[0] = field.add(&one.partials[0], &field.one());
        let verdict = verify(
            &ctx,
            0,
            &params,
            &witness.u,
            &claim,
            &one,
            &mut Fs::new(&CONTEXT),
        );
        assert!(
            matches!(&verdict, Err(VerifyError::Check(m)) if m == BATCH_CHECK),
            "a single shifted partial evaluation was not rejected: {verdict:?}"
        );
    }
}
