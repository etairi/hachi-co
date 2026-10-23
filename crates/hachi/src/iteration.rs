//! One iteration (paper, Fig. 3 and Section 4.3), and the reduction of its output claim to the
//! next iteration's claim (Section 4.5).
//!
//! **Claim.** For a committed witness of $2^r$ chunks $\mathbf f_i$ of $2^m$ ring elements:
//! $\sum_i b_i \cdot \mathbf a^\top \mathbf f_i = T$ ([`Claim`]).
//!
//! **Iteration.**
//! 1. $w_i = \mathbf a^\top \mathbf f_i$, $\hat{\mathbf w} = \mathbf G^{-1}(\mathbf w)$, send
//!    $\mathbf v = \mathbf D \hat{\mathbf w}$; with [`Hiding`](crate::params::Hiding),
//!    $\mathbf v = \mathbf D \hat{\mathbf w} + \mathbf D' \boldsymbol\rho$ with
//!    $\boldsymbol\rho \leftarrow \chi$ from the stream `(seed, "hachi/prove/rho", index)`.
//! 2. Receive $\mathbf c \in \mathcal C^{2^r}$; $\mathbf z = \sum_i c_i \mathbf s_i$ (norm at most
//!    $\beta$), $\hat{\mathbf z} = \mathbf J^{-1}(\mathbf z)$; with hiding also
//!    $\mathbf r_z = \sum_i c_i \mathbf r_i$ (norm at most $\beta_r$) and the digits of
//!    $\boldsymbol\rho$ and $\mathbf r_z$.
//! 3. Ring switching: the quotients of the relation's rows ([`crate::relation`]) and the next
//!    witness table $\tilde w$ of $2^L$ digits. Commit to it packed (below), with hiding under
//!    randomness from the stream `(seed, "hachi/prove/r", index + 1)`, chunk after chunk, and
//!    send the commitment $\mathbf u'$.
//! 4. Receive $\alpha, \tau_0, \tau_1$; the two sumchecks ([`crate::sumcheck`]) reduce everything
//!    to $\tilde w(\mathbf a) = y'$, which the verifier checks against both final claims.
//!
//! **Reduction to the next claim** (Section 4.5, with $\kappa = \log_2 K$). Write a table index as
//! $J \Vert i$, $i$ the last $\kappa$ bits, and $\mathbf a = (\mathbf a_0, \mathbf a_1)$
//! accordingly. The packed next witness is $\hat w_J = \sum_i \tilde w_{J \Vert i} Z_i \in
//! \mathbb F_{q^K}$, with $Z_i$ the basis of Hachi's Eq. (7) (coordinates of `Ext`); $d/K$
//! consecutive $\hat w_J$ go into one ring element by $\psi$, and these form the next message
//! chunks (`pack`). In hypercube form $\hat w$ is `packed_hypercube`.
//! 1. The prover sends the $K$ partial evaluations $y_i = \sum_J \tilde w_{J \Vert i}
//!    \widetilde{eq}(J, \mathbf a_0)$; the verifier checks $y' = \sum_i \widetilde{eq}(i, \mathbf
//!    a_1) y_i$ (`combined_value`, Eq. (24)).
//! 2. A batching sumcheck over the $L - \kappa$ variables of $\hat w$ ([`crate::partials`],
//!    `reduce_prove`) binds the $y_i$ to the committed next witness: it proves
//!    $\hat w(\varphi^{-t}(\mathbf a_0)) = \sum_i Z_i \varphi^{-t}(y_i)$ for $t = 0, \dots, K-1$,
//!    and leaves $\hat w(\mathbf r) =$ `packed_eval` at its point $\mathbf r$.
//!    **This deviates from Section 4.5**, which proves only $(K/d)\,\mathrm{Tr}\_H(p') = \sum_i
//!    y_i Z_i$ (step 4 below at $\mathbf a_0$). That single $\mathbb F_{q^K}$ equation does not
//!    determine the $y_i$: a prover could shift them along its kernel and, with the l-trick on
//!    the last sumcheck round, move $y'$ with them (`crate::partials` and the reduction tests).
//!    The batching binds every $y_i$ instead (the Moore matrix $[\varphi^t(Z_i)]$ is invertible).
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
use crate::relation::{Public, mstar, next_table, quotients};
use crate::sumcheck::{self, RoundMessage, range_poly};
use crate::{Error, Result};
use hachi_co_math::decompose::{digit_range, g_inv, j_inv};
use hachi_co_math::{Ext, Poly};
use hachi_co_sampling::ShakePrg;

const LABEL_V: &[u8] = b"hachi/iter/v";
const LABEL_C: &[u8] = b"hachi/iter/c";
const LABEL_U_NEXT: &[u8] = b"hachi/iter/u-next";
const LABEL_COINS: &[u8] = b"hachi/iter/coins";
const LABEL_FINAL: &[u8] = b"hachi/iter/final";
const LABEL_PARTIALS: &[u8] = b"hachi/iter/partials";
const LABEL_TARGET: &[u8] = b"hachi/iter/next-target";

/// The PRG domain of $\boldsymbol\rho$ in iteration `index` (index `index`).
pub const DOMAIN_RHO: &[u8] = b"hachi/prove/rho";
/// The PRG domain of the randomness of the commitment at level `index + 1` (index `index + 1`).
pub const DOMAIN_NEXT_R: &[u8] = b"hachi/prove/r";

/// A matrix of ring elements, row after row.
type Matrix = Vec<Vec<Poly>>;

/// $\mathbf A_1'$ ($n_A \times \nu$) and $\mathbf D_1'$ ($n_D \times \nu$) of iteration `index`, or
/// two empty matrices without hiding.
fn rand_matrices<const K: usize>(ctx: &Ctx<K>, index: usize) -> Result<(Matrix, Matrix)> {
    match ctx.hiding() {
        None => Ok((Vec::new(), Vec::new())),
        Some(_) => Ok((
            ctx.rand_matrix(b"A'", index, ctx.choices.n_a)?,
            ctx.rand_matrix(b"D'", index, ctx.choices.n_d)?,
        )),
    }
}

/// The randomness of a commitment at level `level` in `layout`: per chunk, $\nu + n_A$ ring
/// elements from the stream `(seed, domain, level)`, chunk after chunk; no chunk without hiding.
///
/// # Errors
/// Sampling errors.
pub fn commitment_randomness<const K: usize>(
    ctx: &Ctx<K>,
    layout: &crate::params::Layout,
    seed: &[u8; 32],
    domain: &[u8],
    level: usize,
) -> Result<Vec<Vec<Poly>>> {
    if ctx.hiding().is_none() {
        return Ok(Vec::new());
    }
    let mut prg = ShakePrg::new(seed, domain, level as u64);
    (0..layout.chunks())
        .map(|_| ctx.noise(&mut prg, ctx.rand_len(ctx.choices.n_a)))
        .collect()
}

/// The claim $\sum_i b_i \cdot \mathbf a^\top \mathbf f_i = T$ on a committed witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    /// $\mathbf b \in \mathcal R_q^{2^r}$.
    pub b: Vec<Poly>,
    /// $\mathbf a \in \mathcal R_q^{2^m}$.
    pub a: Vec<Poly>,
    /// $T \in \mathcal R_q$.
    pub target: Poly,
}

/// The messages of one iteration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IterationProof<const K: usize> {
    /// $\mathbf v = \mathbf D \hat{\mathbf w}$, plus $\mathbf D' \boldsymbol\rho$ with
    /// [`Hiding`](crate::params::Hiding) ($n_D$ ring elements).
    pub v: Vec<Poly>,
    /// The commitment $\mathbf u'$ to the next witness ($n_B$ ring elements).
    pub next_commitment: Vec<Poly>,
    /// The sumcheck rounds ($L$).
    pub rounds: Vec<RoundMessage<K>>,
    /// $y' = \tilde w(\mathbf a)$.
    pub final_eval: Ext<K>,
    /// The $K$ partial evaluations $y_i$.
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

/// The prover of one iteration (`index` numbers the iterations from 0; it selects the
/// matrices and the PRG streams of `seed`, the opening's secret seed, which only hiding uses).
/// Returns the messages, the next committed witness and the next claim.
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
    let (b, delta, tau) = (ctx.choices.b, params.delta, params.tau);
    let layout = &params.layout;
    if witness.layout != *layout {
        return Err(Error::Shape(
            "the witness does not have the iteration's layout".into(),
        ));
    }
    // Step 1.
    let w: Vec<Poly> = witness
        .message
        .iter()
        .map(|fi| ring.inner(&claim.a, fi))
        .collect::<hachi_co_math::Result<_>>()?;
    let w_hat = g_inv(ring, b, delta, &w)?;
    let d_mat = ctx.matrix(b"D", index, ctx.choices.n_d, layout.chunks() * delta)?;
    let mut v = ring.mat_vec(&d_mat, &w_hat)?;
    let (a_rand, d_rand) = rand_matrices(ctx, index)?;
    let rho = if ctx.hiding().is_some() {
        let mut prg = ShakePrg::new(seed, DOMAIN_RHO, index as u64);
        let rho = ctx.noise(&mut prg, ctx.rand_len(ctx.choices.n_d))?;
        let term = ctx.rand_term(&d_rand, &rho)?;
        for (x, y) in v.iter_mut().zip(&term) {
            *x = ring.add(x, y)?;
        }
        rho
    } else {
        Vec::new()
    };
    fs.polys(LABEL_V, &v);
    // Step 2.
    let c: Vec<Poly> = fs
        .challenges(LABEL_C, layout.chunks(), d, ctx.choices.omega)?
        .iter()
        .map(|ci| ring.from_i64(ci))
        .collect::<hachi_co_math::Result<_>>()?;
    let width = layout.width(delta);
    let mut z = vec![ring.zero(); width];
    for (ci, si) in c.iter().zip(&witness.s) {
        for (zx, sx) in z.iter_mut().zip(si) {
            *zx = ring.add(zx, &ring.mul(ci, sx)?)?;
        }
    }
    for zx in &z {
        if ring.inf_norm(zx)? > params.beta {
            return Err(Error::Protocol(format!(
                "the fold exceeds its bound beta = {}",
                params.beta
            )));
        }
    }
    let z_hat = j_inv(ring, b, tau, &z)?;
    // The folded randomness r_z = sum_i c_i r_i, within beta_r.
    let mut r_z = vec![ring.zero(); witness.r.first().map_or(0, Vec::len)];
    for (ci, ri) in c.iter().zip(&witness.r) {
        for (x, y) in r_z.iter_mut().zip(ri) {
            *x = ring.add(x, &ring.mul(ci, y)?)?;
        }
    }
    for x in &r_z {
        if ring.inf_norm(x)? > params.beta_r {
            return Err(Error::Protocol(format!(
                "the folded randomness exceeds its bound beta_r = {}",
                params.beta_r
            )));
        }
    }
    // Step 3.
    let b_mat = ctx.matrix(
        b"B",
        index,
        ctx.choices.n_b,
        layout.chunks() * ctx.choices.n_a * delta,
    )?;
    let a_mat = ctx.matrix(b"A", index, ctx.choices.n_a, width)?;
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
        d_rand: &d_rand,
        v: &v,
        u: &witness.u,
    };
    let q = quotients(
        ctx,
        &public,
        &w,
        &w_hat,
        &witness.t,
        &witness.t_hat,
        &z,
        &rho,
        &r_z,
    )?;
    let mut z_prime = w_hat;
    z_prime.extend(witness.t_hat.iter().cloned());
    z_prime.extend(z_hat);
    if ctx.hiding().is_some() {
        z_prime.extend(j_inv(ring, b, params.tau_rho, &rho)?);
        z_prime.extend(j_inv(ring, b, params.tau_r, &r_z)?);
    }
    let table = next_table(ctx, params, &z_prime, &q)?;
    let r_next = commitment_randomness(ctx, &params.next, seed, DOMAIN_NEXT_R, index + 1)?;
    let next = commit(
        ctx,
        index + 1,
        params.next,
        pack(ctx, params, &table)?,
        r_next,
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
    let w_tab: Vec<Ext<K>> = table.iter().map(|x| field.from_base(*x)).collect();
    let digits = digit_residues(ctx);
    let (rounds, point, final_eval) =
        sumcheck::prove(field, &digits, w_tab, eq_table(field, tau0), g, fs)?;
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
    gamma: b"hachi/iter/batch-gamma",
    round_poly: b"hachi/iter/batch-h",
    challenge: b"hachi/iter/batch-round",
    eval: b"hachi/iter/batch-eval",
};

/// The reduction of the sumcheck's output to the next claim (paper, Section 4.5, with the
/// deviation of [`crate::partials`]). The sumcheck leaves $\tilde w(\mathbf a) =$ `final_eval` at
/// the point $\mathbf a =$ `point`; write $\mathbf a = (\mathbf a_0, \mathbf a_1)$ with
/// $\mathbf a_1$ the last $\kappa$ coordinates. The prover sends the $K$ partial evaluations
/// $y_i = \sum_J \tilde w_{J \Vert i} \widetilde{eq}(J, \mathbf a_0)$ (the verifier checks
/// $\sum_i \widetilde{eq}(i, \mathbf a_1) y_i =$ `final_eval`); a batching sumcheck over the
/// packed table $\hat w_J = \sum_i \tilde w_{J \Vert i} Z_i$ ([`crate::partials`], hypercube
/// form) binds the $y_i$ to it and leaves $\hat w(\mathbf r) =$ `packed_eval` at its point
/// $\mathbf r$. The next claim $(\mathbf b', \mathbf a', p')$ is built at $\mathbf r$, with
/// $p' = \sum_i b'_i \mathbf a'^\top \mathbf f'_i$ and the trace check
/// $(K/d)\,\mathrm{Tr}\_H(p') =$ `packed_eval`. Returns $(y_i)$, the batching rounds,
/// `packed_eval` and the next claim.
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
    fs.exts(LABEL_PARTIALS, &ys);
    if combined_value(ctx, &ys, a1) != final_eval {
        return Err(Error::Protocol(
            "the partial evaluations do not give the sumcheck's value".into(),
        ));
    }
    let packed_table = packed_hypercube(table);
    let (batch_rounds, r, packed_eval) =
        partials::prove_table(field, &packed_table, a0, &ys, &BATCH_LABELS, fs)?;
    let (b_next, a_next) = next_weights(ctx, params, &r)?;
    let target = evaluate_claim(ctx, &b_next, &a_next, &next.message)?;
    fs.polys(LABEL_TARGET, core::slice::from_ref(&target));
    if scaled_trace(ctx, &target)? != packed_eval {
        return Err(Error::Protocol(
            "the trace of the next claim's value disagrees with the packed table".into(),
        ));
    }
    Ok((
        ys,
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
    check_exts(ctx, "partial evaluations", &proof.partials, K)?;
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
        check_exts(ctx, &format!("batching round {}", i + 1), round, 3)?;
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
    let delta = params.delta;
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
    let d_mat = ctx.matrix(b"D", index, ctx.choices.n_d, layout.chunks() * delta)?;
    let b_mat = ctx.matrix(
        b"B",
        index,
        ctx.choices.n_b,
        layout.chunks() * ctx.choices.n_a * delta,
    )?;
    let a_mat = ctx.matrix(b"A", index, ctx.choices.n_a, layout.width(delta))?;
    let (a_rand, d_rand) = rand_matrices(ctx, index)?;
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
        d_rand: &d_rand,
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
/// 4.5 with the deviation of [`crate::partials`]): checks that the partial evaluations
/// reconstruct $y =$ `final_eval`, runs the batching sumcheck's verifier to bind them to the
/// packed next witness, and checks the next claim's trace against `proof.packed_eval` at the
/// batching point.
///
/// # Errors
/// [`VerifyError::Check`] if the partial evaluations, the batching sumcheck or the trace fail.
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
    fs.exts(LABEL_PARTIALS, &proof.partials);
    if combined_value(ctx, &proof.partials, a1) != y {
        return Err(VerifyError::Check("the partial evaluations".into()));
    }
    let r = partials::verify(
        field,
        a0,
        &proof.partials,
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
        };
        let params = Schedule::fixed(&choices, Q, layout, 1).unwrap().iterations[0];
        let chunks = layout.chunks();
        let message: Vec<Vec<Poly>> = (0..chunks)
            .map(|i| polys(&format!("f{i}"), layout.chunk_len(), d))
            .collect();
        let r = commitment_randomness(&ctx, &layout, &SEED, b"test/r", 0).unwrap();
        let witness = commit(&ctx, 0, layout, message.clone(), r).unwrap();
        let a = polys("a", layout.chunk_len(), d);
        let b = polys("b", chunks, d);
        let target = evaluate_claim(&ctx, &b, &a, &message).unwrap();
        (ctx, params, witness, Claim { b, a, target })
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
        // verifier runs the batching sumcheck on the shifted partials, whose conjugate values
        // (hence the challenge gamma and the round-1 claim) differ, and rejects.
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
        let mut forged = proof.clone();
        for (y, d) in forged.partials.iter_mut().zip(delta) {
            *y = field.add(y, &d);
        }
        assert_ne!(forged.partials, proof.partials, "the shift is nontrivial");
        // The old reduction's two checks are blind to the shift: the value Eq. (24) and the
        // packed value f''(a0) are both unchanged.
        assert_eq!(
            combined_value(&ctx, &forged.partials, a1),
            proof.final_eval,
            "the value Eq. (24) is preserved"
        );
        assert_eq!(
            packed(&forged.partials),
            packed(&proof.partials),
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
            matches!(&verdict, Err(VerifyError::Check(m)) if m == "batching sumcheck, round 1"),
            "the shifted partials were accepted or rejected elsewhere: {verdict:?}"
        );
        // A shift of a single partial evaluation alone breaks the value check first.
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
            matches!(&verdict, Err(VerifyError::Check(m)) if m == "the partial evaluations"),
            "a single shifted partial evaluation was not rejected: {verdict:?}"
        );
    }
}
