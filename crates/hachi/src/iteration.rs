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
//! accordingly.
//! 1. The prover sends the $K$ partial evaluations $y_i = \sum_J \tilde w_{J \Vert i}
//!    \widetilde{eq}(J, \mathbf a_0)$; the verifier checks $y' = \sum_i \widetilde{eq}(i, \mathbf
//!    a_1) y_i$.
//! 2. Packing: $\hat w_J = \sum_i \tilde w_{J \Vert i} Z_i \in \mathbb F_{q^K}$, with $Z_i$ the
//!    basis of Hachi's Eq. (7) (coordinates of `Ext`). Then $d/K$ consecutive $\hat w_J$ go into
//!    one ring element by $\psi$, and these form the next message chunks.
//! 3. Split $\mathbf a_0 = (\mathbf a_r, \mathbf a_m, \mathbf a_e)$ along (chunk, element,
//!    position in $\psi$). Then $b'_i = \widetilde{eq}(i, \mathbf a_r)$ and $a'_j =
//!    \widetilde{eq}(j, \mathbf a_m) \cdot \sigma_{-1}(\psi(\widetilde{eq}(\cdot, \mathbf a_e)))$.
//! 4. The prover sends $p' = \sum_i b'_i \mathbf a'^\top \mathbf f'_i$. By Theorem 2,
//!    $(K/d)\,\mathrm{Tr}\_H(p') = \sum_J \hat w_J \widetilde{eq}(J, \mathbf a_0) = \sum_i y_i Z_i$,
//!    which the verifier checks with `Subfield::e_map`. The next claim is
//!    $(\mathbf b', \mathbf a', p')$.

use crate::commit::{Committed, Ctx, commit};
use crate::error::VerifyError;
use crate::fs::Fs;
use crate::mle::{eq_eval, eq_table, mle_eval, powers};
use crate::params::IterParams;
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
    /// $p'$, the next claim's value.
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
fn basis<const K: usize>(i: usize) -> Ext<K> {
    let mut e = Ext([0u64; K]);
    e.0[i] = 1;
    e
}

/// The next claim's $\mathbf b'$ and $\mathbf a'$ from the point $\mathbf a$ (module
/// documentation, step 3), for the next layout's $r'$ and $m'$.
fn next_weights<const K: usize>(
    ctx: &Ctx<K>,
    params: &IterParams,
    point: &[Ext<K>],
) -> Result<(Vec<Poly>, Vec<Poly>)> {
    let field = ctx.field();
    let kappa = ctx.kappa();
    let (r, m) = (params.next.r, params.next.m);
    let a0 = &point[..point.len() - kappa];
    let (a_r, rest) = a0.split_at(r);
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

/// $\sum_i \widetilde{eq}(i, \mathbf a_1) y_i$ and $\sum_i y_i Z_i$.
fn combine<const K: usize>(ctx: &Ctx<K>, ys: &[Ext<K>], a1: &[Ext<K>]) -> (Ext<K>, Ext<K>) {
    let field = ctx.field();
    let eq = eq_table(field, a1);
    let mut value = field.zero();
    let mut packed = field.zero();
    for (i, y) in ys.iter().enumerate() {
        value = field.add(&value, &field.mul(&eq[i], y));
        packed = field.add(&packed, &field.mul(y, &basis::<K>(i)));
    }
    (value, packed)
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
    // The reduction to the next claim.
    let split = point.len() - ctx.kappa();
    let ys = partials(ctx, &table, &point[..split]);
    fs.exts(LABEL_PARTIALS, &ys);
    let (b_next, a_next) = next_weights(ctx, params, &point)?;
    let target = evaluate_claim(ctx, &b_next, &a_next, &next.message)?;
    fs.polys(LABEL_TARGET, core::slice::from_ref(&target));
    let (value, packed) = combine(ctx, &ys, &point[split..]);
    if value != final_eval || scaled_trace(ctx, &target)? != packed {
        return Err(Error::Protocol(
            "the reduction to the next claim is inconsistent".into(),
        ));
    }
    let proof = IterationProof {
        v,
        next_commitment: next.u.clone(),
        rounds,
        final_eval,
        partials: ys,
        next_target: target.clone(),
    };
    let claim = Claim {
        b: b_next,
        a: a_next,
        target,
    };
    Ok((proof, next, claim))
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
    let (ring, field, d) = (&ctx.ring, ctx.field(), ctx.d());
    let layout = &params.layout;
    let delta = params.delta;
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
    for (i, round) in proof.rounds.iter().enumerate() {
        let what = format!("round {}", i + 1);
        check_exts(ctx, &what, &round.h0, round.h0.len())?;
        check_exts(ctx, &what, &round.h_alpha, round.h_alpha.len())?;
    }
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
    let point = &reduced.point;
    let y = proof.final_eval;
    let expected0 = field.mul(
        &eq_eval(field, tau0, point),
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
    fs.exts(LABEL_PARTIALS, &proof.partials);
    let split = point.len() - ctx.kappa();
    let (value, packed) = combine(ctx, &proof.partials, &point[split..]);
    if value != y {
        return Err(VerifyError::Check("the partial evaluations".into()));
    }
    let (b_next, a_next) = next_weights(ctx, params, point)?;
    fs.polys(LABEL_TARGET, core::slice::from_ref(&proof.next_target));
    if scaled_trace(ctx, &proof.next_target)? != packed {
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
