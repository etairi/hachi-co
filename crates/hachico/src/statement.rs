//! The statement of $\mathsf{Eval}$ and its absorption into the Fiat-Shamir transcript
//! (`docs/m1-spec.md`, WP11; plan `docs/hachi-co-plan.md`, finding F5 and the risk "Weak
//! Fiat-Shamir": the transcript absorbs the full statement before any coin).
//!
//! The statement is the public input of $\mathsf{Eval}$ (`6_Collaborative_Hachi.tex`:426): the
//! public parameters $\mathsf{pp}$ (the parameters and $\mathbf A, \mathbf B, \mathbf D$,
//! lines 298-302), the commitment $\mathbf u$ and the point $\mathbf x$. Since the draft's commit
//! `c071784` the value $y$ is not part of it: the verifier reads $y$ from $E(u)$ and outputs it.
//! The base PCS's own parameters (lines 294-297) enter through their digest
//! ([`BasePcs::params_digest`](hachi_co_core::BasePcs::params_digest)). Its commitments are
//! absorbed before the sumcheck's coins, and every opening is made under a context derived from
//! the transcript.

use crate::error::StatementError;
use hachi_co_core::{Challenger, Context, Ext, Poly};
use hachi_co_params::{ChallengeSet, PadRule, Params, TauRule};

/// The domain of the Fiat-Shamir transcript of one $\mathsf{Eval}$
/// ([`hachi_co_core::FsChallenger::new`]): one fresh challenger per proof.
pub const EVAL_DOMAIN: &[u8] = b"hachi-co/eval/v1";

/// The labels of the parameter words of the statement, in the order they are absorbed
/// ([`statement_words`]).
pub const LABELS_PARAMS: [&str; 18] = [
    "hachi-co/eval/params/lambda",
    "hachi-co/eval/params/n",
    "hachi-co/eval/params/t",
    "hachi-co/eval/params/ell",
    "hachi-co/eval/params/alpha",
    "hachi-co/eval/params/kappa",
    "hachi-co/eval/params/b",
    "hachi-co/eval/params/omega",
    "hachi-co/eval/params/challenge_set",
    "hachi-co/eval/params/tau_rule",
    "hachi-co/eval/params/rej_iterations",
    "hachi-co/eval/params/n_A",
    "hachi-co/eval/params/n_B",
    "hachi-co/eval/params/n_D",
    "hachi-co/eval/params/m_u",
    "hachi-co/eval/params/m_v",
    "hachi-co/eval/params/pad",
    "hachi-co/eval/params/q",
];

/// The label of $\mathbf A$ (row-major, $n\_A \cdot \delta 2^m$ ring elements).
pub const LABEL_PP_A: &str = "hachi-co/eval/pp/A";
/// The label of $\mathbf B$ (row-major, $n\_B \cdot (n\_A \delta 2^r + n m\_u)$ ring elements).
pub const LABEL_PP_B: &str = "hachi-co/eval/pp/B";
/// The label of $\mathbf D$ (row-major, $n\_D \cdot w\_D$ ring elements).
pub const LABEL_PP_D: &str = "hachi-co/eval/pp/D";
/// The absorb label of the digest of the base PCS's parameters.
pub const LABEL_PP_BASE: &str = "hachi-co/eval/pp/base-pcs";
/// The label of the evaluation point $\mathbf x \in \mathbb F\_{q^k}^\ell$.
pub const LABEL_X: &str = "hachi-co/eval/x";

/// The parameter words of the statement: the choices of `params`
/// ([`hachi_co_params::Choices`]) with the radius $\omega$ and the modulus $q$ resolved, as
/// 64-bit integers in a fixed order, each with its label from [`LABELS_PARAMS`]:
/// $\lambda$, $n$, $t$, $\ell$, $\alpha$, $\kappa$, $b$, $\omega$ (`params.omega`), the challenge
/// set (0 for the $\ell\_1$ ball, 1 for the sparse set), the $\tau$ rule (0 tight, 1 the earlier
/// draft's, 2 ceil), the number $L$ of iterations of the rejection sampling
/// (`params.rej_iterations`, since the draft's commit `1096ea2`), $n\_A$, $n\_B$, $n\_D$,
/// $m\_u$, $m\_v$, the rule of the pad (0 for
/// [`PadRule::Field`], 1 for [`PadRule::Residues`]), $q$ (`params.q`).
///
/// [`hachi_co_params::derive`] computes every other parameter from these, so they fix the
/// whole parameter set. (A [`Params`] value edited by hand after `derive` is outside this
/// guarantee: build contexts from derived parameters.)
pub fn statement_words(params: &Params) -> [(&'static str, u64); 18] {
    let c = &params.choices;
    let challenge_set = match c.challenge_set {
        ChallengeSet::L1Ball => 0,
        ChallengeSet::Sparse => 1,
    };
    let tau_rule = match c.tau_rule {
        TauRule::Tight => 0,
        TauRule::Draft => 1,
        TauRule::Ceil => 2,
    };
    let pad = match c.pad {
        PadRule::Field => 0,
        PadRule::Residues => 1,
    };
    let values = [
        u64::from(c.lambda),
        c.n,
        c.t,
        u64::from(c.ell),
        u64::from(c.alpha),
        u64::from(c.kappa),
        c.b,
        params.omega,
        challenge_set,
        tau_rule,
        params.rej_iterations,
        c.n_a,
        c.n_b,
        c.n_d,
        c.m_u,
        c.m_v,
        pad,
        params.q,
    ];
    core::array::from_fn(|i| (LABELS_PARAMS[i], values[i]))
}

/// Absorbs the statement of $\mathsf{Eval}$ into `ch`, before any coin is drawn
/// (`docs/m1-spec.md`, WP11), in this order:
///
/// 1. the 18 parameter words of [`statement_words`], each with
///    [`Challenger::absorb_u64`] under its label;
/// 2. $\mathbf A$, $\mathbf B$, $\mathbf D$ (`6_Collaborative_Hachi.tex`:287-292), each as one
///    message of ring elements, row-major ([`Challenger::absorb_polys`] under [`LABEL_PP_A`],
///    [`LABEL_PP_B`], [`LABEL_PP_D`]);
/// 3. the digest of the base PCS's parameters under [`LABEL_PP_BASE`]
///    ([`Challenger::absorb_words`], one vector of four words);
/// 4. the commitment $\mathbf u$ under [`hachi_co_commit::LABEL_U`];
/// 5. $\mathbf x$ under [`LABEL_X`] ([`Challenger::absorb_exts`]).
///
/// The statement is the public input $\mathbbm x\_1 = (\mathbf x, \mathbf u)$ of $\Pi\_A$
/// (`6_Collaborative_Hachi.tex`, $\mathsf{Eval}$ step 1, at the draft's commit `c071784`): the
/// value $y$ is not part of it, since the verifier reads it from $E(u)$ and outputs it.
///
/// The provers and the verifier call it with the same arguments on a fresh challenger; check the
/// statement first ([`check_statement`]), so that only canonical residues are absorbed.
pub fn absorb_statement<const K: usize, H: Challenger<K>>(
    ch: &mut H,
    ctx: &Context<K>,
    base_digest: &[u8; 32],
    u: &[Poly],
    x: &[Ext<K>],
) {
    for (label, value) in statement_words(&ctx.params) {
        ch.absorb_u64(label, value);
    }
    // The rows' concatenation, as the message of `absorb_polys` (`absorb_poly_rows`).
    let pp = ctx.pp();
    ch.absorb_poly_rows(LABEL_PP_A, &pp.a);
    ch.absorb_poly_rows(LABEL_PP_B, &pp.b);
    ch.absorb_poly_rows(LABEL_PP_D, &pp.d);
    let digest: Vec<u64> = base_digest
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| u64::from_le_bytes(*b))
        .collect();
    ch.absorb_words(LABEL_PP_BASE, &[digest]);
    ch.absorb_polys(hachi_co_commit::LABEL_U, u);
    ch.absorb_exts(LABEL_X, x);
}

fn check_residues<'a>(
    what: &'static str,
    residues: impl IntoIterator<Item = &'a u64>,
    q: u64,
) -> Result<(), StatementError> {
    match residues.into_iter().find(|x| **x >= q) {
        Some(value) => Err(StatementError::NonCanonical {
            what,
            value: *value,
            q,
        }),
        None => Ok(()),
    }
}

fn check_len(what: &'static str, expected: usize, actual: usize) -> Result<(), StatementError> {
    if expected != actual {
        return Err(StatementError::Shape {
            what,
            expected,
            actual,
        });
    }
    Ok(())
}

/// Checks that the statement fits the context: $\mathbf u$ has $n\_B$ ring elements of $d$
/// coefficients, $\mathbf x$ has $\ell$ coordinates, and every residue of $\mathbf u$ and
/// $\mathbf x$ is below $q$ (canonical encodings, so that one statement has one transcript).
///
/// # Errors
/// [`StatementError::Shape`] or [`StatementError::NonCanonical`] for the first violation, in the
/// order $\mathbf u$, $\mathbf x$.
pub fn check_statement<const K: usize>(
    ctx: &Context<K>,
    u: &[Poly],
    x: &[Ext<K>],
) -> Result<(), StatementError> {
    let (l, q) = (&ctx.layout, ctx.q());
    check_len("u (n_B ring elements)", l.n_b, u.len())?;
    for p in u {
        check_len("u (d coefficients per entry)", l.d, p.0.len())?;
    }
    check_residues("u", u.iter().flat_map(|p| p.0.iter()), q)?;
    check_len("x (l coordinates)", l.ell, x.len())?;
    check_residues("x", x.iter().flat_map(|e| e.0.iter()), q)
}
