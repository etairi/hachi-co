//! The statement of $\mathsf{Eval}$ and its absorption into the Fiat-Shamir transcript
//! (`docs/m1-spec.md`, WP11; plan `docs/hachi-co-plan.md`, finding F5 and the risk "Weak
//! Fiat-Shamir": the transcript absorbs the full statement before any coin).
//!
//! The statement is the public input of $\mathsf{Eval}$ (`6_Collaborative_Hachi.tex`:408): the
//! public parameters $\mathsf{pp}$ (the parameters and $\mathbf A, \mathbf B, \mathbf D$,
//! lines 293-297), the commitment $\mathbf u$, the point $\mathbf x$ and the claimed evaluation
//! $y$. The base PCS's own parameters (lines 289-292) enter through their digest
//! ([`BasePcs::params_digest`](hachi_co_core::BasePcs::params_digest)). Its commitments are
//! absorbed before the sumcheck's coins, and every opening is made under a context derived from
//! the transcript.

use crate::error::StatementError;
use hachi_co_core::{Challenger, Context, Ext, Matrix, Poly};
use hachi_co_params::{ChallengeSet, Params, TauRule};

/// The domain of the Fiat-Shamir transcript of one $\mathsf{Eval}$
/// ([`hachi_co_core::FsChallenger::new`]): one fresh challenger per proof.
pub const EVAL_DOMAIN: &[u8] = b"hachi-co/eval/v1";

/// The labels of the parameter words of the statement, in the order they are absorbed
/// ([`statement_words`]).
pub const LABELS_PARAMS: [&str; 15] = [
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
    "hachi-co/eval/params/n_A",
    "hachi-co/eval/params/n_B",
    "hachi-co/eval/params/n_D",
    "hachi-co/eval/params/m_rho",
    "hachi-co/eval/params/q",
];

/// The label of $\mathbf A$ (row-major, $n\_A \cdot \delta 2^m$ ring elements).
pub const LABEL_PP_A: &str = "hachi-co/eval/pp/A";
/// The label of $\mathbf B$ (row-major, $n\_B \cdot n\_A \delta 2^r$ ring elements).
pub const LABEL_PP_B: &str = "hachi-co/eval/pp/B";
/// The label of $\mathbf D$ (row-major, $n\_D \cdot w\_D$ ring elements).
pub const LABEL_PP_D: &str = "hachi-co/eval/pp/D";
/// The absorb label of the digest of the base PCS's parameters.
pub const LABEL_PP_BASE: &str = "hachi-co/eval/pp/base-pcs";
/// The label of the evaluation point $\mathbf x \in \mathbb F\_{q^k}^\ell$.
pub const LABEL_X: &str = "hachi-co/eval/x";
/// The label of the claimed evaluation $y \in \mathbb F\_{q^k}$.
pub const LABEL_Y: &str = "hachi-co/eval/y";

/// The parameter words of the statement: the choices of `params`
/// ([`hachi_co_params::Choices`]) with the radius $\omega$ and the modulus $q$ resolved, as
/// 64-bit integers in a fixed order, each with its label from [`LABELS_PARAMS`]:
/// $\lambda$, $n$, $t$, $\ell$, $\alpha$, $\kappa$, $b$, $\omega$ (`params.omega`), the challenge
/// set (0 for the $\ell\_1$ ball, 1 for the sparse set), the $\tau$ rule (0 tight, 1 the draft's,
/// 2 ceil), $n\_A$, $n\_B$, $n\_D$, $m\_\rho$, $q$ (`params.q`).
///
/// [`hachi_co_params::derive`] computes every other parameter from these, so they fix the
/// whole parameter set. (A [`Params`] value edited by hand after `derive` is outside this
/// guarantee: build contexts from derived parameters.)
pub fn statement_words(params: &Params) -> [(&'static str, u64); 15] {
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
        c.n_a,
        c.n_b,
        c.n_d,
        c.m_rho,
        params.q,
    ];
    core::array::from_fn(|i| (LABELS_PARAMS[i], values[i]))
}

/// A matrix as one list of ring elements, row after row (its shape is fixed by the parameter
/// words absorbed before it).
fn rows(m: &Matrix) -> Vec<Poly> {
    m.iter().flatten().cloned().collect()
}

/// Absorbs the statement of $\mathsf{Eval}$ into `ch`, before any coin is drawn
/// (`docs/m1-spec.md`, WP11), in this order:
///
/// 1. the 15 parameter words of [`statement_words`], each with
///    [`Challenger::absorb_u64`] under its label;
/// 2. $\mathbf A$, $\mathbf B$, $\mathbf D$ (`6_Collaborative_Hachi.tex`:283-288), each as one
///    message of ring elements, row-major ([`Challenger::absorb_polys`] under [`LABEL_PP_A`],
///    [`LABEL_PP_B`], [`LABEL_PP_D`]);
/// 3. the digest of the base PCS's parameters under [`LABEL_PP_BASE`]
///    ([`Challenger::absorb_words`], one vector of four words);
/// 4. the commitment $\mathbf u$ under [`hachi_co_commit::LABEL_U`];
/// 5. $\mathbf x$ under [`LABEL_X`] and $y$ under [`LABEL_Y`]
///    ([`Challenger::absorb_exts`]).
///
/// The provers and the verifier call it with the same arguments on a fresh challenger; check the
/// statement first ([`check_statement`]), so that only canonical residues are absorbed.
pub fn absorb_statement<const K: usize, H: Challenger<K>>(
    ch: &mut H,
    ctx: &Context<K>,
    base_digest: &[u8; 32],
    u: &[Poly],
    x: &[Ext<K>],
    y: &Ext<K>,
) {
    for (label, value) in statement_words(&ctx.params) {
        ch.absorb_u64(label, value);
    }
    ch.absorb_polys(LABEL_PP_A, &rows(&ctx.pp.a));
    ch.absorb_polys(LABEL_PP_B, &rows(&ctx.pp.b));
    ch.absorb_polys(LABEL_PP_D, &rows(&ctx.pp.d));
    let digest: Vec<u64> = base_digest
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| u64::from_le_bytes(*b))
        .collect();
    ch.absorb_words(LABEL_PP_BASE, &[digest]);
    ch.absorb_polys(hachi_co_commit::LABEL_U, u);
    ch.absorb_exts(LABEL_X, x);
    ch.absorb_exts(LABEL_Y, core::slice::from_ref(y));
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
/// coefficients, $\mathbf x$ has $\ell$ coordinates, and every residue of $\mathbf u$,
/// $\mathbf x$ and $y$ is below $q$ (canonical encodings, so that one statement has one
/// transcript).
///
/// # Errors
/// [`StatementError::Shape`] or [`StatementError::NonCanonical`] for the first violation, in the
/// order $\mathbf u$, $\mathbf x$, $y$.
pub fn check_statement<const K: usize>(
    ctx: &Context<K>,
    u: &[Poly],
    x: &[Ext<K>],
    y: &Ext<K>,
) -> Result<(), StatementError> {
    let (l, q) = (&ctx.layout, ctx.q());
    check_len("u (n_B ring elements)", l.n_b, u.len())?;
    for p in u {
        check_len("u (d coefficients per entry)", l.d, p.0.len())?;
    }
    check_residues("u", u.iter().flat_map(|p| p.0.iter()), q)?;
    check_len("x (l coordinates)", l.ell, x.len())?;
    check_residues("x", x.iter().flat_map(|e| e.0.iter()), q)?;
    check_residues("y", y.0.iter(), q)
}
