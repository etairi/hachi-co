//! The two sumchecks of one iteration, run with common round challenges (paper, Section 4.3,
//! Fig. 5-7): over $\mathbf x \in \lbrace 0, 1 \rbrace^L$,
//! - the range check $F_0(\mathbf x) = \widetilde{eq}(\tau_0, \mathbf x) \cdot P(\tilde w(\mathbf
//!   x))$ with $P(y) = \prod_{c \in D} (y - c)$ over the digit set $D$ of base $b$, claimed sum
//!   $0$;
//! - the linear check $F_\alpha(\mathbf x) = \tilde w(\mathbf x) \cdot \tilde g(\mathbf x)$, where
//!   $\tilde g$ is the public multilinear table $g(u \Vert \ell) = m^\ast(u) \alpha^\ell$
//!   (`crate::relation`), claimed sum $\sum_i \widetilde{eq}(\tau_1, i) y_i(\alpha)$.
//!
//! Each round the prover sends both round polynomials as their values at $t = 0, 1, \dots$,
//! $\deg$: $|D| + 2$ values for $F_0$ (degree $|D| + 1$) and 3 for $F_\alpha$ (degree 2). After
//! the last round it holds $\tilde w(\mathbf a)$ at the point $\mathbf a$ of the round
//! challenges, bound in order from the first (most significant) variable.

use crate::error::VerifyError;
use crate::fs::Fs;
use crate::mle::interpolate;
use crate::{Error, Result};
use hachi_co_math::{Ext, Fqk};

/// One round's messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundMessage<const K: usize> {
    /// $F_0$'s round polynomial at $t = 0, \dots, |D| + 1$.
    pub h0: Vec<Ext<K>>,
    /// $F_\alpha$'s round polynomial at $t = 0, 1, 2$.
    pub h_alpha: Vec<Ext<K>>,
}

/// $P(y) = \prod_{c \in D} (y - c)$, with the digits given as residues.
pub fn range_poly<const K: usize>(field: &Fqk<K>, digits: &[u64], y: &Ext<K>) -> Ext<K> {
    digits.iter().fold(field.one(), |acc, c| {
        field.mul(&acc, &field.sub(y, &field.from_base(*c)))
    })
}

/// The round labels of the transcript.
const LABEL_H0: &[u8] = b"hachi/sumcheck/h0";
const LABEL_HA: &[u8] = b"hachi/sumcheck/h_alpha";
const LABEL_ROUND: &[u8] = b"hachi/sumcheck/round";

/// Runs the prover on the tables of $\tilde w$, $\widetilde{eq}(\tau_0, \cdot)$ and $\tilde g$
/// (each of $2^L$ values) and returns the round messages, the point $\mathbf a$ and
/// $\tilde w(\mathbf a)$.
///
/// # Errors
/// [`Error::Shape`] if the tables differ in length or are not of a power-of-two length.
pub fn prove<const K: usize>(
    field: &Fqk<K>,
    digits: &[u64],
    mut w: Vec<Ext<K>>,
    mut eq0: Vec<Ext<K>>,
    mut g: Vec<Ext<K>>,
    fs: &mut Fs,
) -> Result<(Vec<RoundMessage<K>>, Vec<Ext<K>>, Ext<K>)> {
    let n = w.len();
    if !n.is_power_of_two() || eq0.len() != n || g.len() != n {
        return Err(Error::Shape(
            "the sumcheck tables must have one power-of-two length".into(),
        ));
    }
    let rounds = n.trailing_zeros() as usize;
    let deg0 = digits.len() + 1;
    let nodes0: Vec<Ext<K>> = (0..=deg0 as u64).map(|t| field.from_base(t)).collect();
    let mut messages = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let half = w.len() / 2;
        let mut h0 = vec![field.zero(); deg0 + 1];
        let mut h_alpha = vec![field.zero(); 3];
        for i in 0..half {
            let (w0, w1) = (w[i], w[i + half]);
            let (e0, e1) = (eq0[i], eq0[i + half]);
            let (g0, g1) = (g[i], g[i + half]);
            let (dw, de, dg) = (
                field.sub(&w1, &w0),
                field.sub(&e1, &e0),
                field.sub(&g1, &g0),
            );
            for (t, node) in nodes0.iter().enumerate() {
                let wt = field.add(&w0, &field.mul(node, &dw));
                let et = field.add(&e0, &field.mul(node, &de));
                h0[t] = field.add(&h0[t], &field.mul(&et, &range_poly(field, digits, &wt)));
                if t < 3 {
                    let gt = field.add(&g0, &field.mul(node, &dg));
                    h_alpha[t] = field.add(&h_alpha[t], &field.mul(&wt, &gt));
                }
            }
        }
        fs.exts(LABEL_H0, &h0);
        fs.exts(LABEL_HA, &h_alpha);
        let r = fs.field_coins::<K>(LABEL_ROUND, field.modulus(), 1)?[0];
        for table in [&mut w, &mut eq0, &mut g] {
            let (lo, hi) = table.split_at(half);
            *table = lo
                .iter()
                .zip(hi)
                .map(|(a, b)| field.add(a, &field.mul(&r, &field.sub(b, a))))
                .collect();
        }
        messages.push(RoundMessage { h0, h_alpha });
        point.push(r);
    }
    Ok((messages, point, w[0]))
}

/// What the verifier is left with: the point $\mathbf a$ and the claims on $F_0(\mathbf a)$ and
/// $F_\alpha(\mathbf a)$.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reduced<const K: usize> {
    /// The point of the round challenges.
    pub point: Vec<Ext<K>>,
    /// The claimed $F_0(\mathbf a)$.
    pub claim0: Ext<K>,
    /// The claimed $F_\alpha(\mathbf a)$.
    pub claim_alpha: Ext<K>,
}

/// Verifies the rounds against the claimed sums $0$ and `claim_alpha`, absorbing the messages
/// and drawing the coins as [`prove`] does.
///
/// # Errors
/// [`VerifyError::Malformed`] for messages of the wrong number or length;
/// [`VerifyError::Check`] if a round's values do not add up to the running claim.
pub fn verify<const K: usize>(
    field: &Fqk<K>,
    digits: usize,
    rounds: &[RoundMessage<K>],
    expected_rounds: usize,
    claim_alpha: Ext<K>,
    fs: &mut Fs,
) -> core::result::Result<Reduced<K>, VerifyError> {
    if rounds.len() != expected_rounds {
        return Err(VerifyError::Malformed(format!(
            "{} sumcheck rounds, {expected_rounds} expected",
            rounds.len()
        )));
    }
    let deg0 = digits + 1;
    let mut claim0 = field.zero();
    let mut claim_a = claim_alpha;
    let mut point = Vec::with_capacity(rounds.len());
    for (i, msg) in rounds.iter().enumerate() {
        if msg.h0.len() != deg0 + 1 || msg.h_alpha.len() != 3 {
            return Err(VerifyError::Malformed(format!(
                "round {}: {} and {} values, {} and 3 expected",
                i + 1,
                msg.h0.len(),
                msg.h_alpha.len(),
                deg0 + 1
            )));
        }
        if field.add(&msg.h0[0], &msg.h0[1]) != field.reduce(&claim0) {
            return Err(VerifyError::Check(format!(
                "range sumcheck, round {}",
                i + 1
            )));
        }
        if field.add(&msg.h_alpha[0], &msg.h_alpha[1]) != field.reduce(&claim_a) {
            return Err(VerifyError::Check(format!(
                "linear sumcheck, round {}",
                i + 1
            )));
        }
        fs.exts(LABEL_H0, &msg.h0);
        fs.exts(LABEL_HA, &msg.h_alpha);
        let r = fs.field_coins::<K>(LABEL_ROUND, field.modulus(), 1)?[0];
        claim0 = interpolate(field, &msg.h0, &r);
        claim_a = interpolate(field, &msg.h_alpha, &r);
        point.push(r);
    }
    Ok(Reduced {
        point,
        claim0,
        claim_alpha: claim_a,
    })
}
