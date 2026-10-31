//! The partial evaluations of a multilinear opening, and the sumcheck that binds them to the
//! committed polynomial. This deviates from the paper's Section 3.2, whose check does not bind
//! them. The same gadget ([`prove_table`], [`verify`]) also binds the partial evaluations of
//! every iteration's next-claim reduction (`iteration`), where the committed polynomial is the
//! packed next witness in hypercube form; the two runs use different transcript labels
//! ([`Labels`]). Below, $f'$ stands for whichever polynomial the caller binds.
//!
//! **The paper's reduction** (Section 3.2). Write $f \in \mathbb Z_q^{\le 1}\[X_1, \dots,
//! X_\ell\]$ as $\sum_i \mathrm{mono}\_i(X_1, \dots, X_\kappa) f_i(X_{\kappa+1}, \dots, X_\ell)$
//! over $i \in \lbrace 0, 1 \rbrace^\kappa$, $\kappa = \log_2 K$, where the $f_i$ have
//! coefficients in $\mathbb F_q$. The committed polynomial is $f' = \sum_i Z_i f_i \in \mathbb
//! F_{q^K}\[X_{\kappa+1}, \dots, X_\ell\]$, with $Z_i$ the basis of the coordinates of `Ext`
//! (Hachi's Eq. (7)). At $\mathbf x$ the prover sends the partial evaluations $y_i = f_i(\mathbf
//! x')$, $\mathbf x' = (x_{\kappa+1}, \dots, x_\ell)$. The verifier checks $f(\mathbf x) =
//! \sum_i \mathrm{mono}\_i(x_1, \dots, x_\kappa) y_i$, and the rest of the opening proves
//! $f'(\mathbf x') = \sum_i y_i Z_i$.
//!
//! **Why that is not enough.** The $y_i$ lie in $\mathbb F_{q^K}$, so $\sum_i y_i Z_i$ does not
//! determine them: $(y_i)\_i \mapsto \sum_i y_i Z_i$ is $\mathbb F_{q^K}$-linear with a kernel
//! of dimension $K - 1$. Unless $(\mathrm{mono}\_i)\_i$ is a multiple of $(Z_i)\_i$, that kernel
//! contains a $\boldsymbol\delta$ with $\sum_i \mathrm{mono}\_i \delta_i = 1$. Adding a multiple
//! of it to the partial evaluations keeps $f'(\mathbf x')$ and moves the value anywhere. Version
//! 2 of the scheme accepted such forgeries with 0, 1 and 2 iterations; the test
//! `partials_moved_along_the_trace_kernel_are_rejected` of `pcs` builds one.
//!
//! **What is proved instead: ring-switching.** This is Construction 3.1 of Diamond and Posen
//! (ePrint 2024/504), which holds for any extension of degree $2^\kappa$ in any characteristic;
//! Akita (ePrint 2026/1983, Section 3.7) calls it the tensor reduction. Write
//! $y_i = \sum_u y_{u,i} Z_u$ with $y_{u,i} \in \mathbb F_q$, and
//! let the *rows* be $\hat y_u = \sum_i y_{u,i} Z_i$: the matrix of coordinates of the partial
//! evaluations, transposed (`transpose`). Write also $\widetilde{eq}(\mathbf x', \mathbf b) =
//! \sum_u A_u(\mathbf b) Z_u$ with $A_u(\mathbf b) \in \mathbb F_q$. As $f_i(\mathbf x') =
//! \sum_{\mathbf b} f_i(\mathbf b) \widetilde{eq}(\mathbf x', \mathbf b)$ and the values
//! $f_i(\mathbf b)$ lie in $\mathbb F_q$, the true partial evaluations have $y_{u,i} =
//! \sum_{\mathbf b} f_i(\mathbf b) A_u(\mathbf b)$, so for every $u$
//! $$\hat y_u = \sum_{\mathbf b} A_u(\mathbf b) T(\mathbf b), \qquad T = \sum_i f_i Z_i,$$
//! with $T$ the values of $f'$ on $\lbrace 0, 1 \rbrace^n$, $n = \ell - \kappa$
//! ([`hypercube_values`]). Unlike the paper's check, these $K$ claims hold only for the true
//! partial evaluations: the $A_u(\mathbf b)$ lie in $\mathbb F_q$, so coordinate $i$ of the
//! right-hand side is $\sum_{\mathbf b} A_u(\mathbf b) f_i(\mathbf b)$, and coordinate $i$ of
//! claim $u$ fixes $y_{u,i}$. (The test `the_rows_determine_the_partials` checks the forgery's
//! $\boldsymbol\delta$ against this.)
//!
//! **Batching.** After the partial evaluations, the verifier draws $\boldsymbol\eta \in \mathbb
//! F_{q^K}^\kappa$, and $\eta_u = \widetilde{eq}(\boldsymbol\eta, \mathrm{bit}(u))$. The $K$
//! claims batch into
//! $$V = \sum_u \eta_u \hat y_u = \sum_{\mathbf b} T(\mathbf b) W(\mathbf b), \qquad W(\mathbf b)
//! = \sum_u \eta_u A_u(\mathbf b)$$
//! (`weights`, one table of $\widetilde{eq}$ and its coordinates). A sumcheck of $n$ rounds
//! ([`prove`], [`verify`]) reduces $V$ to a claim on $T(\mathbf r) W(\mathbf r)$ at the point
//! $\mathbf r$ of its challenges. Each round sends its polynomial, of degree 2, at $0$ and $2$;
//! the verifier takes its value at 1 from the running claim, absorbs the two values and draws
//! the round's challenge. The prover then sends $f'(\mathbf r) = T(\mathbf r)$, and the verifier
//! checks $f'(\mathbf r) W(\mathbf r)$ against the last claim. What is left is the evaluation
//! $f'(\mathbf r)$, which the paper's Section 3.1 reduces to $\mathcal R_q$ as it did
//! $f'(\mathbf x')$ (`pcs`).
//!
//! **The verifier's weight.** In the tensor algebra $\mathbb F_{q^K} \otimes_{\mathbb F_q}
//! \mathbb F_{q^K}$, let
//! $$\prod_j \bigl((1 - x_j) \otimes (1 - r_j) + x_j \otimes r_j\bigr) = \sum_u Z_u \otimes
//! e_u.$$
//! Then $W(\mathbf r) = \sum_u \eta_u e_u$ (`row_weights`): both sides are multilinear in
//! $\mathbf r$, and on $\mathbf r = \mathbf b \in \lbrace 0, 1 \rbrace^n$ the product is
//! $\widetilde{eq}(\mathbf x', \mathbf b) \otimes 1 = \sum_u Z_u \otimes A_u(\mathbf b)$.
//! (Diamond and Posen's faster formula, their Remark 3.4, holds only in characteristic 2.)
//!
//! **The partial evaluation left out.** The verifier knows the value $v = \sum_i w_i y_i$ with
//! public weights: $w_i = \mathrm{mono}\_i(x_1, \dots, x_\kappa)$ here,
//! $\widetilde{eq}(i, \mathbf a_1)$ in an iteration. So the proof leaves out the first $y_i$
//! with $w_i \neq 0$ ([`omit`]), and the verifier derives it from $v$ ([`complete`]): the
//! check of $v$ holds by construction (Hachi, footnote 5).
//!
//! **Earlier versions.** Applying $\varphi : a \mapsto a^q$ ([`Fqk::frobenius`]), the claims
//! above are, by the invertible Moore matrix $(\varphi^t(Z_u))\_{t,u}$, the claims
//! $f'(\varphi^t(\mathbf x')) = \sum_u \varphi^t(Z_u) \hat y_u$ on the orbit of $\mathbf x'$.
//! Version 3 of the encodings proved those, batched by the powers of one challenge. That is the
//! reduction of Hashcaster (Soukhanov, 2024), which Diamond and Posen, Section 1.3, show to be
//! ring-switching read through $\mathbb F_{q^K} \otimes_{\mathbb F_q} \mathbb F_{q^K} \cong
//! \mathbb F_{q^K}^K$; only the batching coefficients differ. The test
//! `the_rows_are_the_claims_on_the_orbit` checks the identity.
//!
//! **Soundness** (interactive bounds; the opening applies Fiat-Shamir to them as to the rest of
//! the protocol). Let $f'$ be the polynomial the commitment binds. If a partial evaluation is
//! false, $\sum_u \widetilde{eq}(\boldsymbol\eta, \mathrm{bit}(u)) (\hat y_u - \sum_{\mathbf b}
//! A_u(\mathbf b) T(\mathbf b))$ is a non-zero multilinear polynomial in $\boldsymbol\eta$, which
//! vanishes with probability at most $\kappa/q^K$ (Schwartz-Zippel). Otherwise the claim $V$ is
//! false, and by the standard analysis of the sumcheck (Lund, Fortnow, Karloff and Nisan) the
//! verifier ends on a true last claim with probability at most $2n/q^K$: two per round, for
//! round polynomials of degree 2. This is the bound of Akita's Theorem 5.1 for one claim. That
//! the commitment binds a single $f'$ across the extractor's forks is Akita's Lemma 10.28;
//! the composition with Hachi's iterations is not proved here.
//!
//! **Cost.** With the partial evaluation left out, the proof gains $2n + 1$ field elements over
//! the paper's $K - 1$. The prover adds $2^n$ field multiplications for $\widetilde{eq}$, $K
//! 2^n$ products of a field element by a coordinate for $W$, and $O(2^n)$ field multiplications
//! for the rounds, small next to the iterations; the verifier adds $O(Kn)$.
//!
//! **Bit order.** As in [`crate::mle`]: a table index has the first variable as its most
//! significant bit, the monomial coefficients of [`hypercube_values`] are ordered as those of
//! `Subfield::monomials`, and the rounds bind the first variable first.

use crate::error::VerifyError;
use crate::fs::Fs;
use crate::mle::{eq_table, interpolate};
use crate::{Error, Result};
use hachi_co_math::{Ext, Fqk};

/// The transcript labels of one run of the batching sumcheck. The same gadget binds the partial
/// evaluations of a multilinear input ([`INPUT_LABELS`], `pcs`) and of every iteration's next
/// claim (`iteration`); each caller passes its own labels so the two runs never share a frame.
pub struct Labels {
    /// The point $\boldsymbol\eta$ that batches the $K$ claims on the rows.
    pub eta: &'static [u8],
    /// A round's polynomial at $0$ and $2$.
    pub round_poly: &'static [u8],
    /// A round's challenge.
    pub challenge: &'static [u8],
    /// The final evaluation $f'(\mathbf r)$.
    pub eval: &'static [u8],
}

/// The labels of the input reduction's batching sumcheck (`pcs`).
pub const INPUT_LABELS: Labels = Labels {
    eta: b"hachi/input/eta",
    round_poly: b"hachi/input/h",
    challenge: b"hachi/input/round",
    eval: b"hachi/input/final",
};

/// One round of the batching sumcheck: its polynomial at $0$ and $2$. The value at 1 is the
/// running claim minus the value at 0.
pub type RoundValues<const K: usize> = [Ext<K>; 2];

/// The $K \times K$ matrix of coordinates of `columns` ($K$ field elements) with its two indices
/// swapped: coordinate $i$ of entry $u$ of the result is coordinate $u$ of `columns[i]`. Of the
/// partial evaluations, these are the rows $\hat y_u$ (module documentation).
fn transpose<const K: usize>(columns: &[Ext<K>]) -> [Ext<K>; K] {
    core::array::from_fn(|u| Ext(core::array::from_fn(|i| columns[i].0[u])))
}

/// The first index with a non-zero weight, whose partial evaluation a proof leaves out (module
/// documentation). `None` only if every weight is zero, which neither kind of weights can be:
/// $\mathrm{mono}\_0 = 1$, and the $\widetilde{eq}(i, \mathbf a_1)$ add up to 1.
fn omitted<const K: usize>(field: &Fqk<K>, weights: &[Ext<K>]) -> Option<usize> {
    weights.iter().position(|w| !field.is_zero(w))
}

/// What a proof sends of the $K$ partial evaluations `partials` for the public `weights`: all
/// but the one the verifier derives (module documentation).
///
/// # Errors
/// [`Error::Protocol`] if every weight is zero.
pub fn omit<const K: usize>(
    field: &Fqk<K>,
    weights: &[Ext<K>],
    partials: &[Ext<K>],
) -> Result<Vec<Ext<K>>> {
    let at = omitted(field, weights)
        .ok_or_else(|| Error::Protocol("every weight of the partial evaluations is zero".into()))?;
    Ok(partials
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != at)
        .map(|(_, y)| *y)
        .collect())
}

/// The $K$ partial evaluations from the $K - 1$ of `sent`: the one left out is the solution of
/// $\sum_i w_i y_i =$ `value` for the public `weights` (module documentation).
///
/// # Errors
/// [`VerifyError::Malformed`] unless there are $K$ weights, not all zero, and $K - 1$ values.
pub fn complete<const K: usize>(
    field: &Fqk<K>,
    weights: &[Ext<K>],
    value: &Ext<K>,
    sent: &[Ext<K>],
) -> core::result::Result<Vec<Ext<K>>, VerifyError> {
    let at = omitted(field, weights)
        .filter(|_| weights.len() == K && sent.len() + 1 == K)
        .ok_or_else(|| {
            VerifyError::Malformed(format!(
                "{} partial evaluations for {} weights, {} and {K} expected",
                sent.len(),
                weights.len(),
                K - 1
            ))
        })?;
    let mut partials = sent.to_vec();
    partials.insert(at, field.zero());
    let rest = batched(field, weights, &partials);
    let inverse = field
        .inv(&weights[at])
        .ok_or_else(|| VerifyError::Malformed("a zero weight was chosen".into()))?;
    partials[at] = field.mul(&field.sub(value, &rest), &inverse);
    Ok(partials)
}

/// The batching coefficients $\eta_u = \widetilde{eq}(\boldsymbol\eta, \mathrm{bit}(u))$ of the
/// $\kappa$ coins drawn under `labels` (module documentation).
///
/// # Errors
/// Sampling errors.
fn batching_coefficients<const K: usize>(
    field: &Fqk<K>,
    labels: &Labels,
    fs: &mut Fs,
) -> Result<Vec<Ext<K>>> {
    let kappa = K.trailing_zeros() as usize;
    Ok(eq_table(
        field,
        &fs.field_coins::<K>(labels.eta, field.modulus(), kappa)?,
    ))
}

/// The table of $W(\mathbf b) = \sum_u \eta_u A_u(\mathbf b)$ on $\lbrace 0, 1 \rbrace^n$ for
/// `etas` $= (\eta_u)\_u$, with $A_u(\mathbf b)$ coordinate $u$ of
/// $\widetilde{eq}(\mathbf x', \mathbf b)$ (module documentation).
fn weights<const K: usize>(field: &Fqk<K>, etas: &[Ext<K>], x: &[Ext<K>]) -> Vec<Ext<K>> {
    eq_table(field, x)
        .iter()
        .map(|e| {
            etas.iter().zip(e.0).fold(field.zero(), |acc, (eta, a)| {
                field.add(&acc, &field.scale(eta, a))
            })
        })
        .collect()
}

/// The $(e_u)\_u$ of $\prod_j ((1 - x_j) \otimes (1 - r_j) + x_j \otimes r_j) = \sum_u Z_u \otimes
/// e_u$ (module documentation), so that $W(\mathbf r) = \sum_u \eta_u e_u$. An element $\sum_u
/// Z_u \otimes a_u$ of the tensor algebra is held by its rows $(a_u)\_u$, with $Z_0 = 1$. The
/// transpose gives its columns: $x \otimes 1$ acts on the columns and $1 \otimes y$ on the rows,
/// each by multiplication in $\mathbb F_{q^K}$.
fn row_weights<const K: usize>(field: &Fqk<K>, x: &[Ext<K>], r: &[Ext<K>]) -> [Ext<K>; K] {
    let mut rows = [field.zero(); K];
    rows[0] = field.one();
    for (xj, rj) in x.iter().zip(r) {
        let columns = transpose(&rows);
        let high = columns.map(|c| field.mul(xj, &c));
        let low: [Ext<K>; K] = core::array::from_fn(|v| field.sub(&columns[v], &high[v]));
        let (high, low) = (transpose(&high), transpose(&low));
        let one_minus = field.sub(&field.one(), rj);
        rows = core::array::from_fn(|u| {
            field.add(&field.mul(&low[u], &one_minus), &field.mul(&high[u], rj))
        });
    }
    rows
}

/// The values on $\lbrace 0, 1 \rbrace^n$ of the multilinear polynomial with the $2^n$
/// monomial coefficients `coeffs`, in the same order: the subset sums $T(\mathbf b) =
/// \sum_{\mathbf j \le \mathbf b} c_{\mathbf j}$ over the $\mathbf j$ whose bits are among
/// those of $\mathbf b$.
///
/// # Errors
/// [`Error::Shape`] unless the length is a power of two.
pub fn hypercube_values<const K: usize>(field: &Fqk<K>, coeffs: &[Ext<K>]) -> Result<Vec<Ext<K>>> {
    if !coeffs.len().is_power_of_two() {
        return Err(Error::Shape(
            "a power-of-two number of coefficients is expected".into(),
        ));
    }
    let mut table = coeffs.to_vec();
    let mut half = 1;
    while half < table.len() {
        for block in table.chunks_exact_mut(2 * half) {
            let (lo, hi) = block.split_at_mut(half);
            for (h, l) in hi.iter_mut().zip(lo.iter()) {
                *h = field.add(h, l);
            }
        }
        half *= 2;
    }
    Ok(table)
}

/// $\sum_u c_u v_u$.
fn batched<const K: usize>(field: &Fqk<K>, coefficients: &[Ext<K>], values: &[Ext<K>]) -> Ext<K> {
    coefficients
        .iter()
        .zip(values)
        .fold(field.zero(), |acc, (c, v)| {
            field.add(&acc, &field.mul(c, v))
        })
}

/// The prover on a polynomial given by its $2^n$ monomial `coeffs` ([`prove_table`] on the
/// hypercube values of `coeffs`, with the input reduction's labels). `pcs` uses this for a
/// multilinear input.
///
/// # Errors
/// As [`prove_table`].
pub fn prove<const K: usize>(
    field: &Fqk<K>,
    coeffs: &[Ext<K>],
    x: &[Ext<K>],
    partials: &[Ext<K>],
    fs: &mut Fs,
) -> Result<(Vec<RoundValues<K>>, Vec<Ext<K>>, Ext<K>)> {
    prove_table(
        field,
        hypercube_values(field, coeffs)?,
        x,
        partials,
        &INPUT_LABELS,
        fs,
    )
}

/// The prover, run after the statement and the partial evaluations were absorbed: draws
/// $\boldsymbol\eta$, runs the rounds on the table `t` of $f'$ on $\lbrace 0, 1 \rbrace^n$
/// (hypercube form, $2^n$ values), and absorbs $f'(\mathbf r)$. Returns the rounds, the point
/// $\mathbf r$ and $f'(\mathbf r)$. `x` is $\mathbf x'$; `partials`, all $K$ of them, serve only
/// to check the honest prover's invariant, the first round's sum $V$; `labels` frame the
/// transcript.
///
/// # Errors
/// [`Error::Shape`] unless `t` has $2^{|\mathbf x'|}$ values and there are $K$ partial
/// evaluations; [`Error::Protocol`] if the partial evaluations are not those of $f'$ at
/// $\mathbf x'$; sampling errors.
pub fn prove_table<const K: usize>(
    field: &Fqk<K>,
    mut t: Vec<Ext<K>>,
    x: &[Ext<K>],
    partials: &[Ext<K>],
    labels: &Labels,
    fs: &mut Fs,
) -> Result<(Vec<RoundValues<K>>, Vec<Ext<K>>, Ext<K>)> {
    if x.len() >= usize::BITS as usize || t.len() != 1 << x.len() || partials.len() != K {
        return Err(Error::Shape(format!(
            "2^{} table values and {K} partial evaluations are expected",
            x.len()
        )));
    }
    let etas = batching_coefficients(field, labels, fs)?;
    let mut w = weights(field, &etas, x);
    let sum = batched(field, &etas, &transpose(partials));
    let mut rounds = Vec::with_capacity(x.len());
    let mut point = Vec::with_capacity(x.len());
    for round in 0..x.len() {
        let half = t.len() / 2;
        let mut h = [field.zero(); 3];
        for i in 0..half {
            let (t0, t1, w0, w1) = (t[i], t[i + half], w[i], w[i + half]);
            let t2 = field.sub(&field.add(&t1, &t1), &t0);
            let w2 = field.sub(&field.add(&w1, &w1), &w0);
            h[0] = field.add(&h[0], &field.mul(&t0, &w0));
            h[1] = field.add(&h[1], &field.mul(&t1, &w1));
            h[2] = field.add(&h[2], &field.mul(&t2, &w2));
        }
        if round == 0 && field.add(&h[0], &h[1]) != sum {
            return Err(Error::Protocol(
                "the partial evaluations are not those of the committed polynomial".into(),
            ));
        }
        let sent = [h[0], h[2]];
        fs.exts(labels.round_poly, &sent);
        let r = fs.field_coins::<K>(labels.challenge, field.modulus(), 1)?[0];
        for table in [&mut t, &mut w] {
            for i in 0..half {
                let (lo, hi) = (table[i], table[i + half]);
                table[i] = field.add(&lo, &field.mul(&r, &field.sub(&hi, &lo)));
            }
            table.truncate(half);
        }
        rounds.push(sent);
        point.push(r);
    }
    let value = t[0];
    fs.exts(labels.eval, &[value]);
    Ok((rounds, point, value))
}

/// The verifier, run after the statement and the partial evaluations were absorbed: draws
/// $\boldsymbol\eta$, follows the rounds from $V = \sum_u \eta_u \hat y_u$ and checks
/// $f'(\mathbf r)$ = `value` against the last claim, absorbing as [`prove_table`] does under the
/// same `labels`. Returns $\mathbf r$, at which the rest of the opening must prove
/// $f'(\mathbf r)$ = `value`. `x` is $\mathbf x'$; `partials`, all $K$ of them, come from
/// [`complete`].
///
/// # Errors
/// [`VerifyError::Malformed`] unless there are $|\mathbf x'|$ rounds and $K$ partial
/// evaluations; [`VerifyError::Check`] if the last claim is not $f'(\mathbf r) W(\mathbf r)$.
pub fn verify<const K: usize>(
    field: &Fqk<K>,
    x: &[Ext<K>],
    partials: &[Ext<K>],
    rounds: &[RoundValues<K>],
    value: &Ext<K>,
    labels: &Labels,
    fs: &mut Fs,
) -> core::result::Result<Vec<Ext<K>>, VerifyError> {
    if rounds.len() != x.len() || partials.len() != K {
        return Err(VerifyError::Malformed(format!(
            "{} rounds of the batching sumcheck and {} partial evaluations, {} and {K} expected",
            rounds.len(),
            partials.len(),
            x.len()
        )));
    }
    let etas = batching_coefficients(field, labels, fs)?;
    let mut claim = batched(field, &etas, &transpose(partials));
    let mut point = Vec::with_capacity(rounds.len());
    for h in rounds {
        fs.exts(labels.round_poly, h);
        let r = fs.field_coins::<K>(labels.challenge, field.modulus(), 1)?[0];
        claim = interpolate(field, &[h[0], field.sub(&claim, &h[0]), h[1]], &r);
        point.push(r);
    }
    let weight = batched(field, &etas, &row_weights(field, x, &point));
    if field.mul(value, &weight) != claim {
        return Err(VerifyError::Check(
            "the batching sumcheck's final value".into(),
        ));
    }
    fs.exts(labels.eval, core::slice::from_ref(value));
    Ok(point)
}

/// A dishonest prover, for the tests: the rounds of [`prove`] on the true tables, followed from
/// the running claim that starts from the batched value of `partials`, and a final value fitted
/// to the last claim. It passes [`verify`] whatever the partial evaluations, with a final value
/// that is not $f'(\mathbf r)$ when they are false: the rest of the opening must reject it.
#[cfg(test)]
pub(crate) fn prove_fitted<const K: usize>(
    field: &Fqk<K>,
    coeffs: &[Ext<K>],
    x: &[Ext<K>],
    partials: &[Ext<K>],
    labels: &Labels,
    fs: &mut Fs,
) -> (Vec<RoundValues<K>>, Vec<Ext<K>>, Ext<K>) {
    let etas = batching_coefficients(field, labels, fs).unwrap();
    let mut t = hypercube_values(field, coeffs).unwrap();
    let mut w = weights(field, &etas, x);
    let mut claim = batched(field, &etas, &transpose(partials));
    let (mut rounds, mut point) = (Vec::new(), Vec::new());
    for _ in 0..x.len() {
        let half = t.len() / 2;
        let mut sent = [field.zero(); 2];
        for i in 0..half {
            let (t0, t1, w0, w1) = (t[i], t[i + half], w[i], w[i + half]);
            let t2 = field.sub(&field.add(&t1, &t1), &t0);
            let w2 = field.sub(&field.add(&w1, &w1), &w0);
            sent[0] = field.add(&sent[0], &field.mul(&t0, &w0));
            sent[1] = field.add(&sent[1], &field.mul(&t2, &w2));
        }
        fs.exts(labels.round_poly, &sent);
        let r = fs
            .field_coins::<K>(labels.challenge, field.modulus(), 1)
            .unwrap()[0];
        for table in [&mut t, &mut w] {
            for i in 0..half {
                let (lo, hi) = (table[i], table[i + half]);
                table[i] = field.add(&lo, &field.mul(&r, &field.sub(&hi, &lo)));
            }
            table.truncate(half);
        }
        claim = interpolate(field, &[sent[0], field.sub(&claim, &sent[0]), sent[1]], &r);
        rounds.push(sent);
        point.push(r);
    }
    let value = field.mul(&claim, &field.inv(&w[0]).unwrap());
    fs.exts(labels.eval, &[value]);
    (rounds, point, value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iteration::basis;
    use crate::mle::mle_eval;
    use hachi_co_math::{Ring, Subfield};

    const K: usize = 4;
    const Q: u64 = 8_388_637;

    /// Deterministic pseudorandom residues (SplitMix64, reduced).
    fn residues(seed: u64, len: usize) -> Vec<u64> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                (z ^ (z >> 31)) % Q
            })
            .collect()
    }

    fn exts(seed: u64, len: usize) -> Vec<Ext<K>> {
        residues(seed, len * K)
            .as_chunks::<K>()
            .0
            .iter()
            .map(|c| Ext(*c))
            .collect()
    }

    fn subfield() -> Subfield<K> {
        Subfield::new(&Ring::new(Q, 16).unwrap()).unwrap()
    }

    /// $\sum_j c_j \mathrm{mono}\_j(\mathbf p)$, the evaluation in monomial form.
    fn evaluate(sub: &Subfield<K>, coeffs: &[Ext<K>], p: &[Ext<K>]) -> Ext<K> {
        let field = sub.field();
        let mono = sub.monomials(p).unwrap();
        coeffs.iter().zip(&mono).fold(field.zero(), |acc, (c, m)| {
            field.add(&acc, &field.mul(c, m))
        })
    }

    /// A random $f$ over $\mathbb F_q$ in $\kappa + n$ variables, as $f' = \sum_i Z_i f_i$
    /// (monomial coefficients) and the $f_i$.
    fn polynomial(seed: u64, n: usize) -> (Vec<Ext<K>>, Vec<Vec<Ext<K>>>) {
        let coeffs = residues(seed, K << n);
        let inner = 1usize << n;
        let packed = (0..inner)
            .map(|j| Ext(core::array::from_fn(|i| coeffs[i * inner + j])))
            .collect();
        let parts = (0..K)
            .map(|i| {
                coeffs[i * inner..(i + 1) * inner]
                    .iter()
                    .map(|c| Ext(core::array::from_fn(|m| if m == 0 { *c } else { 0 })))
                    .collect()
            })
            .collect();
        (packed, parts)
    }

    /// $\varphi^t(x)$, for $t$ of either sign.
    fn conjugate(field: &Fqk<K>, x: &Ext<K>, t: isize) -> Ext<K> {
        (0..t.rem_euclid(K as isize)).fold(*x, |y, _| field.frobenius(&y))
    }

    /// The table of $A_u$: coordinate $u$ of $\widetilde{eq}(\mathbf x', \mathbf b)$.
    fn coordinates(field: &Fqk<K>, x: &[Ext<K>], u: usize) -> Vec<Ext<K>> {
        eq_table(field, x)
            .iter()
            .map(|e| field.from_base(e.0[u]))
            .collect()
    }

    #[test]
    fn the_rows_are_sums_of_the_table_with_base_field_weights() {
        // y^_u = sum_b A_u(b) T(b): the identity the reduction rests on, for the polynomial in
        // monomial form and its table in hypercube form, which also fixes the bit order.
        let sub = subfield();
        let field = sub.field();
        for n in [1usize, 2, 5] {
            let (packed, parts) = polynomial(n as u64, n);
            let x = exts(100 + n as u64, n);
            let partials: Vec<Ext<K>> = parts.iter().map(|p| evaluate(&sub, p, &x)).collect();
            let table = hypercube_values(field, &packed).unwrap();
            let rows = transpose(&partials);
            for (u, row) in rows.iter().enumerate() {
                let sum = table
                    .iter()
                    .zip(coordinates(field, &x, u))
                    .fold(field.zero(), |acc, (t, a)| {
                        field.add(&acc, &field.mul(t, &a))
                    });
                assert_eq!(sum, *row, "n {n}, row {u}");
            }
        }
    }

    #[test]
    fn the_rows_are_the_claims_on_the_orbit() {
        // Diamond and Posen's Eq. (16), and the claims of version 3: f'(phi^t(x')) =
        // sum_u phi^t(Z_u) y^_u, with the Frobenius in the direction that the tests of version 3
        // fixed: phi^{-t}(x') carries sum_i Z_i phi^{-t}(y_i).
        let sub = subfield();
        let field = sub.field();
        for n in [1usize, 2, 5] {
            let (packed, parts) = polynomial(10 + n as u64, n);
            let x = exts(200 + n as u64, n);
            let partials: Vec<Ext<K>> = parts.iter().map(|p| evaluate(&sub, p, &x)).collect();
            let rows = transpose(&partials);
            for t in 0..K as isize {
                let orbit: Vec<Ext<K>> = x.iter().map(|xj| conjugate(field, xj, t)).collect();
                let by_rows = rows.iter().enumerate().fold(field.zero(), |acc, (u, row)| {
                    let z = conjugate(field, &basis::<K>(u), t);
                    field.add(&acc, &field.mul(&z, row))
                });
                assert_eq!(evaluate(&sub, &packed, &orbit), by_rows, "n {n}, t {t}");
                let by_partials = partials
                    .iter()
                    .enumerate()
                    .fold(field.zero(), |acc, (i, y)| {
                        let conjugated = conjugate(field, y, t);
                        field.add(&acc, &field.mul(&basis::<K>(i), &conjugated))
                    });
                assert_eq!(by_partials, by_rows, "n {n}, t {t}");
            }
        }
    }

    #[test]
    fn the_verifier_weight_is_the_extension_of_the_prover_table() {
        // row_weights in the tensor algebra against the multilinear extensions of the tables of
        // the A_u, at random points and at the corners, where it is the table itself.
        let field = subfield().field().clone();
        for n in [0usize, 1, 2, 6] {
            let x = exts(300 + n as u64, n);
            let tables: Vec<Vec<Ext<K>>> = (0..K).map(|u| coordinates(&field, &x, u)).collect();
            for seed in 0..4 {
                let r = exts(400 + seed, n);
                let got = row_weights(&field, &x, &r);
                for u in 0..K {
                    assert_eq!(got[u], mle_eval(&field, &tables[u], &r), "n {n}, row {u}");
                }
            }
            let corners = (0..1usize << n).map(|b| -> Vec<Ext<K>> {
                (0..n)
                    .map(|j| field.from_base(((b >> (n - 1 - j)) & 1) as u64))
                    .collect()
            });
            for (b, corner) in corners.enumerate() {
                let got = row_weights(&field, &x, &corner);
                let at_b: Vec<Ext<K>> = tables.iter().map(|table| table[b]).collect();
                assert_eq!(got.to_vec(), at_b, "n {n}, corner {b}");
            }
            // The batched weight, from one eq table, against the same extension.
            let etas = eq_table(&field, &exts(500, 2));
            assert_eq!(etas.len(), K);
            let table = weights(&field, &etas, &x);
            let r = exts(600, n);
            assert_eq!(
                batched(&field, &etas, &row_weights(&field, &x, &r)),
                mle_eval(&field, &table, &r),
                "n {n}"
            );
        }
    }

    #[test]
    fn hypercube_values_are_the_evaluations_at_the_corners() {
        let sub = subfield();
        let field = sub.field();
        let coeffs = exts(7, 16);
        let table = hypercube_values(field, &coeffs).unwrap();
        for (b, v) in table.iter().enumerate() {
            let corner: Vec<Ext<K>> = (0..4)
                .map(|j| field.from_base(((b >> (3 - j)) & 1) as u64))
                .collect();
            assert_eq!(*v, evaluate(&sub, &coeffs, &corner), "corner {b}");
        }
        assert!(hypercube_values(field, &coeffs[..3]).is_err());
    }

    #[test]
    fn the_rows_determine_the_partials() {
        // A delta in the kernel of the paper's check, sum_i delta_i Z_i = 0, moves the rows; for
        // random deltas the rows vanish only for delta = 0, as they are its coordinates.
        let sub = subfield();
        let field = sub.field();
        let ratio = field.mul(&basis::<K>(1), &field.inv(&basis::<K>(0)).unwrap());
        let mut delta = vec![field.zero(); K];
        delta[0] = field.neg(&ratio);
        delta[1] = field.one();
        let zero = delta.iter().enumerate().fold(field.zero(), |acc, (i, d)| {
            field.add(&acc, &field.mul(d, &basis::<K>(i)))
        });
        assert!(field.is_zero(&zero), "delta is in the kernel");
        assert!(transpose(&delta).iter().any(|v| !field.is_zero(v)));
        assert_eq!(transpose(&transpose(&delta)).to_vec(), delta);
        for seed in 0..50 {
            let delta = exts(1000 + seed, K);
            assert!(transpose(&delta).iter().any(|v| !field.is_zero(v)));
        }
        assert!(
            transpose(&[field.zero(); K])
                .iter()
                .all(|v| field.is_zero(v))
        );
    }

    #[test]
    fn the_partial_evaluation_left_out_is_derived_from_the_value() {
        let field = subfield().field().clone();
        let partials = exts(20, K);
        for (first, weights) in [
            // Non-zero weights; a zero first weight, as eq(i, a1) has when a coordinate is 1;
            // only the last weight left, as at the corners of eq.
            (0, exts(21, K)),
            (1, [vec![field.zero()], exts(22, K - 1)].concat()),
            (K - 1, [vec![field.zero(); K - 1], exts(23, 1)].concat()),
        ] {
            let value = batched(&field, &weights, &partials);
            let sent = omit(&field, &weights, &partials).unwrap();
            assert_eq!(sent.len(), K - 1);
            let mut left = partials.clone();
            left.remove(first);
            assert_eq!(sent, left, "the omitted index is {first}");
            assert_eq!(
                complete(&field, &weights, &value, &sent).unwrap(),
                partials,
                "first non-zero weight at {first}"
            );
            // Another value moves only the partial evaluation left out, and keeps the check.
            let other = field.add(&value, &field.one());
            let moved = complete(&field, &weights, &other, &sent).unwrap();
            assert_ne!(moved[first], partials[first]);
            assert_eq!(batched(&field, &weights, &moved), other);
        }
        // Malformed inputs: no panic.
        let weights = exts(24, K);
        let value = field.one();
        assert!(complete(&field, &weights, &value, &partials).is_err());
        assert!(complete(&field, &weights[1..], &value, &partials[1..]).is_err());
        let zeros = vec![field.zero(); K];
        assert!(complete(&field, &zeros, &value, &partials[1..]).is_err());
        assert!(matches!(
            omit(&field, &zeros, &partials),
            Err(Error::Protocol(_))
        ));
    }

    #[test]
    fn the_rounds_verify_and_altered_rounds_are_rejected() {
        let sub = subfield();
        let field = sub.field();
        let n = 6;
        let (packed, parts) = polynomial(3, n);
        let x = exts(4, n);
        let partials: Vec<Ext<K>> = parts.iter().map(|p| evaluate(&sub, p, &x)).collect();
        let context = [1u8; 32];
        let mut fs = Fs::new(&context);
        let (rounds, point, value) = prove(field, &packed, &x, &partials, &mut fs).unwrap();
        assert_eq!(rounds.len(), n);
        assert_eq!(value, evaluate(&sub, &packed, &point));
        let mut vfs = Fs::new(&context);
        let got = verify(
            field,
            &x,
            &partials,
            &rounds,
            &value,
            &INPUT_LABELS,
            &mut vfs,
        )
        .unwrap();
        assert_eq!(got, point);
        assert_eq!(fs.digest(), vfs.digest(), "the transcripts agree");
        let verdict = |partials: &[Ext<K>], rounds: &[RoundValues<K>], value: &Ext<K>| {
            verify(
                field,
                &x,
                partials,
                rounds,
                value,
                &INPUT_LABELS,
                &mut Fs::new(&context),
            )
        };
        let last = |v: core::result::Result<Vec<Ext<K>>, VerifyError>| matches!(v, Err(VerifyError::Check(m)) if m == "the batching sumcheck's final value");
        // Every value of every round, and the final value.
        for i in 0..n {
            for j in 0..2 {
                let mut bad = rounds.clone();
                bad[i][j] = field.add(&bad[i][j], &field.one());
                assert!(last(verdict(&partials, &bad, &value)), "round {i}, {j}");
            }
        }
        let other = field.add(&value, &field.one());
        assert!(last(verdict(&partials, &rounds, &other)));
        // Another partial evaluation, a missing round, a missing partial evaluation.
        let mut moved = partials.clone();
        moved[2] = field.add(&moved[2], &field.one());
        assert!(last(verdict(&moved, &rounds, &value)));
        assert!(matches!(
            verdict(&partials, &rounds[1..], &value),
            Err(VerifyError::Malformed(_))
        ));
        assert!(matches!(
            verdict(&partials[1..], &rounds, &value),
            Err(VerifyError::Malformed(_))
        ));
        // The honest prover refuses partial evaluations that are not those of f'.
        assert!(matches!(
            prove(field, &packed, &x, &moved, &mut Fs::new(&context)),
            Err(Error::Protocol(_))
        ));
        assert!(matches!(
            prove(field, &packed[1..], &x, &partials, &mut Fs::new(&context)),
            Err(Error::Shape(_))
        ));
    }

    #[test]
    fn fitted_rounds_pass_only_with_another_final_value() {
        // What the sumcheck guarantees by itself: with a false partial evaluation, every round
        // and the final check can be passed, but only with a final value other than f'(r),
        // which the rest of the opening must then refute (`pcs`, test
        // `partials_fitted_by_the_rounds_are_rejected`).
        let sub = subfield();
        let field = sub.field();
        let n = 5;
        let (packed, parts) = polynomial(9, n);
        let x = exts(10, n);
        let mut partials: Vec<Ext<K>> = parts.iter().map(|p| evaluate(&sub, p, &x)).collect();
        let context = [2u8; 32];
        let honest = prove(field, &packed, &x, &partials, &mut Fs::new(&context)).unwrap();
        let fitted = prove_fitted(
            field,
            &packed,
            &x,
            &partials,
            &INPUT_LABELS,
            &mut Fs::new(&context),
        );
        assert_eq!(
            fitted, honest,
            "with true partial evaluations nothing is fitted"
        );
        partials[1] = field.add(&partials[1], &field.one());
        let (rounds, point, value) = prove_fitted(
            field,
            &packed,
            &x,
            &partials,
            &INPUT_LABELS,
            &mut Fs::new(&context),
        );
        let got = verify(
            field,
            &x,
            &partials,
            &rounds,
            &value,
            &INPUT_LABELS,
            &mut Fs::new(&context),
        )
        .unwrap();
        assert_eq!(got, point);
        assert_ne!(value, evaluate(&sub, &packed, &point));
    }
}
