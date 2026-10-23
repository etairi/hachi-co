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
//! **What is proved instead.** Let $\varphi : a \mapsto a^q$ be the Frobenius
//! ([`Fqk::frobenius`]), of order $K$, applied to points coordinatewise. It fixes $\mathbb F_q$,
//! hence the coefficients of the $f_i$, so
//! $$f'(\varphi^{-t}(\mathbf x')) = \sum_i Z_i f_i(\varphi^{-t}(\mathbf x')) = \sum_i Z_i
//! \varphi^{-t}(f_i(\mathbf x')).$$
//! The prover proves the $K$ evaluations $f'(\mathbf p_t) = v_t$, $t = 0, \dots, K - 1$, at the
//! points $\mathbf p_t = \varphi^{-t}(\mathbf x')$ ([`conjugate_points`]), with $v_t = \sum_i
//! Z_i \varphi^{-t}(y_i)$ ([`conjugate_values`]), which the verifier computes from the partial
//! evaluations. The case $t = 0$ is the paper's claim.
//!
//! They hold only for the true partial evaluations. Let $\delta_i = y_i - f_i(\mathbf x')$, so
//! that $\sum_i Z_i \varphi^{-t}(\delta_i) = 0$ for every $t$. Apply $\varphi^t$, multiply by
//! $\varphi^t(c)$ for some $c \in \mathbb F_{q^K}$ and add over $t$: $\sum_i
//! \mathrm{Tr}(c Z_i) \delta_i = 0$, where $\mathrm{Tr} = \sum_t \varphi^t$ is the trace to
//! $\mathbb F_q$. The trace form of a finite field is non-degenerate, so $(Z_i)\_i$ has a dual
//! basis $(Z_j^\vee)\_j$, with $\mathrm{Tr}(Z_j^\vee Z_i) = 1$ if $i = j$ and 0 otherwise, and
//! $c = Z_j^\vee$ gives $\delta_j = 0$. (Equivalently, the Moore matrix $(\varphi^t(Z_i))\_{t,
//! i}$ is invertible.) The test `the_conjugates_determine_the_partials` checks the forgery's
//! $\boldsymbol\delta$ against this.
//!
//! **Batching.** After the partial evaluations, the verifier draws $\gamma \in \mathbb F_{q^K}$.
//! Let $n = \ell - \kappa$. As $f'$ is multilinear, $f'(\mathbf p) = \sum_{\mathbf b} T(\mathbf
//! b) \widetilde{eq}(\mathbf b, \mathbf p)$ over $\mathbf b \in \lbrace 0, 1 \rbrace^n$, with
//! $T$ the values of $f'$ on the hypercube ([`hypercube_values`]). So the $K$ claims batch into
//! $$V = \sum_t \gamma^t v_t = \sum_{\mathbf b} T(\mathbf b) W(\mathbf b), \qquad W(\mathbf b)
//! = \sum_t \gamma^t \widetilde{eq}(\mathbf b, \mathbf p_t).$$
//! A sumcheck of $n$ rounds ([`prove`], [`verify`]) reduces $V$ to a claim on
//! $T(\mathbf r) W(\mathbf r)$ at the point $\mathbf r$ of its challenges. Each round sends its
//! polynomial, of degree 2, at $0, 1, 2$; the verifier checks that the values at 0 and 1 add up
//! to the running claim, absorbs the three and draws the round's challenge. The prover then
//! sends $f'(\mathbf r) = T(\mathbf r)$, and the verifier checks $f'(\mathbf r) \sum_t
//! \gamma^t \widetilde{eq}(\mathbf r, \mathbf p_t)$ against the last claim. What is left is the
//! evaluation $f'(\mathbf r)$, which the paper's Section 3.1 reduces to $\mathcal R_q$ as it
//! did $f'(\mathbf x')$ (`pcs`).
//!
//! **Soundness** (interactive bounds; the opening applies Fiat-Shamir to them as to the rest of
//! the protocol). Let $f'$ be the polynomial the commitment binds. If a partial evaluation is
//! false, $\sum_t \gamma^t (f'(\mathbf p_t) - v_t)$ is a non-zero polynomial in $\gamma$ of
//! degree at most $K - 1$, which vanishes with probability at most $(K - 1)/q^K$. Otherwise the
//! claim $V$ is false, and by the standard analysis of the sumcheck (Lund, Fortnow, Karloff and
//! Nisan) the verifier ends on a true last claim with probability at most $2n/q^K$: two per
//! round, for round polynomials of degree 2.
//!
//! **Cost.** The proof gains $3n + 1$ field elements. The prover adds $O(K 2^n)$ field
//! operations for $W$ and $O(2^n)$ for $T$ and the rounds, small next to the iterations; the
//! verifier adds $O(Kn)$ for the conjugates and $\widetilde{eq}(\mathbf r, \mathbf p_t)$.
//!
//! **Bit order.** As in [`crate::mle`]: a table index has the first variable as its most
//! significant bit, the monomial coefficients of [`hypercube_values`] are ordered as those of
//! `Subfield::monomials`, and the rounds bind the first variable first.

use crate::error::VerifyError;
use crate::fs::Fs;
use crate::iteration::basis;
use crate::mle::{eq_eval, eq_table, interpolate, powers};
use crate::{Error, Result};
use hachi_co_math::{Ext, Fqk};

/// The transcript labels of one run of the batching sumcheck. The same gadget binds the partial
/// evaluations of a multilinear input ([`INPUT_LABELS`], `pcs`) and of every iteration's next
/// claim (`iteration`); each caller passes its own labels so the two runs never share a frame.
pub struct Labels {
    /// The challenge $\gamma$ that batches the $K$ conjugate claims.
    pub gamma: &'static [u8],
    /// A round's polynomial at $0, 1, 2$.
    pub round_poly: &'static [u8],
    /// A round's challenge.
    pub challenge: &'static [u8],
    /// The final evaluation $f'(\mathbf r)$.
    pub eval: &'static [u8],
}

/// The labels of the input reduction's batching sumcheck (`pcs`).
pub const INPUT_LABELS: Labels = Labels {
    gamma: b"hachi/input/gamma",
    round_poly: b"hachi/input/h",
    challenge: b"hachi/input/round",
    eval: b"hachi/input/final",
};

/// One round of the batching sumcheck: its polynomial at $0, 1, 2$.
pub type RoundValues<const K: usize> = [Ext<K>; 3];

/// $\varphi^{-t}(x) = \varphi^{K - t}(x)$ for $0 \le t < K$, by $K - t$ Frobenius maps.
fn conjugate<const K: usize>(field: &Fqk<K>, x: &Ext<K>, t: usize) -> Ext<K> {
    (0..(K - t % K) % K).fold(*x, |y, _| field.frobenius(&y))
}

/// The points $\mathbf p_t = \varphi^{-t}(\mathbf x')$, $t = 0, \dots, K - 1$, of the point
/// `x` $= \mathbf x'$ (module documentation).
pub fn conjugate_points<const K: usize>(field: &Fqk<K>, x: &[Ext<K>]) -> Vec<Vec<Ext<K>>> {
    (0..K)
        .map(|t| x.iter().map(|xj| conjugate(field, xj, t)).collect())
        .collect()
}

/// The values $v_t = \sum_i Z_i \varphi^{-t}(y_i)$, $t = 0, \dots, K - 1$, of the partial
/// evaluations `partials` $= (y_i)\_i$ (module documentation).
pub fn conjugate_values<const K: usize>(field: &Fqk<K>, partials: &[Ext<K>]) -> Vec<Ext<K>> {
    (0..K)
        .map(|t| {
            partials
                .iter()
                .enumerate()
                .fold(field.zero(), |acc, (i, y)| {
                    field.add(&acc, &field.mul(&conjugate(field, y, t), &basis::<K>(i)))
                })
        })
        .collect()
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

/// $\sum_t \gamma^t v_t$.
fn batched<const K: usize>(field: &Fqk<K>, gammas: &[Ext<K>], values: &[Ext<K>]) -> Ext<K> {
    gammas.iter().zip(values).fold(field.zero(), |acc, (g, v)| {
        field.add(&acc, &field.mul(g, v))
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
        &hypercube_values(field, coeffs)?,
        x,
        partials,
        &INPUT_LABELS,
        fs,
    )
}

/// The prover, run after the statement and the partial evaluations were absorbed: draws
/// $\gamma$, runs the rounds on the table `t` of $f'$ on $\lbrace 0, 1 \rbrace^n$ (hypercube
/// form, $2^n$ values), and absorbs $f'(\mathbf r)$. Returns the rounds, the point $\mathbf r$
/// and $f'(\mathbf r)$. `x` is $\mathbf x'$; `partials` serve only to check the honest prover's
/// invariant, the first round's sum $V$; `labels` frame the transcript.
///
/// # Errors
/// [`Error::Shape`] unless `t` has $2^{|\mathbf x'|}$ values and there are $K$ partial
/// evaluations; [`Error::Protocol`] if the partial evaluations are not those of $f'$ at
/// $\mathbf x'$; sampling errors.
pub fn prove_table<const K: usize>(
    field: &Fqk<K>,
    t: &[Ext<K>],
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
    let gamma = fs.field_coins::<K>(labels.gamma, field.modulus(), 1)?[0];
    let gammas = powers(field, &gamma, K);
    let mut t = t.to_vec();
    let mut w = vec![field.zero(); t.len()];
    for (g, p) in gammas.iter().zip(conjugate_points(field, x)) {
        for (wb, e) in w.iter_mut().zip(eq_table(field, &p)) {
            *wb = field.add(wb, &field.mul(g, &e));
        }
    }
    let sum = batched(field, &gammas, &conjugate_values(field, partials));
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
        fs.exts(labels.round_poly, &h);
        let r = fs.field_coins::<K>(labels.challenge, field.modulus(), 1)?[0];
        for table in [&mut t, &mut w] {
            for i in 0..half {
                let (lo, hi) = (table[i], table[i + half]);
                table[i] = field.add(&lo, &field.mul(&r, &field.sub(&hi, &lo)));
            }
            table.truncate(half);
        }
        rounds.push(h);
        point.push(r);
    }
    let value = t[0];
    fs.exts(labels.eval, &[value]);
    Ok((rounds, point, value))
}

/// The verifier, run after the statement and the partial evaluations were absorbed: draws
/// $\gamma$, checks the rounds against $V = \sum_t \gamma^t v_t$ and $f'(\mathbf r)$ = `value`
/// against the last claim, absorbing as [`prove_table`] does under the same `labels`. Returns
/// $\mathbf r$, at which the rest of the opening must prove $f'(\mathbf r)$ = `value`. `x` is
/// $\mathbf x'$.
///
/// # Errors
/// [`VerifyError::Malformed`] unless there are $|\mathbf x'|$ rounds and $K$ partial
/// evaluations; [`VerifyError::Check`] if a round's values at 0 and 1 do not add up to the
/// running claim, or if the last claim is not $f'(\mathbf r) \sum_t \gamma^t
/// \widetilde{eq}(\mathbf r, \mathbf p_t)$.
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
    let gamma = fs.field_coins::<K>(labels.gamma, field.modulus(), 1)?[0];
    let gammas = powers(field, &gamma, K);
    let mut claim = batched(field, &gammas, &conjugate_values(field, partials));
    let mut point = Vec::with_capacity(rounds.len());
    for (i, h) in rounds.iter().enumerate() {
        if field.add(&h[0], &h[1]) != claim {
            return Err(VerifyError::Check(format!(
                "batching sumcheck, round {}",
                i + 1
            )));
        }
        fs.exts(labels.round_poly, h);
        let r = fs.field_coins::<K>(labels.challenge, field.modulus(), 1)?[0];
        claim = interpolate(field, h, &r);
        point.push(r);
    }
    let weight = gammas
        .iter()
        .zip(conjugate_points(field, x))
        .fold(field.zero(), |acc, (g, p)| {
            field.add(&acc, &field.mul(g, &eq_eval(field, &point, &p)))
        });
    if field.mul(value, &weight) != claim {
        return Err(VerifyError::Check(
            "the batching sumcheck's final value".into(),
        ));
    }
    fs.exts(labels.eval, core::slice::from_ref(value));
    Ok(point)
}

/// A dishonest prover, for the tests: the rounds of [`prove`] on the true tables, each with its
/// value at 0 replaced so that it adds up to the running claim that starts from the batched
/// value of `partials`, and a final value fitted to the last claim. It passes [`verify`]
/// whatever the partial evaluations, with a final value that is not $f'(\mathbf r)$ when they
/// are false: the rest of the opening must reject it.
#[cfg(test)]
pub(crate) fn prove_fitted<const K: usize>(
    field: &Fqk<K>,
    coeffs: &[Ext<K>],
    x: &[Ext<K>],
    partials: &[Ext<K>],
    labels: &Labels,
    fs: &mut Fs,
) -> (Vec<RoundValues<K>>, Vec<Ext<K>>, Ext<K>) {
    let gamma = fs
        .field_coins::<K>(labels.gamma, field.modulus(), 1)
        .unwrap()[0];
    let gammas = powers(field, &gamma, K);
    let mut t = hypercube_values(field, coeffs).unwrap();
    let mut w = vec![field.zero(); t.len()];
    for (g, p) in gammas.iter().zip(conjugate_points(field, x)) {
        for (wb, e) in w.iter_mut().zip(eq_table(field, &p)) {
            *wb = field.add(wb, &field.mul(g, &e));
        }
    }
    let mut claim = batched(field, &gammas, &conjugate_values(field, partials));
    let (mut rounds, mut point) = (Vec::new(), Vec::new());
    for _ in 0..x.len() {
        let half = t.len() / 2;
        let mut h = [field.zero(); 3];
        for i in 0..half {
            let (t0, t1, w0, w1) = (t[i], t[i + half], w[i], w[i + half]);
            let t2 = field.sub(&field.add(&t1, &t1), &t0);
            let w2 = field.sub(&field.add(&w1, &w1), &w0);
            h[1] = field.add(&h[1], &field.mul(&t1, &w1));
            h[2] = field.add(&h[2], &field.mul(&t2, &w2));
        }
        h[0] = field.sub(&claim, &h[1]);
        fs.exts(labels.round_poly, &h);
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
        claim = interpolate(field, &h, &r);
        rounds.push(h);
        point.push(r);
    }
    let value = field.mul(&claim, &field.inv(&w[0]).unwrap());
    fs.exts(labels.eval, &[value]);
    (rounds, point, value)
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn the_conjugate_points_carry_the_conjugate_values() {
        // f'(phi^{-t}(x')) = sum_i Z_i phi^{-t}(f_i(x')) for the true partial evaluations,
        // with f' evaluated in monomial form, as the multilinear extension of its hypercube
        // values (mle_eval) and as sum_b T(b) eq(b, p_t): this fixes the direction of the
        // Frobenius and the bit order of the tables.
        let sub = subfield();
        let field = sub.field();
        for n in [1usize, 2, 5] {
            let (packed, parts) = polynomial(n as u64, n);
            let x = exts(100 + n as u64, n);
            let partials: Vec<Ext<K>> = parts.iter().map(|p| evaluate(&sub, p, &x)).collect();
            let table = hypercube_values(field, &packed).unwrap();
            let values = conjugate_values(field, &partials);
            for (t, p) in conjugate_points(field, &x).iter().enumerate() {
                let direct = evaluate(&sub, &packed, p);
                assert_eq!(direct, values[t], "n {n}, t {t}");
                assert_eq!(mle_eval(field, &table, p), direct, "n {n}, t {t}");
                let by_eq = table
                    .iter()
                    .zip(eq_table(field, p))
                    .fold(field.zero(), |acc, (v, e)| {
                        field.add(&acc, &field.mul(v, &e))
                    });
                assert_eq!(by_eq, direct, "n {n}, t {t}");
            }
            // phi(x') carries v_{K-1}, not v_1: the direction matters.
            let forward: Vec<Ext<K>> = x.iter().map(|xj| field.frobenius(xj)).collect();
            assert_eq!(evaluate(&sub, &packed, &forward), values[K - 1], "n {n}");
            assert_ne!(evaluate(&sub, &packed, &forward), values[1], "n {n}");
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
    fn the_conjugates_determine_the_partials() {
        // A delta in the kernel of the paper's check, sum_i delta_i Z_i = 0, moves some v_t
        // with t > 0; and for random deltas the K values (sum_i Z_i phi^{-t}(delta_i))_t vanish
        // only for delta = 0 (the Moore matrix is invertible).
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
        let moved = conjugate_values(field, &delta);
        assert!(field.is_zero(&moved[0]));
        assert!(moved[1..].iter().any(|v| !field.is_zero(v)));
        for seed in 0..50 {
            let delta = exts(1000 + seed, K);
            let values = conjugate_values(field, &delta);
            assert!(values.iter().any(|v| !field.is_zero(v)), "seed {seed}");
        }
        assert!(
            conjugate_values(field, &[field.zero(); K])
                .iter()
                .all(|v| field.is_zero(v))
        );
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
        // Every value of every round, and the final value.
        for i in 0..n {
            for j in 0..3 {
                let mut bad = rounds.clone();
                bad[i][j] = field.add(&bad[i][j], &field.one());
                assert!(
                    matches!(verdict(&partials, &bad, &value), Err(VerifyError::Check(_))),
                    "round {i}, value {j}"
                );
            }
        }
        let other = field.add(&value, &field.one());
        assert!(matches!(
            verdict(&partials, &rounds, &other),
            Err(VerifyError::Check(_))
        ));
        // Another partial evaluation, a missing round, a missing partial evaluation.
        let mut moved = partials.clone();
        moved[2] = field.add(&moved[2], &field.one());
        assert!(matches!(
            verdict(&moved, &rounds, &value),
            Err(VerifyError::Check(m)) if m.contains("round 1")
        ));
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
