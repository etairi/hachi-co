//! Balanced gadget decompositions $\mathbf G^{-1}\_{b,n}$ and $\mathbf J^{-1}\_n$ and their gadget
//! matrices (draft `4_preliminaries.tex`:94-100).
//!
//! **Digits.** Both decompositions output balanced digits in
//! $[\lceil -b/2 \rceil, \lceil b/2 \rceil - 1]$, the digit set of $S_b$
//! (`4_preliminaries.tex`:88). With $k$ digits and $S_k = (b^k - 1)/(b - 1)$, the representable
//! integers are exactly the $b^k$ integers of the window
//! $[\lceil -b/2 \rceil S_k, (\lceil b/2 \rceil - 1) S_k]$, each with a unique digit vector, and
//! the digits are found from the least significant up: the digit is the residue of the running
//! value modulo $b$, moved into the digit set, and the running value is then reduced by it and
//! divided by $b$.
//!
//! **Layout.** $\mathbf G_{b,n} = \mathbf I_n \otimes (1, b, \dots, b^{\delta-1})$ is
//! element-major: the $\delta$ digits of entry $i$ are at positions $i\delta, \dots,
//! i\delta + \delta - 1$, least significant first (`docs/m0-spec.md`, "Digit sets"); the same
//! holds for $\mathbf J_n$ with $\tau$ digits.
//!
//! **Choices where the draft is silent** (see each function): the representative that
//! $\mathbf G^{-1}$ decomposes (the window holds two representatives of some residues when
//! $b^\delta > q$), the window's negative end for odd $b$, and $\tau = 0$ or $\beta = 0$.

use crate::{
    error::{Error, Result},
    ring::{Poly, Ring},
    zq::Zq,
};

/// The digit set $[\lceil -b/2 \rceil, \lceil b/2 \rceil - 1]$ of base $b \ge 2$, as
/// $(-\lfloor b/2 \rfloor, \lfloor (b-1)/2 \rfloor)$ (draft `4_preliminaries.tex`:88, 96).
pub fn digit_range(b: u64) -> (i128, i128) {
    let b = i128::from(b);
    (-(b / 2), (b - 1) / 2)
}

/// $S_k = 1 + b + \dots + b^{k-1}$, or `None` if it exceeds `u128`.
fn digit_sum(b: u64, k: usize) -> Option<u128> {
    let mut s = 0u128;
    for _ in 0..k {
        s = s.checked_mul(u128::from(b))?.checked_add(1)?;
    }
    Some(s)
}

/// $b^k$, or `None` if it exceeds `u128`.
fn power(b: u64, k: usize) -> Option<u128> {
    let mut p = 1u128;
    for _ in 0..k {
        p = p.checked_mul(u128::from(b))?;
    }
    Some(p)
}

/// The window $[\lceil -b/2 \rceil S_k, (\lceil b/2 \rceil - 1) S_k]$,
/// $S_k = (b^k - 1)/(b - 1)$, of the integers that $k$ balanced base-$b$ digits represent, with
/// both ends saturated at the `i128` range.
///
/// This is the window of `docs/m0-spec.md` ("Digit sets"), which writes its negative end as
/// $-\lceil b/2 \rceil S_k$: for even $b$ that is the same number, and for odd $b$ the
/// representable negative end is $\lceil -b/2 \rceil S_k = -(\lceil b/2 \rceil - 1) S_k$, as the
/// draft states (`6_Collaborative_Hachi.tex`:755-761: "the negative endpoint has at least the
/// same magnitude").
pub fn representable_range(b: u64, k: usize) -> (i128, i128) {
    let (lo, hi) = digit_range(b);
    match digit_sum(b, k).and_then(|s| i128::try_from(s).ok()) {
        // Exact, or saturated when the exact end is beyond the i128 range.
        Some(s) => (lo.saturating_mul(s), hi.saturating_mul(s)),
        None => (
            if lo == 0 { 0 } else { i128::MIN },
            if hi == 0 { 0 } else { i128::MAX },
        ),
    }
}

/// $\delta = \lceil \log_b q \rceil$, the least $\delta$ with $b^\delta \ge q$ (draft
/// `4_preliminaries.tex`:61, 96).
///
/// # Errors
/// [`Error::Parameter`] for $q < 2$ or $b < 2$.
pub fn num_digits(q: u64, b: u64) -> Result<usize> {
    if q < 2 || b < 2 {
        return Err(Error::Parameter("num_digits needs q >= 2 and b >= 2"));
    }
    let mut delta = 0;
    while power(b, delta).is_some_and(|p| p < u128::from(q)) {
        delta += 1;
    }
    Ok(delta)
}

/// The tight $\tau$: the least $\tau$ with $(\lceil b/2 \rceil - 1)(b^\tau - 1)/(b - 1) \ge \beta$,
/// so that every integer of $[-\beta, \beta]$ has $\tau$ balanced digits (`docs/m0-spec.md`,
/// "τ rule", the user's decision of 2026-10-17). For $\beta = 0$ this is $\tau = 0$.
///
/// # Errors
/// [`Error::Parameter`] for $b < 2$, and for $b = 2$ with $\beta > 0$ (the digit set
/// $\lbrace -1, 0 \rbrace$ has no positive digit).
pub fn tight_tau(beta: u128, b: u64) -> Result<usize> {
    if b < 2 {
        return Err(Error::Parameter("tight_tau needs b >= 2"));
    }
    let (_, hi) = digit_range(b);
    if hi == 0 && beta > 0 {
        return Err(Error::Parameter("no tau represents beta > 0 in base 2"));
    }
    let hi = hi as u128;
    let mut tau = 0;
    loop {
        match digit_sum(b, tau).and_then(|s| s.checked_mul(hi)) {
            Some(top) if top < beta => tau += 1,
            _ => return Ok(tau),
        }
    }
}

/// The draft's $\tau = \lceil \log_b \beta \rceil + 1$ (`6_Collaborative_Hachi.tex`:276,
/// `4_preliminaries.tex`:65), kept as an option (`docs/m0-spec.md`, "τ rule").
///
/// # Errors
/// [`Error::Parameter`] for $b < 2$ or $\beta = 0$ (where $\log_b \beta$ is undefined).
pub fn draft_tau(beta: u128, b: u64) -> Result<usize> {
    if b < 2 || beta == 0 {
        return Err(Error::Parameter("draft_tau needs b >= 2 and beta >= 1"));
    }
    let mut t = 0;
    while power(b, t).is_some_and(|p| p < beta) {
        t += 1;
    }
    Ok(t + 1)
}

/// Writes the `out.len()` balanced base-$b$ digits of $z$, least significant first, and returns
/// what is left, $(z - \sum_i a_i b^i)/b^{k}$: zero exactly when $z$ is representable.
fn balanced_digits(mut z: i128, b: u64, out: &mut [i64]) -> i128 {
    let b = i128::from(b);
    let (_, hi) = digit_range(b as u64);
    for digit in out.iter_mut() {
        let r = z.rem_euclid(b);
        let a = if r > hi { r - b } else { r };
        *digit = a as i64;
        z = (z - a) / b;
    }
    z
}

/// $b^\delta \ge q$: the window of $\delta$ digits holds a representative of every residue.
fn check_window(zq: &Zq, b: u64, delta: usize) -> Result<()> {
    if power(b, delta).is_some_and(|p| p < u128::from(zq.modulus())) {
        return Err(Error::Parameter("G^-1 needs b^delta >= q"));
    }
    Ok(())
}

fn check_digits(b: u64, k: usize, out: &[i64]) -> Result<()> {
    if b < 2 {
        return Err(Error::Parameter("decomposition base must be at least 2"));
    }
    if k == 0 {
        return Err(Error::Parameter("decomposition needs at least one digit"));
    }
    if out.len() != k {
        return Err(Error::Length {
            expected: k,
            actual: out.len(),
        });
    }
    Ok(())
}

/// $\mathbf G^{-1}$ of one residue: the $\delta$ balanced digits (least significant first, in
/// `out`) of the representative of $x$ (read modulo $q$) in the window of
/// [`representable_range`]`(b, δ)`, so that $\sum_i a_i b^i \equiv x \pmod q$ (draft
/// `4_preliminaries.tex`:96).
///
/// The window has $b^\delta \ge q$ elements, so it holds a representative of every residue, and
/// two of some residues when $b^\delta > q$ (at the toy presets $b^\delta \approx 2q$).
/// **Choice** (the draft is silent; the first version of `docs/m0-spec.md` called the
/// representative unique, which holds only for $b^\delta = q$, and the main session then fixed
/// this rule): the representative is $x$ itself if $x \le (\lceil b/2 \rceil - 1) S_\delta$ and
/// $x - q$ otherwise. It is the rule of the legacy crate (`crates/hachi/src/arithmetic/utils.rs`,
/// `to_window`) generalized to any base, and of the Sage oracle (`ref/hachico_ref/ring.py`), whose
/// recorded digits `tests/oracle.rs` reproduces.
///
/// # Errors
/// [`Error::Parameter`] for $b < 2$, $\delta = 0$ or $b^\delta < q$; [`Error::Length`] unless
/// `out` has $\delta$ entries.
pub fn g_inv_scalar(zq: &Zq, b: u64, delta: usize, x: u64, out: &mut [i64]) -> Result<()> {
    check_digits(b, delta, out)?;
    check_window(zq, b, delta)?;
    let q = zq.modulus();
    let x = i128::from(zq.reduce(x));
    let (_, top) = representable_range(b, delta);
    let representative = if x <= top { x } else { x - i128::from(q) };
    let rest = balanced_digits(representative, b, out);
    debug_assert_eq!(rest, 0, "the window holds the representative");
    Ok(())
}

/// $\mathbf J^{-1}$ of one short integer: the $\tau$ balanced digits of $z$ itself (least
/// significant first, in `out`), with $\sum_i a_i b^i = z$ over $\mathbb Z$ (draft
/// `4_preliminaries.tex`:100).
///
/// # Errors
/// [`Error::NotRepresentable`] if $z$ is outside [`representable_range`]`(b, τ)`;
/// [`Error::Parameter`] for $b < 2$ or $\tau = 0$; [`Error::Length`] unless `out` has $\tau$
/// entries.
pub fn j_inv_scalar(b: u64, tau: usize, z: i64, out: &mut [i64]) -> Result<()> {
    check_digits(b, tau, out)?;
    let (min, max) = representable_range(b, tau);
    let z = i128::from(z);
    if z < min || z > max {
        return Err(Error::NotRepresentable { value: z, min, max });
    }
    let rest = balanced_digits(z, b, out);
    debug_assert_eq!(rest, 0, "the window is the representable set");
    Ok(())
}

/// $\mathbf G^{-1}\_{b,n}(\mathbf t)$ for $\mathbf t \in \mathcal R_q^n$: every coefficient of
/// $t_i$ decomposed by [`g_inv_scalar`], digit $k$ of entry $i$ in output entry $i\delta + k$,
/// digits stored as residues modulo $q$ (draft `4_preliminaries.tex`:96).
///
/// # Errors
/// As [`g_inv_scalar`], and [`Error::Length`] for an entry without $d$ coefficients.
pub fn g_inv(ring: &Ring, b: u64, delta: usize, t: &[Poly]) -> Result<Vec<Poly>> {
    check_window(ring.zq(), b, delta)?;
    decompose(ring, b, delta, t, |x, out| {
        g_inv_scalar(ring.zq(), b, delta, x, out)
    })
}

/// $\mathbf J^{-1}\_n(\mathbf z)$ for $\mathbf z \in \mathcal R_q^n$ read as centered integers:
/// every coefficient decomposed by [`j_inv_scalar`] with $\tau$ digits, in the layout of
/// [`g_inv`], digits stored as residues modulo $q$ (draft `4_preliminaries.tex`:100).
///
/// # Errors
/// As [`j_inv_scalar`] (a coefficient whose centered value is not representable), and
/// [`Error::Length`] for an entry without $d$ coefficients.
pub fn j_inv(ring: &Ring, b: u64, tau: usize, z: &[Poly]) -> Result<Vec<Poly>> {
    decompose(ring, b, tau, z, |x, out| {
        j_inv_scalar(b, tau, ring.zq().to_centered(x), out)
    })
}

fn decompose(
    ring: &Ring,
    b: u64,
    k: usize,
    input: &[Poly],
    digits_of: impl Fn(u64, &mut [i64]) -> Result<()>,
) -> Result<Vec<Poly>> {
    let d = ring.degree();
    let mut digits = vec![0i64; k];
    check_digits(b, k, &digits)?;
    let mut out = vec![ring.zero(); input.len() * k];
    for (i, entry) in input.iter().enumerate() {
        if entry.0.len() != d {
            return Err(Error::Length {
                expected: d,
                actual: entry.0.len(),
            });
        }
        for (j, x) in entry.0.iter().enumerate() {
            digits_of(*x, &mut digits)?;
            for (l, digit) in digits.iter().enumerate() {
                out[i * k + l].0[j] = ring.zq().from_i64(*digit);
            }
        }
    }
    Ok(out)
}

/// $\mathbf G_{b,n} \mathbf s$: entry $i$ is $\sum_{k<\delta} b^k s_{i\delta + k}$ in
/// $\mathcal R_q$, for $\mathbf s \in \mathcal R_q^{n\delta}$ (draft `4_preliminaries.tex`:96).
///
/// # Errors
/// [`Error::Parameter`] for $b < 2$, $\delta = 0$ or a length that is not a multiple of
/// $\delta$; [`Error::Length`] for an entry without $d$ coefficients.
pub fn g_apply(ring: &Ring, b: u64, delta: usize, s: &[Poly]) -> Result<Vec<Poly>> {
    gadget(ring, b, delta, s)
}

/// $\mathbf J_n \mathbf s$: entry $i$ is $\sum_{k<\tau} b^k s_{i\tau + k}$ in $\mathcal R_q$
/// (draft `4_preliminaries.tex`:100).
///
/// # Errors
/// As [`g_apply`].
pub fn j_apply(ring: &Ring, b: u64, tau: usize, s: &[Poly]) -> Result<Vec<Poly>> {
    gadget(ring, b, tau, s)
}

fn gadget(ring: &Ring, b: u64, k: usize, s: &[Poly]) -> Result<Vec<Poly>> {
    if b < 2 || k == 0 {
        return Err(Error::Parameter(
            "gadget needs b >= 2 and at least one digit",
        ));
    }
    if !s.len().is_multiple_of(k) {
        return Err(Error::Parameter(
            "input length is not a multiple of the digit count",
        ));
    }
    let zq = ring.zq();
    let d = ring.degree();
    for entry in s {
        if entry.0.len() != d {
            return Err(Error::Length {
                expected: d,
                actual: entry.0.len(),
            });
        }
    }
    let b = zq.reduce(b);
    Ok(s.chunks(k)
        .map(|digits| {
            // Horner's rule from the most significant digit.
            Poly(
                (0..d)
                    .map(|j| {
                        digits
                            .iter()
                            .rev()
                            .fold(0, |acc, digit| zq.add(zq.mul(acc, b), digit.0[j]))
                    })
                    .collect(),
            )
        })
        .collect())
}
