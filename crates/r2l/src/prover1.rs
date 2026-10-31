//! Prover 1's steps after $\mathcal F\_{\mathsf{AH}}(1^n)$ (`5_Building_Blocks.tex`:142-146): the
//! sum $\mathbf z = \sum\_p \mathbf z\_p$ as integers, the check
//! $\Vert \mathbf z \Vert\_\infty \le \beta$ and $\hat{\mathbf z} =
//! \mathbf J\_{\delta 2^m}^{-1}(\mathbf z)$.
//!
//! Every prover receives $\mathbf z$ from $\mathcal F\_{\mathsf{AH}}(1^n)$; it never reaches the
//! verifier. These steps are not constant time (the norm and $\mathbf J^{-1}$ branch on the
//! values).

use crate::local::{check_len, response_len};
use crate::{Error, Result};
use hachi_co_core::{Context, short};
use hachi_co_math::{Poly, decompose};

/// $\mathbf z = \sum\_{p=1}^{n} \mathbf z\_p$ as integers (`5_Building_Blocks.tex`:143), from the
/// $N$ residues modulo $q$ that $\mathcal F\_{\mathsf{AH}}(1^n)$ returned: their centered
/// representatives. For honest provers this is the integer sum, as $\Vert \mathbf z
/// \Vert\_\infty \le \beta$ and $2\beta \lt q$ (from $q \gt 4\bar\omega\bar\beta$,
/// `6_Collaborative_Hachi.tex`:274).
///
/// # Errors
/// [`Error::Shape`] unless there are $N$ residues; [`Error::NonCanonical`] for a residue $\ge q$.
pub fn z_of_sum<const K: usize>(ctx: &Context<K>, sum: &[u64]) -> Result<Vec<i64>> {
    check_len(
        "output of F_AH(1^n) (N = d delta 2^m)",
        response_len(ctx),
        sum.len(),
    )?;
    let (zq, q) = (ctx.ring.zq(), ctx.q());
    if let Some(value) = sum.iter().copied().find(|x| *x >= q) {
        return Err(Error::NonCanonical {
            what: "output of F_AH(1^n)",
            value,
            q,
        });
    }
    Ok(sum.iter().map(|x| zq.to_centered(*x)).collect())
}

/// Prover 1's check $\Vert \mathbf z \Vert\_\infty \le \beta = n\gamma$
/// (`5_Building_Blocks.tex`:144; $\beta$ of `6_Collaborative_Hachi.tex`:279). Honest responses
/// satisfy $\Vert \mathbf z\_p \Vert\_\infty \le \gamma - \beta\_1$, so the check never fails for
/// honest provers (line 207).
///
/// # Errors
/// [`Error::NormCheck`] with the norm if it exceeds $\beta$.
pub fn norm_check<const K: usize>(ctx: &Context<K>, z: &[i64]) -> Result<()> {
    let (norm, beta) = (short::inf_norm(z), ctx.beta());
    if norm > beta {
        return Err(Error::NormCheck { norm, beta });
    }
    Ok(())
}

/// $\hat{\mathbf z} = \mathbf J\_{\delta 2^m}^{-1}(\mathbf z) \in \mathcal R\_q^{\delta 2^m \tau}$
/// (`5_Building_Blocks.tex`:145-146) of the integer vector $\mathbf z$ (`ref/README.md`, choice
/// 17: $\mathbf J^{-1}$ decomposes the integers themselves, not residues): the $\tau$ balanced
/// digits of coefficient $j$ of entry $k$ ([`decompose::j_inv_scalar`]) are coefficient $j$ of
/// entries $k\tau, \dots, k\tau + \tau - 1$, stored as residues (the element-major layout of
/// `ref/README.md`, "z′ layout": entry $(j\delta + e)\tau + \theta$ holds digit $\theta$ of
/// $z\_{j\delta + e}$). With the tight $\tau$ every $\mathbf z$ with
/// $\Vert \mathbf z \Vert\_\infty \le \beta$ is representable.
///
/// # Errors
/// [`Error::Shape`] unless $\mathbf z$ has $N$ entries; [`Error::Math`]
/// ([`NotRepresentable`](hachi_co_math::Error::NotRepresentable)) for a coefficient outside the
/// window of $\tau$ digits.
pub fn z_hat<const K: usize>(ctx: &Context<K>, z: &[i64]) -> Result<Vec<Poly>> {
    let l = &ctx.layout;
    check_len("z (N = d delta 2^m)", response_len(ctx), z.len())?;
    let (d, tau) = (l.d, l.tau);
    let zq = ctx.ring.zq();
    let mut out = vec![ctx.ring.zero(); l.width_a * tau];
    let mut digits = vec![0i64; tau];
    for (k, entry) in z.chunks(d).enumerate() {
        for (j, x) in entry.iter().enumerate() {
            decompose::j_inv_scalar(ctx.b(), tau, *x, &mut digits)?;
            for (theta, digit) in digits.iter().enumerate() {
                out[k * tau + theta].0[j] = zq.from_i64(*digit);
            }
        }
    }
    crate::ct::wipe(&mut digits);
    Ok(out)
}
