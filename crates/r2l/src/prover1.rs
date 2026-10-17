//! Prover 1's steps after the gather (`5_Building_Blocks.tex`:129-133): the sum
//! $\mathbf z = \sum\_p \mathbf z\_p$, the check $\Vert \mathbf z \Vert\_\infty \le \beta$ and
//! $\hat{\mathbf z} = \mathbf J\_{\delta 2^m}^{-1}(\mathbf z)$.
//!
//! $\mathbf z$ is prover 1's private witness (plan `docs/hachi-co-plan.md`, finding F5); these
//! steps are not constant time (the norm and $\mathbf J^{-1}$ branch on the values).

use crate::local::{check_len, response_len};
use crate::{Error, Result};
use hachi_co_core::{Context, short};
use hachi_co_math::{Poly, decompose};

/// $\mathbf z = \sum\_{p=1}^{n} \mathbf z\_p$ over the integers (`5_Building_Blocks.tex`:130),
/// from the responses gathered at prover 1 (`responses[p - 1]` from prover $p$, flat, $N$ entries
/// each). The responses are messages from other parties: their number and lengths are checked.
///
/// # Errors
/// [`Error::Shape`] unless there are $n$ responses of $N$ entries; [`Error::Overflow`] if a sum
/// leaves `i64` (only for responses far outside $\[-\gamma, \gamma\]$).
pub fn aggregate<const K: usize>(ctx: &Context<K>, responses: &[Vec<i64>]) -> Result<Vec<i64>> {
    let n = response_len(ctx);
    check_len(
        "responses z_p (one per prover)",
        ctx.layout.n,
        responses.len(),
    )?;
    let mut z = vec![0i64; n];
    for response in responses {
        check_len("response z_p (N = d delta 2^m)", n, response.len())?;
        for (zi, x) in z.iter_mut().zip(response) {
            *zi = zi
                .checked_add(*x)
                .ok_or(Error::Overflow("z = sum of the responses"))?;
        }
    }
    Ok(z)
}

/// Prover 1's check $\Vert \mathbf z \Vert\_\infty \le \beta = n\gamma$
/// (`5_Building_Blocks.tex`:131; $\beta$ of `6_Collaborative_Hachi.tex`:275). Honest responses
/// satisfy $\Vert \mathbf z\_p \Vert\_\infty \le \gamma - \beta\_1$, so the check never fails for
/// honest provers (line 193).
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
/// (`5_Building_Blocks.tex`:132-133) of the integer vector $\mathbf z$ (`ref/README.md`, choice
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
