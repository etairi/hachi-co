//! The assembly of prover $p$'s shares $\langle \mathbf z' \rangle\_p$ and
//! $\lbrace \mathbf y \rbrace\_p$ (`5_Building_Blocks.tex`:134, 154-179; `ref/README.md`,
//! "Shares").

use crate::local::{BlockW, check_len, check_polys, fill_first_message_part};
use crate::{AttemptSecrets, Error, Result};
use hachi_co_core::{Context, PartyId, PartyState, build_y};
use hachi_co_math::Poly;

/// Prover $p$'s share $\langle \mathbf z' \rangle\_p \in \mathcal R\_q^\mu$ of
/// $\mathbf z' = (\hat{\mathbf w}, \hat{\mathbf w}\_0, \hat{\mathbf t}\_0, \boldsymbol\rho,
/// \hat{\mathbf t}, \hat{\mathbf z}, u)$ (`5_Building_Blocks.tex`:164-172) for the accepted
/// attempt: the values $p$ holds at their positions ([`hachi_co_core::Layout`]), namely its
/// blocks of $\hat{\mathbf w}$ (from `w`, [`crate::owned_w`]) and of $\hat{\mathbf t}$ (from the
/// state), its own $\hat{\mathbf w}\_{0,p}$, $\hat{\mathbf t}\_{0,p}$ and
/// $\boldsymbol\rho\_p$ (from `secrets`), $\hat{\mathbf z}$ for $p = 1$ (line 134:
/// $\lbrace \hat{\mathbf z} \rbrace\_1 = \hat{\mathbf z}$, $\lbrace \hat{\mathbf z} \rbrace\_p
/// = \mathbf 0$ otherwise), and $\langle u \rangle\_p$ (from $\Pi\_{\mathsf{conv}}$) at
/// position $\mu - 1$; zero elsewhere. So the share is $\mathbf z'$ on $S\_p$
/// ([`hachi_co_core::Layout::s_ranges`]), $\langle u \rangle\_p$ at $\mu - 1$, and zero on the
/// other provers' positions and the unheld ones (`ref/README.md`, choice 6).
///
/// # Errors
/// [`Error::Core`] if the state has the wrong shape; [`Error::Shape`] unless `w` holds $p$'s
/// blocks in order, unless `z_hat` is given exactly for $p = 1$ with $\delta 2^m \tau$ entries,
/// or unless `u_share` has $d$ coefficients; [`Error::NonCanonical`] for a non-canonical residue
/// in the inputs; [`Error::OutOfRange`] for $\boldsymbol\rho\_p \notin S\_b^{m\_\rho}$.
pub fn z_prime_share<const K: usize>(
    ctx: &Context<K>,
    state: &PartyState<K>,
    w: &[BlockW],
    secrets: &AttemptSecrets,
    z_hat: Option<&[Poly]>,
    u_share: &Poly,
) -> Result<Vec<Poly>> {
    let l = &ctx.layout;
    let q = ctx.q();
    state.validate(l)?;
    let p = state.party;
    let mut share = vec![ctx.ring.zero(); l.mu];
    fill_first_message_part(ctx, p, w, secrets, &mut share[..l.t])?;
    for block in &state.blocks {
        check_polys("t_hat_i", &block.t_hat, l.n_a * l.delta, l.d, q)?;
        share[l.t_hat_range(block.index)].clone_from_slice(&block.t_hat);
    }
    match (p, z_hat) {
        (1, Some(z_hat)) => {
            check_polys("z_hat", z_hat, l.width_a * l.tau, l.d, q)?;
            share[l.z_hat_range()].clone_from_slice(z_hat);
        }
        (1, None) => {
            return Err(Error::Shape {
                what: "z_hat at prover 1 (required)",
                expected: l.width_a * l.tau,
                actual: 0,
            });
        }
        (_, None) => {}
        (_, Some(z_hat)) => {
            return Err(Error::Shape {
                what: "z_hat at a prover p > 1 (must be absent)",
                expected: 0,
                actual: z_hat.len(),
            });
        }
    }
    check_polys("<u>_p", core::slice::from_ref(u_share), 1, l.d, q)?;
    share[l.u] = u_share.clone();
    Ok(share)
}

/// Prover $p$'s share $\lbrace \mathbf y \rbrace\_p \in \mathcal R\_q^{n'}$ of
/// $\mathbf y = (\mathbf v, \mathbf u, 0, 0, \mathbf 0)$ (`5_Building_Blocks.tex`:134, 173-179):
/// [`build_y`]`(v, u)` for $p = 1$ and $\mathbf 0$ for $p \gt 1$ (`ref/README.md`, "Shares";
/// the draft's "slices of $\mathbf v, \mathbf u$" by the same procedure as $\hat{\mathbf z}$).
///
/// # Errors
/// [`Error::InvalidParty`]; [`Error::Shape`] or [`Error::NonCanonical`] unless `v` has $n\_D$
/// and `u_com` $n\_B$ polynomials of $d$ canonical residues (checked for every $p$).
pub fn y_share<const K: usize>(
    ctx: &Context<K>,
    party: PartyId,
    v: &[Poly],
    u_com: &[Poly],
) -> Result<Vec<Poly>> {
    let l = &ctx.layout;
    crate::local::check_party(ctx, party)?;
    check_polys("v", v, l.n_d, l.d, ctx.q())?;
    check_polys("u (commitment)", u_com, l.n_b, l.d, ctx.q())?;
    if party == 1 {
        let y = build_y(ctx, v, u_com)?;
        check_len("y", l.n_prime, y.len())?;
        Ok(y)
    } else {
        Ok(vec![ctx.ring.zero(); l.n_prime])
    }
}
