//! Steps 4-6 of $\Pi\_{\mathsf{conv}}$ for one party (`field-to-ring-rok.tex`:124-164): the share
//! $\langle u \rangle\_p = \mathbf d^T \lbrace \mathbf f' \rbrace\_p$, the map $E$ with its matrix,
//! and the party's input $E(\langle u \rangle\_p)$ to $\mathcal F\_{\mathsf{AH}}(E)$.

use crate::error::{Error, Result};
use hachi_co_core::{Context, PartyState, PublicVectors};
use hachi_co_math::Poly;

/// Party $p$'s share $\langle u \rangle\_p = \mathbf d^T \lbrace \mathbf f' \rbrace\_p =
/// \sum\_{i} \sum\_{j \lt 2^m} d\_{i 2^m + j} f'\_{i,j} \in \mathcal R\_q$ over the blocks $i$ of its
/// state (`field-to-ring-rok.tex`:128-131; the oracle's `conv.pi_conv`), with $d\_t$ embedded in
/// $\mathcal R\_q$ (exact products). The sum over the parties is $u = \mathbf d^T \mathbf f' =
/// f'(\mathbf x')$.
///
/// **Choice** (the oracle's, `ref/README.md`, choice 4): $u$ stays secret-shared; only $E(u)$ is
/// opened, and $\langle u \rangle\_p$ is the last entry of party $p$'s share of $\mathbf z'$
/// (`5_Building_Blocks.tex`:171).
///
/// # Errors
/// [`Error::Core`] if the state or the public vectors do not have the shapes of the layout
/// ([`PartyState::validate`], [`PublicVectors::validate`]); [`Error::Shape`] if a block lies
/// beyond $\mathbf d$ (a layout with $r + m \ne \ell'$).
pub fn u_share<const K: usize>(
    ctx: &Context<K>,
    state: &PartyState<K>,
    public: &PublicVectors<K>,
) -> Result<Poly> {
    let (l, ring) = (&ctx.layout, &ctx.ring);
    state.validate(l)?;
    public.validate(l)?;
    let mut acc = ring.zero();
    for block in &state.blocks {
        let start = block.index * l.block_len;
        let d_block = public
            .d
            .get(start..start + l.block_len)
            .ok_or(Error::Shape {
                what: "d (2^l' = 2^r 2^m entries)",
                expected: start + l.block_len,
                actual: public.d.len(),
            })?;
        let d_ring: Vec<Poly> = d_block.iter().map(|e| ctx.sub.embed(e)).collect();
        acc = ring.add(&acc, &ring.inner(&d_ring, &block.f_prime)?)?;
    }
    Ok(acc)
}

/// $E(u) = \frac{k}{d} \mathrm{Tr}\_H\bigl(u \cdot \sigma\_{-1}(v)\bigr) \in \mathcal R\_q^H$
/// (`field-to-ring-rok.tex`:132-135, 162; [`hachi_co_math::Subfield::e_map`]). $E$ is
/// $\mathbb Z\_q$-linear (indeed $\mathcal R\_q^H$-linear), so the parties may apply it to their
/// shares.
///
/// # Errors
/// [`Error::Math`] unless $u$ and $v$ have $d$ coefficients.
pub fn e_map<const K: usize>(ctx: &Context<K>, u: &Poly, v_point: &Poly) -> Result<Poly> {
    Ok(ctx.sub.e_map(u, v_point)?)
}

/// The $d \times d$ matrix $\mathbf E$ over $\mathbb Z\_q$ with $\mathrm{cf}(E(u)) = \mathbf E
/// \cdot \mathrm{cf}(u)$ (`field-to-ring-rok.tex`:137-158), row-major: `e[i][j]` is coefficient
/// $i$ of $E(X^j)$ (the oracle's `conv.e_matrix`, fixture key `conv/E_matrix`;
/// [`hachi_co_math::Subfield::e_matrix`]).
///
/// # Errors
/// [`Error::Math`] unless $v$ has $d$ coefficients.
pub fn e_matrix<const K: usize>(ctx: &Context<K>, v_point: &Poly) -> Result<Vec<Vec<u64>>> {
    Ok(ctx.sub.e_matrix(v_point)?)
}

/// Party $p$'s input to $\mathcal F\_{\mathsf{AH}}(E)$ (`field-to-ring-rok.tex`:160-162):
/// $E(\langle u \rangle\_p)$, the image of its share under the public linear map $E$ (the
/// convention of [`hachi_co_mpc::Comm::fah`]). The sum over the parties is $E(u)$.
///
/// # Errors
/// As [`e_map`].
pub fn fah_e_input<const K: usize>(
    ctx: &Context<K>,
    u_share: &Poly,
    v_point: &Poly,
) -> Result<Poly> {
    e_map(ctx, u_share, v_point)
}
