//! A checker for the reduction $\mathcal R\_{\mathsf{eval}, \mathbb F\_{q^k}} \to
//! \mathcal R\_{\mathsf{eval}, \mathcal R\_q}$ of $\Pi\_{\mathsf{conv}}$ (`field-to-ring-rok.tex`:
//! 1-24, 89-169), on the reconstructed witnesses; for tests.

use crate::error::{Error, Result, check_len, pow2};
use crate::public::public_vectors;
use crate::share::e_map;
use hachi_co_core::{Context, PartyState};
use hachi_co_math::{Ext, Poly};

/// The three conditions of [`check_relation`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvRelation {
    /// The input statement: $f(\mathbf x) = y$ for $\mathbf f = \sum\_p \langle \mathbf f
    /// \rangle\_p$, i.e. $((\mathbf x, y), \langle \mathbf f \rangle) \in
    /// \mathcal R\_{\mathsf{eval}, \mathbb F\_{q^k}}$ (`field-to-ring-rok.tex`:6-13).
    pub input: bool,
    /// The output statement: $u = f'(\mathbf x') = \sum\_t d\_t f'\_t$ over all blocks, i.e.
    /// $((\mathbf x', u), \langle \mathbf f' \rangle) \in \mathcal R\_{\mathsf{eval},
    /// \mathcal R\_q}$ (`field-to-ring-rok.tex`:16-23).
    pub output: bool,
    /// The link that the verifier's check enforces: $E(u) = f(\mathbf x)$, read in
    /// $\mathbb F\_{q^k}$ (`field-to-ring-rok.tex`:160-164, Eq. (conv)).
    pub link: bool,
}

impl ConvRelation {
    /// Whether all three conditions hold.
    pub fn holds(&self) -> bool {
        self.input && self.output && self.link
    }
}

/// $f(\mathbf x) = \sum\_{\mathbf q \in \lbrace 0,1 \rbrace^\ell} f\_{\mathbf q} \prod\_j
/// x\_j^{q\_j}$ for the coefficient vector $\mathbf f$ (index $\mathrm{nat}(\mathbf q)$, $q\_1$ the
/// most significant bit; `ref/README.md`, "Bit strings"), by folding the most significant
/// variable first: $f = f\_0 + x\_1 f\_1$ with $f\_0, f\_1$ the two halves of $\mathbf f$. (A
/// different path from the oracle's monomial inner product, `commit.sample_input`.)
///
/// # Errors
/// [`Error::Shape`] unless $\mathbf f$ has $2^{\mathrm{len}(\mathbf x)}$ entries.
pub fn evaluate<const K: usize>(ctx: &Context<K>, f: &[Ext<K>], x: &[Ext<K>]) -> Result<Ext<K>> {
    let field = ctx.field();
    check_len(
        "f (2^len(x) coefficients)",
        pow2(x.len(), "2^len(x)")?,
        f.len(),
    )?;
    let mut table: Vec<Ext<K>> = f.iter().map(|c| field.reduce(c)).collect();
    for xj in x {
        let half = table.len() / 2;
        let (low, high) = table.split_at(half);
        table = low
            .iter()
            .zip(high)
            .map(|(a, b)| field.add(a, &field.mul(xj, b)))
            .collect();
    }
    Ok(table[0])
}

/// $\mathbf f' \in \mathcal R\_q^{2^{\ell'}}$ assembled from the parties' blocks (block $i$ at
/// $\[i 2^m, (i+1) 2^m)$), with zeros for the blocks that nobody holds:
/// $\mathbf f' = \sum\_p \lbrace \mathbf f' \rbrace\_p$ (`field-to-ring-rok.tex`:85).
///
/// # Errors
/// [`Error::Shape`] unless there are $n$ states; [`Error::InvalidParty`] unless `states[p - 1]`
/// is party $p$'s; [`Error::Core`] for a state of the wrong shape ([`PartyState::validate`]: its
/// blocks must be exactly those of the party).
pub fn f_prime_of_states<const K: usize>(
    ctx: &Context<K>,
    states: &[PartyState<K>],
) -> Result<Vec<Poly>> {
    let l = &ctx.layout;
    check_len("states (one per party)", l.n, states.len())?;
    let mut f_prime = vec![ctx.ring.zero(); pow2(l.ell_prime, "2^l'")?];
    for (p, state) in (1..=l.n).zip(states) {
        if state.party != p {
            return Err(Error::InvalidParty {
                party: state.party,
                n: l.n,
            });
        }
        state.validate(l)?;
        for block in &state.blocks {
            let start = block.index * l.block_len;
            let slot = f_prime
                .get_mut(start..start + l.block_len)
                .ok_or(Error::Shape {
                    what: "f' (2^l' = 2^r 2^m entries)",
                    expected: start + l.block_len,
                    actual: l.n_blocks * l.block_len,
                })?;
            slot.clone_from_slice(&block.f_prime);
        }
    }
    Ok(f_prime)
}

/// The monomials of $\mathbf x'$ computed in $\mathcal R\_q$ with the entries of $\mathbf x'$
/// embedded (Hachi Eq. (7)), in the MSB-first order: an independent path to $\mathbf d$.
fn ring_monomials<const K: usize>(ctx: &Context<K>, x: &[Ext<K>]) -> Result<Vec<Poly>> {
    let ring = &ctx.ring;
    let mut out = vec![ring.monomial(0)];
    for xj in x {
        let e = ctx.sub.embed(xj);
        let mut next = Vec::with_capacity(2 * out.len());
        for v in &out {
            next.push(v.clone());
            next.push(ring.mul(v, &e)?);
        }
        out = next;
    }
    Ok(out)
}

/// Checks the reduction of $\Pi\_{\mathsf{conv}}$ on reconstructed witnesses
/// (`docs/m1-spec.md`, WP8): with $\mathbf f = \sum\_p \langle \mathbf f \rangle\_p$ (the
/// `share` of each state), $\mathbf f'$ assembled from the blocks ([`f_prime_of_states`]) and $u$
/// the sum of the shares $\langle u \rangle\_p$, it reports $f(\mathbf x) = y$ (the input
/// relation), $u = f'(\mathbf x') = \sum\_t d\_t f'\_t$ (the output relation) and
/// $E(u) = f(\mathbf x)$ (the link). $f(\mathbf x)$ is computed by [`evaluate`], and
/// $f'(\mathbf x')$ from the monomials of $\mathbf x'$ formed in $\mathcal R\_q$, independently of
/// the field computation of $\mathbf d$ in [`public_vectors`].
///
/// For an honest run all three hold: $\psi$ turns inner products into traces (Hachi Theorem 2),
/// so $f(\mathbf x) = \sum\_t d\_t \langle \hat{\mathbf f}\_t, \mathrm{monomials}(x\_{\ell\_t +
/// 1}, \dots, x\_\ell) \rangle = \sum\_t d\_t E(f'\_t) = E(u)$, using that $\mathrm{Tr}\_H$ is
/// $\mathcal R\_q^H$-linear and $d\_t \in \mathcal R\_q^H$.
///
/// # Errors
/// As [`public_vectors`] and [`f_prime_of_states`]; [`Error::Shape`] for a $u$ without $d$
/// coefficients.
pub fn check_relation<const K: usize>(
    ctx: &Context<K>,
    x: &[Ext<K>],
    y: &Ext<K>,
    states: &[PartyState<K>],
    u: &Poly,
) -> Result<ConvRelation> {
    let (l, ring, field) = (&ctx.layout, &ctx.ring, ctx.field());
    let public = public_vectors(ctx, x)?;
    check_len("u (d coefficients)", l.d, u.0.len())?;
    let f_prime = f_prime_of_states(ctx, states)?;
    let mut f = vec![field.zero(); pow2(l.ell, "2^l")?];
    for state in states {
        for (acc, c) in f.iter_mut().zip(&state.share) {
            *acc = field.add(acc, c);
        }
    }
    let fx = evaluate(ctx, &f, x)?;
    let monomials = ring_monomials(ctx, &public.x_prime)?;
    let f_prime_at_x_prime = ring.inner(&monomials, &f_prime)?;
    let u_reduced = ring.from_coeffs(&u.0)?;
    let e_u = e_map(ctx, &u_reduced, &public.v_point)?;
    Ok(ConvRelation {
        input: fx == field.reduce(y),
        output: f_prime_at_x_prime == u_reduced,
        link: ctx.sub.extract(&e_u).is_ok_and(|e| e == fx),
    })
}
