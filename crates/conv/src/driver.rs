//! The $\Pi\_{\mathsf{conv}}$ driver of one party (`field-to-ring-rok.tex`:105-169), run against a
//! [`Comm`].

use crate::error::{Error, Result};
use crate::public::public_vectors;
use crate::share::{fah_e_input, u_share};
use hachi_co_core::{Context, PartyState, PublicVectors};
use hachi_co_math::{Ext, Poly};
use hachi_co_mpc::Comm;

/// The [`Comm::fah`] label of $\mathcal F\_{\mathsf{AH}}(E)$ (`field-to-ring-rok.tex`:160-162).
pub const LABEL_FAH_E: &str = "hachi-co/conv/F_AH(E)";
/// The label under which the message $E(u)$ is absorbed into a [`hachi_co_core::Challenger`]
/// (with `absorb_polys`) before the next coins are drawn; $\Pi\_{\mathsf{conv}}$ itself draws no
/// coins.
pub const LABEL_E_U: &str = "hachi-co/conv/E(u)";

/// What $\Pi\_{\mathsf{conv}}$ gives party $p$.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvOutput<const K: usize> {
    /// The public vectors $\mathbf x', \mathbf b, \mathbf a, v, \mathbf d$ ([`public_vectors`]).
    pub public: PublicVectors<K>,
    /// $\langle u \rangle\_p = \mathbf d^T \lbrace \mathbf f' \rbrace\_p$ ([`u_share`]); $u$ stays
    /// secret-shared (`ref/README.md`, choice 4).
    pub u_share: Poly,
    /// $E(u) \in \mathcal R\_q^H$, the output of $\mathcal F\_{\mathsf{AH}}(E)$, which the provers
    /// send to the verifier (`field-to-ring-rok.tex`:160-164).
    pub e_u: Poly,
}

/// $\Pi\_{\mathsf{conv}}$ (`field-to-ring-rok.tex`:105-169) for the party $p$ = `comm.id()` with
/// state `state` = $\mathsf{st}\_p$ from Commit and the evaluation point $\mathbf x \in
/// \mathbb F\_{q^k}^\ell$:
///
/// 1. [`public_vectors`] (steps 1-2, lines 105-123);
/// 2. $\langle u \rangle\_p$ by [`u_share`] from the blocks of $\mathbf f'$ in the state (steps
///    3-4, lines 124-131; $\mathsf{PConv}$ ran in Commit, `ref/README.md`, choice 3);
/// 3. [`fah_e_input`] and [`Comm::fah`] with [`LABEL_FAH_E`]: every party receives
///    $E(u)$ (steps 5-6, lines 132-164).
///
/// The verifier then checks $y = E(u)$ ([`crate::verify`]). With [`hachi_co_mpc::Loopback`]
/// ($n = 1$) the collective call is the identity.
///
/// # Errors
/// [`Error::Network`] or [`Error::InvalidParty`] if `comm` does not match the context or the
/// state belongs to another party; [`Error::Shape`] for a malformed $\mathbf x$ or a malformed
/// output of `comm`; [`Error::Core`] for a state of the wrong shape; the errors of `comm`.
pub fn pi_conv<const K: usize, C: Comm>(
    ctx: &Context<K>,
    comm: &mut C,
    state: &PartyState<K>,
    x: &[Ext<K>],
) -> Result<ConvOutput<K>> {
    let n = ctx.layout.n;
    if comm.n() != n {
        return Err(Error::Network {
            what: "n",
            expected: n as u64,
            actual: comm.n() as u64,
        });
    }
    if comm.modulus() != ctx.q() {
        return Err(Error::Network {
            what: "q",
            expected: ctx.q(),
            actual: comm.modulus(),
        });
    }
    if comm.id() == 0 || comm.id() > n || state.party != comm.id() {
        return Err(Error::InvalidParty {
            party: state.party,
            n,
        });
    }
    let public = public_vectors(ctx, x)?;
    let u_share = u_share(ctx, state, &public)?;
    let local = fah_e_input(ctx, &u_share, &public.v_point)?;
    let output = comm.fah(LABEL_FAH_E, &local.0)?;
    if output.len() != ctx.layout.d {
        return Err(Error::Shape {
            what: "F_AH(E) output (d residues)",
            expected: ctx.layout.d,
            actual: output.len(),
        });
    }
    Ok(ConvOutput {
        public,
        u_share,
        e_u: Poly(output),
    })
}
