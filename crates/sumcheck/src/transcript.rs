//! The messages of the sumcheck: the public transcript the verifier reads, the private data of
//! one party, and the final evaluation claims that the base PCS will have to prove.

use crate::error::VerifyError;
use hachi_co_core::{Blinder, Layout, Poly, SumcheckCoins};
use hachi_co_math::Ext;
use hachi_co_mpc::PartyId;

/// The provers' messages of one round $r$ (`5_Building_Blocks.tex`:293-300): the outputs of
/// $\mathcal F\_{\mathsf{AH}}(\mathbb I)$, $H\_{0,r}(t)$ for $t = 1, \dots, d\_0 + 1$ and $H\_{1,r}(t)$
/// for $t = 1, \dots, d\_1 + 1$ (fixture keys `sumcheck/rounds/<r-1>/H0` and `H1`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundMessage<const K: usize> {
    /// $H\_{0,r}(1), \dots, H\_{0,r}(d\_0 + 1)$.
    pub h0: Vec<Ext<K>>,
    /// $H\_{1,r}(1), \dots, H\_{1,r}(d\_1 + 1)$.
    pub h1: Vec<Ext<K>>,
}

/// The openings of `5_Building_Blocks.tex`:314-320, one per party $p = 1, \dots, n$ (index
/// $p - 1$; fixture keys `sumcheck/final/w`, `g0`, `g1`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinalValues<const K: usize> {
    /// $\langle \widetilde w \rangle\_p(\mathbf a)$.
    pub w: Vec<Ext<K>>,
    /// $g\_{0,p}(\mathbf a)$.
    pub g0: Vec<Ext<K>>,
    /// $g\_{1,p}(\mathbf a)$.
    pub g1: Vec<Ext<K>>,
}

/// Everything the verifier of the sumcheck reads (`5_Building_Blocks.tex`:241, 300, 320), in the
/// shape of the oracle's transcript: the per-party hypercube sums of the blinders, the $\ell''$
/// round messages and the per-party final values. (The base PCS is not modelled in milestone 1:
/// commitments and opening proofs are absent, `ref/README.md`, choice 13.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckTranscript<const K: usize> {
    /// $G\_{0,p} = \sum\_{\mathbf v} g\_{0,p}(\mathbf v)$ for $p = 1, \dots, n$ (fixture key
    /// `sumcheck/G0`).
    pub g0_sums: Vec<Ext<K>>,
    /// $G\_{1,p} = \sum\_{\mathbf v} g\_{1,p}(\mathbf v)$ for $p = 1, \dots, n$ (fixture key
    /// `sumcheck/G1`).
    pub g1_sums: Vec<Ext<K>>,
    /// The round messages, rounds $1, \dots, \ell''$ in order.
    pub rounds: Vec<RoundMessage<K>>,
    /// The final values.
    pub finals: FinalValues<K>,
}

/// `Err(Shape)` unless `actual == expected`.
fn shape(what: &'static str, expected: usize, actual: usize) -> Result<(), VerifyError> {
    if expected != actual {
        return Err(VerifyError::Shape {
            what,
            expected,
            actual,
        });
    }
    Ok(())
}

/// Checks the length and that every coordinate is below $q$.
fn canonical<const K: usize>(
    what: &'static str,
    values: &[Ext<K>],
    len: usize,
    q: u64,
) -> Result<(), VerifyError> {
    shape(what, len, values.len())?;
    for v in values {
        if let Some(bad) = v.0.iter().find(|c| **c >= q) {
            return Err(VerifyError::NonCanonical {
                what,
                value: *bad,
                q,
            });
        }
    }
    Ok(())
}

impl<const K: usize> SumcheckTranscript<K> {
    /// Checks the shape against the layout ($n$ hypercube sums of each kind, $\ell''$ rounds of
    /// $d\_0 + 1$ and $d\_1 + 1$ values, $n$ final values of each kind) and that every coordinate
    /// is a canonical residue in $\[0, q)$. The verifier runs it before anything else, so that a
    /// malformed transcript is an error, never a panic.
    ///
    /// # Errors
    /// [`VerifyError::Shape`] or [`VerifyError::NonCanonical`] for the first violation.
    pub fn validate(&self, layout: &Layout, q: u64) -> Result<(), VerifyError> {
        let n = layout.n;
        canonical("G_0 (one per party)", &self.g0_sums, n, q)?;
        canonical("G_1 (one per party)", &self.g1_sums, n, q)?;
        shape("rounds (l'')", layout.ell_pp, self.rounds.len())?;
        for round in &self.rounds {
            canonical(
                "H_0 round message (d0 + 1 values)",
                &round.h0,
                layout.d0 + 1,
                q,
            )?;
            canonical(
                "H_1 round message (d1 + 1 values)",
                &round.h1,
                layout.d1 + 1,
                q,
            )?;
        }
        canonical("final <w~>_p(a) (one per party)", &self.finals.w, n, q)?;
        canonical("final g_0,p(a) (one per party)", &self.finals.g0, n, q)?;
        canonical("final g_1,p(a) (one per party)", &self.finals.g1, n, q)
    }
}

/// One party's values of one round: its input to $\mathcal F\_{\mathsf{AH}}(\mathbb I)$
/// (`5_Building_Blocks.tex`:293-300; fixture keys `sumcheck/rounds/<r-1>/F_AH_inputs_H0` and
/// `F_AH_inputs_H1`, entry $p - 1$) and the same values without the blinders (for the degree
/// audit, fixture key `sumcheck/round_degrees_unblinded`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundValues<const K: usize> {
    /// $H\_{0,p,r}(t) = F\_{0,p,r}(t) + \rho\, g\_{0,p,r}(t)$ for $t = 1, \dots, d\_0 + 1$.
    pub h0: Vec<Ext<K>>,
    /// $H\_{1,p,r}(t) = F\_{1,p,r}(t) + \rho\, g\_{1,p,r}(t)$ for $t = 1, \dots, d\_1 + 1$.
    pub h1: Vec<Ext<K>>,
    /// $F\_{0,p,r}(t)$, the round polynomial of $F\_{0,p}$ alone, $t = 1, \dots, d\_0 + 1$.
    pub f0: Vec<Ext<K>>,
    /// $F\_{1,p,r}(t)$, $t = 1, \dots, d\_1 + 1$.
    pub f1: Vec<Ext<K>>,
}

/// Party $p$'s private data after [`prove`](crate::prove): what the tests (and, in milestone 3,
/// the base PCS) need besides the public transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckPrivate<const K: usize> {
    /// The party $p$.
    pub party: PartyId,
    /// $\langle \mathbf r \rangle\_p$, $n'$ polynomials of $2d$ residues
    /// ([`quotient_share`](crate::quotient_share)).
    pub quotient: Vec<Poly>,
    /// The hypercube values of $\langle \widetilde w \rangle\_p$, $2^{\ell''}$ residues
    /// ([`w_table`](crate::w_table)): the polynomial committed in `5_Building_Blocks.tex`:229.
    pub w_table: Vec<u64>,
    /// The coefficients of $g\_{0,p}$.
    pub g0: Blinder<K>,
    /// The coefficients of $g\_{1,p}$.
    pub g1: Blinder<K>,
    /// The verifier's coins $\alpha, \rho, \tau\_0, \tau\_1$.
    pub coins: SumcheckCoins<K>,
    /// The point $\mathbf a = (a\_1, \dots, a\_{\ell''})$ of the round coins.
    pub point: Vec<Ext<K>>,
    /// This party's values of every round.
    pub rounds: Vec<RoundValues<K>>,
}

/// Which committed polynomial a [`Claim`] is about (`5_Building_Blocks.tex`:316-318).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClaimedPoly {
    /// The blinder $g\_{0,p}$, of individual degree $d\_0$ in $\ell''$ variables.
    G0,
    /// The blinder $g\_{1,p}$, of individual degree $d\_1$.
    G1,
    /// The witness polynomial $\langle \widetilde w \rangle\_p$, multilinear in $\ell''$ variables.
    W,
}

/// An evaluation claim "the polynomial `poly` committed by party `party` takes the value `value`
/// at the point $\mathbf a$" (`5_Building_Blocks.tex`:322-326), which the base PCS proves in
/// milestone 3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Claim<const K: usize> {
    /// The party $p \in \lbrace 1, \dots, n \rbrace$ that committed to the polynomial.
    pub party: PartyId,
    /// The polynomial.
    pub poly: ClaimedPoly,
    /// The claimed value at the point $\mathbf a$.
    pub value: Ext<K>,
}

impl<const K: usize> Claim<K> {
    /// The oracle's name of the PCS check of this claim: `pcs_open_g0_p<p>`, `pcs_open_g1_p<p>` or
    /// `pcs_open_w_p<p>` (`ref/hachico_ref/sumcheck.py`, `verify`).
    pub fn check_name(&self) -> String {
        let kind = match self.poly {
            ClaimedPoly::G0 => "g0",
            ClaimedPoly::G1 => "g1",
            ClaimedPoly::W => "w",
        };
        format!("pcs_open_{kind}_p{}", self.party)
    }
}

/// The verifier's output: the point $\mathbf a$ and the $3n$ evaluation claims
/// $g\_{0,p}(\mathbf a)$, $g\_{1,p}(\mathbf a)$, $\langle \widetilde w \rangle\_p(\mathbf a)$, in the
/// order of the oracle's checks (party by party: $g\_0$, $g\_1$, $\widetilde w$).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinalClaims<const K: usize> {
    /// $\mathbf a = (a\_1, \dots, a\_{\ell''})$.
    pub point: Vec<Ext<K>>,
    /// The $3n$ claims.
    pub claims: Vec<Claim<K>>,
}
