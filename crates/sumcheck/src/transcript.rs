//! The messages of the sumcheck: the public transcript the verifier reads (with the base-PCS
//! commitments and opening proofs as words), the private data of one party, and the final
//! evaluation claims that the base PCS proves.

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

/// The opened values of `5_Building_Blocks.tex`:314-320, one per party $p = 1, \dots, n$ (index
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

impl<const K: usize> FinalValues<K> {
    /// The values of `poly`, one per party.
    pub fn of(&self, poly: ClaimedPoly) -> &[Ext<K>] {
        match poly {
            ClaimedPoly::W => &self.w,
            ClaimedPoly::G0 => &self.g0,
            ClaimedPoly::G1 => &self.g1,
        }
    }
}

/// Words of the base PCS for the three committed polynomials of every party, one word vector
/// per party $p = 1, \dots, n$ (index $p - 1$): the commitments
/// ([`BasePcs::commitment_to_words`](hachi_co_core::BasePcs::commitment_to_words);
/// `5_Building_Blocks.tex`:220-221, 230, sent at 241) or the opening proofs
/// ([`BasePcs::proof_to_words`](hachi_co_core::BasePcs::proof_to_words); lines 316-320). The
/// sumcheck carries them as words so that its types do not depend on the base PCS; the verifier
/// decodes them with the base PCS it is given.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PcsWords {
    /// The words of $\langle \widetilde w \rangle\_p$.
    pub w: Vec<Vec<u64>>,
    /// The words of $g\_{0,p}$.
    pub g0: Vec<Vec<u64>>,
    /// The words of $g\_{1,p}$.
    pub g1: Vec<Vec<u64>>,
}

impl PcsWords {
    /// The words of `poly`, one vector per party.
    pub fn of(&self, poly: ClaimedPoly) -> &[Vec<u64>] {
        match poly {
            ClaimedPoly::W => &self.w,
            ClaimedPoly::G0 => &self.g0,
            ClaimedPoly::G1 => &self.g1,
        }
    }

    /// The words of `poly`, mutably (for tests that tamper with them).
    pub fn of_mut(&mut self, poly: ClaimedPoly) -> &mut Vec<Vec<u64>> {
        match poly {
            ClaimedPoly::W => &mut self.w,
            ClaimedPoly::G0 => &mut self.g0,
            ClaimedPoly::G1 => &mut self.g1,
        }
    }

    /// The number of words in all.
    pub fn total_words(&self) -> usize {
        [&self.w, &self.g0, &self.g1]
            .into_iter()
            .flatten()
            .map(Vec::len)
            .sum()
    }

    /// One word vector per party for each polynomial.
    fn validate(&self, what: &'static str, n: usize) -> Result<(), VerifyError> {
        for poly in ClaimedPoly::ORDER {
            shape(what, n, self.of(poly).len())?;
        }
        Ok(())
    }
}

/// Everything the verifier of the sumcheck reads (`5_Building_Blocks.tex`:241, 300, 320), in the
/// shape of the oracle's transcript with the base PCS added: the per-party hypercube sums of the
/// blinders and base-PCS commitments, the $\ell''$ round messages, the per-party final values
/// and the base-PCS opening proofs. (The oracle does not model the base PCS, `ref/README.md`,
/// choice 13: its transcripts have no commitments or openings.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckTranscript<const K: usize> {
    /// $G\_{0,p} = \sum\_{\mathbf v} g\_{0,p}(\mathbf v)$ for $p = 1, \dots, n$ (fixture key
    /// `sumcheck/G0`).
    pub g0_sums: Vec<Ext<K>>,
    /// $G\_{1,p} = \sum\_{\mathbf v} g\_{1,p}(\mathbf v)$ for $p = 1, \dots, n$ (fixture key
    /// `sumcheck/G1`).
    pub g1_sums: Vec<Ext<K>>,
    /// The base-PCS commitments to $\langle \widetilde w \rangle\_p$, $g\_{0,p}$, $g\_{1,p}$ of
    /// every party (`5_Building_Blocks.tex`:220-221, 230, sent at 241), sent with the hypercube
    /// sums and absorbed after them, before the coins $\alpha, \rho, \tau\_0, \tau\_1$.
    pub commitments: PcsWords,
    /// The round messages, rounds $1, \dots, \ell''$ in order.
    pub rounds: Vec<RoundMessage<K>>,
    /// The final values.
    pub finals: FinalValues<K>,
    /// The base-PCS proofs that open the commitments to the final values at the point
    /// $\mathbf a$ (`5_Building_Blocks.tex`:314-320).
    pub openings: PcsWords,
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
    /// Checks the shape against the layout ($n$ hypercube sums of each kind, $n$ commitments of
    /// each polynomial, $\ell''$ rounds of $d\_0 + 1$ and $d\_1 + 1$ values, $n$ final values of
    /// each kind, $n$ opening proofs of each polynomial) and that every coordinate of a field
    /// element is a canonical residue in $\[0, q)$. The words of the base PCS are decoded by the
    /// verifier, with the base PCS. The verifier runs this before anything else, so that a
    /// malformed transcript is an error, never a panic.
    ///
    /// # Errors
    /// [`VerifyError::Shape`] or [`VerifyError::NonCanonical`] for the first violation.
    pub fn validate(&self, layout: &Layout, q: u64) -> Result<(), VerifyError> {
        let n = layout.n;
        canonical("G_0 (one per party)", &self.g0_sums, n, q)?;
        canonical("G_1 (one per party)", &self.g1_sums, n, q)?;
        self.commitments
            .validate("base-PCS commitments (one per party and polynomial)", n)?;
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
        canonical("final g_1,p(a) (one per party)", &self.finals.g1, n, q)?;
        self.openings
            .validate("base-PCS opening proofs (one per party and polynomial)", n)
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

/// Party $p$'s private data after [`prove`](crate::prove): what the tests need besides the
/// public transcript (the base PCS's states are dropped once the openings are made).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckPrivate<const K: usize> {
    /// The party $p$.
    pub party: PartyId,
    /// $\langle \mathbf r \rangle\_p$, $n'$ polynomials of $2d$ residues
    /// ([`quotient_share`](crate::quotient_share)).
    pub quotient: Vec<Poly>,
    /// The hypercube values of $\langle \widetilde w \rangle\_p$, $2^{\ell''}$ residues
    /// ([`w_table`](crate::w_table)): the polynomial committed in `5_Building_Blocks.tex`:230
    /// (by its monomial coefficients, [`hachi_co_core::base_pcs::mobius`] of this table).
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
    /// The blinder $g\_{0,p}$, of individual degree $d\_0$ in $\ell''$ variables, committed by
    /// its coefficient vector and opened as an inner product.
    G0,
    /// The blinder $g\_{1,p}$, of individual degree $d\_1$, likewise.
    G1,
    /// The witness polynomial $\langle \widetilde w \rangle\_p$, multilinear in $\ell''$
    /// variables, committed by its monomial coefficients and opened as a multilinear evaluation.
    W,
}

impl ClaimedPoly {
    /// A party's three polynomials in the order of the oracle's checks
    /// (`ref/hachico_ref/sumcheck.py`, `verify`): $g\_0$, $g\_1$, $\widetilde w$. The provers
    /// make, and the verifier checks, the openings in this order.
    pub const ORDER: [Self; 3] = [Self::G0, Self::G1, Self::W];

    /// The short name, `g0`, `g1` or `w`, used in check names and context labels.
    pub fn name(self) -> &'static str {
        match self {
            Self::G0 => "g0",
            Self::G1 => "g1",
            Self::W => "w",
        }
    }

    /// The index of the base PCS's seed ([`PartyRandomness::base_pcs`]) for the commitment to
    /// this polynomial (`opening` false) or for its opening: the position $j$ in
    /// [`ClaimedPoly::ORDER`] for the commitment and $3 + j$ for the opening, so that the six
    /// seeds of a party are distinct.
    ///
    /// [`PartyRandomness::base_pcs`]: hachi_co_core::PartyRandomness::base_pcs
    pub fn seed_index(self, opening: bool) -> u64 {
        let j = match self {
            Self::G0 => 0,
            Self::G1 => 1,
            Self::W => 2,
        };
        if opening { 3 + j } else { j }
    }
}

/// An evaluation claim "the polynomial `poly` committed by party `party` takes the value `value`
/// at the point $\mathbf a$" (`5_Building_Blocks.tex`:322-326), which the base PCS proves.
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
        format!("pcs_open_{}_p{}", self.poly.name(), self.party)
    }
}

/// The point $\mathbf a$ and the $3n$ evaluation claims $g\_{0,p}(\mathbf a)$,
/// $g\_{1,p}(\mathbf a)$, $\langle \widetilde w \rangle\_p(\mathbf a)$, in the order of the
/// oracle's checks (party by party: $g\_0$, $g\_1$, $\widetilde w$), which the verifier checks
/// with the base PCS ([`verify_claims`](crate::verify_claims) returns them once checked).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinalClaims<const K: usize> {
    /// $\mathbf a = (a\_1, \dots, a\_{\ell''})$.
    pub point: Vec<Ext<K>>,
    /// The $3n$ claims.
    pub claims: Vec<Claim<K>>,
}
