//! The byte size of the Eval proof of the composed PCS (`docs/m1-spec.md`, WP11), as a formula
//! in the parameters and in the sizes of the base PCS's items.
//!
//! The proof (`hachi_co::EvalProof` in `crates/hachico`) consists of what the provers send to
//! the verifier in $\mathsf{Eval}$ (6_Collaborative_Hachi.tex:408-419): $E(u) \in \mathcal R\_q$
//! of $\Pi\_{\mathsf{conv}}$ (field-to-ring-rok.tex:160-164), the index $a^\ast$ of the accepted
//! attempt and its first message $\mathbf v \in \mathcal R\_q^{n\_D}$ of the ring-to-linear step
//! (5_Building_Blocks.tex:115-121, 129), and the sumcheck transcript: $G\_{0,p}, G\_{1,p}$ for
//! every prover (5_Building_Blocks.tex:241), the $\ell''$ round messages $H\_{0,r}(1..d\_0+1)$,
//! $H\_{1,r}(1..d\_1+1)$ (lines 293-300) and the final values
//! $\langle \widetilde w \rangle\_p(\mathbf a), g\_{0,p}(\mathbf a), g\_{1,p}(\mathbf a)$
//! (lines 314-320); and the base PCS's parts: the $3n$ commitments to
//! $\langle \widetilde w \rangle\_p, g\_{0,p}, g\_{1,p}$ (lines 220-221, 230, 241) and the $3n$
//! opening proofs (lines 316-320). The challenges are not part of it: the verifier re-derives
//! them.
//!
//! **The encoding** (format version 2) is a 16-byte header ([`HEADER_BYTES`]) followed by
//! little-endian 64-bit words ([`WORD_BYTES`]): one for $a^\ast$, one per length of a base-PCS
//! item ($6n$), one per residue (an element of $\mathbb F\_{q^k}$ taking $k$ residues), and the
//! base-PCS items' own words. [`sizes`] counts the Hachi-Co part alone (every component but the
//! base PCS's, as in format version 1); [`sizes_with_base`] counts the whole encoding, given the
//! number of words of each base-PCS item ([`BasePcsWords`]), which depends on the base PCS. The
//! `hachi-co` tests check that the encoder produces exactly this many bytes.

use crate::cascade::Params;
use crate::error::{Error, Result};
use crate::rules;

/// The bytes of the header of the encoding: an 8-byte magic and a 64-bit version
/// (`hachi_co::EvalProof::to_bytes`).
pub const HEADER_BYTES: u128 = 16;

/// The bytes of every other entry of the encoding (the attempt index, each length, each residue
/// and each word of the base PCS): one little-endian `u64`.
pub const WORD_BYTES: u128 = 8;

/// The number of 64-bit words of each base-PCS item of one prover
/// (`BasePcs::commitment_to_words`, `BasePcs::proof_to_words` in `hachi-co-core`). They depend
/// on the base PCS and on the sizes of the committed polynomials: $\langle \widetilde w
/// \rangle\_p$ is multilinear in $\ell''$ variables ($2^{\ell''}$ coefficients over
/// $\mathbb Z\_q$), and the blinders $g\_{0,p}$, $g\_{1,p}$ are vectors of $1 + \ell'' d\_0$
/// and $1 + \ell'' d\_1$ elements of $\mathbb F\_{q^k}$. Every prover's items have these sizes
/// (the provers publish them with one all-gather each, which needs equal lengths).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BasePcsWords {
    /// A commitment to $\langle \widetilde w \rangle\_p$.
    pub commitment_w: u128,
    /// A commitment to $g\_{0,p}$.
    pub commitment_g0: u128,
    /// A commitment to $g\_{1,p}$.
    pub commitment_g1: u128,
    /// An opening proof of $\langle \widetilde w \rangle\_p(\mathbf a)$.
    pub proof_w: u128,
    /// An opening proof of $g\_{0,p}(\mathbf a)$.
    pub proof_g0: u128,
    /// An opening proof of $g\_{1,p}(\mathbf a)$.
    pub proof_g1: u128,
}

/// The size of the Eval proof's encoding for some [`Params`], in bytes, with a breakdown per
/// component ([`sizes`], [`sizes_with_base`]). With $d$, $k$, $n$, $n\_D$, $\ell''$, $d\_0$,
/// $d\_1$ of the parameters and the words $c\_w, c\_{g\_0}, c\_{g\_1}$ of a commitment and
/// $\pi\_w, \pi\_{g\_0}, \pi\_{g\_1}$ of a proof of each prover ([`BasePcsWords`]):
///
/// | Component | Words | Bytes |
/// |---|---|---|
/// | header | — | $16$ |
/// | $a^\ast$ | $1$ | $8$ |
/// | lengths of the base-PCS items | $6n$ | $48n$ |
/// | $E(u)$ | $d$ residues | $8d$ |
/// | $\mathbf v$ | $n\_D d$ residues | $8 n\_D d$ |
/// | $G\_{0,p}, G\_{1,p}$ | $2nk$ residues | $16nk$ |
/// | base-PCS commitments | $n(c\_w + c\_{g\_0} + c\_{g\_1})$ | $8n(c\_w + c\_{g\_0} + c\_{g\_1})$ |
/// | round messages | $\ell''(d\_0 + d\_1 + 2)k$ residues | $8\ell''(d\_0 + d\_1 + 2)k$ |
/// | final values | $3nk$ residues | $24nk$ |
/// | base-PCS opening proofs | $n(\pi\_w + \pi\_{g\_0} + \pi\_{g\_1})$ | $8n(\pi\_w + \pi\_{g\_0} + \pi\_{g\_1})$ |
///
/// [`sizes`] sets the three base-PCS components to 0 (the Hachi-Co part alone).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProofSizes {
    /// The header, [`HEADER_BYTES`].
    pub header: u128,
    /// The attempt index $a^\ast$: one word.
    pub attempt: u128,
    /// The lengths of the $6n$ base-PCS items: $6n$ words (0 for [`sizes`]).
    pub pcs_lengths: u128,
    /// $E(u)$: $d$ residues.
    pub e_u: u128,
    /// $\mathbf v$: $n\_D d$ residues.
    pub v: u128,
    /// $G\_{0,1}, \dots, G\_{0,n}$ and $G\_{1,1}, \dots, G\_{1,n}$: $2nk$ residues.
    pub g_sums: u128,
    /// The $3n$ base-PCS commitments (0 for [`sizes`]).
    pub pcs_commitments: u128,
    /// The $\ell''$ round messages, $(d\_0 + 1) + (d\_1 + 1)$ field elements each:
    /// $\ell''(d\_0 + d\_1 + 2)k$ residues.
    pub rounds: u128,
    /// The final values, three field elements per prover: $3nk$ residues.
    pub finals: u128,
    /// The $3n$ base-PCS opening proofs (0 for [`sizes`]).
    pub pcs_openings: u128,
    /// The whole encoding: the sum of the components above.
    pub total: u128,
    /// The number of residues of Hachi-Co's own components ($E(u)$, $\mathbf v$, $G$, the round
    /// messages and the final values; the base PCS's words are not counted as residues).
    pub residues: u128,
    /// $\lceil \log\_2 q \rceil$, the bits a residue in $\[0, q)$ needs.
    pub bits_per_residue: u32,
    /// For reference: the size if every residue took $\lceil \log\_2 q \rceil$ bits instead of
    /// 64, i.e. the header, the word of $a^\ast$, $\lceil \mathrm{residues} \cdot \lceil \log\_2
    /// q \rceil / 8 \rceil$ bytes and the base-PCS components as they are. The encoding does not
    /// pack; this only shows the overhead of the word layout.
    pub packed_total: u128,
}

impl ProofSizes {
    /// The components with their names, in the order of the encoding: `header`, `attempt`,
    /// `PCS lengths`, `E(u)`, `v`, `G`, `PCS commitments`, `rounds`, `finals`,
    /// `PCS openings`. Their bytes add up to [`ProofSizes::total`].
    pub fn components(&self) -> [(&'static str, u128); 10] {
        [
            ("header", self.header),
            ("attempt", self.attempt),
            ("PCS lengths", self.pcs_lengths),
            ("E(u)", self.e_u),
            ("v", self.v),
            ("G", self.g_sums),
            ("PCS commitments", self.pcs_commitments),
            ("rounds", self.rounds),
            ("finals", self.finals),
            ("PCS openings", self.pcs_openings),
        ]
    }
}

/// The size of the Hachi-Co part of the Eval proof's encoding for `params` ([`ProofSizes`]; the
/// formula is in its documentation): every component but the base PCS's (its lengths,
/// commitments and opening proofs are 0), which is the whole encoding of format version 1. It
/// uses $d$, $k$, $n$, $n\_D$, $\ell''$, $d\_0$, $d\_1$ and $q$ of `params`.
///
/// # Errors
///
/// [`Error::Overflow`] if a size does not fit in `u128` (impossible for parameters that
/// [`derive`](fn@crate::derive) accepts with $q < 2^{64}$ and $\ell'' \lt 2^{32}$, but the
/// fields of [`Params`] are public).
pub fn sizes(params: &Params) -> Result<ProofSizes> {
    sizes_of(params, None)
}

/// The size of the whole Eval proof's encoding (format version 2) for `params` and a base PCS
/// whose items have the words `base` ([`ProofSizes`]; the formula is in its documentation): the
/// Hachi-Co part of [`sizes`] and the base PCS's $6n$ lengths, $3n$ commitments and $3n$ opening
/// proofs.
///
/// # Errors
///
/// [`Error::Overflow`] if a size does not fit in `u128`.
pub fn sizes_with_base(params: &Params, base: &BasePcsWords) -> Result<ProofSizes> {
    sizes_of(params, Some(base))
}

/// [`sizes`] and [`sizes_with_base`].
fn sizes_of(params: &Params, base: Option<&BasePcsWords>) -> Result<ProofSizes> {
    let overflow = || Error::Overflow("the proof size");
    let mul = |a: u128, b: u128| a.checked_mul(b).ok_or_else(overflow);
    let add = |a: u128, b: u128| a.checked_add(b).ok_or_else(overflow);
    let d = u128::from(params.d);
    let k = u128::from(params.k);
    let n = u128::from(params.choices.n);
    let n_d = u128::from(params.choices.n_d);
    let ell_pp = u128::from(params.ell_dprime);
    let per_round = add(add(u128::from(params.d0), u128::from(params.d1))?, 2)?;
    // Residues per component.
    let e_u = d;
    let v = mul(n_d, d)?;
    let g_sums = mul(mul(2, n)?, k)?;
    let rounds = mul(mul(ell_pp, per_round)?, k)?;
    let finals = mul(mul(3, n)?, k)?;
    let residues = [v, g_sums, rounds, finals]
        .into_iter()
        .try_fold(e_u, &add)?;
    // Words of the base PCS: the lengths, n commitments and n proofs of each polynomial.
    let (lengths, commitments, openings) = match base {
        None => (0, 0, 0),
        Some(b) => (
            mul(6, n)?,
            mul(
                n,
                add(add(b.commitment_w, b.commitment_g0)?, b.commitment_g1)?,
            )?,
            mul(n, add(add(b.proof_w, b.proof_g0)?, b.proof_g1)?)?,
        ),
    };
    let bytes = |words: u128| mul(words, WORD_BYTES);
    let sizes = ProofSizes {
        header: HEADER_BYTES,
        attempt: WORD_BYTES,
        pcs_lengths: bytes(lengths)?,
        e_u: bytes(e_u)?,
        v: bytes(v)?,
        g_sums: bytes(g_sums)?,
        pcs_commitments: bytes(commitments)?,
        rounds: bytes(rounds)?,
        finals: bytes(finals)?,
        pcs_openings: bytes(openings)?,
        total: 0,
        residues,
        bits_per_residue: rules::ceil_log2(u128::from(params.q)),
        packed_total: 0,
    };
    let total = sizes
        .components()
        .into_iter()
        .try_fold(0, |acc, (_, x)| add(acc, x))?;
    let packed = mul(residues, u128::from(sizes.bits_per_residue))?.div_ceil(8);
    let pcs = add(
        add(sizes.pcs_lengths, sizes.pcs_commitments)?,
        sizes.pcs_openings,
    )?;
    let packed_total = add(add(add(HEADER_BYTES, WORD_BYTES)?, packed)?, pcs)?;
    Ok(ProofSizes {
        total,
        packed_total,
        ..sizes
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::derive;
    use crate::presets::toy_choices;

    /// toy-n1 by hand: d = 16, k = 4, n = n_D = 1, l'' = 15, d0 = 14, d1 = 3, q = 8388637
    /// (24 bits).
    #[test]
    fn toy_n1_by_hand() {
        let (params, _) = derive(&toy_choices(1)).unwrap();
        let s = sizes(&params).unwrap();
        assert_eq!(
            s.components(),
            [
                ("header", 16),
                ("attempt", 8),
                ("PCS lengths", 0),
                ("E(u)", 8 * 16),
                ("v", 8 * 16),
                ("G", 8 * 2 * 4),
                ("PCS commitments", 0),
                ("rounds", 8 * 15 * (14 + 3 + 2) * 4),
                ("finals", 8 * 3 * 4),
                ("PCS openings", 0),
            ]
        );
        assert_eq!(s.total, 9560);
        assert_eq!(s.residues, 16 + 16 + 8 + 15 * 19 * 4 + 12);
        assert_eq!(s.bits_per_residue, 24);
        assert_eq!(s.packed_total, 16 + 8 + (1192 * 24) / 8);
        // With a base PCS of 4-word commitments and proofs of 6 + 2^15, 6 + 4 (1 + 15 * 14)
        // and 6 + 4 (1 + 15 * 3) words (ClearPcs at toy-n1).
        let base = BasePcsWords {
            commitment_w: 4,
            commitment_g0: 4,
            commitment_g1: 4,
            proof_w: 6 + (1 << 15),
            proof_g0: 6 + 4 * 211,
            proof_g1: 6 + 4 * 46,
        };
        let w = sizes_with_base(&params, &base).unwrap();
        let pcs_words = 6 + 12 + (6 + 32768) + (6 + 844) + (6 + 184);
        assert_eq!(w.pcs_lengths, 8 * 6);
        assert_eq!(w.pcs_commitments, 8 * 12);
        assert_eq!(w.pcs_openings, 8 * (32774 + 850 + 190));
        assert_eq!(w.total, 9560 + 8 * pcs_words);
        assert_eq!((w.residues, w.bits_per_residue), (s.residues, 24));
        assert_eq!(w.packed_total, s.packed_total + 8 * pcs_words);
        assert_eq!(
            sizes_with_base(&params, &BasePcsWords::default())
                .unwrap()
                .total,
            9560 + 48
        );
    }

    #[test]
    fn overflow_is_an_error() {
        let (mut params, _) = derive(&toy_choices(1)).unwrap();
        params.d0 = u64::MAX;
        params.d1 = u64::MAX;
        assert!(sizes(&params).is_ok());
        let huge = BasePcsWords {
            proof_w: u128::MAX / 2,
            proof_g0: u128::MAX / 2,
            proof_g1: 2,
            ..BasePcsWords::default()
        };
        assert!(matches!(
            sizes_with_base(&params, &huge),
            Err(Error::Overflow(_))
        ));
        params.d = u64::MAX;
        params.k = u64::MAX;
        params.choices.n = u64::MAX;
        assert!(matches!(sizes(&params), Err(Error::Overflow(_))));
    }
}
