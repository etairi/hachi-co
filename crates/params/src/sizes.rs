//! The byte size of the Eval proof of the composed PCS (`docs/m1-spec.md`, WP11), as a formula in
//! the parameters.
//!
//! The proof (`hachi_co::EvalProof` in `crates/hachico`) consists of what the provers send to
//! the verifier in $\mathsf{Eval}$ (6_Collaborative_Hachi.tex:408-419) besides the base PCS:
//! $E(u) \in \mathcal R\_q$ of $\Pi\_{\mathsf{conv}}$ (field-to-ring-rok.tex:160-164), the index
//! $a^\ast$ of the accepted attempt and its first message $\mathbf v \in \mathcal R\_q^{n\_D}$ of
//! the ring-to-linear step (5_Building_Blocks.tex:115-121, 129), and the sumcheck transcript:
//! $G\_{0,p}, G\_{1,p}$ for every prover (5_Building_Blocks.tex:241), the $\ell''$ round messages
//! $H\_{0,r}(1..d\_0+1)$, $H\_{1,r}(1..d\_1+1)$ (lines 293-300) and the final values
//! $\langle \widetilde w \rangle\_p(\mathbf a), g\_{0,p}(\mathbf a), g\_{1,p}(\mathbf a)$
//! (lines 314-320). The challenges are not part of it: the verifier re-derives them.
//!
//! **The encoding** is a 16-byte header ([`HEADER_BYTES`]) followed by little-endian 64-bit
//! words ([`WORD_BYTES`]): one for $a^\ast$, then one per residue, an element of
//! $\mathbb F\_{q^k}$ taking $k$ residues. [`sizes`] counts it; the `hachi-co` tests check that
//! the encoder produces exactly this many bytes.
//!
//! **Not included:** the parts of the base PCS (6_Collaborative_Hachi.tex:289-292;
//! 5_Building_Blocks.tex:220-230, 314-326), which milestone 3 adds: the $3n$ commitments to
//! $g\_{0,p}, g\_{1,p}, \langle \widetilde w \rangle\_p$ and the openings of the $3n$ final claims.

use crate::cascade::Params;
use crate::error::{Error, Result};
use crate::rules;

/// The bytes of the header of the encoding: an 8-byte magic and a 64-bit version
/// (`hachi_co::EvalProof::to_bytes`).
pub const HEADER_BYTES: u128 = 16;

/// The bytes of every other entry of the encoding (the attempt index and each residue): one
/// little-endian `u64`.
pub const WORD_BYTES: u128 = 8;

/// The size of the Eval proof's encoding for some [`Params`], in bytes, with a breakdown per
/// component ([`sizes`]). With $d$, $k$, $n$, $n\_D$, $\ell''$, $d\_0$, $d\_1$ of the parameters:
///
/// | Component | Residues | Bytes |
/// |---|---|---|
/// | header | — | $16$ |
/// | $a^\ast$ | — | $8$ |
/// | $E(u)$ | $d$ | $8d$ |
/// | $\mathbf v$ | $n\_D d$ | $8 n\_D d$ |
/// | $G\_{0,p}, G\_{1,p}$ | $2nk$ | $16nk$ |
/// | round messages | $\ell''(d\_0 + d\_1 + 2)k$ | $8\ell''(d\_0 + d\_1 + 2)k$ |
/// | final values | $3nk$ | $24nk$ |
///
/// The base-PCS parts are not included (see the [module documentation](self)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProofSizes {
    /// The header, [`HEADER_BYTES`].
    pub header: u128,
    /// The attempt index $a^\ast$: one word.
    pub attempt: u128,
    /// $E(u)$: $d$ residues.
    pub e_u: u128,
    /// $\mathbf v$: $n\_D d$ residues.
    pub v: u128,
    /// $G\_{0,1}, \dots, G\_{0,n}$ and $G\_{1,1}, \dots, G\_{1,n}$: $2nk$ residues.
    pub g_sums: u128,
    /// The $\ell''$ round messages, $(d\_0 + 1) + (d\_1 + 1)$ field elements each:
    /// $\ell''(d\_0 + d\_1 + 2)k$ residues.
    pub rounds: u128,
    /// The final values, three field elements per prover: $3nk$ residues.
    pub finals: u128,
    /// The whole encoding: the sum of the components above.
    pub total: u128,
    /// The number of residues in the proof (every component except the header and $a^\ast$).
    pub residues: u128,
    /// $\lceil \log\_2 q \rceil$, the bits a residue in $\[0, q)$ needs.
    pub bits_per_residue: u32,
    /// For reference: the size if every residue took $\lceil \log\_2 q \rceil$ bits instead of
    /// 64, i.e. the header, the word of $a^\ast$ and $\lceil \mathrm{residues} \cdot \lceil \log\_2
    /// q \rceil / 8 \rceil$ bytes. The encoding does not pack; this only shows the overhead of the
    /// word layout.
    pub packed_total: u128,
}

impl ProofSizes {
    /// The components with their names, in the order of the encoding: `header`, `attempt`,
    /// `E(u)`, `v`, `G`, `rounds`, `finals`. Their bytes add up to [`ProofSizes::total`].
    pub fn components(&self) -> [(&'static str, u128); 7] {
        [
            ("header", self.header),
            ("attempt", self.attempt),
            ("E(u)", self.e_u),
            ("v", self.v),
            ("G", self.g_sums),
            ("rounds", self.rounds),
            ("finals", self.finals),
        ]
    }
}

/// The size of the Eval proof's encoding for `params` ([`ProofSizes`]; the formula is in its
/// documentation). It uses $d$, $k$, $n$, $n\_D$, $\ell''$, $d\_0$, $d\_1$ and $q$ of `params`;
/// the base-PCS parts are not included.
///
/// # Errors
///
/// [`Error::Overflow`] if a size does not fit in `u128` (impossible for parameters that
/// [`derive`](fn@crate::derive) accepts with $q < 2^{64}$ and $\ell'' \lt 2^{32}$, but the
/// fields of [`Params`] are public).
pub fn sizes(params: &Params) -> Result<ProofSizes> {
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
    let bytes = |residues: u128| mul(residues, WORD_BYTES);
    let sizes = ProofSizes {
        header: HEADER_BYTES,
        attempt: WORD_BYTES,
        e_u: bytes(e_u)?,
        v: bytes(v)?,
        g_sums: bytes(g_sums)?,
        rounds: bytes(rounds)?,
        finals: bytes(finals)?,
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
    let packed_total = add(add(HEADER_BYTES, WORD_BYTES)?, packed)?;
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
                ("E(u)", 8 * 16),
                ("v", 8 * 16),
                ("G", 8 * 2 * 4),
                ("rounds", 8 * 15 * (14 + 3 + 2) * 4),
                ("finals", 8 * 3 * 4),
            ]
        );
        assert_eq!(s.total, 9560);
        assert_eq!(s.residues, 16 + 16 + 8 + 15 * 19 * 4 + 12);
        assert_eq!(s.bits_per_residue, 24);
        assert_eq!(s.packed_total, 16 + 8 + (1192 * 24) / 8);
    }

    #[test]
    fn overflow_is_an_error() {
        let (mut params, _) = derive(&toy_choices(1)).unwrap();
        params.d0 = u64::MAX;
        params.d1 = u64::MAX;
        assert!(sizes(&params).is_ok());
        params.d = u64::MAX;
        params.k = u64::MAX;
        params.choices.n = u64::MAX;
        assert!(matches!(sizes(&params), Err(Error::Overflow(_))));
    }
}
