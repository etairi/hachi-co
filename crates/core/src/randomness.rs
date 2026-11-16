//! A party's private randomness: the trait the protocol code draws from, and its PRG
//! implementation (`docs/m1-spec.md`, "Randomness and coins").

use crate::data::{Blinder, BlinderKind};
use crate::{Error, Result};
use core::ops::Range;
use hachi_co_math::{Ext, Fqk};
use hachi_co_sampling::{ShakePrg, rej, uniform};
use rand::Rng;
use rayon::prelude::*;
use std::collections::BTreeSet;

#[cfg(doc)]
use hachi_co_params::PadRule;

/// The PRG domain of the masks $\mathbf r\_i^j$ ([`PartyRandomness::mask`]; index = iteration
/// $j$, 0-indexed).
pub const DOMAIN_MASK: &str = "hachi-co/r2l/mask";
/// The PRG domain of the commitment's randomness $\boldsymbol\rho^u\_i$
/// ([`PartyRandomness::rho_u`]; index 0).
pub const DOMAIN_RHO_U: &str = "hachi-co/commit/rho-u";
/// The PRG domain of the first message's randomness $\boldsymbol\rho^v\_i$
/// ([`PartyRandomness::rho_v`]; index 0).
pub const DOMAIN_RHO_V: &str = "hachi-co/r2l/rho-v";
/// The PRG domain of the witness pad $\rho^w\_i$ ([`PartyRandomness::rho_w`]; index 0).
pub const DOMAIN_RHO_W: &str = "hachi-co/sumcheck/rho-w";
/// The PRG domain of the blinder $g\_{0,i}$ ([`PartyRandomness::blinder`]; index 0).
pub const DOMAIN_BLINDER_G0: &str = "hachi-co/sumcheck/g0";
/// The PRG domain of the blinder $g\_{1,i}$ ([`PartyRandomness::blinder`]; index 0).
pub const DOMAIN_BLINDER_G1: &str = "hachi-co/sumcheck/g1";
/// The PRG domain of the base PCS's seeds ([`PartyRandomness::base_pcs`]; index as there).
pub const DOMAIN_BASE_PCS: &str = "hachi-co/sumcheck/base-pcs";

/// One uniform element of $\mathbb F\_{q^K}$: its $K$ coordinates, each
/// [`uniform::uniform_mod_q`] (exact rejection sampling, in coordinate order).
pub(crate) fn uniform_ext<const K: usize>(prg: &mut ShakePrg, q: u64) -> Result<Ext<K>> {
    let coords = uniform::uniform_mod_q(prg, q, K)?;
    let mut out = [0u64; K];
    out.copy_from_slice(&coords);
    Ok(Ext(out))
}

/// The private randomness of one party (`docs/m1-spec.md`, "Randomness and coins" and
/// `hachi-co-core`).
///
/// Protocol code never samples directly: each party draws its private randomness from a
/// `PartyRandomness` value, [`PrgRandomness`] in real runs and the replay provider of the feature
/// `fixtures` (`ReplayRandomness`) in fixture tests, which serves the Sage oracle's recorded
/// values. The same driver code runs in both cases.
///
/// Every method returns a fresh draw from its distribution. Short vectors are flat coefficient
/// vectors: a vector of $L$ ring elements is given as $L d$ integers, coefficient $j$ of element
/// $i$ at index $i d + j$ (the convention of $\chi\_r$, `4_preliminaries.tex`:117, and of
/// `hachi-co-sampling`'s `rej`).
///
/// [`PrgRandomness`] and `ReplayRandomness` implement the trait for every `K`. Generic code with
/// a bound `R: PartyRandomness<K>` calls `rng.mask(..)` directly; on a concrete value, a method
/// whose signature does not mention `K` (`mask`, `rho_u`, `rho_v`, `base_pcs`) needs it spelled out:
/// `PartyRandomness::<4>::mask(&mut rng, iteration, gamma, len)`.
///
/// Commit draws no shares: since the draft's commit `1096ea2` each party commits to its input
/// share as it is, without the resharing $\mathsf{AS.Share}$ of earlier drafts.
pub trait PartyRandomness<const K: usize> {
    /// The mask $\mathbf r\_i^j \leftarrow \chi\_r$ of iteration `iteration` $= j$ (0-indexed,
    /// the draft's $j + 1$) of the ring-to-linear step (`fig:ring-to-linear`, step 3, since the
    /// draft's commit `1096ea2`; `4_preliminaries.tex`:117): `len` $= N = d\delta 2^m$ integers
    /// uniform in $\[-\gamma, \gamma\]$, flat (element $i$, coefficient $j$ at $id + j$).
    ///
    /// # Errors
    /// [`Error::Sampling`] if $\gamma \gt 2^{63} - 1$, and implementation-specific errors.
    fn mask(&mut self, iteration: u64, gamma: u64, len: usize) -> Result<Vec<i64>>;

    /// The masks of the iterations $0, \dots, $ `count` $- 1$, in order: [`PartyRandomness::mask`]
    /// of each ([`PartyRandomness::masks_in`]).
    ///
    /// # Errors
    /// As [`PartyRandomness::mask`].
    fn masks(&mut self, count: usize, gamma: u64, len: usize) -> Result<Vec<Vec<i64>>> {
        self.masks_in(0..count, gamma, len)
    }

    /// The masks of the iterations in `range`, in order: [`PartyRandomness::mask`] of each. An
    /// implementation whose masks come from separate streams may draw them in parallel; the
    /// default draws them one after the other.
    ///
    /// # Errors
    /// As [`PartyRandomness::mask`].
    fn masks_in(&mut self, range: Range<usize>, gamma: u64, len: usize) -> Result<Vec<Vec<i64>>> {
        range.map(|j| self.mask(j as u64, gamma, len)).collect()
    }

    /// The mask of iteration `iteration` again, the one already served for it, for its response:
    /// the ring-to-linear step forms a mask's images when it draws it and its response after the
    /// challenges, and reads it again then rather than holding the $L$ masks in between. A source
    /// that refuses to serve a stream twice ([`PrgRandomness`]) serves a mask again only once it
    /// was drawn, and still refuses a second fresh draw ([`PartyRandomness::mask`]), so that a
    /// mask never serves two evaluations. The default reads it as [`PartyRandomness::mask`] does,
    /// for sources that serve the same value again.
    ///
    /// # Errors
    /// As [`PartyRandomness::mask`].
    fn mask_again(&mut self, iteration: u64, gamma: u64, len: usize) -> Result<Vec<i64>> {
        self.mask(iteration, gamma, len)
    }

    /// The masks of the iterations in `range` again, in order ([`PartyRandomness::mask_again`]
    /// of each); in parallel where the masks come from separate streams.
    ///
    /// # Errors
    /// As [`PartyRandomness::mask_again`].
    fn masks_again(
        &mut self,
        range: Range<usize>,
        gamma: u64,
        len: usize,
    ) -> Result<Vec<Vec<i64>>> {
        range
            .map(|j| self.mask_again(j as u64, gamma, len))
            .collect()
    }

    /// The commitment's randomness $\boldsymbol\rho^u\_i \leftarrow S\_b^{m\_u}$
    /// (`6_Collaborative_Hachi.tex`:382; $S\_b$ of `4_preliminaries.tex`:88): `len` $= m\_u d$
    /// integers uniform in $\[\lceil -b/2 \rceil, \lceil b/2 \rceil - 1\]$, flat.
    ///
    /// # Errors
    /// [`Error::Sampling`] for $b = 0$, and implementation-specific errors.
    fn rho_u(&mut self, b: u64, len: usize) -> Result<Vec<i64>>;

    /// The first message's randomness $\boldsymbol\rho^v\_i \leftarrow S\_b^{m\_v}$, one for all
    /// $L$ iterations (`fig:ring-to-linear`, step 3; $S\_b$ of `4_preliminaries.tex`:88): `len`
    /// $= m\_v d$ integers uniform in $\[\lceil -b/2 \rceil, \lceil b/2 \rceil - 1\]$, flat.
    ///
    /// # Errors
    /// [`Error::Sampling`] for $b = 0$, and implementation-specific errors.
    fn rho_v(&mut self, b: u64, len: usize) -> Result<Vec<i64>>;

    /// The witness pad $\rho^w\_i \leftarrow \mathbb F\_{q^K}$ (`5_Building_Blocks.tex`:252).
    ///
    /// # Errors
    /// Implementation-specific errors.
    fn rho_w(&mut self, field: &Fqk<K>) -> Result<Ext<K>>;

    /// `len` residues uniform in $\[0, q)$, the pad of [`PadRule::Residues`].
    ///
    /// # Errors
    /// Implementation-specific errors.
    #[doc(hidden)]
    fn rho_w_residues(&mut self, q: u64, len: usize) -> Result<Vec<u64>>;

    /// The coefficients of the blinder $g\_{0,i}$ or $g\_{1,i}$ (`5_Building_Blocks.tex`:238-243):
    /// $s\_0$ and the $\ell'' \times D$ coefficients $s\_{j,k}$, all uniform in $\mathbb F\_{q^K}$,
    /// with $D$ = `degree` ($d\_0$ for [`BlinderKind::G0`], $d\_1$ for [`BlinderKind::G1`]).
    ///
    /// # Errors
    /// Implementation-specific errors.
    fn blinder(
        &mut self,
        field: &Fqk<K>,
        kind: BlinderKind,
        ell_pp: usize,
        degree: usize,
    ) -> Result<Blinder<K>>;

    /// A fresh secret 32-byte seed for the base PCS ([`BasePcs`](crate::BasePcs), "Randomness";
    /// `docs/hachi-hiding.md` 6.5). The sumcheck asks for one per commitment and per opening of each
    /// of its three polynomials; the index distinguishes them (`ClaimedPoly::seed_index` of
    /// `hachi-co-sumcheck`).
    ///
    /// # Errors
    /// Implementation-specific.
    fn base_pcs(&mut self, index: u64) -> Result<[u8; 32]>;
}

/// [`PartyRandomness`] from a [`ShakePrg`] keyed by the party's secret 32-byte seed, one PRG
/// domain per kind of randomness.
///
/// | Method | PRG stream (`seed`, domain, index) | draws, in order |
/// |---|---|---|
/// | `mask` | [`DOMAIN_MASK`], iteration | [`rej::mask`] ($\[-\gamma, \gamma\]$, entry by entry) |
/// | `rho_u` | [`DOMAIN_RHO_U`], 0 | [`uniform::short`] ($S\_b$, entry by entry) |
/// | `rho_v` | [`DOMAIN_RHO_V`], 0 | [`uniform::short`] ($S\_b$, entry by entry) |
/// | `rho_w` | [`DOMAIN_RHO_W`], 0 | one element of $\mathbb F\_{q^K}$, its $K$ coordinates by [`uniform::uniform_mod_q`] |
/// | `blinder` | [`DOMAIN_BLINDER_G0`] or [`DOMAIN_BLINDER_G1`], 0 | $s\_0$, then $s\_{j,k}$ row by row, each element as in `rho_w` |
/// | `base_pcs` | [`DOMAIN_BASE_PCS`], index | the first 32 bytes of the stream |
///
/// **No reuse.** Each stream is served at most once: a second request for the same domain and
/// index is [`Error::RandomnessReuse`], because a reused mask leaks the secret it hides
/// (plan `docs/hachi-co-plan.md`, finding F5). A request that fails (for example with an invalid
/// $\gamma$) does not use up its stream. One `PrgRandomness` therefore serves one Commit and one
/// Eval of one party; use a fresh seed for every further Eval.
///
/// The seed must be secret and uniform; the stream is then pseudorandom under the heuristic that
/// keyed SHAKE128 is a PRF (see [`ShakePrg`]). `Debug` hides the seed, and the type is not
/// `Clone`, so that a key is not duplicated by accident.
pub struct PrgRandomness {
    seed: [u8; 32],
    used: BTreeSet<(&'static str, u64)>,
}

impl core::fmt::Debug for PrgRandomness {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PrgRandomness")
            .field("used", &self.used)
            .finish_non_exhaustive()
    }
}

impl PrgRandomness {
    /// The randomness of the party with secret seed `seed`.
    pub fn new(seed: [u8; 32]) -> Self {
        Self {
            seed,
            used: BTreeSet::new(),
        }
    }

    /// The masks of the streams $(\mathrm{seed}, $ [`DOMAIN_MASK`] $, j)$ for $j$ in `range`, in
    /// parallel, as [`PartyRandomness::mask`] draws each.
    fn mask_streams(&self, range: Range<u64>, gamma: u64, len: usize) -> Result<Vec<Vec<i64>>> {
        let seed = self.seed;
        range
            .into_par_iter()
            .map(|j| {
                let mut prg = ShakePrg::new(&seed, DOMAIN_MASK.as_bytes(), j);
                Ok(rej::mask(&mut prg, gamma, len)?)
            })
            .collect()
    }

    /// Runs `draw` on the stream `(seed, domain, index)` and marks the stream used if it
    /// succeeds.
    fn draw<T>(
        &mut self,
        domain: &'static str,
        index: u64,
        draw: impl FnOnce(&mut ShakePrg) -> Result<T>,
    ) -> Result<T> {
        if self.used.contains(&(domain, index)) {
            return Err(Error::RandomnessReuse {
                kind: domain,
                index,
            });
        }
        let out = draw(&mut ShakePrg::new(&self.seed, domain.as_bytes(), index))?;
        self.used.insert((domain, index));
        Ok(out)
    }
}

impl<const K: usize> PartyRandomness<K> for PrgRandomness {
    fn mask(&mut self, iteration: u64, gamma: u64, len: usize) -> Result<Vec<i64>> {
        self.draw(DOMAIN_MASK, iteration, |prg| {
            Ok(rej::mask(prg, gamma, len)?)
        })
    }

    /// Mask $j$ from its own stream $(\mathrm{seed}, $ [`DOMAIN_MASK`] $, j)$, as
    /// [`PartyRandomness::mask`] draws it, the streams in parallel: the same masks. Every stream
    /// is checked unused first, and all are marked used once all are drawn.
    fn masks_in(&mut self, range: Range<usize>, gamma: u64, len: usize) -> Result<Vec<Vec<i64>>> {
        let range = range.start as u64..range.end as u64;
        if let Some(index) = range
            .clone()
            .find(|j| self.used.contains(&(DOMAIN_MASK, *j)))
        {
            return Err(Error::RandomnessReuse {
                kind: DOMAIN_MASK,
                index,
            });
        }
        let masks = self.mask_streams(range.clone(), gamma, len)?;
        self.used.extend(range.map(|j| (DOMAIN_MASK, j)));
        Ok(masks)
    }

    /// The mask already drawn for `iteration`, from its stream again ([`PrgRandomness`] refuses a
    /// mask never drawn with [`Error::MaskNotDrawn`]).
    fn mask_again(&mut self, iteration: u64, gamma: u64, len: usize) -> Result<Vec<i64>> {
        let j = usize::try_from(iteration).unwrap_or(usize::MAX);
        let mut masks =
            PartyRandomness::<K>::masks_again(self, j..j.saturating_add(1), gamma, len)?;
        Ok(masks.pop().unwrap_or_default())
    }

    /// The masks already drawn for the iterations in `range`, from their streams again, the
    /// streams in parallel; a mask never drawn is refused ([`Error::MaskNotDrawn`]).
    fn masks_again(
        &mut self,
        range: Range<usize>,
        gamma: u64,
        len: usize,
    ) -> Result<Vec<Vec<i64>>> {
        let range = range.start as u64..range.end as u64;
        if let Some(index) = range
            .clone()
            .find(|j| !self.used.contains(&(DOMAIN_MASK, *j)))
        {
            return Err(Error::MaskNotDrawn { index });
        }
        self.mask_streams(range, gamma, len)
    }

    fn rho_u(&mut self, b: u64, len: usize) -> Result<Vec<i64>> {
        self.draw(DOMAIN_RHO_U, 0, |prg| Ok(uniform::short(prg, b, len)?))
    }

    fn rho_v(&mut self, b: u64, len: usize) -> Result<Vec<i64>> {
        self.draw(DOMAIN_RHO_V, 0, |prg| Ok(uniform::short(prg, b, len)?))
    }

    fn rho_w(&mut self, field: &Fqk<K>) -> Result<Ext<K>> {
        let q = field.modulus();
        self.draw(DOMAIN_RHO_W, 0, |prg| uniform_ext::<K>(prg, q))
    }

    fn rho_w_residues(&mut self, q: u64, len: usize) -> Result<Vec<u64>> {
        self.draw(DOMAIN_RHO_W, 0, |prg| {
            Ok(uniform::uniform_mod_q(prg, q, len)?)
        })
    }

    fn blinder(
        &mut self,
        field: &Fqk<K>,
        kind: BlinderKind,
        ell_pp: usize,
        degree: usize,
    ) -> Result<Blinder<K>> {
        let domain = match kind {
            BlinderKind::G0 => DOMAIN_BLINDER_G0,
            BlinderKind::G1 => DOMAIN_BLINDER_G1,
        };
        let q = field.modulus();
        self.draw(domain, 0, |prg| {
            let s0 = uniform_ext::<K>(prg, q)?;
            let s = (0..ell_pp)
                .map(|_| {
                    (0..degree)
                        .map(|_| uniform_ext::<K>(prg, q))
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Blinder { s0, s })
        })
    }

    fn base_pcs(&mut self, index: u64) -> Result<[u8; 32]> {
        self.draw(DOMAIN_BASE_PCS, index, |prg| {
            let mut seed = [0u8; 32];
            prg.fill_bytes(&mut seed);
            Ok(seed)
        })
    }
}
