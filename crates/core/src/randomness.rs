//! A party's private randomness: the trait the protocol code draws from, and its PRG
//! implementation (`docs/m1-spec.md`, "Randomness and coins").

use crate::data::{Blinder, BlinderKind};
use crate::{Error, Result};
use hachi_co_math::{Ext, Fqk};
use hachi_co_sampling::{ShakePrg, rej, uniform};
use std::collections::BTreeSet;

/// The PRG domain of the resharing in Commit ([`PartyRandomness::reshare`]; index 0).
pub const DOMAIN_RESHARE: &str = "hachi-co/commit/reshare";
/// The PRG domain of the masks $\mathbf r\_i$ ([`PartyRandomness::mask`]; index = attempt).
pub const DOMAIN_MASK: &str = "hachi-co/r2l/mask";
/// The PRG domain of the commitment randomness $\boldsymbol\rho\_i$ ([`PartyRandomness::rho`];
/// index = attempt).
pub const DOMAIN_RHO: &str = "hachi-co/r2l/rho";
/// The PRG domain of the blinder $g\_{0,i}$ ([`PartyRandomness::blinder`]; index 0).
pub const DOMAIN_BLINDER_G0: &str = "hachi-co/sumcheck/g0";
/// The PRG domain of the blinder $g\_{1,i}$ ([`PartyRandomness::blinder`]; index 0).
pub const DOMAIN_BLINDER_G1: &str = "hachi-co/sumcheck/g1";

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
/// whose signature does not mention `K` (`mask`, `rho`) needs it spelled out:
/// `PartyRandomness::<4>::mask(&mut rng, attempt, gamma, len)`.
pub trait PartyRandomness<const K: usize> {
    /// $\mathsf{AS.Share}(\mathbf x, n)$ (`4_preliminaries.tex`:154; Commit,
    /// `6_Collaborative_Hachi.tex`:303-305) of a vector $\mathbf x \in \mathbb F\_{q^K}^{len}$:
    /// shares $1, \dots, n-1$ uniform in $\mathbb F\_{q^K}^{len}$ and share $n$ equal to
    /// $\mathbf x - \sum\_{j \lt n} \langle \mathbf x \rangle\_j$. Returns the $n$ shares in order;
    /// share $j$ goes to party $j$. For $n = 1$ the only share is $\mathbf x$.
    ///
    /// # Errors
    /// [`Error::Parameter`] for $n = 0$, and implementation-specific errors.
    fn reshare(&mut self, field: &Fqk<K>, input: &[Ext<K>], n: usize) -> Result<Vec<Vec<Ext<K>>>>;

    /// The mask $\mathbf r\_i \leftarrow \chi\_r$ of attempt `attempt` of the ring-to-linear step
    /// (`5_Building_Blocks.tex`:110; `4_preliminaries.tex`:117): `len` $= N = d\delta 2^m$
    /// integers uniform in $\[-\gamma, \gamma\]$, flat (element $i$, coefficient $j$ at $id + j$).
    ///
    /// # Errors
    /// [`Error::Sampling`] if $\gamma \gt 2^{63} - 1$, and implementation-specific errors.
    fn mask(&mut self, attempt: u64, gamma: u64, len: usize) -> Result<Vec<i64>>;

    /// The commitment randomness $\boldsymbol\rho\_i \leftarrow S\_b^{m\_\rho}$ of attempt
    /// `attempt` (`5_Building_Blocks.tex`:110; $S\_b$ of `4_preliminaries.tex`:88): `len`
    /// $= m\_\rho d$ integers uniform in $\[\lceil -b/2 \rceil, \lceil b/2 \rceil - 1\]$, flat.
    ///
    /// # Errors
    /// [`Error::Sampling`] for $b = 0$, and implementation-specific errors.
    fn rho(&mut self, attempt: u64, b: u64, len: usize) -> Result<Vec<i64>>;

    /// The coefficients of the blinder $g\_{0,i}$ or $g\_{1,i}$ (`5_Building_Blocks.tex`:209-214):
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
}

/// [`PartyRandomness`] from a [`ShakePrg`] keyed by the party's secret 32-byte seed, one PRG
/// domain per kind of randomness.
///
/// | Method | PRG stream (`seed`, domain, index) | draws, in order |
/// |---|---|---|
/// | `reshare` | [`DOMAIN_RESHARE`], 0 | shares $1..n-1$, element by element, each element's $K$ coordinates by [`uniform::uniform_mod_q`] |
/// | `mask` | [`DOMAIN_MASK`], attempt | [`rej::mask`] ($\[-\gamma, \gamma\]$, entry by entry) |
/// | `rho` | [`DOMAIN_RHO`], attempt | [`uniform::short`] ($S\_b$, entry by entry) |
/// | `blinder` | [`DOMAIN_BLINDER_G0`] or [`DOMAIN_BLINDER_G1`], 0 | $s\_0$, then $s\_{j,k}$ row by row, each element as in `reshare` |
///
/// **No reuse.** Each stream is served at most once: a second request for the same domain and
/// index is [`Error::RandomnessReuse`], because a reused mask or share leaks the secret it hides
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
    fn reshare(&mut self, field: &Fqk<K>, input: &[Ext<K>], n: usize) -> Result<Vec<Vec<Ext<K>>>> {
        if n == 0 {
            return Err(Error::Parameter("AS.Share needs n >= 1".into()));
        }
        let q = field.modulus();
        self.draw(DOMAIN_RESHARE, 0, |prg| {
            let mut shares = Vec::with_capacity(n);
            let mut last: Vec<Ext<K>> = input.iter().map(|x| field.reduce(x)).collect();
            for _ in 1..n {
                let share = (0..input.len())
                    .map(|_| uniform_ext::<K>(prg, q))
                    .collect::<Result<Vec<_>>>()?;
                for (l, s) in last.iter_mut().zip(&share) {
                    *l = field.sub(l, s);
                }
                shares.push(share);
            }
            shares.push(last);
            Ok(shares)
        })
    }

    fn mask(&mut self, attempt: u64, gamma: u64, len: usize) -> Result<Vec<i64>> {
        self.draw(DOMAIN_MASK, attempt, |prg| Ok(rej::mask(prg, gamma, len)?))
    }

    fn rho(&mut self, attempt: u64, b: u64, len: usize) -> Result<Vec<i64>> {
        self.draw(DOMAIN_RHO, attempt, |prg| Ok(uniform::short(prg, b, len)?))
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
}
