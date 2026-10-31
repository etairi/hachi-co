//! The values the building blocks pass to each other (`docs/m1-spec.md`, `hachi-co-core`,
//! "Inter-block data"): plain structs with public fields. The block crates do not depend on each
//! other; everything they exchange is one of these types.
//!
//! Each type has a `validate` method that checks its shape (lengths and the number of
//! coefficients of every polynomial) against a [`Layout`]; it does not check values.

use crate::{Context, Error, Layout, Result, SpillFile};
use hachi_co_math::{Ext, Poly, decompose};
use hachi_co_mpc::PartyId;

/// A matrix over a polynomial ring, row-major: `m[i][j]` is entry $(i, j)$
/// (`docs/m1-spec.md`, "Conventions").
pub type Matrix = Vec<Vec<Poly>>;

/// `Err(Shape)` unless `actual == expected`.
pub(crate) fn check_len(what: &'static str, expected: usize, actual: usize) -> Result<()> {
    if actual != expected {
        return Err(Error::Shape {
            what,
            expected,
            actual,
        });
    }
    Ok(())
}

/// `count` polynomials of `coeffs` coefficients each.
pub(crate) fn check_polys(
    what: &'static str,
    polys: &[Poly],
    count: usize,
    coeffs: usize,
) -> Result<()> {
    check_len(what, count, polys.len())?;
    for p in polys {
        check_len(what, coeffs, p.0.len())?;
    }
    Ok(())
}

/// $2^e$ as a `usize`, or a shape error.
pub(crate) fn pow2(e: usize, what: &'static str) -> Result<usize> {
    u32::try_from(e)
        .ok()
        .and_then(|e| 1usize.checked_shl(e))
        .ok_or(Error::Overflow(what))
}

/// The inner commitment of one block $i$ of $\mathbf f'$, computed by the prover holding it in
/// Commit (`6_Collaborative_Hachi.tex`:375-381) and kept in that prover's state (line 400).
///
/// Ring elements are [`Poly`] values with $d$ residues; the digits of $\mathbf s\_i$ and
/// $\hat{\mathbf t}\_i$ (balanced, in $\[\lceil -b/2 \rceil, \lceil b/2 \rceil - 1\]$) are stored as
/// residues modulo $q$, in the element-major gadget layout (`ref/README.md`, "Gadgets").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockState {
    /// The block index $i = \mathrm{nat}(\mathbf i) \in \[0, 2^r)$.
    pub index: usize,
    /// $\mathbf f'\_i \in \mathcal R\_q^{2^m}$: entry $j$ is $f'\[i 2^m + j\]$, the output of
    /// $\mathsf{PConv}$ (`field-to-ring-rok.tex`:105-109).
    pub f_prime: Vec<Poly>,
    /// $\mathbf s\_i = \mathbf G\_{2^m}^{-1}(\mathbf f'\_i) \in \mathcal R\_q^{2^m \delta}$
    /// (`6_Collaborative_Hachi.tex`:378): digit $e$ of entry $j$ at $j\delta + e$.
    pub s: Vec<Poly>,
    /// $\mathbf t\_i = \mathbf A \mathbf s\_i \in \mathcal R\_q^{n\_A}$
    /// (`6_Collaborative_Hachi.tex`:379).
    pub t: Vec<Poly>,
    /// $\hat{\mathbf t}\_i = \mathbf G\_{n\_A}^{-1}(\mathbf t\_i) \in \mathcal R\_q^{n\_A \delta}$
    /// (`6_Collaborative_Hachi.tex`:380): digit $e$ of row $a$ at $a\delta + e$.
    pub t_hat: Vec<Poly>,
}

impl BlockState {
    /// Checks $i \lt 2^r$ and the lengths $2^m$, $2^m\delta$, $n\_A$, $n\_A\delta$ (each polynomial
    /// with $d$ coefficients).
    ///
    /// # Errors
    /// [`Error::Shape`] for the first wrong length.
    pub fn validate(&self, layout: &Layout) -> Result<()> {
        self.validate_commitment(layout)?;
        let d = layout.d;
        check_polys("f'_i", &self.f_prime, layout.block_len, d)?;
        check_polys("s_i", &self.s, layout.width_a, d)
    }

    /// [`BlockState::validate`] without $\mathbf f'\_i$ and $\mathbf s\_i$, which a state on
    /// disk does not hold.
    ///
    /// # Errors
    /// [`Error::Shape`] for the first wrong length.
    pub fn validate_commitment(&self, layout: &Layout) -> Result<()> {
        if self.index >= layout.n_blocks {
            return Err(Error::Shape {
                what: "block index (must be below 2^r)",
                expected: layout.n_blocks,
                actual: self.index,
            });
        }
        let d = layout.d;
        check_polys("t_i", &self.t, layout.n_a, d)?;
        check_polys("t_hat_i", &self.t_hat, layout.n_a * layout.delta, d)
    }
}

/// Prover $p$'s private state after Commit: $\mathsf{st}\_p$ of `6_Collaborative_Hachi.tex`:400
/// (the triples $(\mathbf f'\_i, \mathbf s\_i, \hat{\mathbf t}\_i)$ of the blocks it holds, here
/// with $\mathbf t\_i$ as well) plus its share $\langle \mathbf f \rangle\_p$ of the coefficients.
///
/// A state with [`PartyState::spill`] keeps $\mathbf f'\_i$ on disk: its blocks have empty
/// `f_prime` and `s`, its `share` is empty, and [`PartyState::for_each_block`] is the way to read
/// them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartyState<const K: usize> {
    /// The prover $p \in \lbrace 1, \dots, n \rbrace$.
    pub party: PartyId,
    /// $\langle \mathbf f \rangle\_p = \sum\_i \langle \mathbf f\_i \rangle\_p \in
    /// \mathbb F\_{q^k}^{2^\ell}$, its share after the resharing (`6_Collaborative_Hachi.tex`:
    /// 308-313), coefficient $\mathrm{nat}(\mathbf q)$ at index $\mathrm{nat}(\mathbf q)$.
    pub share: Vec<Ext<K>>,
    /// The blocks $p$ holds, [`Layout::blocks_of`]`(p)`, in increasing order of index (none for
    /// the unowned zero blocks, `ref/README.md`, "Blocks and owners").
    pub blocks: Vec<BlockState>,
    /// $\boldsymbol\rho^u\_p \in S\_b^{m\_u}$ (`6_Collaborative_Hachi.tex`:382, 400): $m\_u$
    /// ring elements of residues. Every evaluation of this commitment reuses it
    /// (`t-hiding.tex`:13).
    pub rho_u: Vec<Poly>,
    /// The file that holds $\mathbf f'\_i$ of the blocks, in order, if they are not in memory.
    pub spill: Option<SpillFile>,
}

impl<const K: usize> PartyState<K> {
    /// Checks that $p \in \lbrace 1, \dots, n \rbrace$, that the share has $2^\ell$ entries,
    /// that the blocks are exactly [`Layout::blocks_of`]`(p)`, in order, each valid, and that
    /// $\boldsymbol\rho^u\_p$ has $m\_u$ entries; for a state on disk, the same without the
    /// share, $\mathbf f'\_i$ and $\mathbf s\_i$.
    ///
    /// # Errors
    /// [`Error::InvalidParty`], [`Error::Shape`] or [`Error::Overflow`].
    pub fn validate(&self, layout: &Layout) -> Result<()> {
        if self.party == 0 || self.party > layout.n {
            return Err(Error::InvalidParty {
                party: self.party,
                n: layout.n,
            });
        }
        if self.spill.is_none() {
            check_len("<f>_p", pow2(layout.ell, "2^l")?, self.share.len())?;
        }
        let owned = layout.blocks_of(self.party);
        check_len("blocks of the state", owned.len(), self.blocks.len())?;
        for (block, i) in self.blocks.iter().zip(owned) {
            if block.index != i {
                return Err(Error::Shape {
                    what: "block index in the state (blocks_of(p) in order)",
                    expected: i,
                    actual: block.index,
                });
            }
            if self.spill.is_none() {
                block.validate(layout)?;
            } else {
                block.validate_commitment(layout)?;
            }
        }
        check_polys("rho^u_p", &self.rho_u, layout.m_u, layout.d)
    }

    /// Calls `visit` on every block, in order, with $\mathbf f'\_i$ and $\mathbf s\_i$: the
    /// blocks themselves, or for a state on disk each block read back and decomposed, one at a
    /// time.
    ///
    /// # Errors
    /// The first error of `visit`; [`Error::Spill`] or [`Error::Math`] for a state on disk.
    pub fn for_each_block<E: From<Error>>(
        &self,
        ctx: &Context<K>,
        mut visit: impl FnMut(&BlockState) -> core::result::Result<(), E>,
    ) -> core::result::Result<(), E> {
        let Some(file) = &self.spill else {
            return self.blocks.iter().try_for_each(visit);
        };
        for (slot, block) in self.blocks.iter().enumerate() {
            let f_prime = file.read_block(slot)?;
            let s = decompose::g_inv(&ctx.ring, ctx.b(), ctx.layout.delta, &f_prime)
                .map_err(Error::from)?;
            visit(&BlockState {
                index: block.index,
                f_prime,
                s,
                t: block.t.clone(),
                t_hat: block.t_hat.clone(),
            })?;
        }
        Ok(())
    }
}

/// The public vectors of $\Pi\_{\mathsf{conv}}$ (`field-to-ring-rok.tex`:140-151), computed by
/// every prover and by the verifier from the evaluation point $\mathbf x$ (`ref/README.md`,
/// "Πconv").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicVectors<const K: usize> {
    /// $\mathbf x' = \mathbf 1^{\lceil \log n \rceil} \Vert (x\_1, \dots, x\_{\ell - \alpha +
    /// \kappa}) \in \mathbb F\_{q^k}^{\ell'}$ (line 142).
    pub x_prime: Vec<Ext<K>>,
    /// $\mathbf b = (\prod\_{j \le r} {x'\_j}^{i\_j})\_{\mathbf i \in \lbrace 0,1 \rbrace^r} \in
    /// \mathbb F\_{q^k}^{2^r}$, indexed by $\mathrm{nat}(\mathbf i)$ (line 143).
    pub b: Vec<Ext<K>>,
    /// $\mathbf a = (\prod\_{j \le m} {x'\_{r+j}}^{j\_j})\_{\mathbf j \in \lbrace 0,1 \rbrace^m} \in
    /// \mathbb F\_{q^k}^{2^m}$ (line 144).
    pub a: Vec<Ext<K>>,
    /// $\mathbf d$ with $d\[\mathrm{nat}(\mathbf i) 2^m + \mathrm{nat}(\mathbf j)\] = b\_{\mathbf i}
    /// a\_{\mathbf j}$, in $\mathbb F\_{q^k}^{2^{\ell'}}$ (lines 146-150).
    pub d: Vec<Ext<K>>,
    /// $v = \psi\bigl((\prod\_j x\_{\ell - \alpha + \kappa + j}^{k\_j})\_{\mathbf k}\bigr) \in
    /// \mathcal R\_q$ (line 145), the point of $E$.
    pub v_point: Poly,
    /// $\mathbf b$ embedded in $\mathcal R\_q$ ([`hachi_co_math::Subfield::embed`]), as it enters
    /// the row $\mathbf b^T \mathbf G\_{2^r}$ of $\mathbf M$ (`5_Building_Blocks.tex`:173).
    pub b_ring: Vec<Poly>,
    /// $\mathbf a$ embedded in $\mathcal R\_q$, as in $w\_i = \mathbf a^T \mathbf G\_{2^m}
    /// \mathbf s\_i$ (`5_Building_Blocks.tex`:118) and the $\mathbf c$ row of $\mathbf M$ (line 174).
    pub a_ring: Vec<Poly>,
}

impl<const K: usize> PublicVectors<K> {
    /// Checks the lengths $\ell'$, $2^r$, $2^m$, $2^{\ell'}$ and the polynomials' $d$ coefficients.
    ///
    /// # Errors
    /// [`Error::Shape`] or [`Error::Overflow`].
    pub fn validate(&self, layout: &Layout) -> Result<()> {
        check_len("x'", layout.ell_prime, self.x_prime.len())?;
        check_len("b", layout.n_blocks, self.b.len())?;
        check_len("a", layout.block_len, self.a.len())?;
        check_len("d", pow2(layout.ell_prime, "2^l'")?, self.d.len())?;
        check_len("v", layout.d, self.v_point.0.len())?;
        check_polys("b (ring)", &self.b_ring, layout.n_blocks, layout.d)?;
        check_polys("a (ring)", &self.a_ring, layout.block_len, layout.d)
    }
}

/// Prover $p$'s shares of the linear relation $\mathbf M \mathbf z' = \mathbf y$ after the
/// ring-to-linear step (`5_Building_Blocks.tex`:167-196): $\langle \mathbf z' \rangle\_p$ and
/// $\lbrace \mathbf y \rbrace\_p$, with $\sum\_p \langle \mathbf z' \rangle\_p = \mathbf z'$ and
/// $\sum\_p \lbrace \mathbf y \rbrace\_p = \mathbf y$.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearShare {
    /// $\langle \mathbf z' \rangle\_p \in \mathcal R\_q^{\mu - 1}$: equal to $\mathbf z'$ on
    /// $S\_p$ ([`Layout::s_ranges`]) and $0$ elsewhere, except position $\mu - 2$, which holds
    /// $\langle u \rangle\_p = \mathbf d^T \lbrace \mathbf f' \rbrace\_p$ (`ref/README.md`,
    /// "Shares").
    pub z_prime: Vec<Poly>,
    /// $\lbrace \mathbf y \rbrace\_p \in \mathcal R\_q^{n'}$: $\mathbf y = (\mathbf v, \mathbf u,
    /// 0, 0, \mathbf 0)$ for $p = 1$ and $\mathbf 0$ for $p \gt 1$ (`5_Building_Blocks.tex`:147,
    /// 173-179).
    pub y: Vec<Poly>,
}

impl LinearShare {
    /// The all-zero share: $\mu - 1$ and $n'$ zero polynomials of $d$ coefficients.
    pub fn zero(layout: &Layout) -> Self {
        Self {
            z_prime: vec![Poly(vec![0; layout.d]); layout.len_z],
            y: vec![Poly(vec![0; layout.d]); layout.n_prime],
        }
    }

    /// Party $p$'s shares of a whole witness: $\mathsf{Slice}$ (`4_preliminaries.tex`:167-177) of
    /// $\mathbf z'$ by $S\_1, \dots, S\_n$ ([`Layout::s_ranges`]), with $\langle u \rangle\_p$ =
    /// `u_share` at position $\mu - 2$, and $\lbrace \mathbf y \rbrace\_p = \mathbf y$ for $p = 1$,
    /// $\mathbf 0$ otherwise (`ref/README.md`, "Shares"). Entry $\mu - 2$ of `z_prime` (the whole
    /// $u$) is not used. Summed over $p$, with $\sum\_p \langle u \rangle\_p = u$, the shares give
    /// back $(\mathbf z', \mathbf y)$ when $\mathbf z'$ is zero on the unheld positions
    /// ([`Layout::unheld`]). For tests that split a recorded witness into shares; the
    /// ring-to-linear step assembles each share from the party's own values instead.
    ///
    /// # Errors
    /// [`Error::InvalidParty`] for $p \notin \lbrace 1, \dots, n \rbrace$; [`Error::Shape`] unless
    /// `z_prime` has $\mu - 1$ and `y` $n'$ polynomials and every polynomial $d$ coefficients.
    pub fn slice(
        layout: &Layout,
        z_prime: &[Poly],
        y: &[Poly],
        u_share: &Poly,
        party: PartyId,
    ) -> Result<Self> {
        if party == 0 || party > layout.n {
            return Err(Error::InvalidParty { party, n: layout.n });
        }
        check_polys("z'", z_prime, layout.len_z, layout.d)?;
        check_polys("y", y, layout.n_prime, layout.d)?;
        check_len("<u>_p", layout.d, u_share.0.len())?;
        let mut share = Self::zero(layout);
        for r in layout.s_ranges(party) {
            share.z_prime[r.clone()].clone_from_slice(&z_prime[r]);
        }
        share.z_prime[layout.u] = u_share.clone();
        if party == 1 {
            share.y = y.to_vec();
        }
        Ok(share)
    }

    /// Checks the lengths $\mu - 1$ and $n'$ and the polynomials' $d$ coefficients.
    ///
    /// # Errors
    /// [`Error::Shape`].
    pub fn validate(&self, layout: &Layout) -> Result<()> {
        check_polys("<z'>_p", &self.z_prime, layout.len_z, layout.d)?;
        check_polys("{y}_p", &self.y, layout.n_prime, layout.d)
    }
}

/// Which blinding polynomial of `5_Building_Blocks.tex`:238-243 is meant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlinderKind {
    /// $g\_{0,p}$, of individual degree $d\_0$, which blinds $F\_{0,p}$ (line 240).
    G0,
    /// $g\_{1,p}$, of individual degree $d\_1$, which blinds $F\_{1,p}$ (line 241).
    G1,
}

impl BlinderKind {
    /// The individual degree $D$ of the blinder: $d\_0$ for [`BlinderKind::G0`], $d\_1$ for
    /// [`BlinderKind::G1`] (`6_Collaborative_Hachi.tex`:285).
    pub fn degree(self, layout: &Layout) -> usize {
        match self {
            Self::G0 => layout.d0,
            Self::G1 => layout.d1,
        }
    }
}

/// The coefficients of a blinding polynomial (`5_Building_Blocks.tex`:238-243; `ref/README.md`,
/// "Blinders"): $g(V) = s\_0 + \sum\_{j=1}^{\ell''} \sum\_{k=1}^{D} s\_{j,k} V\_j^k$ with uniform
/// $s\_0, s\_{j,k} \in \mathbb F\_{q^k}$. Only the coefficients live here; the hypercube sum, round
/// polynomials and evaluations are in `hachi-co-sumcheck`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Blinder<const K: usize> {
    /// $s\_0$.
    pub s0: Ext<K>,
    /// `s[j - 1][k - 1]` $= s\_{j,k}$: $\ell''$ rows of $D$ coefficients.
    pub s: Vec<Vec<Ext<K>>>,
}

impl<const K: usize> Blinder<K> {
    /// Checks the shape $\ell'' \times D$.
    ///
    /// # Errors
    /// [`Error::Shape`].
    pub fn validate(&self, ell_pp: usize, degree: usize) -> Result<()> {
        check_len("blinder rows (l'')", ell_pp, self.s.len())?;
        for row in &self.s {
            check_len("blinder row (degree)", degree, row.len())?;
        }
        Ok(())
    }
}

/// The verifier's coins of the sumcheck before the rounds (`5_Building_Blocks.tex`:275-282),
/// uniform in $\mathbb F\_{q^k}$.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckCoins<const K: usize> {
    /// $\alpha$, the point at which the entries of $\mathbf M$ and $\mathbf y$ are evaluated.
    pub alpha: Ext<K>,
    /// $\rho$, the weight of the blinders in $H\_{0,p}$ and $H\_{1,p}$.
    pub rho: Ext<K>,
    /// $\tau\_0 \in \mathbb F\_{q^k}^{\ell'' + \lceil \log n \rceil}$: $\tau\_0\[0..\ell'')$ pairs
    /// with $(U, L)$ and $\tau\_0\[\ell''..)$ with the $\lceil \log n \rceil$ bits of $p - 1$
    /// (`ref/README.md`, "eq").
    pub tau0: Vec<Ext<K>>,
    /// $\tau\_1 \in \mathbb F\_{q^k}^{\lceil \log \nu \rceil}$.
    pub tau1: Vec<Ext<K>>,
}

impl<const K: usize> SumcheckCoins<K> {
    /// Checks the lengths $\ell'' + \lceil \log n \rceil$ and $\lceil \log \nu \rceil$, and
    /// that every coordinate of every coin is a canonical residue below $q$ (a challenger could
    /// otherwise hand the provers and the verifier different representatives of one coin).
    ///
    /// # Errors
    /// [`Error::Shape`]; [`Error::NonCanonical`].
    pub fn validate(&self, layout: &Layout, q: u64) -> Result<()> {
        check_len("tau_0", layout.ell_pp + layout.log_n, self.tau0.len())?;
        check_len("tau_1", layout.log_nu, self.tau1.len())?;
        let coins = [
            ("alpha", core::slice::from_ref(&self.alpha)),
            ("rho", core::slice::from_ref(&self.rho)),
            ("tau_0", self.tau0.as_slice()),
            ("tau_1", self.tau1.as_slice()),
        ];
        for (what, values) in coins {
            if let Some(&value) = values.iter().flat_map(|e| e.0.iter()).find(|&&x| x >= q) {
                return Err(Error::NonCanonical { what, value, q });
            }
        }
        Ok(())
    }
}
