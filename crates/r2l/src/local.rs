//! The party-local steps of the ring-to-linear step (`5_Building_Blocks.tex`:103-130), as pure
//! functions of the party's data: $w\_i$ and $\hat{\mathbf w}\_i$ of the owned blocks, the mask
//! and $\boldsymbol\rho$ of an attempt with their images, the input to
//! $\mathcal F\_{\mathsf{AH}}(\mathbf D)$, the check $\mathbf c \in \mathcal C^{2^r}$, the integer
//! fold and $\mathsf{Rej}$.
//!
//! Short vectors are flat (`docs/m1-spec.md`, "Conventions"): a vector of $L$ ring elements is
//! $Ld$ integers, coefficient $j$ of element $i$ at $id + j$; masks, folds and responses have
//! $N = d\delta 2^m$ entries (`6_Collaborative_Hachi.tex`:275).

use crate::ct::{lift_polys, wipe_polys};
use crate::{Error, Result};
use hachi_co_core::short::center_poly;
use hachi_co_core::{BlockState, Context, PartyId, PartyRandomness, PartyState, PublicVectors};
use hachi_co_math::{Poly, decompose};
use hachi_co_sampling::{l1ball, rej};

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

/// `count` polynomials of $d$ canonical residues each.
pub(crate) fn check_polys(
    what: &'static str,
    polys: &[Poly],
    count: usize,
    d: usize,
    q: u64,
) -> Result<()> {
    check_len(what, count, polys.len())?;
    for p in polys {
        check_len(what, d, p.0.len())?;
        if let Some(value) = p.0.iter().copied().find(|x| *x >= q) {
            return Err(Error::NonCanonical { what, value, q });
        }
    }
    Ok(())
}

/// `Err(InvalidParty)` unless $1 \le p \le n$.
pub(crate) fn check_party<const K: usize>(ctx: &Context<K>, party: PartyId) -> Result<()> {
    let n = ctx.layout.n;
    if party == 0 || party > n {
        return Err(Error::InvalidParty { party, n });
    }
    Ok(())
}

/// $N = d\delta 2^m$, the number of integer coefficients of a mask, a fold and a response
/// (`6_Collaborative_Hachi.tex`:275).
pub fn response_len<const K: usize>(ctx: &Context<K>) -> usize {
    ctx.layout.width_a * ctx.layout.d
}

/// $\mathbf a^T \mathbf G\_{2^m} \mathbf x \in \mathcal R\_q$ for $\mathbf x \in
/// \mathcal R\_q^{\delta 2^m}$ in the element-major gadget layout: the map of
/// `5_Building_Blocks.tex`:106 ($\mathbf x = \mathbf s\_i$) and 113 ($\mathbf x = \mathbf r\_i$);
/// the oracle's `a_times_G`. $\mathbf G\_{2^m}\mathbf x$ is formed with
/// [`decompose::g_apply`] and paired with `a_ring` ($2^m$ entries,
/// [`PublicVectors::a_ring`]) by an exact inner product.
///
/// # Errors
/// [`Error::Shape`] for inputs of the wrong length; [`Error::Math`] for a polynomial without $d$
/// coefficients.
pub fn a_times_g<const K: usize>(ctx: &Context<K>, a_ring: &[Poly], x: &[Poly]) -> Result<Poly> {
    let l = &ctx.layout;
    check_len("a (ring)", l.block_len, a_ring.len())?;
    check_len("x of a^T G x", l.width_a, x.len())?;
    let g = decompose::g_apply(&ctx.ring, ctx.b(), l.delta, x)?;
    Ok(ctx.ring.inner(a_ring, &g)?)
}

/// $w\_i$ and $\hat{\mathbf w}\_i$ of one block (`5_Building_Blocks.tex`:103-108).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockW {
    /// The block index $i = \mathrm{nat}(\mathbf i) \lt 2^r$.
    pub index: usize,
    /// $w\_i = \mathbf a^T \mathbf G\_{2^m} \mathbf s\_i \in \mathcal R\_q$ (line 106).
    pub w: Poly,
    /// $\hat{\mathbf w}\_i = \mathbf G\_1^{-1}(w\_i) \in \mathcal R\_q^\delta$ (line 107): digit
    /// $e$ at index $e$, as residues.
    pub w_hat: Vec<Poly>,
}

/// $w\_i$ and $\hat{\mathbf w}\_i$ of one block of the prover holding it
/// (`5_Building_Blocks.tex`:103-108): [`a_times_g`] of $\mathbf s\_i$ and [`decompose::g_inv`]
/// (the representative rule of `ref/README.md`, choice 2).
///
/// # Errors
/// [`Error::Core`] if the block or the public vectors have the wrong shape; as [`a_times_g`] and
/// [`decompose::g_inv`].
pub fn block_w<const K: usize>(
    ctx: &Context<K>,
    pv: &PublicVectors<K>,
    block: &BlockState,
) -> Result<BlockW> {
    block.validate(&ctx.layout)?;
    pv.validate(&ctx.layout)?;
    let w = a_times_g(ctx, &pv.a_ring, &block.s)?;
    let w_hat = decompose::g_inv(
        &ctx.ring,
        ctx.b(),
        ctx.layout.delta,
        core::slice::from_ref(&w),
    )?;
    Ok(BlockW {
        index: block.index,
        w,
        w_hat,
    })
}

/// Step 1 for prover $p$ (`5_Building_Blocks.tex`:103-108): [`block_w`] of every block it holds,
/// in increasing order of index ([`hachi_co_core::Layout::blocks_of`]).
///
/// # Errors
/// [`Error::Core`] if the state or the public vectors have the wrong shape; as [`block_w`].
pub fn owned_w<const K: usize>(
    ctx: &Context<K>,
    pv: &PublicVectors<K>,
    state: &PartyState<K>,
) -> Result<Vec<BlockW>> {
    state.validate(&ctx.layout)?;
    state.blocks.iter().map(|b| block_w(ctx, pv, b)).collect()
}

/// The images of a mask $\mathbf r\_p$ (`5_Building_Blocks.tex`:111-114).
#[derive(Clone, PartialEq, Eq)]
pub struct MaskImages {
    /// $\mathbf t\_{0,p} = \mathbf A \mathbf r\_p \in \mathcal R\_q^{n\_A}$.
    pub t0: Vec<Poly>,
    /// $\hat{\mathbf t}\_{0,p} = \mathbf G\_{n\_A}^{-1}(\mathbf t\_{0,p}) \in
    /// \mathcal R\_q^{n\_A\delta}$: digit $e$ of row $a$ at $a\delta + e$.
    pub t0_hat: Vec<Poly>,
    /// $w\_{0,p} = \mathbf a^T \mathbf G\_{2^m} \mathbf r\_p \in \mathcal R\_q$.
    pub w0: Poly,
    /// $\hat{\mathbf w}\_{0,p} = \mathbf G\_1^{-1}(w\_{0,p}) \in \mathcal R\_q^\delta$.
    pub w0_hat: Vec<Poly>,
}

/// The images are functions of the secret mask: `Debug` shows their shapes only.
impl core::fmt::Debug for MaskImages {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MaskImages")
            .field("t0", &format_args!("[{} polynomials]", self.t0.len()))
            .field(
                "t0_hat",
                &format_args!("[{} polynomials]", self.t0_hat.len()),
            )
            .field(
                "w0_hat",
                &format_args!("[{} polynomials]", self.w0_hat.len()),
            )
            .finish_non_exhaustive()
    }
}

impl MaskImages {
    /// Checks the lengths $n\_A$, $n\_A\delta$, $d$, $\delta$ and canonical residues.
    pub(crate) fn validate<const K: usize>(&self, ctx: &Context<K>) -> Result<()> {
        let l = &ctx.layout;
        let q = ctx.q();
        check_polys("t_0", &self.t0, l.n_a, l.d, q)?;
        check_polys("t0_hat", &self.t0_hat, l.n_a * l.delta, l.d, q)?;
        check_polys("w_0", core::slice::from_ref(&self.w0), 1, l.d, q)?;
        check_polys("w0_hat", &self.w0_hat, l.delta, l.d, q)
    }
}

/// $\mathbf t\_{0,p}$, $\hat{\mathbf t}\_{0,p}$, $w\_{0,p}$ and $\hat{\mathbf w}\_{0,p}$ of the
/// mask $\mathbf r\_p$ (`5_Building_Blocks.tex`:111-114; the oracle's `ring_to_linear`, step 2).
///
/// `mask` is flat ($N$ integers). The mask is first checked to lie in $\[-\gamma, \gamma\]^N$
/// and lifted to residues, both without branches on its values (best effort); the products by
/// $\mathbf A$ and $\mathbf a^T \mathbf G\_{2^m}$ and the decompositions $\mathbf G^{-1}$ then run
/// in `hachi-co-math`, which is not constant time (see the crate documentation). The lifted copy
/// is wiped before returning.
///
/// # Errors
/// [`Error::Shape`] unless `mask` has $N$ entries; [`Error::Sampling`]
/// ([`MaskOutOfRange`](hachi_co_sampling::Error::MaskOutOfRange)) for a coefficient outside
/// $\[-\gamma, \gamma\]$; [`Error::Parameter`] if $\gamma \ge q$; [`Error::Core`] if the public
/// vectors have the wrong shape.
pub fn mask_images<const K: usize>(
    ctx: &Context<K>,
    pv: &PublicVectors<K>,
    mask: &[i64],
) -> Result<MaskImages> {
    let l = &ctx.layout;
    pv.validate(l)?;
    check_len("mask r_p (N = d delta 2^m)", response_len(ctx), mask.len())?;
    let mut r = lift_polys(mask, l.d, ctx.q(), ctx.gamma(), || {
        hachi_co_sampling::Error::MaskOutOfRange.into()
    })?;
    let images = (|| {
        let t0 = ctx.ring.mat_vec(&ctx.pp.a, &r)?;
        let t0_hat = decompose::g_inv(&ctx.ring, ctx.b(), l.delta, &t0)?;
        let w0 = a_times_g(ctx, &pv.a_ring, &r)?;
        let w0_hat = decompose::g_inv(&ctx.ring, ctx.b(), l.delta, core::slice::from_ref(&w0))?;
        Ok(MaskImages {
            t0,
            t0_hat,
            w0,
            w0_hat,
        })
    })();
    wipe_polys(&mut r);
    images
}

/// Prover $p$'s private values of one attempt (`5_Building_Blocks.tex`:110-114).
///
/// All of it is secret. The mask must never be revealed or reused (plan
/// `docs/hachi-co-plan.md`, finding F5); $\boldsymbol\rho\_p$ and the images enter $p$'s share of
/// $\mathbf z'$ if the attempt is accepted (as copies). `Debug` shows only the attempt index; the
/// type is neither `Clone` nor comparable, so that the mask is not copied or compared by
/// accident, and dropping it wipes the mask, $\boldsymbol\rho\_p$ and the images (best effort, see
/// the crate documentation), also when an error ends the attempt early.
pub struct AttemptSecrets {
    /// The attempt index $a = 0, 1, \dots$ (`ref/README.md`, choice 5).
    pub attempt: u64,
    /// The mask $\mathbf r\_p \leftarrow \chi\_r$, uniform in $\[-\gamma, \gamma\]^N$, flat
    /// (line 110).
    pub mask: Vec<i64>,
    /// $\boldsymbol\rho\_p \leftarrow S\_b^{m\_\rho}$, flat ($m\_\rho d$ integers in
    /// $\[\lceil -b/2 \rceil, \lceil b/2 \rceil - 1\]$, line 110).
    pub rho: Vec<i64>,
    /// The images of the mask ([`mask_images`]).
    pub images: MaskImages,
}

impl core::fmt::Debug for AttemptSecrets {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AttemptSecrets")
            .field("attempt", &self.attempt)
            .finish_non_exhaustive()
    }
}

impl Drop for AttemptSecrets {
    fn drop(&mut self) {
        self.wipe_mask();
        crate::ct::wipe(&mut self.rho);
        let images = &mut self.images;
        wipe_polys(&mut images.t0);
        wipe_polys(&mut images.t0_hat);
        wipe_polys(core::slice::from_mut(&mut images.w0));
        wipe_polys(&mut images.w0_hat);
    }
}

impl AttemptSecrets {
    /// Overwrites the mask with zeros (best effort, see the crate documentation). The driver
    /// calls it as soon as the mask is no longer needed.
    pub fn wipe_mask(&mut self) {
        crate::ct::wipe(&mut self.mask);
    }

    /// $\boldsymbol\rho\_p$ as $m\_\rho$ ring elements with residues in $\[0, q)$ (checked to lie
    /// in $S\_b$).
    pub(crate) fn rho_polys<const K: usize>(&self, ctx: &Context<K>) -> Result<Vec<Poly>> {
        let l = &ctx.layout;
        check_len("rho_p (m_rho d)", l.m_rho * l.d, self.rho.len())?;
        let b = ctx.b();
        // S_b = [ceil(-b/2), ceil(b/2) - 1] = [-floor(b/2), floor((b-1)/2)] (4_preliminaries.tex:88).
        let (min, max) = (-((b / 2) as i64), ((b - 1) / 2) as i64);
        let out_of_range = || Error::OutOfRange {
            what: "rho_p (S_b)",
            min,
            max,
        };
        // Check the asymmetric range exactly with one combined flag, then lift.
        let mut bad = 0u8;
        for x in &self.rho {
            bad |= u8::from(*x < min) | u8::from(*x > max);
        }
        if bad != 0 {
            return Err(out_of_range());
        }
        lift_polys(&self.rho, l.d, ctx.q(), b / 2, out_of_range)
    }
}

/// Step 2 for one attempt (`5_Building_Blocks.tex`:110-114): draws the mask
/// $\mathbf r\_p \leftarrow \chi\_r$ ([`PartyRandomness::mask`], $N$ entries, width $\gamma$)
/// and $\boldsymbol\rho\_p \leftarrow S\_b^{m\_\rho}$ ([`PartyRandomness::rho`], $m\_\rho d$
/// entries), both with index `attempt`, checks what the randomness source returned (lengths and
/// ranges), and computes the [`mask_images`].
///
/// # Errors
/// The errors of the randomness source ([`Error::Core`], e.g.
/// [`RandomnessReuse`](hachi_co_core::Error::RandomnessReuse) for an attempt drawn twice from a
/// [`hachi_co_core::PrgRandomness`]); [`Error::Shape`] or [`Error::OutOfRange`] if it returned
/// malformed values; as [`mask_images`].
pub fn sample_attempt<const K: usize, R: PartyRandomness<K>>(
    ctx: &Context<K>,
    pv: &PublicVectors<K>,
    rng: &mut R,
    attempt: u64,
) -> Result<AttemptSecrets> {
    let l = &ctx.layout;
    let mask = rng.mask(attempt, ctx.gamma(), response_len(ctx))?;
    let mut secrets = AttemptSecrets {
        attempt,
        mask,
        rho: Vec::new(),
        images: MaskImages {
            t0: Vec::new(),
            t0_hat: Vec::new(),
            w0: Poly::default(),
            w0_hat: Vec::new(),
        },
    };
    // On an error below, dropping `secrets` wipes what was drawn.
    check_len(
        "mask r_p (N = d delta 2^m)",
        response_len(ctx),
        secrets.mask.len(),
    )?;
    secrets.rho = rng.rho(attempt, ctx.b(), l.m_rho * l.d)?;
    secrets.rho_polys(ctx)?;
    secrets.images = mask_images(ctx, pv, &secrets.mask)?;
    Ok(secrets)
}

/// Writes prover $p$'s values of $(\hat{\mathbf w}, \hat{\mathbf w}\_0, \hat{\mathbf t}\_0,
/// \boldsymbol\rho)$ into `out` $= \mathbf z'\[0, t)$ (positions of [`hachi_co_core::Layout`]):
/// its blocks of $\hat{\mathbf w}$, its own $\hat{\mathbf w}\_{0,p}$, $\hat{\mathbf t}\_{0,p}$
/// and $\boldsymbol\rho\_p$; other entries are left untouched.
pub(crate) fn fill_first_message_part<const K: usize>(
    ctx: &Context<K>,
    party: PartyId,
    w: &[BlockW],
    secrets: &AttemptSecrets,
    out: &mut [Poly],
) -> Result<()> {
    let l = &ctx.layout;
    check_party(ctx, party)?;
    check_len("z'[0, t)", l.t, out.len())?;
    let owned = l.blocks_of(party);
    check_len("w of the owned blocks", owned.len(), w.len())?;
    for (bw, i) in w.iter().zip(owned) {
        if bw.index != i {
            return Err(Error::Shape {
                what: "block index of w (blocks_of(p) in order)",
                expected: i,
                actual: bw.index,
            });
        }
        check_polys("w_hat_i", &bw.w_hat, l.delta, l.d, ctx.q())?;
        out[l.w_hat_range(i)].clone_from_slice(&bw.w_hat);
    }
    secrets.images.validate(ctx)?;
    out[l.w0_hat_range(party)].clone_from_slice(&secrets.images.w0_hat);
    out[l.t0_hat_range(party)].clone_from_slice(&secrets.images.t0_hat);
    let rho = secrets.rho_polys(ctx)?;
    out[l.rho_range(party)].clone_from_slice(&rho);
    Ok(())
}

/// Prover $p$'s slice $\lbrace (\hat{\mathbf w}, \hat{\mathbf w}\_0, \hat{\mathbf t}\_0,
/// \boldsymbol\rho) \rbrace\_p \in \mathcal R\_q^{t}$ (`5_Building_Blocks.tex`:115-121;
/// $\mathsf{Slice}$ of `4_preliminaries.tex`:167-177): the values $p$ holds at their positions of
/// $\mathbf z'\[0, t)$ and zero elsewhere. It equals the first $t$ entries of $p$'s share of
/// $\mathbf z'$ ([`crate::z_prime_share`]).
///
/// # Errors
/// [`Error::InvalidParty`]; [`Error::Shape`] unless `w` holds the blocks of $p$ in order with
/// $\delta$ digits each; [`Error::Shape`], [`Error::NonCanonical`] or [`Error::OutOfRange`] for
/// malformed `secrets`.
pub fn d_slice<const K: usize>(
    ctx: &Context<K>,
    party: PartyId,
    w: &[BlockW],
    secrets: &AttemptSecrets,
) -> Result<Vec<Poly>> {
    let mut slice = vec![ctx.ring.zero(); ctx.layout.t];
    fill_first_message_part(ctx, party, w, secrets, &mut slice)?;
    Ok(slice)
}

/// Prover $p$'s input to $\mathcal F\_{\mathsf{AH}}(\mathbf D)$ (`5_Building_Blocks.tex`:119-121):
/// $\mathbf D \cdot \lbrace (\hat{\mathbf w}, \hat{\mathbf w}\_0, \hat{\mathbf t}\_0,
/// \boldsymbol\rho) \rbrace\_p \in \mathcal R\_q^{n\_D}$ for the slice of [`d_slice`]. The sum of
/// the provers' inputs is the first message
/// $\mathbf v = \mathbf D\_{\hat{\mathbf w}} \hat{\mathbf w} + \mathbf D\_0 \hat{\mathbf w}\_0 +
/// \mathbf D\_1 \hat{\mathbf t}\_0 + \mathbf D\_\rho \boldsymbol\rho$ (line 120).
///
/// # Errors
/// [`Error::Shape`] unless the slice has $t$ entries; [`Error::Math`] for a polynomial without
/// $d$ coefficients.
pub fn fah_d_input<const K: usize>(ctx: &Context<K>, slice: &[Poly]) -> Result<Vec<Poly>> {
    check_len(
        "slice of (w_hat, w0_hat, t0_hat, rho)",
        ctx.layout.t,
        slice.len(),
    )?;
    Ok(ctx.ring.mat_vec(&ctx.pp.d, slice)?)
}

/// Checks $\mathbf c \in \mathcal C^{2^r}$ (`5_Building_Blocks.tex`:122-126): $2^r$ challenges
/// of $d$ signed coefficients each, with $\Vert c\_i \Vert\_1 \le \omega$, computed exactly
/// ([`l1ball::contains`]). Every prover runs it before using $\mathbf c$ (plan
/// `docs/hachi-co-plan.md`, finding F5), and the verifier runs it on the proof's challenges. The
/// set is the draft's $\ell\_1$ ball; a challenge of the sparse set
/// ([`hachi_co_params::ChallengeSet::Sparse`]) lies in it and passes.
///
/// # Errors
/// [`Error::Shape`] for a wrong number of challenges or of coefficients;
/// [`Error::ChallengeNotInC`] for the first challenge outside $\mathcal C$.
pub fn check_challenges<const K: usize>(ctx: &Context<K>, c: &[Vec<i64>]) -> Result<()> {
    let l = &ctx.layout;
    check_len("c (one challenge per block)", l.n_blocks, c.len())?;
    let omega = ctx.omega();
    for (index, ci) in c.iter().enumerate() {
        check_len("c_i (coefficients)", l.d, ci.len())?;
        if !l1ball::contains(ci, omega) {
            let l1_norm = ci.iter().map(|x| u128::from(x.unsigned_abs())).sum();
            return Err(Error::ChallengeNotInC {
                index,
                l1_norm,
                omega,
            });
        }
    }
    Ok(())
}

/// Prover $p$'s fold $\sum\_{i \in \mathcal I\_p} c\_i \mathbf s\_i$ over
/// $\mathbb Z\[X\]/(X^d + 1)$ (`5_Building_Blocks.tex`:127-128; `ref/README.md`, choice 17): the
/// digits of $\mathbf s\_i$ are taken as centered integers and multiplied exactly by the signed
/// challenge ([`hachi_co_core::short::negacyclic_mul`]). Returned flat: $N$ integers, coefficient
/// $l$ of entry $j \lt \delta 2^m$ at $jd + l$, the layout of the mask. For honest data
/// $\Vert \cdot \Vert\_\infty \le |\mathcal I\_p| \omega \lfloor b/2 \rfloor \le \beta\_1$
/// (line 129); [`respond`] refuses a fold above $\beta\_1$.
///
/// The digits are secret: they are centered without branches on their values (best effort,
/// [`hachi_co_core::short::center_poly`]), the product branches only on the public challenge's
/// zero coefficients and on overflow, and the sums only on overflow, which is impossible within
/// the protocol's bounds.
///
/// # Errors
/// [`Error::Core`] if the state has the wrong shape; as [`check_challenges`];
/// [`Error::NonCanonical`] for a digit $\ge q$; [`Error::Overflow`] (or [`Error::Core`] with
/// [`Overflow`](hachi_co_core::Error::Overflow)) only for digits far outside the digit set.
pub fn fold<const K: usize>(
    ctx: &Context<K>,
    state: &PartyState<K>,
    c: &[Vec<i64>],
) -> Result<Vec<i64>> {
    let l = &ctx.layout;
    state.validate(l)?;
    check_challenges(ctx, c)?;
    let (d, q) = (l.d, ctx.q());
    let mut acc = vec![0i64; response_len(ctx)];
    for block in &state.blocks {
        let ci = &c[block.index];
        for (row, s) in block.s.iter().enumerate() {
            let mut digits = centered_digits(s, q)?;
            let product = hachi_co_core::short::negacyclic_mul(ci, &digits);
            crate::ct::wipe(&mut digits);
            let mut product = product?;
            for (a, x) in acc[row * d..(row + 1) * d].iter_mut().zip(&product) {
                *a = a.checked_add(*x).ok_or(Error::Overflow("the fold"))?;
            }
            crate::ct::wipe(&mut product);
        }
    }
    Ok(acc)
}

/// The centered digits of one entry of $\mathbf s\_i$ ([`center_poly`]); a digit $\ge q$ is this
/// crate's [`Error::NonCanonical`], as for every residue the crate checks itself.
fn centered_digits(s: &Poly, q: u64) -> Result<Vec<i64>> {
    center_poly(s, q, "s_i (digits)").map_err(|e| match e {
        hachi_co_core::Error::NonCanonical { what, value, q } => {
            Error::NonCanonical { what, value, q }
        }
        e => Error::Core(e),
    })
}

/// Prover $p$'s response $\mathbf z\_p = \mathsf{Rej}(\sum\_{i \in \mathcal I\_p} c\_i
/// \mathbf s\_i, \chi\_z, \chi\_r, M; \mathbf r\_p)$ (`5_Building_Blocks.tex`:128;
/// `4_preliminaries.tex`:119): `Some(fold + mask)` if
/// $\Vert \mathrm{fold} + \mathrm{mask} \Vert\_\infty \le \gamma - \beta\_1$, and `None`
/// ($\bot$, the prover announces an abort) otherwise; [`rej::rej`] with the context's $\gamma$
/// and $\beta\_1$.
///
/// [`rej::rej`] checks the hypothesis of the lemma ($\Vert \mathrm{fold} \Vert\_\infty \le
/// \beta\_1$, `4_preliminaries.tex`:123) and the support of the mask, without secret-dependent
/// branches or early exit, and branches once on the combined result (best effort); on rejection
/// it overwrites $\mathbf z$ before freeing it.
///
/// # Errors
/// [`Error::Shape`] unless both vectors have $N$ entries; [`Error::Sampling`] with
/// [`SecretTooLarge`](hachi_co_sampling::Error::SecretTooLarge) if
/// $\Vert \mathrm{fold} \Vert\_\infty \gt \beta\_1$ or
/// [`MaskOutOfRange`](hachi_co_sampling::Error::MaskOutOfRange) if the mask leaves
/// $\[-\gamma, \gamma\]^N$.
pub fn respond<const K: usize>(
    ctx: &Context<K>,
    fold: &[i64],
    mask: &[i64],
) -> Result<Option<Vec<i64>>> {
    let n = response_len(ctx);
    check_len("fold (N = d delta 2^m)", n, fold.len())?;
    check_len("mask r_p (N = d delta 2^m)", n, mask.len())?;
    Ok(rej::rej(fold, ctx.gamma(), ctx.beta1(), mask)?)
}
