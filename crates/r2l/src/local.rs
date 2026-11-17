//! The party-local steps of the ring-to-linear step (`5_Building_Blocks.tex`:115-143, with the
//! $L$ iterations of `fig:ring-to-linear` since the draft's commit `1096ea2`), as pure functions
//! of the party's data: $w\_i$ and $\hat{\mathbf w}\_i$ of the owned blocks, the $L$ masks and
//! $\boldsymbol\rho^v$ with their images, the input to $\mathcal F\_{\mathsf{AH}}(\mathbf D)$,
//! the check $\mathbf c \in \mathcal C^{2^r}$, the integer fold, $\mathsf{Rej}$ and the sets of
//! accepted iterations.
//!
//! Short vectors are flat (`docs/m1-spec.md`, "Conventions"): a vector of $L$ ring elements is
//! $Ld$ integers, coefficient $j$ of element $i$ at $id + j$; masks, folds and responses have
//! $N = d\delta 2^m$ entries (`6_Collaborative_Hachi.tex`:279).

use crate::ct::{lift_polys, wipe_polys};
use crate::{Error, Result};
use core::ops::Range;
use hachi_co_core::short::center_poly;
use hachi_co_core::{
    BlockState, Context, PartyId, PartyRandomness, PartyState, PublicVectors, SparseRows,
};
use hachi_co_math::{Poly, decompose};
use hachi_co_sampling::{l1ball, rej};
use rayon::prelude::*;

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
    // In parallel for long vectors (z-hat), the first wrong entry found in order and reported as
    // the loop below reports it.
    let bad = |p: &Poly| p.0.len() != d || p.0.iter().any(|x| *x >= q);
    let first = if polys.len() >= 4096 {
        polys.par_iter().position_first(bad)
    } else {
        polys.iter().position(bad)
    };
    if let Some(i) = first {
        let p = &polys[i];
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
/// (`6_Collaborative_Hachi.tex`:279).
pub fn response_len<const K: usize>(ctx: &Context<K>) -> usize {
    ctx.layout.width_a * ctx.layout.d
}

/// $\mathbf a^T \mathbf G\_{2^m} \mathbf x \in \mathcal R\_q$ for $\mathbf x \in
/// \mathcal R\_q^{\delta 2^m}$ in the element-major gadget layout: the map of
/// `5_Building_Blocks.tex`:118 ($\mathbf x = \mathbf s\_i$) and 113 ($\mathbf x = \mathbf r\_i$);
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

/// $w\_i$ and $\hat{\mathbf w}\_i$ of one block (`5_Building_Blocks.tex`:115-120).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockW {
    /// The block index $i = \mathrm{nat}(\mathbf i) \lt 2^r$.
    pub index: usize,
    /// $w\_i = \mathbf a^T \mathbf G\_{2^m} \mathbf s\_i \in \mathcal R\_q$ (line 118).
    pub w: Poly,
    /// $\hat{\mathbf w}\_i = \mathbf G\_1^{-1}(w\_i) \in \mathcal R\_q^\delta$ (line 119): digit
    /// $e$ at index $e$, as residues.
    pub w_hat: Vec<Poly>,
}

/// $w\_i$ and $\hat{\mathbf w}\_i$ of one block of the prover holding it
/// (`5_Building_Blocks.tex`:115-120): [`a_times_g`] of $\mathbf s\_i$ and [`decompose::g_inv`]
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

/// Step 1 for prover $p$ (`5_Building_Blocks.tex`:115-120): [`block_w`] of every block it holds,
/// in increasing order of index ([`hachi_co_core::Layout::blocks_of`]). Commit forms
/// $\mathbf s\_i = \mathbf G^{-1}(\mathbf f'\_i)$, so $\mathbf G \mathbf s\_i$ is $\mathbf f'\_i$
/// read modulo $q$, as the product reads it: $w\_i$ is formed from $\mathbf f'\_i$, and a state on
/// disk is read back without being decomposed
/// ([`PartyState::for_each_f_prime_group`](hachi_co_core::PartyState::for_each_f_prime_group)),
/// a group of blocks at a time, in parallel.
///
/// # Errors
/// [`Error::Core`] if the state or the public vectors have the wrong shape; as [`block_w`].
pub fn owned_w<const K: usize>(
    ctx: &Context<K>,
    pv: &PublicVectors<K>,
    state: &PartyState<K>,
) -> Result<Vec<BlockW>> {
    state.validate(&ctx.layout)?;
    pv.validate(&ctx.layout)?;
    let a_ring = ctx.ring.prepare(core::slice::from_ref(&pv.a_ring))?;
    let mut out = Vec::with_capacity(state.blocks.len());
    state.for_each_f_prime_group(ctx, |blocks, f_primes| {
        let ws = blocks
            .par_iter()
            .zip(f_primes.par_iter())
            .map(|(block, f_prime)| {
                block.validate_commitment(&ctx.layout)?;
                let [w] = <[Poly; 1]>::try_from(ctx.ring.mat_vec_prepared(&a_ring, f_prime)?)
                    .map_err(|_| Error::Shape {
                        what: "a^T G s_i (one row)",
                        expected: 1,
                        actual: 0,
                    })?;
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
            })
            .collect::<Result<Vec<_>>>()?;
        out.extend(ws);
        Ok::<(), Error>(())
    })?;
    Ok(out)
}

/// The images of a mask $\mathbf r\_p$ (`5_Building_Blocks.tex`:123-126).
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
/// mask $\mathbf r\_p$ (`5_Building_Blocks.tex`:123-126; the oracle's `ring_to_linear`, step 2).
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
        let t0 = ctx.ring.mat_vec_prepared(&*ctx.a_prepared()?, &r)?;
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

/// The groups of iterations $0, \dots, L - 1$ in which the masks are drawn ([`sample_masks`]) and
/// read again for their responses (`run`): as many as threads, so that the masks of a group are
/// formed in parallel and few are held at once.
pub(crate) fn mask_groups(count: usize) -> impl Iterator<Item = Range<usize>> {
    let group = rayon::current_num_threads().max(1);
    (0..count)
        .step_by(group)
        .map(move |start| start..(start + group).min(count))
}

/// Prover $p$'s private values of the ring-to-linear step (`fig:ring-to-linear`, step 3, since
/// the draft's commit `1096ea2`): the $L$ masks, their images and $\boldsymbol\rho^v\_p$.
///
/// All of it is secret. A mask must never be revealed or reused (plan `docs/hachi-co-plan.md`,
/// finding F5); $\boldsymbol\rho^v\_p$ and the images of all $L$ masks enter $p$'s share of
/// $\mathbf z'$ (as copies). `Debug` shows only the number of masks; the type is neither `Clone`
/// nor comparable, so that a mask is not copied or compared by accident, and dropping it wipes
/// the masks, $\boldsymbol\rho^v\_p$ and the images (best effort, see the crate documentation),
/// also when an error ends the step early.
pub struct MaskSecrets {
    /// The masks $\mathbf r\_p^j \leftarrow \chi\_r$, $j \lt L$ (0-indexed, the draft's $j + 1$),
    /// each uniform in $\[-\gamma, \gamma\]^N$ and flat: held by [`sample_masks`], empty in the
    /// secrets of [`sample_mask_images`] (the driver's), which never holds them all.
    pub masks: Vec<Vec<i64>>,
    /// $\boldsymbol\rho^v\_p \leftarrow S\_b^{m\_v}$, one for all iterations, flat ($m\_v d$
    /// integers in $\[\lceil -b/2 \rceil, \lceil b/2 \rceil - 1\]$).
    pub rho_v: Vec<i64>,
    /// The images of each mask ([`mask_images`]), in the order of the masks.
    pub images: Vec<MaskImages>,
}

impl core::fmt::Debug for MaskSecrets {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MaskSecrets")
            .field("masks", &format_args!("[{} masks]", self.masks.len()))
            .finish_non_exhaustive()
    }
}

impl Drop for MaskSecrets {
    fn drop(&mut self) {
        self.wipe_masks();
        crate::ct::wipe(&mut self.rho_v);
        for images in &mut self.images {
            wipe_polys(&mut images.t0);
            wipe_polys(&mut images.t0_hat);
            wipe_polys(core::slice::from_mut(&mut images.w0));
            wipe_polys(&mut images.w0_hat);
        }
    }
}

impl MaskSecrets {
    /// Overwrites every mask with zeros (best effort, see the crate documentation). The driver
    /// calls it as soon as the masks are no longer needed, right after $\mathsf{Rej}$.
    pub fn wipe_masks(&mut self) {
        for mask in &mut self.masks {
            crate::ct::wipe(mask);
        }
    }

    /// $\boldsymbol\rho^v\_p$ as $m\_v$ ring elements with residues in $\[0, q)$ (checked to lie
    /// in $S\_b$).
    pub(crate) fn rho_polys<const K: usize>(&self, ctx: &Context<K>) -> Result<Vec<Poly>> {
        let l = &ctx.layout;
        check_len("rho^v_p (m_v d)", l.m_v * l.d, self.rho_v.len())?;
        let b = ctx.b();
        // S_b = [ceil(-b/2), ceil(b/2) - 1] = [-floor(b/2), floor((b-1)/2)] (4_preliminaries.tex:88).
        let (min, max) = (-((b / 2) as i64), ((b - 1) / 2) as i64);
        let out_of_range = || Error::OutOfRange {
            what: "rho^v_p (S_b)",
            min,
            max,
        };
        // Check the asymmetric range exactly with one combined flag, then lift.
        let mut bad = 0u8;
        for x in &self.rho_v {
            bad |= u8::from(*x < min) | u8::from(*x > max);
        }
        if bad != 0 {
            return Err(out_of_range());
        }
        lift_polys(&self.rho_v, l.d, ctx.q(), b / 2, out_of_range)
    }
}

/// Step 3 (`fig:ring-to-linear`, since the draft's commit `1096ea2`) as the driver runs it: draws the $L$ masks
/// $\mathbf r\_p^j \leftarrow \chi\_r$ ([`PartyRandomness::mask`] with index $j$, $N$ entries,
/// width $\gamma$) and one $\boldsymbol\rho^v\_p \leftarrow S\_b^{m\_v}$
/// ([`PartyRandomness::rho_v`], $m\_v d$ entries), checks what the randomness source returned
/// (lengths and ranges), and computes the [`mask_images`] of every mask, in parallel, a group of masks at a
/// time, each group wiped once its images are formed: the returned secrets hold no masks, which
/// the driver reads again for their responses ([`PartyRandomness::masks_again`]).
///
/// # Errors
/// The errors of the randomness source ([`Error::Core`], e.g.
/// [`RandomnessReuse`](hachi_co_core::Error::RandomnessReuse) for a second call on one
/// [`hachi_co_core::PrgRandomness`]); [`Error::Shape`] or [`Error::OutOfRange`] if it returned
/// malformed values; as [`mask_images`].
pub fn sample_mask_images<const K: usize, R: PartyRandomness<K>>(
    ctx: &Context<K>,
    pv: &PublicVectors<K>,
    rng: &mut R,
) -> Result<MaskSecrets> {
    let l = &ctx.layout;
    let (count, n) = (l.rej_iterations, response_len(ctx));
    let mut secrets = MaskSecrets {
        masks: Vec::new(),
        rho_v: Vec::new(),
        images: Vec::with_capacity(count),
    };
    // On an error below, dropping `secrets` wipes what was drawn. The masks come from their own
    // streams, a group at a time (in parallel where the randomness allows it,
    // `PartyRandomness::masks_in`): each group's images are formed and the group is wiped, so the
    // L masks are never held at once; `run` reads each mask again for its response
    // (`PartyRandomness::masks_again`).
    for range in mask_groups(count) {
        secrets.masks = rng.masks_in(range.clone(), ctx.gamma(), n)?;
        check_len(
            "masks r_p^j (one per iteration)",
            range.len(),
            secrets.masks.len(),
        )?;
        for mask in &secrets.masks {
            check_len("mask r_p^j (N = d delta 2^m)", n, mask.len())?;
        }
        let images = secrets
            .masks
            .par_iter()
            .map(|mask| mask_images(ctx, pv, mask))
            .collect::<Result<Vec<_>>>()?;
        secrets.wipe_masks();
        secrets.masks.clear();
        secrets.images.extend(images);
    }
    secrets.rho_v = rng.rho_v(ctx.b(), l.m_v * l.d)?;
    secrets.rho_polys(ctx)?;
    Ok(secrets)
}

/// [`sample_mask_images`] with the masks held in the returned secrets, read again from the
/// randomness once their images are formed ([`PartyRandomness::masks_again`]): the same masks, for
/// callers that use them directly.
///
/// # Errors
/// As [`sample_mask_images`].
pub fn sample_masks<const K: usize, R: PartyRandomness<K>>(
    ctx: &Context<K>,
    pv: &PublicVectors<K>,
    rng: &mut R,
) -> Result<MaskSecrets> {
    let l = &ctx.layout;
    let n = response_len(ctx);
    let mut secrets = sample_mask_images(ctx, pv, rng)?;
    secrets.masks = rng.masks_again(0..l.rej_iterations, ctx.gamma(), n)?;
    check_len(
        "masks r_p^j (one per iteration)",
        l.rej_iterations,
        secrets.masks.len(),
    )?;
    for mask in &secrets.masks {
        check_len("mask r_p^j (N = d delta 2^m)", n, mask.len())?;
    }
    Ok(secrets)
}

/// Writes prover $p$'s values of $(\hat{\mathbf w}, \hat{\mathbf w}\_0, \hat{\mathbf t}\_0,
/// \boldsymbol\rho^v)$ into `out` $= \mathbf z'\[0, t)$ (positions of [`hachi_co_core::Layout`]):
/// its blocks of $\hat{\mathbf w}$, its own $\hat{\mathbf w}\_{0,p}^j$ and $\hat{\mathbf t}\_{0,p}^j$
/// of every iteration $j \lt L$, and $\boldsymbol\rho^v\_p$; other entries are left untouched.
pub(crate) fn fill_first_message_part<const K: usize, O: FirstPart + ?Sized>(
    ctx: &Context<K>,
    party: PartyId,
    w: &[BlockW],
    secrets: &MaskSecrets,
    out: &mut O,
) -> Result<()> {
    let l = &ctx.layout;
    check_party(ctx, party)?;
    check_len("z'[0, t)", l.t, out.entries())?;
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
        out.put(l.w_hat_range(i), &bw.w_hat);
    }
    check_len(
        "the images of the masks (one per iteration)",
        l.rej_iterations,
        secrets.images.len(),
    )?;
    for (j, images) in secrets.images.iter().enumerate() {
        images.validate(ctx)?;
        out.put(l.w0_hat_range(party, j), &images.w0_hat);
        out.put(l.t0_hat_range(party, j), &images.t0_hat);
    }
    let rho = secrets.rho_polys(ctx)?;
    out.put(l.rho_v_range(party), &rho);
    Ok(())
}

/// Where [`fill_first_message_part`] writes $\mathbf z'\[0, t)$: a slice of its $t$ entries
/// ([`d_slice`]), or the first $t$ rows of a share held sparsely ([`SparsePrefix`],
/// [`crate::z_prime_share`]).
pub(crate) trait FirstPart {
    /// The entries it takes, $t$.
    fn entries(&self) -> usize;
    /// The entries of `range` set to copies of `values`, of the same length.
    fn put(&mut self, range: Range<usize>, values: &[Poly]);
}

impl FirstPart for [Poly] {
    fn entries(&self) -> usize {
        self.len()
    }

    fn put(&mut self, range: Range<usize>, values: &[Poly]) {
        self[range].clone_from_slice(values);
    }
}

/// The first `len` rows of a share held sparsely, as a [`FirstPart`] of `len` entries.
pub(crate) struct SparsePrefix<'a> {
    pub(crate) rows: &'a mut SparseRows,
    pub(crate) len: usize,
}

impl FirstPart for SparsePrefix<'_> {
    fn entries(&self) -> usize {
        self.len
    }

    fn put(&mut self, range: Range<usize>, values: &[Poly]) {
        assert!(range.end <= self.len, "an entry of z'[0, t)");
        self.rows.put(range, values);
    }
}

/// Prover $p$'s slice $\lbrace (\hat{\mathbf w}, \hat{\mathbf w}\_0, \hat{\mathbf t}\_0,
/// \boldsymbol\rho) \rbrace\_p \in \mathcal R\_q^{t}$ (`5_Building_Blocks.tex`:127-133, the
/// images of all $L$ masks;
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
    secrets: &MaskSecrets,
) -> Result<Vec<Poly>> {
    let mut slice = vec![ctx.ring.zero(); ctx.layout.t];
    fill_first_message_part(ctx, party, w, secrets, &mut slice[..])?;
    Ok(slice)
}

/// Prover $p$'s input to $\mathcal F\_{\mathsf{AH}}(\mathbf D)$ (`5_Building_Blocks.tex`:131-133):
/// $\mathbf D \cdot \lbrace (\hat{\mathbf w}, \hat{\mathbf w}\_0, \hat{\mathbf t}\_0,
/// \boldsymbol\rho) \rbrace\_p \in \mathcal R\_q^{n\_D}$ for the slice of [`d_slice`]. The sum of
/// the provers' inputs is the first message
/// $\mathbf v = \mathbf D\_{\hat{\mathbf w}} \hat{\mathbf w} + \mathbf D\_0 \hat{\mathbf w}\_0 +
/// \mathbf D\_1 \hat{\mathbf t}\_0 + \mathbf D\_\rho \boldsymbol\rho$ (line 132).
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
    Ok(ctx.ring.mat_vec(&ctx.pp().d, slice)?)
}

/// Checks $\mathbf c \in \mathcal C^{2^r}$ (`5_Building_Blocks.tex`:134-138): $2^r$ challenges
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
/// $\mathbb Z\[X\]/(X^d + 1)$ (`5_Building_Blocks.tex`:139-140; `ref/README.md`, choice 17): the
/// digits of $\mathbf s\_i$ are taken as centered integers and multiplied exactly by the signed
/// challenge ([`hachi_co_core::short::negacyclic_mul`]). Returned flat: $N$ integers, coefficient
/// $l$ of entry $j \lt \delta 2^m$ at $jd + l$, the layout of the mask. For honest data
/// $\Vert \cdot \Vert\_\infty \le |\mathcal I\_p| \omega \lfloor b/2 \rfloor \le \beta\_1$
/// (line 141); [`respond`] refuses a fold above $\beta\_1$.
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
    if state.spill.is_some() {
        return fold_from_disk(ctx, state, c);
    }
    let (d, q) = (l.d, ctx.q());
    let mut acc = vec![0i64; response_len(ctx)];
    state.for_each_block(ctx, |block| {
        let ci = &c[block.index];
        acc.par_chunks_mut(d)
            .zip(block.s.par_iter())
            .try_for_each(|(row, s)| {
                let mut digits = centered_digits(s, q)?;
                let product = hachi_co_core::short::negacyclic_mul(ci, &digits);
                crate::ct::wipe(&mut digits);
                let mut product = product?;
                for (a, x) in row.iter_mut().zip(&product) {
                    *a = a.checked_add(*x).ok_or(Error::Overflow("the fold"))?;
                }
                crate::ct::wipe(&mut product);
                Ok::<(), Error>(())
            })
    })?;
    Ok(acc)
}

/// [`fold`] for a state on disk: the digits of $\mathbf s\_i = \mathbf G^{-1}(\mathbf f'\_i)$
/// formed as signed integers where they are multiplied, entry by entry of $\mathbf f'\_i$
/// ([`decompose::g_inv_rows`], the digits of [`decompose::g_inv`]), so that $\mathbf s\_i$ is
/// neither formed as residues nor centered again: the same digits, the same products and sums.
fn fold_from_disk<const K: usize>(
    ctx: &Context<K>,
    state: &PartyState<K>,
    c: &[Vec<i64>],
) -> Result<Vec<i64>> {
    let l = &ctx.layout;
    let (d, delta, b, zq) = (l.d, l.delta, ctx.b(), ctx.ring.zq());
    let mut acc = vec![0i64; response_len(ctx)];
    state.for_each_f_prime_group(ctx, |blocks, f_primes| {
        for (block, f_prime) in blocks.iter().zip(f_primes) {
            let ci = &c[block.index];
            // The delta rows of the fold that the digits of entry e of f'_i reach: e delta + k.
            acc.par_chunks_mut(delta * d)
                .zip(f_prime.par_iter())
                .try_for_each_init(
                    || (vec![0i64; delta * d], vec![0i64; d]),
                    |(digits, scratch), (rows, entry)| {
                        let got =
                            add_digit_products(zq, b, delta, ci, entry, rows, digits, scratch);
                        crate::ct::wipe(digits);
                        crate::ct::wipe(scratch);
                        got
                    },
                )?;
        }
        Ok::<(), Error>(())
    })?;
    Ok(acc)
}

/// Adds $c\_i$ times digit $k$ of `entry` to row $k$ of `rows`, for $k \lt \delta$.
#[allow(clippy::too_many_arguments)]
fn add_digit_products(
    zq: &hachi_co_math::Zq,
    b: u64,
    delta: usize,
    ci: &[i64],
    entry: &Poly,
    rows: &mut [i64],
    digits: &mut [i64],
    scratch: &mut [i64],
) -> Result<()> {
    let d = entry.0.len();
    decompose::g_inv_rows(zq, b, delta, &entry.0, digits, scratch)?;
    for (row, digit_row) in rows.chunks_exact_mut(d).zip(digits.chunks_exact(d)) {
        let mut product = hachi_co_core::short::negacyclic_mul(ci, digit_row)?;
        for (a, x) in row.iter_mut().zip(&product) {
            *a = a.checked_add(*x).ok_or(Error::Overflow("the fold"))?;
        }
        crate::ct::wipe(&mut product);
    }
    Ok(())
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
/// \mathbf s\_i, \chi\_z, \chi\_r, M; \mathbf r\_p)$ (`5_Building_Blocks.tex`:140;
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
