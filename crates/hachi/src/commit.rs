//! The arithmetic context ([`Ctx`]) and the inner-outer commitment (paper, Section 4.1):
//! $\mathbf t_i = \mathbf A \mathbf s_i$, $\hat{\mathbf t}_i = \mathbf G^{-1}(\mathbf t_i)$ and
//! $\mathbf u = \mathbf B \hat{\mathbf t}$, with $\mathbf s_i = \mathbf G^{-1}(\mathbf f_i)$ or
//! $\mathbf s_i = \mathbf f_i$ according to the [`Layout`]. In a masked layout ([`Hiding`]), each
//! chunk ends with a masking row and $\mathbf t_i = \mathbf A \mathbf s_i + \mathbf A'
//! \boldsymbol\varrho_i$ ([`Ctx::rand_matrix`], [`Ctx::rand_term`]).

use crate::params::{HachiChoices, Hiding, Layout, check, height};
use crate::table::{ResidueTable, TableRef};
use crate::{Error, Result};
use hachi_co_math::Prepared;
use hachi_co_math::decompose::{digit_bound, g_inv, g_inv_rows, num_digits};
use hachi_co_math::{Fqk, Poly, Ring, Subfield, Zq};
use hachi_co_sampling::matrix::expand_columns;
use hachi_co_sampling::{ShakePrg, uniform};
use rand::Rng;
use rayon::prelude::*;
use std::borrow::Cow;
use std::ops::Range;

/// The columns of a matrix that one task of [`Ctx::matrix_columns`] expands. Every entry is read
/// from its own stream, so the tasks give the entries of a sequential expansion.
const EXPAND_COLUMNS: usize = 256;

/// Everything both sides need: $q$, the choices, $\mathcal R_q$ of degree $d$, the ring of degree
/// $2d$ in which products of two elements of degree below $d$ are exact (for the quotients of
/// ring switching, paper Section 4.3), the subfield $\mathcal R_q^H \cong \mathbb F_{q^K}$ with
/// $\psi$ and $\mathrm{Tr}\_H$, and $\delta$.
#[derive(Clone, Debug)]
pub struct Ctx<const K: usize> {
    /// The choices.
    pub choices: HachiChoices,
    /// The modulus.
    pub q: u64,
    /// $\mathcal R_q = \mathbb Z_q\[X\]/(X^d + 1)$.
    pub ring: Ring,
    /// $\mathbb Z_q\[X\]/(X^{2d} + 1)$.
    pub ring2: Ring,
    /// $\mathcal R_q^H$ with its maps.
    pub sub: Subfield<K>,
    /// $\delta$, the digits of $\mathbf G^{-1}$.
    pub delta: usize,
}

impl<const K: usize> Ctx<K> {
    /// The context for $q$ and the choices.
    ///
    /// # Errors
    /// [`Error::Parameter`] as [`check`]; arithmetic errors ($q$ not prime, ...).
    pub fn new(q: u64, choices: HachiChoices) -> Result<Self> {
        check(&choices, q, K)?;
        let ring = Ring::new(q, choices.d)?;
        let ring2 = Ring::new(q, 2 * choices.d)?;
        let sub = Subfield::new(&ring)?;
        let delta = num_digits(q, choices.b)?;
        Ok(Self {
            choices,
            q,
            ring,
            ring2,
            sub,
            delta,
        })
    }

    /// $\mathbb F_{q^K}$.
    pub fn field(&self) -> &Fqk<K> {
        self.sub.field()
    }

    /// The ring degree $d$.
    pub fn d(&self) -> usize {
        self.choices.d
    }

    /// $\alpha = \log_2 d$.
    pub fn alpha(&self) -> usize {
        self.choices.d.trailing_zeros() as usize
    }

    /// $\kappa = \log_2 K$.
    pub fn kappa(&self) -> usize {
        K.trailing_zeros() as usize
    }

    /// $d/K$, the field elements packed by $\psi$ into one ring element.
    pub fn d_over_k(&self) -> usize {
        self.choices.d / K
    }

    /// The masking choices, if any.
    pub fn hiding(&self) -> Option<Hiding> {
        self.choices.hiding
    }

    /// The height $n_A$ of the commitment of a witness in `layout`.
    ///
    /// # Errors
    /// [`Error::Parameter`] for a masked layout without [`Hiding`].
    pub fn height(&self, layout: &Layout) -> Result<usize> {
        height(&self.choices, layout)
    }

    /// The choices, or an error for a masked layout without them.
    fn masks(&self) -> Result<Hiding> {
        self.hiding()
            .ok_or_else(|| Error::Parameter("randomness without hiding choices".into()))
    }

    /// The length $\nu + n_A$ of the randomness of a chunk of a masked layout.
    ///
    /// # Errors
    /// [`Error::Parameter`] without [`Hiding`].
    pub fn rand_len(&self) -> Result<usize> {
        let h = self.masks()?;
        Ok(h.nu + h.n_a)
    }

    /// `count` ring elements with uniform digits, the set $S_b$ of Definition 2, read from `prg`
    /// element after element, coefficient after coefficient.
    ///
    /// # Errors
    /// Sampling errors.
    pub fn short(&self, prg: &mut ShakePrg, count: usize) -> Result<Vec<Poly>> {
        let coeffs = uniform::short(prg, self.choices.b, count * self.d())?;
        Ok(coeffs
            .chunks_exact(self.d())
            .map(|c| self.ring.from_i64(c))
            .collect::<hachi_co_math::Result<_>>()?)
    }

    /// $\mathbf A_1'$ of $\mathbf A' = [\mathbf A_1' \mid \mathbf I_{n_A}]$, $n_A \times \nu$, at
    /// level 0.
    ///
    /// # Errors
    /// [`Error::Parameter`] without [`Hiding`]; sampling errors.
    pub fn rand_matrix(&self) -> Result<Vec<Vec<Poly>>> {
        let h = self.masks()?;
        self.matrix(b"A'", 0, h.n_a, h.nu)
    }

    /// $[\mathbf A_1' \mid \mathbf I_{n_A}]\,\boldsymbol\varrho$ for $\mathbf A_1'$ = `m1`
    /// ([`Ctx::rand_matrix`]) and $\boldsymbol\varrho$ = `r` of length $\nu + n_A$.
    ///
    /// # Errors
    /// [`Error::Shape`] for a matrix or randomness of the wrong shape; [`Error::Parameter`]
    /// without [`Hiding`]; arithmetic errors.
    pub fn rand_term(&self, m1: &[Vec<Poly>], r: &[Poly]) -> Result<Vec<Poly>> {
        let h = self.masks()?;
        let rows = m1.len();
        if m1.iter().any(|row| row.len() != h.nu)
            || r.len() != h.nu + rows
            || r.iter().any(|p| p.0.len() != self.d())
        {
            return Err(Error::Shape(format!(
                "a matrix of {rows} rows of {} and randomness of {} ring elements of degree {} \
                 are expected",
                h.nu,
                h.nu + rows,
                self.d()
            )));
        }
        let mut out = self.ring.mat_vec(m1, &r[..h.nu])?;
        for (o, e) in out.iter_mut().zip(&r[h.nu..]) {
            *o = self.ring.add(o, e)?;
        }
        Ok(out)
    }

    /// The public matrix `kind` (`b"A"`, `b"B"`, `b"D"`, or with [`Hiding`] `b"A'"`) of iteration
    /// `iteration`, of shape
    /// `rows` $\times$ `cols`: its seed is squeezed from the choices' seed, and its entries are
    /// those of `hachi_co_sampling::matrix::expand`, expanded as in [`Ctx::matrix_columns`].
    ///
    /// # Errors
    /// Sampling errors.
    pub fn matrix(
        &self,
        kind: &[u8],
        iteration: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<Vec<Poly>>> {
        self.matrix_columns(kind, iteration, rows, cols, 0..cols)
    }

    /// The columns `columns` of [`Ctx::matrix`]`(kind, iteration, rows, cols)`, with the same
    /// entries (`hachi_co_sampling::matrix::expand_columns`), for products formed block by block.
    /// The columns are expanded in parallel, `EXPAND_COLUMNS` per task: every entry has its own
    /// stream, so the entries do not depend on the split.
    ///
    /// # Errors
    /// As [`Ctx::matrix`], and for columns outside the matrix.
    pub fn matrix_columns(
        &self,
        kind: &[u8],
        iteration: usize,
        rows: usize,
        cols: usize,
        columns: Range<usize>,
    ) -> Result<Vec<Vec<Poly>>> {
        let mut domain = b"hachi/pp/".to_vec();
        domain.extend_from_slice(kind);
        let mut seed = [0u8; 32];
        ShakePrg::new(&self.choices.seed, &domain, iteration as u64).fill_bytes(&mut seed);
        if columns.start > columns.end || columns.end > cols {
            return Err(hachi_co_sampling::Error::InvalidParameter(
                "the columns are not within the matrix",
            )
            .into());
        }
        let parts = (columns.start..columns.end)
            .step_by(EXPAND_COLUMNS)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|c0| {
                let c1 = (c0 + EXPAND_COLUMNS).min(columns.end);
                expand_columns(&seed, b"hachi/matrix", rows, cols, c0..c1, self.d(), self.q)
            })
            .collect::<hachi_co_sampling::Result<Vec<_>>>()?;
        let mut m: Vec<Vec<Poly>> = (0..rows)
            .map(|_| Vec::with_capacity(columns.len()))
            .collect();
        for part in parts {
            for (row, entries) in m.iter_mut().zip(part) {
                row.extend(entries.into_iter().map(Poly));
            }
        }
        Ok(m)
    }
}

/// The ring elements per column block of a commitment's matrix: as many columns as keep a block of
/// `rows` rows within `bytes`, at least one.
pub(crate) fn block_columns(bytes: usize, rows: usize, d: usize) -> usize {
    (bytes / (rows.max(1) * d * 8)).max(1)
}

/// The size of a block of a commitment's matrix A ([`block_columns`]): A is expanded, prepared and
/// applied one block of columns at a time, so that a commitment holds one block and not the whole
/// matrix. The products are sums over the columns, exact modulo $q$, so the blocks add up to the
/// products by the whole matrix. With four chunks per task ([`CHUNK_GROUP`]) the size no longer
/// matters to the time: on an AMD EPYC (Zen 3) server the share's commitment of
/// secure-l24-n2-a8 (2048 chunks of 1025 elements, n_A = 7, 27 digits, 5 threads) took 98.6 and
/// 99.9 s with blocks of 16 MiB, 97.9 and 97.0 s with 32, 98.1 and 98.2 s with 64, 98.1 and
/// 99.4 s with 128 and 101.1 and 98.5 s with 256 MiB, alternating. With one chunk per task the
/// small blocks were the slow ones there (136.7 against 110.7 s at 16 and 256 MiB).
pub(crate) const BLOCK_BYTES: usize = 16 << 20;

/// The size of a block of B in a commitment's $\mathbf u = \mathbf B \hat{\mathbf t}$ (at most
/// that of the tests' `block_bytes`): its products are a few seconds of a commitment that takes
/// minutes at large l, so its blocks are small whatever [`BLOCK_BYTES`] is, and the end of the
/// share's commitment, which is the peak of large l, holds one small block of B.
const B_BLOCK_BYTES: usize = 16 << 20;

/// The products of the prepared columns `a` of A by the digits of `elements` (decomposed
/// layouts) or by the elements themselves: [`Ring::mat_vec_prepared_pieces`] on pieces of whole
/// elements, at most [`hachi_co_math::rns::NADDS`] digit vectors each, whose digits are formed by
/// [`hachi_co_math::decompose::g_inv_rows`] into the kernel's buffer, the digits of
/// [`Committed::digits`] in its order. The same products as [`Ring::mat_vec_prepared`] on
/// [`Committed::digits`], without a polynomial or a transform held per digit vector.
fn fused_products<const K: usize>(
    ctx: &Ctx<K>,
    layout: &Layout,
    a: &Prepared,
    elements: &[Poly],
) -> Result<Vec<Poly>> {
    let (ring, d) = (&ctx.ring, ctx.d());
    let zq = ring.zq();
    let (b, delta) = (ctx.choices.b, ctx.delta);
    let per = if layout.decomposed { delta } else { 1 };
    let step = (hachi_co_math::rns::NADDS / per).max(1);
    let pieces = (0..elements.len())
        .step_by(step)
        .map(|e| e * per..(e + step).min(elements.len()) * per);
    let mut scratch = vec![0i64; d];
    Ok(ring.mat_vec_prepared_pieces(a, pieces, |piece, out| {
        let (first, last) = (piece.start / per, piece.end / per);
        for (element, digits) in elements[first..last]
            .iter()
            .zip(out.chunks_exact_mut(per * d))
        {
            if layout.decomposed {
                g_inv_rows(zq, b, delta, &element.0, digits, &mut scratch)?;
            } else {
                for (digit, x) in digits.iter_mut().zip(&element.0) {
                    *digit = zq.to_centered(*x);
                }
            }
        }
        Ok(())
    })?)
}

/// [`fused_products`] for several vectors of the same length at once
/// ([`Ring::mat_vec_prepared_pieces_many`]): each prepared entry of `a` is read once per piece
/// for all of them. Returns the products of `elements[v]` at index `v`, those of
/// [`fused_products`] on it.
fn fused_products_many<const K: usize>(
    ctx: &Ctx<K>,
    layout: &Layout,
    a: &Prepared,
    elements: &[&[Poly]],
) -> Result<Vec<Vec<Poly>>> {
    let (ring, d) = (&ctx.ring, ctx.d());
    let zq = ring.zq();
    let (b, delta) = (ctx.choices.b, ctx.delta);
    let per = if layout.decomposed { delta } else { 1 };
    let step = (hachi_co_math::rns::NADDS / per).max(1);
    let count = elements.first().map_or(0, |e| e.len());
    if elements.iter().any(|e| e.len() != count) {
        return Err(Error::Shape("vectors of one length are expected".into()));
    }
    let pieces = (0..count)
        .step_by(step)
        .map(|e| e * per..(e + step).min(count) * per);
    let mut scratch = vec![0i64; d];
    Ok(
        ring.mat_vec_prepared_pieces_many(a, elements.len(), pieces, |v, piece, out| {
            let (first, last) = (piece.start / per, piece.end / per);
            for (element, digits) in elements[v][first..last]
                .iter()
                .zip(out.chunks_exact_mut(per * d))
            {
                if layout.decomposed {
                    g_inv_rows(zq, b, delta, &element.0, digits, &mut scratch)?;
                } else {
                    for (digit, x) in digits.iter_mut().zip(&element.0) {
                        *digit = zq.to_centered(*x);
                    }
                }
            }
            Ok(())
        })?,
    )
}

/// The chunks of a commitment whose products by a block of A one task forms
/// ([`fused_products_many`]), so that each prepared entry of the block is read once for all of
/// them, not once per chunk. On an AMD EPYC (Zen 3) server the share's commitment of
/// secure-l24-n2-a8 (2048 chunks of 1025 elements, n_A = 7, 27 digits, blocks of 256 MiB,
/// 5 threads) took 130.8 and 124.6 s with one chunk per task, 102.3 and 104.2 s with 2, 92.9 and
/// 96.1 s with 4 and 93.3 and 93.4 s with 8 (`timings_fused_against_digit_vectors` at that
/// shape, alternating).
pub(crate) const CHUNK_GROUP: usize = 4;

/// [`CHUNK_GROUP`] chunks per task, fewer when `chunks` would not give every thread of the pool
/// two tasks.
fn chunk_group(chunks: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    CHUNK_GROUP.min(chunks / (2 * threads)).max(1)
}

/// Ring elements whose coefficients are digits of absolute value below 128, held as one byte per
/// coefficient (centered): $\hat{\mathbf t}$ of a commitment, whose ring elements take $d$ bytes
/// each instead of the $24 + 8d$ of a [`Poly`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Digits {
    /// The coefficients of a ring element.
    d: usize,
    /// Ring element $i$ at $[i d, (i + 1) d)$, centered.
    values: Vec<i8>,
}

impl Digits {
    /// No ring elements, of degree `d`.
    pub fn new(d: usize) -> Self {
        Self {
            d,
            values: Vec::new(),
        }
    }

    /// The number of ring elements.
    pub fn len(&self) -> usize {
        self.values.len().checked_div(self.d).unwrap_or(0)
    }

    /// Whether there are no ring elements.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Appends `polys`, whose canonical coefficients must be digits.
    ///
    /// # Errors
    /// [`Error::Shape`] for a ring element of another degree or a coefficient of absolute value
    /// 128 or more.
    pub fn extend(&mut self, zq: &Zq, polys: &[Poly]) -> Result<()> {
        self.values.reserve(polys.len() * self.d);
        for p in polys {
            if p.0.len() != self.d {
                return Err(Error::Shape(format!(
                    "ring elements of {} coefficients are expected",
                    self.d
                )));
            }
            for x in &p.0 {
                let value = i8::try_from(zq.to_centered(*x))
                    .map_err(|_| Error::Shape("a coefficient is not a digit".into()))?;
                self.values.push(value);
            }
        }
        Ok(())
    }

    /// Appends the ring elements of `other`, of the same degree.
    pub(crate) fn append(&mut self, other: &Self) {
        self.values.extend_from_slice(&other.values);
    }

    /// The centered coefficients of ring element `i`.
    pub fn centered(&self, i: usize) -> &[i8] {
        &self.values[i * self.d..][..self.d]
    }

    /// The canonical coefficients of ring element `i` into `out` ($d$ entries).
    pub fn residues_into(&self, zq: &Zq, i: usize, out: &mut [u64]) {
        for (o, c) in out.iter_mut().zip(self.centered(i)) {
            *o = zq.from_i64(i64::from(*c));
        }
    }

    /// Ring elements `range` (canonical coefficients).
    pub fn polys(&self, zq: &Zq, range: Range<usize>) -> Vec<Poly> {
        range
            .map(|i| {
                let mut p = vec![0u64; self.d];
                self.residues_into(zq, i, &mut p);
                Poly(p)
            })
            .collect()
    }
}

/// A committed witness and everything the prover keeps about it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Committed {
    /// The layout.
    pub layout: Layout,
    /// The message chunks $\mathbf f_i$ ($2^r$ chunks of [`Layout::chunk_len`] ring elements, the
    /// masking row last in a masked layout). The committed vectors $\mathbf s_i$ are not kept:
    /// [`Committed::digits`] recomputes them, as they take $\delta$ times the memory of a
    /// decomposed message.
    pub message: Vec<Vec<Poly>>,
    /// The randomness $\boldsymbol\varrho_i$ of each chunk of a masked layout ($\nu + n_A$ ring
    /// elements each), and no chunk at all otherwise.
    pub r: Vec<Vec<Poly>>,
    /// The inner commitments $\mathbf t_i = \mathbf A \mathbf s_i$, plus
    /// $\mathbf A' \boldsymbol\varrho_i$ in a masked layout ($n_A$ each).
    pub t: Vec<Vec<Poly>>,
    /// $\hat{\mathbf t} = (\mathbf G^{-1}(\mathbf t_1), \dots)$, chunk-major: chunk $i$ occupies
    /// entries $[i n_A \delta, (i + 1) n_A \delta)$, with the height of the layout; its digits as
    /// bytes ([`Digits`]).
    pub t_hat: Digits,
    /// The commitment $\mathbf u = \mathbf B \hat{\mathbf t}$ ($n_B$ ring elements).
    pub u: Vec<Poly>,
}

impl Committed {
    /// The committed vector $\mathbf s_i$ of the chunk `chunk`: its digits in a decomposed
    /// layout, the chunk itself otherwise.
    ///
    /// # Errors
    /// Arithmetic errors.
    pub fn digits<'a, const K: usize>(
        ctx: &Ctx<K>,
        layout: &Layout,
        delta: usize,
        chunk: &'a [Poly],
    ) -> Result<Cow<'a, [Poly]>> {
        Ok(if layout.decomposed {
            Cow::Owned(g_inv(&ctx.ring, ctx.choices.b, delta, chunk)?)
        } else {
            Cow::Borrowed(chunk)
        })
    }
}

/// Checks that `message` has the layout's shape.
fn check_message<const K: usize>(
    ctx: &Ctx<K>,
    layout: &Layout,
    message: &[Vec<Poly>],
) -> Result<()> {
    if message.len() != layout.chunks()
        || message.iter().any(|c| c.len() != layout.chunk_len())
        || message.iter().flatten().any(|p| p.0.len() != ctx.d())
    {
        return Err(Error::Shape(format!(
            "a message of {} chunks of {} ring elements of degree {} is expected",
            layout.chunks(),
            layout.chunk_len(),
            ctx.d()
        )));
    }
    Ok(())
}

/// Commits to `message` in `layout` with the matrices of iteration `iteration` and the
/// randomness `r`: one vector of $\nu + n_A$ ring elements per chunk of a masked layout, none
/// otherwise.
///
/// # Errors
/// [`Error::Shape`] for a message or randomness of the wrong shape; arithmetic errors.
pub fn commit<const K: usize>(
    ctx: &Ctx<K>,
    iteration: usize,
    layout: Layout,
    message: Vec<Vec<Poly>>,
    r: Vec<Vec<Poly>>,
) -> Result<Committed> {
    // Each chunk's products run sequentially in the fused form, so it needs chunks for the pool.
    let fused = message.len() >= 2 * rayon::current_num_threads();
    let group = chunk_group(message.len());
    commit_in_blocks(
        ctx,
        iteration,
        layout,
        message,
        r,
        BLOCK_BYTES,
        fused,
        group,
    )
}

/// Commits to the message whose element $e$ of chunk $c$ is $\psi$ of the $d/K$ values of
/// $\mathbb F_{q^K}$ in the residues `table[(c len + e) d .. (c len + e + 1) d]`
/// (`packed_element`, `len` the layout's chunk length), as [`commit`] does on that message,
/// without the message: each block of the products packs the elements it reads. The committed
/// witness has an empty message, which the caller forms when it needs it. No randomness, so
/// `layout` is not masked.
///
/// # Errors
/// [`Error::Shape`] unless `table` has the layout's $2^r$ chunks of ring elements;
/// [`Error::Shape`] for a masked layout; arithmetic errors.
pub fn commit_table<const K: usize, T: ResidueTable + ?Sized>(
    ctx: &Ctx<K>,
    iteration: usize,
    layout: Layout,
    table: &T,
) -> Result<Committed> {
    let fused = layout.chunks() >= 2 * rayon::current_num_threads();
    commit_table_in_blocks(
        ctx,
        iteration,
        layout,
        table.table_ref(),
        BLOCK_BYTES,
        fused,
        chunk_group(layout.chunks()),
    )
}

/// Commits as [`commit`] to the message whose elements `elements(c, range)` forms, chunk by chunk,
/// when the products read them (each element once), with the randomness `r`. The committed witness
/// has an empty message, which the caller forms when it needs it.
///
/// # Errors
/// As [`commit`], and [`Error::Shape`] for elements of the wrong number or degree; those of
/// `elements`.
pub fn commit_source<const K: usize>(
    ctx: &Ctx<K>,
    iteration: usize,
    layout: Layout,
    elements: &ElementSource<'_>,
    r: Vec<Vec<Poly>>,
) -> Result<Committed> {
    commit_source_parts(ctx, iteration, layout, elements, r, None)
}

/// Prepares part `p` of a message read in parts ([`commit_source_parts`]): its elements are read
/// after this and before the next part is prepared.
pub type PartPrep<'a> = dyn Fn(usize) -> Result<()> + Sync + 'a;

/// [`commit_source`] reading the message in `parts.0` parts of consecutive chunks, in order: part
/// $p$ is prepared by `parts.1(p)` and its chunks are then read (every block of $\mathbf A$'s
/// columns), so that a source can form one part at a time. The same commitment: each chunk's
/// products are summed over the blocks of columns in the same order.
///
/// # Errors
/// As [`commit_source`]; [`Error::Shape`] unless the parts divide the chunks; those of the
/// preparation.
pub fn commit_source_parts<const K: usize>(
    ctx: &Ctx<K>,
    iteration: usize,
    layout: Layout,
    elements: &ElementSource<'_>,
    r: Vec<Vec<Poly>>,
    parts: Option<(usize, &PartPrep<'_>)>,
) -> Result<Committed> {
    let fused = layout.chunks() >= 2 * rayon::current_num_threads();
    let (t, t_hat, u) = products_in_parts(
        ctx,
        iteration,
        &layout,
        &Message::Source(elements),
        &r,
        BLOCK_BYTES,
        fused,
        chunk_group(layout.chunks()),
        parts,
    )?;
    Ok(Committed {
        layout,
        message: Vec::new(),
        r,
        t,
        t_hat,
        u,
    })
}

/// Ring element of a table committed as Section 3.1 commits a polynomial over $\mathbb F_{q^K}$
/// (`hachi-pcs.tex`, Appendix A, the next witness of an iteration): $\psi$ of the $d/K$ residues
/// `block`, each the constant of $\mathbb F_{q^K}$ that it is.
///
/// # Errors
/// [`Error::Shape`] unless `block` has $d/K$ residues; arithmetic errors.
#[cfg(test)]
pub(crate) fn packed_element<const K: usize>(ctx: &Ctx<K>, block: &[u64]) -> Result<Poly> {
    if block.len() != ctx.d_over_k() {
        return Err(Error::Shape(format!(
            "a packed element is {} residues",
            ctx.d_over_k()
        )));
    }
    packed_element_at(ctx, block, 0)
}

/// [`packed_element`] of the $d/K$ residues `table[start .. start + d/K]`, read by index. $\psi$
/// places the constant coordinate of each value directly (`Subfield::psi`): value $j <
/// d/(2K)$ is coefficient $j$, value $d/(2K) + j$ is coefficient $d/2 + j$, and every other
/// coefficient is 0. So a table of digits gives coefficients that are single digits.
///
/// # Errors
/// [`Error::Shape`] unless the table has them; arithmetic errors.
pub(crate) fn packed_element_at<const K: usize, T: ResidueTable + ?Sized>(
    ctx: &Ctx<K>,
    table: &T,
    start: usize,
) -> Result<Poly> {
    let (d, per) = (ctx.d(), ctx.d_over_k());
    if start.checked_add(per).is_none_or(|end| end > table.len()) {
        return Err(Error::Shape(format!(
            "a packed element is {per} residues of the table"
        )));
    }
    let (zq, half) = (ctx.ring.zq(), per / 2);
    let mut out = ctx.ring.zero();
    for j in 0..half {
        out.0[j] = zq.reduce(table[start + j]);
        out.0[d / 2 + j] = zq.reduce(table[start + half + j]);
    }
    Ok(out)
}

/// The $\mathbf t_i$, $\hat{\mathbf t}$ and $\mathbf u$ of a commitment ([`Committed`]).
type Products = (Vec<Vec<Poly>>, Digits, Vec<Poly>);

/// Elements `range` of chunk `c` of a message formed when read ([`commit_source`]).
pub type ElementSource<'a> = dyn Fn(usize, Range<usize>) -> Result<Vec<Poly>> + Sync + 'a;

/// Where a commitment reads its message.
enum Message<'a> {
    /// The chunks themselves.
    Chunks(&'a [Vec<Poly>]),
    /// A table packed by [`packed_element`], `chunk_len` ring elements per chunk.
    Table {
        table: TableRef<'a>,
        chunk_len: usize,
    },
    /// A function that forms the elements.
    Source(&'a ElementSource<'a>),
}

impl Message<'_> {
    /// Elements `range` of chunk `c`.
    fn elements<const K: usize>(
        &self,
        ctx: &Ctx<K>,
        c: usize,
        range: Range<usize>,
    ) -> Result<Cow<'_, [Poly]>> {
        Ok(match self {
            Self::Chunks(message) => Cow::Borrowed(&message[c][range]),
            Self::Table { table, chunk_len } => {
                let per = ctx.d_over_k();
                Cow::Owned(
                    range
                        .map(|e| packed_element_at(ctx, table, (c * chunk_len + e) * per))
                        .collect::<Result<_>>()?,
                )
            }
            Self::Source(source) => {
                let len = range.len();
                let elements = source(c, range)?;
                if elements.len() != len || elements.iter().any(|p| p.0.len() != ctx.d()) {
                    return Err(Error::Shape(format!(
                        "{len} ring elements of degree {} are expected from the source",
                        ctx.d()
                    )));
                }
                Cow::Owned(elements)
            }
        })
    }
}

/// [`commit`] with the matrices' blocks of at most `block_bytes`, the products of A in the
/// fused form (`fused`, [`fused_products`]) or by [`Ring::mat_vec_prepared`] on the digits, and
/// `group` chunks per task in the fused form ([`fused_products_many`]): parameters for the
/// tests, which compare small blocks with one block and the forms.
#[allow(clippy::too_many_arguments)] // the commitment and the parameters of its products
fn commit_in_blocks<const K: usize>(
    ctx: &Ctx<K>,
    iteration: usize,
    layout: Layout,
    message: Vec<Vec<Poly>>,
    r: Vec<Vec<Poly>>,
    block_bytes: usize,
    fused: bool,
    group: usize,
) -> Result<Committed> {
    check_message(ctx, &layout, &message)?;
    let (t, t_hat, u) = products(
        ctx,
        iteration,
        &layout,
        &Message::Chunks(&message),
        &r,
        block_bytes,
        fused,
        group,
    )?;
    Ok(Committed {
        layout,
        message,
        r,
        t,
        t_hat,
        u,
    })
}

/// [`commit_table`] with the parameters of [`commit_in_blocks`].
fn commit_table_in_blocks<const K: usize>(
    ctx: &Ctx<K>,
    iteration: usize,
    layout: Layout,
    table: TableRef<'_>,
    block_bytes: usize,
    fused: bool,
    group: usize,
) -> Result<Committed> {
    let size = layout
        .chunks()
        .checked_mul(layout.chunk_len())
        .and_then(|n| n.checked_mul(ctx.d_over_k()));
    if size != Some(ResidueTable::len(&table)) {
        return Err(Error::Shape(
            "the table does not fit the commitment's layout".into(),
        ));
    }
    let message = Message::Table {
        table,
        chunk_len: layout.chunk_len(),
    };
    let (t, t_hat, u) = products(
        ctx,
        iteration,
        &layout,
        &message,
        &[],
        block_bytes,
        fused,
        group,
    )?;
    Ok(Committed {
        layout,
        message: Vec::new(),
        r: Vec::new(),
        t,
        t_hat,
        u,
    })
}

/// The commitment's $\mathbf t_i$, $\hat{\mathbf t}$ and $\mathbf u$ for the message read from
/// `message` and the randomness `r` ([`commit_in_blocks`]).
#[allow(clippy::too_many_arguments)] // the commitment and the parameters of its products
fn products<const K: usize>(
    ctx: &Ctx<K>,
    iteration: usize,
    layout: &Layout,
    message: &Message<'_>,
    r: &[Vec<Poly>],
    block_bytes: usize,
    fused: bool,
    group: usize,
) -> Result<Products> {
    products_in_parts(
        ctx,
        iteration,
        layout,
        message,
        r,
        block_bytes,
        fused,
        group,
        None,
    )
}

/// [`products`] over the chunks in parts ([`commit_source_parts`]), all of them in one part
/// without `parts`.
#[allow(clippy::too_many_arguments)] // the commitment and the parameters of its products
fn products_in_parts<const K: usize>(
    ctx: &Ctx<K>,
    iteration: usize,
    layout: &Layout,
    message: &Message<'_>,
    r: &[Vec<Poly>],
    block_bytes: usize,
    fused: bool,
    group: usize,
    parts: Option<(usize, &PartPrep<'_>)>,
) -> Result<Products> {
    let (count, prepare) = parts.map_or((1, None), |(n, f)| (n, Some(f)));
    if count == 0 || !layout.chunks().is_multiple_of(count) {
        return Err(Error::Shape(format!(
            "{count} parts do not divide the {} chunks",
            layout.chunks()
        )));
    }
    let per_part = layout.chunks() / count;
    let chunks_of_r = if layout.masked { layout.chunks() } else { 0 };
    if r.len() != chunks_of_r {
        return Err(Error::Shape(format!(
            "randomness for {chunks_of_r} chunks is expected"
        )));
    }
    let (ring, b, delta) = (&ctx.ring, ctx.choices.b, ctx.delta);
    let n_a = ctx.height(layout)?;
    let width = layout.width(delta);
    // t_i = A s_i, A by blocks of whole message elements (`block_columns`).
    let per = if layout.decomposed { delta } else { 1 };
    let elements = (block_columns(block_bytes, n_a, ctx.d()) / per).max(1);
    let mut t: Vec<Vec<Poly>> = vec![vec![ring.zero(); n_a]; layout.chunks()];
    for (part, t_part) in t.chunks_mut(per_part).enumerate() {
        if let Some(prepare) = prepare {
            prepare(part)?;
        }
        let first = part * per_part;
        for e0 in (0..layout.chunk_len()).step_by(elements) {
            let e1 = (e0 + elements).min(layout.chunk_len());
            let block = ctx.matrix_columns(b"A", iteration, n_a, width, e0 * per..e1 * per)?;
            let a = ring.prepare_short(&block, layout.beta_s)?;
            drop(block);
            let group = if fused { group.max(1) } else { 1 };
            t_part
                .par_chunks_mut(group)
                .enumerate()
                .try_for_each(|(g, ts)| -> Result<()> {
                    let elements = (0..ts.len())
                        .map(|j| message.elements(ctx, first + g * group + j, e0..e1))
                        .collect::<Result<Vec<_>>>()?;
                    let products = if !fused {
                        let s = Committed::digits(ctx, layout, delta, &elements[0])?;
                        Vec::from([ring.mat_vec_prepared(&a, &s)?])
                    } else if group == 1 {
                        Vec::from([fused_products(ctx, layout, &a, &elements[0])?])
                    } else {
                        let slices: Vec<&[Poly]> = elements.iter().map(|e| &e[..]).collect();
                        fused_products_many(ctx, layout, &a, &slices)?
                    };
                    for (ti, products) in ts.iter_mut().zip(products) {
                        for (x, y) in ti.iter_mut().zip(products) {
                            *x = ring.add(x, &y)?;
                        }
                    }
                    Ok(())
                })?;
        }
    }
    if !r.is_empty() {
        let a_rand = ctx.rand_matrix()?;
        for (ti, ri) in t.iter_mut().zip(r) {
            let term = ctx.rand_term(&a_rand, ri)?;
            for (x, y) in ti.iter_mut().zip(&term) {
                *x = ring.add(x, y)?;
            }
        }
    }
    // t-hat, the digits of each chunk's t_i in parallel, held as bytes.
    let zq = ring.zq();
    let parts = t
        .par_iter()
        .map(|ti| {
            let mut part = Digits::new(ctx.d());
            part.extend(zq, &g_inv(ring, b, delta, ti)?)?;
            Ok(part)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut t_hat = Digits::new(ctx.d());
    for part in &parts {
        t_hat.append(part);
    }
    drop(parts);
    // u = B t-hat, B by blocks of columns of at most `B_BLOCK_BYTES`, each against its columns of
    // t-hat as ring elements.
    let (n_b, cols) = (ctx.choices.n_b, layout.chunks() * n_a * delta);
    if t_hat.len() != cols {
        return Err(Error::Shape("t-hat does not fit the matrix B".into()));
    }
    let step = block_columns(block_bytes.min(B_BLOCK_BYTES), n_b, ctx.d());
    let mut u = vec![ring.zero(); n_b];
    for c0 in (0..cols).step_by(step) {
        let c1 = (c0 + step).min(cols);
        let block = ctx.matrix_columns(b"B", iteration, n_b, cols, c0..c1)?;
        for (x, y) in u.iter_mut().zip(ring.mat_vec_short(
            &block,
            &t_hat.polys(zq, c0..c1),
            digit_bound(b),
        )?) {
            *x = ring.add(x, &y)?;
        }
    }
    Ok((t, t_hat, u))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hachi_co_math::Ext;
    use hachi_co_sampling::uniform::uniform_mod_q;

    const K: usize = 4;
    /// The modulus of Hachi-Co's toy-n1 preset (24 bits).
    const Q: u64 = 8_388_637;

    fn polys(tag: &str, n: usize, d: usize) -> Vec<Poly> {
        uniform_mod_q(&mut ShakePrg::new(&[5u8; 32], tag.as_bytes(), 0), Q, n * d)
            .unwrap()
            .chunks_exact(d)
            .map(|c| Poly(c.to_vec()))
            .collect()
    }

    #[test]
    #[ignore = "timing: cargo test --release -p hachi --lib timings_fused -- --ignored --nocapture"]
    fn timings_fused_against_digit_vectors() {
        // One commitment of 2^r chunks of 2^m + 1 elements, d = 256, b = 4, n_A = 7 and the 54-bit
        // modulus of secure-l24-n2-a8 (27 digits), r and m from HACHI_COMMIT_RM (default 8,8), by
        // digit vectors and in the fused form, on the Rayon pool.
        use std::time::Instant;
        let q = 15_395_032_342_003_157u64;
        let (r, m): (usize, usize) = std::env::var("HACHI_COMMIT_RM")
            .ok()
            .and_then(|v| {
                let (a, b) = v.split_once(',')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            })
            .unwrap_or((8, 8));
        let ctx = Ctx::<K>::new(
            q,
            HachiChoices {
                n_a: 7,
                ..HachiChoices::toy(256)
            },
        )
        .unwrap();
        let d = ctx.d();
        let layout = Layout {
            r,
            m,
            decomposed: true,
            beta_s: 2,
            masked: false,
            extra: 1,
        };
        let message: Vec<Vec<Poly>> = (0..layout.chunks())
            .map(|i| {
                uniform_mod_q(
                    &mut ShakePrg::new(&[6u8; 32], b"t", i as u64),
                    q,
                    layout.chunk_len() * d,
                )
                .unwrap()
                .chunks_exact(d)
                .map(|c| Poly(c.to_vec()))
                .collect()
            })
            .collect();
        // HACHI_COMMIT_BLOCKS_MIB: the block sizes to time in the fused form (default 256).
        let blocks: Vec<usize> = std::env::var("HACHI_COMMIT_BLOCKS_MIB")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
            .unwrap_or_else(|| Vec::from([256]));
        // HACHI_COMMIT_GROUPS: the chunks per task to time in the fused form (default 1).
        let groups: Vec<usize> = std::env::var("HACHI_COMMIT_GROUPS")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
            .unwrap_or_else(|| Vec::from([1]));
        // HACHI_COMMIT_UNFUSED=0 skips the form by digit vectors.
        let unfused = std::env::var("HACHI_COMMIT_UNFUSED").map_or(true, |v| v != "0");
        let mut results = Vec::new();
        let forms: Vec<(bool, usize, usize)> = unfused
            .then_some((false, BLOCK_BYTES, 1))
            .into_iter()
            .chain(
                blocks
                    .iter()
                    .flat_map(|mib| groups.iter().map(move |g| (true, mib << 20, *g))),
            )
            .collect();
        for (fused, bytes, group) in forms {
            let start = Instant::now();
            let committed = commit_in_blocks(
                &ctx,
                0,
                layout,
                message.clone(),
                Vec::new(),
                bytes,
                fused,
                group,
            )
            .unwrap();
            println!(
                "2^{r} chunks of {} elements, delta {}, {} threads: fused {fused}, blocks of {} MiB, {group} chunks per task: {:.2} s",
                layout.chunk_len(),
                ctx.delta,
                rayon::current_num_threads(),
                bytes >> 20,
                start.elapsed().as_secs_f64()
            );
            results.push(committed.u);
        }
        assert!(results.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn commitments_by_column_blocks_are_the_whole_products() {
        // Columns of a matrix against the whole matrix, then commitments with blocks of one
        // column, of a few and of everything, for a decomposed message and a short one that is
        // not decomposed.
        let ctx = Ctx::<K>::new(Q, HachiChoices::toy(16)).unwrap();
        let d = ctx.d();
        let whole = ctx.matrix(b"A", 1, 3, 10).unwrap();
        for range in [0..10, 0..1, 3..7, 9..10, 4..4] {
            let block = ctx.matrix_columns(b"A", 1, 3, 10, range.clone()).unwrap();
            for (row, full) in block.iter().zip(&whole) {
                assert_eq!(row[..], full[range.clone()], "{range:?}");
            }
        }
        assert!(ctx.matrix_columns(b"A", 1, 3, 10, 5..11).is_err());
        for decomposed in [true, false] {
            let layout = Layout {
                r: 2,
                m: 2,
                decomposed,
                beta_s: 2,
                masked: false,
                extra: 0,
            };
            let short = |p: Poly| {
                Poly(
                    p.0.iter()
                        .map(|x| [0, 1, 2, Q - 1, Q - 2][(*x % 5) as usize])
                        .collect(),
                )
            };
            let message: Vec<Vec<Poly>> = (0..layout.chunks())
                .map(|i| {
                    let chunk = polys(&format!("m/{decomposed}/{i}"), layout.chunk_len(), d);
                    if decomposed {
                        chunk
                    } else {
                        chunk.into_iter().map(short).collect()
                    }
                })
                .collect();
            let one = commit_in_blocks(
                &ctx,
                0,
                layout,
                message.clone(),
                Vec::new(),
                usize::MAX / 2,
                false,
                1,
            )
            .unwrap();
            for bytes in [1, 3 * d * 8, 10 * d * 8, usize::MAX / 2] {
                // Groups of 1, 2, 3 (not dividing the 4 chunks) and more than all chunks.
                for (fused, group) in [(false, 1), (true, 1), (true, 2), (true, 3), (true, 9)] {
                    let blocks = commit_in_blocks(
                        &ctx,
                        0,
                        layout,
                        message.clone(),
                        Vec::new(),
                        bytes,
                        fused,
                        group,
                    )
                    .unwrap();
                    assert_eq!(
                        blocks, one,
                        "decomposed {decomposed}, blocks of {bytes} bytes, fused {fused}, \
                         {group} chunks per task"
                    );
                }
            }
            assert_eq!(commit(&ctx, 0, layout, message, Vec::new()).unwrap(), one);
        }
    }

    #[test]
    fn digits_as_bytes_are_the_ring_elements_of_digits() {
        let ring = Ring::new(Q, 16).unwrap();
        let zq = ring.zq();
        let digits: Vec<Poly> = (0..5i64)
            .map(|i| {
                ring.from_i64(&(0..16).map(|k| (k + i) % 9 - 4).collect::<Vec<_>>())
                    .unwrap()
            })
            .collect();
        let mut held = Digits::new(16);
        assert!(held.is_empty());
        held.extend(zq, &digits[..2]).unwrap();
        held.extend(zq, &digits[2..]).unwrap();
        assert_eq!(held.len(), 5);
        assert_eq!(held.polys(zq, 0..5), digits);
        assert_eq!(held.polys(zq, 3..4), digits[3..4]);
        assert_eq!(held.centered(1)[0], -3);
        let mut out = vec![0u64; 16];
        held.residues_into(zq, 4, &mut out);
        assert_eq!(out, digits[4].0);
        // A coefficient of 128 is not a digit, and the degree must match.
        let big = ring.from_i64(&[128; 16]).unwrap();
        assert!(held.extend(zq, &[big]).is_err());
        assert!(held.extend(zq, &[Poly(vec![0; 8])]).is_err());
        let low = ring.from_i64(&[-128; 16]).unwrap();
        let mut edge = Digits::new(16);
        edge.extend(zq, core::slice::from_ref(&low)).unwrap();
        assert_eq!(edge.polys(zq, 0..1), [low]);
    }

    #[test]
    fn a_task_takes_a_group_of_chunks_when_there_are_enough() {
        let threads = rayon::current_num_threads().max(1);
        assert_eq!(chunk_group(0), 1);
        assert_eq!(chunk_group(2 * threads), 1);
        assert_eq!(chunk_group(2 * threads * CHUNK_GROUP), CHUNK_GROUP);
        assert_eq!(chunk_group(usize::MAX / 4), CHUNK_GROUP);
    }

    #[test]
    fn the_parallel_expansion_gives_the_entries_of_the_sequential_one() {
        // More columns than one task expands, and a range that starts inside a task's columns.
        let ctx = Ctx::<K>::new(Q, HachiChoices::toy(16)).unwrap();
        let (rows, cols) = (3, 2 * EXPAND_COLUMNS + 5);
        let mut seed = [0u8; 32];
        ShakePrg::new(&ctx.choices.seed, b"hachi/pp/B", 2).fill_bytes(&mut seed);
        let want =
            hachi_co_sampling::matrix::expand(&seed, b"hachi/matrix", rows, cols, 16, Q).unwrap();
        let got = ctx.matrix(b"B", 2, rows, cols).unwrap();
        for (row, entries) in got.iter().zip(&want) {
            assert_eq!(row.len(), cols);
            for (p, e) in row.iter().zip(entries) {
                assert_eq!(&p.0, e);
            }
        }
        let range = 7..EXPAND_COLUMNS + 9;
        let part = ctx
            .matrix_columns(b"B", 2, rows, cols, range.clone())
            .unwrap();
        for (row, full) in part.iter().zip(&got) {
            assert_eq!(row[..], full[range.clone()]);
        }
        let reversed = Range { start: 9, end: 7 };
        assert!(ctx.matrix_columns(b"B", 2, rows, cols, reversed).is_err());
        assert_eq!(
            ctx.matrix(b"B", 2, rows, 0).unwrap(),
            vec![Vec::new(); rows]
        );
    }

    #[test]
    fn a_packed_element_is_psi_of_its_residues_as_constants() {
        // The direct placement against Subfield::psi of the constants (c, 0, ..., 0), and the
        // inverse giving the constants back.
        for d in [16usize, 64] {
            let ctx = Ctx::<K>::new(Q, HachiChoices::toy(d)).unwrap();
            let block =
                uniform_mod_q(&mut ShakePrg::new(&[3u8; 32], b"psi", d as u64), Q, d / K).unwrap();
            let constants: Vec<Ext<K>> = block
                .iter()
                .map(|&x| Ext(core::array::from_fn(|c| if c == 0 { x } else { 0 })))
                .collect();
            let packed = packed_element(&ctx, &block).unwrap();
            assert_eq!(packed, ctx.sub.psi(&constants).unwrap());
            assert_eq!(ctx.sub.psi_inv(&packed).unwrap(), constants);
            assert!(packed_element(&ctx, &block[1..]).is_err());
        }
    }

    #[test]
    fn a_commitment_of_a_table_is_the_commitment_of_its_packed_message() {
        // A table against its message packed by psi (d/K constants per ring element): the same t,
        // t-hat and u, for a decomposed layout and a short one, in both forms, with blocks of one
        // element, of a few and of everything. The table form keeps no message and rejects a table
        // of the wrong size and a masked layout.
        let ctx = Ctx::<K>::new(Q, HachiChoices::toy(16)).unwrap();
        let per = ctx.d_over_k();
        for decomposed in [true, false] {
            let layout = Layout {
                r: 2,
                m: 2,
                decomposed,
                beta_s: 2,
                masked: false,
                extra: 0,
            };
            let n = layout.chunks() * layout.chunk_len() * per;
            // A short message needs short packed elements: psi of constants in {0, 1, -1} has
            // coefficients in [-1, 1].
            let table: Vec<u64> = uniform_mod_q(
                &mut ShakePrg::new(&[7u8; 32], b"table", u64::from(decomposed)),
                Q,
                n,
            )
            .unwrap()
            .into_iter()
            .map(|x| {
                if decomposed {
                    x
                } else {
                    [0, 1, Q - 1][(x % 3) as usize]
                }
            })
            .collect();
            let message: Vec<Vec<Poly>> = table
                .chunks_exact(layout.chunk_len() * per)
                .map(|chunk| {
                    chunk
                        .chunks_exact(per)
                        .map(|e| packed_element(&ctx, e).unwrap())
                        .collect()
                })
                .collect();
            for bytes in [1, 3 * ctx.d() * 8, usize::MAX / 2] {
                for (fused, group) in [(false, 1), (true, 1), (true, 3)] {
                    let want = commit_in_blocks(
                        &ctx,
                        1,
                        layout,
                        message.clone(),
                        Vec::new(),
                        bytes,
                        fused,
                        1,
                    )
                    .unwrap();
                    let got = commit_table_in_blocks(
                        &ctx,
                        1,
                        layout,
                        TableRef::Residues(&table),
                        bytes,
                        fused,
                        group,
                    )
                    .unwrap();
                    assert!(got.message.is_empty() && got.r.is_empty());
                    assert_eq!(
                        (&got.layout, &got.t, &got.t_hat, &got.u),
                        (&want.layout, &want.t, &want.t_hat, &want.u),
                        "decomposed {decomposed}, blocks of {bytes} bytes, fused {fused}, \
                         {group} chunks per task"
                    );
                    if !decomposed {
                        // The digits of the short table as bytes: the same commitment.
                        let zq = ctx.ring.zq();
                        let mut digits = crate::table::DigitTable::zeros(zq, table.len());
                        digits.set_residues(zq, 0, &table).unwrap();
                        let bytes_form = commit_table_in_blocks(
                            &ctx,
                            1,
                            layout,
                            TableRef::Digits(&digits),
                            bytes,
                            fused,
                            group,
                        )
                        .unwrap();
                        assert_eq!(
                            (&bytes_form.t, &bytes_form.t_hat, &bytes_form.u),
                            (&got.t, &got.t_hat, &got.u),
                            "the table as bytes"
                        );
                    }
                }
            }
            assert!(commit_table(&ctx, 1, layout, &table[1..]).is_err());
            let masked = Layout {
                masked: true,
                ..layout
            };
            assert!(commit_table(&ctx, 1, masked, &table).is_err());
        }
    }
}
