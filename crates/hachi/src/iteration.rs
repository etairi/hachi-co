//! One iteration (paper, Fig. 3 and Section 4.3), and the reduction of its output claim to the
//! next iteration's claim (Section 4.5).
//!
//! **Claim.** For a committed witness of $2^r$ chunks $\mathbf f_i$ of $2^m$ ring elements:
//! $\sum_i b_i \cdot \mathbf a^\top \mathbf f_i = T$ ([`Claim`]).
//!
//! **Iteration.**
//! 1. $w_i = \mathbf a^\top \mathbf f_i$, $\hat{\mathbf w} = \mathbf G^{-1}(\mathbf w)$, send
//!    $\mathbf v = \mathbf D \hat{\mathbf w}$.
//! 2. Receive $\mathbf c \in \mathcal C^{2^r}$; $\mathbf z = \sum_i c_i \mathbf s_i$ (norm at most
//!    $\beta$), $\hat{\mathbf z} = \mathbf J^{-1}(\mathbf z)$.
//! 3. Ring switching: the quotients of the relation's rows ([`crate::relation`]) and the next
//!    witness table $\tilde w$ of $2^L$ digits. Commit to it packed (below), and send the
//!    commitment $\mathbf u'$.
//! 4. Receive $\alpha, \tau_0, \tau_1$; the two sumchecks ([`crate::sumcheck`]) reduce everything
//!    to $\tilde w(\mathbf a) = y'$, which the verifier checks against both final claims.
//!
//! **Masked level** ([`Hiding`](crate::params::Hiding), `hachi-pcs.tex`, steps 2(a) to 2(c) of
//! `Eval`). In a masked layout, steps 1 and 2 become attempts. An attempt draws
//! masks $\mathbf s_0, \boldsymbol\varrho_0 \leftarrow D_{\mathfrak s}$, sets $w_0 = \mathbf a^\top
//! \mathbf G \mathbf s_0$ and $\mathbf t_0 = \mathbf A \mathbf s_0 + \mathbf A'
//! \boldsymbol\varrho_0$, computes $\mathbf v = \mathbf D(\hat{\mathbf w}, \hat w_0, \hat{\mathbf
//! t}_0)$ and derives $\mathbf c$ from it. It then computes the folds with the masks and passes
//! them through $\mathsf{Rej}$. A rejected attempt leaves no trace: it runs on a copy of the
//! transcript, and only the accepted attempt's $\mathbf v$ enters the proof.
//!
//! **Reduction to the next claim** (`hachi-pcs.tex`, step 2(j) of `Eval`, Hachi's Section 3.1
//! applied to the next witness, a deviation from its Section 4.5). The next witness $\tilde w$ is
//! committed as the polynomial over $\mathbb F_{q^K}$ whose coefficients are its digits, each a
//! constant of $\mathbb F_{q^K}$: $\psi$ packs $d/K$ consecutive digits into one ring element
//! (`pack`), so a table index reads as (chunk, element, position in $\psi$), the position in
//! the last $\alpha - \kappa$ bits.
//! 1. Split the sumcheck's point $\mathbf a = (\mathbf a_r, \mathbf a_m, \mathbf a_e)$ the same
//!    way. Then $b'_i = \widetilde{eq}(i, \mathbf a_r)$ and $a'_j = \widetilde{eq}(j, \mathbf a_m)
//!    \cdot \sigma_{-1}(\psi(\widetilde{eq}(\cdot, \mathbf a_e)))$ (`next_weights`).
//! 2. The prover sends $p' = \sum_i b'_i \mathbf a'^\top \mathbf f'_i$. By Theorem 2,
//!    $(K/d)\,\mathrm{Tr}\_H(p') = \tilde w(\mathbf a) = y'$, which the verifier checks with
//!    `Subfield::e_map`. The next claim is $(\mathbf b', \mathbf a', p')$.
//!
//! No partial evaluations are sent, and the reduction is exact: for any committed ring elements
//! the trace of $p'$ is the value at $\mathbf a$ of the table that $\psi^{-1}$ gives, and the range
//! check of the iteration puts every entry of that table in the digit set. After the last
//! iteration there is no reduction: the prover sends $\tilde w$ in the clear (`pcs`, step 3).
//!
//! **With a tail** ([`crate::tail`]), the last iteration commits its table with the tail instead
//! of step 3's commitment, and its transcript absorbs the tail's commitment where it absorbs
//! $\mathbf u'$ otherwise, under its own label (`prove_next_tail`, [`verify_tail`]). The table
//! is then not sent: the tail proves the claim $\tilde w(\mathbf a) = y'$ (`pcs`, "Tail").

use crate::commit::{
    Committed, Ctx, Digits, ElementSource, block_columns, commit_table, packed_element_at,
};
use crate::error::VerifyError;
use crate::fs::Fs;
use crate::mle::{eq_eval, eq_table, mle_eval, powers};
use crate::params::{IterParams, Layout};
use crate::relation::{Masked, Mat, Part, Public, mstar, next_table_parts, quotients};
use crate::sumcheck::{self, RoundMessage, Tensor, range_poly};
use crate::table::{DigitTable, ResidueTable};
use crate::tail::{Tail, TailOpening, check_words};
use crate::{Error, Result};
use hachi_co_math::decompose::{digit_bound, digit_range, g_apply, g_inv, g_inv_rows, j_inv};
use hachi_co_math::{Ext, Fqk, Poly, Ring, Sparse};
use hachi_co_sampling::ShakePrg;
use hachi_co_sampling::gauss::{WideGaussian, bernoulli_exp};
use rayon::prelude::*;
use std::borrow::Cow;

const LABEL_V: &[u8] = b"hachi/iter/v";
const LABEL_C: &[u8] = b"hachi/iter/c";
const LABEL_U_NEXT: &[u8] = b"hachi/iter/u-next";
const LABEL_COINS: &[u8] = b"hachi/iter/coins";
const LABEL_FINAL: &[u8] = b"hachi/iter/final";
const LABEL_TARGET: &[u8] = b"hachi/iter/next-target";
/// The label of the tail's commitment, absorbed in place of $\mathbf u'$ after the last iteration
/// with a tail (module documentation, "With a tail").
const LABEL_TAIL_COMMITMENT: &[u8] = b"hachi/iter/tail-commitment";

/// The PRG domain of the masks of an attempt: the stream of ring element $e$ of attempt $j$ has
/// index $2^{32} j + e$, so that the elements can be drawn in parallel.
pub const DOMAIN_MASK: &[u8] = b"hachi/prove/mask";
/// The PRG domain of the decision of $\mathsf{Rej}$ in attempt $j$ (index $j$).
pub const DOMAIN_REJ: &[u8] = b"hachi/prove/rej";

/// A matrix of ring elements, row after row.
type Matrix = Vec<Vec<Poly>>;

/// $\mathbf A_1'$ at a masked level, else an empty matrix.
fn rand_matrix<const K: usize>(ctx: &Ctx<K>, params: &IterParams) -> Result<Matrix> {
    if params.layout.masked {
        ctx.rand_matrix()
    } else {
        Ok(Vec::new())
    }
}

/// The randomness of the commitment of a witness in `layout`: per chunk of a masked layout,
/// $\nu + n_A$ ring elements of uniform digits from the stream `(seed, domain, 0)`, chunk after
/// chunk. No chunk otherwise.
///
/// # Errors
/// Sampling errors.
pub fn commitment_randomness<const K: usize>(
    ctx: &Ctx<K>,
    layout: &crate::params::Layout,
    seed: &[u8; 32],
    domain: &[u8],
) -> Result<Vec<Vec<Poly>>> {
    if !layout.masked {
        return Ok(Vec::new());
    }
    let mut prg = ShakePrg::new(seed, domain, 0);
    let len = ctx.rand_len()?;
    (0..layout.chunks())
        .map(|_| ctx.short(&mut prg, len))
        .collect()
}

/// The claim $\sum_i b_i \cdot \mathbf a^\top \mathbf f_i = T$ on a committed witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    /// $\mathbf b \in \mathcal R_q^{2^r}$.
    pub b: Vec<Poly>,
    /// $\mathbf a$, one entry per ring element of a chunk: $\mathbf a \in \mathcal R_q^{2^m}$, or
    /// $(\mathbf a, x^*)$ at a masked level.
    pub a: Vec<Poly>,
    /// $T \in \mathcal R_q$.
    pub target: Poly,
}

/// The messages of one iteration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IterationProof<const K: usize> {
    /// $\mathbf v = \mathbf D \hat{\mathbf w}$, or $\mathbf D (\hat{\mathbf w}, \hat w_0,
    /// \hat{\mathbf t}_0)$ at a masked level ($n_D$ ring elements).
    pub v: Vec<Poly>,
    /// The commitment $\mathbf u'$ to the next witness ($n_B$ ring elements), and none after the
    /// last iteration with a tail, which sends [`IterationProof::tail_commitment`] instead.
    pub next_commitment: Vec<Poly>,
    /// The sumcheck rounds ($L$).
    pub rounds: Vec<RoundMessage<K>>,
    /// $y' = \tilde w(\mathbf a)$.
    pub final_eval: Ext<K>,
    /// $p'$, the next claim's value, and `None` after the last iteration, which has no reduction
    /// (module documentation).
    pub next_target: Option<Poly>,
    /// After the last iteration with a tail: the tail's commitment to the next witness
    /// ([`Tail::commitment_words`] words). `None` otherwise.
    pub tail_commitment: Option<Vec<u64>>,
}

/// What an iteration leaves to the verifier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reduced<const K: usize> {
    /// The next level's claim (step 2(j) of `Eval`).
    Claim(Claim),
    /// After the last iteration: $\tilde w(\mathbf a) = y'$, checked on the witness sent in the
    /// clear (step 3).
    Evaluation {
        /// The sumcheck's point $\mathbf a$.
        point: Vec<Ext<K>>,
        /// $y'$.
        value: Ext<K>,
    },
}

/// The digit set $D$ of base $b$ as residues.
pub fn digit_residues<const K: usize>(ctx: &Ctx<K>) -> Vec<u64> {
    let (lo, hi) = digit_range(ctx.choices.b);
    (lo..=hi).map(|c| ctx.ring.zq().from_i128(c)).collect()
}

/// $\lceil \log_2 \rceil$ of the number of rows.
fn log_rows(rows: usize) -> usize {
    rows.next_power_of_two().trailing_zeros() as usize
}

/// The basis element $Z_i$ of $\mathbb F_{q^K}$ (unit coordinate vector).
#[cfg(test)]
pub(crate) fn basis<const K: usize>(i: usize) -> Ext<K> {
    let mut e = Ext([0u64; K]);
    e.0[i] = 1;
    e
}

/// The next claim's $\mathbf b'$ and $\mathbf a'$ from the sumcheck's point $\mathbf a$ (module
/// documentation, step 1), for the next layout's $r'$ and $m'$. $\mathbf a$ has $L = r' + m' +
/// \alpha - \kappa$ coordinates, split into $(\mathbf a_r, \mathbf a_m, \mathbf a_e)$ along (chunk,
/// element, position in $\psi$).
fn next_weights<const K: usize>(
    ctx: &Ctx<K>,
    params: &IterParams,
    r_point: &[Ext<K>],
) -> Result<(Vec<Poly>, Vec<Poly>)> {
    let field = ctx.field();
    let (r, m) = (params.next.r, params.next.m);
    let (a_r, rest) = r_point.split_at(r);
    let (a_m, a_e) = rest.split_at(m);
    let b: Vec<Poly> = eq_table(field, a_r)
        .iter()
        .map(|e| ctx.sub.embed(e))
        .collect();
    let fe = eq_table(field, a_e);
    let a_e_ring = ctx.ring.automorphism(&ctx.sub.psi(&fe)?, -1)?;
    let a = eq_table(field, a_m)
        .iter()
        .map(|g| ctx.ring.mul(&ctx.sub.embed(g), &a_e_ring))
        .collect::<hachi_co_math::Result<Vec<_>>>()?;
    Ok((b, a))
}

/// [`evaluate_claim`] on the next message, packed from `table` one chunk at a time
/// ([`packed_chunk`]), the chunks in parallel with $\mathbf a$ prepared once: the same sum
/// (exact in $R_q$), without the packed message in memory.
fn evaluate_packed_claim<const K: usize, T: ResidueTable + ?Sized>(
    ctx: &Ctx<K>,
    params: &IterParams,
    b: &[Poly],
    a: &[Poly],
    table: &T,
) -> Result<Poly> {
    let ring = &ctx.ring;
    let a_row = ring.prepare(core::slice::from_ref(&a.to_vec()))?;
    let terms = b
        .par_iter()
        .enumerate()
        .map(|(c, bi)| {
            let fi = packed_chunk(ctx, params, table, c)?;
            if fi.len() != a.len() {
                return Err(Error::Shape(
                    "a chunk does not fit the claim's weights".into(),
                ));
            }
            let product = ring
                .mat_vec_prepared(&a_row, &fi)?
                .pop()
                .ok_or_else(|| Error::Shape("a^T f_i is one row".into()))?;
            Ok(ring.mul(bi, &product)?)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut acc = ring.zero();
    for term in &terms {
        acc = ring.add(&acc, term)?;
    }
    Ok(acc)
}

/// $\sum_i b_i \cdot \mathbf a^\top \mathbf f_i$.
pub fn evaluate_claim<const K: usize>(
    ctx: &Ctx<K>,
    b: &[Poly],
    a: &[Poly],
    message: &[Vec<Poly>],
) -> Result<Poly> {
    let ring = &ctx.ring;
    let mut acc = ring.zero();
    for (bi, fi) in b.iter().zip(message) {
        acc = ring.add(&acc, &ring.mul(bi, &ring.inner(a, fi)?)?)?;
    }
    Ok(acc)
}

/// The next message chunks: $\psi$ of $d/K$ consecutive entries of the table as constants of
/// $\mathbb F_{q^K}$ ([`packed_element_at`]), cut into $2^{r'}$ chunks of $2^{m'}$ elements.
fn pack<const K: usize, T: ResidueTable + ?Sized>(
    ctx: &Ctx<K>,
    params: &IterParams,
    table: &T,
) -> Result<Vec<Vec<Poly>>> {
    (0..params.next.chunks())
        .into_par_iter()
        .map(|c| packed_chunk(ctx, params, table, c))
        .collect()
}

/// Chunk `c` of [`pack`]: element $e$ is $\psi$ of the $d/K$ entries `table[e d/K .. (e + 1)
/// d/K]` as constants, so that a chunk can be packed without the others.
fn packed_chunk<const K: usize, T: ResidueTable + ?Sized>(
    ctx: &Ctx<K>,
    params: &IterParams,
    table: &T,
    c: usize,
) -> Result<Vec<Poly>> {
    let (per, len) = (ctx.d_over_k(), params.next.chunk_len());
    if table.len() != params.next.chunks() * len * per || c >= params.next.chunks() {
        return Err(Error::Shape(
            "the packed witness does not fit the next layout".into(),
        ));
    }
    (0..len)
        .map(|e| packed_element_at(ctx, table, (c * len + e) * per))
        .collect()
}

/// $\tilde w(\mathbf a) = \sum_i \tilde w_i \widetilde{eq}(i, \mathbf a)$ for a table of
/// $2^{|\mathbf a|}$ residues, the first coordinate of $\mathbf a$ its most significant bit (as
/// [`mle_eval`] reads a table), summed by rows of columns (as [`crate::sumcheck::Tensor`]
/// splits the weights): the value that step 3 of `Eval` checks on the final witness.
///
/// # Errors
/// [`Error::Shape`] unless the table has $2^{|\mathbf a|}$ entries.
pub fn table_eval<const K: usize, T: ResidueTable + ?Sized>(
    field: &Fqk<K>,
    table: &T,
    point: &[Ext<K>],
) -> Result<Ext<K>> {
    if u32::try_from(point.len())
        .ok()
        .and_then(|n| 1usize.checked_shl(n))
        != Some(table.len())
    {
        return Err(Error::Shape(
            "a table of 2^n entries for a point of n coordinates".into(),
        ));
    }
    let (a_u, a_l) = point.split_at(point.len() / 2);
    let (rows, cols) = (eq_table(field, a_u), eq_table(field, a_l));
    let width = cols.len();
    Ok(rows
        .par_iter()
        .enumerate()
        .map(|(u, eu)| {
            let row = cols.iter().enumerate().fold(field.zero(), |acc, (l, el)| {
                field.add(&acc, &field.scale(el, table[u * width + l]))
            });
            field.mul(eu, &row)
        })
        .reduce(|| field.zero(), |a, b| field.add(&a, &b)))
}

/// $(K/d)\,\mathrm{Tr}\_H(p)$ as a field element.
fn scaled_trace<const K: usize>(ctx: &Ctx<K>, p: &Poly) -> Result<Ext<K>> {
    let one = ctx.ring.monomial(0);
    Ok(ctx.sub.extract(&ctx.sub.e_map(p, &one)?)?)
}

/// The size of a block of A in the masked fold's products A s0 ([`block_columns`], as for the
/// commitments): every block is prepared once per attempt whatever its size, so smaller blocks
/// cost no time here and hold less memory.
const FOLD_BLOCK_BYTES: usize = 16 << 20;

/// The entries of $\mathbf z$ per task when $\hat{\mathbf z} = \mathbf J^{-1}(\mathbf z)$ is
/// formed as bytes ([`Digits`]): each task holds the $\tau$ ring elements of its entries' digits
/// until it has stored them as bytes (at $d = 256$, $\tau \cdot 512$ KiB).
const Z_HAT_BLOCK: usize = 256;

/// $\mathbf m \mathbf v$ by blocks of `step` columns (at least one), each block of `m` prepared
/// with every prime and freed after its products: the sums over the columns are exact, so this is
/// [`Ring::mat_vec_prepared`] on the whole of `m` prepared.
///
/// # Errors
/// [`Error::Shape`] unless every row of `m` has one entry per entry of `v`; those of the ring.
fn mat_vec_by_blocks(ring: &Ring, m: &Matrix, v: &[Poly], step: usize) -> Result<Vec<Poly>> {
    if m.iter().any(|row| row.len() != v.len()) {
        return Err(Error::Shape(
            "a row of the matrix against a vector of another length".into(),
        ));
    }
    let mut out = vec![ring.zero(); m.len()];
    for c0 in (0..v.len()).step_by(step.max(1)) {
        let c1 = (c0 + step.max(1)).min(v.len());
        let block = ring.prepare_columns(m, c0..c1)?;
        for (x, y) in out
            .iter_mut()
            .zip(ring.mat_vec_prepared(&block, &v[c0..c1])?)
        {
            *x = ring.add(x, &y)?;
        }
    }
    Ok(out)
}

/// $\sum_i c_i \mathbf x_i$ over the chunks' vectors `chunk(i)` of `width` ring elements, in
/// $\mathcal R_q$.
fn fold_chunks<'a, const K: usize, F>(
    ctx: &Ctx<K>,
    sparse: &[Sparse],
    width: usize,
    chunk: F,
) -> Result<Vec<Poly>>
where
    F: Fn(usize) -> Result<std::borrow::Cow<'a, [Poly]>> + Sync,
{
    let ring = &ctx.ring;
    let zeros = || vec![ring.zero(); width];
    sparse
        .par_iter()
        .enumerate()
        .try_fold(zeros, |mut acc, (i, ci)| -> Result<Vec<Poly>> {
            let x = chunk(i)?;
            for (zx, sx) in acc.iter_mut().zip(x.iter()) {
                ring.add_sparse_product(zx, ci, sx)?;
            }
            Ok(acc)
        })
        .try_reduce(zeros, |mut acc, other| {
            for (x, y) in acc.iter_mut().zip(&other) {
                *x = ring.add(x, y)?;
            }
            Ok(acc)
        })
}

/// $\sum_i c_i \mathbf s_i$ over the committed vectors $\mathbf s_i$ of the chunks of `message`
/// ([`Committed::digits`]): for a decomposed layout by [`fold_message_digits`], which forms the
/// digits as it folds, otherwise (or if its integer sums could leave `i64`) by [`fold_chunks`].
fn fold_message<const K: usize>(
    ctx: &Ctx<K>,
    layout: &Layout,
    delta: usize,
    sparse: &[Sparse],
    message: MessageView<'_>,
) -> Result<Vec<Poly>> {
    if layout.decomposed
        && let Some(z) = fold_message_digits(ctx, delta, sparse, message)?
    {
        return Ok(z);
    }
    // The chunks of each part after it is prepared (one part for a message not read in parts):
    // the sums of the parts are the sum over all the chunks.
    let ring = &ctx.ring;
    let mut z: Option<Vec<Poly>> = None;
    for p in 0..message.part_count() {
        let range = message.prepare_part(p)?;
        let first = range.start;
        let part = fold_chunks(
            ctx,
            &sparse[range],
            layout.width(delta),
            |i| match message {
                MessageView::Held(m) => Committed::digits(ctx, layout, delta, &m[first + i]),
                MessageView::Source { .. } => Ok(Cow::Owned(
                    Committed::digits(ctx, layout, delta, &message.chunk(first + i)?)?.into_owned(),
                )),
            },
        )?;
        z = Some(match z {
            None => part,
            Some(mut acc) => {
                for (x, y) in acc.iter_mut().zip(&part) {
                    *x = ring.add(x, y)?;
                }
                acc
            }
        });
    }
    z.ok_or_else(|| Error::Shape("a message of no parts".into()))
}

/// [`fold_chunks`] over the digits of a decomposed message, $\sum_i c_i \mathbf
/// G^{-1}(\mathbf m_i)$, element by element: for each ring element $e$ of a chunk (in parallel),
/// the $\delta$ digit rows of $m_{i,e}$ of every chunk $i$ are formed in a scratch buffer
/// ([`g_inv_rows`]) and $c_i$ times each is added to one of $\delta$ accumulators of signed
/// integers ([`Ring::add_sparse_product_signed`]), which are reduced modulo $q$ at the end.
/// Output entry $e\delta + k$ is digit $k$ of element $e$, as in [`g_inv`]. The integer sums are
/// those of [`fold_chunks`] before reduction, so the result is the same, without the decomposed
/// chunks in memory and without a reduction per term. `None` if the sums could leave `i64`: each
/// is at most $\lfloor b/2 \rfloor \sum_i \Vert c_i \Vert_1$ in absolute value.
///
/// # Errors
/// [`Error::Shape`] unless there is one challenge per chunk and the chunks have one length;
/// arithmetic errors.
fn fold_message_digits<const K: usize>(
    ctx: &Ctx<K>,
    delta: usize,
    sparse: &[Sparse],
    message: MessageView<'_>,
) -> Result<Option<Vec<Poly>>> {
    let b = ctx.choices.b;
    let (len, uniform) = match message {
        MessageView::Held(m) => {
            let len = m.first().map_or(0, Vec::len);
            (len, m.iter().all(|chunk| chunk.len() == len))
        }
        MessageView::Source { chunk_len, .. } => (chunk_len, true),
    };
    if message.chunks() != sparse.len() || !uniform {
        return Err(Error::Shape(
            "one challenge per chunk, and chunks of one length, are expected".into(),
        ));
    }
    let bound = sparse
        .iter()
        .try_fold(0u64, |s, c| s.checked_add(c.l1_norm()))
        .and_then(|s| s.checked_mul(digit_bound(b).max(1)));
    if !bound.is_some_and(|s| s < 1 << 62) {
        return Ok(None);
    }
    // Every partial sum is within the bound, so below 2^31 the sums of i32 accumulators are those
    // of i64 ones, with twice the lanes.
    if bound.is_some_and(|s| s <= i32::MAX as u64) {
        return fold_message_digits_i32(ctx, delta, sparse, message, len).map(Some);
    }
    fold_message_digits_i64(ctx, delta, sparse, message, len).map(Some)
}

/// The loop of [`fold_message_digits`] with `i64` accumulators.
fn fold_message_digits_i64<const K: usize>(
    ctx: &Ctx<K>,
    delta: usize,
    sparse: &[Sparse],
    message: MessageView<'_>,
    len: usize,
) -> Result<Vec<Poly>> {
    let (ring, b, d) = (&ctx.ring, ctx.choices.b, ctx.d());
    // Every element's sums over the chunks, part after part (one part for a message not read in
    // parts): the chunks in the same order, so the same integers.
    let mut accs = vec![0i64; len * delta * d];
    for p in 0..message.part_count() {
        let range = message.prepare_part(p)?;
        accs.par_chunks_mut(delta * d)
            .enumerate()
            .try_for_each_init(
                || (vec![0i64; delta * d], vec![0i64; d]),
                |(digits, scratch), (e, acc)| -> Result<()> {
                    for i in range.clone() {
                        let element = message.element(i, e)?;
                        g_inv_rows(ring.zq(), b, delta, &element.0, digits, scratch)?;
                        for (sum, row) in acc.chunks_exact_mut(d).zip(digits.chunks_exact(d)) {
                            ring.add_sparse_product_signed(sum, &sparse[i], row)?;
                        }
                    }
                    Ok(())
                },
            )?;
    }
    let elements = accs
        .par_chunks(delta * d)
        .map(|acc| {
            acc.chunks_exact(d)
                .map(|sum| Ok(ring.from_i64(sum)?))
                .collect::<Result<Vec<Poly>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(elements.into_iter().flatten().collect())
}

/// [`fold_message_digits`] with `i32` accumulators, for sums that the caller has bounded below
/// $2^{31}$: the digits of each element are formed in `i64` ([`g_inv_rows`]), narrowed (they are
/// at most $b/2$ in absolute value) and added by [`Ring::add_sparse_product_signed_i32`]. The same
/// integers, so the same ring elements.
fn fold_message_digits_i32<const K: usize>(
    ctx: &Ctx<K>,
    delta: usize,
    sparse: &[Sparse],
    message: MessageView<'_>,
    len: usize,
) -> Result<Vec<Poly>> {
    let (ring, b, d) = (&ctx.ring, ctx.choices.b, ctx.d());
    // As in `fold_message_digits_i64`: every element's sums, part after part, the chunks in the
    // same order (every partial sum is within the caller's bound).
    let mut accs = vec![0i32; len * delta * d];
    for p in 0..message.part_count() {
        let range = message.prepare_part(p)?;
        accs.par_chunks_mut(delta * d)
            .enumerate()
            .try_for_each_init(
                || (vec![0i64; delta * d], vec![0i64; d], vec![0i32; delta * d]),
                |(digits, scratch, narrow), (e, acc)| -> Result<()> {
                    for i in range.clone() {
                        let element = message.element(i, e)?;
                        g_inv_rows(ring.zq(), b, delta, &element.0, digits, scratch)?;
                        for (n, x) in narrow.iter_mut().zip(digits.iter()) {
                            *n = i32::try_from(*x).map_err(|_| {
                                Error::Protocol("a digit does not fit 32 bits".into())
                            })?;
                        }
                        for (sum, row) in acc.chunks_exact_mut(d).zip(narrow.chunks_exact(d)) {
                            ring.add_sparse_product_signed_i32(sum, &sparse[i], row)?;
                        }
                    }
                    Ok(())
                },
            )?;
    }
    let elements = accs
        .par_chunks(delta * d)
        .map_init(
            || vec![0i64; d],
            |wide, acc| {
                acc.chunks_exact(d)
                    .map(|sum| {
                        for (w, x) in wide.iter_mut().zip(sum) {
                            *w = i64::from(*x);
                        }
                        Ok(ring.from_i64(wide)?)
                    })
                    .collect::<Result<Vec<Poly>>>()
            },
        )
        .collect::<Result<Vec<_>>>()?;
    Ok(elements.into_iter().flatten().collect())
}

/// The challenges as ring elements and as sparse vectors.
fn folding_challenges<const K: usize>(
    ctx: &Ctx<K>,
    chunks: usize,
    fs: &mut Fs,
) -> Result<(Vec<Poly>, Vec<Sparse>)> {
    let ring = &ctx.ring;
    let coeffs = fs.challenges(LABEL_C, chunks, ctx.d(), ctx.choices.omega)?;
    let c = coeffs
        .iter()
        .map(|ci| ring.from_i64(ci))
        .collect::<hachi_co_math::Result<_>>()?;
    let sparse = coeffs
        .iter()
        .map(|ci| ring.sparse(ci))
        .collect::<hachi_co_math::Result<_>>()?;
    Ok((c, sparse))
}

/// The folds of an iteration: the first message, the challenges, and the rest of $\mathbf z'$.
struct Fold {
    v: Vec<Poly>,
    c: Vec<Poly>,
    z: Vec<Poly>,
    /// The masked parts, at a masked level.
    masks: Option<Masks>,
}

/// What a masked level adds to the witness ([`Masked`]).
struct Masks {
    w0: Poly,
    w0_hat: Vec<Poly>,
    t0: Vec<Poly>,
    t0_hat: Vec<Poly>,
    z_r: Vec<Poly>,
}

impl Masks {
    fn parts(&self) -> Masked<'_> {
        Masked {
            w0: &self.w0,
            w0_hat: &self.w0_hat,
            t0: &self.t0,
            t0_hat: &self.t0_hat,
            z_r: &self.z_r,
        }
    }
}

/// $\mathsf{Rej}(\mathbf v, D_{\mathfrak s}^N, D_{\mathfrak s}^N, M; \mathbf r)$: whether $\mathbf z
/// = \mathbf v + \mathbf r$ is accepted, with probability $\min\lbrace D(\mathbf z) / (M
/// D(\mathbf r)), 1 \rbrace = \min\lbrace \exp(-\gamma), 1 \rbrace$, where
/// $\gamma = (2 \langle \mathbf v, \mathbf r \rangle + \lVert \mathbf v \rVert^2) / (2 \mathfrak
/// s^2) + \ln M$. Everything is an exact ratio of integers, and the bit comes from
/// [`bernoulli_exp`].
fn rej<const K: usize>(
    ctx: &Ctx<K>,
    hiding: &crate::params::Hiding,
    sigma: u64,
    v: &[Poly],
    r: &[Poly],
    prg: &mut ShakePrg,
) -> Result<bool> {
    let ring: &Ring = &ctx.ring;
    let overflow = || Error::Parameter("the rejection test does not fit 128 bits".into());
    let (inner, norm) = v
        .par_iter()
        .zip(r)
        .map(|(x, y)| -> Result<(i128, i128)> {
            let (x, y) = (ring.to_centered(x)?, ring.to_centered(y)?);
            Ok(x.iter().zip(&y).fold((0, 0), |(i, n), (a, b)| {
                let a = i128::from(*a);
                (i + a * i128::from(*b), n + a * a)
            }))
        })
        .try_reduce(|| (0, 0), |a, b| Ok((a.0 + b.0, a.1 + b.1)))?;
    let (log_num, log_den) = hiding.log_rejection();
    let sigma2 = u128::from(sigma) * u128::from(sigma);
    let den = sigma2.checked_mul(log_den).ok_or_else(overflow)?;
    // gamma = (2 <v, r> + |v|^2) phi^2 / den + log_num sigma^2 / den, as log_den = 2 phi^2.
    let phi2 = i128::try_from(log_den / 2).map_err(|_| overflow())?;
    let top = 2i128
        .checked_mul(inner)
        .and_then(|x| x.checked_add(norm))
        .and_then(|x| x.checked_mul(phi2))
        .and_then(|x| {
            let shift = i128::try_from(sigma2.checked_mul(log_num)?).ok()?;
            x.checked_add(shift)
        })
        .ok_or_else(overflow)?;
    match u128::try_from(top) {
        Ok(gamma) => Ok(bernoulli_exp(prg, gamma, den)?),
        Err(_) => Ok(true),
    }
}

/// `count` ring elements of mask, coefficients from `gauss`, each from its own stream of `seed`.
fn draw_masks<const K: usize>(
    ctx: &Ctx<K>,
    gauss: &WideGaussian,
    seed: &[u8; 32],
    attempt: u32,
    first: usize,
    count: usize,
) -> Result<Vec<Poly>> {
    (first..first + count)
        .into_par_iter()
        .map(|element| {
            let index = (u64::from(attempt) << 32) | element as u64;
            let mut prg = ShakePrg::new(seed, DOMAIN_MASK, index);
            let mut x = vec![0i64; ctx.d()];
            gauss.fill(&mut prg, &mut x)?;
            Ok(ctx.ring.from_i64(&x)?)
        })
        .collect()
}

/// Steps 2(a) to 2(c) of `Eval` at a masked level (module documentation). The transcript `fs` gains
/// the accepted attempt's $\mathbf v$ and $\mathbf c$ only.
///
/// # Errors
/// [`Error::Protocol`] if the fold's bound fails (a witness outside the layout's bound) or if no
/// attempt is accepted within [`Hiding::attempts`](crate::params::Hiding::attempts). Arithmetic
/// and sampling errors.
#[allow(clippy::too_many_arguments)] // the parts of the statement and of the witness
fn masked_fold<const K: usize>(
    ctx: &Ctx<K>,
    index: usize,
    params: &IterParams,
    witness: &Committed,
    message: MessageView<'_>,
    claim: &Claim,
    w_hat: &[Poly],
    d_mat: &Matrix,
    a_mat: &Matrix,
    a_rand: &Matrix,
    fs: &mut Fs,
    seed: &[u8; 32],
) -> Result<Fold> {
    let (ring, b, delta) = (&ctx.ring, ctx.choices.b, params.delta);
    let hiding = ctx
        .hiding()
        .ok_or_else(|| Error::Parameter("a masked level without hiding choices".into()))?;
    let layout = &params.layout;
    let width = layout.width(delta);
    let rand_len = ctx.rand_len()?;
    let gauss = WideGaussian::new(params.sigma)?;
    // A s0 by blocks of columns, each prepared in its attempt, rather than all of A prepared once
    // for the attempts: the transforms of A are not held next to the witness (`block_columns`).
    let step = block_columns(FOLD_BLOCK_BYTES, a_mat.len(), ctx.d());
    let n_d = ctx.choices.n_d;
    for attempt in 0..hiding.attempts() {
        // Step 2(a).
        let s0 = draw_masks(ctx, &gauss, seed, attempt, 0, width)?;
        let r0 = draw_masks(ctx, &gauss, seed, attempt, width, rand_len)?;
        let w0 = ring.inner(&claim.a, &g_apply(ring, b, delta, &s0)?)?;
        let mut t0 = mat_vec_by_blocks(ring, a_mat, &s0, step)?;
        let term = ctx.rand_term(a_rand, &r0)?;
        for (x, y) in t0.iter_mut().zip(&term) {
            *x = ring.add(x, y)?;
        }
        let w0_hat = g_inv(ring, b, delta, core::slice::from_ref(&w0))?;
        let t0_hat = g_inv(ring, b, delta, &t0)?;
        let mut columns = w_hat.to_vec();
        columns.extend_from_slice(&w0_hat);
        columns.extend_from_slice(&t0_hat);
        let v = ring.mat_vec_short(d_mat, &columns, digit_bound(b))?;
        debug_assert_eq!(v.len(), n_d);
        let mut attempt_fs = fs.clone();
        attempt_fs.polys(LABEL_V, &v);
        // Step 2(b).
        let (c, sparse) = folding_challenges(ctx, layout.chunks(), &mut attempt_fs)?;
        // Step 2(c).
        let mut v_all = fold_message(ctx, layout, delta, &sparse, message)?;
        let v_r = fold_chunks(ctx, &sparse, rand_len, |i| {
            Ok(std::borrow::Cow::Borrowed(&witness.r[i][..]))
        })?;
        v_all.extend(v_r);
        let mut r_all = s0;
        r_all.extend(r0);
        let mut prg = ShakePrg::new(seed, DOMAIN_REJ, u64::from(attempt));
        let accepted = rej(ctx, &hiding, params.sigma, &v_all, &r_all, &mut prg)?;
        let z_all = v_all
            .iter()
            .zip(&r_all)
            .map(|(x, y)| ring.add(x, y))
            .collect::<hachi_co_math::Result<Vec<_>>>()?;
        let mut within = true;
        for x in &z_all {
            within &= ring.inf_norm(x)? <= params.beta;
        }
        if accepted && within {
            *fs = attempt_fs;
            let (z, z_r) = z_all.split_at(width);
            return Ok(Fold {
                v,
                c,
                z: z.to_vec(),
                masks: Some(Masks {
                    w0,
                    w0_hat,
                    t0,
                    t0_hat,
                    z_r: z_r.to_vec(),
                }),
            });
        }
    }
    Err(Error::Protocol(format!(
        "no fold was accepted in {} attempts at level {index}",
        hiding.attempts()
    )))
}

/// Steps 1 and 2 of an iteration without masks (module documentation).
fn plain_fold<const K: usize>(
    ctx: &Ctx<K>,
    params: &IterParams,
    message: MessageView<'_>,
    w_hat: &[Poly],
    d_mat: &Matrix,
    fs: &mut Fs,
) -> Result<Fold> {
    let (ring, delta) = (&ctx.ring, params.delta);
    let layout = &params.layout;
    let v = ring.mat_vec_short(d_mat, w_hat, digit_bound(ctx.choices.b))?;
    fs.polys(LABEL_V, &v);
    let (c, sparse) = folding_challenges(ctx, layout.chunks(), fs)?;
    let z = fold_message(ctx, layout, delta, &sparse, message)?;
    for zx in &z {
        if ring.inf_norm(zx)? > params.beta {
            return Err(Error::Protocol(format!(
                "the fold exceeds its bound beta = {}",
                params.beta
            )));
        }
    }
    Ok(Fold {
        v,
        c,
        z,
        masks: None,
    })
}

/// The witness of an iteration ([`prove`]), borrowed or owned. An owned witness lets the iteration
/// free its message once steps 1 and 2 have read it, as the rest of the iteration reads only
/// $\mathbf t$, $\hat{\mathbf t}$ and $\mathbf u$; the messages are the same either way.
pub enum Witness<'a> {
    /// The caller keeps the witness.
    Borrowed(&'a Committed),
    /// The iteration may free the message.
    Owned(Committed),
    /// A witness without its message (`committed.message` is empty), whose chunks `source` forms
    /// when steps 1 and 2 read them; the source, and what it reads from, is dropped after step 2.
    Sourced {
        /// The rest of the witness.
        committed: Committed,
        /// Its message, until step 2 is done.
        source: Option<OwnedSource<'a>>,
    },
}

/// A message formed when it is read, owning what it is formed from: the source of
/// [`Witness::Sourced`].
pub struct OwnedSource<'a> {
    /// The number of chunks.
    pub chunks: usize,
    /// The ring elements of a chunk.
    pub chunk_len: usize,
    /// Elements `range` of chunk `c`.
    pub elements: Box<ElementSource<'a>>,
    /// The message in parts of consecutive chunks, formed one at a time ([`OwnedParts`]).
    pub parts: Option<OwnedParts<'a>>,
}

/// A message read in `count` parts of consecutive chunks: `prepare(p)` forms part $p$, after which
/// only its chunks are read, until the next part is prepared. The readers of the message
/// ([`MessageView::prepare_part`]) prepare the parts in order, once per pass over the chunks.
pub struct OwnedParts<'a> {
    /// The number of parts, dividing the chunks.
    pub count: usize,
    /// Forms part `p`.
    pub prepare: Box<PartPrepare<'a>>,
}

/// Forms part `p` of a message read in parts.
pub type PartPrepare<'a> = dyn Fn(usize) -> Result<()> + Send + Sync + 'a;

impl OwnedSource<'_> {
    /// The message as steps 1 and 2 read it.
    pub fn view(&self) -> MessageView<'_> {
        MessageView::Source {
            chunks: self.chunks,
            chunk_len: self.chunk_len,
            elements: &*self.elements,
            parts: self.parts.as_ref().map(|p| (p.count, &*p.prepare)),
        }
    }
}

/// The message of a witness as steps 1 and 2 of its iteration read it: the chunks, or a function
/// that forms the elements of a chunk when they are read (the same elements).
#[derive(Clone, Copy)]
pub enum MessageView<'a> {
    /// The chunks.
    Held(&'a [Vec<Poly>]),
    /// `chunks` chunks of `chunk_len` ring elements, elements `range` of chunk `c` formed by
    /// `elements(c, range)`.
    Source {
        /// The number of chunks.
        chunks: usize,
        /// The ring elements of a chunk.
        chunk_len: usize,
        /// The elements.
        elements: &'a ElementSource<'a>,
        /// The number of parts and their preparation ([`OwnedParts`]), if the message is read in
        /// parts.
        parts: Option<(usize, &'a PartPrepare<'a>)>,
    },
}

impl<'a> MessageView<'a> {
    /// The number of chunks.
    pub fn chunks(&self) -> usize {
        match self {
            Self::Held(m) => m.len(),
            Self::Source { chunks, .. } => *chunks,
        }
    }

    /// The number of parts in which the message is read (1 if it is not read in parts).
    pub fn part_count(&self) -> usize {
        match self {
            Self::Source {
                parts: Some((count, _)),
                ..
            } => (*count).max(1),
            _ => 1,
        }
    }

    /// Prepares part `p` (nothing for a message not read in parts) and returns its chunks: a
    /// pass over the chunks prepares the parts in order and reads each part's chunks after it.
    ///
    /// # Errors
    /// [`Error::Shape`] unless the parts divide the chunks and `p` is one of them; those of the
    /// preparation.
    pub fn prepare_part(&self, p: usize) -> Result<core::ops::Range<usize>> {
        let (count, chunks) = (self.part_count(), self.chunks());
        if p >= count || !chunks.is_multiple_of(count) {
            return Err(Error::Shape(format!(
                "part {p} of {count} parts of {chunks} chunks"
            )));
        }
        if let Self::Source {
            parts: Some((_, prepare)),
            ..
        } = self
        {
            prepare(p)?;
        }
        let per = chunks / count;
        Ok(p * per..(p + 1) * per)
    }

    /// Chunk `c`.
    ///
    /// # Errors
    /// Those of the source; [`Error::Shape`] for a source that forms another number of elements.
    pub fn chunk(&self, c: usize) -> Result<Cow<'a, [Poly]>> {
        match *self {
            Self::Held(m) => Ok(Cow::Borrowed(&m[c])),
            Self::Source {
                chunk_len,
                elements,
                ..
            } => {
                let chunk = elements(c, 0..chunk_len)?;
                if chunk.len() != chunk_len {
                    return Err(Error::Shape(format!(
                        "a chunk of {chunk_len} ring elements is expected from the source"
                    )));
                }
                Ok(Cow::Owned(chunk))
            }
        }
    }

    /// Element `e` of chunk `c`.
    ///
    /// # Errors
    /// As [`MessageView::chunk`].
    pub fn element(&self, c: usize, e: usize) -> Result<Cow<'a, Poly>> {
        match *self {
            Self::Held(m) => Ok(Cow::Borrowed(&m[c][e])),
            Self::Source { elements, .. } => {
                let mut one = elements(c, e..e + 1)?;
                match (one.pop(), one.is_empty()) {
                    (Some(p), true) => Ok(Cow::Owned(p)),
                    _ => Err(Error::Shape(
                        "one ring element is expected from the source".into(),
                    )),
                }
            }
        }
    }
}

impl<'a> From<&'a Committed> for Witness<'a> {
    fn from(committed: &'a Committed) -> Self {
        Self::Borrowed(committed)
    }
}

/// The next committed witness of an iteration ([`prove_next`]): with its message packed, or with
/// the next witness table, whose elements under $\psi$ are the message ([`pack`]).
pub(crate) struct Next {
    /// The committed witness, its message empty while `table` is kept.
    pub(crate) committed: Committed,
    table: Option<DigitTable>,
}

/// What [`prove_next_tail`] leaves after an iteration: the next committed witness, or after the
/// last iteration with a tail the table and the opening of the tail's commitment to it.
pub(crate) enum After<const K: usize> {
    /// The next committed witness ([`Next`]).
    Next(Next),
    /// After the last iteration with a tail.
    Tail {
        /// The table, which the final step does not send.
        table: DigitTable,
        /// The opening of the tail's commitment ([`Tail::commit`]).
        opening: TailOpening,
        /// The sumcheck's point $\mathbf a$, at which the table takes $y'$.
        point: Vec<Ext<K>>,
    },
}

impl Next {
    /// The table of the next witness, which the last iteration sends in the clear (`pcs`, step
    /// 3), and `None` if it was packed into the message.
    pub(crate) fn into_table(self) -> Option<DigitTable> {
        self.table
    }

    /// The witness of the next iteration: owned, or sourced from the table, its elements packed
    /// when steps 1 and 2 read them, as [`packed_chunk`] packs them (the same elements); the
    /// table is dropped with the source after step 2 ([`Witness::Sourced`]).
    ///
    /// # Errors
    /// [`Error::Shape`] unless the table fills the chunks of the witness's layout.
    pub(crate) fn into_witness<const K: usize>(self, ctx: &Ctx<K>) -> Result<Witness<'_>> {
        let Some(table) = self.table else {
            return Ok(Witness::Owned(self.committed));
        };
        let layout = self.committed.layout;
        let (chunks, len, per) = (layout.chunks(), layout.chunk_len(), ctx.d_over_k());
        if table.len() != chunks * len * per {
            return Err(Error::Shape(
                "the packed witness does not fit the next layout".into(),
            ));
        }
        let elements = move |c: usize, range: core::ops::Range<usize>| -> Result<Vec<Poly>> {
            if c >= chunks || range.start > range.end || range.end > len {
                return Err(Error::Shape("elements past the next message".into()));
            }
            range
                .map(|e| packed_element_at(ctx, &table, (c * len + e) * per))
                .collect()
        };
        Ok(Witness::Sourced {
            committed: self.committed,
            source: Some(OwnedSource {
                chunks,
                chunk_len: len,
                elements: Box::new(elements),
                parts: None,
            }),
        })
    }
}

impl From<Committed> for Witness<'_> {
    fn from(committed: Committed) -> Self {
        Self::Owned(committed)
    }
}

impl Witness<'_> {
    /// The committed witness.
    pub fn committed(&self) -> &Committed {
        match self {
            Self::Borrowed(c) => c,
            Self::Owned(c) => c,
            Self::Sourced { committed, .. } => committed,
        }
    }

    /// The message, as steps 1 and 2 read it.
    pub fn message(&self) -> MessageView<'_> {
        match self {
            Self::Borrowed(c) => MessageView::Held(&c.message),
            Self::Owned(c) => MessageView::Held(&c.message),
            Self::Sourced {
                source: Some(source),
                ..
            } => source.view(),
            Self::Sourced {
                committed,
                source: None,
            } => MessageView::Held(&committed.message),
        }
    }

    /// Frees the message of an owned witness and the source of a sourced one (with what it reads
    /// from), and does nothing to a borrowed one.
    fn free_message(&mut self) {
        match self {
            Self::Owned(c) => c.message = Vec::new(),
            Self::Sourced { source, .. } => *source = None,
            Self::Borrowed(_) => {}
        }
    }
}

/// The prover of one iteration (`index` numbers the iterations from 0 and selects the matrices).
/// `seed` is the opening's secret seed, which only a masked level uses. Returns the messages, the
/// next committed witness and the next claim. An owned `witness` ([`Witness`]) has its message
/// freed after step 2.
///
/// # Errors
/// [`Error::Protocol`] if an honest-prover invariant fails (a bug or a witness outside the
/// layout's bound); arithmetic and sampling errors.
pub fn prove<'w, const K: usize>(
    ctx: &Ctx<K>,
    index: usize,
    params: &IterParams,
    witness: impl Into<Witness<'w>>,
    claim: &Claim,
    fs: &mut Fs,
    seed: &[u8; 32],
) -> Result<(IterationProof<K>, Committed, Claim)> {
    prove_with_w(ctx, index, params, witness, claim, None, fs, seed)
}

/// [`prove`] with step 1's $w_i = \mathbf a^\top \mathbf f_i$ given if `w` is (one per chunk), as
/// an opening forms them with its claim: the same values, so the same messages.
///
/// # Errors
/// As [`prove`], and [`Error::Shape`] for another number of $w_i$.
#[allow(clippy::too_many_arguments)] // the inputs of prove and the given w
pub fn prove_with_w<'w, const K: usize>(
    ctx: &Ctx<K>,
    index: usize,
    params: &IterParams,
    witness: impl Into<Witness<'w>>,
    claim: &Claim,
    w: Option<Vec<Poly>>,
    fs: &mut Fs,
    seed: &[u8; 32],
) -> Result<(IterationProof<K>, Committed, Claim)> {
    let (proof, next, claim) = prove_next(ctx, index, params, witness, claim, w, fs, seed, false)?;
    let claim = claim.ok_or_else(|| {
        Error::Protocol("an iteration that is not the last has a next claim".into())
    })?;
    let mut committed = next.committed;
    if let Some(table) = next.table {
        committed.message = pack(ctx, params, &table)?;
    }
    Ok((proof, committed, claim))
}

/// [`prove_with_w`], with the next witness's table kept ([`Next`]) instead of its message packed:
/// the next iteration reads its message from it, and after the last iteration (`last`), which
/// has no reduction and so no next claim, the final step sends it.
#[allow(clippy::too_many_arguments)] // the inputs of prove_with_w and last
pub(crate) fn prove_next<'w, const K: usize>(
    ctx: &Ctx<K>,
    index: usize,
    params: &IterParams,
    witness: impl Into<Witness<'w>>,
    claim: &Claim,
    w: Option<Vec<Poly>>,
    fs: &mut Fs,
    seed: &[u8; 32],
    last: bool,
) -> Result<(IterationProof<K>, Next, Option<Claim>)> {
    match prove_next_tail(ctx, index, params, witness, claim, w, fs, seed, last, None)? {
        (proof, After::Next(next), claim) => Ok((proof, next, claim)),
        (_, After::Tail { .. }, _) => Err(Error::Protocol(
            "an iteration without a tail commits its next witness".into(),
        )),
    }
}

/// [`prove_next`] with a tail if `tail` is given ([`crate::tail`]): the last iteration (`last`)
/// then commits its table with [`Tail::commit`], absorbs that commitment in place of $\mathbf u'$,
/// and leaves the table with the opening ([`After::Tail`]). Before the last iteration `tail` is
/// not read, and without it this is [`prove_next`].
///
/// # Errors
/// As [`prove_next`], those of [`Tail::commit`], and [`Error::Protocol`] for a commitment of
/// another length than [`Tail::commitment_words`].
#[allow(clippy::too_many_arguments)] // the inputs of prove_next and the tail
pub(crate) fn prove_next_tail<'w, const K: usize>(
    ctx: &Ctx<K>,
    index: usize,
    params: &IterParams,
    witness: impl Into<Witness<'w>>,
    claim: &Claim,
    w: Option<Vec<Poly>>,
    fs: &mut Fs,
    seed: &[u8; 32],
    last: bool,
    tail: Option<&dyn Tail<K>>,
) -> Result<(IterationProof<K>, After<K>, Option<Claim>)> {
    let (ring, field, d) = (&ctx.ring, ctx.field(), ctx.d());
    let (b, delta, tau, n_a) = (ctx.choices.b, params.delta, params.tau, params.n_a);
    let layout = &params.layout;
    let mut held = witness.into();
    let witness = held.committed();
    if witness.layout != *layout {
        return Err(Error::Shape(
            "the witness does not have the iteration's layout".into(),
        ));
    }
    // Step 1.
    let message = held.message();
    let w: Vec<Poly> = match w {
        Some(w) if w.len() == message.chunks() => w,
        Some(_) => {
            return Err(Error::Shape("one w_i per chunk is expected".into()));
        }
        None => {
            let a_row = ring.prepare(core::slice::from_ref(&claim.a))?;
            (0..message.chunks())
                .into_par_iter()
                .map(|c| {
                    ring.mat_vec_prepared(&a_row, &message.chunk(c)?)?
                        .pop()
                        .ok_or_else(|| Error::Shape("a^T f_i is one row".into()))
                })
                .collect::<Result<_>>()?
        }
    };
    let w_hat = g_inv(ring, b, delta, &w)?;
    let d_mat = ctx.matrix(b"D", index, ctx.choices.n_d, params.d_cols())?;
    let width = layout.width(delta);
    let a_mat = ctx.matrix(b"A", index, n_a, width)?;
    let a_rand = rand_matrix(ctx, params)?;
    // Step 2.
    let fold = if layout.masked {
        masked_fold(
            ctx, index, params, witness, message, claim, &w_hat, &d_mat, &a_mat, &a_rand, fs, seed,
        )?
    } else {
        plain_fold(ctx, params, message, &w_hat, &d_mat, fs)?
    };
    // Steps 1 and 2 were the last readers of the message: an owned witness frees it here.
    held.free_message();
    let witness = held.committed();
    let Fold { v, c, z, masks } = fold;
    // Step 3. B, the largest of the three matrices at the secure presets, is read by blocks of
    // columns, here and by M*, so that it is never held whole; D and A, which the fold's attempts
    // read, are held until the quotients.
    let b_cols = layout.chunks() * n_a * delta;
    let parts = masks.as_ref().map(Masks::parts);
    let q = quotients(
        ctx,
        &Public {
            params,
            b: &claim.b,
            a: &claim.a,
            target: &claim.target,
            c: &c,
            d_mat: Mat::Held(&d_mat),
            b_mat: Mat::Expanded {
                kind: b"B",
                iteration: index,
                rows: ctx.choices.n_b,
                cols: b_cols,
            },
            a_mat: Mat::Held(&a_mat),
            a_rand: &a_rand,
            v: &v,
            u: &witness.u,
        },
        &w,
        &w_hat,
        &witness.t,
        &witness.t_hat,
        &z,
        parts.as_ref(),
    )?;
    // The quotients were the last readers of the matrices held whole: M* reads them again below,
    // expanded by blocks of columns, so that they are not held through the next table and the
    // next commitment.
    drop((d_mat, a_mat));
    let public = Public {
        params,
        b: &claim.b,
        a: &claim.a,
        target: &claim.target,
        c: &c,
        d_mat: Mat::Expanded {
            kind: b"D",
            iteration: index,
            rows: ctx.choices.n_d,
            cols: params.d_cols(),
        },
        b_mat: Mat::Expanded {
            kind: b"B",
            iteration: index,
            rows: ctx.choices.n_b,
            cols: b_cols,
        },
        a_mat: Mat::Expanded {
            kind: b"A",
            iteration: index,
            rows: n_a,
            cols: width,
        },
        a_rand: &a_rand,
        v: &v,
        u: &witness.u,
    };
    // z' = (w-hat, [w0-hat, t0-hat,] t-hat, z-hat[, z_varrho-hat]), written into the table part
    // by part, t-hat and z-hat from their bytes. z-hat = J^{-1}(z) is formed here, after the
    // quotients (which read z), by blocks of z in parallel, each block's digits kept as bytes, so
    // that its ring elements are never all held (the same digits in the same order: J^{-1} is
    // entry by entry).
    let mut before = w_hat;
    if let Some(m) = &masks {
        before.extend_from_slice(&m.w0_hat);
        before.extend_from_slice(&m.t0_hat);
    }
    let zq = ring.zq();
    let blocks = z
        .par_chunks(Z_HAT_BLOCK)
        .map(|block| -> Result<Digits> {
            let mut part = Digits::new(d);
            part.extend(zq, &j_inv(ring, b, tau, block)?)?;
            Ok(part)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut z_hat = Digits::new(d);
    for part in &blocks {
        z_hat.append(part);
    }
    drop(blocks);
    let z_r_hat = match &masks {
        Some(m) => j_inv(ring, b, tau, &m.z_r)?,
        None => Vec::new(),
    };
    let table = next_table_parts(
        ctx,
        params,
        &[
            Part::Polys(&before),
            Part::Digits(&witness.t_hat),
            Part::Digits(&z_hat),
            Part::Polys(&z_r_hat),
        ],
        &q,
    )?;
    // The table holds z' and the quotients' digits now: their parts, the masks, the fold and w
    // are not read again, so they are freed before the sumcheck rather than at the end.
    drop((before, z_hat, z_r_hat));
    drop(q);
    drop(masks);
    drop((w, z));
    // The next message is the table under psi, element by element: the commitment packs the
    // elements of each block as it reads them, and the message is packed whole only after the
    // reduction, the last reader of the table, so that the two are never held at once. After the
    // last iteration with a tail, the tail commits to the table instead, under its own label.
    let (next, tail_commitment, tail_opening) = match tail.filter(|_| last) {
        Some(tail) => {
            let (words, opening) = tail.commit(params.table_vars, table.values())?;
            check_words(
                "commitment",
                words.len(),
                tail.commitment_words(params.table_vars),
            )?;
            fs.words(LABEL_TAIL_COMMITMENT, &words);
            (None, Some(words), Some(opening))
        }
        None => {
            let next = commit_table(ctx, index + 1, params.next, &table)?;
            fs.polys(LABEL_U_NEXT, &next.u);
            (Some(next), None, None)
        }
    };
    // Step 4.
    let lr = log_rows(params.rows);
    let coins = fs.field_coins::<K>(LABEL_COINS, ctx.q, 1 + params.table_vars + lr)?;
    let (alpha, tau0, tau1) = (
        coins[0],
        &coins[1..=params.table_vars],
        &coins[1 + params.table_vars..],
    );
    let alpha_pows = powers(field, &alpha, d + 1);
    let (m, _sum) = mstar(ctx, &public, &eq_table(field, tau1), &alpha_pows)?;
    // M* was the last reader of the public data and the challenges.
    drop((a_rand, c));
    // Both weight tables as row and column factors (`sumcheck::Tensor`), split after the
    // variables that index m*, as the verifier splits g below: g = m* ⊗ (1, α, ..., α^{d-1}),
    // and eq(τ0, ·) = eq(τ0_U, ·) ⊗ eq(τ0_L, ·) at the same place.
    let (tau_u, tau_l) = tau0.split_at(params.table_vars - ctx.alpha());
    let eq0 = Tensor {
        rows: eq_table(field, tau_u),
        cols: eq_table(field, tau_l),
    };
    let g = Tensor {
        rows: m,
        cols: alpha_pows[..d].to_vec(),
    };
    let digits = digit_residues(ctx);
    // eq0 is the table of tau0, so each round takes the eq factor of its variable out of the
    // range part and computes it at one node fewer (`sumcheck::prove_at`): the same messages.
    let (rounds, point, final_eval) = sumcheck::prove_at(field, &digits, &table, tau0, eq0, g, fs)?;
    fs.exts(LABEL_FINAL, &[final_eval]);
    // Step 2(j), except after the last iteration, whose table is sent in the clear (step 3). The
    // table is kept, a fraction of the message (bytes against ring elements of d residues), for
    // the next iteration to read its message from, or for the final step.
    let claim = if last {
        None
    } else {
        Some(reduce_prove(ctx, params, &table, &point, final_eval, fs)?)
    };
    let proof = IterationProof {
        v,
        next_commitment: next.as_ref().map_or_else(Vec::new, |n| n.u.clone()),
        rounds,
        final_eval,
        next_target: claim.as_ref().map(|c| c.target.clone()),
        tail_commitment,
    };
    let after = match (next, tail_opening) {
        (Some(committed), None) => After::Next(Next {
            committed,
            table: Some(table),
        }),
        (None, Some(opening)) => After::Tail {
            table,
            opening,
            point,
        },
        _ => {
            return Err(Error::Protocol(
                "an iteration commits its table once".into(),
            ));
        }
    };
    Ok((proof, after, claim))
}

/// The reduction of the sumcheck's output to the next claim (`hachi-pcs.tex`, step 2(j) of
/// `Eval`, as Hachi's Section 3.1 reduces a claim on a polynomial over $\mathbb F_{q^K}$; module
/// documentation). The sumcheck leaves $\tilde w(\mathbf a) =$ `final_eval` at its point
/// $\mathbf a =$ `point`. With the weights $(\mathbf b', \mathbf a')$ of [`next_weights`] at
/// $\mathbf a$, the prover sends $p' = \sum_i b'_i \mathbf a'^\top \mathbf f'_i$, whose trace is
/// $(K/d)\,\mathrm{Tr}\_H(p') = \tilde w(\mathbf a)$ by Theorem 2. Returns the next claim
/// $(\mathbf b', \mathbf a', p')$.
///
/// # Errors
/// [`Error::Protocol`] if the trace of $p'$ is not `final_eval` (an honest-prover invariant);
/// arithmetic errors.
fn reduce_prove<const K: usize, T: ResidueTable + ?Sized>(
    ctx: &Ctx<K>,
    params: &IterParams,
    table: &T,
    point: &[Ext<K>],
    final_eval: Ext<K>,
    fs: &mut Fs,
) -> Result<Claim> {
    let (b_next, a_next) = next_weights(ctx, params, point)?;
    let target = evaluate_packed_claim(ctx, params, &b_next, &a_next, table)?;
    fs.polys(LABEL_TARGET, core::slice::from_ref(&target));
    if scaled_trace(ctx, &target)? != final_eval {
        return Err(Error::Protocol(
            "the trace of the next claim's value is not the sumcheck's value".into(),
        ));
    }
    Ok(Claim {
        b: b_next,
        a: a_next,
        target,
    })
}

/// Checks that `polys` are `count` ring elements of degree $d$ with canonical coefficients.
pub fn check_polys<const K: usize>(
    ctx: &Ctx<K>,
    what: &str,
    polys: &[Poly],
    count: usize,
) -> core::result::Result<(), VerifyError> {
    if polys.len() != count || polys.iter().any(|p| p.0.len() != ctx.d()) {
        return Err(VerifyError::Malformed(format!(
            "{what}: {count} ring elements of {} coefficients expected",
            ctx.d()
        )));
    }
    if polys.iter().flat_map(|p| p.0.iter()).any(|x| *x >= ctx.q) {
        return Err(VerifyError::Malformed(format!(
            "{what}: a coefficient is not below q"
        )));
    }
    Ok(())
}

/// Checks that field elements are canonical.
pub fn check_exts<const K: usize>(
    ctx: &Ctx<K>,
    what: &str,
    exts: &[Ext<K>],
    count: usize,
) -> core::result::Result<(), VerifyError> {
    if exts.len() != count || exts.iter().flat_map(|e| e.0).any(|x| x >= ctx.q) {
        return Err(VerifyError::Malformed(format!(
            "{what}: {count} canonical field elements expected"
        )));
    }
    Ok(())
}

/// The verifier of one iteration on the commitment `commitment` and the claim `claim`. Returns
/// the next claim ([`Reduced::Claim`]), or after the last iteration (`last`), which sends no
/// $p'$, the point and the value that the final witness must have ([`Reduced::Evaluation`]); the
/// next commitment is `proof.next_commitment`.
///
/// # Errors
/// [`VerifyError`] for a malformed or rejected proof.
#[allow(clippy::too_many_arguments)] // the iteration's inputs and whether it is the last
pub fn verify<const K: usize>(
    ctx: &Ctx<K>,
    index: usize,
    params: &IterParams,
    commitment: &[Poly],
    claim: &Claim,
    proof: &IterationProof<K>,
    last: bool,
    fs: &mut Fs,
) -> core::result::Result<Reduced<K>, VerifyError> {
    verify_tail(ctx, index, params, commitment, claim, proof, last, None, fs)
}

/// [`verify`] for an opening with a tail whose commitments have `tail_words` words
/// ([`Tail::commitment_words`] at this iteration's $L$): the last iteration must then send the
/// tail's commitment and no $\mathbf u'$, and every other iteration $\mathbf u'$ and no tail
/// commitment, as without a tail (`None`).
///
/// # Errors
/// As [`verify`].
#[allow(clippy::too_many_arguments)] // the inputs of verify and the tail's commitment words
pub fn verify_tail<const K: usize>(
    ctx: &Ctx<K>,
    index: usize,
    params: &IterParams,
    commitment: &[Poly],
    claim: &Claim,
    proof: &IterationProof<K>,
    last: bool,
    tail_words: Option<usize>,
    fs: &mut Fs,
) -> core::result::Result<Reduced<K>, VerifyError> {
    check_polys(ctx, "v", &proof.v, ctx.choices.n_d)?;
    match (&proof.tail_commitment, tail_words.filter(|_| last)) {
        (Some(words), Some(expected)) => {
            if !proof.next_commitment.is_empty() || words.len() != expected {
                return Err(VerifyError::Malformed(format!(
                    "the last iteration with a tail sends a tail commitment of {expected} words \
                     and no next commitment"
                )));
            }
        }
        (None, None) => check_polys(
            ctx,
            "next commitment",
            &proof.next_commitment,
            ctx.choices.n_b,
        )?,
        _ => {
            return Err(VerifyError::Malformed(
                "only the last iteration with a tail sends a tail commitment".into(),
            ));
        }
    }
    match (&proof.next_target, last) {
        (Some(target), false) => check_polys(ctx, "next target", core::slice::from_ref(target), 1)?,
        (None, true) => {}
        _ => {
            return Err(VerifyError::Malformed(
                "every iteration but the last sends the next claim's value".into(),
            ));
        }
    }
    check_exts(
        ctx,
        "final evaluation",
        core::slice::from_ref(&proof.final_eval),
        1,
    )?;
    for (i, round) in proof.rounds.iter().enumerate() {
        let what = format!("round {}", i + 1);
        check_exts(ctx, &what, &round.h0, round.h0.len())?;
        check_exts(ctx, &what, &round.h_alpha, round.h_alpha.len())?;
    }
    let (point, value) = verify_sumcheck(ctx, index, params, commitment, claim, proof, fs)?;
    match &proof.next_target {
        Some(target) => Ok(Reduced::Claim(reduce_verify(
            ctx, params, target, &point, value, fs,
        )?)),
        None => Ok(Reduced::Evaluation { point, value }),
    }
}

/// The range and linear sumchecks of one iteration (paper, Section 4.3): absorbs $\mathbf v$,
/// draws the challenges, absorbs $\mathbf u'$ (or the tail's commitment if the proof has one),
/// draws the coins, and runs the two sumchecks with common challenges. Returns their point
/// $\mathbf a$ and $y = \tilde w(\mathbf a) =$ `proof.final_eval`, which it checks against both
/// final claims and then absorbs. Shared by [`verify`] and the reduction tests.
///
/// # Errors
/// [`VerifyError::Check`] if a sumcheck's final value is wrong; arithmetic errors.
fn verify_sumcheck<const K: usize>(
    ctx: &Ctx<K>,
    index: usize,
    params: &IterParams,
    commitment: &[Poly],
    claim: &Claim,
    proof: &IterationProof<K>,
    fs: &mut Fs,
) -> core::result::Result<(Vec<Ext<K>>, Ext<K>), VerifyError> {
    let (ring, field, d) = (&ctx.ring, ctx.field(), ctx.d());
    let layout = &params.layout;
    let (delta, n_a) = (params.delta, params.n_a);
    fs.polys(LABEL_V, &proof.v);
    let c: Vec<Poly> = fs
        .challenges(LABEL_C, layout.chunks(), d, ctx.choices.omega)?
        .iter()
        .map(|ci| ring.from_i64(ci))
        .collect::<hachi_co_math::Result<_>>()?;
    match &proof.tail_commitment {
        Some(words) => fs.words(LABEL_TAIL_COMMITMENT, words),
        None => fs.polys(LABEL_U_NEXT, &proof.next_commitment),
    }
    let lr = log_rows(params.rows);
    let coins = fs.field_coins::<K>(LABEL_COINS, ctx.q, 1 + params.table_vars + lr)?;
    let (alpha, tau0, tau1) = (
        coins[0],
        &coins[1..=params.table_vars],
        &coins[1 + params.table_vars..],
    );
    let d_mat = ctx.matrix(b"D", index, ctx.choices.n_d, params.d_cols())?;
    let b_mat = ctx.matrix(b"B", index, ctx.choices.n_b, layout.chunks() * n_a * delta)?;
    let a_mat = ctx.matrix(b"A", index, n_a, layout.width(delta))?;
    let a_rand = rand_matrix(ctx, params)?;
    let public = Public {
        params,
        b: &claim.b,
        a: &claim.a,
        target: &claim.target,
        c: &c,
        d_mat: Mat::Held(&d_mat),
        b_mat: Mat::Held(&b_mat),
        a_mat: Mat::Held(&a_mat),
        a_rand: &a_rand,
        v: &proof.v,
        u: commitment,
    };
    let alpha_pows = powers(field, &alpha, d + 1);
    let (m, sum) = mstar(ctx, &public, &eq_table(field, tau1), &alpha_pows)?;
    let digits = digit_residues(ctx);
    let reduced = sumcheck::verify(
        field,
        digits.len(),
        &proof.rounds,
        params.table_vars,
        sum,
        fs,
    )?;
    let point = reduced.point;
    let y = proof.final_eval;
    let expected0 = field.mul(
        &eq_eval(field, tau0, &point),
        &range_poly(field, &digits, &y),
    );
    if reduced.claim0 != expected0 {
        return Err(VerifyError::Check(
            "the range sumcheck's final value".into(),
        ));
    }
    let split_l = params.table_vars - ctx.alpha();
    let g_at = field.mul(
        &mle_eval(field, &m, &point[..split_l]),
        &mle_eval(field, &alpha_pows[..d], &point[split_l..]),
    );
    if reduced.claim_alpha != field.mul(&y, &g_at) {
        return Err(VerifyError::Check(
            "the linear sumcheck's final value".into(),
        ));
    }
    fs.exts(LABEL_FINAL, &[y]);
    Ok((point, y))
}

/// The verifier's reduction to the next claim (the mirror of [`reduce_prove`]): the weights at
/// the sumcheck's point and the check $(K/d)\,\mathrm{Tr}\_H(p') = y$ of the value $p' =$ `target`
/// that the prover sent.
///
/// # Errors
/// [`VerifyError::Check`] if the trace fails.
fn reduce_verify<const K: usize>(
    ctx: &Ctx<K>,
    params: &IterParams,
    target: &Poly,
    point: &[Ext<K>],
    y: Ext<K>,
    fs: &mut Fs,
) -> core::result::Result<Claim, VerifyError> {
    let (b_next, a_next) = next_weights(ctx, params, point)?;
    fs.polys(LABEL_TARGET, core::slice::from_ref(target));
    if scaled_trace(ctx, target)? != y {
        return Err(VerifyError::Check(
            "the trace of the next claim's value".into(),
        ));
    }
    Ok(Claim {
        b: b_next,
        a: a_next,
        target: target.clone(),
    })
}

/// What steps 1 and 2 of [`prove`] compute at a masked level, with the level's matrices: the
/// shared instance of the cross-check against the Sage spec (`tests/fixtures/check_masked_spec.sage`),
/// which a test of `pcs` writes.
#[cfg(test)]
pub(crate) struct MaskedLevel {
    /// $w_i = \mathbf a^\top \mathbf f_i$ and their digits $\hat{\mathbf w}$.
    pub(crate) w: Vec<Poly>,
    pub(crate) w_hat: Vec<Poly>,
    /// $\mathbf D$, $\mathbf A$, $\mathbf A_1'$ and $\mathbf B$ of the level.
    pub(crate) d_mat: Matrix,
    pub(crate) a_mat: Matrix,
    pub(crate) a_rand: Matrix,
    pub(crate) b_mat: Matrix,
    /// The accepted attempt: $\mathbf v$, $\mathbf c$, $\mathbf z$ and the masked parts.
    pub(crate) v: Vec<Poly>,
    pub(crate) c: Vec<Poly>,
    pub(crate) z: Vec<Poly>,
    pub(crate) w0: Poly,
    pub(crate) w0_hat: Vec<Poly>,
    pub(crate) t0: Vec<Poly>,
    pub(crate) t0_hat: Vec<Poly>,
    pub(crate) z_r: Vec<Poly>,
}

/// Steps 1 and 2 of [`prove`] at a masked level, computed as [`prove`] computes them.
#[cfg(test)]
pub(crate) fn masked_level<const K: usize>(
    ctx: &Ctx<K>,
    index: usize,
    params: &IterParams,
    witness: &Committed,
    claim: &Claim,
    fs: &mut Fs,
    seed: &[u8; 32],
) -> Result<MaskedLevel> {
    let (ring, b, delta, n_a) = (&ctx.ring, ctx.choices.b, params.delta, params.n_a);
    let layout = &params.layout;
    let a_row = ring.prepare(core::slice::from_ref(&claim.a))?;
    let w = witness
        .message
        .iter()
        .map(|fi| {
            ring.mat_vec_prepared(&a_row, fi)?
                .pop()
                .ok_or_else(|| Error::Shape("a^T f_i is one row".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    let w_hat = g_inv(ring, b, delta, &w)?;
    let d_mat = ctx.matrix(b"D", index, ctx.choices.n_d, params.d_cols())?;
    let a_mat = ctx.matrix(b"A", index, n_a, layout.width(delta))?;
    let a_rand = rand_matrix(ctx, params)?;
    let fold = masked_fold(
        ctx,
        index,
        params,
        witness,
        MessageView::Held(&witness.message),
        claim,
        &w_hat,
        &d_mat,
        &a_mat,
        &a_rand,
        fs,
        seed,
    )?;
    let b_mat = ctx.matrix(b"B", index, ctx.choices.n_b, layout.chunks() * n_a * delta)?;
    let masks = fold
        .masks
        .ok_or_else(|| Error::Shape("not a masked level".into()))?;
    Ok(MaskedLevel {
        w,
        w_hat,
        d_mat,
        a_mat,
        a_rand,
        b_mat,
        v: fold.v,
        c: fold.c,
        z: fold.z,
        w0: masks.w0,
        w0_hat: masks.w0_hat,
        t0: masks.t0,
        t0_hat: masks.t0_hat,
        z_r: masks.z_r,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::{Ctx, commit};
    use crate::params::{HachiChoices, Layout, Schedule};
    use hachi_co_sampling::ShakePrg;
    use hachi_co_sampling::uniform::uniform_mod_q;

    const K: usize = 4;
    /// The modulus of Hachi-Co's toy-n1 preset (24 bits).
    const Q: u64 = 8_388_637;
    const CONTEXT: [u8; 32] = [3u8; 32];
    const SEED: [u8; 32] = [1u8; 32];
    /// The verifier's message when the trace of the next claim's value is not $y'$.
    const TRACE_CHECK: &str = "the trace of the next claim's value";

    /// `n` pseudorandom ring elements of degree `d`.
    fn polys(tag: &str, n: usize, d: usize) -> Vec<Poly> {
        uniform_mod_q(&mut ShakePrg::new(&[5u8; 32], tag.as_bytes(), 0), Q, n * d)
            .unwrap()
            .chunks_exact(d)
            .map(|c| Poly(c.to_vec()))
            .collect()
    }

    /// An honest one-iteration instance (no hiding): the context, the iteration's parameters, a
    /// committed first witness and a valid claim on it.
    fn honest_instance() -> (Ctx<K>, IterParams, Committed, Claim) {
        let choices = HachiChoices {
            fixed_iterations: Some(1),
            ..HachiChoices::toy(16)
        };
        let ctx = Ctx::<K>::new(Q, choices.clone()).unwrap();
        let d = ctx.d();
        let layout = Layout {
            r: 2,
            m: 2,
            decomposed: true,
            beta_s: 2,
            masked: false,
            extra: 0,
        };
        let params = Schedule::fixed(&choices, Q, K, layout, 1)
            .unwrap()
            .iterations[0];
        let chunks = layout.chunks();
        let message: Vec<Vec<Poly>> = (0..chunks)
            .map(|i| polys(&format!("f{i}"), layout.chunk_len(), d))
            .collect();
        let witness = commit(&ctx, 0, layout, message.clone(), Vec::new()).unwrap();
        let a = polys("a", layout.chunk_len(), d);
        let b = polys("b", chunks, d);
        let target = evaluate_claim(&ctx, &b, &a, &message).unwrap();
        (ctx, params, witness, Claim { b, a, target })
    }

    /// The acceptance rate of [`rej`] on masks of $D_{\mathfrak s}$, over `trials` attempts, and the
    /// mean of $\mathbf z_0^2 / \mathfrak s^2$ over the accepted ones.
    fn rejection_rate(v: &[Poly], ctx: &Ctx<K>, sigma: u64, trials: u32) -> (f64, f64) {
        let hiding = ctx.hiding().unwrap();
        let gauss = WideGaussian::new(sigma).unwrap();
        let (mut accepted, mut sum) = (0u32, 0f64);
        let mut seeds = ShakePrg::new(&SEED, b"test/rej", 0);
        for _ in 0..trials {
            let r: Vec<Poly> = (0..v.len())
                .map(|_| {
                    let mut x = vec![0i64; ctx.d()];
                    gauss.fill(&mut seeds, &mut x).unwrap();
                    ctx.ring.from_i64(&x).unwrap()
                })
                .collect();
            if rej(ctx, &hiding, sigma, v, &r, &mut seeds).unwrap() {
                accepted += 1;
                let z = ctx.ring.add(&v[0], &r[0]).unwrap();
                let z0 = ctx.ring.to_centered(&z).unwrap()[0] as f64 / sigma as f64;
                sum += z0 * z0;
            }
        }
        (
            f64::from(accepted) / f64::from(trials),
            sum / f64::from(accepted),
        )
    }

    #[test]
    fn rejection_keeps_one_in_m_and_hides_the_fold() {
        // Lemma 21: every attempt is accepted with probability in [1 / M-bar, 1 / M], and an accepted
        // z follows D_s whatever v is. With v = 0 the rate is 1 / M exactly. A v at the lemma's bound
        // has the same rate and the same second moment, within 5 standard errors.
        let choices = HachiChoices::toy_hiding(16);
        let ctx = Ctx::<K>::new(Q, choices).unwrap();
        let hiding = ctx.hiding().unwrap();
        let (num, den) = hiding.log_rejection();
        let m = (num as f64 / den as f64).exp();
        let elements = 20usize;
        let trials = 40_000u32;
        let norm = 3 * (elements * ctx.d()) as u64;
        let sigma = (u64::from(hiding.phi) * norm).div_ceil(16) * 16;
        let zero = vec![ctx.ring.zero(); elements];
        // A v of alternating signs, with ||v||_2 about sigma / (2 phi).
        let large: Vec<Poly> = (0..elements)
            .map(|i| {
                let c: Vec<i64> = (0..ctx.d())
                    .map(|j| if (i + j) % 2 == 0 { 3 } else { -3 } * (elements * ctx.d()) as i64 / 40)
                    .collect();
                ctx.ring.from_i64(&c).unwrap()
            })
            .collect();
        let se_rate = ((1.0 / m) * (1.0 - 1.0 / m) / f64::from(trials)).sqrt();
        for (name, v) in [("zero", &zero), ("large", &large)] {
            let (rate, second) = rejection_rate(v, &ctx, sigma, trials);
            assert!(
                (rate - 1.0 / m).abs() < 5.0 * se_rate,
                "{name}: rate {rate}, 1/M {}",
                1.0 / m
            );
            let se = (2.0 / (f64::from(trials) / m)).sqrt();
            assert!(
                (second - 1.0).abs() < 5.0 * se,
                "{name}: E[z^2] / s^2 = {second}"
            );
        }
    }

    #[test]
    fn honest_iterations_prove_and_verify() {
        // An iteration before the last sends p' and leaves the next claim, the last one sends
        // neither and leaves its point and value, with the same transcript up to the reduction.
        let (ctx, params, witness, claim) = honest_instance();
        let mut fs_p = Fs::new(&CONTEXT);
        let (proof, next, next_claim) =
            prove(&ctx, 0, &params, &witness, &claim, &mut fs_p, &SEED).unwrap();
        assert_eq!(proof.next_target.as_ref(), Some(&next_claim.target));
        assert_eq!(
            evaluate_claim(&ctx, &next_claim.b, &next_claim.a, &next.message).unwrap(),
            next_claim.target,
            "the next claim holds on the packed next witness"
        );
        let mut fs_v = Fs::new(&CONTEXT);
        let reduced = verify(
            &ctx, 0, &params, &witness.u, &claim, &proof, false, &mut fs_v,
        )
        .unwrap();
        assert_eq!(reduced, Reduced::Claim(next_claim));
        assert_eq!(fs_p.digest(), fs_v.digest(), "the transcripts agree");
        let mut fs_last = Fs::new(&CONTEXT);
        let (last, kept, none) = prove_next(
            &ctx,
            0,
            &params,
            &witness,
            &claim,
            None,
            &mut fs_last,
            &SEED,
            true,
        )
        .unwrap();
        assert!(last.next_target.is_none() && none.is_none());
        let table = kept.into_table().unwrap();
        let mut fs_v = Fs::new(&CONTEXT);
        let Reduced::Evaluation { point, value } =
            verify(&ctx, 0, &params, &witness.u, &claim, &last, true, &mut fs_v).unwrap()
        else {
            panic!("the last iteration leaves an evaluation");
        };
        assert_eq!(table_eval(ctx.field(), &table, &point).unwrap(), value);
        assert_eq!(fs_last.digest(), fs_v.digest(), "the transcripts agree");
        // Each kind refuses the other's shape.
        let mut fs_v = Fs::new(&CONTEXT);
        assert!(
            verify(
                &ctx, 0, &params, &witness.u, &claim, &proof, true, &mut fs_v
            )
            .is_err()
        );
        let mut fs_v = Fs::new(&CONTEXT);
        assert!(
            verify(
                &ctx, 0, &params, &witness.u, &claim, &last, false, &mut fs_v
            )
            .is_err()
        );
    }

    #[test]
    fn a_false_next_value_is_rejected() {
        // p' + 1 has the trace y' + 1 (Tr_H(1) = d/K), so the verifier rejects it at the trace
        // check of the reduction.
        let (ctx, params, witness, claim) = honest_instance();
        let (proof, _next, _next_claim) = prove(
            &ctx,
            0,
            &params,
            &witness,
            &claim,
            &mut Fs::new(&CONTEXT),
            &SEED,
        )
        .unwrap();
        let mut forged = proof.clone();
        let target = forged.next_target.as_mut().unwrap();
        target.0[0] = ctx.ring.zq().add(target.0[0], 1);
        let verdict = verify(
            &ctx,
            0,
            &params,
            &witness.u,
            &claim,
            &forged,
            false,
            &mut Fs::new(&CONTEXT),
        );
        assert!(
            matches!(&verdict, Err(VerifyError::Check(m)) if m == TRACE_CHECK),
            "a false next value was accepted or rejected elsewhere: {verdict:?}"
        );
    }

    #[test]
    fn the_reduction_is_exact_for_any_committed_elements() {
        // The argument of Section 3.1 that needs no partial evaluations: for any ring elements f'
        // (not only packings of digits), (K/d) Tr_H(sum_i b'_i a'^T f'_i) with the weights at a
        // point a is the value at a of the table over F_{q^K} that psi^{-1} reads from f'.
        let (ctx, params, _, _) = honest_instance();
        let field = ctx.field();
        let (chunks, len, d) = (params.next.chunks(), params.next.chunk_len(), ctx.d());
        let message: Vec<Vec<Poly>> = (0..chunks)
            .map(|c| polys(&format!("any{c}"), len, d))
            .collect();
        let table: Vec<Ext<K>> = message
            .iter()
            .flatten()
            .flat_map(|f| ctx.sub.psi_inv(f).unwrap())
            .collect();
        assert_eq!(table.len(), 1 << params.table_vars);
        for seed in 0..3u64 {
            let point: Vec<Ext<K>> = uniform_mod_q(
                &mut ShakePrg::new(&[11u8; 32], b"point", seed),
                Q,
                params.table_vars * K,
            )
            .unwrap()
            .as_chunks::<K>()
            .0
            .iter()
            .map(|c| Ext(*c))
            .collect();
            let (b, a) = next_weights(&ctx, &params, &point).unwrap();
            let p = evaluate_claim(&ctx, &b, &a, &message).unwrap();
            assert_eq!(
                scaled_trace(&ctx, &p).unwrap(),
                mle_eval(field, &table, &point),
                "seed {seed}"
            );
        }
    }

    #[test]
    fn the_packed_claim_and_the_table_value_are_the_direct_ones() {
        // The claim on the next message packed chunk by chunk against evaluate_claim on the packed
        // message, and table_eval against mle_eval on the table as constants of F_{q^K}, for an
        // iteration's next layout and a random table.
        let (ctx, params, _, _) = honest_instance();
        let (field, d, per) = (ctx.field(), ctx.d(), ctx.d_over_k());
        let len = params.next.chunks() * params.next.chunk_len() * per;
        assert_eq!(len, 1 << params.table_vars);
        let table: Vec<u64> =
            uniform_mod_q(&mut ShakePrg::new(&[9u8; 32], b"table", 0), Q, len).unwrap();
        let message = pack(&ctx, &params, &table).unwrap();
        let b = polys("b", params.next.chunks(), d);
        let a = polys("a", params.next.chunk_len(), d);
        assert_eq!(
            evaluate_packed_claim(&ctx, &params, &b, &a, &table).unwrap(),
            evaluate_claim(&ctx, &b, &a, &message).unwrap()
        );
        let point: Vec<Ext<K>> = uniform_mod_q(
            &mut ShakePrg::new(&[9u8; 32], b"point", 0),
            Q,
            params.table_vars * K,
        )
        .unwrap()
        .as_chunks::<K>()
        .0
        .iter()
        .map(|c| Ext(*c))
        .collect();
        let constants: Vec<Ext<K>> = table
            .iter()
            .map(|&x| Ext(core::array::from_fn(|c| if c == 0 { x } else { 0 })))
            .collect();
        assert_eq!(
            table_eval(field, &table, &point).unwrap(),
            mle_eval(field, &constants, &point)
        );
        assert!(table_eval(field, &table[1..], &point).is_err());
        assert!(packed_chunk(&ctx, &params, &table[1..], 0).is_err());
        assert!(packed_chunk(&ctx, &params, &table, params.next.chunks()).is_err());
    }

    #[test]
    fn the_next_message_from_the_table_is_the_packed_message() {
        // The next witness sourced from its table (Next::into_witness) reads as the message packed
        // whole (pack), chunk by chunk and element by element; elements past the message and a
        // table that does not fill the layout are refused; without a table the witness is owned.
        let (ctx, params, _, _) = honest_instance();
        let (zq, per) = (ctx.ring.zq(), ctx.d_over_k());
        let len = params.next.chunks() * params.next.chunk_len() * per;
        // Digits of base b, as the next table holds them.
        let (low, high) = digit_range(ctx.choices.b);
        let span = u64::try_from(high - low + 1).unwrap();
        let bytes: Vec<i8> = uniform_mod_q(&mut ShakePrg::new(&[7u8; 32], b"digits", 0), span, len)
            .unwrap()
            .into_iter()
            .map(|x| i8::try_from(i128::from(x) + low).unwrap())
            .collect();
        let mut table = DigitTable::zeros(zq, len);
        table.set_values(0, &bytes).unwrap();
        let packed = pack(&ctx, &params, &table).unwrap();
        let committed = commit_table(&ctx, 1, params.next, &table).unwrap();
        let next = Next {
            committed: committed.clone(),
            table: Some(table),
        };
        let witness = next.into_witness(&ctx).unwrap();
        let message = witness.message();
        assert!(matches!(message, MessageView::Source { .. }));
        assert_eq!(message.chunks(), packed.len());
        for (c, chunk) in packed.iter().enumerate() {
            assert_eq!(&*message.chunk(c).unwrap(), &chunk[..], "chunk {c}");
            let last = chunk.len() - 1;
            assert_eq!(*message.element(c, last).unwrap(), chunk[last], "chunk {c}");
        }
        assert!(message.chunk(packed.len()).is_err());
        let short = Next {
            committed: committed.clone(),
            table: Some(DigitTable::zeros(zq, len - per)),
        };
        assert!(short.into_witness(&ctx).is_err());
        let owned = Next {
            committed,
            table: None,
        };
        assert!(matches!(owned.into_witness(&ctx), Ok(Witness::Owned(_))));
    }

    #[test]
    fn products_by_column_blocks_are_the_whole_products() {
        // Wide vectors (the masks of a masked fold are not short), blocks of 1, 3, 7 and every
        // column, against the matrix prepared whole.
        let ctx = Ctx::<K>::new(Q, HachiChoices::toy(16)).unwrap();
        let ring = &ctx.ring;
        let m = ctx.matrix(b"A", 1, 3, 23).unwrap();
        let v: Vec<Poly> = ctx.matrix(b"v", 1, 1, 23).unwrap().pop().unwrap();
        let want = ring
            .mat_vec_prepared(&ring.prepare(&m).unwrap(), &v)
            .unwrap();
        for step in [1, 3, 7, 23, 100] {
            assert_eq!(
                mat_vec_by_blocks(ring, &m, &v, step).unwrap(),
                want,
                "step {step}"
            );
        }
        assert!(mat_vec_by_blocks(ring, &m, &v[1..], 3).is_err());
    }

    #[test]
    fn the_fold_by_elements_is_the_fold_by_chunks() {
        // fold_message_digits against fold_chunks over Committed::digits: bases that are powers
        // of two and one that is not, the least number of digits and one more, challenges with
        // coefficients of both signs up to 3 in absolute value (one at X^(d-1)), and residues at
        // both ends of Z_q.
        for b in [2u64, 3, 4, 8, 16] {
            let choices = HachiChoices {
                b,
                ..HachiChoices::toy(16)
            };
            let ctx = Ctx::<K>::new(Q, choices).unwrap();
            let (ring, d) = (&ctx.ring, ctx.d());
            let least = hachi_co_math::decompose::num_digits(Q, b).unwrap();
            let layout = Layout {
                r: 2,
                m: 1,
                decomposed: true,
                beta_s: 2,
                masked: false,
                extra: 0,
            };
            let (chunks, len) = (4, 3);
            for delta in [least, least + 1] {
                let mut message: Vec<Vec<Poly>> = (0..chunks)
                    .map(|i| polys(&format!("fold/{b}/{delta}/{i}"), len, d))
                    .collect();
                message[0][0].0[0] = Q - 1;
                message[1][2].0[d - 1] = 0;
                message[2][1].0[3] = (Q - 1) / 2;
                message[3][0].0[1] = Q.div_ceil(2);
                let sparse: Vec<Sparse> = (0..chunks)
                    .map(|i| {
                        let mut c = vec![0i64; d];
                        for (t, x) in [1, -1, 2, -3, 1].into_iter().enumerate() {
                            c[(3 * i + 5 * t) % d] = x;
                        }
                        c[d - 1] = -1;
                        ring.sparse(&c).unwrap()
                    })
                    .collect();
                let want = fold_chunks(&ctx, &sparse, len * delta, |i| {
                    Committed::digits(&ctx, &layout, delta, &message[i])
                })
                .unwrap();
                let got = fold_message_digits(&ctx, delta, &sparse, MessageView::Held(&message))
                    .unwrap()
                    .expect("the sums fit i64");
                assert_eq!(got, want, "b {b} delta {delta}");
                // Both widths of accumulators, whichever the bound chose above.
                for got in [
                    fold_message_digits_i32(&ctx, delta, &sparse, MessageView::Held(&message), len)
                        .unwrap(),
                    fold_message_digits_i64(&ctx, delta, &sparse, MessageView::Held(&message), len)
                        .unwrap(),
                ] {
                    assert_eq!(got, want, "b {b} delta {delta}, both widths");
                }
                assert_eq!(
                    fold_message(&ctx, &layout, delta, &sparse, MessageView::Held(&message))
                        .unwrap(),
                    want
                );
            }
            assert!(
                fold_message_digits(&ctx, least, &[], MessageView::Held(&[vec![ring.zero()]]))
                    .is_err()
            );
        }
    }
}
