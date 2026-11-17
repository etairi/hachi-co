//! The hypercube tables and public values of the sumcheck (`5_Building_Blocks.tex`:234-307;
//! oracle `ref/hachico_ref/sumcheck.py`): the witness table $\langle \widetilde w \rangle\_p$, the
//! row indicators $e\_p$ and $\mathrm{ind}$, the range product, the table of
//! $\widetilde M^\ast$, and the claimed sum $a$.
//!
//! **Hypercube** (`ref/README.md`, "Hypercube"; fixture convention `hypercube`):
//! $V = (U, L)$ with $\lvert U \rvert = \ell\_U = \lceil \log(\mu + n') \rceil$ and
//! $\lvert L \rvert = \ell\_L = \log 2d$; the point $(u\_0, l\_0)$ has index $u\_0 \cdot 2d + l\_0$,
//! $V\_1$ the most significant bit. The draft's 1-indexed $\chi\_{\mathsf{bit}(u) \Vert
//! \mathsf{bit}(\ell)}$ is the entry $(u - 1) \cdot 2d + (\ell - 1)$.

use crate::arith::{Arith, Montgomery};
use crate::eq::eq_table;
use crate::error::{Error, Result, check_len, pow2};
use crate::quotient::{check_matrix, check_row};
use core::ops::Range;
use hachi_co_core::{Context, Layout, Poly, SparseRows};
use hachi_co_math::{Ext, Fqk, Zq};
use rayon::prelude::*;
use std::borrow::Cow;

/// $2^{\ell''}$, the number of points of the hypercube.
///
/// # Errors
/// [`Error::Parameter`] if it does not fit in `usize`.
pub fn table_size(layout: &Layout) -> Result<usize> {
    pow2(layout.ell_pp, "the hypercube")
}

/// The hypercube values of party $p$'s witness polynomial $\langle \widetilde w \rangle\_p(U, L)$
/// (`5_Building_Blocks.tex`:254-259; the oracle's `w_table`; fixture convention `hypercube`), over
/// $\mathbb Z\_q$, $2^{\ell''}$ entries, without the pad: the entry $(u\_0, l\_0)$ is coefficient
/// $l\_0$ of $\langle z'\_{u\_0} \rangle\_p$ for $u\_0 \lt \mu - 1$, $l\_0 \lt d$; coefficient
/// $l\_0$ of $\langle r\_{u\_0 - \mu} \rangle\_p$ for $\mu \le u\_0 \lt \mu + n'$,
/// $l\_0 \lt 2d$; and $0$ otherwise, in particular in the pad's row $\mu - 1$. Residues are
/// reduced modulo $q$.
///
/// # Errors
/// [`Error::Shape`] unless `z_share` has $\mu - 1$ polynomials of $d$ coefficients and `r_share` $n'$
/// polynomials of $2d$ coefficients; [`Error::Parameter`] for a hypercube larger than `usize`.
pub fn w_table<const K: usize>(
    ctx: &Context<K>,
    z_share: &[Poly],
    r_share: &[Poly],
) -> Result<Vec<u64>> {
    w_table_rows(ctx, z_share, r_share)
}

/// The polynomials of a party's share $\langle \mathbf z' \rangle\_p$ as the sumcheck reads
/// them: a dense list, or [`SparseRows`], whose rows not held are zero.
pub(crate) trait ZRows: Sync {
    /// The number of polynomials.
    fn count(&self) -> usize;
    /// Polynomial `j`, the zero polynomial if it is not held.
    fn row(&self, j: usize) -> &Poly;
    /// Whether polynomial `j` is held: one not held is zero.
    fn held(&self, j: usize) -> bool;
}

impl ZRows for [Poly] {
    fn count(&self) -> usize {
        self.len()
    }

    fn row(&self, j: usize) -> &Poly {
        &self[j]
    }

    fn held(&self, _: usize) -> bool {
        true
    }
}

impl ZRows for SparseRows {
    fn count(&self) -> usize {
        self.len()
    }

    fn row(&self, j: usize) -> &Poly {
        self.get(j)
    }

    fn held(&self, j: usize) -> bool {
        self.is_held(j)
    }
}

/// The shapes [`w_table`] checks: $\mu - 1$ polynomials of $d$ coefficients in `z_share` and
/// $n'$ of $2d$ in `r_share`.
fn check_shares<Z: ZRows + ?Sized>(l: &Layout, z_share: &Z, r_share: &[Poly]) -> Result<()> {
    check_len("<z'>_p", l.len_z, z_share.count())?;
    check_len("<r>_p", l.n_prime, r_share.len())?;
    for j in 0..z_share.count() {
        check_len("<z'>_p (coefficients)", l.d, z_share.row(j).0.len())?;
    }
    for p in r_share {
        check_len("<r>_p (coefficients)", 2 * l.d, p.0.len())?;
    }
    Ok(())
}

/// [`w_table`] on a share of $\mathbf z'$ as [`ZRows`]: the same table, the rows not held left
/// zero.
///
/// # Errors
/// As [`w_table`].
pub(crate) fn w_table_rows<const K: usize, Z: ZRows + ?Sized>(
    ctx: &Context<K>,
    z_share: &Z,
    r_share: &[Poly],
) -> Result<Vec<u64>> {
    let l = &ctx.layout;
    let q = ctx.q();
    let two_d = 2 * l.d;
    check_shares(l, z_share, r_share)?;
    let mut table = vec![0u64; table_size(l)?];
    if l.len_z > l.mu || (l.mu + l.n_prime) * two_d > table.len() {
        return Err(Error::Shape {
            what: "rows of the table (z' then r from row mu)",
            expected: table.len() / two_d,
            actual: l.mu + l.n_prime,
        });
    }
    // The rows of z' (those held) and of r, each written in parallel (disjoint rows).
    let (z_rows, rest) = table.split_at_mut(l.mu * two_d);
    z_rows
        .par_chunks_mut(two_d)
        .take(z_share.count())
        .enumerate()
        .filter(|(u, _)| z_share.held(*u))
        .for_each(|(u, row)| {
            for (t, c) in row.iter_mut().zip(&z_share.row(u).0) {
                *t = c % q;
            }
        });
    rest.par_chunks_mut(two_d)
        .zip(r_share.par_iter())
        .for_each(|(row, p)| {
            for (t, c) in row.iter_mut().zip(&p.0) {
                *t = c % q;
            }
        });
    Ok(table)
}

/// A party's base-field table of [`w_table`] as the rounds of
/// [`PartyProver`](crate::PartyProver) read it: entry `x` as a residue and as its centered
/// representative. [`DenseTable`] reads the table itself, [`CompactTable`] its compact form; both
/// give the same entries.
pub(crate) trait BaseTable: Sync {
    /// The number of entries, $2^{\ell''}$.
    fn entries(&self) -> usize;
    /// Entry `x`, reduced modulo $q$.
    fn residue(&self, x: usize) -> u64;
    /// Entry `x` centered, in $(-q/2, q/2]$ ([`Zq::to_centered`]).
    fn centered(&self, x: usize) -> i64;
}

/// The table of [`w_table`] as it is (reduced entries).
pub(crate) struct DenseTable<'a> {
    zq: Zq,
    table: &'a [u64],
}

impl<'a> DenseTable<'a> {
    pub(crate) fn new(zq: Zq, table: &'a [u64]) -> Self {
        Self { zq, table }
    }
}

impl BaseTable for DenseTable<'_> {
    #[inline(always)]
    fn entries(&self) -> usize {
        self.table.len()
    }

    #[inline(always)]
    fn residue(&self, x: usize) -> u64 {
        self.table[x]
    }

    #[inline(always)]
    fn centered(&self, x: usize) -> i64 {
        self.zq.to_centered(self.table[x])
    }
}

/// The table of [`w_table`] held compactly: every entry outside the range `large` as its centered
/// value in one byte, and the entries of `large` as residues. A party's share of $\mathbf z'$ is
/// zero outside its slice $S\_p$ (the slices condition of $\mathcal R\_3$) and its entries in the
/// slice are digits of absolute value below $b$, so only the rows of [`ind_rows`] ($\langle u
/// \rangle$, the pad's and the quotient's) hold large values: $2^{\ell''}$ bytes and $(n' + 2)\, 2d$
/// residues in place of $2^{\ell''}$ residues, with the same entries
/// ([`CompactTable::from_dense`] checks that they fit).
#[derive(Clone, Debug)]
pub struct CompactTable {
    zq: Zq,
    small: Vec<i8>,
    /// The first entry of the range held as residues.
    large_start: usize,
    large: Vec<u64>,
}

impl CompactTable {
    /// `table` (reduced entries) with the entries `large` held as residues, if the centered value
    /// of every other entry lies in $[-128, 127]$; `None` otherwise (or for a range outside the
    /// table).
    pub fn from_dense(zq: Zq, table: &[u64], large: Range<usize>) -> Option<Self> {
        if large.start > large.end || large.end > table.len() {
            return None;
        }
        const CHUNK: usize = 1 << 16;
        let mut small = vec![0i8; table.len()];
        let fits = small.par_chunks_mut(CHUNK).enumerate().all(|(c, chunk)| {
            chunk.iter_mut().enumerate().all(|(i, s)| {
                let x = c * CHUNK + i;
                if large.contains(&x) {
                    return true;
                }
                match i8::try_from(zq.to_centered(table[x])) {
                    Ok(v) => {
                        *s = v;
                        true
                    }
                    Err(_) => false,
                }
            })
        });
        fits.then(|| Self {
            zq,
            small,
            large_start: large.start,
            large: table[large].to_vec(),
        })
    }

    /// The table of [`w_table`] on the same shares, formed compactly without the table itself:
    /// the same entries as [`CompactTable::from_dense`] on it with the entries of the rows of
    /// [`ind_rows`] as residues, or `None` where that would be `None`.
    ///
    /// # Errors
    /// Those of [`w_table`] for shares of the wrong shape.
    pub fn from_share<const K: usize>(
        ctx: &Context<K>,
        z_share: &[Poly],
        r_share: &[Poly],
    ) -> Result<Option<Self>> {
        Self::from_rows(ctx, z_share, r_share)
    }

    /// [`CompactTable::from_share`] on a share of $\mathbf z'$ as [`ZRows`]: the same table, the
    /// rows not held left zero.
    ///
    /// # Errors
    /// As [`CompactTable::from_share`].
    pub(crate) fn from_rows<const K: usize, Z: ZRows + ?Sized>(
        ctx: &Context<K>,
        z_share: &Z,
        r_share: &[Poly],
    ) -> Result<Option<Self>> {
        let l = &ctx.layout;
        let (q, zq, two_d) = (ctx.q(), *ctx.field().zq(), 2 * l.d);
        check_shares(l, z_share, r_share)?;
        let size = table_size(l)?;
        if l.len_z > l.mu || (l.mu + l.n_prime) * two_d > size {
            return Err(Error::Shape {
                what: "rows of the table (z' then r from row mu)",
                expected: size / two_d,
                actual: l.mu + l.n_prime,
            });
        }
        let rows = ind_rows(l);
        let large = rows.start * two_d..rows.end * two_d;
        // The entries of w_table at row u, column c (reduced), as it writes them.
        let entry = |u: usize, c: usize| -> u64 {
            if u < l.len_z {
                if c < l.d { z_share.row(u).0[c] % q } else { 0 }
            } else if u >= l.mu && u < l.mu + l.n_prime {
                r_share[u - l.mu].0[c] % q
            } else {
                0
            }
        };
        let mut small = vec![0i8; size];
        let fits = small
            .par_chunks_mut(two_d)
            .enumerate()
            .take(l.mu + l.n_prime)
            .all(|(u, row)| {
                // A row of z' not held is zero, as `small` is.
                rows.contains(&u)
                    || (u < l.len_z && !z_share.held(u))
                    || row.iter_mut().enumerate().all(|(c, s)| {
                        match i8::try_from(zq.to_centered(entry(u, c))) {
                            Ok(v) => {
                                *s = v;
                                true
                            }
                            Err(_) => false,
                        }
                    })
            });
        Ok(fits.then(|| Self {
            zq,
            small,
            large_start: large.start,
            large: large.map(|x| entry(x / two_d, x % two_d)).collect(),
        }))
    }

    /// The dense table again, reduced: the table given to [`CompactTable::from_dense`].
    pub fn to_dense(&self) -> Vec<u64> {
        (0..self.small.len())
            .into_par_iter()
            .map(|x| self.residue(x))
            .collect()
    }

    /// The bytes and residues it holds.
    pub fn bytes(&self) -> usize {
        self.small.len() + 8 * self.large.len()
    }

    /// A digest of the table (FNV-1a over its bytes, eight at a time, then its length, the start of
    /// its residues and the residues), against which a later reader checks that the table did not
    /// change in between: not a binding.
    pub fn digest(&self) -> u64 {
        let fnv = |h: u64, w: u64| (h ^ w).wrapping_mul(FNV_PRIME);
        let blocks: Vec<u64> = self
            .small
            .par_chunks(FNV_BLOCK)
            .map(|block| {
                block.chunks(8).fold(FNV_OFFSET, |h, word| {
                    let w = word
                        .iter()
                        .enumerate()
                        .fold(0u64, |w, (i, v)| w | (u64::from(*v as u8) << (8 * i)));
                    fnv(h, w)
                })
            })
            .collect();
        let h = blocks.into_iter().fold(FNV_OFFSET, fnv);
        let h = fnv(fnv(h, self.small.len() as u64), self.large_start as u64);
        self.large.iter().fold(h, |h, x| fnv(h, *x))
    }
}

impl BaseTable for CompactTable {
    #[inline(always)]
    fn entries(&self) -> usize {
        self.small.len()
    }

    #[inline(always)]
    fn residue(&self, x: usize) -> u64 {
        let i = x.wrapping_sub(self.large_start);
        if i < self.large.len() {
            self.large[i]
        } else {
            self.zq.from_i64(i64::from(self.small[x]))
        }
    }

    #[inline(always)]
    fn centered(&self, x: usize) -> i64 {
        let i = x.wrapping_sub(self.large_start);
        if i < self.large.len() {
            self.zq.to_centered(self.large[i])
        } else {
            i64::from(self.small[x])
        }
    }
}

/// The monomial coefficients of a party's table held compactly, with a field pad, formed in
/// `parts` parts of consecutive coefficients for
/// [`BasePcs::commit_multilinear_ext_parts`](hachi_co_core::BasePcs::commit_multilinear_ext_parts):
/// the coefficients of [`LiftedCoeffs`](hachi_co_core::base_pcs::LiftedCoeffs) on the same table and
/// pad, without the table or its coefficients held whole. Part $p$ of $P$ coefficients (indices
/// $p P + x$) is the Möbius transform over the bits of $x$ of $\sum\_{h \subseteq p} (-1)^{\lvert p
/// \rvert - \lvert h \rvert} T\[h P + x\]$, the first $\log \mathrm{parts}$ levels of the Möbius
/// transform of the table $T$ taken at $p$ (the transform's levels commute); the pad is added as
/// `LiftedCoeffs` adds it. The fingerprint hashes the parts' blocks of $2^{16}$ in order, as
/// `LiftedCoeffs::fingerprint` hashes its coefficients.
pub(crate) struct CompactCoeffs<'a, const K: usize> {
    field: &'a Fqk<K>,
    table: &'a CompactTable,
    pad: Option<(usize, Ext<K>)>,
    parts: usize,
    /// The part formed last and its coefficients over $\mathbb Z\_q$.
    current: std::sync::RwLock<(usize, Vec<u64>)>,
    /// The FNV-1a state over the blocks of the parts formed so far, and the next part.
    hash: std::sync::Mutex<(u64, usize)>,
}

/// FNV-1a, as `LiftedCoeffs::fingerprint` takes it.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// The blocks that `LiftedCoeffs::fingerprint` hashes.
const FNV_BLOCK: usize = 1 << 16;

impl<'a, const K: usize> CompactCoeffs<'a, K> {
    /// The coefficients of `table` with the pad `pad` in `parts` parts: a power of two dividing
    /// the table whose parts hold a multiple of $2^{16}$ coefficients, or one part.
    pub(crate) fn new(
        field: &'a Fqk<K>,
        table: &'a CompactTable,
        pad: Option<(usize, Ext<K>)>,
        parts: usize,
    ) -> Result<Self> {
        let n = table.entries();
        if !parts.is_power_of_two()
            || !n.is_multiple_of(parts)
            || (parts > 1 && !(n / parts).is_multiple_of(FNV_BLOCK))
        {
            return Err(Error::Parameter(format!(
                "{parts} parts of {n} coefficients"
            )));
        }
        Ok(Self {
            field,
            table,
            pad: pad.map(|(at, v)| (at, field.reduce(&v))),
            parts,
            current: std::sync::RwLock::new((usize::MAX, Vec::new())),
            hash: std::sync::Mutex::new((FNV_OFFSET, 0)),
        })
    }

    /// Forms part `p`, which must be the part after the last one formed for the fingerprint.
    pub(crate) fn prepare(&self, p: usize) -> hachi_co_core::Result<()> {
        let n = self.table.entries();
        let len = n / self.parts;
        if p >= self.parts {
            return Err(hachi_co_core::Error::Shape {
                what: "part of the coefficients",
                expected: self.parts,
                actual: p,
            });
        }
        let (zq, table) = (*self.field.zq(), self.table);
        let mut current = self.current.write().expect("the coefficients' lock");
        let mut part = std::mem::take(&mut current.1);
        part.clear();
        part.resize(len, 0);
        part.par_chunks_mut(1 << 12)
            .enumerate()
            .for_each(|(i, out)| {
                for (j, o) in out.iter_mut().enumerate() {
                    let x = (i << 12) + j;
                    // The sum over the subsets h of p, with the sign of |p| - |h|.
                    let (mut h, mut acc) = (p, 0u64);
                    loop {
                        let v = table.residue(h * len + x);
                        acc = if (p ^ h).count_ones().is_multiple_of(2) {
                            zq.add(acc, v)
                        } else {
                            zq.sub(acc, v)
                        };
                        if h == 0 {
                            break;
                        }
                        h = (h - 1) & p;
                    }
                    *o = acc;
                }
            });
        hachi_co_core::base_pcs::mobius(&zq, &mut part)?;
        let mut hash = self.hash.lock().expect("the fingerprint's lock");
        if hash.1 == p {
            let fnv = |h: u64, w: u64| (h ^ w).wrapping_mul(FNV_PRIME);
            let blocks: Vec<u64> = part
                .par_chunks(FNV_BLOCK)
                .map(|block| block.iter().fold(FNV_OFFSET, |h, x| fnv(h, *x)))
                .collect();
            hash.0 = blocks.into_iter().fold(hash.0, fnv);
            hash.1 = p + 1;
        }
        *current = (p, part);
        Ok(())
    }

    /// Coefficients `start .. start + out.len()`, within the part formed last.
    pub(crate) fn read(&self, start: usize, out: &mut [Ext<K>]) {
        let f = self.field;
        let current = self.current.read().expect("the coefficients' lock");
        let (p, part) = (&current.0, &current.1);
        let first = p * part.len();
        for (i, o) in (start..).zip(out.iter_mut()) {
            let x = f.from_base(part[i - first]);
            *o = match self.pad {
                Some((at, v)) if i & at == at => {
                    if (i ^ at).count_ones().is_multiple_of(2) {
                        f.add(&x, &v)
                    } else {
                        f.sub(&x, &v)
                    }
                }
                _ => x,
            };
        }
    }

    /// The fingerprint, once every part was formed in order: that of `LiftedCoeffs` on the
    /// same table and pad.
    pub(crate) fn fingerprint(&self) -> u64 {
        let fnv = |h: u64, w: u64| (h ^ w).wrapping_mul(FNV_PRIME);
        let hash = self.hash.lock().expect("the fingerprint's lock");
        debug_assert_eq!(hash.1, self.parts, "every part formed");
        let mut h = fnv(hash.0, self.table.entries() as u64);
        if let Some((at, value)) = self.pad {
            h = value
                .0
                .iter()
                .fold(fnv(h, at as u64 + 1), |h, x| fnv(h, *x));
        }
        h
    }
}

/// The rows of a table of `two_d` entries per row whose entries are all zero, the pad's row
/// excluded (a field pad is held outside the table, which has 0 at its index): the rows on which
/// $\langle \widetilde w \rangle\_p$ vanishes, where $F\_{0,p}$ and $F\_{1,p}$ vanish too.
pub(crate) fn zero_rows<B: BaseTable>(
    table: &B,
    two_d: usize,
    pad_row: Option<usize>,
) -> Vec<bool> {
    let rows = table.entries() / two_d;
    (0..rows)
        .into_par_iter()
        .map(|u| Some(u) != pad_row && (u * two_d..(u + 1) * two_d).all(|x| table.residue(x) == 0))
        .collect()
}

/// The table `base` of [`w_table`] over $\mathbb F\_{q^K}$, with the pad's entry `pad` (its index
/// and value) set when the pad is a field element: the hypercube values of
/// $\langle \widetilde w \rangle\_p$, the pad included.
pub(crate) fn lift_table<const K: usize>(
    field: &Fqk<K>,
    base: &[u64],
    pad: Option<(usize, Ext<K>)>,
) -> Vec<Ext<K>> {
    let mut table: Vec<Ext<K>> = base.iter().map(|x| field.from_base(*x)).collect();
    if let Some((at, value)) = pad {
        table[at] = value;
    }
    table
}

/// The table index of the pad's entry $(\mu, 1)$ (`5_Building_Blocks.tex`:256): row $\mu - 1$,
/// coefficient 0.
pub fn pad_index(layout: &Layout) -> usize {
    layout.pad * 2 * layout.d
}

/// Writes the residues of [`PadRule::Residues`](hachi_co_params::PadRule) into the pad's row, from
/// its first entry.
///
/// # Errors
/// [`Error::Shape`] unless there are $2k$ residues.
pub fn put_pad_residues(layout: &Layout, table: &mut [u64], residues: &[u64]) -> Result<()> {
    check_len("pad residues (2k)", 2 * layout.k, residues.len())?;
    let at = pad_index(layout);
    table[at..at + residues.len()].copy_from_slice(residues);
    Ok(())
}

/// The rows $u$ of the draft's $\sum\_{u = \mu - 1}^{\mu + n'} \chi\_{\mathsf{bit}(u)}(U)$
/// (`5_Building_Blocks.tex`:302): the 1-indexed rows $\mu - 1, \dots, \mu + n'$ are the 0-indexed
/// $\mu - 2, \dots, \mu + n' - 1$, i.e. $\langle u \rangle$, the pad's row and the $n'$ quotient
/// rows (the oracle's `ind_rows`; `ref/README.md`, choice 8).
pub fn ind_rows(layout: &Layout) -> Range<usize> {
    layout.u..layout.mu + layout.n_prime
}

/// The $0/1$ table over $U$ ($2^{\ell\_U}$ entries) that is 1 on the rows `rows`: $e\_p(U) =
/// \sum\_{u \in S\_p} \chi\_{\mathsf{bit}(u)}(U)$ for the ranges [`Layout::s_ranges`]`(p)`, or
/// $\mathrm{ind}(U)$ for [`ind_rows`] (`5_Building_Blocks.tex`:294, 302). The oracle's
/// `row_indicator` broadcasts it over $L$; both factors are constant in $L$, so the provers keep
/// them as tables over $U$.
///
/// # Errors
/// [`Error::Shape`] if a range leaves $\[0, 2^{\ell\_U})$.
pub fn row_indicator(layout: &Layout, rows: &[Range<usize>]) -> Result<Vec<u64>> {
    let size = pow2(layout.ell_u, "the U part of the hypercube")?;
    let mut table = vec![0u64; size];
    for r in rows {
        if r.start > r.end || r.end > size {
            return Err(Error::Shape {
                what: "row range (within the U part of the hypercube)",
                expected: size,
                actual: r.end,
            });
        }
        for x in &mut table[r.clone()] {
            *x = 1;
        }
    }
    Ok(table)
}

/// The range polynomial $\prod\_{c \in \lbrace -b+1, \dots, b-1 \rbrace} (W - c)$ of
/// $F\_{0,p}$ (`5_Building_Blocks.tex`:301; the oracle's `range_product`), evaluated as
/// $W \prod\_{c=1}^{b-1} (W^2 - c^2)$: $b$ multiplications instead of $2b - 2$ (the empty product
/// for $b = 0$ is $1$).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeProduct {
    b: u64,
    /// $c^2 \bmod q$ for $c = 1, \dots, b - 1$.
    squares: Vec<u64>,
}

impl RangeProduct {
    /// The range product for the base $b$.
    pub fn new(zq: &Zq, b: u64) -> Self {
        Self {
            b,
            squares: (1..b.max(1)).map(|c| zq.mul(c, c)).collect(),
        }
    }

    /// $c^2 \bmod q$ for $c = 1, \dots, b - 1$.
    pub(crate) fn squares(&self) -> &[u64] {
        &self.squares
    }

    /// $\prod\_{c = -b+1}^{b-1} (w - c)$.
    pub fn eval<const K: usize>(&self, field: &Fqk<K>, w: &Ext<K>) -> Ext<K> {
        if self.b == 0 {
            return field.one();
        }
        let mut acc = *w;
        if let Some((first, rest)) = self.squares.split_first() {
            let w2 = field.square(w);
            let mut prod = sub_base(field, &w2, *first);
            for c2 in rest {
                prod = field.mul(&prod, &sub_base(field, &w2, *c2));
            }
            acc = field.mul(&prod, w);
        }
        acc
    }
}

/// $\prod\_{c = -b+1}^{b-1} (w - c)$ ([`RangeProduct`] for one evaluation).
pub fn range_product<const K: usize>(field: &Fqk<K>, w: &Ext<K>, b: u64) -> Ext<K> {
    RangeProduct::new(field.zq(), b).eval(field, w)
}

/// $x - c$ for a base-field constant $c$ (only coordinate $a\_0$ changes, Hachi Eq. (7)).
pub(crate) fn sub_base<const K: usize>(field: &Fqk<K>, x: &Ext<K>, c: u64) -> Ext<K> {
    let mut out = field.reduce(x);
    out.0[0] = field.zq().sub(out.0[0], c);
    out
}

/// $(x^0, x^1, \dots, x^{\mathrm{count} - 1})$.
pub fn powers<const K: usize>(field: &Fqk<K>, x: &Ext<K>, count: usize) -> Vec<Ext<K>> {
    let mut out = Vec::with_capacity(count);
    let mut current = field.one();
    for _ in 0..count {
        out.push(current);
        current = field.mul(&current, x);
    }
    out
}

/// $\sum\_j c\_j \alpha^j$ for a polynomial with base-field coefficients $c\_j$, given the powers
/// $\alpha^j$ (at least as many as coefficients).
pub(crate) fn eval_at<const K: usize>(field: &Fqk<K>, coeffs: &[u64], powers: &[Ext<K>]) -> Ext<K> {
    let zq = field.zq();
    let mut acc = field.zero();
    for (c, p) in coeffs.iter().zip(powers) {
        let c = zq.reduce(*c);
        if c != 0 {
            acc = field.add(&acc, &field.scale(p, c));
        }
    }
    acc
}

/// Checks the $d \times d$ matrix of $E$ and the length of the $\widetilde{eq}(\tau\_1, \cdot)$
/// table ($2^{\lceil \log \nu \rceil}$).
fn check_e_and_eq1(layout: &Layout, e_matrix: &[Vec<u64>], eq1_len: usize) -> Result<()> {
    check_len("E matrix (rows)", layout.d, e_matrix.len())?;
    for row in e_matrix {
        check_len("E matrix (columns)", layout.d, row.len())?;
    }
    check_len(
        "eq(tau_1) table (2^ceil(log nu) entries)",
        pow2(layout.log_nu, "the eq(tau_1) table")?,
        eq1_len,
    )
}

/// $\widetilde{eq}(\tau\_1, \cdot)$ as two half tables: entry $i$ of
/// [`eq_table`](crate::eq_table)$(\tau\_1)$ is $\mathrm{hi}\[i \gg s\] \cdot \mathrm{lo}\[i \bmod 2^s\]$, with
/// $\mathrm{hi}$ the table of the first $\lvert \tau\_1 \rvert - s$ coordinates and $\mathrm{lo}$
/// that of the last $s = \lfloor \lvert \tau\_1 \rvert / 2 \rfloor$ (the most significant bit
/// goes with the first coordinate, as in [`eq_table`](crate::eq_table)). The product is exact, so each entry is that of
/// the whole table, which [`mstar_table_rows`] then never holds ($2^{\lceil \log \nu \rceil}$
/// values, 2 GiB at secure-l24-n2-a8).
///
/// With Montgomery's form ([`Montgomery`], where the modulus admits it) `hi` is held in it, so
/// that the product $\phi(\mathrm{hi}) \cdot \mathrm{lo}$ is the canonical entry, in about half the
/// time of the canonical product.
#[derive(Clone, Debug)]
pub struct EqSplit<const K: usize> {
    /// In the working representation of `mont`.
    hi: Vec<Ext<K>>,
    lo: Vec<Ext<K>>,
    s: usize,
    mont: Option<Montgomery<K>>,
}

impl<const K: usize> EqSplit<K> {
    /// The two halves of $\widetilde{eq}(\boldsymbol\tau, \cdot)$.
    ///
    /// # Errors
    /// As [`eq_table`](crate::eq_table).
    pub fn new(field: &Fqk<K>, tau: &[Ext<K>]) -> Result<Self> {
        let s = tau.len() / 2;
        let (hi, lo) = tau.split_at(tau.len() - s);
        let mont = Montgomery::<K>::new(field.modulus());
        let mut hi = eq_table(field, hi)?;
        if let Some(m) = &mont {
            hi.iter_mut().for_each(|v| *v = m.enter(v));
        }
        Ok(Self {
            hi,
            lo: eq_table(field, lo)?,
            s,
            mont,
        })
    }

    /// The number of entries, $2^{\lvert \boldsymbol\tau \rvert}$.
    pub fn len(&self) -> usize {
        self.hi.len() << self.s
    }

    /// Whether there are no entries (never: the empty point has the table $(1)$).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Entry $i$ of the whole table, for $i$ below [`EqSplit::len`].
    #[inline]
    pub fn at(&self, field: &Fqk<K>, i: usize) -> Ext<K> {
        let (hi, lo) = (&self.hi[i >> self.s], &self.lo[i & ((1 << self.s) - 1)]);
        match &self.mont {
            Some(m) => m.mul(hi, lo),
            None => field.mul(hi, lo),
        }
    }
}

/// The weights $c\_u = \sum\_{i' \lt n'} \widetilde{eq}(\tau\_1, \mathsf{bit}(i'))\, M\_{i',u}(\alpha)$
/// of the columns $u \lt \mu - 1$ of $\mathbf M$ (fixture convention `M_tilde`), from the powers
/// $\alpha^0, \dots, \alpha^{d-1}$. Zero coefficients of $\mathbf M$ are skipped.
pub(crate) fn column_weights<const K: usize>(
    field: &Fqk<K>,
    m: &[Vec<Poly>],
    eq1: &[Ext<K>],
    apow: &[Ext<K>],
) -> Vec<Ext<K>> {
    let cols = m.first().map_or(0, Vec::len);
    let mut c = vec![field.zero(); cols];
    for (row, e) in m.iter().zip(eq1) {
        add_row_weights(field, &mut c, row, e, apow);
    }
    c
}

/// [`column_weights`] with the $n'$ rows of $\mathbf M$ read one at a time from `rows`, in order,
/// each of `cols` polynomials of `d` coefficients and dropped after it is added: the same weights
/// (the same additions in the same order), without holding $\mathbf M$.
///
/// # Errors
/// [`Error::Shape`] for a row of the wrong shape, and the errors of `rows`.
pub(crate) fn column_weights_rows<const K: usize>(
    field: &Fqk<K>,
    rows: &(dyn Fn(usize) -> hachi_co_core::Result<Vec<Poly>> + Sync),
    n_rows: usize,
    cols: usize,
    d: usize,
    eq1: &[Ext<K>],
    apow: &[Ext<K>],
) -> Result<Vec<Ext<K>>> {
    let mut c = vec![field.zero(); cols];
    for (i, e) in eq1.iter().enumerate().take(n_rows) {
        let row = rows(i)?;
        check_row(&row, cols, d)?;
        add_row_weights(field, &mut c, &row, e, apow);
        // The row's many ring elements are dropped in parallel.
        row.into_par_iter().for_each(drop);
    }
    Ok(c)
}

/// Adds row $i'$ of $\mathbf M$, of weight $e = \widetilde{eq}(\tau\_1, \mathsf{bit}(i'))$, to the
/// column weights of [`column_weights`]: $c\_u \mathrel{+}= \sum\_j M\_{i',u,j} \beta\_j$ with
/// $\beta\_j = e \alpha^j$. The columns run in parallel, each over its coefficients in order.
fn add_row_weights<const K: usize>(
    field: &Fqk<K>,
    c: &mut [Ext<K>],
    row: &[Poly],
    e: &Ext<K>,
    apow: &[Ext<K>],
) {
    let zq = field.zq();
    let beta: Vec<Ext<K>> = apow.iter().map(|p| field.mul(e, p)).collect();
    c.par_iter_mut().zip(row.par_iter()).for_each(|(cu, poly)| {
        for (coef, b) in poly.0.iter().zip(&beta) {
            let coef = zq.reduce(*coef);
            if coef != 0 {
                *cu = field.add(cu, &field.scale(b, coef));
            }
        }
    });
}

/// The hypercube values of
/// $$M^\ast(U, L) = \sum\_{i' \in \[\nu\]} \widetilde{eq}(\tau\_1, \mathsf{bit}(i'))\,
/// \widetilde M\_\alpha(\mathsf{bit}(i'), U, L)$$
/// (`5_Building_Blocks.tex`:286-292, 303; the oracle's `mstar_table`), $2^{\ell''}$ elements of
/// $\mathbb F\_{q^k}$, by the fixture convention `M_tilde`: at $(u\_0, l\_0)$,
/// - $c\_u \alpha^{l\_0}$ for $u\_0 \lt \mu - 1$ and every $l\_0 \lt 2d$, with
///   $c\_u = \sum\_{i\_0 \lt n'} \mathrm{eq}\_1\[i\_0\] M\_{i\_0, u\_0}(\alpha)$ (first term, $u \le \mu - 1$;
///   `ref/README.md`, choice 10);
/// - $-\mathrm{eq}\_1\[u\_0 - \mu\] (\alpha^d + 1) \alpha^{l\_0}$ for $\mu \le u\_0 \lt \mu + n'$ (first
///   term, $u = i' + \mu$);
/// - plus $\sum\_{j\_0 \lt d} \mathrm{eq}\_1\[n' + j\_0\] E\[j\_0\]\[l\_0\]$ at $u\_0 = \mu - 2$,
///   $l\_0 \lt d$ (second term, the $E$ rows; choice 9);
/// - plus $\mathrm{eq}\_1\[n' + d + u\_0 + (l\_0 - d)(\mu + n')\]$ for $u\_0 \lt \mu + n'$,
///   $l\_0 \ge d$ (third term, the high coefficients);
/// - and $0$ at the pad's entries $(\mu - 1, l\_0)$, $l\_0 \lt d$, and on the padding rows
///   $u\_0 \ge \mu + n'$,
///
/// where $\mathrm{eq}\_1$ is [`eq_table`](crate::eq_table)$(\tau\_1)$ and $E\[j\]\[l\]$ is
/// coefficient $j$ of $E(X^l)$ ([`hachi_co_math::Subfield::e_matrix`]). The cost is
/// $O(\text{nonzero coefficients of } \mathbf M + 2^{\ell''} + d^2)$ operations; the verifier
/// evaluates the multilinear extension at one point without the table
/// ([`mstar_eval_factorized`](crate::mstar_eval_factorized)).
///
/// # Errors
/// [`Error::Shape`] unless $\mathbf M$ is $n' \times (\mu - 1)$ with $d$ coefficients per entry,
/// $E$ is $d \times d$ and $\mathrm{eq}\_1$ has $2^{\lceil \log \nu \rceil}$ entries.
pub fn mstar_table<const K: usize>(
    ctx: &Context<K>,
    m: &[Vec<Poly>],
    e_matrix: &[Vec<u64>],
    eq1: &[Ext<K>],
    alpha: &Ext<K>,
) -> Result<Vec<Ext<K>>> {
    let l = &ctx.layout;
    check_matrix(m, l.n_prime, l.len_z, l.d)?;
    check_e_and_eq1(l, e_matrix, eq1.len())?;
    mstar_table_with(
        ctx,
        |i| Ok(Cow::Borrowed(&m[i][..])),
        e_matrix,
        |i| eq1[i],
        alpha,
    )
}

/// [`mstar_table`] with the rows of $\mathbf M$ read one at a time from `rows` (row $i$ of the
/// $n'$, each read once, in order) and $\mathrm{eq}\_1$ as an [`EqSplit`], so that neither
/// $\mathbf M$ nor the whole $\mathrm{eq}\_1$ is held: the same table, as the column weights add
/// up row by row in the same order and every entry of $\mathrm{eq}\_1$ is the same.
///
/// # Errors
/// As [`mstar_table`], for a row of the wrong shape when it is read, and those of `rows`.
pub fn mstar_table_rows<'m, const K: usize>(
    ctx: &Context<K>,
    rows: impl FnMut(usize) -> Result<Cow<'m, [Poly]>>,
    e_matrix: &[Vec<u64>],
    eq1: &EqSplit<K>,
    alpha: &Ext<K>,
) -> Result<Vec<Ext<K>>> {
    check_e_and_eq1(&ctx.layout, e_matrix, eq1.len())?;
    let field = ctx.field();
    mstar_table_with(ctx, rows, e_matrix, |i| eq1.at(field, i), alpha)
}

/// The table of [`mstar_table`] from the rows of $\mathbf M$ and the entries of
/// $\mathrm{eq}\_1$ (checked by the callers, but the rows' shapes, checked as they are read).
fn mstar_table_with<'m, const K: usize>(
    ctx: &Context<K>,
    mut rows: impl FnMut(usize) -> Result<Cow<'m, [Poly]>>,
    e_matrix: &[Vec<u64>],
    eq1: impl Fn(usize) -> Ext<K> + Sync,
    alpha: &Ext<K>,
) -> Result<Vec<Ext<K>>> {
    let l = &ctx.layout;
    let field = ctx.field();
    let (d, mu, np) = (l.d, l.mu, l.n_prime);
    let two_d = 2 * d;
    let apow = powers(field, alpha, two_d);
    let mut c = vec![field.zero(); l.len_z];
    for i in 0..np {
        let row = rows(i)?;
        check_row(&row, l.len_z, d)?;
        add_row_weights(field, &mut c, &row, &eq1(i), &apow[..d]);
    }
    let mut table = vec![field.zero(); table_size(l)?];
    // First term, u0 < mu - 1: c_u alpha^{l0} for all l0 < 2d (rows in parallel).
    table
        .par_chunks_mut(two_d)
        .zip(c.par_iter())
        .for_each(|(row, cu)| {
            for (t, p) in row.iter_mut().zip(&apow) {
                *t = field.mul(cu, p);
            }
        });
    // First term, u0 = mu + i0: -eq1[i0] (alpha^d + 1) alpha^{l0}.
    let ad1 = field.add(&apow[d], &field.one());
    for i0 in 0..np {
        let coef = field.neg(&field.mul(&eq1(i0), &ad1));
        for (l0, p) in apow.iter().enumerate() {
            table[(mu + i0) * two_d + l0] = field.mul(&coef, p);
        }
    }
    // Second term: rows n' + 1 .. n' + d (1-indexed) at u0 = mu - 2, l0 < d.
    let zq = field.zq();
    for l0 in 0..d {
        let mut acc = field.zero();
        for (j0, row) in e_matrix.iter().enumerate() {
            let coef = zq.reduce(row[l0]);
            if coef != 0 {
                acc = field.add(&acc, &field.scale(&eq1(np + j0), coef));
            }
        }
        let x = l.u * two_d + l0;
        table[x] = field.add(&table[x], &acc);
    }
    // Third term: u0 < mu + n', l0 >= d, i' - 1 = n' + d + u0 + (l0 - d)(mu + n') (rows in
    // parallel; the second term is on the columns l0 < d).
    table
        .par_chunks_mut(two_d)
        .take(mu + np)
        .enumerate()
        .for_each(|(u0, row)| {
            for (l0, t) in row.iter_mut().enumerate().skip(d) {
                let i = np + d + u0 + (l0 - d) * (mu + np);
                *t = field.add(t, &eq1(i));
            }
        });
    Ok(table)
}

/// The parts of [`mstar_table`] from which each entry is formed when it is read
/// ([`MStarParts::at`]), so that the table of $2^{\ell''}$ entries is not held before the first
/// bind (4 GiB at secure-l24-n2-a8): the row factors of the first term ($c\_u$ for
/// $u\_0 \lt \mu - 1$, $-\mathrm{eq}\_1\[u\_0 - \mu\] (\alpha^d + 1)$ for $\mu \le u\_0 \lt \mu + n'$, zero
/// otherwise), the powers $\alpha^{l\_0}$, the second term's row and $\mathrm{eq}\_1$ as an
/// [`EqSplit`]. Each entry is formed by the field operations that [`mstar_table_rows`] uses for
/// it, so it is the same element.
#[derive(Clone, Debug)]
pub struct MStarParts<const K: usize> {
    d: usize,
    mu: usize,
    np: usize,
    /// The row of the second term ($\mathbf E$).
    u: usize,
    /// $\log 2d$: the point $(u\_0, l\_0)$ has index $u\_0 \cdot 2d + l\_0$.
    shift: usize,
    size: usize,
    apow: Vec<Ext<K>>,
    /// The first term's factor of each row $u\_0 \lt \mu + n'$, in the working representation of
    /// `mont`: $\phi(c) \alpha^{l\_0}$ is the canonical product.
    rows: Vec<Ext<K>>,
    /// Montgomery's form, where the modulus admits it.
    mont: Option<Montgomery<K>>,
    /// $\sum\_{j\_0} \mathrm{eq}\_1\[n' + j\_0\] E\[j\_0\]\[l\_0\]$ for $l\_0 \lt d$.
    second: Vec<Ext<K>>,
    eq1: EqSplit<K>,
}

impl<const K: usize> MStarParts<K> {
    /// The parts of [`mstar_table_rows`] with the same inputs: the rows of $\mathbf M$ are read
    /// once, in order, for the column weights.
    ///
    /// # Errors
    /// As [`mstar_table_rows`]; [`Error::Parameter`] if $2d$ is not a power of two.
    pub fn from_rows<'m>(
        ctx: &Context<K>,
        mut rows: impl FnMut(usize) -> Result<Cow<'m, [Poly]>>,
        e_matrix: &[Vec<u64>],
        eq1: EqSplit<K>,
        alpha: &Ext<K>,
    ) -> Result<Self> {
        let l = &ctx.layout;
        check_e_and_eq1(l, e_matrix, eq1.len())?;
        let field = ctx.field();
        let (d, mu, np) = (l.d, l.mu, l.n_prime);
        let two_d = 2 * d;
        if !two_d.is_power_of_two() {
            return Err(Error::Parameter(format!(
                "2d = {two_d} is not a power of two"
            )));
        }
        let size = table_size(l)?;
        if (mu + np) * two_d > size || l.len_z > mu + np || (l.u + 1) * two_d > size {
            return Err(Error::Shape {
                what: "rows of M* (mu + n' rows of 2d points)",
                expected: size / two_d,
                actual: mu + np,
            });
        }
        let apow = powers(field, alpha, two_d);
        let mut c = vec![field.zero(); l.len_z];
        for i in 0..np {
            let row = rows(i)?;
            check_row(&row, l.len_z, d)?;
            add_row_weights(field, &mut c, &row, &eq1.at(field, i), &apow[..d]);
            // A row built for this call is dropped in parallel (its ring elements are many).
            if let Cow::Owned(row) = row {
                row.into_par_iter().for_each(drop);
            }
        }
        let mut factors = vec![field.zero(); mu + np];
        factors[..l.len_z].copy_from_slice(&c);
        let ad1 = field.add(&apow[d], &field.one());
        for i0 in 0..np {
            factors[mu + i0] = field.neg(&field.mul(&eq1.at(field, i0), &ad1));
        }
        let zq = field.zq();
        let second = (0..d)
            .map(|l0| {
                let mut acc = field.zero();
                for (j0, row) in e_matrix.iter().enumerate() {
                    let coef = zq.reduce(row[l0]);
                    if coef != 0 {
                        acc = field.add(&acc, &field.scale(&eq1.at(field, np + j0), coef));
                    }
                }
                acc
            })
            .collect();
        let mont = Montgomery::<K>::new(field.modulus());
        if let Some(m) = &mont {
            factors.iter_mut().for_each(|v| *v = m.enter(v));
        }
        Ok(Self {
            d,
            mu,
            np,
            u: l.u,
            shift: two_d.trailing_zeros() as usize,
            size,
            apow,
            rows: factors,
            mont,
            second,
            eq1,
        })
    }

    /// The number of entries, $2^{\ell''}$.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Whether there are no entries (never: the hypercube has at least $2d$ points).
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Entry $x$ of [`mstar_table`], for $x$ below [`MStarParts::len`].
    #[inline]
    pub fn at(&self, field: &Fqk<K>, x: usize) -> Ext<K> {
        let (u0, l0) = (x >> self.shift, x & ((1 << self.shift) - 1));
        let mut v = match (self.rows.get(u0), &self.mont) {
            (Some(factor), Some(m)) => m.mul(factor, &self.apow[l0]),
            (Some(factor), None) => field.mul(factor, &self.apow[l0]),
            (None, _) => field.zero(),
        };
        if u0 == self.u && l0 < self.d {
            v = field.add(&v, &self.second[l0]);
        }
        if u0 < self.mu + self.np && l0 >= self.d {
            let i = self.np + self.d + u0 + (l0 - self.d) * (self.mu + self.np);
            v = field.add(&v, &self.eq1.at(field, i));
        }
        v
    }

    /// The whole table, [`MStarParts::at`] at every point.
    pub fn table(&self, field: &Fqk<K>) -> Vec<Ext<K>> {
        (0..self.size)
            .into_par_iter()
            .map(|x| self.at(field, x))
            .collect()
    }
}

/// The claimed sum
/// $$a = \sum\_{i' \in \[n'\]} \widetilde{eq}(\tau\_1, \mathsf{bit}(i'))\, y\_{i'}(\alpha) +
/// \sum\_{i' = n'+1}^{n'+d} \widetilde{eq}(\tau\_1, \mathsf{bit}(i'))\, y\_{i' - n' - 1}$$
/// (`5_Building_Blocks.tex`:295-296; the oracle's `claimed_sum`), where $\mathbf y \in
/// \mathcal R\_q^{n'}$ is the right-hand side of $\mathbf M \mathbf z' = \mathbf y$ and
/// $y = \sum\_{j \lt d} y\_j X^j$ is the claimed evaluation embedded in $\mathcal R\_q^H$
/// (`y_ring`; `ref/README.md`, choice 12).
///
/// # Errors
/// [`Error::Shape`] unless $\mathbf y$ has $n'$ polynomials and every polynomial (and `y_ring`)
/// $d$ coefficients, and $\mathrm{eq}\_1$ has $2^{\lceil \log \nu \rceil}$ entries.
pub fn claimed_sum<const K: usize>(
    ctx: &Context<K>,
    eq1: &[Ext<K>],
    y: &[Poly],
    y_ring: &Poly,
    alpha: &Ext<K>,
) -> Result<Ext<K>> {
    let l = &ctx.layout;
    let field = ctx.field();
    check_len("y (one entry per row of M)", l.n_prime, y.len())?;
    check_len(
        "eq(tau_1) table (2^ceil(log nu) entries)",
        pow2(l.log_nu, "the eq(tau_1) table")?,
        eq1.len(),
    )?;
    check_len("y as a ring element", l.d, y_ring.0.len())?;
    let apow = powers(field, alpha, l.d);
    let mut a = field.zero();
    for (yi, e) in y.iter().zip(eq1) {
        check_len("y (coefficients)", l.d, yi.0.len())?;
        a = field.add(&a, &field.mul(e, &eval_at(field, &yi.0, &apow)));
    }
    let zq = field.zq();
    for (yj, e) in y_ring.0.iter().zip(&eq1[l.n_prime..]) {
        a = field.add(&a, &field.scale(e, zq.reduce(*yj)));
    }
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// splitmix64.
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// [`CompactTable::from_share`] against [`CompactTable::from_dense`] on the table of
    /// [`w_table`] for shares shaped as a party's (digits on rows of the slice, residues on the
    /// rows of [`ind_rows`], zero elsewhere), for toy contexts of one to three parties, and `None`
    /// for both once an entry outside those rows is not a byte.
    #[test]
    fn the_compact_table_from_the_share_is_the_compact_dense_table() {
        use hachi_co_params::{derive, presets::toy_choices};
        for n in 1..=3u64 {
            let (params, _) = derive(&toy_choices(n)).unwrap();
            let ctx = Context::<4>::setup(params, &[n as u8 + 60; 32]).unwrap();
            let l = &ctx.layout;
            let (q, zq, two_d) = (ctx.q(), *ctx.field().zq(), 2 * l.d);
            let mut state = 60 + n;
            let rows = ind_rows(l);
            for p in 1..=l.n {
                let slice = row_indicator(l, &l.s_ranges(p)).unwrap();
                let mut z: Vec<Poly> = (0..l.len_z)
                    .map(|u| {
                        Poly(
                            (0..l.d)
                                .map(|_| {
                                    if rows.contains(&u) {
                                        next(&mut state) % q
                                    } else if slice[u] == 1 {
                                        zq.from_i64((next(&mut state) % 31) as i64 - 15)
                                    } else {
                                        0
                                    }
                                })
                                .collect(),
                        )
                    })
                    .collect();
                let r: Vec<Poly> = (0..l.n_prime)
                    .map(|_| Poly((0..two_d).map(|_| next(&mut state) % q).collect()))
                    .collect();
                let large = rows.start * two_d..rows.end * two_d;
                let dense = w_table(&ctx, &z, &r).unwrap();
                let from_dense = CompactTable::from_dense(zq, &dense, large.clone()).unwrap();
                let from_share = CompactTable::from_share(&ctx, &z, &r).unwrap().unwrap();
                assert_eq!(from_share.to_dense(), dense, "n = {n}, p = {p}");
                assert_eq!(from_share.to_dense(), from_dense.to_dense());
                // The share held sparsely, its rows of the slice and of ind_rows: the same tables.
                let mut sparse = SparseRows::new(l.len_z, l.d);
                for u in (0..l.len_z).filter(|u| slice[*u] == 1 || rows.contains(u)) {
                    sparse.set(u, z[u].clone());
                }
                assert!(
                    l.n == 1 || sparse.held() < l.len_z,
                    "n = {n}, p = {p}: rows not held"
                );
                assert_eq!(w_table_rows(&ctx, &sparse, &r).unwrap(), dense);
                let from_rows = CompactTable::from_rows(&ctx, &sparse, &r).unwrap().unwrap();
                assert_eq!(from_rows.to_dense(), dense, "n = {n}, p = {p}, sparse");
                assert_eq!(from_rows.digest(), from_share.digest());
                // An entry of the slice that is not a byte: neither form is compact.
                if let Some(u) = (0..l.len_z).find(|u| slice[*u] == 1 && !rows.contains(u)) {
                    z[u].0[0] = 1000;
                    let dense = w_table(&ctx, &z, &r).unwrap();
                    assert!(CompactTable::from_dense(zq, &dense, large).is_none());
                    assert!(CompactTable::from_share(&ctx, &z, &r).unwrap().is_none());
                    sparse.set(u, z[u].clone());
                    assert!(
                        CompactTable::from_rows(&ctx, &sparse, &r)
                            .unwrap()
                            .is_none()
                    );
                }
            }
        }
    }

    /// [`CompactCoeffs`] in 1, 2 and 4 parts against [`LiftedCoeffs`] on the same table and pad:
    /// a table of $2^{18}$ entries with small entries, zero rows and a range of residues, every
    /// coefficient read part by part and the fingerprint.
    #[test]
    fn the_coefficients_in_parts_are_the_lifted_coefficients() {
        use hachi_co_core::base_pcs::LiftedCoeffs;
        let q = 18211211930825437u64; // secure-l24-n2-a8
        let f = Fqk::<4>::new(q).unwrap();
        let zq = *f.zq();
        let mut state = 77u64;
        let n = 1usize << 18;
        let large = 1000..3000;
        let dense: Vec<u64> = (0..n)
            .map(|x| {
                if large.contains(&x) {
                    next(&mut state) % q
                } else if (x / 512) % 3 == 0 {
                    0
                } else {
                    zq.from_i64((next(&mut state) % 15) as i64 - 7)
                }
            })
            .collect();
        let at = 5 * 512 + 3;
        let mut table = dense.clone();
        table[at] = 0;
        let pad_value = Ext::<4>(core::array::from_fn(|_| next(&mut state) % q));
        let compact = CompactTable::from_dense(zq, &table, large).unwrap();
        let lifted = LiftedCoeffs::from_table(&f, table.clone(), Some((at, pad_value))).unwrap();
        let want = lifted.to_vec();
        for parts in [1usize, 2, 4] {
            let coeffs = CompactCoeffs::new(&f, &compact, Some((at, pad_value)), parts).unwrap();
            let per = n / parts;
            let mut got = vec![f.zero(); n];
            for (p, out) in got.chunks_mut(per).enumerate() {
                coeffs.prepare(p).unwrap();
                // Read in runs of 64, as the commitment reads them.
                for (k, run) in out.chunks_mut(64).enumerate() {
                    coeffs.read(p * per + 64 * k, run);
                }
            }
            assert_eq!(got, want, "{parts} parts");
            assert_eq!(coeffs.fingerprint(), lifted.fingerprint(), "{parts} parts");
        }
    }

    #[test]
    fn the_split_eq_table_is_the_eq_table() {
        let q = 134217757u64; // toy-n4
        let f = Fqk::<4>::new(q).unwrap();
        let mut state = 51u64;
        for len in 0..=9usize {
            let tau: Vec<Ext<4>> = (0..len)
                .map(|_| Ext(core::array::from_fn(|_| next(&mut state) % q)))
                .collect();
            let whole = eq_table(&f, &tau).unwrap();
            let split = EqSplit::new(&f, &tau).unwrap();
            assert_eq!(split.len(), whole.len(), "{len} coordinates");
            for (i, want) in whole.iter().enumerate() {
                assert_eq!(split.at(&f, i), *want, "{len} coordinates, entry {i}");
            }
        }
    }

    #[test]
    fn m_star_and_the_quotient_by_rows_are_those_of_the_whole_matrix() {
        // Toy contexts of one and two parties, M from random public values, a random tau_1, E
        // and alpha: the table from the whole M and the whole eq table against the table from
        // rows (borrowed, and built by `MRows`) and the split table; the quotient share likewise.
        use hachi_co_core::{LinearShare, MRows, build_m};
        use hachi_co_params::{derive, presets::toy_choices};
        for (n, seed) in [(1u64, 52u64), (2, 53)] {
            let (params, _) = derive(&toy_choices(n)).unwrap();
            let ctx = Context::<4>::setup(params, &[seed as u8; 32]).unwrap();
            let (l, f) = (&ctx.layout, ctx.field());
            let (q, d) = (ctx.q(), l.d);
            let mut state = seed;
            let mut ext = || Ext::<4>(core::array::from_fn(|_| next(&mut state) % q));
            let b_ring: Vec<Poly> = (0..l.n_blocks).map(|_| ctx.sub.embed(&ext())).collect();
            let a_ring: Vec<Poly> = (0..l.block_len).map(|_| ctx.sub.embed(&ext())).collect();
            let tau1: Vec<Ext<4>> = (0..l.log_nu).map(|_| ext()).collect();
            let alpha = ext();
            let mut state = seed + 100;
            let c: Vec<Vec<i64>> = (0..l.n_blocks)
                .map(|_| (0..d).map(|_| (next(&mut state) % 3) as i64 - 1).collect())
                .collect();
            let e_matrix: Vec<Vec<u64>> = (0..d)
                .map(|_| (0..d).map(|_| next(&mut state) % q).collect())
                .collect();
            let m = build_m(&ctx, &b_ring, &a_ring, &c, 0).unwrap();
            let rows = MRows::new(&ctx, &b_ring, &a_ring, &c, 0).unwrap();
            let eq1 = eq_table(f, &tau1).unwrap();
            let split = EqSplit::new(f, &tau1).unwrap();
            let want = mstar_table(&ctx, &m, &e_matrix, &eq1, &alpha).unwrap();
            let borrowed = |i: usize| Ok(Cow::Borrowed(&m[i][..]));
            assert_eq!(
                mstar_table_rows(&ctx, borrowed, &e_matrix, &split, &alpha).unwrap(),
                want,
                "n = {n}, borrowed rows"
            );
            let built = |i: usize| Ok(Cow::Owned(rows.row(i)?));
            assert_eq!(
                mstar_table_rows(&ctx, built, &e_matrix, &split, &alpha).unwrap(),
                want,
                "n = {n}, rows of MRows"
            );
            // M* from its parts, each entry formed when read: the same table.
            let built = |i: usize| Ok(Cow::Owned(rows.row(i)?));
            let parts =
                MStarParts::from_rows(&ctx, built, &e_matrix, split.clone(), &alpha).unwrap();
            assert_eq!(parts.len(), want.len(), "n = {n}");
            assert_eq!(parts.table(f), want, "n = {n}, parts");
            for x in [0, 1, want.len() / 2, want.len() - 1] {
                assert_eq!(parts.at(f, x), want[x], "n = {n}, entry {x}");
            }
            let short = &e_matrix[1..];
            assert!(MStarParts::from_rows(&ctx, borrowed, short, split.clone(), &alpha).is_err());
            let share = LinearShare {
                z_prime: (0..l.len_z)
                    .map(|_| Poly((0..d).map(|_| next(&mut state) % q).collect()))
                    .collect(),
                y: (0..l.n_prime)
                    .map(|_| Poly((0..d).map(|_| next(&mut state) % q).collect()))
                    .collect(),
            };
            let quotient = crate::quotient_share(&ctx, &m, &share).unwrap();
            let built = |i: usize| Ok(Cow::Owned(rows.row(i)?));
            assert_eq!(
                crate::quotient_share_rows(&ctx, built, &share).unwrap(),
                quotient,
                "n = {n}"
            );
            // Every third row held sparsely, the others zero in the dense share: the same
            // quotient.
            let mut dense = share.clone();
            let mut sparse = SparseRows::new(l.len_z, d);
            for (j, z) in dense.z_prime.iter_mut().enumerate() {
                if j % 3 == 0 {
                    sparse.set(j, z.clone());
                } else {
                    z.0.fill(0);
                }
            }
            let built = |i: usize| Ok(Cow::Owned(rows.row(i)?));
            assert_eq!(
                crate::quotient::quotient_rows(&ctx, built, &sparse, &dense.y).unwrap(),
                crate::quotient_share(&ctx, &m, &dense).unwrap(),
                "n = {n}, sparse"
            );
        }
    }

    #[test]
    fn range_product_matches_the_literal_product() {
        let q = 134217757u64; // toy-n4
        let f = Fqk::<4>::new(q).unwrap();
        for b in 0..=6u64 {
            let rp = RangeProduct::new(f.zq(), b);
            for w in [
                f.zero(),
                f.one(),
                f.from_base(q - 1),
                f.from_base(b),
                f.from_base(q - b),
                Ext([5, 7, 11, 13]),
            ] {
                let mut lit = f.one();
                for c in -(b as i64) + 1..=(b as i64) - 1 {
                    lit = f.mul(&lit, &sub_base(&f, &w, f.zq().from_i64(c)));
                }
                assert_eq!(rp.eval(&f, &w), lit, "b = {b}");
            }
            // The roots are exactly the integers -b+1..b-1 (the failure boundary is +-b).
            for c in -(b as i64) + 1..=(b as i64) - 1 {
                assert!(f.is_zero(&rp.eval(&f, &f.from_base(f.zq().from_i64(c)))));
            }
            if b > 0 {
                for c in [b as i64, -(b as i64)] {
                    assert!(!f.is_zero(&rp.eval(&f, &f.from_base(f.zq().from_i64(c)))));
                }
            }
        }
    }

    #[test]
    fn powers_and_evaluation() {
        let q = 8388637u64;
        let f = Fqk::<4>::new(q).unwrap();
        let x = Ext([3, 1, 4, 1]);
        let p = powers(&f, &x, 5);
        assert_eq!(p[0], f.one());
        assert_eq!(p[4], f.pow(&x, &[4]));
        // 2 + 3x + 0x^2 + (q-1)x^3
        let v = eval_at(&f, &[2, 3, 0, q - 1], &p);
        let want = f.sub(&f.add(&f.from_base(2), &f.scale(&x, 3)), &p[3]);
        assert_eq!(v, want);
    }
}
