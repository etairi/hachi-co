//! Where everything sits: the positions of the witness $\mathbf z'$, the rows of $\mathbf M$, the
//! ownership of the blocks of $\mathbf f'$, the partition $(S\_1, \dots, S\_n)$ and the sizes of the
//! sumcheck's hypercube ([`Layout`]).

use crate::{Error, Result};
use core::ops::Range;
use hachi_co_mpc::PartyId;
use hachi_co_params::Params;

/// A named block of $\mathbf z'$: its name, first position and length, as in the fixture key
/// `ring_to_linear/layout/blocks` (`ref/hachico_ref/r2l.py`, `Layout.blocks`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NamedBlock {
    /// One of `w_hat`, `w0_hat`, `t0_hat`, `rho`, `t_hat`, `z_hat`, `u`.
    pub name: &'static str,
    /// The first position, 0-indexed.
    pub start: usize,
    /// The number of positions.
    pub len: usize,
}

/// The positions in $\mathbf z'$, the rows of $\mathbf M$, the block ownership, the partition
/// $(S\_p)\_p$ and the hypercube sizes, all derived from [`Params`] by [`Layout::new`] (the oracle's
/// `r2l.Layout` plus ownership and the hypercube; `docs/m1-spec.md`, `hachi-co-core`).
///
/// All positions are 0-indexed (`ref/README.md`, "Indexing": the draft's 1-indexed position $u$ is
/// our $u - 1$). The layout of $\mathbf z' \in \mathcal R\_q^\mu$ is that of
/// `5_Building_Blocks.tex`:164-172 and of the oracle (`ref/hachico_ref/r2l.py`, class `Layout`):
/// $$\mathbf z' = \bigl\[ \hat{\mathbf w}\ (2^r\delta) \mid \hat{\mathbf w}\_0\ (n\delta) \mid
/// \hat{\mathbf t}\_0\ (n n\_A \delta) \mid \boldsymbol\rho\ (n m\_\rho) \mid
/// \hat{\mathbf t}\ (2^r n\_A \delta) \mid \hat{\mathbf z}\ (2^m \delta \tau) \mid
/// \langle u \rangle\ (1) \bigr\],$$
/// with the element-major gadget layout inside each block: $\hat{\mathbf w}\_i =
/// \mathbf G\_1^{-1}(w\_i)$ at $w + i\delta + e$; $\hat{\mathbf w}\_{0,p}$ at $w\_0 + (p-1)\delta +
/// e$; $\hat{\mathbf t}\_{0,p}$ at $t\_0 + (p-1) n\_A \delta + a\delta + e$; $\boldsymbol\rho\_p$ at
/// $\rho + (p-1) m\_\rho + j$; $\hat{\mathbf t}\_i$ at $t + i n\_A \delta + a \delta + e$; digit
/// $\theta$ of $z\_{j\delta + e}$ at $z + (j\delta + e)\tau + \theta$ (`ref/README.md`, "z′
/// layout"). The rows of $\mathbf M \in \mathcal R\_q^{n' \times \mu}$ are $\mathbf D$ ($n\_D$),
/// $\mathbf B$ ($n\_B$), $\mathbf b^T \mathbf G\_{2^r}$ (1), the $\mathbf c$ row (1) and the
/// $\mathbf A$ rows ($n\_A$) (`5_Building_Blocks.tex`:157-163).
///
/// Every field is a `usize`, so that block code can index vectors directly; the derivation checks
/// every product for overflow. The struct is non-exhaustive: it is built only by [`Layout::new`],
/// which checks its invariants.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Layout {
    // ---- dimensions (6_Collaborative_Hachi.tex:265-287) ----
    /// The number of provers $n \ge 1$.
    pub n: usize,
    /// $\lceil \log\_2 n \rceil$ (`6_Collaborative_Hachi.tex`:274), with $\lceil \log 1 \rceil = 0$.
    pub log_n: usize,
    /// The ring degree $d = 2^\alpha$ (`6_Collaborative_Hachi.tex`:270).
    pub d: usize,
    /// The extension degree $k = 2^\kappa$ of $\mathbb F\_{q^k} \cong \mathcal R\_q^H$
    /// (`6_Collaborative_Hachi.tex`:274).
    pub k: usize,
    /// The number of variables $\ell$ of the committed polynomial ($2^\ell$ coefficients).
    pub ell: usize,
    /// $\ell' = \lceil \log n \rceil + \ell - \alpha + \kappa$, the number of variables of $f'$
    /// (`6_Collaborative_Hachi.tex`:274).
    pub ell_prime: usize,
    /// $r = \max(\lceil \ell'/2 \rceil, \lceil \log n \rceil)$ (`6_Collaborative_Hachi.tex`:274).
    pub r: usize,
    /// $m = \ell' - r$ (`6_Collaborative_Hachi.tex`:274).
    pub m: usize,
    /// $\delta = \lceil \log\_b q \rceil$, the digits of $\mathbf G^{-1}$
    /// (`6_Collaborative_Hachi.tex`:274).
    pub delta: usize,
    /// $\tau$, the digits of $\mathbf J^{-1}$ (`6_Collaborative_Hachi.tex`:276, tight rule of
    /// `docs/m0-spec.md`).
    pub tau: usize,
    /// The height $n\_A$ of $\mathbf A$.
    pub n_a: usize,
    /// The height $n\_B$ of $\mathbf B$.
    pub n_b: usize,
    /// The height $n\_D$ of $\mathbf D$.
    pub n_d: usize,
    /// $m\_\rho$, the ring elements of each $\boldsymbol\rho\_p \in S\_b^{m\_\rho}$
    /// (`5_Building_Blocks.tex`:110).
    pub m_rho: usize,
    /// $2^r$, the number of blocks of $\mathbf f'$.
    pub n_blocks: usize,
    /// $2^m$, the ring elements per block.
    pub block_len: usize,
    /// $2^{r - \lceil \log n \rceil}$, the blocks held by each prover ($|\mathcal I\_i|$,
    /// `5_Building_Blocks.tex`:127).
    pub blocks_per_party: usize,
    /// The width $\delta 2^m$ of $\mathbf A$ (`6_Collaborative_Hachi.tex`:285).
    pub width_a: usize,
    /// The width $n\_A \delta 2^r$ of $\mathbf B$ (`6_Collaborative_Hachi.tex`:286).
    pub width_b: usize,
    /// The width $\delta(2^r + n + n n\_A) + n m\_\rho$ of $\mathbf D$
    /// (`6_Collaborative_Hachi.tex`:287); $\mathbf D$ acts on $\mathbf z'\[0..t)$.
    pub width_d: usize,
    /// $\mu = \mathrm{len}(\mathbf z')$ (`6_Collaborative_Hachi.tex`:278).
    pub mu: usize,
    /// $n' = n\_A + n\_B + n\_D + 2 = \mathrm{len}(\mathbf y)$ (`6_Collaborative_Hachi.tex`:279).
    pub n_prime: usize,
    /// $\nu = n' + d + (\mu + n') d$ (`6_Collaborative_Hachi.tex`:281).
    pub nu: usize,
    /// $d\_0 = 2b + n' + 1$, the degree of $g\_{0,p}$ (`6_Collaborative_Hachi.tex`:281).
    pub d0: usize,
    /// $d\_1 = 3$, the degree of $g\_{1,p}$ (`6_Collaborative_Hachi.tex`:281).
    pub d1: usize,
    // ---- hypercube of the sumcheck (5_Building_Blocks.tex:207, 243-249) ----
    /// $\ell\_U = \lceil \log\_2(\mu + n') \rceil = |U|$ (`ref/README.md`, "Hypercube").
    pub ell_u: usize,
    /// $\ell\_L = \log\_2 2d = |L|$.
    pub ell_l: usize,
    /// $\ell'' = \ell\_U + \ell\_L$ (`6_Collaborative_Hachi.tex`:280).
    pub ell_pp: usize,
    /// $\lceil \log\_2 \nu \rceil$, the length of $\tau\_1$ (`5_Building_Blocks.tex`:248).
    pub log_nu: usize,
    // ---- offsets of z' (5_Building_Blocks.tex:164-172) ----
    /// Start of $\hat{\mathbf w}$ (always 0).
    pub w: usize,
    /// Start of $\hat{\mathbf w}\_0 = (\hat{\mathbf w}\_{0,1}, \dots, \hat{\mathbf w}\_{0,n})$.
    pub w0: usize,
    /// Start of $\hat{\mathbf t}\_0 = (\hat{\mathbf t}\_{0,1}, \dots, \hat{\mathbf t}\_{0,n})$.
    pub t0: usize,
    /// Start of $\boldsymbol\rho = (\boldsymbol\rho\_1, \dots, \boldsymbol\rho\_n)$.
    pub rho: usize,
    /// Start of $\hat{\mathbf t}$; equals the width of $\mathbf D$.
    pub t: usize,
    /// Start of $\hat{\mathbf z}$.
    pub z: usize,
    /// The position of $\langle u \rangle$, the last entry: $u = \mu - 1$.
    pub u: usize,
    // ---- rows of M (5_Building_Blocks.tex:157-163) ----
    /// First row of the $\mathbf D$ block (always 0).
    pub row_d: usize,
    /// First row of the $\mathbf B$ block ($= n\_D$).
    pub row_b: usize,
    /// The row $\mathbf b^T \mathbf G\_{2^r} \mid \cdots \mid -1$ ($= n\_D + n\_B$).
    pub row_bg: usize,
    /// The row $\mathbf c^T \otimes \mathbf G\_1 \mid \mathbf 1\_n^T \otimes \mathbf G\_1 \mid \cdots
    /// \mid -\mathbf a^T \mathbf G\_{2^m} \mathbf J\_{\delta 2^m}$ ($= n\_D + n\_B + 1$).
    pub row_c: usize,
    /// First of the $n\_A$ rows $\mathbf 1\_n^T \otimes \mathbf G\_{n\_A} \mid \mathbf c^T \otimes
    /// \mathbf G\_{n\_A} \mid -\mathbf A \mathbf J\_{\delta 2^m}$ ($= n\_D + n\_B + 2$).
    pub row_a: usize,
}

/// `x` as a `usize`, or a layout error naming it.
fn to_usize<T>(x: T, what: &str) -> Result<usize>
where
    T: TryInto<usize> + Copy + core::fmt::Display,
{
    x.try_into()
        .map_err(|_| Error::Layout(format!("{what} = {x} does not fit in usize")))
}

fn overflow(what: &str) -> Error {
    Error::Layout(format!("{what} overflows usize"))
}

fn mul(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| overflow(what))
}

fn add(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_add(b).ok_or_else(|| overflow(what))
}

fn pow2(e: usize, what: &str) -> Result<usize> {
    u32::try_from(e)
        .ok()
        .and_then(|e| 1usize.checked_shl(e))
        .ok_or_else(|| overflow(what))
}

/// $\lceil \log\_2 x \rceil$, the least $k$ with $2^k \ge x$, with $\lceil \log 1 \rceil = 0$ (and
/// $0$ for $x = 0$).
pub(crate) fn ceil_log2(x: usize) -> usize {
    if x <= 1 {
        0
    } else {
        (usize::BITS - (x - 1).leading_zeros()) as usize
    }
}

/// Appends `r` to the sorted, disjoint list `out`, merging it with the last range if they touch;
/// empty ranges are skipped.
fn push_merged(out: &mut Vec<Range<usize>>, r: Range<usize>) {
    if r.is_empty() {
        return;
    }
    match out.last_mut() {
        Some(last) if last.end == r.start => last.end = r.end,
        _ => out.push(r),
    }
}

impl Layout {
    /// The layout for `params`, with every invariant checked.
    ///
    /// Besides the conversions to `usize`, it checks (`docs/m1-spec.md`, `hachi-co-core`) that
    /// $u + 1 = \mu$ (`6_Collaborative_Hachi.tex`:278), that $t$ equals the width of $\mathbf D$
    /// (line 287), that $\mathrm{row}\_A + n\_A = n'$ (line 279), and that $S\_1, \dots, S\_n$ and
    /// the unheld positions partition $\[0, \mu - 1)$ (`5_Building_Blocks.tex`:197 and
    /// `ref/README.md`, choice 6); and also that the widths of $\mathbf A$ and $\mathbf B$ match
    /// the blocks they act on (lines 285-286), that $\nu = n' + d + (\mu + n') d$ (line 281),
    /// that $\ell'' = \lceil \log(\mu + n') \rceil + \log 2d$ (line 280), that
    /// $\lceil \log n \rceil$ is right, that $r \ge \lceil \log n \rceil$ and that $d$ is a
    /// power of two.
    ///
    /// # Errors
    /// [`Error::Layout`] naming the first violated condition.
    pub fn new(params: &Params) -> Result<Self> {
        let c = &params.choices;
        let n = to_usize(c.n, "n")?;
        let log_n = to_usize(params.log_n, "ceil(log n)")?;
        let d = to_usize(params.d, "d")?;
        let k = to_usize(params.k, "k")?;
        let ell = to_usize(c.ell, "l")?;
        let ell_prime = to_usize(params.ell_prime, "l'")?;
        let r = to_usize(params.r, "r")?;
        let m = to_usize(params.m, "m")?;
        let delta = to_usize(params.delta, "delta")?;
        let tau = to_usize(params.tau, "tau")?;
        let n_a = to_usize(c.n_a, "n_A")?;
        let n_b = to_usize(c.n_b, "n_B")?;
        let n_d = to_usize(c.n_d, "n_D")?;
        let m_rho = to_usize(c.m_rho, "m_rho")?;
        let width_a = to_usize(params.width_a, "the width of A")?;
        let width_b = to_usize(params.width_b, "the width of B")?;
        let width_d = to_usize(params.width_d, "the width of D")?;
        let mu = to_usize(params.mu, "mu")?;
        let n_prime = to_usize(params.n_prime, "n'")?;
        let nu = to_usize(params.nu, "nu")?;
        let d0 = to_usize(params.d0, "d0")?;
        let d1 = to_usize(params.d1, "d1")?;
        let ell_pp = to_usize(params.ell_dprime, "l''")?;

        let fail = |msg: String| Err(Error::Layout(msg));
        if n == 0 {
            return fail("there must be at least one prover".into());
        }
        if log_n != ceil_log2(n) {
            return fail(format!(
                "ceil(log n) = {log_n} is wrong for n = {n} (6_Collaborative_Hachi.tex:274)"
            ));
        }
        if d < 2 || !d.is_power_of_two() {
            return fail(format!("d = {d} is not a power of two >= 2"));
        }
        if r < log_n {
            return fail(format!(
                "r = {r} < ceil(log n) = {log_n} (6_Collaborative_Hachi.tex:274)"
            ));
        }
        let n_blocks = pow2(r, "2^r")?;
        let block_len = pow2(m, "2^m")?;
        let blocks_per_party = pow2(r - log_n, "2^(r - ceil(log n))")?;

        // Offsets of z' (5_Building_Blocks.tex:164-172).
        let w = 0;
        let w0 = add(w, mul(n_blocks, delta, "2^r delta")?, "w0")?;
        let t0 = add(w0, mul(n, delta, "n delta")?, "t0")?;
        let n_na_delta = mul(mul(n, n_a, "n n_A")?, delta, "n n_A delta")?;
        let rho = add(t0, n_na_delta, "rho")?;
        let t = add(rho, mul(n, m_rho, "n m_rho")?, "t")?;
        let t_hat_len = mul(mul(n_blocks, n_a, "2^r n_A")?, delta, "2^r n_A delta")?;
        let z = add(t, t_hat_len, "z")?;
        let z_hat_len = mul(mul(block_len, delta, "2^m delta")?, tau, "2^m delta tau")?;
        let u = add(z, z_hat_len, "u")?;
        if add(u, 1, "u + 1")? != mu {
            return fail(format!(
                "u + 1 = {} but mu = {mu} (6_Collaborative_Hachi.tex:278)",
                u + 1
            ));
        }
        if t != width_d {
            return fail(format!(
                "t = {t} but the width of D is {width_d} (6_Collaborative_Hachi.tex:287)"
            ));
        }
        if t_hat_len != width_b {
            return fail(format!(
                "t-hat has {t_hat_len} entries but the width of B is {width_b} \
                 (6_Collaborative_Hachi.tex:286)"
            ));
        }
        if mul(block_len, delta, "2^m delta")? != width_a {
            return fail(format!(
                "delta 2^m = {} but the width of A is {width_a} (6_Collaborative_Hachi.tex:285)",
                block_len * delta
            ));
        }

        // Rows of M (5_Building_Blocks.tex:157-163).
        let row_d = 0;
        let row_b = n_d;
        let row_bg = add(n_d, n_b, "n_D + n_B")?;
        let row_c = add(row_bg, 1, "row_c")?;
        let row_a = add(row_c, 1, "row_A")?;
        if add(row_a, n_a, "row_A + n_A")? != n_prime {
            return fail(format!(
                "row_A + n_A = {} but n' = {n_prime} (6_Collaborative_Hachi.tex:279)",
                row_a + n_a
            ));
        }

        // Hypercube (5_Building_Blocks.tex:207; ref/README.md, "Hypercube").
        let mu_np = add(mu, n_prime, "mu + n'")?;
        let ell_u = ceil_log2(mu_np);
        // log 2d = alpha + 1 (d is a power of two).
        let ell_l = d.trailing_zeros() as usize + 1;
        if ell_u + ell_l != ell_pp {
            return fail(format!(
                "l'' = {ell_pp} but ceil(log(mu + n')) + log 2d = {} (6_Collaborative_Hachi.tex:280)",
                ell_u + ell_l
            ));
        }
        let expected_nu = add(
            add(n_prime, d, "n' + d")?,
            mul(mu_np, d, "(mu + n') d")?,
            "nu",
        )?;
        if nu != expected_nu {
            return fail(format!(
                "nu = {nu} but n' + d + (mu + n') d = {expected_nu} (6_Collaborative_Hachi.tex:281)"
            ));
        }
        let log_nu = ceil_log2(nu);

        let layout = Self {
            n,
            log_n,
            d,
            k,
            ell,
            ell_prime,
            r,
            m,
            delta,
            tau,
            n_a,
            n_b,
            n_d,
            m_rho,
            n_blocks,
            block_len,
            blocks_per_party,
            width_a,
            width_b,
            width_d,
            mu,
            n_prime,
            nu,
            d0,
            d1,
            ell_u,
            ell_l,
            ell_pp,
            log_nu,
            w,
            w0,
            t0,
            rho,
            t,
            z,
            u,
            row_d,
            row_b,
            row_bg,
            row_c,
            row_a,
        };
        layout.check_partition()?;
        Ok(layout)
    }

    /// Checks that $S\_1, \dots, S\_n$ and the unheld positions partition $\[0, \mu - 1)$.
    fn check_partition(&self) -> Result<()> {
        let mut count = vec![0u8; self.u];
        let ranges = (1..=self.n)
            .flat_map(|p| self.s_ranges(p))
            .chain(self.unheld());
        for range in ranges {
            if range.end > self.u {
                return Err(Error::Layout(format!(
                    "the range {range:?} leaves [0, mu - 1) = [0, {})",
                    self.u
                )));
            }
            for c in &mut count[range] {
                *c = c.saturating_add(1);
            }
        }
        if let Some(pos) = count.iter().position(|c| *c != 1) {
            return Err(Error::Layout(format!(
                "position {pos} is covered {} times by S_1..S_n and the unheld positions \
                 (5_Building_Blocks.tex:197)",
                count[pos]
            )));
        }
        Ok(())
    }

    /// The prover holding block $i$ of $\mathbf f'$: $\lfloor i / 2^{r - \lceil \log n \rceil}
    /// \rfloor + 1$ if that is at most $n$ (`6_Collaborative_Hachi.tex`:406, `ref/README.md`,
    /// "Blocks and owners"), and `None` for an all-zero block held by nobody (for $n = 3$: blocks 6
    /// and 7) and for $i \ge 2^r$.
    pub fn owner_of_block(&self, i: usize) -> Option<PartyId> {
        if i >= self.n_blocks {
            return None;
        }
        let p = i / self.blocks_per_party + 1;
        (p <= self.n).then_some(p)
    }

    /// The blocks of prover $p$, $\mathcal I\_p$ of `5_Building_Blocks.tex`:127: the contiguous
    /// range $\[(p-1) \cdot 2^{r - \lceil \log n \rceil}, p \cdot 2^{r - \lceil \log n \rceil})$; empty for
    /// $p \notin \lbrace 1, \dots, n \rbrace$.
    pub fn blocks_of(&self, p: PartyId) -> Range<usize> {
        if p == 0 || p > self.n {
            return 0..0;
        }
        (p - 1) * self.blocks_per_party..p * self.blocks_per_party
    }

    /// The $\delta$ positions of $\hat{\mathbf w}\_i = \mathbf G\_1^{-1}(w\_i)$ in $\mathbf z'$
    /// (`5_Building_Blocks.tex`:107, 165); empty for $i \ge 2^r$.
    pub fn w_hat_range(&self, i: usize) -> Range<usize> {
        if i >= self.n_blocks {
            return 0..0;
        }
        self.w + i * self.delta..self.w + (i + 1) * self.delta
    }

    /// The $\delta$ positions of $\hat{\mathbf w}\_{0,p}$ (`5_Building_Blocks.tex`:113, 166);
    /// empty for $p \notin \lbrace 1, \dots, n \rbrace$.
    pub fn w0_hat_range(&self, p: PartyId) -> Range<usize> {
        if p == 0 || p > self.n {
            return 0..0;
        }
        self.w0 + (p - 1) * self.delta..self.w0 + p * self.delta
    }

    /// The $n\_A \delta$ positions of $\hat{\mathbf t}\_{0,p}$ (`5_Building_Blocks.tex`:112, 167),
    /// row $a$ of $\mathbf t\_{0,p}$ at offset $a\delta$; empty for $p \notin \lbrace 1, \dots, n
    /// \rbrace$.
    pub fn t0_hat_range(&self, p: PartyId) -> Range<usize> {
        if p == 0 || p > self.n {
            return 0..0;
        }
        let len = self.n_a * self.delta;
        self.t0 + (p - 1) * len..self.t0 + p * len
    }

    /// The $m\_\rho$ positions of $\boldsymbol\rho\_p$ (`5_Building_Blocks.tex`:110, 168); empty
    /// for $p \notin \lbrace 1, \dots, n \rbrace$.
    pub fn rho_range(&self, p: PartyId) -> Range<usize> {
        if p == 0 || p > self.n {
            return 0..0;
        }
        self.rho + (p - 1) * self.m_rho..self.rho + p * self.m_rho
    }

    /// The $n\_A \delta$ positions of $\hat{\mathbf t}\_i = \mathbf G\_{n\_A}^{-1}(\mathbf t\_i)$
    /// (`6_Collaborative_Hachi.tex`:375; `5_Building_Blocks.tex`:169), row $a$ of $\mathbf t\_i$
    /// at offset $a\delta$; empty for $i \ge 2^r$.
    pub fn t_hat_range(&self, i: usize) -> Range<usize> {
        if i >= self.n_blocks {
            return 0..0;
        }
        let len = self.n_a * self.delta;
        self.t + i * len..self.t + (i + 1) * len
    }

    /// The $2^m \delta \tau$ positions of $\hat{\mathbf z} = \mathbf J\_{\delta 2^m}^{-1}(\mathbf z)$
    /// (`5_Building_Blocks.tex`:133, 170): digit $\theta$ of $z\_j$, $j \lt \delta 2^m$, is at
    /// $z + j\tau + \theta$.
    pub fn z_hat_range(&self) -> Range<usize> {
        self.z..self.u
    }

    /// The pieces of $S\_p$ in increasing order of position (some may be empty).
    fn components(&self, p: PartyId) -> [Range<usize>; 6] {
        if p == 0 || p > self.n {
            return core::array::from_fn(|_| 0..0);
        }
        let blocks = self.blocks_of(p);
        let t_len = self.n_a * self.delta;
        [
            self.w + blocks.start * self.delta..self.w + blocks.end * self.delta,
            self.w0_hat_range(p),
            self.t0_hat_range(p),
            self.rho_range(p),
            self.t + blocks.start * t_len..self.t + blocks.end * t_len,
            if p == 1 { self.z_hat_range() } else { 0..0 },
        ]
    }

    /// $S\_p$ (`5_Building_Blocks.tex`:197; `ref/README.md`, "Shares" and choice 6): the positions
    /// of $\mathbf z'$ that prover $p$ holds, as sorted, disjoint, non-adjacent ranges in
    /// $\[0, \mu - 1)$ (adjacent pieces are merged, as in the fixture key
    /// `ring_to_linear/partition`). They are $p$'s blocks of $\hat{\mathbf w}$ and
    /// $\hat{\mathbf t}$, its own $\hat{\mathbf w}\_{0,p}$, $\hat{\mathbf t}\_{0,p}$ and
    /// $\boldsymbol\rho\_p$, and, for $p = 1$, $\hat{\mathbf z}$. Position $\mu - 1$
    /// ($\langle u \rangle$) is in no $S\_p$. Empty for $p \notin \lbrace 1, \dots, n \rbrace$.
    pub fn s_ranges(&self, p: PartyId) -> Vec<Range<usize>> {
        let mut out = Vec::with_capacity(6);
        for r in self.components(p) {
            push_merged(&mut out, r);
        }
        out
    }

    /// Whether position `pos` of $\mathbf z'$ is in $S\_p$ (see [`Layout::s_ranges`]); false for
    /// $p \notin \lbrace 1, \dots, n \rbrace$.
    pub fn in_s(&self, p: PartyId, pos: usize) -> bool {
        self.components(p).iter().any(|r| r.contains(&pos))
    }

    /// The positions of the all-zero blocks that nobody holds (the $\hat{\mathbf w}$ and
    /// $\hat{\mathbf t}$ entries of blocks $i \ge n 2^{r - \lceil \log n \rceil}$; non-empty only
    /// when $n$ is not a power of two), as sorted disjoint ranges (`ref/README.md`, choice 6;
    /// fixture key `ring_to_linear/unheld`). Together with $S\_1, \dots, S\_n$ they partition
    /// $\[0, \mu - 1)$.
    pub fn unheld(&self) -> Vec<Range<usize>> {
        let first = self.n * self.blocks_per_party;
        let t_len = self.n_a * self.delta;
        let mut out = Vec::with_capacity(2);
        push_merged(&mut out, self.w + first * self.delta..self.w0);
        push_merged(&mut out, self.t + first * t_len..self.z);
        out
    }

    /// The seven blocks of $\mathbf z'$ in order, named as in the fixture key
    /// `ring_to_linear/layout/blocks`: `w_hat`, `w0_hat`, `t0_hat`, `rho`, `t_hat`, `z_hat`, `u`
    /// (`5_Building_Blocks.tex`:164-172).
    pub fn named_blocks(&self) -> [NamedBlock; 7] {
        let block = |name, start: usize, end: usize| NamedBlock {
            name,
            start,
            len: end - start,
        };
        [
            block("w_hat", self.w, self.w0),
            block("w0_hat", self.w0, self.t0),
            block("t0_hat", self.t0, self.rho),
            block("rho", self.rho, self.t),
            block("t_hat", self.t, self.z),
            block("z_hat", self.z, self.u),
            block("u", self.u, self.mu),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_log2_matches_its_definition() {
        for x in 1usize..=4100 {
            let k = ceil_log2(x);
            assert!(1usize << k >= x, "x = {x}");
            assert!(k == 0 || 1usize << (k - 1) < x, "x = {x}");
        }
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(usize::MAX), usize::BITS as usize);
    }

    #[test]
    fn merging_joins_touching_ranges_only() {
        let mut v = Vec::new();
        for r in [0..2, 2..5, 5..5, 6..7, 7..9, 10..10] {
            push_merged(&mut v, r);
        }
        assert_eq!(v, vec![0..5, 6..9]);
    }
}
