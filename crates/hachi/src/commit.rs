//! The arithmetic context ([`Ctx`]) and the inner-outer commitment (paper, Section 4.1):
//! $\mathbf t_i = \mathbf A \mathbf s_i$, $\hat{\mathbf t}_i = \mathbf G^{-1}(\mathbf t_i)$ and
//! $\mathbf u = \mathbf B \hat{\mathbf t}$, with $\mathbf s_i = \mathbf G^{-1}(\mathbf f_i)$ or
//! $\mathbf s_i = \mathbf f_i$ according to the [`Layout`]. In a masked layout ([`Hiding`]), each
//! chunk ends with a masking row and $\mathbf t_i = \mathbf A \mathbf s_i + \mathbf A'
//! \boldsymbol\varrho_i$ ([`Ctx::rand_matrix`], [`Ctx::rand_term`]).

use crate::params::{HachiChoices, Hiding, Layout, check, height};
use crate::{Error, Result};
use hachi_co_math::decompose::{digit_bound, g_inv, num_digits};
use hachi_co_math::{Fqk, Poly, Ring, Subfield};
use hachi_co_sampling::matrix::expand;
use hachi_co_sampling::{ShakePrg, uniform};
use rand::Rng;
use rayon::prelude::*;
use std::borrow::Cow;

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
    /// expanded by `hachi_co_sampling::matrix::expand`.
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
        let mut domain = b"hachi/pp/".to_vec();
        domain.extend_from_slice(kind);
        let mut seed = [0u8; 32];
        ShakePrg::new(&self.choices.seed, &domain, iteration as u64).fill_bytes(&mut seed);
        let m = expand(&seed, b"hachi/matrix", rows, cols, self.d(), self.q)?;
        Ok(m.into_iter()
            .map(|row| row.into_iter().map(Poly).collect())
            .collect())
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
    /// entries $[i n_A \delta, (i + 1) n_A \delta)$, with the height of the layout.
    pub t_hat: Vec<Poly>,
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
    check_message(ctx, &layout, &message)?;
    let chunks_of_r = if layout.masked { layout.chunks() } else { 0 };
    if r.len() != chunks_of_r {
        return Err(Error::Shape(format!(
            "randomness for {chunks_of_r} chunks is expected"
        )));
    }
    let (ring, b, delta) = (&ctx.ring, ctx.choices.b, ctx.delta);
    let n_a = ctx.height(&layout)?;
    let width = layout.width(delta);
    let a = ring.prepare_short(&ctx.matrix(b"A", iteration, n_a, width)?, layout.beta_s)?;
    let bmat = ctx.matrix(
        b"B",
        iteration,
        ctx.choices.n_b,
        layout.chunks() * n_a * delta,
    )?;
    let mut t: Vec<Vec<Poly>> = message
        .par_iter()
        .map(|chunk| {
            let s = Committed::digits(ctx, &layout, delta, chunk)?;
            Ok(ring.mat_vec_prepared(&a, &s)?)
        })
        .collect::<Result<_>>()?;
    if !r.is_empty() {
        let a_rand = ctx.rand_matrix()?;
        for (ti, ri) in t.iter_mut().zip(&r) {
            let term = ctx.rand_term(&a_rand, ri)?;
            for (x, y) in ti.iter_mut().zip(&term) {
                *x = ring.add(x, y)?;
            }
        }
    }
    let mut t_hat = Vec::with_capacity(layout.chunks() * n_a * delta);
    for ti in &t {
        t_hat.extend(g_inv(ring, b, delta, ti)?);
    }
    let u = ring.mat_vec_short(&bmat, &t_hat, digit_bound(b))?;
    Ok(Committed {
        layout,
        message,
        r,
        t,
        t_hat,
        u,
    })
}
