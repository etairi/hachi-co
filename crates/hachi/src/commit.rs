//! The arithmetic context ([`Ctx`]) and the inner-outer commitment (paper, Section 4.1):
//! $\mathbf t_i = \mathbf A \mathbf s_i$, $\hat{\mathbf t}_i = \mathbf G^{-1}(\mathbf t_i)$ and
//! $\mathbf u = \mathbf B \hat{\mathbf t}$, with $\mathbf s_i = \mathbf G^{-1}(\mathbf f_i)$ or
//! $\mathbf s_i = \mathbf f_i$ according to the [`Layout`]. With [`Hiding`],
//! $\mathbf t_i = \mathbf A \mathbf s_i + \mathbf A' \mathbf r_i$ ([`Ctx::rand_matrix`],
//! [`Ctx::rand_term`]).

use crate::params::{HachiChoices, Hiding, Layout, Noise, check};
use crate::{Error, Result};
use hachi_co_math::decompose::{g_inv, num_digits};
use hachi_co_math::{Fqk, Poly, Ring, Subfield};
use hachi_co_sampling::matrix::expand;
use hachi_co_sampling::{ShakePrg, gauss, uniform};
use rand::Rng;

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

    /// The randomized commitments' choices, if any.
    pub fn hiding(&self) -> Option<Hiding> {
        self.choices.hiding
    }

    /// The length $\nu + n$ of the randomness of a commitment with $n$ rows ($n_A$ for the inner
    /// commitments, $n_D$ for $\mathbf v$), or 0 without [`Hiding`].
    pub fn rand_len(&self, rows: usize) -> usize {
        self.hiding().map_or(0, |h| h.nu + rows)
    }

    /// `count` ring elements whose coefficients follow $\chi$ ([`Noise`]), read from `prg`
    /// element after element, coefficient after coefficient: [`gauss::gaussian`] for a Gaussian,
    /// [`uniform::short`]$(b)$ for digits.
    ///
    /// # Errors
    /// [`Error::Parameter`] without [`Hiding`]; sampling errors.
    pub fn noise(&self, prg: &mut ShakePrg, count: usize) -> Result<Vec<Poly>> {
        let hiding = self
            .hiding()
            .ok_or_else(|| Error::Parameter("randomness without hiding choices".into()))?;
        let coeffs = match hiding.noise {
            Noise::Digits => uniform::short(prg, self.choices.b, count * self.d())?,
            Noise::Gaussian { log_sigma } => gauss::gaussian(prg, log_sigma, count * self.d())?,
        };
        Ok(coeffs
            .chunks_exact(self.d())
            .map(|c| self.ring.from_i64(c))
            .collect::<hachi_co_math::Result<_>>()?)
    }

    /// The matrix $\mathbf M_1'$ of $[\mathbf M_1' \mid \mathbf I_{rows}]$: the matrix `kind`
    /// (`b"A'"` or `b"D'"`) of iteration `iteration`, $rows \times \nu$ ([`Hiding`]).
    ///
    /// # Errors
    /// [`Error::Parameter`] without [`Hiding`]; sampling errors.
    pub fn rand_matrix(
        &self,
        kind: &[u8],
        iteration: usize,
        rows: usize,
    ) -> Result<Vec<Vec<Poly>>> {
        let hiding = self
            .hiding()
            .ok_or_else(|| Error::Parameter("randomness without hiding choices".into()))?;
        self.matrix(kind, iteration, rows, hiding.nu)
    }

    /// $[\mathbf M_1' \mid \mathbf I_{rows}]\,\mathbf r$ for $\mathbf M_1'$ = `m1`
    /// ([`Ctx::rand_matrix`], $rows \times \nu$) and the randomness $\mathbf r$ of length
    /// $\nu + rows$.
    ///
    /// # Errors
    /// [`Error::Shape`] for a matrix or randomness of the wrong shape; [`Error::Parameter`]
    /// without [`Hiding`]; arithmetic errors.
    pub fn rand_term(&self, m1: &[Vec<Poly>], r: &[Poly]) -> Result<Vec<Poly>> {
        let hiding = self
            .hiding()
            .ok_or_else(|| Error::Parameter("randomness without hiding choices".into()))?;
        let rows = m1.len();
        if m1.iter().any(|row| row.len() != hiding.nu)
            || r.len() != hiding.nu + rows
            || r.iter().any(|p| p.0.len() != self.d())
        {
            return Err(Error::Shape(format!(
                "a matrix of {rows} rows of {} and randomness of {} ring elements of degree {} \
                 are expected",
                hiding.nu,
                hiding.nu + rows,
                self.d()
            )));
        }
        let mut out = self.ring.mat_vec(m1, &r[..hiding.nu])?;
        for (o, e) in out.iter_mut().zip(&r[hiding.nu..]) {
            *o = self.ring.add(o, e)?;
        }
        Ok(out)
    }

    /// The public matrix `kind` (`b"A"`, `b"B"`, `b"D"`, or with [`Hiding`] `b"A'"`, `b"D'"`) of
    /// iteration `iteration`, of shape
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
    /// The message chunks $\mathbf f_i$ ($2^r$ chunks of $2^m$ ring elements).
    pub message: Vec<Vec<Poly>>,
    /// The committed vectors $\mathbf s_i$.
    pub s: Vec<Vec<Poly>>,
    /// The randomness $\mathbf r_i$ of each chunk ($\nu + n_A$ ring elements each), or no
    /// chunk at all without [`Hiding`].
    pub r: Vec<Vec<Poly>>,
    /// The inner commitments $\mathbf t_i = \mathbf A \mathbf s_i$, plus
    /// $\mathbf A' \mathbf r_i$ with [`Hiding`] ($n_A$ each).
    pub t: Vec<Vec<Poly>>,
    /// $\hat{\mathbf t} = (\mathbf G^{-1}(\mathbf t_1), \dots)$, chunk-major: chunk $i$ occupies
    /// entries $[i n_A \delta, (i + 1) n_A \delta)$.
    pub t_hat: Vec<Poly>,
    /// The commitment $\mathbf u = \mathbf B \hat{\mathbf t}$ ($n_B$ ring elements).
    pub u: Vec<Poly>,
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
/// randomness `r`: one vector of $\nu + n_A$ ring elements per chunk with [`Hiding`], none
/// without.
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
    let chunks_of_r = if ctx.hiding().is_some() {
        layout.chunks()
    } else {
        0
    };
    if r.len() != chunks_of_r {
        return Err(Error::Shape(format!(
            "randomness for {chunks_of_r} chunks is expected"
        )));
    }
    let (ring, b, delta) = (&ctx.ring, ctx.choices.b, ctx.delta);
    let width = layout.width(delta);
    let a = ctx.matrix(b"A", iteration, ctx.choices.n_a, width)?;
    let bmat = ctx.matrix(
        b"B",
        iteration,
        ctx.choices.n_b,
        layout.chunks() * ctx.choices.n_a * delta,
    )?;
    let s: Vec<Vec<Poly>> = if layout.decomposed {
        message
            .iter()
            .map(|chunk| g_inv(ring, b, delta, chunk))
            .collect::<hachi_co_math::Result<_>>()?
    } else {
        message.clone()
    };
    let mut t: Vec<Vec<Poly>> = s
        .iter()
        .map(|si| ring.mat_vec(&a, si))
        .collect::<hachi_co_math::Result<_>>()?;
    if !r.is_empty() {
        let a_rand = ctx.rand_matrix(b"A'", iteration, ctx.choices.n_a)?;
        for (ti, ri) in t.iter_mut().zip(&r) {
            let term = ctx.rand_term(&a_rand, ri)?;
            for (x, y) in ti.iter_mut().zip(&term) {
                *x = ring.add(x, y)?;
            }
        }
    }
    let mut t_hat = Vec::with_capacity(layout.chunks() * ctx.choices.n_a * delta);
    for ti in &t {
        t_hat.extend(g_inv(ring, b, delta, ti)?);
    }
    let u = ring.mat_vec(&bmat, &t_hat)?;
    Ok(Committed {
        layout,
        message,
        s,
        r,
        t,
        t_hat,
        u,
    })
}
