//! The arithmetic context ([`Ctx`]) and the inner-outer commitment (paper, Section 4.1):
//! $\mathbf t_i = \mathbf A \mathbf s_i$, $\hat{\mathbf t}_i = \mathbf G^{-1}(\mathbf t_i)$ and
//! $\mathbf u = \mathbf B \hat{\mathbf t}$, with $\mathbf s_i = \mathbf G^{-1}(\mathbf f_i)$ or
//! $\mathbf s_i = \mathbf f_i$ according to the [`Layout`].

use crate::params::{HachiChoices, Layout, check};
use crate::{Error, Result};
use hachi_co_math::decompose::{g_inv, num_digits};
use hachi_co_math::{Fqk, Poly, Ring, Subfield};
use hachi_co_sampling::ShakePrg;
use hachi_co_sampling::matrix::expand;
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

    /// The public matrix `kind` (`b"A"`, `b"B"` or `b"D"`) of iteration `iteration`, of shape
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
    /// The inner commitments $\mathbf t_i = \mathbf A \mathbf s_i$ ($n_A$ each).
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

/// Commits to `message` in `layout` with the matrices of iteration `iteration`.
///
/// # Errors
/// [`Error::Shape`] for a message of the wrong shape; arithmetic errors.
pub fn commit<const K: usize>(
    ctx: &Ctx<K>,
    iteration: usize,
    layout: Layout,
    message: Vec<Vec<Poly>>,
) -> Result<Committed> {
    check_message(ctx, &layout, &message)?;
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
    let t: Vec<Vec<Poly>> = s
        .iter()
        .map(|si| ring.mat_vec(&a, si))
        .collect::<hachi_co_math::Result<_>>()?;
    let mut t_hat = Vec::with_capacity(layout.chunks() * ctx.choices.n_a * delta);
    for ti in &t {
        t_hat.extend(g_inv(ring, b, delta, ti)?);
    }
    let u = ring.mat_vec(&bmat, &t_hat)?;
    Ok(Committed {
        layout,
        message,
        s,
        t,
        t_hat,
        u,
    })
}
