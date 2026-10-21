//! The context of one Hachi-Co instance: the parameters, their [`Layout`], the rings
//! $\mathcal R\_q$ and $\mathbb Z\_q\[X\]/(X^{2d} + 1)$, the subfield $\mathcal R\_q^H \cong
//! \mathbb F\_{q^k}$, and the public matrices $\mathbf A, \mathbf B, \mathbf D$ of
//! $\mathsf{Setup}$ (`6_Collaborative_Hachi.tex`:263-298).

use crate::data::{Matrix, check_len, check_polys};
use crate::{Error, Layout, Result};
use hachi_co_math::{Fqk, Poly, Ring, Subfield};
use hachi_co_params::Params;
use hachi_co_sampling::matrix;

/// The PRG domain of $\mathbf A$ in [`Context::setup`].
pub const SETUP_DOMAIN_A: &[u8] = b"hachi-co/setup/A";
/// The PRG domain of $\mathbf B$ in [`Context::setup`].
pub const SETUP_DOMAIN_B: &[u8] = b"hachi-co/setup/B";
/// The PRG domain of $\mathbf D$ in [`Context::setup`].
pub const SETUP_DOMAIN_D: &[u8] = b"hachi-co/setup/D";

/// The public matrices of $\mathsf{Setup}$ (`6_Collaborative_Hachi.tex`:283-288), row-major,
/// entries in $\mathcal R\_q$ with canonical residues:
/// $\mathbf A \in \mathcal R\_q^{n\_A \times \delta 2^m}$,
/// $\mathbf B \in \mathcal R\_q^{n\_B \times n\_A \delta 2^r}$ and
/// $\mathbf D \in \mathcal R\_q^{n\_D \times (\delta(2^r + n + n n\_A) + n m\_\rho)}$.
/// The parameters of the base PCS (lines 289-292) are not part of them: the base PCS is a value
/// of its own ([`BasePcs`](crate::BasePcs)), set up over the same $q$ and $k$ and passed to Eval
/// and Verify.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicParams {
    /// $\mathbf A$, the inner commitment key: $\mathbf t\_i = \mathbf A \mathbf s\_i$.
    pub a: Matrix,
    /// $\mathbf B$, the outer commitment key: $\mathbf u = \mathbf B \hat{\mathbf t}$.
    pub b: Matrix,
    /// $\mathbf D$, the key of the first message $\mathbf v$ of the ring-to-linear step
    /// (`5_Building_Blocks.tex`:120).
    pub d: Matrix,
}

/// A matrix of `rows` rows of `cols` polynomials with $d$ canonical coefficients each.
fn check_matrix(
    what: &'static str,
    m: &Matrix,
    rows: usize,
    cols: usize,
    d: usize,
    q: u64,
) -> Result<()> {
    check_len(what, rows, m.len())?;
    for row in m {
        check_polys(what, row, cols, d)?;
        for value in row.iter().flat_map(|p| p.0.iter()) {
            if *value >= q {
                return Err(Error::NonCanonical {
                    what,
                    value: *value,
                    q,
                });
            }
        }
    }
    Ok(())
}

impl PublicParams {
    /// Checks the shapes $n\_A \times \delta 2^m$, $n\_B \times n\_A \delta 2^r$ and
    /// $n\_D \times w\_D$, that every entry has $d$ coefficients, and that every residue is below
    /// $q$ (the matrices are absorbed into transcripts, so their encoding must be canonical).
    ///
    /// # Errors
    /// [`Error::Shape`] or [`Error::NonCanonical`].
    pub fn validate(&self, layout: &Layout, q: u64) -> Result<()> {
        let d = layout.d;
        check_matrix("A", &self.a, layout.n_a, layout.width_a, d, q)?;
        check_matrix("B", &self.b, layout.n_b, layout.width_b, d, q)?;
        check_matrix("D", &self.d, layout.n_d, layout.width_d, d, q)
    }
}

/// Everything the parties and the verifier share about one instance (`docs/m1-spec.md`,
/// `hachi-co-core`): the parameters, the [`Layout`], $\mathcal R\_q$ (`ring`),
/// $\mathbb Z\_q\[X\]/(X^{2d} + 1)$ (`ring2`, for the quotients of `5_Building_Blocks.tex`:198-200),
/// the subfield $\mathcal R\_q^H \cong \mathbb F\_{q^K}$ (`sub`) and the public matrices (`pp`).
///
/// Built by [`Context::new`], [`Context::setup`] or (feature `fixtures`) `Fixture::context`. The
/// fields are public for reading; the struct is non-exhaustive so that it is always built through
/// a validating constructor.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Context<const K: usize> {
    /// The derived parameters (`6_Collaborative_Hachi.tex`:265-281; `hachi-co-params`).
    pub params: Params,
    /// The positions, rows, ownership and hypercube sizes derived from `params`.
    pub layout: Layout,
    /// $\mathcal R\_q = \mathbb Z\_q\[X\]/(X^d + 1)$ (`4_preliminaries.tex`:67).
    pub ring: Ring,
    /// $\mathbb Z\_q\[X\]/(X^{2d} + 1)$, in which the quotient shares
    /// $\langle \mathbf r \rangle\_p$ live (`5_Building_Blocks.tex`:198-200).
    pub ring2: Ring,
    /// $\mathcal R\_q^H \cong \mathbb F\_{q^K}$ with $\psi$, $\psi^{-1}$, $\mathrm{Tr}\_H$ and $E$
    /// (`4_preliminaries.tex`:73-84; Hachi, Lemma 5 and Theorem 2).
    pub sub: Subfield<K>,
    /// The public matrices $\mathbf A, \mathbf B, \mathbf D$.
    pub pp: PublicParams,
}

impl<const K: usize> Context<K> {
    /// The context of `params` with the public matrices `pp`.
    ///
    /// Checks that $K = k$, builds the [`Layout`] (with its invariants), $\mathcal R\_q$,
    /// $\mathbb Z\_q\[X\]/(X^{2d} + 1)$ and $\mathcal R\_q^H$ (which need a prime
    /// $q \equiv 5 \pmod 8$, $q \lt 2^{63}$, $2 \le d \le 2048$ and $k \mid d/2$), checks that the
    /// base $b \ge 2$, that $\gamma$, $\beta\_1$, $\beta$ and $\omega$ fit in `i64` (the masks,
    /// responses and challenges are `i64`), and validates the shapes and residues of `pp`
    /// ([`PublicParams::validate`]). It does not enforce the constraint report of
    /// `hachi-co-params` (toy parameters miss the knowledge-error target on purpose); check
    /// [`hachi_co_params::Report::is_admissible`] where that matters.
    ///
    /// # Errors
    /// [`Error::Parameter`], [`Error::Layout`], [`Error::Math`], [`Error::Shape`] or
    /// [`Error::NonCanonical`].
    pub fn new(params: Params, pp: PublicParams) -> Result<Self> {
        if params.k != K as u64 {
            return Err(Error::Parameter(format!(
                "the context is built for K = {K} but the parameters have k = {}",
                params.k
            )));
        }
        let layout = Layout::new(&params)?;
        let ring = Ring::new(params.q, layout.d)?;
        let ring2 = Ring::new(params.q, 2 * layout.d)?;
        let sub = Subfield::<K>::new(&ring)?;
        if params.choices.b < 2 {
            return Err(Error::Parameter(format!(
                "the decomposition base b = {} is below 2",
                params.choices.b
            )));
        }
        let max = i64::MAX as u128;
        for (name, value) in [
            ("gamma", params.gamma),
            ("beta_1", params.beta1),
            ("beta", params.beta),
            ("omega", u128::from(params.omega)),
        ] {
            if value > max {
                return Err(Error::Parameter(format!(
                    "{name} = {value} does not fit in i64"
                )));
            }
        }
        pp.validate(&layout, params.q)?;
        Ok(Self {
            params,
            layout,
            ring,
            ring2,
            sub,
            pp,
        })
    }

    /// $\mathsf{Setup}$ (`6_Collaborative_Hachi.tex`:283-288) from a 32-byte seed: entry
    /// $(i, j)$ of each matrix is uniform in $\mathcal R\_q$, read from its own stream
    /// [`hachi_co_sampling::matrix::expand`] with the domains [`SETUP_DOMAIN_A`],
    /// [`SETUP_DOMAIN_B`] and [`SETUP_DOMAIN_D`] (index $i \cdot \mathrm{cols} + j$). The Sage
    /// oracle uses its own PRG (`ref/README.md`, choice 16), so its fixtures carry the matrices
    /// and the fixture tests read them instead.
    ///
    /// # Errors
    /// As [`Context::new`], and [`Error::Sampling`] if the samplers fail.
    pub fn setup(params: Params, seed: &[u8; 32]) -> Result<Self> {
        let layout = Layout::new(&params)?;
        let (q, d) = (params.q, layout.d);
        let expand = |domain: &[u8], rows: usize, cols: usize| -> Result<Matrix> {
            Ok(matrix::expand(seed, domain, rows, cols, d, q)?
                .into_iter()
                .map(|row| row.into_iter().map(Poly).collect())
                .collect())
        };
        let pp = PublicParams {
            a: expand(SETUP_DOMAIN_A, layout.n_a, layout.width_a)?,
            b: expand(SETUP_DOMAIN_B, layout.n_b, layout.width_b)?,
            d: expand(SETUP_DOMAIN_D, layout.n_d, layout.width_d)?,
        };
        Self::new(params, pp)
    }

    /// The field $\mathbb F\_{q^K} \cong \mathcal R\_q^H$ of the coordinates (Hachi, Lemma 5).
    pub fn field(&self) -> &Fqk<K> {
        self.sub.field()
    }

    /// The modulus $q$ (`6_Collaborative_Hachi.tex`:272).
    pub fn q(&self) -> u64 {
        self.ring.modulus()
    }

    /// The decomposition base $b$ (`6_Collaborative_Hachi.tex`:269).
    pub fn b(&self) -> u64 {
        self.params.choices.b
    }

    /// The challenge radius $\omega$: $\mathcal C = \lbrace c : \Vert c \Vert\_1 \le \omega
    /// \rbrace$ (`5_Building_Blocks.tex`:122).
    pub fn omega(&self) -> u64 {
        self.params.omega
    }

    /// The mask width $\gamma = 2nN\beta\_1$ (`6_Collaborative_Hachi.tex`:275): masks are uniform
    /// in $\[-\gamma, \gamma\]^N$. ([`Context::new`] checks that it fits in `i64`; if `params` is
    /// changed afterwards, the value saturates at `u64::MAX`.)
    pub fn gamma(&self) -> u64 {
        u64::try_from(self.params.gamma).unwrap_or(u64::MAX)
    }

    /// $\beta\_1 = 2^{r - \lceil \log n \rceil} \omega b$, the bound on each prover's fold
    /// (`6_Collaborative_Hachi.tex`:275; `5_Building_Blocks.tex`:129). Saturating as
    /// [`Context::gamma`].
    pub fn beta1(&self) -> u64 {
        u64::try_from(self.params.beta1).unwrap_or(u64::MAX)
    }

    /// $\beta = n\gamma$, the bound that prover 1 checks on $\Vert \mathbf z \Vert\_\infty$
    /// (`6_Collaborative_Hachi.tex`:275; `5_Building_Blocks.tex`:131). Saturating as
    /// [`Context::gamma`].
    pub fn beta(&self) -> u64 {
        u64::try_from(self.params.beta).unwrap_or(u64::MAX)
    }
}
