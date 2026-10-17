//! The free choices of $\mathsf{Setup}(1^\lambda, 1^n, 1^t, 1^{2^\ell})$
//! (6_Collaborative_Hachi.tex:263-281).
//!
//! The draft fixes only the asymptotic behaviour of these quantities (lines 266-273); a concrete
//! parameter set picks them, and [`derive`](fn@crate::derive) computes everything else.

use serde::{Deserialize, Serialize};

/// The radius $\omega$ of the challenge set (6_Collaborative_Hachi.tex:271, "$\omega = O(d)$").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Omega {
    /// A fixed radius $\omega \ge 1$.
    Fixed(u64),
    /// The smallest $\omega \ge 1$ whose challenge set reaches the knowledge-error target
    /// $2^{r+1}/\lvert C \rvert \le 2^{-\lambda}$, for the chosen [`ChallengeSet`]
    /// ([`rules::min_omega`](crate::rules::min_omega)).
    MinForTarget,
}

/// The challenge set $C$ from which each of the $2^r$ folding challenges is drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChallengeSet {
    /// The draft's $\ell_1$ ball $C = \lbrace c \in \mathcal R_q : \lVert c \rVert_1 \le \omega
    /// \rbrace$ (5_Building_Blocks.tex:122).
    #[default]
    L1Ball,
    /// Exactly $\omega$ nonzero coefficients in $\lbrace \pm 1 \rbrace$ (the single-prover
    /// prototype's set). Every such $c$ has $\lVert c \rVert_1 = \omega$, so it is a subset of
    /// the $\ell_1$ ball and the relaxed-opening bound $\bar\omega = 2\omega$ still applies.
    Sparse,
}

/// How the number of digits $\tau$ of $\mathbf J^{-1}$ is chosen from $\beta$
/// ([`rules::tau`](crate::rules::tau)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TauRule {
    /// The least $\tau$ with $(\lceil b/2 \rceil - 1)(b^\tau - 1)/(b - 1) \ge \beta$: every
    /// $z$ with $\lVert z \rVert_\infty \le \beta$ is representable (user decision of 2026-10-17,
    /// `docs/m0-spec.md`). The default.
    #[default]
    Tight,
    /// The draft's $\tau = \lceil \log_b \beta \rceil + 1$ (6_Collaborative_Hachi.tex:276).
    Draft,
    /// $\tau = \lceil \log_b \beta \rceil$, which does not represent $\beta$ in general
    /// (balanced digits cover only about $b^\tau/2$); kept to show the failure in the report.
    Ceil,
}

/// How the modulus $q$ is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QChoice {
    /// A given modulus; [`derive`](fn@crate::derive) only checks it (the report flags what
    /// fails). Must be at least 2 so that $\delta$ is defined.
    Fixed(u64),
    /// The smallest prime $p \equiv 5 \pmod 8$ with
    /// $p > \max(16\omega b^\tau,\ 1 + 2^{\ell+3} n \omega^2 b,\ 2\beta + 1)$, where $\tau$ and
    /// $\beta$ are computed with $\delta = \lceil \log_b p \rceil$ (`docs/m0-spec.md`,
    /// "Choosing q"). This leaves no MSIS margin: secure presets fix $q$ from the Sage script.
    Smallest,
}

/// The inputs of [`derive`](fn@crate::derive): the arguments of $\mathsf{Setup}$ and the free
/// choices of 6_Collaborative_Hachi.tex:266-273.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Choices {
    /// The security parameter $\lambda$, in bits. In this crate it only enters
    /// [`Omega::MinForTarget`] and the knowledge-error check of the report.
    pub lambda: u32,
    /// The number of provers $n \ge 1$.
    pub n: u64,
    /// The threshold $t$ (4_preliminaries.tex:131: $t \in [n-1]$, $t < n/2$). The construction
    /// does not use it; it is recorded and not checked.
    pub t: u64,
    /// The number of variables $\ell$ of the committed polynomial ($2^\ell$ coefficients).
    pub ell: u32,
    /// $\alpha$, with ring degree $d = 2^\alpha$.
    pub alpha: u32,
    /// $\kappa$, with extension degree $k = 2^\kappa$ ($\mathcal R_q^H \cong \mathbb F_{q^k}$).
    pub kappa: u32,
    /// The decomposition base $b$ (the draft requires $b \ge 3$; the cascade is evaluated for
    /// every $b \ge 2$ so that the report can flag $b = 2$).
    pub b: u64,
    /// The challenge radius $\omega$.
    pub omega: Omega,
    /// The challenge set.
    pub challenge_set: ChallengeSet,
    /// The rule for $\tau$.
    pub tau_rule: TauRule,
    /// The height $n_A \ge 1$ of $\mathbf A$.
    pub n_a: u64,
    /// The height $n_B \ge 1$ of $\mathbf B$.
    pub n_b: u64,
    /// The height $n_D \ge 1$ of $\mathbf D$.
    pub n_d: u64,
    /// The number $m_\rho$ of ring elements of each prover's commitment randomness
    /// $\boldsymbol\rho_i \in S_b^{m_\rho}$ (5_Building_Blocks.tex:110); the draft requires
    /// $m_\rho > n_D$.
    pub m_rho: u64,
    /// The modulus.
    pub q: QChoice,
}
