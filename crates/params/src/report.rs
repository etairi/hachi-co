//! The constraint report: every condition of the draft's $\mathsf{Setup}$ and of
//! `docs/m0-spec.md`, with whether it holds and how severe a violation is.
//!
//! **Severity.** A condition is [`Severity::Fatal`] when the draft's Setup, a definition the
//! construction relies on, or the implementation's conventions *require* it: with it violated
//! the parameters are not an instance of the construction (or not one this implementation can
//! run). It is a [`Severity::Warning`] when it is a *property the parameters are meant to
//! have* (a completeness bound the draft asserts, or the knowledge-error target $\lambda$),
//! whose failure leaves the construction well defined but weaker than claimed. This is how the
//! plan's design rule ("only the ones the chosen preset requires are fatal",
//! `docs/hachi-co-plan.md` §4) is realised: the toy presets are insecure by design and keep a
//! failing knowledge-error warning, while every secure preset satisfies every condition.

use crate::cascade::Params;
use crate::error::Result;
use crate::rules;
use core::fmt;
use serde::Serialize;

/// How serious a violated condition is (see the [module documentation](self)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Severity {
    /// Required by the draft's Setup, by a definition, or by the implementation's conventions.
    Fatal,
    /// A claimed property or the security target; the construction stays well defined.
    Warning,
}

/// The conditions the report checks, in the order of `docs/m0-spec.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum ConstraintId {
    /// $q$ is prime (6_Collaborative_Hachi.tex:272). Fatal.
    QPrime,
    /// $q \equiv 5 \pmod 8$ (6_Collaborative_Hachi.tex:272), needed for
    /// $\mathcal R_q^H \cong \mathbb F_{q^k}$ (Hachi, Lemma 5). Fatal.
    QFiveMod8,
    /// $q > 1 + 2^{\ell+3} n \omega^2 b$ (6_Collaborative_Hachi.tex:272; the draft does not
    /// derive it). Fatal.
    QAboveRangeGuard,
    /// $q > 4\bar\omega\bar\beta$ (6_Collaborative_Hachi.tex:272): otherwise the
    /// $\mathsf{MSIS}$ bound $2\bar\omega\bar\beta$ of relaxed binding (line 563) is at least
    /// $q/2$ and the instance is trivially solvable. Fatal.
    QAboveBindingGuard,
    /// $2\beta < q$, so that every $z$ with $\lVert z \rVert_\infty \le \beta$ has a unique
    /// centered representative (`docs/m0-spec.md`, "Choosing q"). Fatal.
    CenteredResponses,
    /// $1 \le \alpha - \kappa \le \ell$ (6_Collaborative_Hachi.tex:273). Only the lower bound
    /// can fail in a report: $\alpha - \kappa > \ell$ makes $m < 0$, so
    /// [`derive`](fn@crate::derive) returns [`Error::NegativeExponent`](crate::Error::NegativeExponent).
    /// Fatal.
    AlphaMinusKappa,
    /// $k \mid d/2$ (4_preliminaries.tex:73), needed for
    /// $H = \langle\sigma_{-1}, \sigma_{4k+1}\rangle$; for powers of two it is equivalent to
    /// $\alpha - \kappa \ge 1$. Fatal.
    KDividesHalfD,
    /// $b \ge 3$ (6_Collaborative_Hachi.tex:269). Fatal.
    BAtLeast3,
    /// $m_\rho > n_D$ (6_Collaborative_Hachi.tex:269), needed for the $\mathsf{MLWE}$ hiding
    /// argument with secret dimension $m_\rho - n_D$ (5_Building_Blocks.tex:193). Fatal.
    MRhoAboveND,
    /// $q < 2^{63}$: residues are `u64` with 128-bit products (`docs/m0-spec.md`,
    /// "Modulus"). Fatal.
    QBelow2Pow63,
    /// $M \le e^{1/n}$ (6_Collaborative_Hachi.tex:275), which gives the completeness bound
    /// $M^{-n} \ge 1/e$ (5_Building_Blocks.tex:193). Always true for the cascade's $\gamma$
    /// ([`rules::rejection_constant_within`]). Warning.
    RejectionConstant,
    /// $\mathbf J^{-1}$ represents $\beta$ with $\tau$ digits:
    /// $(\lceil b/2 \rceil - 1)(b^\tau - 1)/(b - 1) \ge \beta$ (4_preliminaries.tex:100 defines
    /// $\mathbf J^{-1}$ on every $\mathbf t$ with $\lVert \mathbf t \rVert_\infty \le \beta$).
    /// Fatal.
    JInvRepresentable,
    /// The knowledge error of the folding challenge, $2^{r+1}/\lvert C \rvert \le 2^{-\lambda}$
    /// ([`rules::knowledge_error_holds`]). Warning.
    KnowledgeError,
}

impl ConstraintId {
    /// Every condition, in report order.
    pub const ALL: [ConstraintId; 13] = [
        ConstraintId::QPrime,
        ConstraintId::QFiveMod8,
        ConstraintId::QAboveRangeGuard,
        ConstraintId::QAboveBindingGuard,
        ConstraintId::CenteredResponses,
        ConstraintId::AlphaMinusKappa,
        ConstraintId::KDividesHalfD,
        ConstraintId::BAtLeast3,
        ConstraintId::MRhoAboveND,
        ConstraintId::QBelow2Pow63,
        ConstraintId::RejectionConstant,
        ConstraintId::JInvRepresentable,
        ConstraintId::KnowledgeError,
    ];

    /// The severity of a violation of this condition.
    pub const fn severity(self) -> Severity {
        match self {
            ConstraintId::RejectionConstant | ConstraintId::KnowledgeError => Severity::Warning,
            _ => Severity::Fatal,
        }
    }

    /// The condition as a formula (plain text).
    pub const fn statement(self) -> &'static str {
        match self {
            ConstraintId::QPrime => "q prime",
            ConstraintId::QFiveMod8 => "q = 5 mod 8",
            ConstraintId::QAboveRangeGuard => "q > 1 + 2^(l+3) n omega^2 b",
            ConstraintId::QAboveBindingGuard => "q > 4 omega_bar beta_bar",
            ConstraintId::CenteredResponses => "2 beta < q",
            ConstraintId::AlphaMinusKappa => "1 <= alpha - kappa <= l",
            ConstraintId::KDividesHalfD => "k | d/2",
            ConstraintId::BAtLeast3 => "b >= 3",
            ConstraintId::MRhoAboveND => "m_rho > n_D",
            ConstraintId::QBelow2Pow63 => "q < 2^63",
            ConstraintId::RejectionConstant => "M <= e^(1/n)",
            ConstraintId::JInvRepresentable => "(ceil(b/2) - 1)(b^tau - 1)/(b - 1) >= beta",
            ConstraintId::KnowledgeError => "2^(r+1)/|C| <= 2^-lambda",
        }
    }

    /// Where the condition comes from.
    pub const fn source(self) -> &'static str {
        match self {
            ConstraintId::QPrime
            | ConstraintId::QFiveMod8
            | ConstraintId::QAboveRangeGuard
            | ConstraintId::QAboveBindingGuard => "6_Collaborative_Hachi.tex:272",
            ConstraintId::CenteredResponses => "docs/m0-spec.md (unique centered responses)",
            ConstraintId::AlphaMinusKappa => "6_Collaborative_Hachi.tex:273",
            ConstraintId::KDividesHalfD => "4_preliminaries.tex:73",
            ConstraintId::BAtLeast3 | ConstraintId::MRhoAboveND => "6_Collaborative_Hachi.tex:269",
            ConstraintId::QBelow2Pow63 => "docs/m0-spec.md (64-bit residues)",
            ConstraintId::RejectionConstant => "6_Collaborative_Hachi.tex:275",
            ConstraintId::JInvRepresentable => {
                "4_preliminaries.tex:100; docs/m0-spec.md (tau rule)"
            }
            ConstraintId::KnowledgeError => "Hachi Lemma 4; docs/m0-spec.md (Omega::MinForTarget)",
        }
    }
}

/// One checked condition.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Constraint {
    /// Which condition.
    pub id: ConstraintId,
    /// Its severity ([`ConstraintId::severity`]).
    pub severity: Severity,
    /// Whether the parameters satisfy it.
    pub holds: bool,
    /// The values compared, for humans.
    pub detail: String,
}

/// The result of checking every [`ConstraintId`] against a [`Params`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Report {
    /// One entry per condition, in the order of [`ConstraintId::ALL`] (so the entry of `id` is
    /// at index `id as usize`).
    constraints: Vec<Constraint>,
}

impl Report {
    /// Checks every condition against `p` (called by [`derive`](fn@crate::derive), the only
    /// source of reports).
    ///
    /// # Errors
    ///
    /// Only [`Error::Undecided`](crate::Error::Undecided) from
    /// [`rules::rejection_constant_within`], which cannot happen for parameters produced by
    /// [`derive`](fn@crate::derive).
    pub(crate) fn new(p: &Params) -> Result<Self> {
        let c = &p.choices;
        let q = u128::from(p.q);
        let alpha_minus_kappa = i64::from(c.alpha) - i64::from(c.kappa);
        let size = rules::challenge_set_size(c.challenge_set, p.d, p.omega);
        let log2_error = f64::from(p.r) + 1.0 - rules::log2_biguint(&size);
        let mut constraints = Vec::with_capacity(ConstraintId::ALL.len());
        let mut push = |id: ConstraintId, holds: bool, detail: String| {
            constraints.push(Constraint {
                id,
                severity: id.severity(),
                holds,
                detail,
            });
        };
        push(
            ConstraintId::QPrime,
            rules::is_prime_u64(p.q),
            format!("q = {}", p.q),
        );
        push(
            ConstraintId::QFiveMod8,
            p.q % 8 == 5,
            format!("q mod 8 = {}", p.q % 8),
        );
        push(
            ConstraintId::QAboveRangeGuard,
            q > p.range_guard,
            format!("q = {}, 1 + 2^(l+3) n omega^2 b = {}", p.q, p.range_guard),
        );
        push(
            ConstraintId::QAboveBindingGuard,
            q > p.binding_guard,
            format!(
                "q = {}, 4 omega_bar beta_bar = 16 omega b^tau = {}",
                p.q, p.binding_guard
            ),
        );
        push(
            ConstraintId::CenteredResponses,
            p.beta.checked_mul(2).is_some_and(|two_beta| two_beta < q),
            format!("2 beta = 2 * {}, q = {}", p.beta, p.q),
        );
        push(
            ConstraintId::AlphaMinusKappa,
            alpha_minus_kappa >= 1 && alpha_minus_kappa <= i64::from(c.ell),
            format!("alpha - kappa = {alpha_minus_kappa}, l = {}", c.ell),
        );
        push(
            ConstraintId::KDividesHalfD,
            p.d >= 2 && (p.d / 2).is_multiple_of(p.k),
            format!("k = {}, d = {}", p.k, p.d),
        );
        push(ConstraintId::BAtLeast3, c.b >= 3, format!("b = {}", c.b));
        push(
            ConstraintId::MRhoAboveND,
            c.m_rho > c.n_d,
            format!("m_rho = {}, n_D = {}", c.m_rho, c.n_d),
        );
        push(
            ConstraintId::QBelow2Pow63,
            p.q < 1 << 63,
            format!("log2 q = {:.3}", (p.q as f64).log2()),
        );
        push(
            ConstraintId::RejectionConstant,
            rules::rejection_constant_within(c.n, p.big_n, p.gamma, p.beta1)?,
            format!(
                "ln M = {:.6}, 1/n = {:.6}",
                p.log2_rejection_m * core::f64::consts::LN_2,
                1.0 / c.n as f64
            ),
        );
        push(
            ConstraintId::JInvRepresentable,
            p.j_inv_capacity >= p.beta,
            format!(
                "tau = {} ({:?} rule), capacity = {}, beta = {}",
                p.tau, c.tau_rule, p.j_inv_capacity, p.beta
            ),
        );
        push(
            ConstraintId::KnowledgeError,
            rules::knowledge_error_holds(&size, p.r, c.lambda),
            format!(
                "log2(2^(r+1)/|C|) = {log2_error:.2} ({:?}, d = {}, omega = {}, r = {}), target -{}",
                c.challenge_set, p.d, p.omega, p.r, c.lambda
            ),
        );
        debug_assert!(constraints.iter().map(|c| c.id).eq(ConstraintId::ALL));
        Ok(Self { constraints })
    }

    /// Every checked condition, in the order of [`ConstraintId::ALL`].
    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    /// The entry for `id`.
    pub fn get(&self, id: ConstraintId) -> &Constraint {
        // Report::new pushes the conditions in the order of ConstraintId::ALL, which is the
        // declaration order of the enum, and the field is private.
        let entry = &self.constraints[id as usize];
        debug_assert_eq!(entry.id, id);
        entry
    }

    /// Whether `id` holds.
    pub fn holds(&self, id: ConstraintId) -> bool {
        self.get(id).holds
    }

    /// The violated conditions.
    pub fn violations(&self) -> impl Iterator<Item = &Constraint> {
        self.constraints.iter().filter(|c| !c.holds)
    }

    /// The violated conditions of severity [`Severity::Fatal`].
    pub fn fatal_violations(&self) -> impl Iterator<Item = &Constraint> {
        self.violations().filter(|c| c.severity == Severity::Fatal)
    }

    /// No [`Severity::Fatal`] condition is violated: the parameters are an instance of the
    /// construction that this implementation can run (they may still miss the security target).
    pub fn is_admissible(&self) -> bool {
        self.fatal_violations().next().is_none()
    }

    /// Every condition holds, warnings included.
    pub fn all_hold(&self) -> bool {
        self.violations().next().is_none()
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for c in &self.constraints {
            let status = match (c.holds, c.severity) {
                (true, _) => "ok     ",
                (false, Severity::Fatal) => "FATAL  ",
                (false, Severity::Warning) => "warning",
            };
            writeln!(
                f,
                "{status}  {:<44} {}  [{}]",
                c.id.statement(),
                c.detail,
                c.id.source()
            )?;
        }
        Ok(())
    }
}
