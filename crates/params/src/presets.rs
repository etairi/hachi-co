//! Named parameter sets: the toy presets of `docs/m0-spec.md`, the development presets of
//! `docs/m1-spec.md` (WP11) and the secure 64-bit presets that `parameters.sage` writes to
//! `crates/params/presets/secure.json`.
//!
//! The JSON file is embedded at compile time ([`SECURE_JSON`]). Each secure entry stores the
//! [`Choices`] (inside its [`Params`]), the parameters Sage derived from them, and the lattice
//! estimator's results; the test `tests/secure_presets.rs` re-derives every entry in Rust and
//! requires exact agreement.

use crate::cascade::{Params, derive};
use crate::choices::{ChallengeSet, Choices, Omega, QChoice, TauRule};
use crate::error::{Error, Result};
use crate::report::Report;
use serde::{Deserialize, Serialize};

/// The contents of `crates/params/presets/secure.json`, embedded at compile time.
pub const SECURE_JSON: &str = include_str!("../presets/secure.json");

/// Whether a preset is meant to be secure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PresetKind {
    /// Small and insecure, for the Sage oracle and for tests (`docs/m0-spec.md`, "Toy presets").
    Toy,
    /// Insecure like the toys (the same choices with $n = 1$ and a larger $\ell$), for the
    /// development and the end-to-end tests of the composed PCS (`docs/m1-spec.md`, WP11):
    /// [`dev_presets`].
    Dev,
    /// Meets the $\lambda$-bit target according to the lattice estimator's estimates for the
    /// $\mathsf{MSIS}$ and $\mathsf{MLWE}$ instances of the draft (6_Collaborative_Hachi.tex:563,
    /// 5_Building_Blocks.tex:193), satisfies every condition of the report, and has
    /// $q < 2^{63}$. Estimates, not proofs (the draft's security proofs are unfinished).
    Secure,
}

/// A named parameter set.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Preset {
    /// The name, e.g. `toy-n2`, `dev-l16` or `secure-l20-n2-a10`.
    pub name: String,
    /// Toy, development or secure.
    pub kind: PresetKind,
    /// The choices to [`derive`](fn@crate::derive) from.
    pub choices: Choices,
    /// The estimator's results (secure presets only).
    pub security: Option<SecurityEstimates>,
}

impl Preset {
    /// [`derive`](fn@crate::derive) on [`Preset::choices`].
    ///
    /// # Errors
    ///
    /// As [`derive`](fn@crate::derive); never for the presets of this module (the tests derive all of them).
    pub fn derive(&self) -> Result<(Params, Report)> {
        derive(&self.choices)
    }
}

/// The estimate for one $\mathsf{MSIS}_{q, d, n_X, w, B}$ instance in the $\infty$-norm
/// (4_preliminaries.tex:105-108), as computed by `parameters.sage`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MsisEstimate {
    /// The number of equations over $\mathbb Z_q$, $n_X d$.
    pub rank: u64,
    /// The number of integer columns of the matrix, $w d$.
    pub columns: u128,
    /// The number of columns the cheapest attack uses: `parameters.sage` evaluates `SIS.lattice`
    /// on $m'$ columns for a range of $m'$ and keeps the minimum (see `presets/README.md`).
    pub estimator_columns: u64,
    /// The $\infty$-norm bound $B$.
    pub bound: u128,
    /// $\mathrm{rank} \cdot \log_2(q/(2B + 1))$, the security against the trivial
    /// linear-algebra attack that the estimator does not model.
    pub trivial_bits: f64,
    /// $\log_2$ of the estimator's cost (`SIS.lattice`, minimised over the number of columns
    /// used), or `None` if no width gave a finite cost (every evaluation infinite or beyond
    /// $2^{1000}$).
    pub lattice_bits: Option<f64>,
    /// How [`MsisEstimate::lattice_bits`] was obtained.
    pub method: String,
}

/// The estimate for the $\mathsf{MLWE}_{q, d, n_D, m_\rho - n_D, b}$ instance of the hiding
/// argument (5_Building_Blocks.tex:193), as computed by `parameters.sage`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlweEstimate {
    /// The secret dimension over $\mathbb Z$, $(m_\rho - n_D) d$.
    pub secret_dim: u64,
    /// The number of samples over $\mathbb Z$, $n_D d$.
    pub samples: u64,
    /// Secret and error coefficients are uniform in this range, the coefficients of $S_b$
    /// (4_preliminaries.tex:88).
    pub digit_range: (i64, i64),
    /// $\log_2$ of the cheapest attack's cost, or `None` if every attack was infinite.
    pub bits: Option<f64>,
    /// The cheapest attack.
    pub attack: String,
}

/// The estimator's results for a secure preset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecurityEstimates {
    /// The target $\lambda$.
    pub lambda: u32,
    /// $\mathsf{MSIS}$ for $\mathbf A$ with bound $2\bar\omega\bar\beta$
    /// (6_Collaborative_Hachi.tex:563).
    pub msis_a: MsisEstimate,
    /// $\mathsf{MSIS}$ for $\mathbf B$ with bound $2\bar\gamma$ (6_Collaborative_Hachi.tex:563).
    pub msis_b: MsisEstimate,
    /// $\mathsf{MLWE}$ for the hiding of the first message (5_Building_Blocks.tex:193).
    pub mlwe: MlweEstimate,
}

/// The lattice estimator that produced `secure.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EstimatorInfo {
    /// The repository URL.
    pub repository: String,
    /// The git revision.
    pub revision: String,
}

/// One entry of `secure.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecurePresetEntry {
    /// The preset's name.
    pub name: String,
    /// The parameters as derived by Sage; `params.choices` are the choices.
    pub params: Params,
    /// The estimator's results.
    pub security: SecurityEstimates,
}

/// The whole of `secure.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecurePresetFile {
    /// The format version (1).
    pub format: u32,
    /// What wrote the file.
    pub generator: String,
    /// The command that wrote it.
    pub command: String,
    /// The SageMath version.
    pub sage_version: String,
    /// The lattice estimator.
    pub estimator: EstimatorInfo,
    /// The date of the run.
    pub generated: String,
    /// The presets.
    pub presets: Vec<SecurePresetEntry>,
}

/// The toy choices for $n$ provers (`docs/m0-spec.md`, "Toy presets"): $\ell = 6$, $\alpha = 4$,
/// $\kappa = 2$, $b = 4$, $\omega = 2$ in the $\ell_1$ ball, $n_A = n_B = n_D = 1$,
/// $m_\rho = 2$, the tight $\tau$ and the smallest admissible $q$.
///
/// The brief leaves $\lambda$ and $t$ open. They do not affect any derived value here; $t$ is
/// the largest threshold below $n/2$, and $\lambda = 115$ as for the secure presets, so the
/// report states plainly that the toys miss the knowledge-error target (a warning).
pub fn toy_choices(n: u64) -> Choices {
    Choices {
        lambda: 115,
        n,
        t: n.saturating_sub(1) / 2,
        ell: 6,
        alpha: 4,
        kappa: 2,
        b: 4,
        omega: Omega::Fixed(2),
        challenge_set: ChallengeSet::L1Ball,
        tau_rule: TauRule::Tight,
        n_a: 1,
        n_b: 1,
        n_d: 1,
        m_rho: 2,
        q: QChoice::Smallest,
    }
}

/// The three toy presets `toy-n1`, `toy-n2`, `toy-n3` (`docs/m0-spec.md`).
pub fn toy_presets() -> Vec<Preset> {
    (1..=3)
        .map(|n| Preset {
            name: format!("toy-n{n}"),
            kind: PresetKind::Toy,
            choices: toy_choices(n),
            security: None,
        })
        .collect()
}

/// The numbers of variables $\ell$ of the development presets `dev-l12`, `dev-l16`, `dev-l20`.
pub const DEV_ELLS: [u32; 3] = [12, 16, 20];

/// The development choices with $\ell$ variables (`docs/m1-spec.md`, WP11): the toy choices
/// of [`toy_choices`]`(1)` with this $\ell$, that is $n = 1$, $\alpha = 4$, $\kappa = 2$, $b = 4$,
/// $\omega = 2$ (fixed) in the $\ell\_1$ ball, $n\_A = n\_B = n\_D = 1$, $m\_\rho = 2$, the tight
/// $\tau$ and the smallest admissible $q$.
///
/// **Not secure.** Like the toys, these choices miss the knowledge-error target by far
/// ($\lvert C \rvert$ is tiny with $d = 16$ and $\omega = 2$) and $q$ has no $\mathsf{MSIS}$
/// margin; they exist to run the protocol at realistic sizes of the hypercube
/// ($\ell'' = 18, 21, 23$ for $\ell = 12, 16, 20$).
pub fn dev_choices(ell: u32) -> Choices {
    Choices {
        ell,
        ..toy_choices(1)
    }
}

/// The development presets `dev-l12`, `dev-l16`, `dev-l20`: [`dev_choices`] at each $\ell$ of
/// [`DEV_ELLS`], of kind [`PresetKind::Dev`]. **Not secure** (see [`dev_choices`]).
pub fn dev_presets() -> Vec<Preset> {
    DEV_ELLS
        .iter()
        .map(|&ell| Preset {
            name: format!("dev-l{ell}"),
            kind: PresetKind::Dev,
            choices: dev_choices(ell),
            security: None,
        })
        .collect()
}

/// Parses the embedded [`SECURE_JSON`].
///
/// # Errors
///
/// [`Error::Json`] if the file does not parse (the tests parse it).
pub fn secure_preset_file() -> Result<SecurePresetFile> {
    Ok(serde_json::from_str(SECURE_JSON)?)
}

/// The secure presets of `secure.json`.
///
/// # Errors
///
/// As [`secure_preset_file`].
pub fn secure_presets() -> Result<Vec<Preset>> {
    Ok(secure_preset_file()?
        .presets
        .into_iter()
        .map(|e| Preset {
            name: e.name,
            kind: PresetKind::Secure,
            choices: e.params.choices,
            security: Some(e.security),
        })
        .collect())
}

/// The toy presets, then the development presets, then the secure presets.
///
/// # Errors
///
/// As [`secure_preset_file`].
pub fn presets() -> Result<Vec<Preset>> {
    let mut all = toy_presets();
    all.extend(dev_presets());
    all.extend(secure_presets()?);
    Ok(all)
}

/// The preset called `name`.
///
/// # Errors
///
/// [`Error::UnknownPreset`] if there is none, or as [`secure_preset_file`].
pub fn preset(name: &str) -> Result<Preset> {
    presets()?
        .into_iter()
        .find(|p| p.name == name)
        .ok_or_else(|| Error::UnknownPreset(name.to_owned()))
}
