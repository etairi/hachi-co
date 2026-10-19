//! The Sage oracle's fixtures (`ref/fixtures/toy-n{1,2,3,4,8}.json`, format
//! `hachi-co-ref-fixture/v1`; `ref/README.md`, "Fixture format"), for the tests of every block
//! crate (feature `fixtures`).
//!
//! [`Fixture`] loads a fixture, derives and cross-checks its parameters, builds its [`Context`]
//! with the recorded $\mathbf A, \mathbf B, \mathbf D$, and decodes values by JSON pointer
//! (RFC 6901: `/ring_to_linear/attempts/0/c`; a key containing `~` or `/` is escaped as `~0`
//! resp. `~1`, see [`escape_key`]). The decoders check the shape and that every residue is in
//! $\[0, q)$. [`sha256_residues`] is the fixtures' digest convention.

use crate::{Context, Error, Layout, Matrix, PublicParams, Result};
use hachi_co_math::{Ext, Poly};
use hachi_co_params::{Params, TauRule, derive, presets::toy_choices};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// The SHA-256 digest, as lowercase hex, of `residues` written as little-endian `u64` in order:
/// the convention of the fixtures' digests of large objects (`ref/README.md`, "Encodings"; the
/// oracle's `fixtures.digest`). The caller passes canonical residues in row-major order of the
/// stated shape (the oracle reduces modulo $q$ before hashing); for $\mathbf M$ that is
/// $n' \times \mu \times d$ (fixture key `ring_to_linear/M_sha256`).
pub fn sha256_residues<I: IntoIterator<Item = u64>>(residues: I) -> String {
    let mut h = Sha256::new();
    for r in residues {
        h.update(r.to_le_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// `key` escaped for use as one segment of a JSON pointer (RFC 6901): `~` becomes `~0` and `/`
/// becomes `~1`. Some keys of `relations` and `negative_tests` contain `~`.
pub fn escape_key(key: &str) -> String {
    key.replace('~', "~0").replace('/', "~1")
}

/// One fixture of the Sage oracle (`ref/README.md`, "Fixture format").
#[derive(Clone, Debug, PartialEq)]
pub struct Fixture {
    name: String,
    root: Value,
    n: usize,
    q: u64,
    d: usize,
    k: usize,
}

fn err(pointer: &str, what: impl core::fmt::Display) -> Error {
    Error::Fixture(format!("{pointer}: {what}"))
}

fn as_array<'a>(v: &'a Value, pointer: &str) -> Result<&'a [Value]> {
    v.as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| err(pointer, "expected a list"))
}

fn as_u64(v: &Value, pointer: &str) -> Result<u64> {
    v.as_u64().ok_or_else(|| {
        err(
            pointer,
            format!("expected a non-negative integer, found {v}"),
        )
    })
}

fn as_i64(v: &Value, pointer: &str) -> Result<i64> {
    v.as_i64()
        .ok_or_else(|| err(pointer, format!("expected an integer, found {v}")))
}

/// Pushes every integer of a nested list onto `out`, row-major.
fn flatten_ints(v: &Value, pointer: &str, out: &mut Vec<i64>) -> Result<()> {
    match v {
        Value::Array(items) => items.iter().try_for_each(|x| flatten_ints(x, pointer, out)),
        _ => {
            out.push(as_i64(v, pointer)?);
            Ok(())
        }
    }
}

impl Fixture {
    /// The names of the five fixtures, for $n = 1, 2, 3, 4, 8$: one per toy preset, in the
    /// order of [`TOY_NS`](hachi_co_params::presets::TOY_NS) (`docs/m0-spec.md`, "Toy presets",
    /// with $n = 4, 8$ added).
    pub const TOY: [&'static str; 5] = ["toy-n1", "toy-n2", "toy-n3", "toy-n4", "toy-n8"];

    /// The format the loader accepts (fixture key `format`).
    pub const FORMAT: &'static str = "hachi-co-ref-fixture/v1";

    /// The path of fixture `name`: `ref/fixtures/<name>.json` under the workspace root (two
    /// levels above this crate's manifest).
    pub fn path(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("ref")
            .join("fixtures")
            .join(format!("{name}.json"))
    }

    /// Loads fixture `name` (for example `"toy-n2"`) from [`Fixture::path`].
    ///
    /// # Errors
    /// [`Error::Fixture`] for a name with characters other than ASCII letters, digits, `-` and
    /// `_`, a missing or unreadable file, invalid JSON, or as [`Fixture::from_json`].
    pub fn load(name: &str) -> Result<Self> {
        if name.is_empty()
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(Error::Fixture(format!("invalid fixture name {name:?}")));
        }
        Self::from_path(&Self::path(name))
    }

    /// Loads the fixture at `path`; its name is the file stem.
    ///
    /// # Errors
    /// [`Error::Fixture`] for an unreadable file or invalid JSON, or as [`Fixture::from_json`].
    pub fn from_path(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Fixture(format!("cannot read {}: {e}", path.display())))?;
        let root: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Fixture(format!("{}: invalid JSON: {e}", path.display())))?;
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_owned();
        Self::from_json(name, root)
    }

    /// Wraps a parsed fixture (for instance a modified copy of [`Fixture::json`] in a negative
    /// test). Checks the `format` key and reads $n$, $q$, $d$ and $k$ from `params`.
    ///
    /// # Errors
    /// [`Error::Fixture`] for another format or missing parameters.
    pub fn from_json(name: impl Into<String>, root: Value) -> Result<Self> {
        let format = root.pointer("/format").and_then(Value::as_str);
        if format != Some(Self::FORMAT) {
            return Err(Error::Fixture(format!(
                "format {format:?}, expected {:?}",
                Self::FORMAT
            )));
        }
        let read = |key: &str| -> Result<u64> {
            let pointer = format!("/params/{key}");
            let v = root
                .pointer(&pointer)
                .ok_or_else(|| err(&pointer, "missing"))?;
            as_u64(v, &pointer)
        };
        let size = |key: &str| -> Result<usize> {
            usize::try_from(read(key)?).map_err(|_| Error::Fixture(format!("{key} overflows")))
        };
        let (n, q, d, k) = (size("n")?, read("q")?, size("d")?, size("k")?);
        Ok(Self {
            name: name.into(),
            root,
            n,
            q,
            d,
            k,
        })
    }

    /// The fixture's name, e.g. `toy-n2`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The number of provers $n$ (`params/n`).
    pub fn n(&self) -> usize {
        self.n
    }

    /// The modulus $q$ (`params/q`).
    pub fn q(&self) -> u64 {
        self.q
    }

    /// The ring degree $d$ (`params/d`).
    pub fn d(&self) -> usize {
        self.d
    }

    /// The extension degree $k$ (`params/k`).
    pub fn k(&self) -> usize {
        self.k
    }

    /// The whole parsed fixture.
    pub fn json(&self) -> &Value {
        &self.root
    }

    /// The value at the JSON pointer `pointer`.
    ///
    /// # Errors
    /// [`Error::Fixture`] if there is none.
    pub fn get(&self, pointer: &str) -> Result<&Value> {
        self.root
            .pointer(pointer)
            .ok_or_else(|| err(pointer, "missing"))
    }

    /// The parameters: [`derive`](fn@derive)`(&`[`toy_choices`]`(n))` for the fixture's $n$, cross-checked
    /// field by field against the fixture's `params` section. Every key of that section is
    /// compared: the choices, $q$, every derived integer (including $\ell\_U$, $\ell\_L$ and
    /// $\lceil \log \nu \rceil$ through [`Layout`]), $|\mathcal C|$ (`l1ball_size`), and the
    /// floats $\ln M$, $1/M$ and $1 - M^{-n}$ to a relative $10^{-12}$; a key the check does not
    /// know is an error too.
    ///
    /// # Errors
    /// [`Error::Params`] if the cascade fails, [`Error::Fixture`] listing every disagreement.
    pub fn params(&self) -> Result<Params> {
        let n = self.n as u64;
        let (params, _report) =
            derive(&toy_choices(n)).map_err(|e| Error::Params(e.to_string()))?;
        let layout = Layout::new(&params)?;
        let section = self
            .get("/params")?
            .as_object()
            .ok_or_else(|| err("/params", "expected an object"))?;
        let c = &params.choices;
        let mut bad = Vec::new();
        let mut known = Vec::new();
        let mut int = |key: &'static str, got: u128| {
            known.push(key);
            match section.get(key).and_then(Value::as_u64) {
                Some(want) if u128::from(want) == got => {}
                other => bad.push(format!("{key}: fixture {other:?}, Rust {got}")),
            }
        };
        let ints: [(&'static str, u128); 38] = [
            ("n", c.n.into()),
            ("ell", c.ell.into()),
            ("alpha", c.alpha.into()),
            ("kappa", c.kappa.into()),
            ("b", c.b.into()),
            ("omega", params.omega.into()),
            ("n_A", c.n_a.into()),
            ("n_B", c.n_b.into()),
            ("n_D", c.n_d.into()),
            ("m_rho", c.m_rho.into()),
            ("q", params.q.into()),
            ("d", params.d.into()),
            ("k", params.k.into()),
            ("delta", params.delta.into()),
            ("logn", params.log_n.into()),
            ("ell_prime", params.ell_prime.into()),
            ("r", params.r.into()),
            ("m", params.m.into()),
            ("beta1", params.beta1),
            ("N", params.big_n),
            ("gamma", params.gamma),
            ("beta", params.beta),
            ("tau", params.tau.into()),
            ("beta_bar", params.beta_bar),
            ("omega_bar", params.omega_bar.into()),
            ("gamma_bar", params.gamma_bar.into()),
            ("mu", params.mu),
            ("n_prime", params.n_prime.into()),
            ("ell_U", layout.ell_u as u128),
            ("ell_L", layout.ell_l as u128),
            ("ell_pp", params.ell_dprime.into()),
            ("d0", params.d0.into()),
            ("d1", params.d1.into()),
            ("nu", params.nu),
            ("log_nu", layout.log_nu as u128),
            ("width_A", params.width_a),
            ("width_B", params.width_b),
            ("width_D", params.width_d),
        ];
        for (key, got) in ints {
            int(key, got);
        }
        let mut text = |key: &'static str, got: String| {
            known.push(key);
            match section.get(key).and_then(Value::as_str) {
                Some(want) if want == got => {}
                other => bad.push(format!("{key}: fixture {other:?}, Rust {got:?}")),
            }
        };
        text("name", format!("toy-n{n}"));
        let rule = match c.tau_rule {
            TauRule::Tight => "tight",
            TauRule::Draft => "draft",
            TauRule::Ceil => "ceil",
        };
        text("tau_rule", rule.to_owned());
        let ln_m = params.log2_rejection_m * core::f64::consts::LN_2;
        let acceptance = (-ln_m).exp();
        let abort = 1.0 - acceptance.powi(i32::try_from(n).unwrap_or(i32::MAX));
        let mut float = |key: &'static str, got: f64| {
            known.push(key);
            match section.get(key).and_then(Value::as_f64) {
                Some(want) if (want - got).abs() <= 1e-12 * want.abs().max(got.abs()) => {}
                other => bad.push(format!("{key}: fixture {other:?}, Rust {got}")),
            }
        };
        float("ln_M", ln_m);
        float("acceptance_per_prover", acceptance);
        float("abort_probability", abort);
        known.push("l1ball_size");
        let ball = hachi_co_sampling::l1ball::count(layout.d, params.omega).to_string();
        let recorded = section.get("l1ball_size").map(Value::to_string);
        if recorded.as_deref() != Some(ball.as_str()) {
            bad.push(format!("l1ball_size: fixture {recorded:?}, Rust {ball}"));
        }
        for key in section.keys() {
            if !known.contains(&key.as_str()) {
                bad.push(format!("{key}: not cross-checked"));
            }
        }
        let preset = self.root.pointer("/preset").and_then(Value::as_str);
        if preset != Some(format!("toy-n{n}").as_str()) {
            bad.push(format!("preset: fixture {preset:?}, Rust toy-n{n}"));
        }
        if !bad.is_empty() {
            return Err(Error::Fixture(format!(
                "{}: parameters differ: {}",
                self.name,
                bad.join("; ")
            )));
        }
        Ok(params)
    }

    /// The context of the fixture: [`Fixture::params`] and the recorded matrices `setup/A`,
    /// `setup/B`, `setup/D` ([`Context::new`]).
    ///
    /// # Errors
    /// As [`Fixture::params`], the decoders and [`Context::new`] (for example for $K \ne k$).
    pub fn context<const K: usize>(&self) -> Result<Context<K>> {
        let params = self.params()?;
        let pp = PublicParams {
            a: self.matrix("/setup/A")?,
            b: self.matrix("/setup/B")?,
            d: self.matrix("/setup/D")?,
        };
        Context::new(params, pp)
    }

    /// A non-negative integer.
    ///
    /// # Errors
    /// [`Error::Fixture`] if the value is missing or not a non-negative integer.
    pub fn u64(&self, pointer: &str) -> Result<u64> {
        as_u64(self.get(pointer)?, pointer)
    }

    /// A non-negative integer as `usize` (a count or an index).
    ///
    /// # Errors
    /// As [`Fixture::u64`], or if it does not fit in `usize`.
    pub fn usize(&self, pointer: &str) -> Result<usize> {
        usize::try_from(self.u64(pointer)?).map_err(|_| err(pointer, "does not fit in usize"))
    }

    /// A boolean.
    ///
    /// # Errors
    /// [`Error::Fixture`] if the value is missing or not a boolean.
    pub fn bool(&self, pointer: &str) -> Result<bool> {
        self.get(pointer)?
            .as_bool()
            .ok_or_else(|| err(pointer, "expected a boolean"))
    }

    /// A string (for example a digest).
    ///
    /// # Errors
    /// [`Error::Fixture`] if the value is missing or not a string.
    pub fn str(&self, pointer: &str) -> Result<&str> {
        self.get(pointer)?
            .as_str()
            .ok_or_else(|| err(pointer, "expected a string"))
    }

    /// Every integer of a (possibly nested) list of signed integers, flattened in row-major
    /// order: a short ring element ($d$ values), a mask $\mathbf r\_i$ recorded as
    /// $\delta 2^m \times d$ ($N$ values, element $i$'s coefficient $j$ at $id + j$), a list of
    /// party indices.
    ///
    /// # Errors
    /// [`Error::Fixture`] if the value is missing or contains a non-integer.
    pub fn ints(&self, pointer: &str) -> Result<Vec<i64>> {
        let mut out = Vec::new();
        flatten_ints(self.get(pointer)?, pointer, &mut out)?;
        Ok(out)
    }

    /// A list of lists of signed integers (for example the challenges $\mathbf c$, $2^r \times d$).
    ///
    /// # Errors
    /// [`Error::Fixture`] if the value is not a list of lists of integers.
    pub fn int_rows(&self, pointer: &str) -> Result<Vec<Vec<i64>>> {
        as_array(self.get(pointer)?, pointer)?
            .iter()
            .map(|row| {
                as_array(row, pointer)?
                    .iter()
                    .map(|x| as_i64(x, pointer))
                    .collect()
            })
            .collect()
    }

    /// One residue: an integer in $\[0, q)$.
    fn residue(&self, v: &Value, pointer: &str) -> Result<u64> {
        let x = as_u64(v, pointer)?;
        if x >= self.q {
            return Err(err(
                pointer,
                format!("residue {x} is not below q = {}", self.q),
            ));
        }
        Ok(x)
    }

    /// A flat list of residues, with `len` entries if given.
    fn residue_list(&self, v: &Value, pointer: &str, len: Option<usize>) -> Result<Vec<u64>> {
        let items = as_array(v, pointer)?;
        if let Some(len) = len
            && items.len() != len
        {
            return Err(err(
                pointer,
                format!("expected {len} residues, found {}", items.len()),
            ));
        }
        items.iter().map(|x| self.residue(x, pointer)).collect()
    }

    /// Every residue of a (possibly nested) list, flattened in row-major order, each checked to be
    /// in $\[0, q)$.
    ///
    /// # Errors
    /// [`Error::Fixture`] if the value is missing, contains a non-integer, or a residue $\ge q$.
    pub fn residues(&self, pointer: &str) -> Result<Vec<u64>> {
        let mut ints = Vec::new();
        flatten_ints(self.get(pointer)?, pointer, &mut ints)?;
        ints.into_iter()
            .map(|x| {
                u64::try_from(x)
                    .ok()
                    .filter(|x| *x < self.q)
                    .ok_or_else(|| err(pointer, format!("{x} is not a residue modulo {}", self.q)))
            })
            .collect()
    }

    /// A list of lists of residues (for example the $d \times d$ matrix of $E$).
    ///
    /// # Errors
    /// [`Error::Fixture`] if the value is not a list of lists of residues.
    pub fn residue_rows(&self, pointer: &str) -> Result<Vec<Vec<u64>>> {
        as_array(self.get(pointer)?, pointer)?
            .iter()
            .map(|row| self.residue_list(row, pointer, None))
            .collect()
    }

    /// A ring element of $\mathcal R\_q$: a list of $d$ residues.
    ///
    /// # Errors
    /// [`Error::Fixture`] for another length or a residue $\ge q$.
    pub fn poly(&self, pointer: &str) -> Result<Poly> {
        Ok(Poly(self.residue_list(
            self.get(pointer)?,
            pointer,
            Some(self.d),
        )?))
    }

    /// A list of ring elements of $\mathcal R\_q$ ($d$ residues each).
    ///
    /// # Errors
    /// As [`Fixture::poly`], for any entry.
    pub fn polys(&self, pointer: &str) -> Result<Vec<Poly>> {
        self.polys_of_len(pointer, self.d)
    }

    /// A list of polynomials of `len` residues each (for example the quotient shares, elements of
    /// $\mathbb Z\_q\[X\]/(X^{2d} + 1)$ with $2d$ residues).
    ///
    /// # Errors
    /// [`Error::Fixture`] for another length or a residue $\ge q$.
    pub fn polys_of_len(&self, pointer: &str, len: usize) -> Result<Vec<Poly>> {
        as_array(self.get(pointer)?, pointer)?
            .iter()
            .map(|p| Ok(Poly(self.residue_list(p, pointer, Some(len))?)))
            .collect()
    }

    /// A matrix over $\mathcal R\_q$, row-major (for example `setup/A`).
    ///
    /// # Errors
    /// As [`Fixture::poly`], for any entry.
    pub fn matrix(&self, pointer: &str) -> Result<Matrix> {
        as_array(self.get(pointer)?, pointer)?
            .iter()
            .map(|row| {
                as_array(row, pointer)?
                    .iter()
                    .map(|p| Ok(Poly(self.residue_list(p, pointer, Some(self.d))?)))
                    .collect()
            })
            .collect()
    }

    /// An element of $\mathbb F\_{q^K}$: a list of $K$ residues, the coordinates in Hachi's basis.
    fn ext_of<const K: usize>(&self, v: &Value, pointer: &str) -> Result<Ext<K>> {
        if K != self.k {
            return Err(err(
                pointer,
                format!("decoding with K = {K}, but the fixture has k = {}", self.k),
            ));
        }
        let coords = self.residue_list(v, pointer, Some(K))?;
        Ok(Ext(core::array::from_fn(|c| coords[c])))
    }

    /// An element of $\mathbb F\_{q^K}$ ($K$ residues, `ref/README.md`, "Encodings").
    ///
    /// # Errors
    /// [`Error::Fixture`] for $K \ne k$, another length or a residue $\ge q$.
    pub fn ext<const K: usize>(&self, pointer: &str) -> Result<Ext<K>> {
        self.ext_of(self.get(pointer)?, pointer)
    }

    /// A list of elements of $\mathbb F\_{q^K}$.
    ///
    /// # Errors
    /// As [`Fixture::ext`], for any entry.
    pub fn exts<const K: usize>(&self, pointer: &str) -> Result<Vec<Ext<K>>> {
        as_array(self.get(pointer)?, pointer)?
            .iter()
            .map(|x| self.ext_of(x, pointer))
            .collect()
    }

    /// A list of lists of elements of $\mathbb F\_{q^K}$ (for example a blinder's $s$,
    /// $\ell'' \times D$, or the reshared shares of one party, $n \times 2^\ell$).
    ///
    /// # Errors
    /// As [`Fixture::ext`], for any entry.
    pub fn ext_rows<const K: usize>(&self, pointer: &str) -> Result<Vec<Vec<Ext<K>>>> {
        as_array(self.get(pointer)?, pointer)?
            .iter()
            .map(|row| {
                as_array(row, pointer)?
                    .iter()
                    .map(|x| self.ext_of(x, pointer))
                    .collect()
            })
            .collect()
    }
}
