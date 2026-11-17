//! The parameters of a commitment, chosen by the rule of the reference implementation.
//!
//! **The rule** (`greyhound.c`, `init_polcomctx`, of `lattice-dogs/labrador` at 8b6626b). Here
//! $\ell$ is the number of ring elements of the polynomial (`len`), $L$ the bit length of $q$
//! (the reference's `LOGQ`, which is the bit length of each of its moduli $2^{L} - c$), $N = 64$,
//! $\tau_1 = 32$ and $\tau_2 = 8$ the weights of a challenge and $T = 14$ its operator-norm
//! bound. For the expansion factor $f = 2, \dots, 8$ of the polynomial's decomposition, with
//! base $2^b$, $b = \lfloor (L + \lfloor f/2 \rfloor) / f \rfloor$, the inner rank
//! $\kappa = 1, \dots, 32$ is the first for which
//!
//! - $m = \mathrm{round}(\sqrt{\ell(\kappa + 1)/2})$, $n = \lceil \ell / m \rceil$,
//! - $v_z = 2^{2b}/12 \cdot n (\tau_1 + 4\tau_2)$, $b_u = \mathrm{round}(\log_2(12 v_z)/4)$,
//!   $f_u = \mathrm{round}(L / b_u)$,
//! - $\beta^2 = N \cdot \lfloor \lfloor (2^{2b_u}/12 + v_z/2^{2b_u}) m f \rfloor +
//!   (2^{2b_u}(f_u - 1) + 2^{2(L - (f_u - 1) b_u)})/12 \cdot (\kappa + 1) n \rfloor$ (the
//!   reference truncates to integers where shown),
//!
//! make Module-SIS of rank $\kappa$ hard for the bound $6 T s 2^{b_u} \beta$ (here with $\beta$
//! raised by [`NORM_MARGIN`], a deviation), with $s$ the slack
//! of the proof system that proves the reduced relation. Then the outer rank $\kappa_1 =
//! 1, \dots, 32$ is the first for which Module-SIS of rank $\kappa_1$ is hard for $2 s \beta$, and
//! the first $f$ for which both exist is taken. The polynomial is cut into $n$ columns of $m$
//! ring elements (the last zero-padded), each decomposed into $f$ digits; the inner commitments
//! and the evaluations $w_i$ into $f_u$ digits of base $2^{b_u}$, and the amortized opening $z$
//! into two.
//!
//! **Large moduli.** The rule takes the first $f$ with ranks of at most 32, which for 56-bit
//! moduli is $f = 2$ (digits of 28 bits) at the largest ranks: for $q = 2^{56} - 27$ and degrees
//! $2^{21}$ and $2^{23}$, $\kappa = 32$ and a predicted witness norm of about $4 \cdot 10^8$.
//! [`Params::with_expansion`] runs the rule for a given $f$ instead.
//!
//! **Hardness** is the reference's `sis_secure`: rank $k$ is hard for the Euclidean bound $B$
//! when $\log_2 B < \min(\log_2 q,\ 2\sqrt{\log_2 q \log_2 \delta \cdot N k})$, the length of the
//! shortest vector that lattice reduction of root Hermite factor $\delta$ finds in the best
//! sublattice. The reference fixes $\delta = 1.00444$. Here $\log_2 q$ is the exact logarithm
//! (the reference uses $L$, which differs from it by less than $2^{-26}$ for its moduli).
//!
//! **Slack.** The reference sets $s = 2$ (`SLACK`, marked `FIXME`), while its LaBRADOR accepts
//! projections up to $\sqrt{256}\beta$, which by BS23 Lemma 4.1 only guarantees $\sqrt{256/30}
//! \beta \approx 2.92\beta$. jali's LaBRADOR accepts projections up to $\sqrt{337}\beta$ and its
//! extractor guarantees $\sqrt{337/30}\beta \approx 3.35\beta$, so [`Security::labrador`], the
//! default, takes $s = \sqrt{337/30}$. [`Security::REFERENCE`] takes $s = 2$, which reproduces
//! the reference's parameters.
use crate::error::{Error, Result};

/// The degree of the ring $\mathcal R_q = \mathbb Z_q\[X\]/(X^{64} + 1)$.
pub const N: usize = 64;
/// The number of coefficients $\pm 1$ of a challenge.
pub const TAU1: usize = 32;
/// The number of coefficients $\pm 2$ of a challenge.
pub const TAU2: usize = 8;
/// The bound on the operator norm of a challenge.
pub const T: f64 = 14.0;
/// The largest commitment rank the rule tries.
pub const MAX_RANK: usize = 32;
/// The root Hermite factor of the reference's hardness rule.
pub const REFERENCE_DELTA: f64 = 1.00444;
/// Moduli must be below $2^{62}$ (centred values, digits and sums stay within `i64`).
pub const MAX_MODULUS_BITS: u32 = 62;
/// The factor on the predicted norm in the choice of the ranks (the reference has none): the
/// verifier checks the ranks against the actual norm of the witness, which differs from the
/// prediction by up to 0.1% in the runs measured, so a choice at the edge of a rank fails
/// honest proofs. Without it, $q = 93266870910952069$, degree $2^{21}$ and $f = 4$ gave
/// $\kappa = 14$ and an honest proof that the verifier rejected (hco-dev1, 17 November 2026).
pub const NORM_MARGIN: f64 = 1.01;

/// The security rule of the parameter choice and of the verifier's checks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Security {
    /// The factor by which the norm of an extracted witness of the reduced relation may exceed
    /// the relation's bound.
    pub slack: f64,
    /// The root Hermite factor that lattice reduction is assumed not to reach.
    pub delta: f64,
}

impl Security {
    /// The reference's rule: slack 2 and $\delta = 1.00444$.
    pub const REFERENCE: Self = Self {
        slack: 2.0,
        delta: REFERENCE_DELTA,
    };

    /// The slack of jali's LaBRADOR, $\sqrt{337/30}$, with $\delta = 1.00444$ (the default).
    pub fn labrador() -> Self {
        Self {
            slack: (337.0f64 / 30.0).sqrt(),
            delta: REFERENCE_DELTA,
        }
    }

    /// The root Hermite factor $\delta_0(b)$ of BKZ with block size $b$ (Chen's formula, as jali
    /// computes it), to set [`Security::delta`] for a block size: for example $b = 439$, about
    /// 128 bits in the core-SVP model $2^{0.292 b}$, which jali's default policy requires of its
    /// own commitments.
    pub fn delta_for_block_size(b: u32) -> f64 {
        use std::f64::consts::{E, PI};
        let b = f64::from(b);
        (b / (2.0 * PI * E) * (PI * b).powf(1.0 / b)).powf(1.0 / (2.0 * (b - 1.0)))
    }
}

impl Default for Security {
    fn default() -> Self {
        Self::labrador()
    }
}

/// The bit length of $q$, the reference's `LOGQ` for its moduli $2^{\mathrm{LOGQ}} - c$.
pub fn bit_length(q: u64) -> u32 {
    64 - q.leading_zeros()
}

/// The reference's `sis_secure`: Module-SIS of rank `rank` over $\mathcal R_q$ is hard for the
/// Euclidean bound `norm` (module documentation).
pub fn sis_secure(q: u64, rank: usize, norm: f64, delta: f64) -> bool {
    let logq = (q as f64).log2();
    let maxlog = 2.0 * (logq * delta.log2() * N as f64).sqrt() * (rank as f64).sqrt();
    norm.log2() < maxlog.min(logq)
}

/// The parameters of a commitment to a polynomial of `len` ring elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Params {
    /// The modulus, a prime $q \equiv 5 \pmod 8$ below $2^{62}$.
    pub q: u64,
    /// The number of ring elements of the polynomial: its degree is below $64 \ell$.
    pub len: usize,
    /// Ring elements per column ($m$).
    pub m: usize,
    /// Columns ($n$, the reference's `n`; the paper's $r$).
    pub n: usize,
    /// Digits of the polynomial ($f$, the paper's $\delta_0$).
    pub f: usize,
    /// $\log_2$ of their base ($b$).
    pub b: u32,
    /// Digits of the inner commitments and of the evaluations $w_i$ ($f_u$, the paper's $\delta$).
    pub fu: usize,
    /// $\log_2$ of their base ($b_u$), also that of the amortized opening $z = z_0 + 2^{b_u} z_1$.
    pub bu: u32,
    /// The rank of the inner commitments ($\kappa$, the paper's $n$).
    pub kappa: usize,
    /// The rank of the outer commitments ($\kappa_1$, the paper's $n_1$).
    pub kappa1: usize,
}

/// The reference's prediction of the squared witness norm, with its integer truncations.
#[allow(clippy::too_many_arguments)]
fn predicted_norm_sq(
    logq: u32,
    m: usize,
    n: usize,
    f: usize,
    b: u32,
    fu: usize,
    bu: u32,
    kappa: usize,
) -> u64 {
    let varz = 2f64.powi(2 * b as i32) / 12.0 * n as f64 * (TAU1 + 4 * TAU2) as f64;
    let e2bu = 2f64.powi(2 * bu as i32);
    let mut normsq = ((e2bu / 12.0 + varz / e2bu) * m as f64 * f as f64) as u64;
    let top = 2f64.powi(2 * (logq as i32 - ((fu - 1) * bu as usize) as i32));
    normsq = (normsq as f64 + (e2bu * (fu - 1) as f64 + top) / 12.0 * (kappa + 1) as f64 * n as f64)
        as u64;
    normsq * N as u64
}

impl Params {
    /// The parameters of the reference's rule (module documentation) for a polynomial of `len`
    /// ring elements over $\mathcal R_q$.
    ///
    /// # Errors
    /// [`Error::Parameter`] for $q$ outside $(2^{16}, 2^{62})$, $q \not\equiv 5 \pmod 8$ (primality
    /// is checked by [`crate::PublicParams::new`]), `len` zero, or if no $f \le 8$ gives ranks of
    /// at most 32.
    pub fn derive(len: usize, q: u64, security: Security) -> Result<Self> {
        for f in 2..=8usize {
            if let Some(params) = Self::for_expansion(len, q, security, f)? {
                return Ok(params);
            }
        }
        Err(Error::Parameter(
            "no expansion factor up to 8 gives commitment ranks up to 32",
        ))
    }

    /// The reference's rule for the given expansion factor $f$ only, instead of the first $f$
    /// that works: at large moduli the first is often $f = 2$ with the largest rank tried (for
    /// $q = 2^{56} - 27$ and degree $2^{21}$, $\kappa = 32$).
    ///
    /// # Errors
    /// As [`Params::derive`], and [`Error::Parameter`] if no ranks up to 32 exist for this $f$.
    pub fn with_expansion(len: usize, q: u64, security: Security, f: usize) -> Result<Self> {
        if !(1..=16).contains(&f) {
            return Err(Error::Parameter("the expansion factor must be in [1, 16]"));
        }
        Self::for_expansion(len, q, security, f)?.ok_or(Error::Parameter(
            "no commitment ranks up to 32 for this expansion factor",
        ))
    }

    /// One iteration of the reference's loop over $f$.
    fn for_expansion(len: usize, q: u64, security: Security, f: usize) -> Result<Option<Self>> {
        if q <= 1 << 16 || bit_length(q) > MAX_MODULUS_BITS || q % 8 != 5 {
            return Err(Error::Parameter(
                "the modulus must be 5 mod 8, above 2^16 and below 2^62",
            ));
        }
        if len == 0 || len > 1 << 40 {
            return Err(Error::Parameter("the length must be in [1, 2^40]"));
        }
        if !(security.slack.is_finite() && security.slack >= 1.0 && security.delta > 1.0) {
            return Err(Error::Parameter(
                "the slack must be at least 1 and delta above 1",
            ));
        }
        let logq = bit_length(q);
        let tau = (TAU1 + 4 * TAU2) as f64;
        let b = (logq + f as u32 / 2) / f as u32;
        let mut chosen = None;
        for kappa in 1..=MAX_RANK {
            let m = ((len * (kappa + 1)) as f64 / 2.0).sqrt().round() as usize;
            let m = m.max(1);
            let n = (len as f64 / m as f64).ceil() as usize;
            let varz = 2f64.powi(2 * b as i32) / 12.0 * n as f64 * tau;
            let bu = ((0.25 * (12.0 * varz).log2()).round() as u32).max(1);
            let fu = ((f64::from(logq) / f64::from(bu)).round() as usize).max(1);
            if (fu - 1) * bu as usize > logq as usize {
                continue;
            }
            let normsq = predicted_norm_sq(logq, m, n, f, b, fu, bu, kappa);
            let norm = (normsq as f64).sqrt() * NORM_MARGIN;
            if sis_secure(
                q,
                kappa,
                6.0 * T * security.slack * 2f64.powi(bu as i32) * norm,
                security.delta,
            ) {
                chosen = Some((kappa, m, n, bu, fu, norm));
                break;
            }
        }
        let Some((kappa, m, n, bu, fu, norm)) = chosen else {
            return Ok(None);
        };
        let Some(kappa1) =
            (1..=MAX_RANK).find(|&k| sis_secure(q, k, 2.0 * security.slack * norm, security.delta))
        else {
            return Ok(None);
        };
        let params = Self {
            q,
            len,
            m,
            n,
            f,
            b,
            fu,
            bu,
            kappa,
            kappa1,
        };
        params.validate()?;
        Ok(Some(params))
    }

    /// The shape conditions on parameters, whoever chose them: $q$ as in [`Params::derive`],
    /// $mn \ge \ell$, bases and digit counts that the integers hold, ranks of 1 to 64, and at
    /// most $2^{32}$ ring elements in any vector.
    ///
    /// # Errors
    /// [`Error::Parameter`] naming the condition.
    pub fn validate(&self) -> Result<()> {
        let bad = |what| Err(Error::Parameter(what));
        if self.q <= 1 << 16 || bit_length(self.q) > MAX_MODULUS_BITS || self.q % 8 != 5 {
            return bad("modulus");
        }
        if self.len == 0 || self.m == 0 || self.n == 0 {
            return bad("lengths");
        }
        if self.m.checked_mul(self.n).is_none_or(|x| x < self.len) {
            return bad("m n must cover the polynomial");
        }
        if !(1..=16).contains(&self.f) || !(1..=MAX_MODULUS_BITS).contains(&self.b) {
            return bad("decomposition of the polynomial");
        }
        if !(1..=16).contains(&self.fu) || !(1..=MAX_MODULUS_BITS).contains(&self.bu) {
            return bad("decomposition of the commitments");
        }
        // The low digits must stay within the bits of i64 (digits::Digits).
        if (self.f - 1) * self.b as usize > 62 || (self.fu - 1) * self.bu as usize > 62 {
            return bad("too many digits for their base");
        }
        if !(1..=64).contains(&self.kappa) || !(1..=64).contains(&self.kappa1) {
            return bad("ranks");
        }
        let lengths = [
            self.m.checked_mul(self.f),
            self.kappa
                .checked_mul(self.fu)
                .and_then(|x| x.checked_mul(self.n)),
            self.fu.checked_mul(self.n),
        ];
        if lengths.iter().any(|x| x.is_none_or(|x| x > 1 << 32)) {
            return bad("vector lengths");
        }
        Ok(())
    }

    /// The reference's prediction of the squared norm of the witness (the square of the
    /// "Predicted witness norm" it prints).
    pub fn predicted_norm_sq(&self) -> u64 {
        predicted_norm_sq(
            bit_length(self.q),
            self.m,
            self.n,
            self.f,
            self.b,
            self.fu,
            self.bu,
            self.kappa,
        )
    }

    /// The length $mf$ of the decomposed columns and of the amortized opening.
    pub fn z_len(&self) -> usize {
        self.m * self.f
    }

    /// The length $\kappa f_u n$ of the decomposed inner commitments $\hat t$.
    pub fn t_len(&self) -> usize {
        self.kappa * self.fu * self.n
    }

    /// The length $f_u n$ of the decomposed evaluations $\hat w$.
    pub fn w_len(&self) -> usize {
        self.fu * self.n
    }

    /// The lengths of the four witness vectors $z_0, z_1, \hat t, \hat w$ of the reduced relation.
    pub fn witness_lengths(&self) -> [usize; 4] {
        [self.z_len(), self.z_len(), self.t_len(), self.w_len()]
    }

    /// The bits of one residue in the encodings, $\lceil \log_2 q \rceil$.
    pub fn residue_bits(&self) -> u32 {
        bit_length(self.q - 1)
    }

    /// The verifier's checks of the reference (`polcom_reduce`): Module-SIS of rank $\kappa$
    /// hard for $6 T s 2^{b_u} \beta$ and of rank $\kappa_1$ hard for $2 s \beta$, for the bound
    /// $\beta^2$ = `norm_sq` that the proof states.
    ///
    /// # Errors
    /// [`Error::InvalidProof`] naming the commitment that is not binding for that bound.
    pub fn check_norm(&self, norm_sq: u128, security: Security) -> Result<()> {
        let norm = (norm_sq as f64).sqrt();
        if !sis_secure(
            self.q,
            self.kappa,
            6.0 * T * security.slack * 2f64.powi(self.bu as i32) * norm,
            security.delta,
        ) {
            return Err(Error::InvalidProof(
                "inner commitments not binding for the norm",
            ));
        }
        if !sis_secure(
            self.q,
            self.kappa1,
            2.0 * security.slack * norm,
            security.delta,
        ) {
            return Err(Error::InvalidProof(
                "outer commitments not binding for the norm",
            ));
        }
        Ok(())
    }

    /// The size the reference prints as "Proof size" of the evaluation proof: $u_1$ and $u_2$,
    /// $2 \kappa_1 N L$ bits, in bytes.
    pub fn reference_proof_bytes(&self) -> usize {
        2 * self.kappa1 * N * bit_length(self.q) as usize / 8
    }

    /// The canonical encoding: each field as `LE64`, in the order of the struct.
    pub fn to_bytes(&self) -> Vec<u8> {
        let fields = [
            self.q,
            self.len as u64,
            self.m as u64,
            self.n as u64,
            self.f as u64,
            u64::from(self.b),
            self.fu as u64,
            u64::from(self.bu),
            self.kappa as u64,
            self.kappa1 as u64,
        ];
        fields.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// The parameters from [`Params::to_bytes`], validated.
    ///
    /// # Errors
    /// [`Error::Encoding`] for a wrong length or a field out of range; the errors of
    /// [`Params::validate`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 80 {
            return Err(Error::Encoding);
        }
        let field = |i: usize| u64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().expect("8"));
        let small = |i: usize| usize::try_from(field(i)).map_err(|_| Error::Encoding);
        let bits = |i: usize| u32::try_from(field(i)).map_err(|_| Error::Encoding);
        let params = Self {
            q: field(0),
            len: small(1)?,
            m: small(2)?,
            n: small(3)?,
            f: small(4)?,
            b: bits(5)?,
            fu: small(6)?,
            bu: bits(7)?,
            kappa: small(8)?,
            kappa1: small(9)?,
        };
        params.validate()?;
        Ok(params)
    }
}

#[cfg(test)]
mod tests;
