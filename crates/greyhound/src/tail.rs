//! A tail for Hachi's final witnesses: commitments to tables of small digits, and one
//! LaBRADOR proof of several multilinear evaluation claims over $\mathbb F_{q^K}$.
//!
//! **Claims.** A table of $2^v$ digits $d_x$, $|d_x| \le$ `bound`, and a point $\mathbf a \in
//! \mathbb F_{q^K}^v$, with value $\sum_x d_x \widetilde{eq}(x, \mathbf a)$, the first coordinate
//! of $\mathbf a$ the most significant bit of $x$ (the convention of Hachi's `table_eval`).
//!
//! **Commitment** (Greyhound's, [`mod@crate::commit`], with $f = 1$: the digits are already
//! short).
//! Ring element $e$ of the table is $\sum_{c < 64} d_{64e + c} X^c$, column $i$ holds ring elements
//! $im, \dots, im + m - 1$ for $m = 2^{\mu}$ and $n = 2^{v - 6 - \mu}$ columns, the inner
//! commitments are $t_i = A \mathbf s_i$, in $f_u$ digits of base $2^{b_u}$ ($\hat t$, row $\rho$ of
//! column $i$ at $(\rho n + i) f_u$), and the commitment is $u_1 = B \hat t$, packed into words.
//!
//! **Evaluation** (Greyhound's front end with the powers of $x$ replaced by the tensor of $eq$
//! factors). Index $x = 64(im + j) + c$ splits the point into column, row and coefficient
//! parts, $\widetilde{eq}(x, \mathbf a) = e^{col}_i e^{row}_j e^{coef}_c$. With the coordinates
//! $e^{row}_{j,k}$ of $e^{row}_j$ in the basis of $\mathbb F_{q^K}$ (Hachi's Eq. (7)):
//!
//! - $w_{i,k} = \sum_j e^{row}_{j,k} \mathbf s_{i,j} \in \mathcal R_q$, in digits $\hat w$ (block
//!   $k$ at $k n f_u$, column $i$ at $i f_u$ within it), and $u_2 = D \hat w$;
//! - $z = \sum_i c_i \mathbf s_i$ for challenges from the transcript, in two digits $z = z_0 +
//!   2^{b_u} z_1$;
//! - the value: $\sum_{i,k,c} (w_{i,k})_c \, e^{col}_i \beta_k e^{coef}_c$ for the basis
//!   $\beta_k$, so coordinate $l$ of the value is $\sum_{i,k} \mathrm{ct}(\sigma_{-1}(g_{i,k,l})
//!   w_{i,k})$ with $g_{i,k,l} = \sum_c (e^{col}_i \beta_k e^{coef}_c)_l X^c$.
//!
//! The relation of one claim: $B \hat t = u_1$ ($\kappa_1$ rows), $D \hat w = u_2$ ($\kappa_1$
//! rows), $\langle e^{row}_{\cdot,k}, z_0 + 2^{b_u} z_1 \rangle = \sum_i c_i w_{i,k}$ ($K$ rows),
//! $A (z_0 + 2^{b_u} z_1) = \sum_i c_i t_i$ ($\kappa$ rows), and $K$ constant-term rows for the
//! value's coordinates. All claims of a proof share one statement and one LaBRADOR proof by
//! jali, whose bound $\beta^2$ is the squared norm of all witnesses together; each claim's
//! commitment ranks are chosen for that joint norm (the extractor's bound on any one claim).
//!
//! **Encodings** are words (`u64`). A commitment is the residues of $u_1$, $\lceil \log_2 q
//! \rceil$ bits each, packed from the least significant bit. A proof is the packed $u_2$ of
//! each claim, then $\beta^2$ (two words), one word with the length in bytes of jali's proof in
//! its low 48 bits and the attempt number $a$ (below) in its high 16 bits, and jali's proof in
//! words, zero-padded to a fixed length ([`TailParams::proof_words`]). Decoders check every
//! length, residue and padding bit, refuse $a \ge 64$, and never panic.
//!
//! **Restarts.** The prover absorbs an attempt number $a < 64$ ([`TAIL_ATTEMPTS`]) after the
//! claims and before it squeezes the amortization challenges $c_i$, and restarts with $a + 1$
//! when the witness norm fails the binding checks (the SIS conditions of every claim's ranks
//! $\kappa, \kappa_1$ at the stated norm $\beta$, which the verifier checks too) or when jali's
//! proof exceeds its reserve ([`CAP_FACTOR`] times jali's planner's prediction plus
//! [`CAP_BYTES`]). Both depend on the tables and the transcript. Tables of independent digits
//! keep $\|z\|$ concentrated, but tables whose columns are equal (all $-8$, or $-8, 7$
//! alternating) give $z = (\sum_i c_i) \mathbf s$, whose norm is heavy-tailed. jali chooses each
//! level's parameters from the witness's norm and garbage, and writes the last witness in codes
//! of its values, so its proof length varies too. In the samples of 17 November 2026
//! (hco-dev1, lambda 100 and 115, q of 48.4, 53.3 and 56.4 bits, single tables of $2^{24}$ and
//! $2^{28}$ digits and batches of $2^{27}$ or $2^{28}$, $2^{20}$ and $2^{18}$, five digit
//! distributions, 20 contexts each), the squared norm stayed within 0.5% of its mean for
//! independent digits and spread from 0.38 to 1.33 times the joint one for equal columns. None
//! of 600 proofs on independent digits restarted, and 8 of 380 on equal columns did, at most
//! twice. jali's proofs took at most 1.038 times the prediction, against the reserve's 1.08
//! plus 2 KiB, and no attempt exceeded it. The cost of restarts: a prover may choose among 64
//! sets of amortization challenges, so the soundness error of the amortization step grows by
//! at most a factor 64 (6 bits). [`TailParams::prove`] fails with [`Error::Parameter`] when all
//! 64 attempts do.
//!
//! **Security** (not knowledge soundness, which is out of scope here): Greyhound's rule of
//! [`crate::params`] with jali's slack $\sqrt{337/30}$ and $\delta = \delta_0(b)$ for the BKZ
//! block size $b = \lceil \lambda / 0.292 \rceil$ (core-SVP), and jali's levels under the policy
//! of block size $\max(b, 346)$ (jali's least). Not zero knowledge, variable time.
use crate::challenge::Challenge;
use crate::commit::{digit_bound, to_polys};
use crate::digits::Digits;
use crate::error::{Error, Result};
use crate::key;
use crate::params::{N, NORM_MARGIN, Security, T, TAU1, TAU2, bit_length, sis_secure};
use crate::relation::{Constraint, Layout, Relation, Term, flatten, norm_sq_of, scaled};
use hachi_co_math::{Ext, Fqk, Poly, Prepared, Ring, rns::NADDS};
use hachi_co_sampling::ShakePrg;
use hachi_co_transcript::Transcript;
use rand::Rng;
use rayon::prelude::*;
use std::sync::Arc;

/// The domain of the tail's transcript, matrices and seeds.
pub const TAIL_DOMAIN: &[u8] = b"hachi-co-greyhound/v1/tail";
/// The largest extension degree $K$ (the width of $D$).
pub const MAX_K: usize = 4;
/// The degree $K$ the parameters are chosen for (Hachi-Co's).
pub const PARAM_K: usize = 4;
/// The core-SVP cost exponent per block: $\lambda$ bits for block size $\lambda / 0.292$.
pub const CORE_SVP: f64 = 0.292;

/// The parameters of one claim's commitment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClaimShape {
    /// Variables: the table has $2^v$ digits.
    pub vars: usize,
    /// $\log_2 m$.
    pub mbits: usize,
    /// Ring elements per column.
    pub m: usize,
    /// Columns.
    pub n: usize,
    /// Digits of $t_i$ and $w_{i,k}$, and $\log_2$ of their base (also that of $z = z_0 + 2^{b_u}
    /// z_1$).
    pub fu: usize,
    /// $\log_2$ of the base.
    pub bu: u32,
    /// Inner and outer ranks.
    pub kappa: usize,
    /// The outer rank.
    pub kappa1: usize,
}

impl ClaimShape {
    /// The lengths of $z_0, z_1, \hat t, \hat w$ in ring elements, for degree $K$.
    pub fn lengths(&self, k: usize) -> [usize; 4] {
        [
            self.m,
            self.m,
            self.kappa * self.fu * self.n,
            k * self.fu * self.n,
        ]
    }
}

/// The public matrices of one table size.
struct TailKey {
    vars: usize,
    a: Vec<Vec<Poly>>,
    b: Vec<Vec<Poly>>,
    d: Vec<Vec<Poly>>,
    a_prepared: Prepared,
}

/// The parameters of a tail for claims on tables of $2^{v_i}$ digits, in order.
pub struct TailParams {
    q: u64,
    bound: u64,
    lambda: u32,
    seed: [u8; 32],
    ring: Ring,
    security: Security,
    policy: jali::labrador::Policy,
    shapes: Vec<ClaimShape>,
    keys: Vec<Arc<TailKey>>,
    /// The joint squared norm the ranks were chosen for.
    joint_norm_sq: f64,
    /// jali's prediction of its proof's bytes for that norm.
    predicted_bytes: usize,
    /// The words reserved for jali's proof.
    cap_words: usize,
    /// Tests only: attempts below this one restart (the restart path).
    #[cfg(test)]
    forced_restarts: u8,
}

/// The attempts of [`TailParams::prove`] (see Restarts in the module documentation).
pub const TAIL_ATTEMPTS: u8 = 64;
/// The position of the attempt number in a proof's length word.
const ATTEMPT_SHIFT: u32 = 48;

/// The outcome of one attempt of [`TailParams::prove`].
enum Attempt {
    /// The proof's words.
    Done(Vec<u64>),
    /// A restart, with the error to return if every attempt restarts.
    Restart(&'static str),
}

/// The reserve for jali's proof over its planner's prediction: a factor and bytes. jali's
/// proof depends on the witness (it chooses each level's parameters from the witness's norm
/// and garbage, and writes the last witness in codes of its values), so the reserve is
/// empirical: see Restarts in the module documentation for the samples it was set from. An
/// attempt whose proof exceeds it restarts.
pub const CAP_FACTOR: f64 = 1.08;
/// The bytes added to the reserve.
pub const CAP_BYTES: usize = 2048;

impl core::fmt::Debug for TailParams {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TailParams")
            .field("q", &self.q)
            .field("bound", &self.bound)
            .field("lambda", &self.lambda)
            .field("shapes", &self.shapes)
            .field("cap_words", &self.cap_words)
            .finish_non_exhaustive()
    }
}

/// What the prover keeps of a commitment: the digits $\hat t$.
#[derive(Clone, Debug)]
pub struct TailOpening {
    vars: usize,
    t_hat: Vec<i64>,
}

/// A claim: the commitment of a table of $2^v$ digits, a point in $\mathbb F_{q^K}^v$ and the
/// value of the table's multilinear extension there.
#[derive(Clone, Copy, Debug)]
pub struct Claim<'a, const K: usize> {
    /// $v$.
    pub vars: usize,
    /// The commitment's words.
    pub commitment: &'a [u64],
    /// The point, its first coordinate the most significant bit of the index.
    pub point: &'a [Ext<K>],
    /// $\sum_x d_x \widetilde{eq}(x, \mathbf a)$.
    pub value: Ext<K>,
}

/// A claim with what the prover knows: the digits and the opening.
#[derive(Clone, Copy, Debug)]
pub struct ProverClaim<'a, const K: usize> {
    /// The claim.
    pub claim: Claim<'a, K>,
    /// The table's $2^v$ digits.
    pub digits: &'a [i8],
    /// The opening of [`TailParams::commit`].
    pub opening: &'a TailOpening,
}

/// The BKZ block size of $\lambda$ bits in the core-SVP model.
pub fn block_size(lambda: u32) -> u32 {
    (f64::from(lambda) / CORE_SVP).ceil() as u32
}

/// Residues of `bits` bits packed into words from the least significant bit.
fn pack_words(values: impl IntoIterator<Item = u64>, bits: u32) -> Vec<u64> {
    let (mut out, mut acc, mut held) = (Vec::new(), 0u128, 0u32);
    for v in values {
        acc |= u128::from(v) << held;
        held += bits;
        while held >= 64 {
            out.push(acc as u64);
            acc >>= 64;
            held -= 64;
        }
    }
    if held > 0 {
        out.push(acc as u64);
    }
    out
}

/// Words of `count` residues of `bits` bits.
fn words_for(count: usize, bits: u32) -> usize {
    (count * bits as usize).div_ceil(64)
}

/// The inverse of [`pack_words`]: exactly [`words_for`] words, residues below $q$, zero padding.
fn unpack_words(words: &[u64], bits: u32, count: usize, q: u64) -> Result<Vec<u64>> {
    if words.len() != words_for(count, bits) {
        return Err(Error::Encoding);
    }
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let (mut out, mut acc, mut held) = (Vec::with_capacity(count), 0u128, 0u32);
    let mut input = words.iter();
    for _ in 0..count {
        if held < bits {
            acc |= u128::from(*input.next().ok_or(Error::Encoding)?) << held;
            held += 64;
        }
        let v = (acc as u64) & mask;
        acc >>= bits;
        held -= bits;
        if v >= q {
            return Err(Error::Encoding);
        }
        out.push(v);
    }
    if acc != 0 || input.next().is_some() {
        return Err(Error::Encoding);
    }
    Ok(out)
}

/// $\widetilde{eq}(x, \mathbf a)$ for every $x < 2^{|\mathbf a|}$, $a_0$ the most significant bit
/// (Hachi's `eq_table`).
fn eq_table<const K: usize>(field: &Fqk<K>, a: &[Ext<K>]) -> Vec<Ext<K>> {
    let mut out = vec![field.one()];
    for aj in a {
        let mut next = vec![field.zero(); 2 * out.len()];
        next.par_chunks_mut(2)
            .zip(out.par_iter())
            .for_each(|(children, v)| {
                children[1] = field.mul(v, aj);
                children[0] = field.sub(v, &children[1]);
            });
        out = next;
    }
    out
}

/// The predicted squared norm of one claim's witness for the inner rank $\kappa$: the rule of
/// [`crate::params`] with $f = 1$ and digits of second moment at most $\mathrm{bound}^2$ (any
/// table, not only uniform ones).
fn claim_norm_sq(
    logq: u32,
    bound: u64,
    m: usize,
    n: usize,
    fu: usize,
    bu: u32,
    kappa: usize,
) -> f64 {
    let var_d = (bound * bound) as f64;
    let varz = var_d * n as f64 * (TAU1 + 4 * TAU2) as f64;
    let e2bu = 2f64.powi(2 * bu as i32);
    let top = 2f64.powi(2 * (logq as i32 - ((fu - 1) * bu as usize) as i32));
    let z = (e2bu / 12.0 + varz / e2bu) * m as f64;
    let digits = (e2bu * (fu - 1) as f64 + top) / 12.0 * ((kappa + PARAM_K) * n) as f64;
    N as f64 * (z + digits)
}

/// The base and digits of $z$, $t_i$ and $w_{i,k}$ for $n$ columns.
fn bases(logq: u32, bound: u64, n: usize) -> (u32, usize) {
    let var_d = (bound * bound) as f64;
    let varz = var_d * n as f64 * (TAU1 + 4 * TAU2) as f64;
    let bu = ((0.25 * (12.0 * varz).log2()).round() as u32).clamp(1, 61);
    let mut fu = ((f64::from(logq) / f64::from(bu)).round() as usize).max(1);
    while (fu - 1) * bu as usize > logq as usize {
        fu -= 1;
    }
    (bu, fu)
}

/// The shape of a claim on $2^v$ digits whose ranks hold for the joint squared norm `joint`
/// (its own predicted norm when `joint` is smaller): the $\mu$ of the fewest witness ring
/// elements $2m + (\kappa + K) f_u n$, each with its least ranks.
fn shape_for(
    q: u64,
    vars: usize,
    bound: u64,
    security: Security,
    joint: f64,
) -> Result<(ClaimShape, f64)> {
    let logq = bit_length(q);
    let total = vars - 6;
    let mut best: Option<(usize, ClaimShape, f64)> = None;
    for mbits in 0..=total {
        let (m, n) = (1usize << mbits, 1usize << (total - mbits));
        let (bu, fu) = bases(logq, bound, n);
        let mut chosen = None;
        for kappa in 1..=64 {
            let own = claim_norm_sq(logq, bound, m, n, fu, bu, kappa);
            let norm = own.max(joint).sqrt() * NORM_MARGIN;
            if sis_secure(
                q,
                kappa,
                6.0 * T * security.slack * 2f64.powi(bu as i32) * norm,
                security.delta,
            ) {
                chosen = Some((kappa, own, norm));
                break;
            }
        }
        let Some((kappa, own, norm)) = chosen else {
            continue;
        };
        let Some(kappa1) =
            (1..=64).find(|&k| sis_secure(q, k, 2.0 * security.slack * norm, security.delta))
        else {
            continue;
        };
        let cost = 2 * m + (kappa + PARAM_K) * fu * n;
        let shape = ClaimShape {
            vars,
            mbits,
            m,
            n,
            fu,
            bu,
            kappa,
            kappa1,
        };
        if best.as_ref().is_none_or(|(c, _, _)| cost < *c) {
            best = Some((cost, shape, own));
        }
    }
    best.map(|(_, s, own)| (s, own))
        .ok_or(Error::Parameter("no tail ranks up to 64 for this table"))
}

impl TailParams {
    /// Parameters for claims on tables of $2^{v_i}$ digits (`vars`, in this order) with
    /// $|d| \le$ `bound` over $\mathcal R_q$ ($q$ a prime $\equiv 5 \pmod 8$ below $2^{62}$),
    /// at $\lambda$ bits (module documentation), and the matrices from `seed`.
    ///
    /// # Errors
    /// [`Error::Parameter`] for an invalid input or no ranks; arithmetic errors.
    pub fn new(q: u64, vars: &[usize], bound: u64, lambda: u32, seed: [u8; 32]) -> Result<Self> {
        if vars.is_empty() || vars.iter().any(|v| !(6..=40).contains(v)) {
            return Err(Error::Parameter("tables of 2^6 to 2^40 digits"));
        }
        if !(1..=1 << 20).contains(&bound) {
            return Err(Error::Parameter("the digit bound must be in [1, 2^20]"));
        }
        if !(64..=256).contains(&lambda) {
            return Err(Error::Parameter("lambda must be in [64, 256]"));
        }
        if q <= 1 << 16 || bit_length(q) > crate::params::MAX_MODULUS_BITS || q % 8 != 5 {
            return Err(Error::Parameter(
                "the modulus must be 5 mod 8, above 2^16 and below 2^62",
            ));
        }
        let ring = Ring::new(q, N)?;
        let block = block_size(lambda);
        let security = Security {
            slack: (337.0f64 / 30.0).sqrt(),
            delta: Security::delta_for_block_size(block),
        };
        let policy = jali::labrador::Policy {
            min_block_size: block.max(346),
        };
        // Ranks for the joint norm: start from each claim alone, then twice with the sum.
        let mut joint = 0f64;
        let mut shapes = Vec::new();
        for _ in 0..3 {
            let results: Vec<(ClaimShape, f64)> = vars
                .iter()
                .map(|v| shape_for(q, *v, bound, security, joint))
                .collect::<Result<_>>()?;
            joint = results.iter().map(|(_, own)| own).sum();
            shapes = results.into_iter().map(|(s, _)| s).collect();
        }
        let mut keys: Vec<Arc<TailKey>> = Vec::new();
        for s in &shapes {
            if keys.iter().any(|k| k.vars == s.vars) {
                continue;
            }
            let domain = |name: &str| format!("hachi-co-greyhound/v1/tail/{name}/{}", s.vars);
            let a = key::expand(&seed, domain("A").as_bytes(), s.kappa, s.m, q);
            let b = key::expand(
                &seed,
                domain("B").as_bytes(),
                s.kappa1,
                s.kappa * s.fu * s.n,
                q,
            );
            let d = key::expand(
                &seed,
                domain("D").as_bytes(),
                s.kappa1,
                MAX_K * s.fu * s.n,
                q,
            );
            let a_prepared = ring.prepare_short(&a, bound)?;
            keys.push(Arc::new(TailKey {
                vars: s.vars,
                a,
                b,
                d,
                a_prepared,
            }));
        }
        let mut params = Self {
            q,
            bound,
            lambda,
            seed,
            ring,
            security,
            policy,
            shapes,
            keys,
            joint_norm_sq: joint,
            predicted_bytes: 0,
            cap_words: 0,
            #[cfg(test)]
            forced_restarts: 0,
        };
        // jali's planner predicts its proof from the statement's shape and bound alone.
        let lengths = params.lengths(PARAM_K);
        let layout = Layout::choose(&lengths);
        let jring = jali::labrador::Rq::new(q, N)?;
        params.predicted_bytes = jali::labrador::predicted_proof_bytes(
            &jring,
            layout.r,
            layout.n,
            joint as u128,
            &params.policy,
        )?;
        params.cap_words =
            ((params.predicted_bytes as f64 * CAP_FACTOR) as usize + CAP_BYTES).div_ceil(8);
        Ok(params)
    }

    /// jali's prediction of its proof's bytes (before [`CAP_FACTOR`] and [`CAP_BYTES`]).
    pub fn predicted_labrador_bytes(&self) -> usize {
        self.predicted_bytes
    }

    /// The joint squared norm the commitment ranks hold for.
    pub fn joint_norm_sq(&self) -> f64 {
        self.joint_norm_sq
    }

    /// The shapes of the claims, in order.
    pub fn shapes(&self) -> &[ClaimShape] {
        &self.shapes
    }

    /// The modulus.
    pub fn modulus(&self) -> u64 {
        self.q
    }

    /// The security rule of the commitments.
    pub fn security(&self) -> Security {
        self.security
    }

    /// jali's policy for its levels.
    pub fn policy(&self) -> jali::labrador::Policy {
        self.policy
    }

    fn shape(&self, vars: usize) -> Result<&ClaimShape> {
        self.shapes
            .iter()
            .find(|s| s.vars == vars)
            .ok_or(Error::Parameter("no claim on tables of this size"))
    }

    fn key(&self, vars: usize) -> Result<&TailKey> {
        self.keys
            .iter()
            .find(|k| k.vars == vars)
            .map(|k| k.as_ref())
            .ok_or(Error::Parameter("no claim on tables of this size"))
    }

    fn residue_bits(&self) -> u32 {
        bit_length(self.q - 1)
    }

    /// The lengths of all witness vectors, four per claim, for degree $K$.
    fn lengths(&self, k: usize) -> Vec<usize> {
        self.shapes.iter().flat_map(|s| s.lengths(k)).collect()
    }

    /// The words of a commitment to a table of $2^v$ digits.
    pub fn commitment_words(&self, vars: usize) -> usize {
        self.shape(vars)
            .map(|s| words_for(s.kappa1 * N, self.residue_bits()))
            .unwrap_or(0)
    }

    /// The words of a proof for the claims of [`TailParams::new`]: the packed $u_2$ of each
    /// claim, $\beta^2$, the length and attempt word, and a fixed number of words for jali's
    /// proof: its planner's prediction for the statement's shape and the joint norm the ranks
    /// were chosen for, times [`CAP_FACTOR`], plus [`CAP_BYTES`]. [`TailParams::prove`]
    /// restarts an attempt whose proof exceeds it.
    pub fn proof_words(&self) -> usize {
        let bits = self.residue_bits();
        self.shapes
            .iter()
            .map(|s| words_for(s.kappa1 * N, bits))
            .sum::<usize>()
            + 3
            + self.cap_words
    }

    /// Commit to a table of $2^v$ digits of absolute value at most the bound.
    ///
    /// # Errors
    /// [`Error::Shape`] for a table of the wrong size or a digit out of bounds;
    /// [`Error::Parameter`] for a size without a claim.
    pub fn commit(&self, vars: usize, digits: &[i8]) -> Result<(Vec<u64>, TailOpening)> {
        let s = *self.shape(vars)?;
        let key = self.key(vars)?;
        if digits.len() != 1usize << vars {
            return Err(Error::Shape("a table of 2^vars digits"));
        }
        if digits
            .iter()
            .any(|d| u64::from(d.unsigned_abs()) > self.bound)
        {
            return Err(Error::Shape("a digit beyond the bound"));
        }
        let ring = &self.ring;
        let zq = ring.zq();
        let (m, n) = (s.m, s.n);
        let pieces: Vec<core::ops::Range<usize>> = (0..m)
            .step_by(NADDS)
            .map(|a| a..(a + NADDS).min(m))
            .collect();
        let groups: Vec<core::ops::Range<usize>> =
            (0..n).step_by(8).map(|a| a..(a + 8).min(n)).collect();
        let t: Vec<Vec<Poly>> = groups
            .par_iter()
            .map(|cols| {
                ring.mat_vec_prepared_pieces_many(
                    &key.a_prepared,
                    cols.len(),
                    pieces.iter().cloned(),
                    |v, piece, out| {
                        let col = cols.start + v;
                        for (slot, e) in out.as_chunks_mut::<N>().0.iter_mut().zip(piece) {
                            let at = (col * m + e) * N;
                            for (o, d) in slot.iter_mut().zip(&digits[at..at + N]) {
                                *o = i64::from(*d);
                            }
                        }
                        Ok(())
                    },
                )
            })
            .collect::<core::result::Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect();
        let tdigits = Digits::new(s.bu, s.fu);
        let mut t_hat = vec![0i64; s.kappa * s.fu * n * N];
        t_hat
            .par_chunks_mut(n * s.fu * N)
            .enumerate()
            .for_each(|(rho, row)| {
                for (i, ti) in t.iter().enumerate() {
                    for (c, x) in ti[rho].0.iter().enumerate() {
                        let centred = zq.to_centered(*x);
                        for j in 0..s.fu {
                            row[(i * s.fu + j) * N + c] = tdigits.digit(centred, j);
                        }
                    }
                }
            });
        let u1 = ring.mat_vec_short(
            &key.b,
            &to_polys(ring, &t_hat),
            digit_bound(self.q, s.bu, s.fu),
        )?;
        let words = pack_words(
            u1.iter().flat_map(|p| p.0.iter().copied()),
            self.residue_bits(),
        );
        Ok((words, TailOpening { vars, t_hat }))
    }

    /// The transcript's start: the context, the parameters and the seed.
    fn transcript(&self, context: &[u8; 32]) -> Transcript {
        let mut t = Transcript::new(TAIL_DOMAIN);
        t.append_message(b"context", context);
        let mut p = Vec::new();
        for x in [
            self.q,
            self.bound,
            u64::from(self.lambda),
            self.shapes.len() as u64,
        ] {
            p.extend_from_slice(&x.to_le_bytes());
        }
        for s in &self.shapes {
            for x in [s.vars, s.mbits, s.fu, s.bu as usize, s.kappa, s.kappa1] {
                p.extend_from_slice(&(x as u64).to_le_bytes());
            }
        }
        t.append_message(b"params", &p);
        t.append_message(b"seed", &self.seed);
        t
    }

    /// Absorb a claim and its $u_2$.
    fn absorb_claim<const K: usize>(t: &mut Transcript, claim: &Claim<'_, K>, u2: &[u64]) {
        t.append_u64(b"vars", claim.vars as u64);
        let words = |w: &[u64]| -> Vec<u8> { w.iter().flat_map(|x| x.to_le_bytes()).collect() };
        t.append_message(b"commitment", &words(claim.commitment));
        let point: Vec<u64> = claim.point.iter().flat_map(|e| e.0).collect();
        t.append_message(b"point", &words(&point));
        t.append_message(b"value", &words(&claim.value.0));
        t.append_message(b"u2", &words(u2));
    }

    /// The checks of a claim's shape against the parameters (both sides).
    fn check_claim<const K: usize>(
        &self,
        field: &Fqk<K>,
        index: usize,
        claim: &Claim<'_, K>,
    ) -> Result<ClaimShape> {
        if K == 0 || K > MAX_K || field.modulus() != self.q {
            return Err(Error::Parameter(
                "the field must be F_{q^K} for this q and K <= 4",
            ));
        }
        let s = *self
            .shapes
            .get(index)
            .ok_or(Error::Shape("more claims than the parameters"))?;
        if claim.vars != s.vars || claim.point.len() != s.vars {
            return Err(Error::Shape("a claim of another size than the parameters'"));
        }
        if claim.commitment.len() != words_for(s.kappa1 * N, self.residue_bits()) {
            return Err(Error::Shape("a commitment of the wrong length"));
        }
        let q = self.q;
        if claim
            .point
            .iter()
            .chain([&claim.value])
            .any(|e| e.0.iter().any(|c| *c >= q))
        {
            return Err(Error::Shape("field elements must have coordinates below q"));
        }
        Ok(s)
    }

    /// The eq factors of a claim's point: columns, rows and coefficients.
    fn factors<const K: usize>(
        field: &Fqk<K>,
        s: &ClaimShape,
        point: &[Ext<K>],
    ) -> [Vec<Ext<K>>; 3] {
        let nbits = s.vars - 6 - s.mbits;
        [
            eq_table(field, &point[..nbits]),
            eq_table(field, &point[nbits..nbits + s.mbits]),
            eq_table(field, &point[nbits + s.mbits..]),
        ]
    }

    /// $w_{i,k}$ for every column, block $k$ at $kn$: $\sum_j e^{row}_{j,k} \mathbf s_{i,j}$.
    fn evaluations<const K: usize>(
        &self,
        s: &ClaimShape,
        digits: &[i8],
        rows: &[Ext<K>],
    ) -> Vec<[u64; N]> {
        let zq = self.ring.zq();
        let (m, n) = (s.m, s.n);
        let per: Vec<Vec<[u64; N]>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut acc = vec![[0i128; N]; K];
                for (j, e) in rows.iter().enumerate() {
                    let at = (i * m + j) * N;
                    let ds = &digits[at..at + N];
                    for (k, a) in acc.iter_mut().enumerate() {
                        let w = e.0[k] as i64 as i128;
                        for (x, d) in a.iter_mut().zip(ds) {
                            *x += w * i128::from(*d);
                        }
                    }
                }
                acc.iter()
                    .map(|a| core::array::from_fn(|c| zq.from_i128(a[c])))
                    .collect()
            })
            .collect();
        // Reorder to block k at k n.
        let mut out = vec![[0u64; N]; K * n];
        for (i, ws) in per.into_iter().enumerate() {
            for (k, w) in ws.into_iter().enumerate() {
                out[k * n + i] = w;
            }
        }
        out
    }

    /// $z = \sum_i c_i \mathbf s_i$ over the integers.
    fn amortize(&self, s: &ClaimShape, digits: &[i8], cs: &[Challenge]) -> Vec<i64> {
        let (m, n) = (s.m, s.n);
        const R: usize = 32;
        let chunks: Vec<Vec<i64>> = (0..m.div_ceil(R))
            .into_par_iter()
            .map(|chunk| {
                let j0 = chunk * R;
                let rows = R.min(m - j0);
                let mut acc = vec![0i64; N * R];
                let mut digit_rows = vec![0i64; N * R];
                for (i, c) in cs.iter().enumerate().take(n) {
                    for jj in 0..R {
                        if jj < rows {
                            let at = (i * m + j0 + jj) * N;
                            for (cc, d) in digits[at..at + N].iter().enumerate() {
                                digit_rows[cc * R + jj] = i64::from(*d);
                            }
                        } else {
                            for cc in 0..N {
                                digit_rows[cc * R + jj] = 0;
                            }
                        }
                    }
                    for (pos, v) in c.coeffs.iter().enumerate().filter(|(_, v)| **v != 0) {
                        let v = i64::from(*v);
                        let (low, high) = acc.split_at_mut(pos * R);
                        let (stays, wraps) = digit_rows.split_at((N - pos) * R);
                        for (a, x) in high.iter_mut().zip(stays) {
                            *a += v * *x;
                        }
                        for (a, x) in low.iter_mut().zip(wraps) {
                            *a -= v * *x;
                        }
                    }
                }
                acc
            })
            .collect();
        let mut z = vec![0i64; m * N];
        for (chunk, acc) in chunks.iter().enumerate() {
            let j0 = chunk * R;
            for jj in 0..R.min(m - j0) {
                for c in 0..N {
                    z[(j0 + jj) * N + c] = acc[c * R + jj];
                }
            }
        }
        z
    }

    /// The constraints of claim `index` (vectors $4\,index$ to $4\,index + 3$).
    #[allow(clippy::too_many_arguments)]
    fn constraints<const K: usize>(
        &self,
        field: &Fqk<K>,
        index: usize,
        s: &ClaimShape,
        claim: &Claim<'_, K>,
        u1: &[u64],
        u2: &[u64],
        cs: &[Challenge],
        full: &mut Vec<Constraint>,
        constant_term: &mut Vec<Constraint>,
    ) -> Result<()> {
        let zq = self.ring.zq();
        let key = self.key(s.vars)?;
        let (m, n, fu) = (s.m, s.n, s.fu);
        let [cols, rows, coefs] = Self::factors(field, s, claim.point);
        let base = 4 * index;
        let two_bu = zq.pow(2, u64::from(s.bu));
        // gamma_{i fu + j} = -2^{j bu} c_i.
        let mut gamma = vec![0u64; n * fu * N];
        for (i, c) in cs.iter().enumerate() {
            let mut scale = 1u64;
            for j in 0..fu {
                let at = (i * fu + j) * N;
                for (o, v) in gamma[at..at + N].iter_mut().zip(c.coeffs) {
                    *o = zq.neg(zq.mul(zq.from_i64(i64::from(v)), scale));
                }
                scale = zq.mul(scale, two_bu);
            }
        }
        let wlen = K * fu * n;
        for (r, row) in key.b.iter().enumerate() {
            full.push(Constraint {
                terms: vec![Term {
                    vector: base + 2,
                    offset: 0,
                    phi: flatten(row),
                }],
                b: u1[r * N..(r + 1) * N].to_vec(),
            });
        }
        for (r, row) in key.d.iter().enumerate() {
            full.push(Constraint {
                terms: vec![Term {
                    vector: base + 3,
                    offset: 0,
                    phi: flatten(&row[..wlen]),
                }],
                b: u2[r * N..(r + 1) * N].to_vec(),
            });
        }
        for k in 0..K {
            let mut phi = vec![0u64; m * N];
            for (j, e) in rows.iter().enumerate() {
                phi[j * N] = e.0[k];
            }
            let phi1 = scaled(&self.ring, &phi, two_bu);
            full.push(Constraint {
                terms: vec![
                    Term {
                        vector: base,
                        offset: 0,
                        phi,
                    },
                    Term {
                        vector: base + 1,
                        offset: 0,
                        phi: phi1,
                    },
                    Term {
                        vector: base + 3,
                        offset: k * n * fu,
                        phi: gamma.clone(),
                    },
                ],
                b: vec![0; N],
            });
        }
        for (rho, row) in key.a.iter().enumerate() {
            let a = flatten(row);
            let a1 = scaled(&self.ring, &a, two_bu);
            full.push(Constraint {
                terms: vec![
                    Term {
                        vector: base,
                        offset: 0,
                        phi: a,
                    },
                    Term {
                        vector: base + 1,
                        offset: 0,
                        phi: a1,
                    },
                    Term {
                        vector: base + 2,
                        offset: rho * n * fu,
                        phi: gamma.clone(),
                    },
                ],
                b: vec![0; N],
            });
        }
        // psi_l at (k n + i) fu + j: 2^{j bu} sigma_{-1}(g_{i,k,l}), g_{i,k,l} = sum_c
        // (e^col_i beta_k e^coef_c)_l X^c.
        let mut psi = vec![vec![0u64; wlen * N]; K];
        let products: Vec<Vec<Ext<K>>> = (0..n * K)
            .into_par_iter()
            .map(|ik| {
                let (i, k) = (ik / K, ik % K);
                let mut beta = [0u64; K];
                beta[k] = 1;
                let p = field.mul(&cols[i], &Ext(beta));
                coefs.iter().map(|e| field.mul(&p, e)).collect()
            })
            .collect();
        for (ik, es) in products.iter().enumerate() {
            let (i, k) = (ik / K, ik % K);
            for (l, out) in psi.iter_mut().enumerate() {
                let mut g = [0u64; N];
                for (c, e) in es.iter().enumerate() {
                    g[c] = e.0[l];
                }
                // sigma_{-1}(g): g_0, then -g_{64-t} at t.
                let mut sg = [0u64; N];
                sg[0] = g[0];
                for t in 1..N {
                    sg[t] = zq.neg(g[N - t]);
                }
                let mut scale = 1u64;
                for j in 0..fu {
                    let at = ((k * n + i) * fu + j) * N;
                    for (o, v) in out[at..at + N].iter_mut().zip(&sg) {
                        *o = zq.mul(*v, scale);
                    }
                    scale = zq.mul(scale, two_bu);
                }
            }
        }
        for (l, phi) in psi.into_iter().enumerate() {
            let mut b = vec![0u64; N];
            b[0] = claim.value.0[l];
            constant_term.push(Constraint {
                terms: vec![Term {
                    vector: base + 3,
                    offset: 0,
                    phi,
                }],
                b,
            });
        }
        Ok(())
    }

    /// The binding checks of every claim for the joint norm.
    fn check_norm(&self, norm_sq: u128) -> Result<()> {
        let norm = (norm_sq as f64).sqrt();
        for s in &self.shapes {
            let sec = self.security;
            if !sis_secure(
                self.q,
                s.kappa,
                6.0 * T * sec.slack * 2f64.powi(s.bu as i32) * norm,
                sec.delta,
            ) || !sis_secure(self.q, s.kappa1, 2.0 * sec.slack * norm, sec.delta)
            {
                return Err(Error::InvalidProof(
                    "commitments not binding for the stated norm",
                ));
            }
        }
        Ok(())
    }

    fn labrador_seed(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        ShakePrg::new(&self.seed, b"hachi-co-greyhound/v1/tail/labrador", 0).fill_bytes(&mut out);
        out
    }

    fn challenge_seed_index(claim: usize, column: usize) -> u64 {
        ((claim as u64) << 40) | column as u64
    }

    /// Prove the claims (in the order of [`TailParams::new`]) in one LaBRADOR proof, with up to
    /// [`TAIL_ATTEMPTS`] attempts (see Restarts in the module documentation).
    ///
    /// # Errors
    /// [`Error::Shape`] or [`Error::Parameter`] for claims that do not match the parameters,
    /// [`Error::Labrador`] if jali refuses (among them a wrong value, which no witness
    /// satisfies), and [`Error::Parameter`] if in every attempt the witness norm fails the
    /// binding checks or jali's proof exceeds its reserve in [`TailParams::proof_words`].
    pub fn prove<const K: usize>(
        &self,
        field: &Fqk<K>,
        claims: &[ProverClaim<'_, K>],
        context: &[u8; 32],
    ) -> Result<Vec<u64>> {
        if claims.len() != self.shapes.len() {
            return Err(Error::Shape("one claim per table of the parameters"));
        }
        let bits = self.residue_bits();
        let mut t = self.transcript(context);
        let mut fronts = Vec::with_capacity(claims.len());
        for (index, pc) in claims.iter().enumerate() {
            let s = self.check_claim(field, index, &pc.claim)?;
            if pc.digits.len() != 1usize << s.vars || pc.opening.vars != s.vars {
                return Err(Error::Shape("digits or opening of another table size"));
            }
            if pc.opening.t_hat.len() != s.kappa * s.fu * s.n * N {
                return Err(Error::Shape("an opening of the wrong length"));
            }
            let [_, rows, _] = Self::factors(field, &s, pc.claim.point);
            let w = self.evaluations(&s, pc.digits, &rows);
            let wdigits = Digits::new(s.bu, s.fu);
            let zq = self.ring.zq();
            let mut w_hat = vec![0i64; K * s.fu * s.n * N];
            for (kn, wi) in w.iter().enumerate() {
                for (c, v) in wi.iter().enumerate() {
                    let centred = zq.to_centered(*v);
                    for j in 0..s.fu {
                        w_hat[(kn * s.fu + j) * N + c] = wdigits.digit(centred, j);
                    }
                }
            }
            let key = self.key(s.vars)?;
            let wlen = K * s.fu * s.n;
            let d_rows: Vec<Vec<Poly>> = key.d.iter().map(|r| r[..wlen].to_vec()).collect();
            let u2 = self.ring.mat_vec_short(
                &d_rows,
                &to_polys(&self.ring, &w_hat),
                digit_bound(self.q, s.bu, s.fu),
            )?;
            let u2: Vec<u64> = u2.iter().flat_map(|p| p.0.iter().copied()).collect();
            Self::absorb_claim(&mut t, &pc.claim, &pack_words(u2.iter().copied(), bits));
            fronts.push((s, u2, w_hat));
        }
        let mut reason = "";
        for attempt in 0..TAIL_ATTEMPTS {
            match self.attempt(field, claims, &fronts, &t, attempt, bits)? {
                Attempt::Done(proof) => return Ok(proof),
                Attempt::Restart(why) => reason = why,
            }
        }
        Err(Error::Parameter(reason))
    }

    /// Attempt `attempt` of [`TailParams::prove`] from the transcript `base` after the claims:
    /// its amortization challenges, the witness and its norm, and jali's proof. Restarts when
    /// the norm fails the binding checks or jali's proof exceeds its reserve.
    fn attempt<const K: usize>(
        &self,
        field: &Fqk<K>,
        claims: &[ProverClaim<'_, K>],
        fronts: &[(ClaimShape, Vec<u64>, Vec<i64>)],
        base: &Transcript,
        attempt: u8,
        bits: u32,
    ) -> Result<Attempt> {
        #[cfg(test)]
        {
            if attempt < self.forced_restarts {
                return Ok(Attempt::Restart("a restart forced by a test"));
            }
        }
        let mut t = base.clone();
        t.append_message(b"attempt", &[attempt]);
        let seed = t.challenge_seed(b"challenges");
        let mut vectors: Vec<Vec<i64>> = Vec::with_capacity(4 * claims.len());
        let mut all_cs = Vec::with_capacity(claims.len());
        for (index, (pc, (s, _, w_hat))) in claims.iter().zip(fronts).enumerate() {
            let cs: Vec<Challenge> = (0..s.n)
                .into_par_iter()
                .map(|i| Challenge::sample(&seed, Self::challenge_seed_index(index, i)))
                .collect();
            let z = self.amortize(s, pc.digits, &cs);
            let zd = Digits::new(s.bu, 2);
            let z0: Vec<i64> = z.par_iter().map(|v| zd.digit(*v, 0)).collect();
            let z1: Vec<i64> = z.par_iter().map(|v| zd.digit(*v, 1)).collect();
            vectors.push(z0);
            vectors.push(z1);
            vectors.push(pc.opening.t_hat.clone());
            vectors.push(w_hat.clone());
            all_cs.push(cs);
        }
        let slices: Vec<&[i64]> = vectors.iter().map(Vec::as_slice).collect();
        let norm_sq = norm_sq_of(&slices);
        if self.check_norm(norm_sq).is_err() {
            return Ok(Attempt::Restart(
                "the witness norm fails the binding checks in every attempt",
            ));
        }
        t.append_message(b"norm", &norm_sq.to_le_bytes());
        let mut full = Vec::new();
        let mut constant_term = Vec::new();
        for (index, (pc, (s, u2, _))) in claims.iter().zip(fronts).enumerate() {
            let u1 = unpack_words(pc.claim.commitment, bits, s.kappa1 * N, self.q)?;
            self.constraints(
                field,
                index,
                s,
                &pc.claim,
                &u1,
                u2,
                &all_cs[index],
                &mut full,
                &mut constant_term,
            )?;
        }
        let lengths = self.lengths(K);
        let relation = Relation {
            lengths: lengths.clone(),
            norm_sq,
            full,
            constant_term,
        };
        let ring = jali::labrador::Rq::new(self.q, N)?;
        let layout = Layout::choose(&lengths);
        let statement = relation.into_jali(&ring, &layout)?;
        let witness = layout.jali_witness_of(&ring, &lengths, &slices)?;
        let ctx = labrador_context(&t);
        let proof = jali::labrador::prove(
            &statement,
            &witness,
            &self.labrador_seed(),
            &ctx,
            &self.policy,
        )?;
        let bytes = proof.as_bytes();
        if bytes.len() > self.cap_words * 8 {
            return Ok(Attempt::Restart(
                "the LaBRADOR proof exceeds its fixed length in every attempt",
            ));
        }
        let mut out = Vec::with_capacity(self.proof_words());
        for (_, u2, _) in fronts {
            out.extend(pack_words(u2.iter().copied(), bits));
        }
        out.push(norm_sq as u64);
        out.push((norm_sq >> 64) as u64);
        out.push(bytes.len() as u64 | (u64::from(attempt) << ATTEMPT_SHIFT));
        for chunk in bytes.chunks(8) {
            let mut w = [0u8; 8];
            w[..chunk.len()].copy_from_slice(chunk);
            out.push(u64::from_le_bytes(w));
        }
        out.resize(self.proof_words(), 0);
        Ok(Attempt::Done(out))
    }

    /// Verify a proof of the claims (in the order of [`TailParams::new`]).
    ///
    /// # Errors
    /// [`Error::Encoding`] for a proof that does not decode, [`Error::InvalidProof`] or
    /// [`Error::Labrador`] when it is rejected, [`Error::Shape`] for claims that do not match.
    pub fn verify<const K: usize>(
        &self,
        field: &Fqk<K>,
        claims: &[Claim<'_, K>],
        proof: &[u64],
        context: &[u8; 32],
    ) -> Result<()> {
        if claims.len() != self.shapes.len() {
            return Err(Error::Shape("one claim per table of the parameters"));
        }
        if proof.len() != self.proof_words() {
            return Err(Error::Encoding);
        }
        let bits = self.residue_bits();
        let mut at = 0;
        let mut u2s = Vec::with_capacity(claims.len());
        for (index, claim) in claims.iter().enumerate() {
            let s = self.check_claim(field, index, claim)?;
            let len = words_for(s.kappa1 * N, bits);
            u2s.push(unpack_words(
                &proof[at..at + len],
                bits,
                s.kappa1 * N,
                self.q,
            )?);
            at += len;
        }
        let norm_sq = u128::from(proof[at]) | (u128::from(proof[at + 1]) << 64);
        let attempt = proof[at + 2] >> ATTEMPT_SHIFT;
        if attempt >= u64::from(TAIL_ATTEMPTS) {
            return Err(Error::Encoding);
        }
        let byte_len = usize::try_from(proof[at + 2] & ((1 << ATTEMPT_SHIFT) - 1))
            .map_err(|_| Error::Encoding)?;
        at += 3;
        if byte_len > self.cap_words * 8 || byte_len == 0 {
            return Err(Error::Encoding);
        }
        let words = &proof[at..];
        let used = byte_len.div_ceil(8);
        let mut bytes: Vec<u8> = words[..used].iter().flat_map(|w| w.to_le_bytes()).collect();
        if bytes[byte_len..].iter().any(|b| *b != 0) || words[used..].iter().any(|w| *w != 0) {
            return Err(Error::Encoding);
        }
        bytes.truncate(byte_len);
        let labrador = jali::labrador::Proof::from_bytes(bytes)?;
        self.check_norm(norm_sq)?;
        let mut t = self.transcript(context);
        for (claim, u2) in claims.iter().zip(&u2s) {
            Self::absorb_claim(&mut t, claim, &pack_words(u2.iter().copied(), bits));
        }
        t.append_message(b"attempt", &[attempt as u8]);
        let seed = t.challenge_seed(b"challenges");
        t.append_message(b"norm", &norm_sq.to_le_bytes());
        let mut full = Vec::new();
        let mut constant_term = Vec::new();
        for (index, (claim, u2)) in claims.iter().zip(&u2s).enumerate() {
            let s = self.shapes[index];
            let cs: Vec<Challenge> = (0..s.n)
                .into_par_iter()
                .map(|i| Challenge::sample(&seed, Self::challenge_seed_index(index, i)))
                .collect();
            let u1 = unpack_words(claim.commitment, bits, s.kappa1 * N, self.q)?;
            self.constraints(
                field,
                index,
                &s,
                claim,
                &u1,
                u2,
                &cs,
                &mut full,
                &mut constant_term,
            )?;
        }
        let lengths = self.lengths(K);
        let relation = Relation {
            lengths: lengths.clone(),
            norm_sq,
            full,
            constant_term,
        };
        let ring = jali::labrador::Rq::new(self.q, N)?;
        let statement = relation.into_jali(&ring, &Layout::choose(&lengths))?;
        jali::labrador::verify(
            &statement,
            &labrador,
            &self.labrador_seed(),
            &labrador_context(&t),
            &self.policy,
        )?;
        Ok(())
    }
}

/// jali's context for a transcript: the tail's domain and the digest.
fn labrador_context(t: &Transcript) -> Vec<u8> {
    let mut out = TAIL_DOMAIN.to_vec();
    out.extend_from_slice(&t.digest());
    out
}

/// Knobs for the tests of the restart path.
#[cfg(test)]
impl TailParams {
    /// Restart every attempt below `attempts`.
    fn with_forced_restarts(mut self, attempts: u8) -> Self {
        self.forced_restarts = attempts;
        self
    }

    /// A reserve of `words` words for jali's proof.
    fn with_reserve_words(mut self, words: usize) -> Self {
        self.cap_words = words;
        self
    }

    /// A root Hermite factor for the binding checks of [`TailParams::check_norm`] (the ranks
    /// stay those chosen at construction).
    fn with_check_delta(mut self, delta: f64) -> Self {
        self.security.delta = delta;
        self
    }

    /// The index of a proof's length and attempt word.
    fn length_word(&self) -> usize {
        self.shapes
            .iter()
            .map(|s| self.commitment_words(s.vars))
            .sum::<usize>()
            + 2
    }
}

#[cfg(test)]
mod tests;
