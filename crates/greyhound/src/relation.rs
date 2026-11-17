//! The reduction of an evaluation proof to a statement of LaBRADOR's principal relation
//! (`polcom_reduce`), and the statement in jali's form.
//!
//! **The relation.** Witness vectors $z_0, z_1 \in \mathcal R^{mf}$, $\hat t \in
//! \mathcal R^{\kappa f_u n}$ and $\hat w \in \mathcal R^{f_u n}$ with $\Vert z_0 \Vert^2 + \Vert z_1
//! \Vert^2 + \Vert \hat t \Vert^2 + \Vert \hat w \Vert^2 \le \beta^2$ and, for $\gamma_{i f_u + j}
//! = -2^{j b_u} c_i$:
//!
//! 1. $\langle B_r, \hat t \rangle = u_{1,r}$ for each row $r < \kappa_1$ (the commitment);
//! 2. $\langle D_r, \hat w \rangle = u_{2,r}$ for each row $r < \kappa_1$;
//! 3. $\langle \varphi, z_0 \rangle + 2^{b_u} \langle \varphi, z_1 \rangle + \langle \gamma, \hat w
//!    \rangle = 0$ with $\varphi_{km + j} = 2^{kb} \bar x^j$: $a^\top z = \sum_i c_i w_i$;
//! 4. $\langle A_k, z_0 \rangle + 2^{b_u} \langle A_k, z_1 \rangle + \langle \gamma, \hat t_k
//!    \rangle = 0$ for each row $k < \kappa$, $\hat t_k$ the digits of row $k$ of all $t_i$:
//!    $Az = \sum_i c_i t_i$;
//! 5. the constant coefficient of $\langle \psi, \hat w \rangle$ is $y$, with $\psi_{i f_u + j} =
//!    2^{j b_u} \bar x^{mi} \sigma_{-1}(a)$: $\sum_i \bar x^{mi} \mathrm{ct}(\sigma_{-1}(a) w_i) =
//!    f(x)$.
//!
//! The first four hold in $\mathcal R_q$ (jali's [`jali::labrador::Statement::push`]), the last
//! in its constant coefficient ([`jali::labrador::Statement::push_constant_term`]), as the
//! reference's five sparse constraints, the first, second and fourth of which have $\kappa_1$,
//! $\kappa_1$ and $\kappa$ rows. The bound is the $\beta^2$ the proof states, as in the reference.
//!
//! **jali's form.** jali's statements have $r$ vectors of one rank $n'$, so the four vectors are
//! concatenated in this order and cut into $r$ vectors of $n'$ polynomials, the last
//! zero-padded ([`Layout`]). The padding enters no constraint: a witness with other short values
//! there satisfies the same constraints, and its other entries are a witness of the relation
//! above with a smaller norm.
use crate::challenge::Challenge;
use crate::commit::Commitment;
use crate::error::{Error, Result};
use crate::eval::{FrontProof, FrontWitness, transcript};
use crate::params::{N, Params};
use crate::pcs::PublicParams;
use hachi_co_math::{Poly, Ring};
use rayon::prelude::*;
use std::sync::Arc;

/// A term $\langle \varphi, v[o..o + \ell] \rangle$: the vector, the offset $o$ in polynomials
/// and the $64 \ell$ canonical coefficients of $\varphi$.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Term {
    /// The witness vector: 0 for $z_0$, 1 for $z_1$, 2 for $\hat t$, 3 for $\hat w$.
    pub vector: usize,
    /// The first polynomial of the vector that the term reads.
    pub offset: usize,
    /// $\varphi$, $64 \ell$ residues.
    pub phi: Vec<u64>,
}

/// A linear constraint $\sum \langle \varphi, v[o..o + \ell] \rangle = b$.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint {
    /// The terms.
    pub terms: Vec<Term>,
    /// $b$, 64 residues (a constant-term constraint uses its constant coefficient).
    pub b: Vec<u64>,
}

/// The reduced relation (module documentation).
#[derive(Clone, Debug)]
pub struct Relation {
    /// The lengths of the witness vectors in polynomials: $z_0, z_1, \hat t, \hat w$ for one
    /// evaluation proof (and four per claim in [`crate::tail`]).
    pub lengths: Vec<usize>,
    /// $\beta^2$.
    pub norm_sq: u128,
    /// The constraints that vanish in $\mathcal R_q$: $\kappa_1 + \kappa_1 + 1 + \kappa$ rows.
    pub full: Vec<Constraint>,
    /// The constraint whose constant coefficient vanishes.
    pub constant_term: Vec<Constraint>,
}

/// $\sigma_{-1}(a)$ for $a = \sum_{l<64} x^l X^l$: coefficients $1, -x^{63}, \dots, -x$.
pub(crate) fn sigma_powers(ring: &Ring, x: u64) -> Vec<u64> {
    let zq = ring.zq();
    let mut out = vec![0u64; N];
    out[0] = 1;
    let mut acc = 1u64;
    for l in 1..N {
        acc = zq.mul(acc, x);
        out[N - l] = zq.neg(acc);
    }
    out
}

/// The row of a public matrix, flattened.
pub(crate) fn flatten(row: &[Poly]) -> Vec<u64> {
    row.iter().flat_map(|p| p.0.iter().copied()).collect()
}

/// $s \cdot v$ coefficientwise.
pub(crate) fn scaled(ring: &Ring, v: &[u64], s: u64) -> Vec<u64> {
    let zq = ring.zq();
    v.par_iter().map(|x| zq.mul(*x, s)).collect()
}

impl Relation {
    /// The verifier's reduction: the challenges from the transcript, then
    /// [`Relation::with_challenges`].
    ///
    /// # Errors
    /// [`Error::Shape`] for inputs of the wrong size.
    pub fn new(
        pp: &PublicParams,
        com: &Commitment,
        x: u64,
        y: u64,
        proof: &FrontProof,
    ) -> Result<Self> {
        let (_, cs) = transcript(pp, com, x, y, &proof.u2);
        Self::with_challenges(pp, com, x, y, proof, &cs)
    }

    /// The relation for the challenges `cs` (module documentation).
    ///
    /// # Errors
    /// [`Error::Shape`] for inputs of the wrong size.
    pub fn with_challenges(
        pp: &PublicParams,
        com: &Commitment,
        x: u64,
        y: u64,
        proof: &FrontProof,
        cs: &[Challenge],
    ) -> Result<Self> {
        let p = pp.params();
        let ring = pp.ring();
        let zq = ring.zq();
        let (m, n, f, fu) = (p.m, p.n, p.f, p.fu);
        if x >= p.q
            || y >= p.q
            || com.u1.len() != p.kappa1
            || proof.u2.len() != p.kappa1
            || cs.len() != n
        {
            return Err(Error::Shape(
                "point, value, commitment or proof of the wrong size",
            ));
        }
        let key = pp.key();
        let xbar = zq.pow(x, N as u64);
        let two_bu = zq.pow(2, u64::from(p.bu));
        // gamma_{i fu + j} = -2^{j bu} c_i.
        let mut gamma = vec![0u64; p.w_len() * N];
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
        let mut full = Vec::with_capacity(2 * p.kappa1 + 1 + p.kappa);
        // 1. The commitment u1 = B t-hat.
        let rows: Vec<Constraint> = key
            .b
            .par_iter()
            .zip(&com.u1)
            .map(|(row, u)| Constraint {
                terms: vec![Term {
                    vector: 2,
                    offset: 0,
                    phi: flatten(row),
                }],
                b: u.0.clone(),
            })
            .collect();
        full.extend(rows);
        // 2. u2 = D w-hat.
        for (row, u) in key.d.iter().zip(&proof.u2) {
            full.push(Constraint {
                terms: vec![Term {
                    vector: 3,
                    offset: 0,
                    phi: flatten(row),
                }],
                b: u.0.clone(),
            });
        }
        // 3. a^T z = sum_i c_i w_i, with phi_{k m + j} = 2^{k b} xbar^j.
        let mut phi = vec![0u64; p.z_len() * N];
        let two_b = zq.pow(2, u64::from(p.b));
        let mut digit_scale = 1u64;
        for k in 0..f {
            let mut s = digit_scale;
            for j in 0..m {
                phi[(k * m + j) * N] = s;
                s = zq.mul(s, xbar);
            }
            digit_scale = zq.mul(digit_scale, two_b);
        }
        let phi1 = scaled(ring, &phi, two_bu);
        full.push(Constraint {
            terms: vec![
                Term {
                    vector: 0,
                    offset: 0,
                    phi,
                },
                Term {
                    vector: 1,
                    offset: 0,
                    phi: phi1,
                },
                Term {
                    vector: 3,
                    offset: 0,
                    phi: gamma.clone(),
                },
            ],
            b: vec![0; N],
        });
        // 4. A z = sum_i c_i t_i, row by row.
        let rows: Vec<Constraint> = key
            .a
            .par_iter()
            .enumerate()
            .map(|(k, row)| {
                let a = flatten(row);
                let a1 = scaled(ring, &a, two_bu);
                Constraint {
                    terms: vec![
                        Term {
                            vector: 0,
                            offset: 0,
                            phi: a,
                        },
                        Term {
                            vector: 1,
                            offset: 0,
                            phi: a1,
                        },
                        Term {
                            vector: 2,
                            offset: k * n * fu,
                            phi: gamma.clone(),
                        },
                    ],
                    b: vec![0; N],
                }
            })
            .collect();
        full.extend(rows);
        // 5. ct(<psi, w-hat>) = y, psi_{i fu + j} = 2^{j bu} xbar^{m i} sigma_{-1}(a).
        let sigma = sigma_powers(ring, x);
        let xbar_m = zq.pow(xbar, m as u64);
        let mut psi = vec![0u64; p.w_len() * N];
        let mut column = 1u64;
        for i in 0..n {
            let mut s = column;
            for j in 0..fu {
                let at = (i * fu + j) * N;
                for (o, v) in psi[at..at + N].iter_mut().zip(&sigma) {
                    *o = zq.mul(*v, s);
                }
                s = zq.mul(s, two_bu);
            }
            column = zq.mul(column, xbar_m);
        }
        let mut b = vec![0u64; N];
        b[0] = y;
        Ok(Self {
            lengths: p.witness_lengths().to_vec(),
            norm_sq: proof.norm_sq,
            full,
            constant_term: vec![Constraint {
                terms: vec![Term {
                    vector: 3,
                    offset: 0,
                    phi: psi,
                }],
                b,
            }],
        })
    }

    /// The number of coefficients of all $\varphi$ of the relation.
    pub fn coefficients(&self) -> usize {
        self.full
            .iter()
            .chain(&self.constant_term)
            .flat_map(|c| &c.terms)
            .map(|t| t.phi.len())
            .sum()
    }

    /// Whether `witness` satisfies the relation: its lengths, its norm and every constraint,
    /// with the exact products of the workspace arithmetic (the reference's
    /// `principle_verify`).
    ///
    /// # Errors
    /// [`Error::Witness`] naming the first condition that fails.
    pub fn check(&self, ring: &Ring, witness: &FrontWitness) -> Result<()> {
        let vectors: Vec<&[i64]> = witness.vectors.iter().map(Vec::as_slice).collect();
        self.check_vectors(ring, &vectors)
    }

    /// [`Relation::check`] for witness vectors given as centred coefficients, one slice per
    /// entry of [`Relation::lengths`].
    ///
    /// # Errors
    /// [`Error::Witness`] naming the first condition that fails.
    pub fn check_vectors(&self, ring: &Ring, vectors: &[&[i64]]) -> Result<()> {
        if vectors.len() != self.lengths.len()
            || vectors
                .iter()
                .zip(&self.lengths)
                .any(|(v, len)| v.len() != len * N)
        {
            return Err(Error::Witness("vector lengths"));
        }
        if norm_sq_of(vectors) > self.norm_sq {
            return Err(Error::Witness("norm"));
        }
        let zq = ring.zq();
        let canonical: Vec<Vec<u64>> = vectors
            .iter()
            .map(|v| v.par_iter().map(|x| zq.from_i64(*x)).collect())
            .collect();
        let value = |c: &Constraint| -> Result<Vec<u64>> {
            let mut acc = vec![0u64; N];
            for t in &c.terms {
                let len = t.phi.len() / N;
                let Some(v) = canonical.get(t.vector) else {
                    return Err(Error::Witness("term out of range"));
                };
                if t.phi.len() % N != 0 || (t.offset + len) * N > v.len() {
                    return Err(Error::Witness("term out of range"));
                }
                let phis = t.phi.as_chunks::<N>().0;
                let ws = v[t.offset * N..(t.offset + len) * N].as_chunks::<N>().0;
                let s = ring.rns().inner(phis, ws)?;
                for (a, x) in acc.iter_mut().zip(s) {
                    *a = zq.add(*a, x);
                }
            }
            Ok(acc)
        };
        let full: Vec<bool> = self
            .full
            .par_iter()
            .map(|c| value(c).map(|v| v == c.b))
            .collect::<Result<_>>()?;
        if full.iter().any(|ok| !ok) {
            return Err(Error::Witness("a constraint over R_q"));
        }
        for c in &self.constant_term {
            if value(c)?[0] != c.b[0] {
                return Err(Error::Witness("the constant-term constraint"));
            }
        }
        Ok(())
    }

    /// The statement in jali's form for `layout` (module documentation), consuming the
    /// relation.
    ///
    /// # Errors
    /// [`Error::Labrador`] if jali does not take the modulus (jali 0.3.1 requires $q < 2^{62}$) or
    /// the dimensions; [`Error::Parameter`] for a layout that does not hold the witness.
    pub fn into_jali(
        self,
        ring: &Arc<jali::labrador::Rq>,
        layout: &Layout,
    ) -> Result<jali::labrador::Statement> {
        let base = layout.offsets(&self.lengths)?;
        let mut st =
            jali::labrador::Statement::new(ring.clone(), layout.r, layout.n, self.norm_sq)?;
        let convert = |c: Constraint| -> jali::labrador::Constraint {
            let mut out = jali::labrador::Constraint::new(c.b);
            for t in c.terms {
                let len = t.phi.len() / N;
                if len == 0 || t.vector >= base.len() {
                    continue;
                }
                let start = base[t.vector] + t.offset;
                let (first, last) = (start / layout.n, (start + len - 1) / layout.n);
                if first == last {
                    out = out.linear(first, start % layout.n, t.phi);
                    continue;
                }
                let mut pos = 0;
                while pos < len {
                    let global = start + pos;
                    let (i, o) = (global / layout.n, global % layout.n);
                    let take = (len - pos).min(layout.n - o);
                    out = out.linear(i, o, t.phi[pos * N..(pos + take) * N].to_vec());
                    pos += take;
                }
            }
            out
        };
        let full: Vec<_> = self.full.into_par_iter().map(convert).collect();
        for c in full {
            st.push(c)?;
        }
        for c in self.constant_term {
            st.push_constant_term(convert(c))?;
        }
        Ok(st)
    }
}

/// $\sum \Vert v \Vert^2$ of centred coefficients, saturating.
pub(crate) fn norm_sq_of(vectors: &[&[i64]]) -> u128 {
    vectors.iter().flat_map(|v| v.iter()).fold(0u128, |acc, x| {
        let a = u128::from(x.unsigned_abs());
        acc.saturating_add(a * a)
    })
}

/// How the witness vectors fill jali's $r$ vectors of $n'$ polynomials.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// jali's multiplicity $r$.
    pub r: usize,
    /// jali's rank $n'$.
    pub n: usize,
}

/// Rank granularity: $n'$ is a multiple of it, so that jali can split the next amortized
/// opening into parts of equal length (it takes divisors of $n'$).
const RANK_STEP: usize = 16;

/// The default multiplicity of jali's statement.
pub const DEFAULT_MULTIPLICITY: usize = 8;

impl Layout {
    /// The default layout: [`Layout::with_multiplicity`] of [`DEFAULT_MULTIPLICITY`] vectors.
    /// Measured on hco-dev1 for degrees $2^{21}$, $2^{23}$ and $2^{25}$ at $q = 2^{32} - 99$
    /// with $r$ from 2 to 32 (17 November 2026), 8 vectors gave the smallest LaBRADOR proof at
    /// $2^{21}$ and $2^{23}$ and one 0.3% above the smallest (6 vectors) at $2^{25}$, in about
    /// the least proving time.
    pub fn choose(lengths: &[usize]) -> Self {
        Self::with_multiplicity(lengths, DEFAULT_MULTIPLICITY)
    }

    /// The layout of about `r` vectors: $n' = \lceil T/r \rceil$ rounded up to a multiple of 16,
    /// and $r = \lceil T/n' \rceil$.
    pub fn with_multiplicity(lengths: &[usize], r: usize) -> Self {
        let total: usize = lengths.iter().sum();
        let n = total.max(1).div_ceil(r.max(1)).next_multiple_of(RANK_STEP);
        Self {
            r: total.max(1).div_ceil(n),
            n,
        }
    }

    /// The offsets of the vectors in the concatenation, after checking that the layout holds
    /// them.
    pub(crate) fn offsets(&self, lengths: &[usize]) -> Result<Vec<usize>> {
        let total: usize = lengths.iter().sum();
        if self.n == 0 || self.r.checked_mul(self.n).is_none_or(|x| x < total) {
            return Err(Error::Parameter("the layout does not hold the witness"));
        }
        Ok(lengths
            .iter()
            .scan(0, |at, len| {
                let start = *at;
                *at += len;
                Some(start)
            })
            .collect())
    }

    /// The witness in jali's form.
    ///
    /// # Errors
    /// [`Error::Parameter`] for a layout that does not hold the witness.
    pub fn jali_witness(
        &self,
        ring: &jali::labrador::Rq,
        params: &Params,
        witness: &FrontWitness,
    ) -> Result<jali::labrador::Witness> {
        let vectors: Vec<&[i64]> = witness.vectors.iter().map(Vec::as_slice).collect();
        self.jali_witness_of(ring, &params.witness_lengths(), &vectors)
    }

    /// The witness in jali's form for vectors of `lengths` polynomials, given as centred
    /// coefficients.
    ///
    /// # Errors
    /// [`Error::Parameter`] for a layout that does not hold the witness or vectors of other
    /// lengths.
    pub fn jali_witness_of(
        &self,
        ring: &jali::labrador::Rq,
        lengths: &[usize],
        vectors: &[&[i64]],
    ) -> Result<jali::labrador::Witness> {
        self.offsets(lengths)?;
        if vectors.len() != lengths.len()
            || vectors.iter().zip(lengths).any(|(v, l)| v.len() != l * N)
        {
            return Err(Error::Parameter("witness vectors of other lengths"));
        }
        let mut flat = Vec::with_capacity(self.r * self.n * N);
        for v in vectors {
            flat.extend_from_slice(v);
        }
        flat.resize(self.r * self.n * N, 0);
        let vectors: Vec<Vec<i64>> = flat.chunks_exact(self.n * N).map(<[i64]>::to_vec).collect();
        Ok(jali::labrador::Witness::new(ring, vectors))
    }
}
