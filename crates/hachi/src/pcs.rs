//! The polynomial commitment scheme: [`HachiPcs`].
//!
//! Two kinds of committed objects, both with evaluation points in $\mathbb F_{q^K}$:
//! - **Multilinear polynomials** $f \in \mathbb Z_q^{\le 1}\[X_1, \dots, X_\ell\]$ given by
//!   their $2^\ell$ monomial coefficients ([`HachiPcs::commit_multilinear`]); coefficient $i$
//!   multiplies $X_1^{i_1} \cdots X_\ell^{i_\ell}$ with $\mathrm{bit}(i) = (i_1, \dots, i_\ell)$
//!   most significant first. Openings reduce $f(\mathbf x) = y$ to a claim over $\mathcal R_q$
//!   by the paper's Section 3.2 then Section 3.1: the prover sends the $K$ partial evaluations
//!   $y_i = f_i(x_{\kappa+1}, \dots, x_\ell)$ over the first $\kappa$ variables, the packed
//!   polynomial $f' = \sum_i f_i Z_i$ goes into ring elements by $\psi$, and the prover sends
//!   $Y = F(\mathbf x_{\mathrm{high}})$, checked by $E(Y) = (K/d)\,\mathrm{Tr}\_H(Y \sigma_{-1}(v))
//!   = f'(\mathbf x')$ (Theorem 2).
//! - **Vectors** $\mathbf s \in \mathbb F_{q^K}^N$ ([`HachiPcs::commit_vector`]), opened as an
//!   inner product $\langle \mathbf s, \mathbf e \rangle$ with any public weights: by Theorem 2,
//!   $\langle \mathbf s, \mathbf e \rangle = (K/d)\,\mathrm{Tr}\_H(\sum_t \psi(\mathbf s_t)
//!   \sigma_{-1}(\psi(\mathbf e_t)))$ over blocks of $d/K$. The prover sends that sum $Y$; the
//!   weights are not a tensor, so the claim has a single chunk ($r = 0$).
//!
//! Then the iterations of the [`Schedule`] ([`crate::iteration`]) and the final step, which
//! sends the last witness in the clear (paper, Section 4.4): the verifier recomputes its
//! commitment and checks the claim directly. No step hides anything: this is the non-hiding
//! scheme (hiding comes with the Hwang-Seo-Song changes).

use crate::commit::{Committed, Ctx, commit};
use crate::error::VerifyError;
use crate::fs::Fs;
use crate::iteration::{self, Claim, IterationProof, check_exts, check_polys, evaluate_claim};
use crate::params::{HachiChoices, Layout, Schedule, best_layout, max_digit};
use crate::sumcheck::RoundMessage;
use crate::{Error, Result};
use hachi_co_math::decompose::digit_range;
use hachi_co_math::{Ext, Poly};

/// The format version of [`Proof::to_words`] and [`Commitment::to_words`].
pub const VERSION: u64 = 1;

/// A commitment $\mathbf u \in \mathcal R_q^{n_B}$.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commitment {
    /// $\mathbf u$.
    pub u: Vec<Poly>,
}

/// What the input of a committed object was.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Input<const K: usize> {
    Multilinear { vars: usize, coeffs: Vec<u64> },
    Vector { values: Vec<Ext<K>> },
}

/// The prover's state after committing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State<const K: usize> {
    input: Input<K>,
    committed: Committed,
}

/// An evaluation proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof<const K: usize> {
    /// The $K$ partial evaluations of a multilinear opening (empty for an inner product).
    pub partials: Vec<Ext<K>>,
    /// The ring element $Y$ of the reduction to $\mathcal R_q$.
    pub ring_value: Poly,
    /// The iterations.
    pub iterations: Vec<IterationProof<K>>,
    /// The final witness's message, chunk after chunk.
    pub final_message: Vec<Poly>,
}

/// Hachi with fixed parameters.
#[derive(Clone, Debug)]
pub struct HachiPcs<const K: usize> {
    ctx: Ctx<K>,
}

/// Transcript labels.
const LABEL_STATEMENT: &[u8] = b"hachi/statement";
const LABEL_COMMITMENT: &[u8] = b"hachi/commitment";
const LABEL_POINT: &[u8] = b"hachi/point";
const LABEL_VALUE: &[u8] = b"hachi/value";
const LABEL_PARTIALS: &[u8] = b"hachi/input/partials";
const LABEL_RING_VALUE: &[u8] = b"hachi/input/ring-value";

impl<const K: usize> HachiPcs<K> {
    /// Hachi over $\mathbb F_{q^K}$ with the given choices.
    ///
    /// # Errors
    /// [`Error::Parameter`] for unusable choices (see [`crate::params::check`]).
    pub fn new(q: u64, choices: HachiChoices) -> Result<Self> {
        Ok(Self {
            ctx: Ctx::new(q, choices)?,
        })
    }

    /// The arithmetic context.
    pub fn ctx(&self) -> &Ctx<K> {
        &self.ctx
    }

    /// The layout of a multilinear polynomial in `vars` $\ge \alpha$ variables: $2^{\ell-\alpha}$
    /// ring elements, decomposed, split to minimise the first iteration's next witness.
    ///
    /// # Errors
    /// [`Error::Parameter`] if `vars` $< \alpha$.
    pub fn multilinear_layout(&self, vars: usize) -> Result<Layout> {
        let n = vars.checked_sub(self.ctx.alpha()).ok_or_else(|| {
            Error::Parameter(format!(
                "a polynomial needs at least alpha = {} variables",
                self.ctx.alpha()
            ))
        })?;
        best_layout(
            &self.ctx.choices,
            self.ctx.q,
            n,
            true,
            max_digit(self.ctx.choices.b),
        )
    }

    /// The layout of a vector of `len` $\ge 1$ field elements: one chunk of
    /// $2^{\lceil \log_2 \lceil len K/d \rceil \rceil}$ ring elements, decomposed.
    ///
    /// # Errors
    /// [`Error::Shape`] for an empty vector.
    pub fn vector_layout(&self, len: usize) -> Result<Layout> {
        if len == 0 {
            return Err(Error::Shape("an empty vector cannot be committed".into()));
        }
        let blocks = len.div_ceil(self.ctx.d_over_k());
        Ok(Layout {
            r: 0,
            m: blocks.next_power_of_two().trailing_zeros() as usize,
            decomposed: true,
            beta_s: max_digit(self.ctx.choices.b),
        })
    }

    /// The schedule of an opening of an object of layout `first`.
    ///
    /// # Errors
    /// [`Error::Parameter`] if an iteration cannot be parametrised.
    pub fn schedule(&self, first: Layout) -> Result<Schedule> {
        Schedule::derive(&self.ctx.choices, self.ctx.q, first)
    }

    /// The number of words of [`Commitment::to_words`]: $3 + n_B d$.
    pub fn commitment_words(&self) -> usize {
        3 + self.ctx.choices.n_b * self.ctx.d()
    }

    /// The number of words of [`Proof::to_words`] for an opening of a multilinear polynomial in
    /// `vars` variables. It depends only on `vars` and the parameters.
    ///
    /// # Errors
    /// As [`HachiPcs::multilinear_layout`] and [`HachiPcs::schedule`].
    pub fn multilinear_proof_words(&self, vars: usize) -> Result<usize> {
        self.proof_words(self.multilinear_layout(vars)?, K)
    }

    /// The number of words of [`Proof::to_words`] for an inner-product opening of a vector of
    /// `len` elements. It depends only on `len` and the parameters.
    ///
    /// # Errors
    /// As [`HachiPcs::vector_layout`] and [`HachiPcs::schedule`].
    pub fn vector_proof_words(&self, len: usize) -> Result<usize> {
        self.proof_words(self.vector_layout(len)?, 0)
    }

    /// The words of [`Proof::to_words`] for a first witness of layout `first` and `partials`
    /// partial evaluations: the version and $d$ (2), the partial evaluations ($1 + K \cdot$
    /// `partials`), $Y$ ($d$), the number of iterations (1), per iteration $\mathbf v$
    /// ($1 + n_D d$), $\mathbf u'$ ($1 + n_B d$), the rounds ($1 + L (2 + K (b + 2) + 3K)$, with
    /// $b + 2$ values of $H_0$ for the $b$ digits and 3 of $H_\alpha$), $y'$ ($K$), the partial
    /// evaluations ($1 + K^2$) and $p'$ ($d$), and the final message ($1 + 2^{r+m} d$).
    fn proof_words(&self, first: Layout, partials: usize) -> Result<usize> {
        let (d, c) = (self.ctx.d(), &self.ctx.choices);
        let schedule = self.schedule(first)?;
        let (lo, hi) = digit_range(c.b);
        let digits = usize::try_from(hi - lo + 1)
            .map_err(|_| Error::Parameter("the digit set is empty".into()))?;
        let round = (1 + K * (digits + 2)) + (1 + K * 3);
        let mut words = 2 + (1 + K * partials) + d + 1;
        for it in &schedule.iterations {
            words += (1 + c.n_d * d) + (1 + c.n_b * d) + (1 + it.table_vars * round);
            words += K + (1 + K * K) + d;
        }
        let last = schedule.final_layout;
        Ok(words + 1 + last.chunks() * last.chunk_len() * d)
    }

    /// Commits to the multilinear polynomial with monomial coefficients `coeffs` ($2^\ell$
    /// residues, $\ell \ge \alpha$; module documentation for the order).
    ///
    /// # Errors
    /// [`Error::Shape`] unless the length is a power of two of at least $d$; arithmetic errors.
    pub fn commit_multilinear(&self, coeffs: &[u64]) -> Result<(Commitment, State<K>)> {
        let ctx = &self.ctx;
        if !coeffs.len().is_power_of_two() || coeffs.len() < ctx.d() {
            return Err(Error::Shape(format!(
                "2^l coefficients with l >= alpha = {} are expected",
                ctx.alpha()
            )));
        }
        let vars = coeffs.len().trailing_zeros() as usize;
        let layout = self.multilinear_layout(vars)?;
        let coeffs: Vec<u64> = coeffs.iter().map(|c| ctx.ring.zq().reduce(*c)).collect();
        let message = self.multilinear_message(&layout, vars, &coeffs)?;
        let committed = commit(ctx, 0, layout, message)?;
        Ok((
            Commitment {
                u: committed.u.clone(),
            },
            State {
                input: Input::Multilinear { vars, coeffs },
                committed,
            },
        ))
    }

    /// The message chunks of a multilinear polynomial: $f'_j = \sum_i f_{i \Vert j} Z_i$, then
    /// $\psi$ of $d/K$ consecutive $f'_j$.
    fn multilinear_message(
        &self,
        layout: &Layout,
        vars: usize,
        coeffs: &[u64],
    ) -> Result<Vec<Vec<Poly>>> {
        let ctx = &self.ctx;
        let inner = 1usize << (vars - ctx.kappa());
        let packed: Vec<Ext<K>> = (0..inner)
            .map(|j| Ext(core::array::from_fn(|i| coeffs[i * inner + j])))
            .collect();
        let elements = packed
            .chunks_exact(ctx.d_over_k())
            .map(|block| ctx.sub.psi(block))
            .collect::<hachi_co_math::Result<Vec<_>>>()?;
        Ok(elements
            .chunks_exact(layout.chunk_len())
            .map(<[Poly]>::to_vec)
            .collect())
    }

    /// Commits to the vector `values` ($N \ge 1$ field elements).
    ///
    /// # Errors
    /// [`Error::Shape`] for an empty vector; arithmetic errors.
    pub fn commit_vector(&self, values: &[Ext<K>]) -> Result<(Commitment, State<K>)> {
        let ctx = &self.ctx;
        let layout = self.vector_layout(values.len())?;
        let values: Vec<Ext<K>> = values.iter().map(|v| ctx.field().reduce(v)).collect();
        let message = vec![self.vector_blocks(&layout, &values)?];
        let committed = commit(ctx, 0, layout, message)?;
        Ok((
            Commitment {
                u: committed.u.clone(),
            },
            State {
                input: Input::Vector { values },
                committed,
            },
        ))
    }

    /// $\psi$ of the blocks of $d/K$ entries of `values`, zero-padded to $2^m$ blocks.
    fn vector_blocks(&self, layout: &Layout, values: &[Ext<K>]) -> Result<Vec<Poly>> {
        let ctx = &self.ctx;
        let per = ctx.d_over_k();
        let mut padded = values.to_vec();
        padded.resize(layout.chunk_len() * per, ctx.field().zero());
        Ok(padded
            .chunks_exact(per)
            .map(|block| ctx.sub.psi(block))
            .collect::<hachi_co_math::Result<Vec<_>>>()?)
    }

    /// The transcript after the statement: the choices, the object's shape, the commitment,
    /// the point or weights, and the value.
    fn statement(
        &self,
        context: &[u8; 32],
        kind: u64,
        size: usize,
        commitment: &[Poly],
        point: &[Ext<K>],
        value: &Ext<K>,
    ) -> Fs {
        let c = &self.ctx.choices;
        let mut fs = Fs::new(context);
        let mut words = vec![
            VERSION,
            kind,
            size as u64,
            self.ctx.q,
            K as u64,
            c.d as u64,
            c.b,
            c.omega,
            c.n_a as u64,
            c.n_b as u64,
            c.n_d as u64,
            c.final_words as u64,
            c.max_iterations as u64,
            c.fixed_iterations.map_or(u64::MAX, |n| n as u64),
        ];
        words.extend(
            c.seed
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| u64::from_le_bytes(*b)),
        );
        fs.words(LABEL_STATEMENT, &words);
        fs.polys(LABEL_COMMITMENT, commitment);
        fs.exts(LABEL_POINT, point);
        fs.exts(LABEL_VALUE, core::slice::from_ref(value));
        fs
    }

    /// The claim of the first iteration of a multilinear opening and the checks of the
    /// reduction: the value from the partial evaluations, and $E(Y) = f'(\mathbf x')$.
    fn multilinear_claim(
        &self,
        layout: &Layout,
        point: &[Ext<K>],
        partials: &[Ext<K>],
        ring_value: &Poly,
    ) -> Result<(Claim, Ext<K>, bool)> {
        let ctx = &self.ctx;
        let field = ctx.field();
        let kappa = ctx.kappa();
        let vars = point.len();
        let mono_hi = ctx.sub.monomials(&point[..kappa])?;
        let mut value = field.zero();
        let mut packed = field.zero();
        for (i, y) in partials.iter().enumerate() {
            value = field.add(&value, &field.mul(&mono_hi[i], y));
            let mut z = Ext([0u64; K]);
            z.0[i] = 1;
            packed = field.add(&packed, &field.mul(y, &z));
        }
        let ring_vars = vars - ctx.alpha();
        let x_high = &point[kappa..kappa + ring_vars];
        let x_low = &point[kappa + ring_vars..];
        let v_low = ctx.sub.psi(&ctx.sub.monomials(x_low)?)?;
        let traced = ctx.sub.extract(&ctx.sub.e_map(ring_value, &v_low)?)?;
        let b = ctx
            .sub
            .monomials(&x_high[..layout.r])?
            .iter()
            .map(|e| ctx.sub.embed(e))
            .collect();
        let a = ctx
            .sub
            .monomials(&x_high[layout.r..])?
            .iter()
            .map(|e| ctx.sub.embed(e))
            .collect();
        Ok((
            Claim {
                b,
                a,
                target: ring_value.clone(),
            },
            value,
            traced == packed,
        ))
    }

    /// The weights $\sigma_{-1}(\psi(\mathbf e_t))$ of an inner product, zero-padded.
    fn vector_weights(&self, layout: &Layout, weights: &[Ext<K>]) -> Result<Vec<Poly>> {
        self.vector_blocks(layout, weights)?
            .iter()
            .map(|p| Ok(self.ctx.ring.automorphism(p, -1)?))
            .collect()
    }

    /// Runs the iterations and the final step for the prover.
    fn prove_rest(
        &self,
        layout: Layout,
        committed: &Committed,
        claim: Claim,
        fs: &mut Fs,
    ) -> Result<(Vec<IterationProof<K>>, Vec<Poly>)> {
        let schedule = self.schedule(layout)?;
        let mut witness = committed.clone();
        let mut claim = claim;
        let mut proofs = Vec::with_capacity(schedule.iterations.len());
        for (index, params) in schedule.iterations.iter().enumerate() {
            let (proof, next, next_claim) =
                iteration::prove(&self.ctx, index, params, &witness, &claim, fs)?;
            proofs.push(proof);
            witness = next;
            claim = next_claim;
        }
        Ok((proofs, witness.message.concat()))
    }

    /// Proves $f(\mathbf x)$ for a committed multilinear polynomial; returns the value and the
    /// proof. `context` binds the proof to the caller's transcript.
    ///
    /// # Errors
    /// [`Error::Shape`] for a point of the wrong length or a state of a vector; prover errors.
    pub fn prove_multilinear(
        &self,
        state: &State<K>,
        point: &[Ext<K>],
        context: &[u8; 32],
    ) -> Result<(Ext<K>, Proof<K>)> {
        let ctx = &self.ctx;
        let field = ctx.field();
        let Input::Multilinear { vars, coeffs } = &state.input else {
            return Err(Error::Shape(
                "this state is not a multilinear polynomial".into(),
            ));
        };
        if point.len() != *vars {
            return Err(Error::Shape(format!(
                "a point of {vars} coordinates is expected"
            )));
        }
        let point: Vec<Ext<K>> = point.iter().map(|x| field.reduce(x)).collect();
        let kappa = ctx.kappa();
        let inner = 1usize << (vars - kappa);
        let mono = ctx.sub.monomials(&point[kappa..])?;
        let partials: Vec<Ext<K>> = (0..K)
            .map(|i| {
                coeffs[i * inner..(i + 1) * inner]
                    .iter()
                    .zip(&mono)
                    .fold(field.zero(), |acc, (c, m)| {
                        field.add(&acc, &field.scale(m, *c))
                    })
            })
            .collect();
        let layout = state.committed.layout;
        let ring_value = evaluate_claim(
            ctx,
            &ctx.sub
                .monomials(&point[kappa..kappa + layout.r])?
                .iter()
                .map(|e| ctx.sub.embed(e))
                .collect::<Vec<_>>(),
            &ctx.sub
                .monomials(&point[kappa + layout.r..kappa + vars - ctx.alpha()])?
                .iter()
                .map(|e| ctx.sub.embed(e))
                .collect::<Vec<_>>(),
            &state.committed.message,
        )?;
        let (claim, value, traced) =
            self.multilinear_claim(&layout, &point, &partials, &ring_value)?;
        if !traced {
            return Err(Error::Protocol(
                "the trace check of the input reduction fails".into(),
            ));
        }
        let mut fs = self.statement(context, 1, *vars, &state.committed.u, &point, &value);
        fs.exts(LABEL_PARTIALS, &partials);
        fs.polys(LABEL_RING_VALUE, core::slice::from_ref(&ring_value));
        let (iterations, final_message) =
            self.prove_rest(layout, &state.committed, claim, &mut fs)?;
        Ok((
            value,
            Proof {
                partials,
                ring_value,
                iterations,
                final_message,
            },
        ))
    }

    /// Proves $\langle \mathbf s, \mathbf e \rangle$ for a committed vector $\mathbf s$ and
    /// public weights $\mathbf e$ of the same length; returns the value and the proof.
    ///
    /// # Errors
    /// [`Error::Shape`] for weights of the wrong length or a state of a polynomial; prover
    /// errors.
    pub fn prove_inner_product(
        &self,
        state: &State<K>,
        weights: &[Ext<K>],
        context: &[u8; 32],
    ) -> Result<(Ext<K>, Proof<K>)> {
        let ctx = &self.ctx;
        let field = ctx.field();
        let Input::Vector { values } = &state.input else {
            return Err(Error::Shape("this state is not a vector".into()));
        };
        if weights.len() != values.len() {
            return Err(Error::Shape(format!(
                "{} weights are expected",
                values.len()
            )));
        }
        let weights: Vec<Ext<K>> = weights.iter().map(|x| field.reduce(x)).collect();
        let value = values
            .iter()
            .zip(&weights)
            .fold(field.zero(), |acc, (v, e)| {
                field.add(&acc, &field.mul(v, e))
            });
        let layout = state.committed.layout;
        let a = self.vector_weights(&layout, &weights)?;
        let b = vec![ctx.ring.monomial(0)];
        let ring_value = evaluate_claim(ctx, &b, &a, &state.committed.message)?;
        let one = ctx.ring.monomial(0);
        if ctx.sub.extract(&ctx.sub.e_map(&ring_value, &one)?)? != value {
            return Err(Error::Protocol(
                "the trace check of the inner product fails".into(),
            ));
        }
        let mut fs = self.statement(
            context,
            2,
            values.len(),
            &state.committed.u,
            &weights,
            &value,
        );
        fs.polys(LABEL_RING_VALUE, core::slice::from_ref(&ring_value));
        let claim = Claim {
            b,
            a,
            target: ring_value.clone(),
        };
        let (iterations, final_message) =
            self.prove_rest(layout, &state.committed, claim, &mut fs)?;
        Ok((
            value,
            Proof {
                partials: Vec::new(),
                ring_value,
                iterations,
                final_message,
            },
        ))
    }

    /// Runs the iterations' verifiers and the final step.
    fn verify_rest(
        &self,
        layout: Layout,
        commitment: &[Poly],
        claim: Claim,
        proof: &Proof<K>,
        fs: &mut Fs,
    ) -> core::result::Result<(), VerifyError> {
        let ctx = &self.ctx;
        let schedule = self.schedule(layout)?;
        if proof.iterations.len() != schedule.iterations.len() {
            return Err(VerifyError::Malformed(format!(
                "{} iterations, {} expected",
                proof.iterations.len(),
                schedule.iterations.len()
            )));
        }
        let mut u = commitment.to_vec();
        let mut claim = claim;
        for (index, (params, it)) in schedule
            .iterations
            .iter()
            .zip(&proof.iterations)
            .enumerate()
        {
            claim = iteration::verify(ctx, index, params, &u, &claim, it, fs)?;
            u.clone_from(&it.next_commitment);
        }
        let last = schedule.final_layout;
        let count = last.chunks() * last.chunk_len();
        check_polys(ctx, "final message", &proof.final_message, count)?;
        if !last.decomposed {
            for p in &proof.final_message {
                if ctx.ring.inf_norm(p)? > last.beta_s {
                    return Err(VerifyError::Check("the final witness is not short".into()));
                }
            }
        }
        let message: Vec<Vec<Poly>> = proof
            .final_message
            .chunks_exact(last.chunk_len())
            .map(<[Poly]>::to_vec)
            .collect();
        let recomputed = commit(ctx, schedule.iterations.len(), last, message.clone())?;
        if recomputed.u != u {
            return Err(VerifyError::Check(
                "the final witness does not open the commitment".into(),
            ));
        }
        if evaluate_claim(ctx, &claim.b, &claim.a, &message)? != claim.target {
            return Err(VerifyError::Check(
                "the final witness does not satisfy the claim".into(),
            ));
        }
        Ok(())
    }

    /// Checks the commitment's shape.
    fn check_commitment(&self, commitment: &Commitment) -> core::result::Result<(), VerifyError> {
        check_polys(&self.ctx, "commitment", &commitment.u, self.ctx.choices.n_b)
    }

    /// Verifies a multilinear opening: $f(\mathbf x) = $ `value` for the polynomial in `vars`
    /// variables committed in `commitment`.
    ///
    /// # Errors
    /// [`VerifyError`] for a malformed or rejected proof.
    pub fn verify_multilinear(
        &self,
        commitment: &Commitment,
        vars: usize,
        point: &[Ext<K>],
        value: &Ext<K>,
        proof: &Proof<K>,
        context: &[u8; 32],
    ) -> core::result::Result<(), VerifyError> {
        let ctx = &self.ctx;
        self.check_commitment(commitment)?;
        if point.len() != vars {
            return Err(VerifyError::Malformed(format!(
                "a point of {vars} coordinates is expected"
            )));
        }
        check_exts(ctx, "point", point, vars)?;
        check_exts(ctx, "value", core::slice::from_ref(value), 1)?;
        check_exts(ctx, "partial evaluations", &proof.partials, K)?;
        check_polys(
            ctx,
            "ring value",
            core::slice::from_ref(&proof.ring_value),
            1,
        )?;
        let layout = self.multilinear_layout(vars)?;
        let (claim, from_partials, traced) =
            self.multilinear_claim(&layout, point, &proof.partials, &proof.ring_value)?;
        if from_partials != *value {
            return Err(VerifyError::Check(
                "the partial evaluations of the input".into(),
            ));
        }
        if !traced {
            return Err(VerifyError::Check(
                "the trace of the input reduction".into(),
            ));
        }
        let mut fs = self.statement(context, 1, vars, &commitment.u, point, value);
        fs.exts(LABEL_PARTIALS, &proof.partials);
        fs.polys(LABEL_RING_VALUE, core::slice::from_ref(&proof.ring_value));
        self.verify_rest(layout, &commitment.u, claim, proof, &mut fs)
    }

    /// Verifies an inner-product opening: $\langle \mathbf s, \mathbf e \rangle = $ `value` for
    /// the vector of length `len` committed in `commitment`.
    ///
    /// # Errors
    /// [`VerifyError`] for a malformed or rejected proof.
    pub fn verify_inner_product(
        &self,
        commitment: &Commitment,
        len: usize,
        weights: &[Ext<K>],
        value: &Ext<K>,
        proof: &Proof<K>,
        context: &[u8; 32],
    ) -> core::result::Result<(), VerifyError> {
        let ctx = &self.ctx;
        self.check_commitment(commitment)?;
        check_exts(ctx, "weights", weights, len)?;
        check_exts(ctx, "value", core::slice::from_ref(value), 1)?;
        if !proof.partials.is_empty() {
            return Err(VerifyError::Malformed(
                "an inner product has no partial evaluations".into(),
            ));
        }
        check_polys(
            ctx,
            "ring value",
            core::slice::from_ref(&proof.ring_value),
            1,
        )?;
        let layout = self.vector_layout(len)?;
        let one = ctx.ring.monomial(0);
        if ctx.sub.extract(&ctx.sub.e_map(&proof.ring_value, &one)?)? != *value {
            return Err(VerifyError::Check("the trace of the inner product".into()));
        }
        let mut fs = self.statement(context, 2, len, &commitment.u, weights, value);
        fs.polys(LABEL_RING_VALUE, core::slice::from_ref(&proof.ring_value));
        let claim = Claim {
            b: vec![one],
            a: self.vector_weights(&layout, weights)?,
            target: proof.ring_value.clone(),
        };
        self.verify_rest(layout, &commitment.u, claim, proof, &mut fs)
    }
}

// ------------------------------------------------------------------------------- encodings ----

/// A reader of words that never panics.
struct Reader<'a> {
    words: &'a [u64],
    at: usize,
}

impl Reader<'_> {
    fn word(&mut self) -> Result<u64> {
        let w = *self
            .words
            .get(self.at)
            .ok_or_else(|| Error::Shape("the encoding ends early".into()))?;
        self.at += 1;
        Ok(w)
    }

    fn len(&mut self, max: usize) -> Result<usize> {
        let n = usize::try_from(self.word()?).unwrap_or(usize::MAX);
        if n > max || n > self.words.len() {
            return Err(Error::Shape("a length in the encoding is too large".into()));
        }
        Ok(n)
    }

    fn polys(&mut self, d: usize) -> Result<Vec<Poly>> {
        let n = self.len(self.words.len() / d.max(1) + 1)?;
        (0..n).map(|_| self.poly(d)).collect()
    }

    fn poly(&mut self, d: usize) -> Result<Poly> {
        let end = self.at.checked_add(d).filter(|e| *e <= self.words.len());
        let end = end.ok_or_else(|| Error::Shape("the encoding ends early".into()))?;
        let p = Poly(self.words[self.at..end].to_vec());
        self.at = end;
        Ok(p)
    }

    fn exts<const K: usize>(&mut self) -> Result<Vec<Ext<K>>> {
        let n = self.len(self.words.len() / K + 1)?;
        (0..n).map(|_| self.ext()).collect()
    }

    fn ext<const K: usize>(&mut self) -> Result<Ext<K>> {
        let mut e = Ext([0u64; K]);
        for x in &mut e.0 {
            *x = self.word()?;
        }
        Ok(e)
    }
}

fn push_polys(out: &mut Vec<u64>, polys: &[Poly]) {
    out.push(polys.len() as u64);
    for p in polys {
        out.extend_from_slice(&p.0);
    }
}

fn push_exts<const K: usize>(out: &mut Vec<u64>, exts: &[Ext<K>]) {
    out.push(exts.len() as u64);
    for e in exts {
        out.extend_from_slice(&e.0);
    }
}

impl Commitment {
    /// `[VERSION, n_B, d, coefficients…]`.
    pub fn to_words(&self) -> Vec<u64> {
        let d = self.u.first().map_or(0, |p| p.0.len());
        let mut out = vec![VERSION, self.u.len() as u64, d as u64];
        for p in &self.u {
            out.extend_from_slice(&p.0);
        }
        out
    }

    /// The inverse of [`Commitment::to_words`]; the shape is checked by the verifier.
    ///
    /// # Errors
    /// [`Error::Shape`] for a malformed encoding.
    pub fn from_words(words: &[u64]) -> Result<Self> {
        let mut r = Reader { words, at: 0 };
        if r.word()? != VERSION {
            return Err(Error::Shape("unknown commitment version".into()));
        }
        let n = r.len(words.len())?;
        let d = r.len(words.len())?;
        let u = (0..n).map(|_| r.poly(d)).collect::<Result<Vec<_>>>()?;
        if r.at != words.len() {
            return Err(Error::Shape("trailing words after the commitment".into()));
        }
        Ok(Self { u })
    }
}

impl<const K: usize> Proof<K> {
    /// A canonical encoding: version, $d$, the partial evaluations, $Y$, every iteration (its
    /// $\mathbf v$, $\mathbf u'$, rounds, final value, partial evaluations and $p'$), and the
    /// final message, every list length-prefixed.
    pub fn to_words(&self) -> Vec<u64> {
        let d = self.ring_value.0.len();
        let mut out = vec![VERSION, d as u64];
        push_exts(&mut out, &self.partials);
        out.extend_from_slice(&self.ring_value.0);
        out.push(self.iterations.len() as u64);
        for it in &self.iterations {
            push_polys(&mut out, &it.v);
            push_polys(&mut out, &it.next_commitment);
            out.push(it.rounds.len() as u64);
            for round in &it.rounds {
                push_exts(&mut out, &round.h0);
                push_exts(&mut out, &round.h_alpha);
            }
            out.extend_from_slice(&it.final_eval.0);
            push_exts(&mut out, &it.partials);
            out.extend_from_slice(&it.next_target.0);
        }
        push_polys(&mut out, &self.final_message);
        out
    }

    /// The inverse of [`Proof::to_words`]; shapes are checked by the verifier.
    ///
    /// # Errors
    /// [`Error::Shape`] for a malformed encoding.
    pub fn from_words(words: &[u64]) -> Result<Self> {
        let mut r = Reader { words, at: 0 };
        if r.word()? != VERSION {
            return Err(Error::Shape("unknown proof version".into()));
        }
        let d = r.len(words.len())?;
        let partials = r.exts()?;
        let ring_value = r.poly(d)?;
        let count = r.len(words.len())?;
        let mut iterations = Vec::with_capacity(count);
        for _ in 0..count {
            let v = r.polys(d)?;
            let next_commitment = r.polys(d)?;
            let rounds_n = r.len(words.len())?;
            let mut rounds = Vec::with_capacity(rounds_n);
            for _ in 0..rounds_n {
                let h0 = r.exts()?;
                let h_alpha = r.exts()?;
                rounds.push(RoundMessage { h0, h_alpha });
            }
            let final_eval = r.ext()?;
            let it_partials = r.exts()?;
            let next_target = r.poly(d)?;
            iterations.push(IterationProof {
                v,
                next_commitment,
                rounds,
                final_eval,
                partials: it_partials,
                next_target,
            });
        }
        let final_message = r.polys(d)?;
        if r.at != words.len() {
            return Err(Error::Shape("trailing words after the proof".into()));
        }
        Ok(Self {
            partials,
            ring_value,
            iterations,
            final_message,
        })
    }
}
