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
//! commitment and checks the claim directly.
//!
//! **Hiding.** With [`HachiChoices::hiding`] every commitment is randomized
//! ([`Hiding`](crate::params::Hiding), `docs/hachi-hiding.md` §6; the draft's item (i),
//! 4_preliminaries.tex:334).
//! - Under MLWE the inner commitments $\mathbf t_i$ are pseudorandom, so $\mathbf u$ is
//!   indistinguishable from the commitment of uniform $\mathbf t_i$, which does not depend on
//!   the committed object: the commitment hides it. This holds at an MLWE rank $\nu$ of real
//!   security; the toy choices have $\nu = 1$.
//! - The final step also sends the last commitment's randomness, and the verifier checks that it
//!   lies in the support of $\chi$.
//! - The openings reveal what they revealed before, the last witness and its randomness
//!   included. So a commitment hides only until it is opened, and with no iteration the opening
//!   sends the committed object itself.
//!
//! **Randomness.** The prover's methods take a secret 32-byte seed, which only hiding uses; a
//! commitment draws from the PRG stream `(seed, "hachi/commit/r", 0)`, an opening from the
//! streams of [`crate::iteration`]. Every commitment and every opening needs a fresh seed: an
//! opening of the same state with a reused seed repeats its randomness.
//!
//! **Schedule.** Unless [`HachiChoices::fixed_iterations`] fixes it, an opening runs the number
//! of iterations, from 0 to [`HachiChoices::max_iterations`], whose proof has the fewest words
//! (the fewer iterations on a tie; [`crate::params`], "How many iterations"). The prover and the
//! verifier derive it from the size of the object and the parameters.
//!
//! **Encodings** ([`HachiPcs::commitment_to_words`], [`HachiPcs::proof_to_words`]). A header of
//! plain 64-bit words, then fields packed least significant bit first into 64-bit words, the
//! last word padded with zero bits. A residue takes $\lceil \log_2 q \rceil$ bits. After at
//! least one iteration the final witness is a table of digits of base $b$ (its range was proved
//! by the last iteration's sumcheck), sent as $\psi^{-1}$ of its ring elements, each digit as its
//! index in the digit set: $\lceil \log_2 b \rceil$ bits. Without iterations the final witness
//! is the input, sent as residues. The decoders know the parameters, so a header holds only
//! what they do not fix:
//! - commitment: `[VERSION]`, then the $n_B d$ coefficients of $\mathbf u$;
//! - proof: `[VERSION, #partials, k, L_1, …, L_k, N]` for $k$ iterations of $L_i$ sumcheck
//!   rounds and a final message of $N$ ring elements, then the partial evaluations, $Y$, per
//!   iteration $\mathbf v$, $\mathbf u'$, the rounds ($b + 2$ and 3 field elements each), $y'$,
//!   the $K$ partial evaluations and $p'$, and the final message. With hiding the header ends
//!   with the number $R$ of ring elements of the final randomness, which follows the final
//!   message, each coefficient $x$ as $x - \mathrm{lo}$ in
//!   $\lceil \log_2(\mathrm{hi} - \mathrm{lo} + 1) \rceil$ bits for the support
//!   $[\mathrm{lo}, \mathrm{hi}]$ of $\chi$. Without hiding the encodings are those of the
//!   non-hiding scheme, unchanged.
//!
//! Every value has a single encoding: the decoders reject residues not below $q$, digit indices
//! outside the digit set, randomness outside the support of $\chi$, non-zero padding, and any
//! length other than the one the header and the parameters imply. The counts in a header are
//! checked against the schedule by the verifier.

use crate::commit::{Committed, Ctx, commit};
use crate::error::VerifyError;
use crate::fs::Fs;
use crate::iteration::{
    self, Claim, IterationProof, check_exts, check_polys, commitment_randomness, evaluate_claim,
};
use crate::params::{HachiChoices, Layout, Schedule, best_layout, max_digit};
use crate::sumcheck::RoundMessage;
use crate::{Error, Result};
use hachi_co_math::decompose::digit_range;
use hachi_co_math::{Ext, Poly};

/// The version of the encodings ([`HachiPcs::commitment_to_words`],
/// [`HachiPcs::proof_to_words`]), also absorbed with the statement: 2 since values are packed
/// and the schedule minimises the proof's size (version 1 had a 64-bit word per value and
/// stopped at the first iteration that did not shrink the witness).
pub const VERSION: u64 = 2;

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
    /// With hiding, the final witness's commitment randomness, chunk after chunk ($\nu + n_A$
    /// ring elements per chunk); empty without.
    pub final_randomness: Vec<Poly>,
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

/// The PRG domain of a commitment's randomness (index 0).
pub const DOMAIN_COMMIT_R: &[u8] = b"hachi/commit/r";

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

    /// The schedule of an opening of a multilinear polynomial in `vars` variables (module
    /// documentation, "Schedule").
    ///
    /// # Errors
    /// As [`HachiPcs::multilinear_layout`]; [`Error::Parameter`] if the fixed number of
    /// iterations cannot be parametrised.
    pub fn multilinear_schedule(&self, vars: usize) -> Result<Schedule> {
        self.schedule(self.multilinear_layout(vars)?, K)
    }

    /// The schedule of an inner-product opening of a vector of `len` elements (module
    /// documentation, "Schedule").
    ///
    /// # Errors
    /// As [`HachiPcs::vector_layout`]; [`Error::Parameter`] if the fixed number of iterations
    /// cannot be parametrised.
    pub fn vector_schedule(&self, len: usize) -> Result<Schedule> {
        self.schedule(self.vector_layout(len)?, 0)
    }

    /// The schedule of an opening of a first witness of layout `first` whose proof carries
    /// `partials` partial evaluations: [`HachiChoices::fixed_iterations`] iterations if set,
    /// otherwise the candidate of [`Schedule::candidates`] whose proof has the fewest words, the
    /// one with fewer iterations on a tie.
    fn schedule(&self, first: Layout, partials: usize) -> Result<Schedule> {
        let (c, q) = (&self.ctx.choices, self.ctx.q);
        if let Some(n) = c.fixed_iterations {
            return Schedule::fixed(c, q, first, n);
        }
        let mut best: Option<(usize, Schedule)> = None;
        for schedule in Schedule::candidates(c, q, first) {
            let words = self.proof_words(&schedule, partials);
            if best.as_ref().is_none_or(|(least, _)| words < *least) {
                best = Some((words, schedule));
            }
        }
        best.map(|(_, s)| s)
            .ok_or_else(|| Error::Parameter("no schedule to choose from".into()))
    }

    /// The bits of a residue in the encodings: $\lceil \log_2 q \rceil$.
    fn residue_bits(&self) -> usize {
        (u64::BITS - (self.ctx.q - 1).leading_zeros()) as usize
    }

    /// The bits of a digit of the final witness: $\lceil \log_2 b \rceil$ (the digit set has
    /// $b$ elements).
    fn digit_bits(&self) -> usize {
        (u64::BITS - (self.ctx.choices.b - 1).leading_zeros()) as usize
    }

    /// With hiding, the support $[\mathrm{lo}, \mathrm{hi}]$ of $\chi$ and the bits of a
    /// coefficient of the final randomness, $\lceil \log_2(\mathrm{hi} - \mathrm{lo} + 1)
    /// \rceil$; `None` without.
    fn noise_field(&self) -> Option<(i64, i64, usize)> {
        let h = self.ctx.hiding()?;
        // The range is valid: `check` accepted the choices.
        let (lo, hi) = h.noise.range(self.ctx.choices.b).ok()?;
        let width = (hi - lo) as u64;
        Some((lo, hi, (u64::BITS - width.leading_zeros()) as usize))
    }

    /// The number of words of [`HachiPcs::commitment_to_words`]: the version, then $n_B d$
    /// packed residues.
    pub fn commitment_words(&self) -> usize {
        1 + (self.ctx.choices.n_b * self.ctx.d() * self.residue_bits()).div_ceil(64)
    }

    /// The number of words of [`HachiPcs::proof_to_words`] for an opening of a multilinear
    /// polynomial in `vars` variables. It depends only on `vars` and the parameters.
    ///
    /// # Errors
    /// As [`HachiPcs::multilinear_schedule`].
    pub fn multilinear_proof_words(&self, vars: usize) -> Result<usize> {
        Ok(self.proof_words(&self.multilinear_schedule(vars)?, K))
    }

    /// The number of words of [`HachiPcs::proof_to_words`] for an inner-product opening of a
    /// vector of `len` elements. It depends only on `len` and the parameters.
    ///
    /// # Errors
    /// As [`HachiPcs::vector_schedule`].
    pub fn vector_proof_words(&self, len: usize) -> Result<usize> {
        Ok(self.proof_words(&self.vector_schedule(len)?, 0))
    }

    /// The words of [`HachiPcs::proof_to_words`] for the schedule `schedule` and `partials`
    /// partial evaluations (module documentation, "Encodings"): the header ($4 + k$ words for
    /// $k$ iterations, one more with hiding), then $\lceil \log_2 q \rceil$ bits per residue:
    /// the partial evaluations ($K$ residues each), $Y$ ($d$), per iteration $\mathbf v$
    /// ($n_D d$), $\mathbf u'$ ($n_B d$), the rounds ($L$ rounds of $b + 2$ and 3 field
    /// elements), $y'$ ($K$), the partial evaluations ($K^2$) and $p'$ ($d$); the final message,
    /// $2^{r+m} d$ residues without iterations, else as many digits of
    /// $\lceil \log_2 b \rceil$ bits; and with hiding the final randomness,
    /// $2^r (\nu + n_A) d$ coefficients of the bits of [`Noise`](crate::params::Noise)'s support.
    fn proof_words(&self, schedule: &Schedule, partials: usize) -> usize {
        let (d, c) = (self.ctx.d(), &self.ctx.choices);
        let round = K * (c.b as usize + 2 + 3);
        let mut residues = K * partials + d;
        for it in &schedule.iterations {
            residues += (c.n_d + c.n_b) * d + it.table_vars * round + K + K * K + d;
        }
        let last = schedule.final_layout;
        let entries = last.chunks() * last.chunk_len() * d;
        let entry_bits = if schedule.iterations.is_empty() {
            self.residue_bits()
        } else {
            self.digit_bits()
        };
        let mut bits = residues * self.residue_bits() + entries * entry_bits;
        let mut header = 4 + schedule.iterations.len();
        if let Some((_, _, noise_bits)) = self.noise_field() {
            header += 1;
            bits += last.chunks() * self.ctx.rand_len(c.n_a) * d * noise_bits;
        }
        header + bits.div_ceil(64)
    }

    /// Commits to the multilinear polynomial with monomial coefficients `coeffs` ($2^\ell$
    /// residues, $\ell \ge \alpha$; module documentation for the order), with randomness from
    /// the secret `seed` if the choices hide (module documentation, "Randomness").
    ///
    /// # Errors
    /// [`Error::Shape`] unless the length is a power of two of at least $d$; arithmetic errors.
    pub fn commit_multilinear(
        &self,
        coeffs: &[u64],
        seed: &[u8; 32],
    ) -> Result<(Commitment, State<K>)> {
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
        let r = commitment_randomness(ctx, &layout, seed, DOMAIN_COMMIT_R, 0)?;
        let committed = commit(ctx, 0, layout, message, r)?;
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

    /// Commits to the vector `values` ($N \ge 1$ field elements), with randomness from the
    /// secret `seed` if the choices hide.
    ///
    /// # Errors
    /// [`Error::Shape`] for an empty vector; arithmetic errors.
    pub fn commit_vector(
        &self,
        values: &[Ext<K>],
        seed: &[u8; 32],
    ) -> Result<(Commitment, State<K>)> {
        let ctx = &self.ctx;
        let layout = self.vector_layout(values.len())?;
        let values: Vec<Ext<K>> = values.iter().map(|v| ctx.field().reduce(v)).collect();
        let message = vec![self.vector_blocks(&layout, &values)?];
        let r = commitment_randomness(ctx, &layout, seed, DOMAIN_COMMIT_R, 0)?;
        let committed = commit(ctx, 0, layout, message, r)?;
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
        // Only with hiding, after the other words: a non-hiding statement is unchanged.
        if let Some(h) = c.hiding {
            words.push(h.nu as u64);
            words.extend(h.noise.words());
        }
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

    /// Runs the iterations and the final step for the prover; `partials` is the number of
    /// partial evaluations the proof carries (it enters the choice of the schedule). Returns the
    /// iterations, the final message and the final randomness.
    fn prove_rest(
        &self,
        layout: Layout,
        partials: usize,
        committed: &Committed,
        claim: Claim,
        fs: &mut Fs,
        seed: &[u8; 32],
    ) -> Result<(Vec<IterationProof<K>>, Vec<Poly>, Vec<Poly>)> {
        let schedule = self.schedule(layout, partials)?;
        let mut witness = committed.clone();
        let mut claim = claim;
        let mut proofs = Vec::with_capacity(schedule.iterations.len());
        for (index, params) in schedule.iterations.iter().enumerate() {
            let (proof, next, next_claim) =
                iteration::prove(&self.ctx, index, params, &witness, &claim, fs, seed)?;
            proofs.push(proof);
            witness = next;
            claim = next_claim;
        }
        Ok((proofs, witness.message.concat(), witness.r.concat()))
    }

    /// Proves $f(\mathbf x)$ for a committed multilinear polynomial; returns the value and the
    /// proof. `context` binds the proof to the caller's transcript; `seed` is the opening's
    /// fresh secret seed (module documentation, "Randomness").
    ///
    /// # Errors
    /// [`Error::Shape`] for a point of the wrong length or a state of a vector; prover errors.
    pub fn prove_multilinear(
        &self,
        state: &State<K>,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
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
        let (iterations, final_message, final_randomness) =
            self.prove_rest(layout, K, &state.committed, claim, &mut fs, seed)?;
        Ok((
            value,
            Proof {
                partials,
                ring_value,
                iterations,
                final_message,
                final_randomness,
            },
        ))
    }

    /// Proves $\langle \mathbf s, \mathbf e \rangle$ for a committed vector $\mathbf s$ and
    /// public weights $\mathbf e$ of the same length; returns the value and the proof. `seed` as
    /// for [`HachiPcs::prove_multilinear`].
    ///
    /// # Errors
    /// [`Error::Shape`] for weights of the wrong length or a state of a polynomial; prover
    /// errors.
    pub fn prove_inner_product(
        &self,
        state: &State<K>,
        weights: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
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
        let (iterations, final_message, final_randomness) =
            self.prove_rest(layout, 0, &state.committed, claim, &mut fs, seed)?;
        Ok((
            value,
            Proof {
                partials: Vec::new(),
                ring_value,
                iterations,
                final_message,
                final_randomness,
            },
        ))
    }

    /// Runs the iterations' verifiers and the final step; `partials` as for
    /// [`HachiPcs::prove_rest`].
    fn verify_rest(
        &self,
        layout: Layout,
        partials: usize,
        commitment: &[Poly],
        claim: Claim,
        proof: &Proof<K>,
        fs: &mut Fs,
    ) -> core::result::Result<(), VerifyError> {
        let ctx = &self.ctx;
        let schedule = self.schedule(layout, partials)?;
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
        let rand_len = ctx.rand_len(ctx.choices.n_a);
        check_polys(
            ctx,
            "final randomness",
            &proof.final_randomness,
            last.chunks() * rand_len,
        )?;
        if let Some((lo, hi, _)) = self.noise_field() {
            let zq = ctx.ring.zq();
            for x in proof.final_randomness.iter().flat_map(|p| &p.0) {
                if !(lo..=hi).contains(&zq.to_centered(*x)) {
                    return Err(VerifyError::Check(
                        "the final randomness is outside the support of chi".into(),
                    ));
                }
            }
        }
        let message: Vec<Vec<Poly>> = proof
            .final_message
            .chunks_exact(last.chunk_len())
            .map(<[Poly]>::to_vec)
            .collect();
        let r: Vec<Vec<Poly>> = if rand_len == 0 {
            Vec::new()
        } else {
            proof
                .final_randomness
                .chunks_exact(rand_len)
                .map(<[Poly]>::to_vec)
                .collect()
        };
        let recomputed = commit(ctx, schedule.iterations.len(), last, message.clone(), r)?;
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
        self.verify_rest(layout, K, &commitment.u, claim, proof, &mut fs)
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
        self.verify_rest(layout, 0, &commitment.u, claim, proof, &mut fs)
    }
}

// ------------------------------------------------------------------------------- encodings ----

/// Fields packed least significant bit first into 64-bit words (module documentation,
/// "Encodings").
#[derive(Default)]
struct BitWriter {
    words: Vec<u64>,
    bits: usize,
}

impl BitWriter {
    /// Appends the low `width` bits of `value` (`width` $\le 64$; 0 appends nothing).
    fn push(&mut self, value: u64, width: usize) {
        if width == 0 {
            return;
        }
        let value = if width >= 64 {
            value
        } else {
            value & ((1u64 << width) - 1)
        };
        let at = self.bits % 64;
        match self.words.last_mut() {
            Some(last) if at != 0 => {
                *last |= value << at;
                if at + width > 64 {
                    self.words.push(value >> (64 - at));
                }
            }
            _ => self.words.push(value),
        }
        self.bits += width;
    }
}

/// A reader of the fields of [`BitWriter`] that never panics.
struct BitReader<'a> {
    words: &'a [u64],
    bit: usize,
}

impl BitReader<'_> {
    /// The next field of `width` bits (`width` $\le 64$).
    fn take(&mut self, width: usize) -> Result<u64> {
        if width == 0 {
            return Ok(0);
        }
        let end = self
            .bit
            .checked_add(width)
            .filter(|end| *end <= self.words.len().saturating_mul(64))
            .ok_or_else(|| Error::Shape("the encoding ends early".into()))?;
        let (i, at) = (self.bit / 64, self.bit % 64);
        let mut value = self.words.get(i).copied().unwrap_or(0) >> at;
        if at + width > 64 {
            value |= self.words.get(i + 1).copied().unwrap_or(0) << (64 - at);
        }
        self.bit = end;
        Ok(if width >= 64 {
            value
        } else {
            value & ((1u64 << width) - 1)
        })
    }

    /// Checks that the fields fill the words exactly, up to zero padding in the last word.
    fn finish(&self) -> Result<()> {
        if self.bit.div_ceil(64) != self.words.len() {
            return Err(Error::Shape("trailing words after the encoding".into()));
        }
        let at = self.bit % 64;
        if at != 0 && self.words.last().is_some_and(|w| w >> at != 0) {
            return Err(Error::Shape("non-zero padding bits".into()));
        }
        Ok(())
    }
}

impl<const K: usize> HachiPcs<K> {
    /// Appends the coefficients of `polys` as residues.
    fn put_polys(&self, w: &mut BitWriter, polys: &[Poly]) {
        for x in polys.iter().flat_map(|p| &p.0) {
            w.push(*x, self.residue_bits());
        }
    }

    /// Appends the coordinates of `exts` as residues.
    fn put_exts(&self, w: &mut BitWriter, exts: &[Ext<K>]) {
        for x in exts.iter().flat_map(|e| e.0) {
            w.push(x, self.residue_bits());
        }
    }

    /// The next residue.
    ///
    /// # Errors
    /// [`Error::Shape`] if the encoding ends or the residue is not below $q$.
    fn take_residue(&self, r: &mut BitReader<'_>) -> Result<u64> {
        let x = r.take(self.residue_bits())?;
        if x >= self.ctx.q {
            return Err(Error::Shape("a residue is not below q".into()));
        }
        Ok(x)
    }

    /// The next ring element, as $d$ residues.
    fn take_poly(&self, r: &mut BitReader<'_>) -> Result<Poly> {
        (0..self.ctx.d())
            .map(|_| self.take_residue(r))
            .collect::<Result<Vec<_>>>()
            .map(Poly)
    }

    /// The next `count` ring elements.
    fn take_polys(&self, r: &mut BitReader<'_>, count: usize) -> Result<Vec<Poly>> {
        (0..count).map(|_| self.take_poly(r)).collect()
    }

    /// The next field element, as $K$ residues.
    fn take_ext(&self, r: &mut BitReader<'_>) -> Result<Ext<K>> {
        let mut e = Ext([0u64; K]);
        for x in &mut e.0 {
            *x = self.take_residue(r)?;
        }
        Ok(e)
    }

    /// The next `count` field elements.
    fn take_exts(&self, r: &mut BitReader<'_>, count: usize) -> Result<Vec<Ext<K>>> {
        (0..count).map(|_| self.take_ext(r)).collect()
    }

    /// The next ring element of a final witness sent as digits: $d$ indices into the digit set,
    /// the table that $\psi$ packs into the ring element.
    ///
    /// # Errors
    /// [`Error::Shape`] if the encoding ends or an index is outside the digit set.
    fn take_digit_poly(&self, r: &mut BitReader<'_>) -> Result<Poly> {
        let (zq, b) = (self.ctx.ring.zq(), self.ctx.choices.b);
        let (lo, _) = digit_range(b);
        let mut table = vec![Ext([0u64; K]); self.ctx.d_over_k()];
        for x in table.iter_mut().flat_map(|e| e.0.iter_mut()) {
            let index = r.take(self.digit_bits())?;
            if index >= b {
                return Err(Error::Shape("a digit index outside the digit set".into()));
            }
            *x = zq.from_i128(lo + i128::from(index));
        }
        Ok(self.ctx.sub.psi(&table)?)
    }

    /// The encoding of a commitment: `[VERSION]`, then the $n_B d$ coefficients of
    /// $\mathbf u$ packed (module documentation, "Encodings"); [`HachiPcs::commitment_words`]
    /// words.
    pub fn commitment_to_words(&self, com: &Commitment) -> Vec<u64> {
        let mut w = BitWriter::default();
        self.put_polys(&mut w, &com.u);
        let mut out = vec![VERSION];
        out.extend(w.words);
        out
    }

    /// The inverse of [`HachiPcs::commitment_to_words`].
    ///
    /// # Errors
    /// [`Error::Shape`] for any input that [`HachiPcs::commitment_to_words`] does not produce
    /// from a commitment of $n_B$ ring elements: another version or length, a residue not below
    /// $q$, or non-zero padding.
    pub fn commitment_from_words(&self, words: &[u64]) -> Result<Commitment> {
        let Some((&version, stream)) = words.split_first() else {
            return Err(Error::Shape("the encoding ends early".into()));
        };
        if version != VERSION {
            return Err(Error::Shape("unknown commitment version".into()));
        }
        if self.commitment_words() != words.len() {
            return Err(Error::Shape(format!(
                "a commitment has {} words",
                self.commitment_words()
            )));
        }
        let mut r = BitReader {
            words: stream,
            bit: 0,
        };
        let u = self.take_polys(&mut r, self.ctx.choices.n_b)?;
        r.finish()?;
        Ok(Commitment { u })
    }

    /// The canonical encoding of a proof made by this scheme's prover (module documentation,
    /// "Encodings"). For any other value it returns some words without panicking, but the
    /// decoder need not give the value back: values are cut to their fields, and a final witness
    /// that is not a table of digits is not kept.
    pub fn proof_to_words(&self, proof: &Proof<K>) -> Vec<u64> {
        let mut out = vec![
            VERSION,
            proof.partials.len() as u64,
            proof.iterations.len() as u64,
        ];
        out.extend(proof.iterations.iter().map(|it| it.rounds.len() as u64));
        out.push(proof.final_message.len() as u64);
        let noise = self.noise_field();
        if noise.is_some() {
            out.push(proof.final_randomness.len() as u64);
        }
        let mut w = BitWriter::default();
        self.put_exts(&mut w, &proof.partials);
        self.put_polys(&mut w, core::slice::from_ref(&proof.ring_value));
        for it in &proof.iterations {
            self.put_polys(&mut w, &it.v);
            self.put_polys(&mut w, &it.next_commitment);
            for round in &it.rounds {
                self.put_exts(&mut w, &round.h0);
                self.put_exts(&mut w, &round.h_alpha);
            }
            self.put_exts(&mut w, core::slice::from_ref(&it.final_eval));
            self.put_exts(&mut w, &it.partials);
            self.put_polys(&mut w, core::slice::from_ref(&it.next_target));
        }
        if proof.iterations.is_empty() {
            self.put_polys(&mut w, &proof.final_message);
        } else {
            let zq = self.ctx.ring.zq();
            let (lo, _) = digit_range(self.ctx.choices.b);
            let blank = vec![Ext([0u64; K]); self.ctx.d_over_k()];
            for p in &proof.final_message {
                let table = self.ctx.sub.psi_inv(p).unwrap_or_else(|_| blank.clone());
                for x in table.iter().flat_map(|e| e.0) {
                    let index = i128::from(zq.to_centered(x)) - lo;
                    w.push(index as u64, self.digit_bits());
                }
            }
        }
        if let Some((lo, _, bits)) = noise {
            let zq = self.ctx.ring.zq();
            for x in proof.final_randomness.iter().flat_map(|p| &p.0) {
                w.push(
                    (i128::from(zq.to_centered(*x)) - i128::from(lo)) as u64,
                    bits,
                );
            }
        }
        out.extend(w.words);
        out
    }

    /// The inverse of [`HachiPcs::proof_to_words`]. The counts of the header (iterations,
    /// rounds, final ring elements, and with hiding the final randomness) are checked against
    /// the schedule by the verifier.
    ///
    /// # Errors
    /// [`Error::Shape`] for any input that [`HachiPcs::proof_to_words`] does not produce from a
    /// proof of this scheme's shape: another version, a number of partial evaluations other
    /// than 0 and $K$, a length other than the one the header implies, a residue not below
    /// $q$, a digit index outside the digit set, a randomness coefficient outside the support
    /// of $\chi$, or non-zero padding.
    pub fn proof_from_words(&self, words: &[u64]) -> Result<Proof<K>> {
        let shape = |what: &str| Error::Shape(what.into());
        let [version, partials, count, rest @ ..] = words else {
            return Err(shape("the encoding ends early"));
        };
        if *version != VERSION {
            return Err(shape("unknown proof version"));
        }
        let partials = match *partials {
            0 => 0,
            p if p == K as u64 => K,
            _ => return Err(shape("a proof has 0 or K partial evaluations")),
        };
        // The header: k round counts and the final count.
        let count = usize::try_from(*count)
            .ok()
            .filter(|k| *k < rest.len())
            .ok_or_else(|| shape("the encoding ends early"))?;
        let (round_counts, rest) = rest.split_at(count);
        let Some((&final_count, stream)) = rest.split_first() else {
            return Err(shape("the encoding ends early"));
        };
        let noise = self.noise_field();
        let (rand_count, stream) = match noise {
            None => (0, stream),
            Some(_) => {
                let Some((&r, stream)) = stream.split_first() else {
                    return Err(shape("the encoding ends early"));
                };
                (r, stream)
            }
        };
        let too_large = || shape("a count in the header is too large");
        let round_counts = round_counts
            .iter()
            .map(|l| usize::try_from(*l).map_err(|_| too_large()))
            .collect::<Result<Vec<_>>>()?;
        let final_count = usize::try_from(final_count).map_err(|_| too_large())?;
        let rand_count = usize::try_from(rand_count).map_err(|_| too_large())?;
        // The length the header implies, checked before anything is allocated.
        let (d, c) = (self.ctx.d(), &self.ctx.choices);
        let round = K * (c.b as usize + 2 + 3);
        let fixed = (c.n_d + c.n_b) * d + K + K * K + d;
        let mut residues = Some(K * partials + d);
        for l in &round_counts {
            residues =
                residues.and_then(|r| l.checked_mul(round)?.checked_add(fixed)?.checked_add(r));
        }
        let entry_bits = if count == 0 {
            self.residue_bits()
        } else {
            self.digit_bits()
        };
        let noise_bits = noise.map_or(0, |(_, _, bits)| bits);
        let bits = residues
            .and_then(|r| r.checked_mul(self.residue_bits()))
            .and_then(|b| {
                final_count
                    .checked_mul(d)?
                    .checked_mul(entry_bits)?
                    .checked_add(b)
            })
            .and_then(|b| {
                rand_count
                    .checked_mul(d)?
                    .checked_mul(noise_bits)?
                    .checked_add(b)
            })
            .ok_or_else(too_large)?;
        if bits.div_ceil(64) != stream.len() {
            return Err(shape("the length does not match the header"));
        }
        let mut r = BitReader {
            words: stream,
            bit: 0,
        };
        let partials = self.take_exts(&mut r, partials)?;
        let ring_value = self.take_poly(&mut r)?;
        let mut iterations = Vec::with_capacity(count);
        for l in round_counts {
            let v = self.take_polys(&mut r, c.n_d)?;
            let next_commitment = self.take_polys(&mut r, c.n_b)?;
            let rounds = (0..l)
                .map(|_| {
                    Ok(RoundMessage {
                        h0: self.take_exts(&mut r, c.b as usize + 2)?,
                        h_alpha: self.take_exts(&mut r, 3)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            iterations.push(IterationProof {
                v,
                next_commitment,
                rounds,
                final_eval: self.take_ext(&mut r)?,
                partials: self.take_exts(&mut r, K)?,
                next_target: self.take_poly(&mut r)?,
            });
        }
        let final_message = if count == 0 {
            self.take_polys(&mut r, final_count)?
        } else {
            (0..final_count)
                .map(|_| self.take_digit_poly(&mut r))
                .collect::<Result<Vec<_>>>()?
        };
        let final_randomness = match noise {
            None => Vec::new(),
            Some((lo, hi, bits)) => {
                let zq = self.ctx.ring.zq();
                let width = (hi - lo) as u64;
                (0..rand_count)
                    .map(|_| {
                        (0..d)
                            .map(|_| {
                                let x = r.take(bits)?;
                                if x > width {
                                    return Err(shape(
                                        "a randomness coefficient outside the support of chi",
                                    ));
                                }
                                Ok(zq.from_i128(i128::from(lo) + i128::from(x)))
                            })
                            .collect::<Result<Vec<_>>>()
                            .map(Poly)
                    })
                    .collect::<Result<Vec<_>>>()?
            }
        };
        r.finish()?;
        Ok(Proof {
            partials,
            ring_value,
            iterations,
            final_message,
            final_randomness,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{BitReader, BitWriter};

    /// The low `width` bits.
    fn low(value: u64, width: usize) -> u64 {
        if width >= 64 {
            value
        } else {
            value & ((1u64 << width) - 1)
        }
    }

    #[test]
    fn fields_of_every_width_round_trip() {
        // 4000 fields of widths 0 to 64 in a pseudorandom order, so that every offset in a
        // word meets every width.
        let mut state = 0x5eed_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state ^ (state >> 29)
        };
        let fields: Vec<(u64, usize)> = (0..4000)
            .map(|_| {
                let width = (next() % 65) as usize;
                (next(), width)
            })
            .collect();
        let mut w = BitWriter::default();
        for (value, width) in &fields {
            w.push(*value, *width);
        }
        let total: usize = fields.iter().map(|(_, width)| width).sum();
        assert_eq!(w.words.len(), total.div_ceil(64));
        let mut r = BitReader {
            words: &w.words,
            bit: 0,
        };
        for (value, width) in &fields {
            assert_eq!(
                r.take(*width).unwrap(),
                low(*value, *width),
                "width {width}"
            );
        }
        r.finish().unwrap();
        let mut end = BitReader {
            words: &w.words,
            bit: 64 * w.words.len(),
        };
        assert!(end.take(1).is_err());
        assert_eq!(end.take(0).unwrap(), 0);
    }

    #[test]
    fn values_are_cut_and_padding_and_trailing_words_are_refused() {
        let mut w = BitWriter::default();
        w.push(u64::MAX, 3);
        w.push(0, 61);
        assert_eq!(w.words, vec![7]);
        w.push(u64::MAX, 64);
        assert_eq!(w.words, vec![7, u64::MAX]);
        // A field across two words.
        let mut w = BitWriter::default();
        w.push(0, 60);
        w.push(0b1_0110_1011, 9);
        assert_eq!(w.words, vec![0b1011 << 60, 0b1_0110]);
        let mut r = BitReader {
            words: &w.words,
            bit: 60,
        };
        assert_eq!(r.take(9).unwrap(), 0b1_0110_1011);
        r.finish().unwrap();
        // Padding bits must be zero, and no word may be left over.
        let mut r = BitReader {
            words: &[1 << 63],
            bit: 0,
        };
        r.take(3).unwrap();
        assert!(r.finish().is_err());
        let r = BitReader {
            words: &[0, 0],
            bit: 64,
        };
        assert!(r.finish().is_err());
        let r = BitReader { words: &[], bit: 0 };
        r.finish().unwrap();
    }
}
