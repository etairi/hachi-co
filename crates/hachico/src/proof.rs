//! The Eval proof and its canonical byte encoding (`docs/m1-spec.md`, WP11), with the base
//! PCS's commitments and opening proofs.

use crate::error::{DecodeError, ProofError};
use hachi_co_core::{Context, Ext, Layout, Poly};
use hachi_co_params::sizes::BasePcsWords;
use hachi_co_r2l::MAX_ATTEMPTS;
use hachi_co_sumcheck::{ClaimedPoly, FinalValues, PcsWords, RoundMessage, SumcheckTranscript};

/// The first 8 bytes of every encoded [`EvalProof`].
pub const MAGIC: [u8; 8] = *b"HCO-EVAL";

/// The version of the encoding, the second 8 bytes of the header (little-endian): 2 since the
/// proof carries the base PCS's commitments and opening proofs (version 1, the encoding without
/// a base PCS, had neither).
pub const VERSION: u64 = 2;

/// The length of the header: [`MAGIC`] and [`VERSION`].
pub const HEADER_LEN: usize = 16;

// The size formula of `hachi-co-params` counts the same header.
const _: () = assert!(HEADER_LEN as u128 == hachi_co_params::sizes::HEADER_BYTES);
const _: () = assert!(8 == hachi_co_params::sizes::WORD_BYTES);

/// The order of the base-PCS items in the encoding: the polynomials in the order of
/// [`PcsWords`] ($\langle \widetilde w \rangle$, $g\_0$, $g\_1$), and the parties in order within
/// each.
const PCS_ORDER: [ClaimedPoly; 3] = [ClaimedPoly::W, ClaimedPoly::G0, ClaimedPoly::G1];

/// What the provers send to the verifier in $\mathsf{Eval}$ (`6_Collaborative_Hachi.tex`:
/// 408-419): the public messages of $\Pi\_{\mathsf{conv}}$, of the accepted attempt of the
/// ring-to-linear step and of the sumcheck, whose transcript carries the base PCS's commitments
/// and opening proofs as words. The verifier's coins are not part of it: it re-derives them, the
/// challenges $\mathbf c$ included.
///
/// The fields are public; [`EvalProof::validate`] checks them against a context, and the
/// verifier calls it first. The words of the base PCS are decoded by the verifier, with the base
/// PCS it is given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalProof<const K: usize> {
    /// $E(u) \in \mathcal R\_q^H$, the output of $\mathcal F\_{\mathsf{AH}}(E)$
    /// (`field-to-ring-rok.tex`:160-164): $d$ residues.
    pub e_u: Poly,
    /// The index $a^\ast$ of the accepted attempt of the ring-to-linear step
    /// (`5_Building_Blocks.tex`:129; Fiat-Shamir with aborts, `ref/README.md`, choice 5), below
    /// [`MAX_ATTEMPTS`].
    pub attempt: u64,
    /// The accepted attempt's first message $\mathbf v \in \mathcal R\_q^{n\_D}$
    /// (`5_Building_Blocks.tex`:120-121).
    pub v: Vec<Poly>,
    /// The sumcheck transcript (`5_Building_Blocks.tex`:220-241, 293-300, 314-320): $G\_{0,p},
    /// G\_{1,p}$, the base-PCS commitments, the $\ell''$ round messages, the final values and the
    /// base-PCS opening proofs of every prover.
    pub sumcheck: SumcheckTranscript<K>,
}

/// `a * b` or an overflow.
fn mul(a: usize, b: usize) -> Result<usize, DecodeError> {
    a.checked_mul(b).ok_or(DecodeError::Overflow)
}

/// `a + b` or an overflow.
fn add(a: usize, b: usize) -> Result<usize, DecodeError> {
    a.checked_add(b).ok_or(DecodeError::Overflow)
}

/// The words of the encoding besides the base PCS's items: $a^\ast$, the $6n$ lengths of the
/// items and the residues, $1 + 6n + d + n\_D d + k(2n + \ell''(d\_0 + d\_1 + 2) + 3n)$.
fn fixed_words(l: &Layout) -> Result<usize, DecodeError> {
    // Field elements: 2n sums, l'' rounds of (d0 + 1) + (d1 + 1) values, 3n final values.
    let per_round = add(add(l.d0, l.d1)?, 2)?;
    let exts = add(add(mul(2, l.n)?, mul(l.ell_pp, per_round)?)?, mul(3, l.n)?)?;
    let head = add(1, mul(6, l.n)?)?;
    add(add(head, add(l.d, mul(l.n_d, l.d)?)?)?, mul(exts, l.k)?)
}

/// The number of bytes of [`EvalProof::to_bytes`] for a proof of `layout` with a base PCS whose
/// items of each prover have the words `base`: the header ([`HEADER_LEN`]) and
/// $8(1 + 6n + d + n\_D d + k(2n + \ell''(d\_0 + d\_1 + 2) + 3n) + n(c\_w + c\_{g\_0} + c\_{g\_1} +
/// \pi\_w + \pi\_{g\_0} + \pi\_{g\_1}))$ bytes: one little-endian word for $a^\ast$, one per
/// length of a base-PCS item, one per residue (a field element has $k$ residues), and the
/// items' own words. This is the formula of [`hachi_co_params::sizes_with_base`], computed here
/// from the layout.
///
/// # Errors
/// [`DecodeError::Overflow`] if the length does not fit in `usize`.
pub fn encoded_len(layout: &Layout, base: &BasePcsWords) -> Result<usize, DecodeError> {
    let fixed = fixed_words(layout)?;
    let per_party = [
        base.commitment_w,
        base.commitment_g0,
        base.commitment_g1,
        base.proof_w,
        base.proof_g0,
        base.proof_g1,
    ]
    .into_iter()
    .try_fold(0u128, u128::checked_add)
    .ok_or(DecodeError::Overflow)?;
    let pcs = usize::try_from(per_party)
        .map_err(|_| DecodeError::Overflow)
        .and_then(|w| mul(w, layout.n))?;
    add(HEADER_LEN, mul(add(fixed, pcs)?, 8)?)
}

fn check_len(what: &'static str, expected: usize, actual: usize) -> Result<(), ProofError> {
    if expected != actual {
        return Err(ProofError::Shape {
            what,
            expected,
            actual,
        });
    }
    Ok(())
}

fn check_canonical<'a>(
    what: &'static str,
    residues: impl IntoIterator<Item = &'a u64>,
    q: u64,
) -> Result<(), ProofError> {
    match residues.into_iter().find(|x| **x >= q) {
        Some(value) => Err(ProofError::NonCanonical {
            what,
            value: *value,
            q,
        }),
        None => Ok(()),
    }
}

/// `len` field elements, each with canonical coordinates.
fn check_exts<const K: usize>(
    what: &'static str,
    values: &[Ext<K>],
    len: usize,
    q: u64,
) -> Result<(), ProofError> {
    check_len(what, len, values.len())?;
    check_canonical(what, values.iter().flat_map(|e| e.0.iter()), q)
}

/// The attempt index must be below the cap.
fn check_attempt(attempt: u64) -> Result<(), ProofError> {
    if attempt >= MAX_ATTEMPTS {
        return Err(ProofError::AttemptOutOfRange {
            attempt,
            cap: MAX_ATTEMPTS,
        });
    }
    Ok(())
}

/// The base-PCS items in the order of the encoding ([`PCS_ORDER`], party by party).
fn pcs_items(words: &PcsWords) -> impl Iterator<Item = &Vec<u64>> {
    PCS_ORDER.into_iter().flat_map(|poly| words.of(poly))
}

/// Reads little-endian words from the body of an encoding.
struct Reader<'a> {
    words: core::slice::Iter<'a, [u8; 8]>,
    expected: usize,
    actual: usize,
    q: u64,
}

impl Reader<'_> {
    fn word(&mut self) -> Result<u64, DecodeError> {
        let chunk = self.words.next().ok_or(DecodeError::Truncated {
            expected: self.expected,
            actual: self.actual,
        })?;
        Ok(u64::from_le_bytes(*chunk))
    }

    fn residue(&mut self, what: &'static str) -> Result<u64, DecodeError> {
        let value = self.word()?;
        if value >= self.q {
            return Err(ProofError::NonCanonical {
                what,
                value,
                q: self.q,
            }
            .into());
        }
        Ok(value)
    }

    fn poly(&mut self, what: &'static str, d: usize) -> Result<Poly, DecodeError> {
        Ok(Poly(
            (0..d)
                .map(|_| self.residue(what))
                .collect::<Result<_, _>>()?,
        ))
    }

    fn exts<const K: usize>(
        &mut self,
        what: &'static str,
        count: usize,
    ) -> Result<Vec<Ext<K>>, DecodeError> {
        (0..count)
            .map(|_| {
                let mut coords = [0u64; K];
                for c in &mut coords {
                    *c = self.residue(what)?;
                }
                Ok(Ext(coords))
            })
            .collect()
    }

    /// The base-PCS items of `n` parties with the given lengths (in the order of
    /// [`PCS_ORDER`]), as they are.
    fn pcs_words(&mut self, n: usize, lengths: &[usize]) -> Result<PcsWords, DecodeError> {
        let mut out = PcsWords::default();
        let mut lengths = lengths.iter();
        for poly in PCS_ORDER {
            for _ in 0..n {
                let len = *lengths.next().ok_or(DecodeError::Overflow)?;
                let words = (0..len)
                    .map(|_| self.word())
                    .collect::<Result<Vec<_>, _>>()?;
                out.of_mut(poly).push(words);
            }
        }
        Ok(out)
    }
}

impl<const K: usize> EvalProof<K> {
    /// Checks the proof against the context: $a^\ast$ below [`MAX_ATTEMPTS`]; $E(u)$ with $d$
    /// residues; $\mathbf v$ with $n\_D$ ring elements of $d$ residues; $n$ sums $G\_{0,p}$ and
    /// $n$ sums $G\_{1,p}$; $n$ base-PCS commitments of each polynomial; $\ell''$ round messages
    /// of $d\_0 + 1$ and $d\_1 + 1$ field elements; $n$ final values of each kind; $n$ base-PCS
    /// opening proofs of each polynomial; and every residue below $q$. The words of the base
    /// PCS are not interpreted here (the verifier decodes them with the base PCS). Exactly the
    /// proofs that pass have an encoding ([`EvalProof::to_bytes`]) that
    /// [`EvalProof::from_bytes`] accepts.
    ///
    /// # Errors
    /// The first violation, in the order of the encoding.
    pub fn validate(&self, ctx: &Context<K>) -> Result<(), ProofError> {
        let (l, q) = (&ctx.layout, ctx.q());
        check_attempt(self.attempt)?;
        let t = &self.sumcheck;
        for poly in PCS_ORDER {
            check_len(
                "base-PCS commitments (one per prover and polynomial)",
                l.n,
                t.commitments.of(poly).len(),
            )?;
            check_len(
                "base-PCS opening proofs (one per prover and polynomial)",
                l.n,
                t.openings.of(poly).len(),
            )?;
        }
        check_len("E(u) (d residues)", l.d, self.e_u.0.len())?;
        check_canonical("E(u)", &self.e_u.0, q)?;
        check_len("v (n_D ring elements)", l.n_d, self.v.len())?;
        for p in &self.v {
            check_len("v (d residues per entry)", l.d, p.0.len())?;
        }
        check_canonical("v", self.v.iter().flat_map(|p| p.0.iter()), q)?;
        check_exts("G_0 (one per prover)", &t.g0_sums, l.n, q)?;
        check_exts("G_1 (one per prover)", &t.g1_sums, l.n, q)?;
        check_len("round messages (l'')", l.ell_pp, t.rounds.len())?;
        for round in &t.rounds {
            check_exts("H_0 (d0 + 1 values)", &round.h0, l.d0.saturating_add(1), q)?;
            check_exts("H_1 (d1 + 1 values)", &round.h1, l.d1.saturating_add(1), q)?;
        }
        check_exts("final <w~>_p(a) (one per prover)", &t.finals.w, l.n, q)?;
        check_exts("final g_0,p(a) (one per prover)", &t.finals.g0, l.n, q)?;
        check_exts("final g_1,p(a) (one per prover)", &t.finals.g1, l.n, q)
    }

    /// The number of bytes of [`EvalProof::to_bytes`] for this proof, counted without encoding
    /// it: the header, $a^\ast$, one length word per base-PCS item, every residue and the words
    /// of the base-PCS items as they are (saturating at `usize::MAX`). For a proof that passes
    /// [`EvalProof::validate`] it is [`encoded_len`] with the words of its base-PCS items.
    pub fn encoded_len(&self) -> usize {
        let t = &self.sumcheck;
        let exts = t.g0_sums.len()
            + t.g1_sums.len()
            + t.rounds
                .iter()
                .map(|r| r.h0.len() + r.h1.len())
                .sum::<usize>()
            + t.finals.w.len()
            + t.finals.g0.len()
            + t.finals.g1.len();
        let items = pcs_items(&t.commitments).count() + pcs_items(&t.openings).count();
        let words = 1usize
            .saturating_add(items)
            .saturating_add(t.commitments.total_words())
            .saturating_add(t.openings.total_words())
            .saturating_add(self.e_u.0.len())
            .saturating_add(self.v.iter().map(|p| p.0.len()).sum::<usize>());
        K.saturating_mul(exts)
            .saturating_add(words)
            .saturating_mul(8)
            .saturating_add(HEADER_LEN)
    }

    /// The canonical encoding: the 16-byte header ([`MAGIC`], then [`VERSION`] as a
    /// little-endian `u64`) followed by little-endian `u64` words in this order:
    ///
    /// 1. $a^\ast$;
    /// 2. the lengths, in words, of the $6n$ base-PCS items of 5 and 8, in their order there
    ///    (the commitments, then the opening proofs), so that the decoder knows the exact length
    ///    of the encoding before reading the rest;
    /// 3. the $d$ residues of $E(u)$; the residues of $\mathbf v$, ring element after ring
    ///    element;
    /// 4. $G\_{0,1}, \dots, G\_{0,n}$, then $G\_{1,1}, \dots, G\_{1,n}$;
    /// 5. the base-PCS commitments to $\langle \widetilde w \rangle\_p$ for $p = 1, \dots, n$,
    ///    then to $g\_{0,p}$, then to $g\_{1,p}$, each as its words
    ///    ([`BasePcs::commitment_to_words`](hachi_co_core::BasePcs::commitment_to_words));
    /// 6. for each round $r = 1, \dots, \ell''$: $H\_{0,r}(1), \dots, H\_{0,r}(d\_0 + 1)$, then
    ///    $H\_{1,r}(1), \dots, H\_{1,r}(d\_1 + 1)$;
    /// 7. the final values $\langle \widetilde w \rangle\_p(\mathbf a)$ for $p = 1, \dots, n$, then
    ///    $g\_{0,p}(\mathbf a)$, then $g\_{1,p}(\mathbf a)$;
    /// 8. the base-PCS opening proofs, in the order of 5
    ///    ([`BasePcs::proof_to_words`](hachi_co_core::BasePcs::proof_to_words));
    ///
    /// where a field element is its $K$ coordinates in Hachi's basis. The fields are written as
    /// they are; for a proof that passes [`EvalProof::validate`] the length is [`encoded_len`]
    /// with the words of its base-PCS items, and [`EvalProof::from_bytes`] returns the proof.
    pub fn to_bytes(&self) -> Vec<u8> {
        let t = &self.sumcheck;
        let mut out = Vec::with_capacity(self.encoded_len());
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        let put = |out: &mut Vec<u8>, words: &mut dyn Iterator<Item = u64>| {
            for w in words {
                out.extend_from_slice(&w.to_le_bytes());
            }
        };
        let exts = |list: &[Ext<K>]| list.iter().flat_map(|e| e.0).collect::<Vec<u64>>();
        put(&mut out, &mut core::iter::once(self.attempt));
        put(
            &mut out,
            &mut pcs_items(&t.commitments)
                .chain(pcs_items(&t.openings))
                .map(|item| item.len() as u64),
        );
        put(&mut out, &mut self.e_u.0.iter().copied());
        put(
            &mut out,
            &mut self.v.iter().flat_map(|p| p.0.iter().copied()),
        );
        put(&mut out, &mut exts(&t.g0_sums).into_iter());
        put(&mut out, &mut exts(&t.g1_sums).into_iter());
        put(
            &mut out,
            &mut pcs_items(&t.commitments).flat_map(|i| i.iter().copied()),
        );
        for round in &t.rounds {
            put(&mut out, &mut exts(&round.h0).into_iter());
            put(&mut out, &mut exts(&round.h1).into_iter());
        }
        for list in [&t.finals.w, &t.finals.g0, &t.finals.g1] {
            put(&mut out, &mut exts(list).into_iter());
        }
        put(
            &mut out,
            &mut pcs_items(&t.openings).flat_map(|i| i.iter().copied()),
        );
        out
    }

    /// Decodes [`EvalProof::to_bytes`] for the context `ctx`, accepting exactly the canonical
    /// encodings of the proofs that pass [`EvalProof::validate`]: the header must be [`MAGIC`] and
    /// [`VERSION`]; the length must be [`encoded_len`] of the context with the lengths of the
    /// base-PCS items written in the encoding (read first, so that the length is checked before
    /// the rest is read); every residue must be below $q$ and $a^\ast$ below [`MAX_ATTEMPTS`].
    /// The base PCS's words are taken as they are (the verifier decodes them). Never panics,
    /// and allocates no more than the encoding's length.
    ///
    /// # Errors
    /// [`DecodeError::Truncated`] for fewer bytes than the header and the lengths of the
    /// base-PCS items (then `expected` is the length with empty items), or than the length they
    /// imply; [`DecodeError::BadMagic`], [`DecodeError::UnsupportedVersion`];
    /// [`DecodeError::TrailingBytes`] for more bytes than that length;
    /// [`DecodeError::Invalid`] for a non-canonical residue or an attempt index at or beyond the
    /// cap; [`DecodeError::Overflow`] if the length overflows.
    pub fn from_bytes(ctx: &Context<K>, bytes: &[u8]) -> Result<Self, DecodeError> {
        let (l, q) = (&ctx.layout, ctx.q());
        let shortest = encoded_len(l, &BasePcsWords::default())?;
        let actual = bytes.len();
        let truncated = DecodeError::Truncated {
            expected: shortest,
            actual,
        };
        let Some((header, body)) = bytes.split_at_checked(HEADER_LEN) else {
            return Err(truncated);
        };
        let (&[magic, version], _) = header.as_chunks::<8>() else {
            return Err(truncated);
        };
        if magic != MAGIC {
            return Err(DecodeError::BadMagic);
        }
        let version = u64::from_le_bytes(version);
        if version != VERSION {
            return Err(DecodeError::UnsupportedVersion(version));
        }
        let (words, _) = body.as_chunks::<8>();
        let items = mul(6, l.n)?;
        let Some(table) = words.get(1..add(1, items)?) else {
            return Err(truncated);
        };
        let lengths = table
            .iter()
            .map(|w| usize::try_from(u64::from_le_bytes(*w)).map_err(|_| DecodeError::Overflow))
            .collect::<Result<Vec<_>, _>>()?;
        let pcs = lengths.iter().try_fold(0usize, |acc, x| add(acc, *x))?;
        let expected = add(shortest, mul(pcs, 8)?)?;
        if actual < expected {
            return Err(DecodeError::Truncated { expected, actual });
        }
        if actual > expected {
            return Err(DecodeError::TrailingBytes { expected, actual });
        }
        // The body has 8 (expected - HEADER_LEN) / 8 bytes, so nothing remains.
        let mut r = Reader {
            words: words.iter(),
            expected,
            actual,
            q,
        };
        let attempt = r.word()?;
        check_attempt(attempt)?;
        for _ in 0..items {
            r.word()?;
        }
        let e_u = r.poly("E(u)", l.d)?;
        let v = (0..l.n_d)
            .map(|_| r.poly("v", l.d))
            .collect::<Result<Vec<_>, _>>()?;
        let g0_sums = r.exts("G_0", l.n)?;
        let g1_sums = r.exts("G_1", l.n)?;
        let (com_lengths, proof_lengths) = lengths.split_at(3 * l.n);
        let commitments = r.pcs_words(l.n, com_lengths)?;
        let rounds = (0..l.ell_pp)
            .map(|_| {
                Ok(RoundMessage {
                    h0: r.exts("H_0", l.d0 + 1)?,
                    h1: r.exts("H_1", l.d1 + 1)?,
                })
            })
            .collect::<Result<Vec<_>, DecodeError>>()?;
        let finals = FinalValues {
            w: r.exts("final <w~>_p(a)", l.n)?,
            g0: r.exts("final g_0,p(a)", l.n)?,
            g1: r.exts("final g_1,p(a)", l.n)?,
        };
        let openings = r.pcs_words(l.n, proof_lengths)?;
        Ok(Self {
            e_u,
            attempt,
            v,
            sumcheck: SumcheckTranscript {
                g0_sums,
                g1_sums,
                commitments,
                rounds,
                finals,
                openings,
            },
        })
    }
}
