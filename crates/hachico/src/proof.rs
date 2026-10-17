//! The Eval proof and its canonical byte encoding (`docs/m1-spec.md`, WP11).

use crate::error::{DecodeError, ProofError};
use hachi_co_core::{Context, Ext, Layout, Poly};
use hachi_co_r2l::MAX_ATTEMPTS;
use hachi_co_sumcheck::{FinalValues, RoundMessage, SumcheckTranscript};

/// The first 8 bytes of every encoded [`EvalProof`].
pub const MAGIC: [u8; 8] = *b"HCO-EVAL";

/// The version of the encoding, the second 8 bytes of the header (little-endian).
pub const VERSION: u64 = 1;

/// The length of the header: [`MAGIC`] and [`VERSION`].
pub const HEADER_LEN: usize = 16;

// The size formula of `hachi-co-params` counts the same header.
const _: () = assert!(HEADER_LEN as u128 == hachi_co_params::sizes::HEADER_BYTES);
const _: () = assert!(8 == hachi_co_params::sizes::WORD_BYTES);

/// What the provers send to the verifier in $\mathsf{Eval}$ (`6_Collaborative_Hachi.tex`:
/// 408-419), without the base PCS (milestone 1): the public messages of $\Pi\_{\mathsf{conv}}$,
/// of the accepted attempt of the ring-to-linear step and of the sumcheck. The verifier's coins
/// are not part of it: it re-derives them, the challenges $\mathbf c$ included.
///
/// The fields are public; [`EvalProof::validate`] checks them against a context, and the
/// verifier calls it first.
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
    /// The sumcheck transcript (`5_Building_Blocks.tex`:241, 293-300, 314-320): $G\_{0,p},
    /// G\_{1,p}$, the $\ell''$ round messages and the final values of every prover.
    pub sumcheck: SumcheckTranscript<K>,
}

/// The number of bytes of [`EvalProof::to_bytes`] for a proof of `layout`: the header
/// ([`HEADER_LEN`]) and $8(1 + d + n\_D d + k(2n + \ell''(d\_0 + d\_1 + 2) + 3n))$ bytes, one
/// little-endian word for $a^\ast$ and one per residue (a field element has $k$ residues). This
/// is the formula of [`hachi_co_params::sizes`](fn@hachi_co_params::sizes), computed here from
/// the layout.
///
/// # Errors
/// [`DecodeError::Overflow`] if the length does not fit in `usize`.
pub fn encoded_len(layout: &Layout) -> Result<usize, DecodeError> {
    let l = layout;
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or(DecodeError::Overflow);
    let add = |a: usize, b: usize| a.checked_add(b).ok_or(DecodeError::Overflow);
    // Field elements: 2n sums, l'' rounds of (d0 + 1) + (d1 + 1) values, 3n final values.
    let per_round = add(add(l.d0, l.d1)?, 2)?;
    let exts = add(add(mul(2, l.n)?, mul(l.ell_pp, per_round)?)?, mul(3, l.n)?)?;
    // Words: a*, E(u), v and the coordinates of the field elements.
    let words = add(add(add(1, l.d)?, mul(l.n_d, l.d)?)?, mul(exts, l.k)?)?;
    add(HEADER_LEN, mul(words, 8)?)
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
}

impl<const K: usize> EvalProof<K> {
    /// Checks the proof against the context: $a^\ast$ below [`MAX_ATTEMPTS`]; $E(u)$ with $d$
    /// residues; $\mathbf v$ with $n\_D$ ring elements of $d$ residues; $n$ sums $G\_{0,p}$ and
    /// $n$ sums $G\_{1,p}$; $\ell''$ round messages of $d\_0 + 1$ and $d\_1 + 1$ field elements;
    /// $n$ final values of each kind; and every residue below $q$. Exactly the proofs that pass
    /// have an encoding ([`EvalProof::to_bytes`]) that [`EvalProof::from_bytes`] accepts.
    ///
    /// # Errors
    /// The first violation, in the order of the encoding.
    pub fn validate(&self, ctx: &Context<K>) -> Result<(), ProofError> {
        let (l, q) = (&ctx.layout, ctx.q());
        check_attempt(self.attempt)?;
        check_len("E(u) (d residues)", l.d, self.e_u.0.len())?;
        check_canonical("E(u)", &self.e_u.0, q)?;
        check_len("v (n_D ring elements)", l.n_d, self.v.len())?;
        for p in &self.v {
            check_len("v (d residues per entry)", l.d, p.0.len())?;
        }
        check_canonical("v", self.v.iter().flat_map(|p| p.0.iter()), q)?;
        let t = &self.sumcheck;
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

    /// The canonical encoding: the 16-byte header ([`MAGIC`], then [`VERSION`] as a
    /// little-endian `u64`) followed by little-endian `u64` words in this order:
    ///
    /// 1. $a^\ast$;
    /// 2. the $d$ residues of $E(u)$;
    /// 3. the residues of $\mathbf v$, ring element after ring element;
    /// 4. $G\_{0,1}, \dots, G\_{0,n}$, then $G\_{1,1}, \dots, G\_{1,n}$;
    /// 5. for each round $r = 1, \dots, \ell''$: $H\_{0,r}(1), \dots, H\_{0,r}(d\_0 + 1)$, then
    ///    $H\_{1,r}(1), \dots, H\_{1,r}(d\_1 + 1)$;
    /// 6. the final values $\langle \widetilde w \rangle\_p(\mathbf a)$ for $p = 1, \dots, n$, then
    ///    $g\_{0,p}(\mathbf a)$, then $g\_{1,p}(\mathbf a)$;
    ///
    /// where a field element is its $K$ coordinates in Hachi's basis. The fields are written as
    /// they are; for a proof that passes [`EvalProof::validate`] the length is
    /// [`encoded_len`] and [`EvalProof::from_bytes`] returns the proof.
    pub fn to_bytes(&self) -> Vec<u8> {
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
        let words = 1 + self.e_u.0.len() + self.v.iter().map(|p| p.0.len()).sum::<usize>();
        let len = K
            .saturating_mul(exts)
            .saturating_add(words)
            .saturating_mul(8)
            .saturating_add(HEADER_LEN);
        let mut out = Vec::with_capacity(len);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        let mut push = |x: u64| out.extend_from_slice(&x.to_le_bytes());
        push(self.attempt);
        self.e_u.0.iter().for_each(|c| push(*c));
        self.v.iter().flat_map(|p| &p.0).for_each(|c| push(*c));
        let ext_lists = [&t.g0_sums, &t.g1_sums]
            .into_iter()
            .chain(t.rounds.iter().flat_map(|r| [&r.h0, &r.h1]))
            .chain([&t.finals.w, &t.finals.g0, &t.finals.g1]);
        for list in ext_lists {
            list.iter().flat_map(|e| e.0).for_each(&mut push);
        }
        out
    }

    /// Decodes [`EvalProof::to_bytes`] for the context `ctx`, accepting exactly the canonical
    /// encodings of the proofs that pass [`EvalProof::validate`]: the header must be [`MAGIC`] and
    /// [`VERSION`], the length must be [`encoded_len`] of the context, every residue must be below
    /// $q$ and $a^\ast$ below [`MAX_ATTEMPTS`]. Never panics.
    ///
    /// # Errors
    /// [`DecodeError::Truncated`] for fewer bytes than the header or than the context's length;
    /// [`DecodeError::BadMagic`], [`DecodeError::UnsupportedVersion`];
    /// [`DecodeError::TrailingBytes`] for more bytes than the context's length;
    /// [`DecodeError::Invalid`] for a non-canonical residue or an attempt index at or beyond the
    /// cap; [`DecodeError::Overflow`] if the context's length overflows.
    pub fn from_bytes(ctx: &Context<K>, bytes: &[u8]) -> Result<Self, DecodeError> {
        let (l, q) = (&ctx.layout, ctx.q());
        let expected = encoded_len(l)?;
        let actual = bytes.len();
        let truncated = DecodeError::Truncated { expected, actual };
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
        if actual < expected {
            return Err(truncated);
        }
        if actual > expected {
            return Err(DecodeError::TrailingBytes { expected, actual });
        }
        // The body has 8 (expected - HEADER_LEN) / 8 bytes, so nothing remains.
        let (words, _) = body.as_chunks::<8>();
        let mut r = Reader {
            words: words.iter(),
            expected,
            actual,
            q,
        };
        let attempt = r.word()?;
        check_attempt(attempt)?;
        let e_u = r.poly("E(u)", l.d)?;
        let v = (0..l.n_d)
            .map(|_| r.poly("v", l.d))
            .collect::<Result<Vec<_>, _>>()?;
        let g0_sums = r.exts("G_0", l.n)?;
        let g1_sums = r.exts("G_1", l.n)?;
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
        Ok(Self {
            e_u,
            attempt,
            v,
            sumcheck: SumcheckTranscript {
                g0_sums,
                g1_sums,
                rounds,
                finals,
            },
        })
    }
}
