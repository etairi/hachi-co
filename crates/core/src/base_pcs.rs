//! The interface to the base polynomial commitment scheme of the collaborative sumcheck
//! (`5_Building_Blocks.tex`:209-241, 314-333), and the conversions around it.
//!
//! Each prover commits, on its own, to three polynomials before the sumcheck's coins and opens
//! them at the sumcheck's point $\mathbf a$ after the last round (the authors' answers of
//! 2026-10-21, plan `docs/hachi-co-plan.md`, §0):
//! - its share polynomial $\langle \widetilde w \rangle\_p$, multilinear in $\ell''$ variables
//!   with $\mathbb Z\_q$ coefficients, defined by its values on the hypercube
//!   (`5_Building_Blocks.tex`:223-229): the values are converted to monomial coefficients
//!   ([`mobius`]), committed with [`BasePcs::commit_multilinear`] and opened as a multilinear
//!   evaluation ([`BasePcs::prove_multilinear`]);
//! - its blinders $g\_{0,p}$ and $g\_{1,p}$ (lines 209-214): their coefficient vectors over
//!   $\mathbb F\_{q^k}$ are committed with [`BasePcs::commit_vector`] and opened as inner products
//!   with public weights ([`BasePcs::prove_inner_product`]).
//!
//! The trait is implemented by `ClearPcs` (feature `fixtures`), a reference stand-in that
//! reveals the committed vector, and by an adapter around Hachi.

use crate::data::check_len;
use crate::{Error, Result};
use core::fmt::Debug;
use hachi_co_math::{Ext, Fqk, Zq};

/// A base polynomial commitment scheme over $\mathbb F\_{q^K}$, as the collaborative sumcheck
/// uses it (`5_Building_Blocks.tex`:209-241, 314-333): every prover commits on its own, before
/// the sumcheck's coins, and opens at the public point $\mathbf a$ (or against public weights
/// derived from it) after the last round; the verifier checks every opening against the final
/// value the provers announced.
///
/// **Coefficient convention.** A multilinear polynomial $f \in \mathbb Z\_q\[X\_1, \dots,
/// X\_\ell\]$ is given by its $2^\ell$ monomial coefficients: `coeffs[i]` multiplies
/// $X\_1^{i\_1} \cdots X\_\ell^{i\_\ell}$, where $(i\_1, \dots, i\_\ell) = \mathsf{bit}(i)$ is
/// MSB-first, so $X\_1$ goes with the most significant bit of $i$:
/// $$f(\mathbf a) = \sum\_{i \lt 2^\ell} \mathtt{coeffs}\[i\] \prod\_{j=1}^{\ell} a\_j^{i\_j}$$
/// ([`evaluate_multilinear`]). This is the `nat` convention of Hachi-Co's hypercube
/// (`ref/README.md`, "Hypercube"; `4_preliminaries.tex`:44): the point $\mathbf x \in \lbrace 0,
/// 1 \rbrace^\ell$ has index $\mathrm{nat}(\mathbf x) = \sum\_j x\_j 2^{\ell - j}$, and
/// [`mobius`] turns the table of the values $f(\mathbf x)$, at index $\mathrm{nat}(\mathbf x)$,
/// into these coefficients. A vector $\mathbf v \in \mathbb F\_{q^K}^{len}$ is opened against
/// public weights $\mathbf e$ as $\langle \mathbf v, \mathbf e \rangle = \sum\_j v\_j e\_j$
/// ([`inner_product`]). Residues are canonical, in $\[0, q)$; a field element is its $K$
/// coordinates in Hachi's basis ([`Ext`]).
///
/// **Contexts.** Every opening is made and checked under a 32-byte `context`. The caller
/// derives it from its Fiat-Shamir transcript ([`Challenger::context`](crate::Challenger::context))
/// with a label that names the instance (in the sumcheck: the party and the polynomial), so that
/// the prover and the verifier pass the same value, and the openings of different instances, or
/// of different proofs, have different contexts. A scheme with its own Fiat-Shamir transcript
/// starts it from the context (before absorbing the commitment, the point or the weights and the
/// value), which binds each opening proof to its instance and to the outer transcript: a proof
/// made under one context must not verify under another. A scheme without coins does not need
/// it for soundness; `ClearPcs` checks it anyway, so that the tests catch a context mismatch.
///
/// **Randomness.** The provers' methods take a secret 32-byte `seed` for the scheme's own
/// randomness (a hiding commitment's, `docs/hachi-hiding.md` 6.5): each commitment and each opening
/// needs a fresh seed, which the sumcheck draws from
/// [`PartyRandomness::base_pcs`](crate::PartyRandomness::base_pcs). A deterministic scheme, such
/// as `ClearPcs` or Hachi without hiding, ignores it.
///
/// **Words.** Commitments and proofs travel as `u64` words: through
/// [`hachi_co_mpc::Comm::all_gather`], in the Fiat-Shamir transcript (the commitments) and in
/// the proof encoding. The encodings belong to an instance of the scheme, so that they can use
/// its parameters (Hachi packs residues in $\lceil \log\_2 q \rceil$ bits). On one instance,
/// `commitment_from_words` and `proof_from_words` must give back every commitment and proof the
/// instance makes from its encoding, accept at most one encoding of any value (whatever they
/// decode re-encodes to the same words), and return an error, never panic, on any input they
/// cannot decode: the verifier decodes words it reads from a proof. A decoder may accept values
/// of a shape that no statement has (Hachi's checks only the lengths its header implies); the
/// verify methods reject those. The provers publish their
/// commitments and their proofs with one `all_gather` call each, which needs the same length at
/// every party: the number of words of a commitment or of a proof must depend only on the size
/// of the committed polynomial (as for `ClearPcs`).
///
/// **Errors.** The provers' methods return an [`Error`] (for example [`Error::Shape`] or
/// [`Error::BasePcs`]); the verifier's methods return the reason of a rejection as a `String`,
/// which the caller reports together with the party and the polynomial. No method panics on
/// malformed input.
pub trait BasePcs<const K: usize>: Send + Sync {
    /// A commitment: public, sent to the verifier and absorbed before the sumcheck's coins
    /// (`5_Building_Blocks.tex`:220-221, 230, 241).
    type Commitment: Clone + Debug + PartialEq + Eq + Send + Sync;
    /// What a prover keeps after committing, to open later (private).
    type State: Send + Sync;
    /// An opening proof (`5_Building_Blocks.tex`:316-318).
    type Proof: Clone + Debug + PartialEq + Eq + Send + Sync;

    /// Commits to the multilinear polynomial with the monomial coefficients `coeffs`
    /// ($2^\ell$ residues of $\mathbb Z\_q$, in the convention of the trait), with the secret
    /// `seed` (see "Randomness").
    ///
    /// # Errors
    /// Implementation-specific, for example [`Error::Shape`] unless the length is a power of
    /// two.
    fn commit_multilinear(
        &self,
        coeffs: &[u64],
        seed: &[u8; 32],
    ) -> Result<(Self::Commitment, Self::State)>;

    /// Commits to the vector `v` over $\mathbb F\_{q^K}$ (a blinder's coefficients), with the
    /// secret `seed`.
    ///
    /// # Errors
    /// Implementation-specific.
    fn commit_vector(
        &self,
        v: &[Ext<K>],
        seed: &[u8; 32],
    ) -> Result<(Self::Commitment, Self::State)>;

    /// Opens a commitment of [`commit_multilinear`](Self::commit_multilinear) at `point`
    /// ($\ell$ coordinates) under `context`, with the secret `seed`: returns $f(\mathbf a)$ and
    /// a proof of it.
    ///
    /// # Errors
    /// Implementation-specific, for example a state of a vector commitment or a point of the
    /// wrong length.
    fn prove_multilinear(
        &self,
        st: &Self::State,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> Result<(Ext<K>, Self::Proof)>;

    /// Opens a commitment of [`commit_vector`](Self::commit_vector) against `weights` (one per
    /// entry) under `context`, with the secret `seed`: returns $\langle \mathbf v, \mathbf e
    /// \rangle$ and a proof of it.
    ///
    /// # Errors
    /// Implementation-specific, for example a state of a multilinear commitment or weights of
    /// the wrong length.
    fn prove_inner_product(
        &self,
        st: &Self::State,
        weights: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> Result<(Ext<K>, Self::Proof)>;

    /// Checks that `proof` opens `com`, a commitment to a multilinear polynomial in `num_vars`
    /// variables, to `value` at `point`, under `context`.
    ///
    /// # Errors
    /// The reason of the rejection.
    fn verify_multilinear(
        &self,
        com: &Self::Commitment,
        num_vars: usize,
        point: &[Ext<K>],
        value: &Ext<K>,
        proof: &Self::Proof,
        context: &[u8; 32],
    ) -> core::result::Result<(), String>;

    /// Checks that `proof` opens `com`, a commitment to a vector of `len` elements, to the
    /// inner product `value` with `weights`, under `context`.
    ///
    /// # Errors
    /// The reason of the rejection.
    fn verify_inner_product(
        &self,
        com: &Self::Commitment,
        len: usize,
        weights: &[Ext<K>],
        value: &Ext<K>,
        proof: &Self::Proof,
        context: &[u8; 32],
    ) -> core::result::Result<(), String>;

    /// A digest of the scheme's own public parameters. Hachi-Co absorbs it into the statement
    /// of every proof (`6_Collaborative_Hachi.tex`:289-292: $\mathsf{pp}\_{\mathsf{PCS}}$ is part
    /// of $\mathsf{pp}$), so that a proof is bound to the base PCS it was made with. Two
    /// instances with different parameters must have different digests.
    fn params_digest(&self) -> [u8; 32];

    /// The words of a commitment (see "Words").
    fn commitment_to_words(&self, com: &Self::Commitment) -> Vec<u64>;

    /// The commitment of `words`, the inverse of
    /// [`commitment_to_words`](Self::commitment_to_words).
    ///
    /// # Errors
    /// For any input that `commitment_to_words` does not produce.
    fn commitment_from_words(&self, words: &[u64]) -> Result<Self::Commitment>;

    /// The words of a proof (see "Words").
    fn proof_to_words(&self, p: &Self::Proof) -> Vec<u64>;

    /// The proof of `words`, the inverse of [`proof_to_words`](Self::proof_to_words).
    ///
    /// # Errors
    /// For any input that `proof_to_words` does not produce.
    fn proof_from_words(&self, words: &[u64]) -> Result<Self::Proof>;
}

/// Checks that a table of a multilinear polynomial has $2^\ell$ entries for some $\ell \ge 0$.
fn check_power_of_two(what: &'static str, len: usize) -> Result<()> {
    if !len.is_power_of_two() {
        return Err(Error::Shape {
            what,
            expected: len.checked_next_power_of_two().unwrap_or(0),
            actual: len,
        });
    }
    Ok(())
}

/// One butterfly pass per variable: `op(low, high)` replaces `high` for every pair of entries
/// that differ only in one bit, the entry with the bit unset being `low`.
fn butterflies(table: &mut [u64], op: impl Fn(u64, u64) -> u64) {
    let mut half = 1;
    while half < table.len() {
        for chunk in table.chunks_exact_mut(2 * half) {
            let (low, high) = chunk.split_at_mut(half);
            for (l, h) in low.iter().zip(high.iter_mut()) {
                *h = op(*l, *h);
            }
        }
        half *= 2;
    }
}

/// The Möbius transform over $\mathbb Z\_q$, in place: the values $f(\mathbf x)$ of a multilinear
/// polynomial $f$ on $\lbrace 0, 1 \rbrace^\ell$, at index $\mathrm{nat}(\mathbf x)$, become its
/// monomial coefficients in the convention of [`BasePcs`]. With $S(i)$ the set of positions of
/// the bits of $i$,
/// $$c\_i = \sum\_{x \,:\, S(x) \subseteq S(i)} (-1)^{\lvert S(i) \rvert - \lvert S(x) \rvert} f(x),$$
/// the inverse of $f(x) = \sum\_{i \,:\, S(i) \subseteq S(x)} c\_i$ ([`zeta`]): the monomial
/// $\prod\_j X\_j^{i\_j}$ is 1 at a point of the hypercube if every variable in it is 1 there,
/// and 0 otherwise. The bits of the index play the same role in values and coefficients, so the
/// MSB-first convention carries over. $\ell 2^{\ell - 1}$ subtractions; every entry is reduced
/// modulo $q$. Hachi-Co converts the table of $\langle \widetilde w \rangle\_p$
/// (`5_Building_Blocks.tex`:223-229) with it before committing (the authors' answer 2 of
/// 2026-10-21, plan `docs/hachi-co-plan.md`, §0).
///
/// # Errors
/// [`Error::Shape`] unless the length is a power of two.
pub fn mobius(zq: &Zq, table: &mut [u64]) -> Result<()> {
    check_power_of_two("hypercube table (2^l entries)", table.len())?;
    for x in table.iter_mut() {
        *x = zq.reduce(*x);
    }
    butterflies(table, |low, high| zq.sub(high, low));
    Ok(())
}

/// The zeta transform over $\mathbb Z\_q$, in place: the inverse of [`mobius`], from the
/// monomial coefficients of a multilinear polynomial to its values on the hypercube,
/// $f(x) = \sum\_{i \,:\, S(i) \subseteq S(x)} c\_i$. Every entry is reduced modulo $q$.
///
/// # Errors
/// [`Error::Shape`] unless the length is a power of two.
pub fn zeta(zq: &Zq, table: &mut [u64]) -> Result<()> {
    check_power_of_two("monomial coefficients (2^l entries)", table.len())?;
    for x in table.iter_mut() {
        *x = zq.reduce(*x);
    }
    butterflies(table, |low, high| zq.add(high, low));
    Ok(())
}

/// $f(\mathbf a) = \sum\_i \mathtt{coeffs}\[i\] \prod\_j a\_j^{i\_j}$ for the multilinear polynomial
/// with the monomial coefficients `coeffs` (the convention of [`BasePcs`]) at a point of
/// $\mathbb F\_{q^K}^\ell$. Binds $X\_1$ first: $f = f\_0 + X\_1 f\_1$, where $f\_0$ and $f\_1$
/// are given by the first and the second half of the coefficients. $2^\ell$ multiplications.
///
/// # Errors
/// [`Error::Shape`] unless `coeffs` has $2^\ell$ entries for a point of $\ell$ coordinates;
/// [`Error::Overflow`] if $2^\ell$ does not fit in `usize`.
pub fn evaluate_multilinear<const K: usize>(
    field: &Fqk<K>,
    coeffs: &[u64],
    point: &[Ext<K>],
) -> Result<Ext<K>> {
    let size = u32::try_from(point.len())
        .ok()
        .and_then(|e| 1usize.checked_shl(e))
        .ok_or(Error::Overflow("2^l coefficients"))?;
    check_len(
        "multilinear coefficients (2^l for a point of l coordinates)",
        size,
        coeffs.len(),
    )?;
    let Some((first, rest)) = point.split_first() else {
        return Ok(field.from_base(coeffs[0]));
    };
    let (low, high) = coeffs.split_at(size / 2);
    let mut table: Vec<Ext<K>> = low
        .iter()
        .zip(high)
        .map(|(c0, c1)| field.add(&field.from_base(*c0), &field.scale(first, *c1)))
        .collect();
    for a in rest {
        let half = table.len() / 2;
        let (low, high) = table.split_at_mut(half);
        for (l, h) in low.iter_mut().zip(high.iter()) {
            *l = field.add(l, &field.mul(a, h));
        }
        table.truncate(half);
    }
    Ok(table[0])
}

/// $\langle \mathbf v, \mathbf e \rangle = \sum\_j v\_j e\_j$ over $\mathbb F\_{q^K}$.
///
/// # Errors
/// [`Error::Shape`] unless the vectors have the same length.
pub fn inner_product<const K: usize>(
    field: &Fqk<K>,
    v: &[Ext<K>],
    weights: &[Ext<K>],
) -> Result<Ext<K>> {
    check_len(
        "inner product (vectors of equal length)",
        v.len(),
        weights.len(),
    )?;
    Ok(v.iter().zip(weights).fold(field.zero(), |acc, (x, e)| {
        field.add(&acc, &field.mul(x, e))
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The toy-n2 modulus.
    const Q: u64 = 33554501;

    /// A splitmix64 stream (seeded per test).
    struct Stream(u64);

    impl Stream {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn residues(&mut self, len: usize) -> Vec<u64> {
            (0..len).map(|_| self.next() % Q).collect()
        }

        fn ext(&mut self) -> Ext<4> {
            Ext(core::array::from_fn(|_| self.next() % Q))
        }
    }

    /// Bit $j$ ($1$-based, MSB-first) of an index of $\ell$ bits.
    fn bit(i: usize, j: usize, ell: usize) -> bool {
        (i >> (ell - j)) & 1 == 1
    }

    /// The Möbius transform from its definition, $c\_i = \sum\_{S(x) \subseteq S(i)}
    /// (-1)^{|S(i)| - |S(x)|} f(x)$, and the zeta transform as the subset sums, at every
    /// $\ell \le 6$; the two are inverse; the transforms of an unreduced table are those of its
    /// residues.
    #[test]
    fn mobius_and_zeta_match_brute_force() {
        let zq = Zq::new(Q).unwrap();
        let mut rng = Stream(0x4d6f_6269_7573);
        for ell in 0..=6usize {
            let size = 1usize << ell;
            let values = rng.residues(size);
            let mut coeffs = values.clone();
            mobius(&zq, &mut coeffs).unwrap();
            for (i, c) in coeffs.iter().enumerate() {
                let mut want = 0u64;
                for (x, v) in values.iter().enumerate() {
                    if x & !i == 0 {
                        let sign = (i.count_ones() - x.count_ones()) % 2;
                        want = if sign == 0 {
                            zq.add(want, *v)
                        } else {
                            zq.sub(want, *v)
                        };
                    }
                }
                assert_eq!(*c, want, "l = {ell}, i = {i}");
            }
            let mut back = coeffs.clone();
            zeta(&zq, &mut back).unwrap();
            assert_eq!(back, values, "l = {ell}");
            for (x, v) in values.iter().enumerate() {
                let sum = coeffs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i & !x == 0)
                    .fold(0, |acc, (_, c)| zq.add(acc, *c));
                assert_eq!(sum, *v, "l = {ell}, x = {x}");
            }
            let mut unreduced: Vec<u64> = values.iter().map(|v| v + Q).collect();
            mobius(&zq, &mut unreduced).unwrap();
            assert_eq!(unreduced, coeffs);
        }
    }

    /// The coefficients of the Möbius transform evaluate (in the convention of the trait) to the
    /// multilinear extension of the values: $f(\mathbf a) = \sum\_x \widetilde{eq}(\mathbf a,
    /// \mathbf x) f(\mathbf x)$, with $\widetilde{eq}$ as a product over MSB-first bits; at a
    /// hypercube point, the value of the table; and [`evaluate_multilinear`] against the sum
    /// over the monomials with every power computed separately.
    #[test]
    fn coefficients_evaluate_to_the_multilinear_extension() {
        let zq = Zq::new(Q).unwrap();
        let field = Fqk::<4>::new(Q).unwrap();
        let mut rng = Stream(0x6576_616c);
        for ell in 0..=6usize {
            let size = 1usize << ell;
            let values = rng.residues(size);
            let mut coeffs = values.clone();
            mobius(&zq, &mut coeffs).unwrap();
            let point: Vec<Ext<4>> = (0..ell).map(|_| rng.ext()).collect();
            let one = field.one();
            let mle = values.iter().enumerate().fold(field.zero(), |acc, (x, v)| {
                let eq = (1..=ell).fold(one, |e, j| {
                    let a = &point[j - 1];
                    let factor = if bit(x, j, ell) {
                        *a
                    } else {
                        field.sub(&one, a)
                    };
                    field.mul(&e, &factor)
                });
                field.add(&acc, &field.scale(&eq, *v))
            });
            let monomials = coeffs.iter().enumerate().fold(field.zero(), |acc, (i, c)| {
                let term = (1..=ell)
                    .filter(|j| bit(i, *j, ell))
                    .fold(field.from_base(*c), |t, j| field.mul(&t, &point[j - 1]));
                field.add(&acc, &term)
            });
            let fast = evaluate_multilinear(&field, &coeffs, &point).unwrap();
            assert_eq!(fast, mle, "l = {ell}");
            assert_eq!(fast, monomials, "l = {ell}");
            // At a hypercube point: the table entry, with X_1 the most significant bit.
            if ell > 0 {
                let x = size - 2;
                let corner: Vec<Ext<4>> = (1..=ell)
                    .map(|j| field.from_base(u64::from(bit(x, j, ell))))
                    .collect();
                assert_eq!(
                    evaluate_multilinear(&field, &coeffs, &corner).unwrap(),
                    field.from_base(values[x])
                );
            }
        }
    }

    #[test]
    fn shapes_are_checked() {
        let zq = Zq::new(Q).unwrap();
        let field = Fqk::<4>::new(Q).unwrap();
        for len in [0usize, 3, 5, 6, 12] {
            let mut t = vec![1u64; len];
            assert!(
                matches!(mobius(&zq, &mut t), Err(Error::Shape { actual, .. }) if actual == len)
            );
            assert!(matches!(zeta(&zq, &mut t), Err(Error::Shape { .. })));
        }
        let mut one = vec![Q + 5];
        mobius(&zq, &mut one).unwrap();
        assert_eq!(one, vec![5]);
        let point = [field.one(); 2];
        assert!(evaluate_multilinear(&field, &[1, 2, 3], &point).is_err());
        assert!(evaluate_multilinear(&field, &[1, 2, 3, 4, 5, 6, 7, 8], &point).is_err());
        assert_eq!(
            evaluate_multilinear(&field, &[7], &[]).unwrap(),
            field.from_base(7)
        );
        let long = vec![field.one(); 64];
        assert_eq!(
            evaluate_multilinear(&field, &[1], &long),
            Err(Error::Overflow("2^l coefficients"))
        );
        assert!(inner_product(&field, &[field.one()], &[]).is_err());
        let v = [Ext([1, 2, 3, 4]), Ext([5, 6, 7, 8])];
        let e = [Ext([2, 0, 0, 0]), Ext([0, 1, 0, 0])];
        let want = field.add(&field.mul(&v[0], &e[0]), &field.mul(&v[1], &e[1]));
        assert_eq!(inner_product(&field, &v, &e).unwrap(), want);
        assert_eq!(inner_product::<4>(&field, &[], &[]).unwrap(), field.zero());
    }
}
