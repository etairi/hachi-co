//! $\mathcal R_q^H$, $\mathrm{Tr}_H$, $\psi$, $\psi^{-1}$ and $E$ against their definitions
//! (draft `4_preliminaries.tex`:69-84, `field-to-ring-rok.tex`:132-164, Hachi Lemma 5 and
//! Theorem 2): the trace as a sum of automorphisms over $H$, membership as invariance under $H$,
//! $\psi$ as a sum of products with monomials, $\psi^{-1}$ by the trace formula, $E$ by the
//! schoolbook product and the summed trace; Theorem 2; $\psi^{-1} \circ \psi = \mathrm{id}$ and
//! $\psi \circ \psi^{-1} = \mathrm{id}$; $E(\psi(\mathbf a)) = \langle \mathbf a,
//! \mathrm{monomials}(\mathbf x) \rangle$; the matrix of $E$ (seeds 50-59).
mod common;

use common::*;
use hachi_co_math::{Error, Ext, Poly, Ring, Subfield};
use rand::RngExt;

fn random_ext<const K: usize>(rng: &mut rand_chacha::ChaCha8Rng, q: u64) -> Ext<K> {
    Ext(core::array::from_fn(|_| rng.random_range(0..q)))
}

/// $\mathrm{Tr}_H(a) = \sum_{h \in H} \sigma_h(a)$, with $H$ recomputed here as
/// $\lbrace \pm (4K\alpha + 1) \bmod 2d \rbrace$ (Hachi, proof of Theorem 2, first claim).
fn trace_by_definition(ring: &Ring, k: usize, a: &Poly) -> Poly {
    let d = ring.degree();
    let mut out = ring.zero();
    for h in h_by_formula(d, k) {
        out = ring
            .add(&out, &ring.automorphism(a, h as i64).unwrap())
            .unwrap();
    }
    out
}

fn h_by_formula(d: usize, k: usize) -> Vec<usize> {
    let mut h: Vec<usize> = (0..d / (2 * k))
        .flat_map(|alpha| {
            let x = (4 * k * alpha + 1) % (2 * d);
            [x, (2 * d - x) % (2 * d)]
        })
        .collect();
    h.sort_unstable();
    h.dedup();
    h
}

fn inner<const K: usize>(s: &Subfield<K>, a: &[Ext<K>], b: &[Ext<K>]) -> Ext<K> {
    let f = s.field();
    a.iter()
        .zip(b)
        .fold(f.zero(), |acc, (x, y)| f.add(&acc, &f.mul(x, y)))
}

fn check_subfield<const K: usize>(q: u64, d: usize, seed: u64) {
    let mut rng = rng(seed);
    let ring = Ring::new(q, d).unwrap();
    let s = Subfield::<K>::new(&ring).unwrap();
    let f = s.field();
    let zq = *ring.zq();
    let full = d <= 256;
    // H.
    let h = s.h_exponents();
    assert_eq!(h.len(), d / K);
    assert_eq!(h, h_by_formula(d, K).as_slice());
    for x in h {
        for g in [2 * d - 1, (4 * K + 1) % (2 * d)] {
            assert!(
                h.contains(&(x * g % (2 * d))),
                "closed under the generators"
            );
        }
    }
    // Embedding, extraction and membership as invariance under H.
    let x: Ext<K> = random_ext(&mut rng, q);
    let ex = s.embed(&x);
    assert_eq!(s.extract(&ex).unwrap(), x);
    let u = Poly(residues(&mut rng, q, d));
    let fixed = |a: &Poly| {
        h.iter()
            .all(|e| ring.automorphism(a, *e as i64).unwrap() == *a)
    };
    let candidates = [
        ex.clone(),
        u.clone(),
        s.trace(&u).unwrap(),
        ring.add(&u, &ring.automorphism(&u, -1).unwrap()).unwrap(),
        ring.zero(),
        ring.monomial(d as i64 / 2),
        ring.monomial((d / (2 * K)) as i64),
    ];
    for a in &candidates {
        assert_eq!(
            s.is_in_subfield(a),
            fixed(a),
            "q {q} d {d} K {K}: membership = invariance"
        );
        assert_eq!(s.extract(a).is_ok(), fixed(a));
    }
    assert!(!s.is_in_subfield(&u) || d / K == 1);
    if d / K > 1 {
        assert_eq!(s.extract(&u), Err(Error::NotInSubfield));
    }
    // Field products are products in R_q.
    let y: Ext<K> = random_ext(&mut rng, q);
    assert_eq!(
        s.embed(&f.mul(&x, &y)),
        ring.mul(&ex, &s.embed(&y)).unwrap()
    );
    // The trace.
    let worst = Poly(vec![(q - 1) / 2; d]);
    for a in [&u, &worst, &ex] {
        assert_eq!(
            s.trace(a).unwrap(),
            trace_by_definition(&ring, K, a),
            "trace q {q} d {d} K {K}"
        );
    }
    // Theorem 2 with the trace by definition: Tr(psi(a) sigma_{-1}(psi(b))) = (d/K) <a, b>.
    let a: Vec<Ext<K>> = (0..d / K).map(|_| random_ext(&mut rng, q)).collect();
    let b: Vec<Ext<K>> = (0..d / K).map(|_| random_ext(&mut rng, q)).collect();
    let (pa, pb) = (s.psi(&a).unwrap(), s.psi(&b).unwrap());
    let product = ring.mul(&pa, &ring.automorphism(&pb, -1).unwrap()).unwrap();
    let lhs = trace_by_definition(&ring, K, &product);
    let rhs = s.embed(&f.scale(&inner(&s, &a, &b), (d / K) as u64));
    assert_eq!(lhs, rhs, "Theorem 2 q {q} d {d} K {K}");
    assert_eq!(s.trace(&product).unwrap(), rhs);
    // psi by its definition: sum_i a_i X^i + X^{d/2} sum_i a_{s+i} X^i.
    let half = d / (2 * K);
    let mut want = ring.zero();
    for i in 0..half {
        let low = ring.mul_monomial(&s.embed(&a[i]), i as i64).unwrap();
        let high = ring
            .mul_monomial(&s.embed(&a[half + i]), (d / 2 + i) as i64)
            .unwrap();
        want = ring.add(&ring.add(&want, &low).unwrap(), &high).unwrap();
    }
    assert_eq!(pa, want, "psi q {q} d {d} K {K}");
    // Round trips, both ways.
    assert_eq!(s.psi_inv(&pa).unwrap(), a);
    assert_eq!(s.psi(&s.psi_inv(&u).unwrap()).unwrap(), u);
    assert_eq!(s.psi(&s.psi_inv(&worst).unwrap()).unwrap(), worst);
    // psi^{-1} by the trace formula: (K/d) Tr_H(u X^{-e_i}).
    let k_over_d = zq.mul(K as u64, zq.inv(d as u64).unwrap());
    let psi_inv_u = s.psi_inv(&u).unwrap();
    let indices: Vec<usize> = if full {
        (0..d / K).collect()
    } else {
        vec![0, 1, half - 1, half, d / K - 1]
    };
    for i in indices {
        let e = if i < half { i } else { d / 2 + (i - half) } as i64;
        let t = trace_by_definition(&ring, K, &ring.mul_monomial(&u, -e).unwrap());
        let coords = s.extract(&ring.scalar_mul(k_over_d, &t).unwrap()).unwrap();
        assert_eq!(psi_inv_u[i], coords, "psi_inv q {q} d {d} K {K} i {i}");
    }
    // Monomials by their definition (b_1 the most significant bit).
    let n_vars = (d / K).trailing_zeros() as usize;
    let point: Vec<Ext<K>> = (0..n_vars).map(|_| random_ext(&mut rng, q)).collect();
    let mono = s.monomials(&point).unwrap();
    assert_eq!(mono.len(), d / K);
    for (idx, m) in mono.iter().enumerate() {
        let mut want = f.one();
        for (j, xj) in point.iter().enumerate() {
            if (idx >> (n_vars - 1 - j)) & 1 == 1 {
                want = f.mul(&want, xj);
            }
        }
        assert_eq!(*m, want);
    }
    // E(psi(a)) = <a, monomials(x)> with v = psi(monomials(x)).
    let v = s.psi(&mono).unwrap();
    assert_eq!(
        s.e_map(&pa, &v).unwrap(),
        s.embed(&inner(&s, &a, &mono)),
        "E property q {q} d {d} K {K}"
    );
    // E by its definition, with the schoolbook product and the summed trace.
    if full {
        let w = ring.automorphism(&v, -1).unwrap();
        let t = trace_by_definition(&ring, K, &ring.mul_schoolbook(&u, &w).unwrap());
        assert_eq!(
            s.e_map(&u, &v).unwrap(),
            ring.scalar_mul(k_over_d, &t).unwrap()
        );
    }
    // E is additive, and its matrix acts on coefficients.
    let u2 = Poly(residues(&mut rng, q, d));
    let sum = s.e_map(&ring.add(&u, &u2).unwrap(), &v).unwrap();
    assert_eq!(
        sum,
        ring.add(&s.e_map(&u, &v).unwrap(), &s.e_map(&u2, &v).unwrap())
            .unwrap()
    );
    if d <= 1024 {
        let m = s.e_matrix(&v).unwrap();
        assert_eq!(m.len(), d);
        for input in [&u, &worst] {
            let e = s.e_map(input, &v).unwrap();
            for (row, want) in m.iter().zip(&e.0) {
                let got = row
                    .iter()
                    .zip(&input.0)
                    .fold(0u64, |acc, (mij, uj)| zq.add(acc, zq.mul(*mij, *uj)));
                assert_eq!(got, *want, "E matrix q {q} d {d} K {K}");
            }
        }
    }
}

#[test]
fn subfield_maps_equal_their_definitions_for_small_rings() {
    for (i, q) in ALL_MODULI.into_iter().enumerate() {
        let seed = 50 + i as u64;
        for d in [2usize, 4, 8, 16, 32, 64, 256] {
            check_subfield::<1>(q, d, seed);
            if d >= 4 {
                check_subfield::<2>(q, d, seed);
            }
            if d >= 8 {
                check_subfield::<4>(q, d, seed);
            }
            if d >= 16 {
                check_subfield::<8>(q, d, seed);
            }
        }
    }
}

#[test]
fn subfield_maps_equal_their_definitions_for_large_rings() {
    for (i, q) in MODULI.into_iter().enumerate() {
        let seed = 55 + i as u64;
        check_subfield::<4>(q, 1024, seed);
        check_subfield::<8>(q, 1024, seed);
    }
    check_subfield::<4>(Q61, 2048, 58);
    check_subfield::<1>(Q32, 2048, 58);
}

#[test]
fn invalid_subfields_and_shapes_are_rejected() {
    let ring = Ring::new(Q61, 8).unwrap();
    assert!(
        matches!(Subfield::<8>::new(&ring), Err(Error::Parameter(_))),
        "8 does not divide d/2 = 4"
    );
    assert!(matches!(
        Subfield::<3>::new(&Ring::new(Q61, 16).unwrap()),
        Err(Error::Parameter(_))
    ));
    let other = Ring::new(97, 16).unwrap();
    assert_eq!(
        Subfield::<4>::new(&other).unwrap_err(),
        Error::ModulusClass(97)
    );
    let ring = Ring::new(Q61, 16).unwrap();
    let s = Subfield::<4>::new(&ring).unwrap();
    let short = Poly(vec![0; 15]);
    assert!(!s.is_in_subfield(&short));
    assert!(matches!(s.extract(&short), Err(Error::Length { .. })));
    assert!(matches!(s.trace(&short), Err(Error::Length { .. })));
    assert!(matches!(s.psi_inv(&short), Err(Error::Length { .. })));
    assert!(matches!(
        s.e_map(&short, &ring.zero()),
        Err(Error::Length { .. })
    ));
    assert!(matches!(
        s.e_map(&ring.zero(), &short),
        Err(Error::Length { .. })
    ));
    assert!(matches!(s.e_matrix(&short), Err(Error::Length { .. })));
    assert_eq!(
        s.psi(&[Ext([0; 4]); 3]),
        Err(Error::Length {
            expected: 4,
            actual: 3
        })
    );
    assert!(s.monomials(&[Ext([0; 4]); 48]).is_err());
    assert_eq!(s.monomials(&[]).unwrap(), vec![s.field().one()]);
}
