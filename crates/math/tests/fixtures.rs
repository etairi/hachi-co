//! Exact agreement with the Sage reference `ref/hachico_ref/subfield.py`, through the fixtures
//! that `ref/fixtures/gen_math_fixtures.py` writes to `tests/fixtures/` (`docs/m0-spec.md`,
//! exit criteria): the three moduli $2^{32} - 99$, $2^{52} - 395$, $2^{61} - 259$ and
//! $(d, K) \in \lbrace (16,1), (16,2), (16,4), (16,8), (64,4), (64,8), (1024,4) \rbrace$. The
//! format of each case is documented in the generator.
use hachi_co_math::{Ext, Fqk, Poly, Ring, Subfield};
use serde::Deserialize;

#[derive(Deserialize)]
struct File {
    meta: Meta,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct Meta {
    generator: String,
    reference: String,
    sage: String,
    modulus_name: String,
}

#[derive(Deserialize)]
struct Case {
    q: u64,
    d: usize,
    k: usize,
    seed: u64,
    h_exponents: Vec<usize>,
    field: FieldCase,
    embed: (Vec<u64>, Vec<u64>),
    u: Vec<u64>,
    trace_u: Vec<u64>,
    sigma: Option<Vec<(i64, Vec<u64>)>>,
    product: Option<(Vec<u64>, Vec<u64>)>,
    in_subfield: (bool, bool),
    psi: PsiCase,
    theorem2: Theorem2Case,
    e_map: EMapCase,
}

#[derive(Deserialize)]
struct FieldCase {
    elements: Vec<Vec<u64>>,
    products: Vec<Vec<Vec<u64>>>,
    sums: Vec<Vec<u64>>,
    inverses: Vec<(usize, Vec<u64>)>,
    powers: Vec<(usize, Vec<u64>, Vec<u64>)>,
    frobenius: Vec<Vec<u64>>,
}

#[derive(Deserialize)]
struct PsiCase {
    a: Vec<Vec<u64>>,
    psi_a: Vec<u64>,
    psi_inv_u: Vec<Vec<u64>>,
}

#[derive(Deserialize)]
struct Theorem2Case {
    b: Vec<Vec<u64>>,
    lhs: Vec<u64>,
    inner: Vec<u64>,
}

#[derive(Deserialize)]
struct EMapCase {
    point: Vec<Vec<u64>>,
    monomials: Vec<Vec<u64>>,
    v: Vec<u64>,
    e_u: Vec<u64>,
    e_psi_a: Vec<u64>,
    inner_a_monomials: Vec<u64>,
    matrix: Option<Vec<Vec<u64>>>,
}

fn ext<const K: usize>(v: &[u64]) -> Ext<K> {
    Ext(v.try_into().expect("K coordinates"))
}

fn exts<const K: usize>(v: &[Vec<u64>]) -> Vec<Ext<K>> {
    v.iter().map(|x| ext::<K>(x)).collect()
}

fn inner<const K: usize>(f: &Fqk<K>, a: &[Ext<K>], b: &[Ext<K>]) -> Ext<K> {
    a.iter()
        .zip(b)
        .fold(f.zero(), |acc, (x, y)| f.add(&acc, &f.mul(x, y)))
}

fn check<const K: usize>(c: &Case) {
    let label = format!("q {} d {} K {} seed {}", c.q, c.d, c.k, c.seed);
    let ring = Ring::new(c.q, c.d).unwrap();
    let s = Subfield::<K>::new(&ring).unwrap();
    let f = s.field();
    assert_eq!(s.h_exponents(), c.h_exponents.as_slice(), "{label}: H");

    // Field: the full product table of the listed elements (the basis among them), sums,
    // inverses, powers and the Frobenius.
    let el = exts::<K>(&c.field.elements);
    for (i, x) in el.iter().enumerate() {
        for (j, y) in el.iter().enumerate() {
            assert_eq!(
                f.mul(x, y),
                ext::<K>(&c.field.products[i][j]),
                "{label}: product {i} {j}"
            );
        }
        assert_eq!(
            f.frobenius(x),
            ext::<K>(&c.field.frobenius[i]),
            "{label}: frobenius {i}"
        );
    }
    for (i, sum) in c.field.sums.iter().enumerate() {
        assert_eq!(f.add(&el[i], &el[i + 1]), ext::<K>(sum), "{label}: sum {i}");
    }
    for (i, inv) in &c.field.inverses {
        assert_eq!(f.inv(&el[*i]), Some(ext::<K>(inv)), "{label}: inverse {i}");
    }
    for (i, e, power) in &c.field.powers {
        assert_eq!(f.pow(&el[*i], e), ext::<K>(power), "{label}: power {i}");
    }

    // Ring: embedding, trace, automorphisms, product, membership.
    let x = ext::<K>(&c.embed.0);
    assert_eq!(s.embed(&x), Poly(c.embed.1.clone()), "{label}: embed");
    let u = Poly(c.u.clone());
    let trace = s.trace(&u).unwrap();
    assert_eq!(
        s.extract(&trace).unwrap(),
        ext::<K>(&c.trace_u),
        "{label}: trace"
    );
    if let Some(sigma) = &c.sigma {
        for (i, value) in sigma {
            assert_eq!(
                ring.automorphism(&u, *i).unwrap(),
                Poly(value.clone()),
                "{label}: sigma {i}"
            );
        }
    }
    if let Some((w, product)) = &c.product {
        let w = Poly(w.clone());
        assert_eq!(
            ring.mul(&u, &w).unwrap(),
            Poly(product.clone()),
            "{label}: ring product"
        );
        assert_eq!(ring.mul_schoolbook(&u, &w).unwrap(), Poly(product.clone()));
    }
    assert_eq!(
        s.is_in_subfield(&s.embed(&x)),
        c.in_subfield.0,
        "{label}: in_subfield"
    );
    assert_eq!(
        s.is_in_subfield(&u),
        c.in_subfield.1,
        "{label}: in_subfield"
    );

    // psi and its inverse.
    let a = exts::<K>(&c.psi.a);
    let psi_a = Poly(c.psi.psi_a.clone());
    assert_eq!(s.psi(&a).unwrap(), psi_a, "{label}: psi");
    assert_eq!(
        s.psi_inv(&u).unwrap(),
        exts::<K>(&c.psi.psi_inv_u),
        "{label}: psi_inv"
    );
    assert_eq!(s.psi_inv(&psi_a).unwrap(), a, "{label}: psi_inv psi");

    // Theorem 2.
    let b = exts::<K>(&c.theorem2.b);
    let product = ring
        .mul(&psi_a, &ring.automorphism(&s.psi(&b).unwrap(), -1).unwrap())
        .unwrap();
    let lhs = s.extract(&s.trace(&product).unwrap()).unwrap();
    assert_eq!(lhs, ext::<K>(&c.theorem2.lhs), "{label}: Theorem 2 lhs");
    let ab = inner(f, &a, &b);
    assert_eq!(ab, ext::<K>(&c.theorem2.inner), "{label}: <a, b>");
    assert_eq!(lhs, f.scale(&ab, (c.d / c.k) as u64), "{label}: Theorem 2");

    // E.
    let point = exts::<K>(&c.e_map.point);
    let mono = s.monomials(&point).unwrap();
    assert_eq!(mono, exts::<K>(&c.e_map.monomials), "{label}: monomials");
    let v = s.psi(&mono).unwrap();
    assert_eq!(v, Poly(c.e_map.v.clone()), "{label}: v");
    assert_eq!(
        s.extract(&s.e_map(&u, &v).unwrap()).unwrap(),
        ext::<K>(&c.e_map.e_u),
        "{label}: E(u)"
    );
    let e_psi_a = s.extract(&s.e_map(&psi_a, &v).unwrap()).unwrap();
    assert_eq!(e_psi_a, ext::<K>(&c.e_map.e_psi_a), "{label}: E(psi(a))");
    assert_eq!(
        inner(f, &a, &mono),
        ext::<K>(&c.e_map.inner_a_monomials),
        "{label}: <a, mono>"
    );
    assert_eq!(e_psi_a, inner(f, &a, &mono), "{label}: E property");
    if let Some(matrix) = &c.e_map.matrix {
        assert_eq!(&s.e_matrix(&v).unwrap(), matrix, "{label}: E matrix");
    } else {
        assert_eq!(c.d, 1024, "{label}: matrices are omitted only for d = 1024");
    }
}

#[test]
fn the_crate_agrees_with_the_sage_fixtures() {
    let expected_cases = [
        (16, 1),
        (16, 2),
        (16, 4),
        (16, 8),
        (64, 4),
        (64, 8),
        (1024, 4),
    ];
    for (name, q) in [
        ("q32", 4294967197u64),
        ("q52", 4503599627370101),
        ("q61", 2305843009213693693),
    ] {
        let path = format!(
            "{}/tests/fixtures/subfield_{name}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let file: File = serde_json::from_str(&text).unwrap();
        assert_eq!(file.meta.modulus_name, name);
        assert_eq!(file.meta.reference, "ref/hachico_ref/subfield.py");
        let shapes: Vec<(usize, usize)> = file.cases.iter().map(|c| (c.d, c.k)).collect();
        assert_eq!(shapes, expected_cases, "{path}: cases");
        for c in &file.cases {
            assert_eq!(c.q, q);
            match c.k {
                1 => check::<1>(c),
                2 => check::<2>(c),
                4 => check::<4>(c),
                8 => check::<8>(c),
                k => panic!("unexpected k = {k}"),
            }
        }
    }
}
