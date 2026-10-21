//! The size formula of the Eval proof (`sizes`, `docs/m1-spec.md`, WP11): the components add up,
//! each component follows its formula (recomputed here from the parameters), and the toy and
//! development presets have the sizes computed by hand. The `hachi-co` tests check the formula
//! against real encodings.

use hachi_co_params::sizes::{HEADER_BYTES, WORD_BYTES};
use hachi_co_params::{BasePcsWords, presets, sizes, sizes_with_base};

#[test]
fn components_follow_the_formula_for_every_preset() {
    for p in presets().unwrap() {
        let (x, _) = p.derive().unwrap();
        let s = sizes(&x).unwrap();
        let (d, k, n, n_d) = (
            u128::from(x.d),
            u128::from(x.k),
            u128::from(x.choices.n),
            u128::from(x.choices.n_d),
        );
        let ell_pp = u128::from(x.ell_dprime);
        let per_round = u128::from(x.d0 + 1 + x.d1 + 1) * k;
        assert_eq!((s.header, s.attempt), (16, 8), "{}", p.name);
        assert_eq!(s.e_u, 8 * d, "{}", p.name);
        assert_eq!(s.v, 8 * n_d * d, "{}", p.name);
        assert_eq!(s.g_sums, 8 * 2 * n * k, "{}", p.name);
        assert_eq!(s.rounds, 8 * ell_pp * per_round, "{}", p.name);
        assert_eq!(s.finals, 8 * 3 * n * k, "{}", p.name);
        let sum: u128 = s.components().iter().map(|(_, b)| b).sum();
        assert_eq!(s.total, sum, "{}", p.name);
        assert_eq!(
            s.total,
            HEADER_BYTES + WORD_BYTES * (1 + s.residues),
            "{}",
            p.name
        );
        // ceil(log2 q): q is an odd prime, so its bit length.
        assert_eq!(s.bits_per_residue, 64 - x.q.leading_zeros(), "{}", p.name);
        assert!(x.q > 1 << (s.bits_per_residue - 1) && x.q < 1 << s.bits_per_residue);
        let packed = (s.residues * u128::from(s.bits_per_residue)).div_ceil(8);
        assert_eq!(s.packed_total, 16 + 8 + packed, "{}", p.name);
        assert!(s.packed_total < s.total, "{}", p.name);
    }
}

/// Hand-computed totals: toy-n1 (n = 1, l'' = 15, 2^23 < q < 2^24), toy-n2 (n = 2, l'' = 15,
/// 2^25 < q < 2^26), toy-n3 (n = 3, l'' = 16, 2^27 < q < 2^28), toy-n4 (n = 4, l'' = 16,
/// 2^27 < q < 2^28), toy-n8 (n = 8, l'' = 17, 2^29 < q < 2^30) and the development presets
/// (n = 1, l'' = 18, 21, 23, q just above 2^29, 2^33, 2^37); d0 = 14 for all of them.
#[test]
fn toy_and_dev_sizes_by_hand() {
    // total = 16 + 8 (1 + 16 + 16 + 8n + 76 l'' + 12n); residues = 32 + 20n + 76 l''.
    let table: [(&str, u128, u128, u32); 8] = [
        ("toy-n1", 9560, 1192, 24),
        ("toy-n2", 9720, 1212, 26),
        ("toy-n3", 10488, 1308, 28),
        ("toy-n4", 10648, 1328, 28),
        ("toy-n8", 11896, 1484, 30),
        ("dev-l12", 11384, 1420, 30),
        ("dev-l16", 13208, 1648, 34),
        ("dev-l20", 14424, 1800, 38),
    ];
    for (name, total, residues, bits) in table {
        let p = hachi_co_params::preset(name).unwrap();
        let s = sizes(&p.derive().unwrap().0).unwrap();
        assert_eq!(
            (s.total, s.residues, s.bits_per_residue),
            (total, residues, bits),
            "{name}"
        );
        assert_eq!(
            s.packed_total,
            24 + (residues * u128::from(bits)).div_ceil(8),
            "{name}"
        );
    }
}

/// With the base PCS: the Hachi-Co components are those of `sizes`, and the base PCS adds
/// 6n length words, n commitments and n proofs of each polynomial, for arbitrary word counts at
/// every preset; the Hachi-Co part alone has no base-PCS bytes.
#[test]
fn the_base_pcs_adds_its_lengths_commitments_and_proofs() {
    let base = BasePcsWords {
        commitment_w: 4,
        commitment_g0: 5,
        commitment_g1: 6,
        proof_w: 1000,
        proof_g0: 70,
        proof_g1: 9,
    };
    for p in presets().unwrap() {
        let (x, _) = p.derive().unwrap();
        let (plain, with) = (sizes(&x).unwrap(), sizes_with_base(&x, &base).unwrap());
        let n = u128::from(x.choices.n);
        assert_eq!(
            (plain.pcs_lengths, plain.pcs_commitments, plain.pcs_openings),
            (0, 0, 0),
            "{}",
            p.name
        );
        assert_eq!(with.pcs_lengths, 8 * 6 * n, "{}", p.name);
        assert_eq!(with.pcs_commitments, 8 * n * 15, "{}", p.name);
        assert_eq!(with.pcs_openings, 8 * n * 1079, "{}", p.name);
        assert_eq!(
            with.total,
            plain.total + 8 * n * (6 + 15 + 1079),
            "{}",
            p.name
        );
        let sum: u128 = with.components().iter().map(|(_, b)| b).sum();
        assert_eq!(with.total, sum, "{}", p.name);
        for ((name, a), (_, b)) in plain.components().iter().zip(with.components()) {
            if !name.starts_with("PCS") {
                assert_eq!(*a, b, "{}: {name}", p.name);
            }
        }
        assert_eq!(
            with.packed_total,
            plain.packed_total + 8 * n * 1100,
            "{}",
            p.name
        );
    }
}
