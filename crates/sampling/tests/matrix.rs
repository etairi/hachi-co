//! Seeded matrices: each entry comes from its own stream (row-major index), the values are
//! residues, and the seed and domain separate matrices. Byte-exact values are in `tests/kat.rs`.

use hachi_co_sampling::matrix::{expand, expand_entry};
use hachi_co_sampling::uniform::uniform_mod_q;
use hachi_co_sampling::{Error, ShakePrg};

const SEED: [u8; 32] = [0x42; 32];
const Q: u64 = 8388637; // toy-n1 modulus

#[test]
fn entries_are_independent_streams_in_row_major_order() {
    let (rows, cols, d) = (3, 4, 16);
    let m = expand(&SEED, b"A", rows, cols, d, Q).unwrap();
    assert_eq!(m.len(), rows);
    for (i, row) in m.iter().enumerate() {
        assert_eq!(row.len(), cols);
        for (j, entry) in row.iter().enumerate() {
            let index = (i * cols + j) as u64;
            assert_eq!(entry, &expand_entry(&SEED, b"A", index, d, Q).unwrap());
            let direct = uniform_mod_q(&mut ShakePrg::new(&SEED, b"A", index), Q, d).unwrap();
            assert_eq!(entry, &direct, "entry ({i}, {j})");
            assert_eq!(entry.len(), d);
            assert!(entry.iter().all(|x| *x < Q));
        }
    }
    // A sub-matrix with the same number of columns is a prefix of the rows.
    let top = expand(&SEED, b"A", 2, cols, d, Q).unwrap();
    assert_eq!(&m[..2], &top[..]);
}

#[test]
fn seed_and_domain_separate_matrices() {
    let a = expand(&SEED, b"A", 2, 2, 8, Q).unwrap();
    let b = expand(&SEED, b"B", 2, 2, 8, Q).unwrap();
    let a2 = expand(&[0x43; 32], b"A", 2, 2, 8, Q).unwrap();
    for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        assert_ne!(a[i][j], b[i][j], "domains A and B agree at ({i}, {j})");
        assert_ne!(a[i][j], a2[i][j], "seeds agree at ({i}, {j})");
    }
    // Distinct entries of one matrix differ (distinct indices).
    assert_ne!(a[0][1], a[1][0]);
}

#[test]
fn degenerate_shapes_and_errors() {
    assert_eq!(expand(&SEED, b"A", 0, 5, 16, Q), Ok(vec![]));
    assert_eq!(expand(&SEED, b"A", 2, 0, 16, Q), Ok(vec![vec![], vec![]]));
    assert_eq!(expand(&SEED, b"A", 1, 1, 0, Q), Ok(vec![vec![vec![]]]));
    assert_eq!(
        expand(&SEED, b"A", 1, 2, 3, 1),
        Ok(vec![vec![vec![0; 3]; 2]])
    );
    assert_eq!(expand(&SEED, b"A", 1, 1, 4, 0), Err(Error::EmptyRange));
    assert_eq!(expand(&SEED, b"A", 0, 0, 4, 0), Err(Error::EmptyRange));
    assert_eq!(expand_entry(&SEED, b"A", 0, 4, 0), Err(Error::EmptyRange));
    // The largest index is a valid stream.
    assert_eq!(expand_entry(&SEED, b"A", u64::MAX, 4, Q).unwrap().len(), 4);
}

#[test]
fn large_moduli_stay_in_range() {
    for q in [(1u64 << 61) - 1, (1 << 63) - 25, u64::MAX] {
        let m = expand(&SEED, b"wide", 2, 2, 64, q).unwrap();
        assert!(m.iter().flatten().flatten().all(|x| *x < q), "q = {q}");
    }
}
