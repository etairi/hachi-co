//! The discrete Gaussian sampler (`hachi_co_sampling::gauss`).
//!
//! What is tested:
//! - the tables have the pinned digests of `tests/fixtures/gauss_tables.py --digest`, are
//!   non-increasing and end in a nonzero entry;
//! - samples follow the distribution the table defines (chi-square) with its variance, and stay
//!   within the bound;
//! - a try reads three words and follows the documented rule, including the repetition;
//! - the repetition cap and an unsupported width are errors.
//!
//! The tables themselves are checked against $D_{\mathbb Z,\sigma}$ and against Sage's sampler in
//! `tests/fixtures/gauss_check.sage`.

mod common;

use common::{Constant, Scripted, assert_chi_square};
use hachi_co_sampling::uniform::MAX_TRIES;
use hachi_co_sampling::{Error, ShakePrg, gauss};
use shake::{ExtendableOutput, Shake256, Update, XofReader};

/// SHAKE256 of LE64(length) and every entry as 16 little-endian bytes, 32 bytes in hex (the
/// digest of `gauss_tables.py --digest`).
fn digest(table: &[u128]) -> String {
    let mut h = Shake256::default();
    h.update(&(table.len() as u64).to_le_bytes());
    for t in table {
        h.update(&t.to_le_bytes());
    }
    let mut out = [0u8; 32];
    h.finalize_xof().read(&mut out);
    out.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn the_tables_have_the_pinned_digests_and_shapes() {
    // From `sage -python crates/sampling/tests/fixtures/gauss_tables.py --digest`.
    let pinned = [
        (
            13,
            "e6356c3b562ebbef593848be323aef0e6f04b6fde5ebfd7126f79ab6d1a1db4f",
        ),
        (
            26,
            "8ceeeac4419ab3de2b9535833cd9efbae68e8b4d52bddfce2ad2282603df17e5",
        ),
        (
            52,
            "7e6a023301124e0af489cc04a5ee314c1cd9514d1fc5a49ec8c5d684ce2b8fee",
        ),
        (
            105,
            "faf0e017cdb439e906636d3358b3bff89d75849ec076145b0ced2703a53db4e9",
        ),
        (
            211,
            "65db6f695523a5d0a6271c3f74589f31b9ca214019e76860a10f8687ae408017",
        ),
    ];
    for (j, (len, d)) in pinned.iter().enumerate() {
        let table = gauss::table(j as u32).unwrap();
        assert_eq!(table.len(), *len, "sigma 2^{j}: length");
        assert_eq!(digest(table), *d, "sigma 2^{j}: digest");
        assert!(table.windows(2).all(|w| w[0] >= w[1]), "non-increasing");
        assert!(*table.last().unwrap() > 0, "the last entry is nonzero");
        assert_eq!(gauss::bound(j as u32).unwrap(), *len as u64);
    }
    for j in [gauss::MAX_LOG_SIGMA + 1, u32::MAX] {
        assert!(matches!(gauss::table(j), Err(Error::InvalidParameter(_))));
        assert!(matches!(gauss::bound(j), Err(Error::InvalidParameter(_))));
        let mut prg = ShakePrg::new(&[0; 32], b"", 0);
        assert!(matches!(
            gauss::gaussian(&mut prg, j, 1),
            Err(Error::InvalidParameter(_))
        ));
    }
}

/// The probabilities of the construction for `table`: entry $x + B$ is $\Pr[x]$ for
/// $\lvert x \rvert \le B$, $B$ the table's length.
fn pmf(table: &[u128]) -> Vec<f64> {
    let scale = 2f64.powi(128);
    let tail: Vec<f64> = table.iter().map(|t| *t as f64 / scale).collect();
    let len = table.len();
    // Pr[K = i] = T_{i-1} - T_i over 2^128, with T_{-1} = 2^128 and T_len = 0.
    let pk: Vec<f64> = (0..=len)
        .map(|i| {
            let above = if i == 0 { 1.0 } else { tail[i - 1] };
            above - tail.get(i).copied().unwrap_or(0.0)
        })
        .collect();
    let accept = 1.0 - pk[0] / 2.0;
    let mut out = vec![0.0; 2 * len + 1];
    out[len] = pk[0] / 2.0 / accept;
    for i in 1..=len {
        out[len + i] = pk[i] / 2.0 / accept;
        out[len - i] = pk[i] / 2.0 / accept;
    }
    out
}

#[test]
fn samples_follow_the_table_and_stay_within_the_bound() {
    let n = 200_000;
    for j in 0..=gauss::MAX_LOG_SIGMA {
        let sigma = 1i64 << j;
        let table = gauss::table(j).unwrap();
        let bound = table.len() as i64;
        let mut prg = ShakePrg::new(&[j as u8 + 1; 32], b"gauss-test", u64::from(j));
        let xs = gauss::gaussian(&mut prg, j, n).unwrap();
        assert_eq!(xs.len(), n);
        assert!(xs.iter().all(|x| x.abs() <= bound), "sigma {sigma}: bound");
        // Bins: x < -3 sigma, every x in [-3 sigma, 3 sigma], x > 3 sigma.
        let p = pmf(table);
        let c = 3 * sigma;
        let at = |x: i64| p[(x + bound) as usize];
        let mut expected = vec![0.0; (2 * c + 3) as usize];
        for x in -bound..=bound {
            let bin = (x.clamp(-c - 1, c + 1) + c + 1) as usize;
            expected[bin] += n as f64 * at(x);
        }
        let mut observed = vec![0u64; expected.len()];
        for x in &xs {
            observed[((*x).clamp(-c - 1, c + 1) + c + 1) as usize] += 1;
        }
        assert!(expected.iter().all(|e| *e >= 5.0), "sigma {sigma}: bins");
        assert_chi_square(&format!("sigma {sigma}"), &observed, &expected);
        // The variance against the table's, within 5 standard errors.
        let var: f64 = (-bound..=bound).map(|x| (x * x) as f64 * at(x)).sum();
        let fourth: f64 = (-bound..=bound).map(|x| (x as f64).powi(4) * at(x)).sum();
        let mean_sq = xs.iter().map(|x| (x * x) as f64).sum::<f64>() / n as f64;
        let se = ((fourth - var * var) / n as f64).sqrt();
        assert!(
            (mean_sq - var).abs() < 5.0 * se,
            "sigma {sigma}: E[x^2] = {mean_sq}, table {var}, se {se}"
        );
        // And the table's variance is sigma^2 up to the discretization (|err| < 1e-6 sigma^2
        // for sigma >= 1; computed: the discrete Gaussian's variance is sigma^2 minus a term of
        // order exp(-2 pi^2 sigma^2)).
        assert!(
            (var - (sigma * sigma) as f64).abs() < 1e-6 * (sigma * sigma) as f64,
            "sigma {sigma}: table variance {var}"
        );
    }
}

#[test]
fn a_try_reads_three_words_and_follows_the_rule() {
    let table = gauss::table(0).unwrap();
    let len = table.len() as i64;
    let t0 = table[0];
    let split = |v: u128| (v as u64, (v >> 64) as u64);
    // V = 0 is below every entry: K = len. Sign 0 gives -K, sign 1 gives K.
    for (sign, want) in [(0u64, -len), (1, len), (2, -len), (3, len)] {
        let mut rng = Scripted::new(&[0, 0, sign]);
        assert_eq!(gauss::gaussian(&mut rng, 0, 1).unwrap(), vec![want]);
        assert_eq!(rng.read, 3);
    }
    // V = T_0 - 1 gives K = 1, V = T_0 gives K = 0 (with sign 0: the sample 0).
    let (a, b) = split(t0 - 1);
    let mut rng = Scripted::new(&[a, b, 1]);
    assert_eq!(gauss::gaussian(&mut rng, 0, 1).unwrap(), vec![1]);
    let (a, b) = split(t0);
    let mut rng = Scripted::new(&[a, b, 0]);
    assert_eq!(gauss::gaussian(&mut rng, 0, 1).unwrap(), vec![0]);
    // K = 0 with sign 1 is drawn again: the second try decides.
    let mut rng = Scripted::new(&[a, b, 1, 0, 0, 0]);
    assert_eq!(gauss::gaussian(&mut rng, 0, 1).unwrap(), vec![-len]);
    assert_eq!(rng.read, 6);
    // Two samples read two tries.
    let mut rng = Scripted::new(&[0, 0, 0, u64::MAX, u64::MAX, 0]);
    assert_eq!(gauss::gaussian(&mut rng, 0, 2).unwrap(), vec![-len, 0]);
    assert_eq!(rng.read, 6);
}

#[test]
fn the_repetition_cap_is_an_error() {
    // Every word all ones: V = 2^128 - 1 is above every entry (K = 0) and the sign is 1.
    let mut rng = Constant(u64::MAX, 0);
    assert_eq!(
        gauss::gaussian(&mut rng, 3, 1),
        Err(Error::RejectionLimit(MAX_TRIES))
    );
    assert_eq!(rng.1, 3 * MAX_TRIES as usize);
}
