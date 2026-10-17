// Golden test for the legacy prototype: a deterministic witness at l = 20, the full
// setup/commit/prove/verify, and a SHA-256 over the commitment and the proof. The expected values
// were computed on the code before it moved into this crate (commit eb0eefc), so any change in the
// legacy behaviour, or in its dependencies, shows up here.
use std::io::Write;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use hachi::arithmetic::fs::Serialise;
use hachi::arithmetic::utils::{Logarithm, multi_lin_coeff_int, rand_int};
use hachi::hachi::{Hachi, commit::Commit, prove::Prove, setup::Setup, verify::Verify};
use hachi::stream::{Stream, file_stream::U64FileStream};

#[test]
fn golden_l20() {
    let l = 20;
    let params = Hachi::setup(l, true);
    let path = std::env::temp_dir().join(format!("hachi-golden-{}.bin", std::process::id()));
    {
        let mut rng = ChaCha12Rng::from_seed([7u8; 32]);
        let mut f = std::fs::File::create(&path).unwrap();
        let mut buf = Vec::with_capacity(4 << l);
        for _ in 0..1u64 << l {
            buf.extend_from_slice(&(rand_int(params.q, params.q.log(), &mut rng) as u32).to_le_bytes());
        }
        f.write_all(&buf).unwrap();
    }
    let mut witness = U64FileStream::init(path.to_str().unwrap(), 0);
    let com = Hachi::commit(&mut witness, &params);
    let x: Vec<u64> = (0..params.l).map(|i| (1234 + 7 * i as u64) % params.q).collect();
    let proof = Hachi::prove(&mut witness, &params, &x, &com);
    witness.reset();
    let (pr, pm) = (params.r, params.m);
    let mut x_perm = Vec::with_capacity(params.l);
    x_perm.extend_from_slice(&x[pr + pm..params.l]);
    x_perm.extend_from_slice(&x[pr..pr + pm]);
    x_perm.extend_from_slice(&x[0..pr]);
    let mut buf = vec![0u64; 1 << l];
    witness.read(&mut buf);
    let mut y = 0u64;
    for j in 0..1usize << l {
        let a = multi_lin_coeff_int(&x_perm, j, params.l, params.q);
        y = (y + a * buf[j]) % params.q;
    }
    Hachi::verify(&params, &x, y, &com.u, &proof);
    let mut bytes = Vec::new();
    bytes.extend(com.u.serialise());
    bytes.extend(proof.y.serialise());
    bytes.extend(proof.v.serialise());
    bytes.extend(proof.u_dash.serialise());
    for u in &proof.univariates_f_alpha { bytes.extend(u.serialise()); }
    for u in &proof.univariates_f_0 { bytes.extend(u.serialise()); }
    bytes.extend(format!("{}", proof.y_dash).into_bytes());
    std::fs::remove_file(&path).ok();
    let digest = sha256::digest(&bytes[..]);
    println!("GOLDEN y={} sha256={}", y, digest);
    assert_eq!(y, 2980921180);
    assert_eq!(digest, "7803646e9c5b046b4d61baa82c827ffbec3962e0405b4ef818d4c15716a73b9a");
}
