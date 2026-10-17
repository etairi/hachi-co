//! Agreement with the Sage protocol oracle's recorded decompositions and commitments
//! (`ref/fixtures/toy-n{1,2,3}.json`, format `hachi-co-ref-fixture/v1`, described in
//! `ref/README.md`; written by `ref/run_oracle.py`, which uses the same $\mathbf G^{-1}$
//! representative rule: $x$ if $x \le (\lceil b/2 \rceil - 1) S_\delta$, else $x - q$).
//!
//! For every toy preset: $\delta$ and the tight $\tau$; for every owned block of $\mathbf f'$,
//! $\mathbf s = \mathbf G^{-1}(\mathbf f'_i)$, $\mathbf G \mathbf s = \mathbf f'_i$,
//! $\mathbf t = \mathbf A \mathbf s$ (an exact product in $\mathcal R_q$) and
//! $\hat{\mathbf t} = \mathbf G^{-1}(\mathbf t)$; $\hat{\mathbf w} = \mathbf G^{-1}(\mathbf w)$;
//! for every attempt and prover, $\hat{\mathbf t}_0 = \mathbf G^{-1}(\mathbf t_0)$ and
//! $\hat{\mathbf w}_0 = \mathbf G^{-1}(\mathbf w_0)$; and
//! $\hat{\mathbf z} = \mathbf J^{-1}(\mathbf z)$ as recorded in $\mathbf z'$, with
//! $\mathbf J \hat{\mathbf z} = \mathbf z$.
use hachi_co_math::{
    Poly, Ring,
    decompose::{g_apply, g_inv, j_apply, j_inv, num_digits, tight_tau},
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    format: String,
    params: Params,
    setup: Setup,
    commit: Commit,
    ring_to_linear: RingToLinear,
}

#[derive(Deserialize)]
struct Params {
    q: u64,
    d: usize,
    b: u64,
    delta: usize,
    tau: usize,
    beta: u128,
    m: usize,
    #[serde(rename = "n_A")]
    n_a: usize,
}

#[derive(Deserialize)]
struct Setup {
    #[serde(rename = "A")]
    a: Vec<Vec<Vec<u64>>>,
}

#[derive(Deserialize)]
struct Commit {
    f_prime: Vec<Vec<u64>>,
    blocks: Vec<Block>,
}

#[derive(Deserialize)]
struct Block {
    index: usize,
    s: Vec<Vec<u64>>,
    t: Vec<Vec<u64>>,
    t_hat: Vec<Vec<u64>>,
}

#[derive(Deserialize)]
struct RingToLinear {
    w: Vec<Vec<u64>>,
    w_hat: Vec<Vec<u64>>,
    attempts: Vec<Attempt>,
    z: Vec<Vec<i64>>,
    z_prime: Vec<Vec<u64>>,
    layout: Layout,
}

#[derive(Deserialize)]
struct Attempt {
    provers: Vec<ProverAttempt>,
}

#[derive(Deserialize)]
struct ProverAttempt {
    prover: usize,
    t0: Vec<Vec<u64>>,
    t0_hat: Vec<Vec<u64>>,
    w0: Vec<u64>,
    w0_hat: Vec<Vec<u64>>,
}

#[derive(Deserialize)]
struct Layout {
    blocks: Vec<(String, usize, usize)>,
}

fn polys(v: &[Vec<u64>]) -> Vec<Poly> {
    v.iter().map(|c| Poly(c.clone())).collect()
}

#[test]
fn decompositions_and_commitments_agree_with_the_protocol_oracle() {
    for n in 1..=3 {
        let path = format!(
            "{}/../../ref/fixtures/toy-n{n}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let fx: Fixture = serde_json::from_str(&text).unwrap();
        assert_eq!(fx.format, "hachi-co-ref-fixture/v1");
        let p = &fx.params;
        let ring = Ring::new(p.q, p.d).unwrap();
        assert_eq!(num_digits(p.q, p.b).unwrap(), p.delta, "toy-n{n}: delta");
        assert_eq!(tight_tau(p.beta, p.b).unwrap(), p.tau, "toy-n{n}: tau");
        let (b, delta, tau) = (p.b, p.delta, p.tau);

        // Commit: s = G^{-1}(f'_i), G s = f'_i, t = A s, t_hat = G^{-1}(t).
        let a: Vec<Vec<Poly>> = fx.setup.a.iter().map(|row| polys(row)).collect();
        assert_eq!(a.len(), p.n_a);
        let width = 1 << p.m;
        assert!(!fx.commit.blocks.is_empty());
        for block in &fx.commit.blocks {
            let f_block = polys(&fx.commit.f_prime[block.index * width..(block.index + 1) * width]);
            let s = g_inv(&ring, b, delta, &f_block).unwrap();
            assert_eq!(s, polys(&block.s), "toy-n{n} block {}: s", block.index);
            assert_eq!(g_apply(&ring, b, delta, &s).unwrap(), f_block);
            let t = ring.mat_vec(&a, &s).unwrap();
            assert_eq!(
                t,
                polys(&block.t),
                "toy-n{n} block {}: t = A s",
                block.index
            );
            assert_eq!(
                g_inv(&ring, b, delta, &t).unwrap(),
                polys(&block.t_hat),
                "toy-n{n}: t_hat"
            );
        }

        // Ring-to-linear: w_hat, and per attempt and prover t0_hat and w0_hat.
        let r2l = &fx.ring_to_linear;
        assert_eq!(
            g_inv(&ring, b, delta, &polys(&r2l.w)).unwrap(),
            polys(&r2l.w_hat),
            "toy-n{n}: w_hat"
        );
        let mut checked = 0;
        for attempt in &r2l.attempts {
            for prover in &attempt.provers {
                let label = format!("toy-n{n} prover {}", prover.prover);
                let t0 = polys(&prover.t0);
                assert_eq!(
                    g_inv(&ring, b, delta, &t0).unwrap(),
                    polys(&prover.t0_hat),
                    "{label}: t0_hat"
                );
                let w0 = vec![Poly(prover.w0.clone())];
                assert_eq!(
                    g_inv(&ring, b, delta, &w0).unwrap(),
                    polys(&prover.w0_hat),
                    "{label}: w0_hat"
                );
                checked += 1;
            }
        }
        assert!(checked >= n);

        // z_hat = J^{-1}(z), read from its block of z'.
        let (_, start, len) = r2l
            .layout
            .blocks
            .iter()
            .find(|(name, _, _)| name == "z_hat")
            .cloned()
            .expect("z_hat block");
        let z: Vec<Poly> = r2l.z.iter().map(|c| ring.from_i64(c).unwrap()).collect();
        assert_eq!(len, z.len() * tau);
        let z_hat = j_inv(&ring, b, tau, &z).unwrap();
        assert_eq!(
            z_hat,
            polys(&r2l.z_prime[start..start + len]),
            "toy-n{n}: z_hat"
        );
        assert_eq!(j_apply(&ring, b, tau, &z_hat).unwrap(), z);
    }
}
