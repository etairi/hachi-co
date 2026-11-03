//! The shared instance of the cross-check against the Sage spec: one masked opening of each kind (a
//! polynomial over the field, a vector and a merged commitment) at the toy choices with masks, written
//! to `tests/fixtures/masked_spec.json` with the prover's secrets, its challenges and the values of
//! step 1 and of the masked level. `tests/fixtures/check_masked_spec.sage` recomputes the equations
//! of `hachi-pcs.tex`, Appendix A.2, on it with Sage's own arithmetic.
//!
//! The values come from the openings themselves: the test replays each opening's transcript up to
//! the first iteration, and the replayed iteration must equal the proof's.
//!
//! `HACHI_WRITE_FIXTURES=1 cargo test -p hachi masked_spec` rewrites the file. Otherwise the test
//! fails when the file differs from what the code computes, and then the file and the Sage check
//! must be redone.

use super::*;
use crate::iteration::{self, MaskedLevel};
use crate::params::IterParams;

const K: usize = 4;
/// The modulus of Hachi-Co's toy-n1 preset (24 bits).
const Q: u64 = 8_388_637;
const CONTEXT: [u8; 32] = [5u8; 32];
const COMMIT_SEED: [u8; 32] = [6u8; 32];
const OPEN_SEED: [u8; 32] = [7u8; 32];

/// `len` pseudorandom field elements, determined by `tag`.
fn exts(tag: &str, len: usize) -> Vec<Ext<K>> {
    let mut prg = ShakePrg::new(&[8u8; 32], tag.as_bytes(), 0);
    uniform_mod_q(&mut prg, Q, len * K)
        .unwrap()
        .as_chunks::<K>()
        .0
        .iter()
        .map(|c| Ext(*c))
        .collect()
}

/// The choices of every instance: the toy ones with masks, and one iteration.
fn pcs() -> HachiPcs<K> {
    let choices = HachiChoices {
        fixed_iterations: Some(1),
        ..HachiChoices::toy_hiding(16)
    };
    HachiPcs::<K>::new(Q, choices).unwrap()
}

/// A JSON list.
fn list<T>(items: &[T], f: impl Fn(&T) -> String) -> String {
    format!("[{}]", items.iter().map(f).collect::<Vec<_>>().join(","))
}

/// A ring element as its $d$ residues, a field element as its $K$ coordinates.
fn poly(p: &Poly) -> String {
    list(&p.0, u64::to_string)
}
fn polys(ps: &[Poly]) -> String {
    list(ps, poly)
}
fn matrix(m: &[Vec<Poly>]) -> String {
    list(m, |row| polys(row))
}
fn ext(x: &Ext<K>) -> String {
    list(&x.0, u64::to_string)
}
fn exts_json(xs: &[Ext<K>]) -> String {
    list(xs, ext)
}

/// A JSON object, one key per line, in the order of insertion.
#[derive(Default)]
struct Object(Vec<(&'static str, String)>);

impl Object {
    fn put(&mut self, key: &'static str, value: String) {
        self.0.push((key, value));
    }

    fn num(&mut self, key: &'static str, value: impl ToString) {
        self.put(key, value.to_string());
    }

    fn text(&self) -> String {
        let fields: Vec<String> = self
            .0
            .iter()
            .map(|(k, v)| format!("      \"{k}\": {v}"))
            .collect();
        format!("    {{\n{}\n    }}", fields.join(",\n"))
    }
}

/// Level 0 of an opening whose transcript `fs` stands before the iterations: the replayed
/// iteration must be the proof's first one, and [`iteration::masked_level`] gives its values.
fn level_zero(
    pcs: &HachiPcs<K>,
    state: &State<K>,
    reduction: Reduction,
    claim: &Claim,
    fs: &Fs,
    proof: &Proof<K>,
) -> (IterParams, MaskedLevel) {
    let schedule = pcs.schedule(state.committed.layout, reduction).unwrap();
    let params = schedule.iterations[0];
    let (first, _, _) = iteration::prove(
        pcs.ctx(),
        0,
        &params,
        &state.committed,
        claim,
        &mut fs.clone(),
        &OPEN_SEED,
    )
    .unwrap();
    assert_eq!(first, proof.iterations[0], "the replay is the opening");
    let level = iteration::masked_level(
        pcs.ctx(),
        0,
        &params,
        &state.committed,
        claim,
        &mut fs.clone(),
        &OPEN_SEED,
    )
    .unwrap();
    assert_eq!(level.v, first.v);
    (params, level)
}

/// The fields every instance has: the choices, the commitment with its secrets, the matrices of
/// level 0, the first claim and the masked level.
fn common(
    o: &mut Object,
    pcs: &HachiPcs<K>,
    state: &State<K>,
    claim: &Claim,
    params: &IterParams,
    level: &MaskedLevel,
) {
    let (ctx, layout) = (pcs.ctx(), &state.committed.layout);
    let hiding = ctx.hiding().unwrap();
    let c = &ctx.choices;
    o.num("q", ctx.q);
    o.num("d", ctx.d());
    o.num("k", K);
    o.num("b", c.b);
    o.num("omega", c.omega);
    o.num("delta", params.delta);
    o.num("n_a", params.n_a);
    o.num("nu", hiding.nu);
    o.num("n_b", c.n_b);
    o.num("n_d", c.n_d);
    o.num("phi", hiding.phi);
    // The statistical security parameter, not the coin lambda of a merged opening.
    o.num("sec_lambda", hiding.lambda);
    o.num("r", layout.r);
    o.num("m", layout.m);
    o.num("extra", layout.extra);
    o.num("width", layout.width(params.delta));
    o.num("sigma", params.sigma);
    o.num("beta", params.beta);
    o.num("tau", params.tau);
    o.put("chunks", list(&state.committed.message, |ch| polys(ch)));
    let digits: Vec<Vec<Poly>> = state
        .committed
        .message
        .iter()
        .map(|ch| {
            Committed::digits(ctx, layout, params.delta, ch)
                .unwrap()
                .into_owned()
        })
        .collect();
    o.put("s", list(&digits, |s| polys(s)));
    o.put("rho", list(&state.committed.r, |r| polys(r)));
    o.put("t", list(&state.committed.t, |t| polys(t)));
    o.put("t_hat", polys(&state.committed.t_hat));
    o.put("u", polys(&state.committed.u));
    o.put("A", matrix(&level.a_mat));
    o.put("A1p", matrix(&level.a_rand));
    o.put("B", matrix(&level.b_mat));
    o.put("D", matrix(&level.d_mat));
    o.put("claim_b", polys(&claim.b));
    o.put("claim_a", polys(&claim.a));
    o.put("claim_target", poly(&claim.target));
    o.put("w", polys(&level.w));
    o.put("w_hat", polys(&level.w_hat));
    o.put("v", polys(&level.v));
    o.put("c", polys(&level.c));
    o.put("z", polys(&level.z));
    o.put("z_rho", polys(&level.z_r));
    o.put("w0", poly(&level.w0));
    o.put("w0_hat", polys(&level.w0_hat));
    o.put("t0", polys(&level.t0));
    o.put("t0_hat", polys(&level.t0_hat));
}

/// A masked polynomial over the field in 6 variables.
fn multilinear() -> String {
    let pcs = pcs();
    let vars = 6;
    let coeffs = exts("f", 1 << vars);
    let point = exts("x", vars);
    let (com, state) = pcs.commit_multilinear_ext(&coeffs, &COMMIT_SEED).unwrap();
    let (value, proof) = pcs
        .prove_multilinear_ext(&state, &point, &CONTEXT, &OPEN_SEED)
        .unwrap();
    pcs.verify_multilinear_ext(&com, vars, &point, &value, &proof, &CONTEXT)
        .unwrap();
    // Step 1, as the prover runs it.
    let (b, a, v) = pcs.ring_weights(&state.committed.layout, &point).unwrap();
    let mut fs = pcs.statement(&CONTEXT, 3, vars, &state.committed.u, &point, &value);
    let (mask_eval, claim) = pcs.masked_claim(&state, b, a, &v, &value, &mut fs).unwrap();
    assert_eq!(Some(mask_eval), proof.mask_eval);
    assert_eq!(claim.target, proof.ring_value);
    let x_star = pcs.ctx().sub.extract(claim.a.last().unwrap()).unwrap();
    let (params, level) = level_zero(&pcs, &state, Reduction::Masked, &claim, &fs, &proof);
    let mut o = Object::default();
    o.put("kind", "\"multilinear\"".into());
    o.num("vars", vars);
    o.put("coeffs", exts_json(&coeffs));
    o.put("point", exts_json(&point));
    o.put("value", ext(&value));
    o.put("v_trace", poly(&v));
    o.put("y_mask", ext(&mask_eval));
    o.put("x_star", ext(&x_star));
    common(&mut o, &pcs, &state, &claim, &params, &level);
    o.text()
}

/// A masked vector of 20 entries.
fn vector() -> String {
    let pcs = pcs();
    let len = 20;
    let (values, weights) = (exts("s", len), exts("e", len));
    let (com, state) = pcs.commit_vector(&values, &COMMIT_SEED).unwrap();
    let (value, proof) = pcs
        .prove_inner_product(&state, &weights, &CONTEXT, &OPEN_SEED)
        .unwrap();
    pcs.verify_inner_product(&com, len, &weights, &value, &proof, &CONTEXT)
        .unwrap();
    let ring = &pcs.ctx().ring;
    let a = pcs
        .vector_weights(&state.committed.layout, &weights)
        .unwrap();
    let one = ring.monomial(0);
    let mut fs = pcs.statement(&CONTEXT, 2, len, &state.committed.u, &weights, &value);
    let (mask_eval, claim) = pcs
        .masked_claim(&state, vec![one.clone()], a, &one, &value, &mut fs)
        .unwrap();
    assert_eq!(Some(mask_eval), proof.mask_eval);
    assert_eq!(claim.target, proof.ring_value);
    let x_star = pcs.ctx().sub.extract(claim.a.last().unwrap()).unwrap();
    let (params, level) = level_zero(&pcs, &state, Reduction::Masked, &claim, &fs, &proof);
    let mut o = Object::default();
    o.put("kind", "\"vector\"".into());
    o.num("len", len);
    o.put("values", exts_json(&values));
    o.put("weights", exts_json(&weights));
    o.put("value", ext(&value));
    o.put("y_mask", ext(&mask_eval));
    o.put("x_star", ext(&x_star));
    common(&mut o, &pcs, &state, &claim, &params, &level);
    o.text()
}

/// A merged commitment: a polynomial over the field in 6 variables and vectors of 9 and 5 entries.
fn merged() -> String {
    let pcs = pcs();
    let vars = 6;
    let coeffs = exts("f", 1 << vars);
    let (g0, g1) = (exts("g0", 9), exts("g1", 5));
    let point = exts("x", vars);
    let (e0, e1) = (exts("e0", 9), exts("e1", 5));
    let (com, state) = pcs
        .commit_merged(&coeffs, [&g0, &g1], &COMMIT_SEED)
        .unwrap();
    let (values, proof) = pcs
        .prove_merged(&state, &point, [&e0, &e1], &CONTEXT, &OPEN_SEED)
        .unwrap();
    pcs.verify_merged(&com, vars, &point, [&e0, &e1], &values, &proof, &CONTEXT)
        .unwrap();
    // Step 1 from the proof's values, as the verifier replays it.
    let layout = state.committed.layout;
    let (b, a, v) = pcs.ring_weights(&layout, &point).unwrap();
    let mut fs = pcs.merged_statement(
        &CONTEXT,
        vars,
        &state.committed.u,
        &point,
        [&e0, &e1],
        &values,
    );
    let mask_evals = [
        proof.mask_eval.unwrap(),
        proof.blinder_mask_evals[0],
        proof.blinder_mask_evals[1],
    ];
    fs.exts(LABEL_MASK_EVAL, &mask_evals);
    let x_star = pcs.mask_coin(&mut fs).unwrap();
    let targets = [
        proof.ring_value.clone(),
        proof.blinder_ring_values[0].clone(),
        proof.blinder_ring_values[1].clone(),
    ];
    fs.polys(LABEL_RING_VALUE, &targets);
    let lambda = pcs.nonzero_coin(&mut fs, LABEL_MERGED_COIN).unwrap();
    let vector_weights = [
        pcs.block_weights(&e0).unwrap(),
        pcs.block_weights(&e1).unwrap(),
    ];
    let claim = pcs
        .merged_claim(&layout, b, a, &vector_weights, &x_star, &lambda, &targets)
        .unwrap();
    let (params, level) = level_zero(&pcs, &state, Reduction::Merged, &claim, &fs, &proof);
    let mut o = Object::default();
    o.put("kind", "\"merged\"".into());
    o.num("vars", vars);
    o.put("coeffs", exts_json(&coeffs));
    o.put("g0", exts_json(&g0));
    o.put("g1", exts_json(&g1));
    o.put("point", exts_json(&point));
    o.put("e0", exts_json(&e0));
    o.put("e1", exts_json(&e1));
    o.put("values", exts_json(&values));
    o.put("v_trace", poly(&v));
    o.put("y_masks", exts_json(&mask_evals));
    o.put("x_star", ext(&x_star));
    o.put("Y_stars", polys(&targets));
    o.put("lambda", ext(&lambda));
    common(&mut o, &pcs, &state, &claim, &params, &level);
    o.text()
}

#[test]
fn the_masked_spec_fixture_is_current() {
    let text = format!(
        "{{\n  \"generator\": \"crates/hachi/src/pcs/spec_fixture.rs\",\n  \"instances\": [\n{},\n{},\n{}\n  ]\n}}\n",
        multilinear(),
        vector(),
        merged()
    );
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/masked_spec.json");
    if std::env::var_os("HACHI_WRITE_FIXTURES").is_some() {
        std::fs::write(&path, &text).unwrap();
    }
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(
        committed == text,
        "{} differs from what the code computes: rerun with HACHI_WRITE_FIXTURES=1, then the Sage check",
        path.display()
    );
}
