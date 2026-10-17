//! $\mathsf{Open}(\mathsf{pp}, \mathbf{st}, \mathbf u, f, c)$ (`6_Collaborative_Hachi.tex`:393-404;
//! the oracle's `commit.open_check`), a test oracle that reports every condition.

use crate::error::{Result, check_len, pow2};
use crate::polyconv::{f_hat_of_f_prime, fold_slots};
use crate::units::is_unit;
use hachi_co_core::{BlockState, Context, Layout, PartyState};
use hachi_co_math::{Ext, Poly, decompose};

/// The global relaxation factor $c^\* = (1, \dots, 1) \in \mathcal Z$
/// (`6_Collaborative_Hachi.tex`:395): $2^r$ copies of the constant polynomial $1$, as signed
/// coefficient vectors.
pub fn c_star(layout: &Layout) -> Vec<Vec<i64>> {
    let mut one = vec![0i64; layout.d];
    one[0] = 1;
    vec![one; layout.n_blocks]
}

/// The named conditions of Open (`6_Collaborative_Hachi.tex`:393-404), each `true` when it holds;
/// the names in [`OpenChecks::NAMES`] are the oracle's (fixture key `open/checks`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenChecks {
    /// `c_in_slack_space`: $c \in \mathcal Z$, i.e. every $c\_i$ is a unit of $\mathcal R\_q$
    /// ([`is_unit`]) with $\Vert c\_i \Vert\_1 \le \bar\omega$ over centered coefficients
    /// (line 394; `ref/README.md`, choice 18).
    pub c_in_slack_space: bool,
    /// `no_block_in_several_states`: no block $\mathbf i$ has entries in several states
    /// (line 397). **Choice:** two entries for the same block in one state count as several.
    pub no_block_in_several_states: bool,
    /// `A_s_eq_G_that`: $\mathbf A \mathbf s\_i = \mathbf G\_{n\_A} \hat{\mathbf t}\_i$ for every
    /// $i \in \[2^r\]$ (line 401).
    pub a_s_eq_g_t_hat: bool,
    /// `norm_c_s_le_beta_bar`: $\Vert c\_i \mathbf s\_i \Vert\_\infty \le \bar\beta$ for every $i$,
    /// the product taken in $\mathcal R\_q$ and the norm over centered coefficients (line 401).
    pub norm_c_s_le_beta_bar: bool,
    /// `B_that_eq_u`: $\mathbf B \hat{\mathbf t} = \mathbf u$ (line 402), comparing the reduced
    /// product with $\mathbf u$ as given.
    pub b_t_hat_eq_u: bool,
    /// `norm_that_le_gamma_bar`: $\Vert \hat{\mathbf t} \Vert\_\infty \le \bar\gamma$ (line 402).
    pub norm_t_hat_le_gamma_bar: bool,
    /// `f_eq_f_bar`: $f\_{\mathbf q} = \bar f\_{\mathbf q}$ for every $\mathbf q \in \lbrace 0,1
    /// \rbrace^\ell$, comparing residues modulo $q$ (line 403).
    pub f_eq_f_bar: bool,
}

impl OpenChecks {
    /// The oracle's names of the conditions, in its order (`ref/hachico_ref/commit.py`,
    /// `open_check`).
    pub const NAMES: [&'static str; 7] = [
        "c_in_slack_space",
        "no_block_in_several_states",
        "A_s_eq_G_that",
        "norm_c_s_le_beta_bar",
        "B_that_eq_u",
        "norm_that_le_gamma_bar",
        "f_eq_f_bar",
    ];

    /// The conditions with their oracle names, in the order of [`OpenChecks::NAMES`].
    pub fn named(&self) -> [(&'static str, bool); 7] {
        let values = [
            self.c_in_slack_space,
            self.no_block_in_several_states,
            self.a_s_eq_g_t_hat,
            self.norm_c_s_le_beta_bar,
            self.b_t_hat_eq_u,
            self.norm_t_hat_le_gamma_bar,
            self.f_eq_f_bar,
        ];
        core::array::from_fn(|i| (Self::NAMES[i], values[i]))
    }

    /// Whether every condition holds.
    pub fn all(&self) -> bool {
        self.named().iter().all(|(_, ok)| *ok)
    }

    /// The names of the conditions that fail, in the order of [`OpenChecks::NAMES`] (the form of
    /// the fixture key `negative_tests/open/*/failing`).
    pub fn failing(&self) -> Vec<&'static str> {
        self.named()
            .iter()
            .filter(|(_, ok)| !*ok)
            .map(|(name, _)| *name)
            .collect()
    }
}

/// The result of [`open`]: the output bit and every condition, plus the two norms (fixture keys
/// `open/bit`, `open/checks`, `open/max_norm_c_s`, `open/max_norm_t_hat`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenReport {
    /// $\mathsf{Open}(\mathsf{pp}, \mathbf{st}, \mathbf u, f, c)$: `true` (output 1) iff every
    /// condition holds.
    pub bit: bool,
    /// The conditions.
    pub checks: OpenChecks,
    /// $\max\_i \Vert c\_i \mathbf s\_i \Vert\_\infty$.
    pub max_norm_c_s: u64,
    /// $\Vert \hat{\mathbf t} \Vert\_\infty$.
    pub max_norm_t_hat: u64,
}

/// $\Vert c \Vert\_1$ over the centered representatives of the coefficients, saturating.
fn l1_norm(ctx_ring: &hachi_co_math::Ring, c: &Poly) -> Result<u128> {
    Ok(ctx_ring.to_centered(c)?.iter().fold(0u128, |acc, x| {
        acc.saturating_add(u128::from(x.unsigned_abs()))
    }))
}

/// $\mathsf{Open}(\mathsf{pp}, \mathbf{st}, \mathbf u, f, c)$ (`6_Collaborative_Hachi.tex`:
/// 393-404), evaluating and reporting every condition, as the oracle's `open_check` does (the
/// draft outputs 0 as soon as $c \notin \mathcal Z$; the bit is the same).
///
/// Inputs: the $n$ states (`states[p - 1]` = $\mathsf{st}\_p$; only their blocks are used), the
/// commitment $\mathbf u \in \mathcal R\_q^{n\_B}$, the coefficient vector $\mathbf f \in
/// \mathbb F\_{q^k}^{2^\ell}$ of $f$, and the relaxation factor $c = (c\_1, \dots, c\_{2^r})$ as
/// signed coefficient vectors read modulo $q$, with `None` for $c^\*$ ([`c_star`]).
///
/// 1. *Assemble* (line 397): $\mathbf s\_i, \hat{\mathbf t}\_i$ from the state holding block $i$;
///    zero if none does; if several do, the condition `no_block_in_several_states` fails and the
///    first entry (in party order, then in the order of the state) is used, as in the oracle.
/// 2. *Decode* (line 398): $(f'\_{\mathbf i \Vert \mathbf j})\_{\mathbf j} = \mathbf
///    G\_{2^m} \mathbf s\_i$, $\hat{\mathbf f} = \psi^{-1}(\mathbf f')$
///    ([`f_hat_of_f_prime`]), $\bar f\_{\mathbf q} = \sum\_{\mathbf p} \hat f\_{\mathbf p \Vert
///    \mathbf q}$ ([`fold_slots`]).
/// 3. *Conditions* (lines 399-404): see [`OpenChecks`].
///
/// # Errors
/// [`Error::Shape`](crate::Error::Shape) unless there are $n$ states, $n\_B$ polynomials in
/// $\mathbf u$ (with $d$ coefficients each), $2^\ell$ entries in $\mathbf f$ and $2^r$ rows of
/// $d$ coefficients in $c$; [`Error::Core`](crate::Error::Core) for a block of the wrong shape
/// (index $\ge 2^r$ included).
pub fn open<const K: usize>(
    ctx: &Context<K>,
    states: &[PartyState<K>],
    u: &[Poly],
    f: &[Ext<K>],
    c: Option<&[Vec<i64>]>,
) -> Result<OpenReport> {
    let (l, ring, field) = (&ctx.layout, &ctx.ring, ctx.field());
    let (b, delta) = (ctx.b(), l.delta);
    check_len("states (one per party)", l.n, states.len())?;
    for block in states.iter().flat_map(|s| &s.blocks) {
        block.validate(l)?;
    }
    check_len("u (n_B entries)", l.n_b, u.len())?;
    for entry in u {
        check_len("u (d coefficients per entry)", l.d, entry.0.len())?;
    }
    check_len("f (2^l coefficients)", pow2(l.ell, "2^l")?, f.len())?;
    let c_rows = match c {
        Some(rows) => rows.to_vec(),
        None => c_star(l),
    };
    check_len("c (2^r entries)", l.n_blocks, c_rows.len())?;
    let c_polys = c_rows
        .iter()
        .map(|row| {
            check_len("c_i (d coefficients)", l.d, row.len())?;
            Ok(ring.from_i64(row)?)
        })
        .collect::<Result<Vec<_>>>()?;

    // The slack space Z (line 394).
    let mut c_in_slack_space = true;
    for ci in &c_polys {
        c_in_slack_space &=
            l1_norm(ring, ci)? <= u128::from(ctx.params.omega_bar) && is_unit(ring, ci)?;
    }

    // Step 1: assemble the opening.
    let mut held: Vec<Option<&BlockState>> = vec![None; l.n_blocks];
    let mut no_block_in_several_states = true;
    for block in states.iter().flat_map(|s| &s.blocks) {
        match held[block.index] {
            Some(_) => no_block_in_several_states = false,
            None => held[block.index] = Some(block),
        }
    }
    let zero_s = vec![ring.zero(); l.width_a];
    let zero_t_hat = vec![ring.zero(); l.n_a * delta];
    let s_of = |i: usize| held[i].map_or(&zero_s, |blk| &blk.s);
    let t_hat_of = |i: usize| held[i].map_or(&zero_t_hat, |blk| &blk.t_hat);

    // Step 2: decode the message.
    let mut f_prime = Vec::with_capacity(l.n_blocks * l.block_len);
    for i in 0..l.n_blocks {
        f_prime.extend(decompose::g_apply(ring, b, delta, s_of(i))?);
    }
    let f_bar = fold_slots(ctx, &f_hat_of_f_prime(ctx, &f_prime)?)?;

    // Step 3: the conditions.
    let mut a_s_eq_g_t_hat = true;
    let mut max_norm_c_s = 0u64;
    for (i, ci) in c_polys.iter().enumerate() {
        let lhs = ring.mat_vec(&ctx.pp.a, s_of(i))?;
        let rhs = decompose::g_apply(ring, b, delta, t_hat_of(i))?;
        a_s_eq_g_t_hat &= lhs == rhs;
        for s_e in s_of(i) {
            max_norm_c_s = max_norm_c_s.max(ring.inf_norm(&ring.mul(ci, s_e)?)?);
        }
    }
    let t_hat: Vec<Poly> = (0..l.n_blocks)
        .flat_map(|i| t_hat_of(i).iter().cloned())
        .collect();
    let b_t_hat_eq_u = ring.mat_vec(&ctx.pp.b, &t_hat)? == u;
    let max_norm_t_hat = hachi_co_core::short::inf_norm_polys(ring, &t_hat)?;
    let f_eq_f_bar = f.iter().zip(&f_bar).all(|(x, y)| field.reduce(x) == *y);

    let checks = OpenChecks {
        c_in_slack_space,
        no_block_in_several_states,
        a_s_eq_g_t_hat,
        norm_c_s_le_beta_bar: u128::from(max_norm_c_s) <= ctx.params.beta_bar,
        b_t_hat_eq_u,
        norm_t_hat_le_gamma_bar: max_norm_t_hat <= ctx.params.gamma_bar,
        f_eq_f_bar,
    };
    Ok(OpenReport {
        bit: checks.all(),
        checks,
        max_norm_c_s,
        max_norm_t_hat,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_failing_follow_the_oracle_order() {
        let mut checks = OpenChecks {
            c_in_slack_space: true,
            no_block_in_several_states: true,
            a_s_eq_g_t_hat: true,
            norm_c_s_le_beta_bar: true,
            b_t_hat_eq_u: true,
            norm_t_hat_le_gamma_bar: true,
            f_eq_f_bar: true,
        };
        assert!(checks.all());
        assert!(checks.failing().is_empty());
        checks.f_eq_f_bar = false;
        checks.a_s_eq_g_t_hat = false;
        assert!(!checks.all());
        assert_eq!(checks.failing(), vec!["A_s_eq_G_that", "f_eq_f_bar"]);
        assert_eq!(checks.named()[2], ("A_s_eq_G_that", false));
    }
}
