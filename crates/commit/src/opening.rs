//! $\mathsf{Open}(\mathsf{pp}, \mathbf{st}, \mathbf u, f, \mathbf c)$ as the draft states it since
//! commit `28721d6` (`6_Collaborative_Hachi.tex`, $\mathsf{Open}$; the oracle's
//! `commit.open_check`), a test oracle that reports every condition.

use crate::error::{Result, check_len, pow2};
use crate::polyconv::{f_hat_of_f_prime, fold_slots};
use crate::units::is_unit;
use hachi_co_core::{BlockState, Context, Layout, PartyState};
use hachi_co_math::{Ext, Poly, decompose};

/// The global relaxation factor $\mathbf c^\* = (1, \dots, 1) \in \mathcal Z$
/// (`6_Collaborative_Hachi.tex`:405): $2^r$ copies of the constant polynomial $1$, as signed
/// coefficient vectors.
pub fn c_star(layout: &Layout) -> Vec<Vec<i64>> {
    let mut one = vec![0i64; layout.d];
    one[0] = 1;
    vec![one; layout.n_blocks]
}

/// The named conditions of Open, each `true` when it holds; the names in [`OpenChecks::NAMES`]
/// are the oracle's (fixture key `open/checks`), in the order of the draft's text: the slack space,
/// the assembly by the partition $T$, then the checks (a) to (e).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenChecks {
    /// `c_in_slack_space`: $\mathbf c \in \mathcal Z$, i.e. every $c\_i$ is a unit of
    /// $\mathcal R\_q$ ([`is_unit`]) with $\Vert c\_i \Vert\_1 \le \bar\omega$ over centered
    /// coefficients (`ref/README.md`, choice 18).
    pub c_in_slack_space: bool,
    /// `states_hold_their_blocks`: the state of every prover $i$ has an entry for each block of
    /// its part $T\_i$ of the partition $T$ in $\mathsf{pp}$ ([`Layout::blocks_of`]), which the
    /// assembly reads. Entries for other blocks are ignored, and the blocks of $T\_0$ (held by
    /// nobody) are assembled as zero.
    pub states_hold_their_blocks: bool,
    /// `f_eq_f_bar`: (a) $f\_{\mathbf q} = \bar f\_{\mathbf q}$ for every
    /// $\mathbf q \in \lbrace 0,1 \rbrace^\ell$, comparing residues modulo $q$, with $\bar f$
    /// decoded from the $\mathbf f'\_{\mathbf i}$ of the states.
    pub f_eq_f_bar: bool,
    /// `f_prime_eq_G_s`: (b) $\mathbf f'\_{\mathbf i} = \mathbf G\_{2^m} \mathbf s\_{\mathrm{nat}(\mathbf i)}$
    /// for every block.
    pub f_prime_eq_g_s: bool,
    /// `A_s_eq_G_that`: (c) $\mathbf A \mathbf s\_{i'} = \mathbf G\_{n\_A} \hat{\mathbf t}\_{i'}$ for
    /// every $i' \in \[2^r\]$.
    pub a_s_eq_g_t_hat: bool,
    /// `norm_c_s_le_beta_bar`: (c) $\Vert c\_{i'} \mathbf s\_{i'} \Vert\_\infty \le \bar\beta$ for
    /// every $i'$, the product taken in $\mathcal R\_q$ and the norm over centered coefficients.
    pub norm_c_s_le_beta_bar: bool,
    /// `B_that_eq_u`: (d) $\mathbf B (\hat{\mathbf t}, \boldsymbol\rho^u) = \mathbf u$, comparing
    /// the reduced product with $\mathbf u$ as given.
    pub b_t_hat_eq_u: bool,
    /// `norm_that_le_gamma_bar`: (d) $\Vert (\hat{\mathbf t}, \boldsymbol\rho^u) \Vert\_\infty \le
    /// \bar\gamma$.
    pub norm_t_hat_le_gamma_bar: bool,
    /// `unheld_blocks_zero`: (e) $\mathbf f'\_{\mathsf{bit}(i')}$, $\mathbf s\_{i'}$ and
    /// $\hat{\mathbf t}\_{i'}$ are zero for every $i' \in T\_0$. The assembly sets them to zero, so
    /// this holds whenever the assembly is the draft's; it is kept to mirror the text.
    pub unheld_blocks_zero: bool,
}

impl OpenChecks {
    /// The oracle's names of the conditions, in its order (`ref/hachico_ref/commit.py`,
    /// `open_check`).
    pub const NAMES: [&'static str; 9] = [
        "c_in_slack_space",
        "states_hold_their_blocks",
        "f_eq_f_bar",
        "f_prime_eq_G_s",
        "A_s_eq_G_that",
        "norm_c_s_le_beta_bar",
        "B_that_eq_u",
        "norm_that_le_gamma_bar",
        "unheld_blocks_zero",
    ];

    /// The conditions with their oracle names, in the order of [`OpenChecks::NAMES`].
    pub fn named(&self) -> [(&'static str, bool); 9] {
        let values = [
            self.c_in_slack_space,
            self.states_hold_their_blocks,
            self.f_eq_f_bar,
            self.f_prime_eq_g_s,
            self.a_s_eq_g_t_hat,
            self.norm_c_s_le_beta_bar,
            self.b_t_hat_eq_u,
            self.norm_t_hat_le_gamma_bar,
            self.unheld_blocks_zero,
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
    /// $\mathsf{Open}(\mathsf{pp}, \mathbf{st}, \mathbf u, f, \mathbf c)$: `true` (output 1) iff
    /// every condition holds.
    pub bit: bool,
    /// The conditions.
    pub checks: OpenChecks,
    /// $\max\_i \Vert c\_i \mathbf s\_i \Vert\_\infty$.
    pub max_norm_c_s: u64,
    /// $\Vert (\hat{\mathbf t}, \boldsymbol\rho^u) \Vert\_\infty$.
    pub max_norm_t_hat: u64,
}

/// $\Vert c \Vert\_1$ over the centered representatives of the coefficients, saturating.
fn l1_norm(ctx_ring: &hachi_co_math::Ring, c: &Poly) -> Result<u128> {
    Ok(ctx_ring.to_centered(c)?.iter().fold(0u128, |acc, x| {
        acc.saturating_add(u128::from(x.unsigned_abs()))
    }))
}

/// $\mathsf{Open}(\mathsf{pp}, \mathbf{st}, \mathbf u, f, \mathbf c)$ as the draft states it since
/// commit `28721d6`, evaluating and reporting every condition, as the oracle's `open_check` does
/// (the draft outputs 0 as soon as $\mathbf c \notin \mathcal Z$; the bit is the same).
///
/// Inputs: the $n$ states (`states[p - 1]` = $\mathsf{st}\_p$; only their blocks and
/// $\boldsymbol\rho^u\_p$ are used), the commitment $\mathbf u \in \mathcal R\_q^{n\_B}$, the
/// coefficient vector $\mathbf f \in \mathbb F\_{q^k}^{2^\ell}$ of $f$, and the relaxation factor
/// $\mathbf c = (c\_1, \dots, c\_{2^r})$ as signed coefficient vectors read modulo $q$, with
/// `None` for $\mathbf c^\*$ ([`c_star`]).
///
/// 1. *Assemble*: prover $p$'s state supplies $(\mathbf f'\_{\mathsf{bit}(i')}, \mathbf s\_{i'},
///    \hat{\mathbf t}\_{i'})$ for the blocks $i'$ of its part $T\_p$ ([`Layout::blocks_of`]); a
///    missing block is assembled as zero and fails `states_hold_their_blocks`; the blocks of
///    $T\_0$ are zero; $\boldsymbol\rho^u$ from the states.
/// 2. *Decode*: $\hat{\mathbf f} = \psi^{-1}(\mathbf f')$ ([`f_hat_of_f_prime`]) from the
///    $\mathbf f'$ of the states, $\bar f\_{\mathbf q} = \sum\_{\mathbf p} \hat f\_{\mathbf p \Vert
///    \mathbf q}$ ([`fold_slots`]).
/// 3. *Conditions* (a) to (e): see [`OpenChecks`].
///
/// # Errors
/// [`Error::Shape`](crate::Error::Shape) unless there are $n$ states, $n\_B$ polynomials in
/// $\mathbf u$ (with $d$ coefficients each), $2^\ell$ entries in $\mathbf f$ and $2^r$ rows of
/// $d$ coefficients in $\mathbf c$; [`Error::Core`](crate::Error::Core) for a block of the wrong
/// shape (index $\ge 2^r$ included).
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
    for state in states {
        check_len("rho^u_p (m_u entries)", l.m_u, state.rho_u.len())?;
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

    // The slack space Z.
    let mut c_in_slack_space = true;
    for ci in &c_polys {
        c_in_slack_space &=
            l1_norm(ring, ci)? <= u128::from(ctx.params.omega_bar) && is_unit(ring, ci)?;
    }

    // Step 1: assemble the opening by the partition T: prover p supplies the blocks of T_p.
    let mut held: Vec<Option<&BlockState>> = vec![None; l.n_blocks];
    let mut states_hold_their_blocks = true;
    for (p0, state) in states.iter().enumerate() {
        for i in l.blocks_of(p0 + 1) {
            match state.blocks.iter().find(|blk| blk.index == i) {
                Some(blk) => held[i] = Some(blk),
                None => states_hold_their_blocks = false,
            }
        }
    }
    let zero_f = vec![ring.zero(); l.block_len];
    let zero_s = vec![ring.zero(); l.width_a];
    let zero_t_hat = vec![ring.zero(); l.n_a * delta];
    let f_prime_of = |i: usize| held[i].map_or(&zero_f, |blk| &blk.f_prime);
    let s_of = |i: usize| held[i].map_or(&zero_s, |blk| &blk.s);
    let t_hat_of = |i: usize| held[i].map_or(&zero_t_hat, |blk| &blk.t_hat);

    // Step 2: decode the polynomial from the f' of the states.
    let mut f_prime = Vec::with_capacity(l.n_blocks * l.block_len);
    for i in 0..l.n_blocks {
        f_prime.extend(f_prime_of(i).iter().cloned());
    }
    let f_bar = fold_slots(ctx, &f_hat_of_f_prime(ctx, &f_prime)?)?;

    // Step 3: the conditions (a) to (e).
    let f_eq_f_bar = f.iter().zip(&f_bar).all(|(x, y)| field.reduce(x) == *y);
    let mut f_prime_eq_g_s = true;
    let mut a_s_eq_g_t_hat = true;
    let mut max_norm_c_s = 0u64;
    for (i, ci) in c_polys.iter().enumerate() {
        let g_s = decompose::g_apply(ring, b, delta, s_of(i))?;
        f_prime_eq_g_s &= f_prime_of(i).as_slice() == g_s.as_slice();
        let lhs = ring.mat_vec(&ctx.pp().a, s_of(i))?;
        let rhs = decompose::g_apply(ring, b, delta, t_hat_of(i))?;
        a_s_eq_g_t_hat &= lhs == rhs;
        for s_e in s_of(i) {
            max_norm_c_s = max_norm_c_s.max(ring.inf_norm(&ring.mul(ci, s_e)?)?);
        }
    }
    let t_rho: Vec<Poly> = (0..l.n_blocks)
        .flat_map(|i| t_hat_of(i).iter().cloned())
        .chain(states.iter().flat_map(|s| s.rho_u.iter().cloned()))
        .collect();
    let b_t_hat_eq_u = ring.mat_vec(&ctx.pp().b, &t_rho)? == u;
    let max_norm_t_hat = hachi_co_core::short::inf_norm_polys(ring, &t_rho)?;
    let is_zero = |polys: &[Poly]| polys.iter().all(|p| p.0.iter().all(|x| *x == 0));
    let unheld_blocks_zero = (0..l.n_blocks)
        .filter(|i| l.owner_of_block(*i).is_none())
        .all(|i| is_zero(f_prime_of(i)) && is_zero(s_of(i)) && is_zero(t_hat_of(i)));

    let checks = OpenChecks {
        c_in_slack_space,
        states_hold_their_blocks,
        f_eq_f_bar,
        f_prime_eq_g_s,
        a_s_eq_g_t_hat,
        norm_c_s_le_beta_bar: u128::from(max_norm_c_s) <= ctx.params.beta_bar,
        b_t_hat_eq_u,
        norm_t_hat_le_gamma_bar: max_norm_t_hat <= ctx.params.gamma_bar,
        unheld_blocks_zero,
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
            states_hold_their_blocks: true,
            f_eq_f_bar: true,
            f_prime_eq_g_s: true,
            a_s_eq_g_t_hat: true,
            norm_c_s_le_beta_bar: true,
            b_t_hat_eq_u: true,
            norm_t_hat_le_gamma_bar: true,
            unheld_blocks_zero: true,
        };
        assert!(checks.all());
        assert!(checks.failing().is_empty());
        checks.f_eq_f_bar = false;
        checks.a_s_eq_g_t_hat = false;
        assert!(!checks.all());
        assert_eq!(checks.failing(), vec!["f_eq_f_bar", "A_s_eq_G_that"]);
        assert_eq!(checks.named()[4], ("A_s_eq_G_that", false));
    }
}
