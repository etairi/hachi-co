//! The inputs of the building blocks as the oracle recorded them (feature `fixtures`): every
//! party's state after Commit, the public vectors and the shares of $\Pi\_{\mathsf{conv}}$, the
//! commitment and the number of attempts. The block crates' tests build their inputs from
//! these, so that each crate is tested on the oracle's values without running the blocks before
//! it.

use crate::{BlockState, Context, Error, Fixture, PartyState, Poly, PublicVectors, Result};

impl Fixture {
    /// Party $p$'s state after Commit (`6_Collaborative_Hachi.tex`:390) as recorded: its share
    /// $\langle \mathbf f \rangle\_p$ (`commit/f_shares`), and for every block $i$ it owns
    /// ([`Layout::blocks_of`](crate::Layout::blocks_of)) the slice of $\mathbf f'$
    /// (`commit/f_prime`) and $\mathbf s\_i, \mathbf t\_i, \hat{\mathbf t}\_i$ from the entry of
    /// `commit/blocks` with that index. Every recorded block of $p$ must be one it owns and vice
    /// versa; the state is validated.
    ///
    /// # Errors
    /// [`Error::Fixture`] for a missing or misattributed block; the decoders' and
    /// [`PartyState::validate`]'s errors.
    pub fn party_state<const K: usize>(&self, ctx: &Context<K>, p: usize) -> Result<PartyState<K>> {
        let l = &ctx.layout;
        let f_prime = self.polys("/commit/f_prime")?;
        let recorded = self.get("/commit/blocks")?.as_array().map_or(0, Vec::len);
        let mut owners = Vec::with_capacity(recorded);
        for j in 0..recorded {
            owners.push((
                self.usize(&format!("/commit/blocks/{j}/index"))?,
                self.usize(&format!("/commit/blocks/{j}/owner"))?,
            ));
        }
        let mut blocks = Vec::new();
        for i in l.blocks_of(p) {
            let j = owners
                .iter()
                .position(|&(index, _)| index == i)
                .ok_or_else(|| Error::Fixture(format!("block {i} of party {p} is not recorded")))?;
            if owners[j].1 != p {
                return Err(Error::Fixture(format!(
                    "block {i} is recorded for party {}, the layout gives it to party {p}",
                    owners[j].1
                )));
            }
            let key = |k: &str| self.polys(&format!("/commit/blocks/{j}/{k}"));
            let slice = f_prime
                .get(i * l.block_len..(i + 1) * l.block_len)
                .ok_or_else(|| Error::Fixture(format!("commit/f_prime has no block {i}")))?;
            blocks.push(BlockState {
                index: i,
                f_prime: slice.to_vec(),
                s: key("s")?,
                t: key("t")?,
                t_hat: key("t_hat")?,
            });
        }
        let recorded_for_p = owners.iter().filter(|&&(_, owner)| owner == p).count();
        if recorded_for_p != blocks.len() {
            return Err(Error::Fixture(format!(
                "{recorded_for_p} blocks are recorded for party {p}, the layout gives it {}",
                blocks.len()
            )));
        }
        let state = PartyState {
            party: p,
            share: self.exts(&format!("/commit/f_shares/{}", p - 1))?,
            blocks,
        };
        state.validate(l)?;
        Ok(state)
    }

    /// The public vectors of $\Pi\_{\mathsf{conv}}$ as recorded (`conv/x_prime`, `b`, `a`, `d`,
    /// `v_point`), with $\mathbf b$ and $\mathbf a$ embedded in $\mathcal R\_q$; validated.
    ///
    /// # Errors
    /// The decoders' and [`PublicVectors::validate`]'s errors.
    pub fn public_vectors<const K: usize>(&self, ctx: &Context<K>) -> Result<PublicVectors<K>> {
        let b = self.exts::<K>("/conv/b")?;
        let a = self.exts::<K>("/conv/a")?;
        let pv = PublicVectors {
            x_prime: self.exts("/conv/x_prime")?,
            b_ring: b.iter().map(|x| ctx.sub.embed(x)).collect(),
            a_ring: a.iter().map(|x| ctx.sub.embed(x)).collect(),
            b,
            a,
            d: self.exts("/conv/d")?,
            v_point: self.poly("/conv/v_point")?,
        };
        pv.validate(&ctx.layout)?;
        Ok(pv)
    }

    /// Party $p$'s share $\langle u \rangle\_p$ of $\Pi\_{\mathsf{conv}}$ (`conv/u_shares`).
    ///
    /// # Errors
    /// The decoder's errors.
    pub fn u_share(&self, p: usize) -> Result<Poly> {
        self.poly(&format!("/conv/u_shares/{}", p.wrapping_sub(1)))
    }

    /// The commitment $\mathbf u$ (`commit/u`).
    ///
    /// # Errors
    /// The decoder's errors.
    pub fn commitment(&self) -> Result<Vec<Poly>> {
        self.polys("/commit/u")
    }

    /// The number of recorded attempts of the ring-to-linear step
    /// (`ring_to_linear/attempts`).
    ///
    /// # Errors
    /// [`Error::Fixture`] if the entry is not an array.
    pub fn attempt_count(&self) -> Result<usize> {
        self.get("/ring_to_linear/attempts")?
            .as_array()
            .map(Vec::len)
            .ok_or_else(|| Error::Fixture("ring_to_linear/attempts is not an array".into()))
    }
}
