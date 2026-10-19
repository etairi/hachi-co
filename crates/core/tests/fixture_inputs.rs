//! The recorded inputs of the building blocks (`Fixture::party_state`, `public_vectors`,
//! `u_share`, `commitment`, `attempt_count`) for every fixture: every party's state validates and
//! holds exactly the blocks the layout gives it, the blocks of all parties are the owned blocks,
//! the public vectors validate, there is one share of u per party, and party indices outside
//! 1..=n are errors.
#![cfg(feature = "fixtures")]

use hachi_co_core::Fixture;

const K: usize = 4;

#[test]
fn every_fixture_yields_valid_inputs_for_every_party() {
    for name in Fixture::TOY {
        let fx = Fixture::load(name).unwrap();
        let ctx = fx.context::<K>().unwrap();
        let l = &ctx.layout;
        let n = fx.n();
        let mut owned = Vec::new();
        for p in 1..=n {
            let st = fx.party_state(&ctx, p).unwrap();
            assert_eq!(st.party, p, "{name}");
            let indices: Vec<_> = st.blocks.iter().map(|b| b.index).collect();
            assert_eq!(
                indices,
                l.blocks_of(p).collect::<Vec<_>>(),
                "{name}, party {p}"
            );
            owned.extend(indices);
            assert_eq!(fx.u_share(p).unwrap().0.len(), l.d, "{name}, party {p}");
        }
        owned.sort_unstable();
        let all: Vec<_> = (0..l.n_blocks)
            .filter(|&i| l.owner_of_block(i).is_some())
            .collect();
        assert_eq!(
            owned, all,
            "{name}: the parties' blocks are the owned blocks"
        );
        fx.public_vectors(&ctx).unwrap();
        assert_eq!(fx.commitment().unwrap().len(), l.n_b, "{name}");
        assert!(fx.attempt_count().unwrap() >= 1, "{name}");
        assert!(fx.party_state(&ctx, n + 1).is_err(), "{name}: party n + 1");
        assert!(fx.u_share(n + 1).is_err(), "{name}: share n + 1");
        assert!(fx.u_share(0).is_err(), "{name}: share 0");
    }
}
