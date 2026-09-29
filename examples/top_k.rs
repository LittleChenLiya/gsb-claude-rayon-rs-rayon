//! Small demo of the stable parallel top-k selection.
//!
//! Run with: cargo run --example top_k

use rayon::prelude::*;

#[derive(Debug)]
struct Player {
    name: &'static str,
    score: u32,
}

fn main() {
    let players = vec![
        Player {
            name: "alice",
            score: 42,
        },
        Player {
            name: "bob",
            score: 17,
        },
        Player {
            name: "carol",
            score: 99,
        },
        Player {
            name: "dave",
            score: 42,
        }, // tie with alice
        Player {
            name: "erin",
            score: 73,
        },
        Player {
            name: "frank",
            score: 8,
        },
    ];

    // Top 3 by score, highest first. Ties keep the input order, so
    // alice ranks ahead of dave, exactly like a stable sort would.
    let podium = players.par_iter().top_k_by(3, |a, b| b.score.cmp(&a.score));

    println!("top 3 players by score:");
    for (rank, player) in podium.iter().enumerate() {
        println!("  #{} {:>5} ({})", rank + 1, player.name, player.score);
    }

    // The k smallest numbers of a large range, without collecting
    // and sorting the whole thing first.
    let top: Vec<u64> = (0..1_000_000usize)
        .into_par_iter()
        .map(|x| {
            (x as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .reverse_bits()
        })
        .top_k(5);
    println!("5 smallest hashed values: {top:?}");
}
