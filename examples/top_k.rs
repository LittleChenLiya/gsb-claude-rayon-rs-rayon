//! A small demo of IndexedParallelIterator::top_k.
//!
//! Run with: cargo run --example top_k

use rayon::prelude::*;

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
            score: 42,
        }, // ties with alice, listed after her
        Player {
            name: "dave",
            score: 99,
        },
        Player {
            name: "erin",
            score: 5,
        },
    ];

    // Highest scores first; equal scores keep the original input order.
    let podium = players.par_iter().top_k(3, |a, b| b.score.cmp(&a.score));
    println!("top 3 players:");
    for (rank, player) in podium.iter().enumerate() {
        println!(
            "  #{} {:>6} ({} points)",
            rank + 1,
            player.name,
            player.score
        );
    }

    // Large input, small k: only bounded candidate sets are kept in
    // memory, the ten million items are never collected in full.
    let top = (0..10_000_000usize)
        .into_par_iter()
        .top_k(5, |a, b| b.cmp(a));
    println!("top 5 of 0..10_000_000: {top:?}");
}
