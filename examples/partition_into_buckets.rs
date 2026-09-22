//! Stable bucketing demo: partition numbers into buckets by their last
//! digit, then print each bucket's sequence (truncated) and item count.
//!
//! Run with: cargo run --example partition_into_buckets

use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let start = Instant::now();

    // Deterministic pseudo-random data.
    let data: Vec<u64> = (0..1_000_000u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15) % 1_000)
        .collect();

    let buckets = data
        .into_par_iter()
        .partition_into_buckets(10, |&x| (x % 10) as usize)
        .expect("10 is a valid bucket count");

    for (b, bucket) in buckets.iter().enumerate() {
        let preview: Vec<u64> = bucket.iter().take(10).copied().collect();
        println!(
            "bucket {b}: count = {:7}, first items = {preview:?}",
            bucket.len()
        );
    }

    let total: usize = buckets.iter().map(Vec::len).sum();
    println!("total items: {total} (in {:?})", start.elapsed());
}
