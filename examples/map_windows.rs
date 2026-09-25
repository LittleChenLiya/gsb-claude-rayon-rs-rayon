//! Demonstrates `ParallelSlice::par_map_windows`: each element of the
//! slice is treated as the center of a neighborhood window, and a closure
//! maps that window to exactly one output.
//!
//! Run with: `cargo run --example map_windows`

use rayon::prelude::*;
use rayon::slice::WindowBoundary;
use std::time::Instant;

fn main() {
    let start = Instant::now();

    let data: Vec<i32> = (1..=12).collect();
    println!("input: {data:?}");

    // Truncated boundaries: every element is a center, edge windows shrink.
    println!("\n-- WindowBoundary::Truncate (left = 2, right = 1) --");
    let truncated: Vec<String> = data
        .par_map_windows(2, 1, WindowBoundary::Truncate, |window, pos, index| {
            format!(
                "center {index:2} (value {:2}, window pos {pos}): {window:?} -> sum {}",
                window[pos],
                window.iter().sum::<i32>(),
            )
        })
        .collect();
    for line in &truncated {
        println!("{line}");
    }

    // Skipped boundaries: only complete windows produce output, and the
    // center index keeps referring to the original slice.
    println!("\n-- WindowBoundary::Skip (left = 2, right = 1) --");
    let skipped: Vec<String> = data
        .par_map_windows(2, 1, WindowBoundary::Skip, |window, pos, index| {
            format!(
                "center {index:2} (value {:2}, window pos {pos}): {window:?} -> sum {}",
                window[pos],
                window.iter().sum::<i32>(),
            )
        })
        .collect();
    for line in &skipped {
        println!("{line}");
    }

    let elapsed = start.elapsed();
    println!(
        "\n{} outputs computed in {elapsed:?}",
        truncated.len() + skipped.len()
    );
    assert!(elapsed.as_secs() < 90, "example took too long");
}
