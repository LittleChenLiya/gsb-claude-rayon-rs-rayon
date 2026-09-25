//! Demonstrates par_map_windows: for every center element we print the
//! window boundaries (as indices into the original slice), the window
//! contents, and the mapped result, under both boundary policies.
//!
//! Run with: cargo run --example par_map_windows

use rayon::prelude::*;
use rayon::slice::{WindowBoundary, WindowCenter};

fn show(boundary: WindowBoundary) {
    let data: Vec<i32> = (1..=8).collect();
    let left = 2;
    let right = 1;

    println!(
        "policy = {:?}, data = {:?}, left = {}, right = {}",
        boundary, data, left, right
    );

    // Outputs come back in stable center-index order.
    let rows: Vec<String> = data
        .par_map_windows(
            left,
            right,
            boundary,
            |window: &[i32], center: WindowCenter| {
                let start = center.index - center.position;
                let end = start + window.len();
                format!(
                    "  center index {} (window position {}): data[{}..{}] = {:?} -> sum = {}",
                    center.index,
                    center.position,
                    start,
                    end,
                    window,
                    window.iter().sum::<i32>()
                )
            },
        )
        .collect();

    for row in &rows {
        println!("{}", row);
    }
    println!("  {} output(s)", rows.len());
    println!();
}

fn main() {
    show(WindowBoundary::Truncate);
    show(WindowBoundary::Skip);
}
