use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

/// Serial reference implementation of an inclusive segmented scan.
fn reference_sum(values: &[i64], starts: &[bool]) -> Vec<i64> {
    let mut out = Vec::with_capacity(values.len());
    let mut acc = 0;
    for (&value, &is_start) in values.iter().zip(starts) {
        if is_start {
            acc = 0;
        }
        acc += value;
        out.push(acc);
    }
    out
}

fn reference_concat(values: &[String], starts: &[bool]) -> Vec<String> {
    let mut out = Vec::with_capacity(values.len());
    let mut acc = String::new();
    for (value, &is_start) in values.iter().zip(starts) {
        if is_start {
            acc.clear();
        }
        acc.push_str(value);
        out.push(acc.clone());
    }
    out
}

/// Deterministic pseudo-random data, so failures are reproducible.
fn generate(size: usize, chunk: usize) -> (Vec<i64>, Vec<bool>) {
    let mut values = Vec::with_capacity(size);
    let mut starts = Vec::with_capacity(size);
    let mut state = 0x1234_5678_9abc_def0u64;
    for i in 0..size {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        values.push(((state >> 33) % 199) as i64 - 99);
        // Mix regular chunk boundaries with random segment starts.
        starts.push(i % chunk == 0 || (state >> 40) % 13 == 0);
    }
    (values, starts)
}

#[test]
fn segmented_scan_basic() {
    let values = vec![1, 2, 3, 4, 5];
    let starts = vec![true, false, true, false, false];

    let scanned: Vec<i32> = values
        .into_par_iter()
        .zip(starts)
        .segmented_scan(0, |a, b| a + b)
        .collect();

    assert_eq!(scanned, [1, 3, 3, 7, 12]);
}

#[test]
fn segmented_scan_first_item_implicitly_starts_segment() {
    // No explicit segment-start marker at all: the whole input is one
    // segment, opened implicitly by the first item.
    let values = vec![10, 20, 30];
    let starts = vec![false, false, false];

    let scanned: Vec<i32> = values
        .into_par_iter()
        .zip(starts)
        .segmented_scan(0, |a, b| a + b)
        .collect();

    assert_eq!(scanned, [10, 30, 60]);
}

#[test]
fn segmented_scan_empty() {
    let scanned: Vec<i32> = Vec::<(i32, bool)>::new()
        .into_par_iter()
        .segmented_scan(0, |a, b| a + b)
        .collect();

    assert_eq!(scanned, Vec::<i32>::new());
}

#[test]
fn segmented_scan_non_commutative() {
    // String concatenation is associative but not commutative, so this
    // proves the left-to-right operand order is preserved.
    let values: Vec<String> = ["a", "b", "c", "d", "e"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let starts = vec![true, false, true, false, false];

    let scanned: Vec<String> = values
        .into_par_iter()
        .zip(starts)
        .segmented_scan(String::new(), |a, b| {
            let mut s = a.clone();
            s.push_str(b);
            s
        })
        .collect();

    assert_eq!(scanned, ["a", "ab", "c", "cd", "cde"]);
}

#[test]
fn segmented_scan_matches_serial_reference() {
    let sizes = [
        0usize, 1, 2, 3, 15, 16, 17, 100, 1023, 1024, 1025, 2048, 4095, 10_000, 100_003,
    ];
    let chunks = [1usize, 2, 3, 7, 64, 1000];

    for threads in [1, 2, 4] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for &size in &sizes {
            for &chunk in &chunks {
                let (values, starts) = generate(size, chunk);
                let expected = reference_sum(&values, &starts);
                let actual: Vec<i64> = pool.install(|| {
                    values
                        .par_iter()
                        .copied()
                        .zip(starts.par_iter().copied())
                        .segmented_scan(0, |a, b| a + b)
                        .collect()
                });
                assert_eq!(
                    actual, expected,
                    "mismatch with threads={threads}, size={size}, chunk={chunk}"
                );
            }
        }
    }
}

#[test]
fn segmented_scan_non_commutative_across_blocks() {
    // Larger inputs with a non-commutative operation, checked against the
    // serial reference across thread counts and segment lengths. This
    // exercises the cross-block prefix propagation.
    for threads in [1, 2, 4] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for &size in &[0usize, 1, 5, 1023, 1024, 1025, 5000] {
            for &chunk in &[1usize, 2, 5, 13] {
                let values: Vec<String> = (0..size).map(|i| format!("{i}:")).collect();
                let starts: Vec<bool> = (0..size).map(|i| i % chunk == 0).collect();
                let expected = reference_concat(&values, &starts);
                let actual: Vec<String> = pool.install(|| {
                    values
                        .par_iter()
                        .cloned()
                        .zip(starts.par_iter().copied())
                        .segmented_scan(String::new(), |a, b| {
                            let mut s = a.clone();
                            s.push_str(b);
                            s
                        })
                        .collect()
                });
                assert_eq!(
                    actual, expected,
                    "mismatch with threads={threads}, size={size}, chunk={chunk}"
                );
            }
        }
    }
}
