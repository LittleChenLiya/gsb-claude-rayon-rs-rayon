#![cfg(test)]

use crate::prelude::*;
use rand::distr::Uniform;
use rand::seq::IndexedRandom;
use rand::{RngExt, rng};
use std::cmp::Ordering::{Equal, Greater, Less};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

macro_rules! sort {
    ($f:ident, $name:ident) => {
        #[test]
        fn $name() {
            let rng = &mut rng();

            for len in (0..25).chain(500..501) {
                for &modulus in &[5, 10, 100] {
                    let dist = Uniform::new(0, modulus).unwrap();
                    for _ in 0..100 {
                        let v: Vec<i32> = rng.sample_iter(&dist).take(len).collect();

                        // Test sort using `<` operator.
                        let mut tmp = v.clone();
                        tmp.$f(|a, b| a.cmp(b));
                        assert!(tmp.is_sorted());

                        // Test sort using `>` operator.
                        let mut tmp = v.clone();
                        tmp.$f(|a, b| b.cmp(a));
                        assert!(tmp.is_sorted_by(|a, b| a >= b));
                    }
                }
            }

            // Test sort with many duplicates.
            for &len in &[1_000, 10_000, 100_000] {
                for &modulus in &[5, 10, 100, 10_000] {
                    let dist = Uniform::new(0, modulus).unwrap();
                    let mut v: Vec<i32> = rng.sample_iter(&dist).take(len).collect();

                    v.$f(|a, b| a.cmp(b));
                    assert!(v.is_sorted());
                }
            }

            // Test sort with many pre-sorted runs.
            for &len in &[1_000, 10_000, 100_000] {
                let len_dist = Uniform::new(0, len).unwrap();
                for &modulus in &[5, 10, 1000, 50_000] {
                    let dist = Uniform::new(0, modulus).unwrap();
                    let mut v: Vec<i32> = rng.sample_iter(&dist).take(len).collect();

                    v.sort();
                    v.reverse();

                    for _ in 0..5 {
                        let a = rng.sample(&len_dist);
                        let b = rng.sample(&len_dist);
                        if a < b {
                            v[a..b].reverse();
                        } else {
                            v.swap(a, b);
                        }
                    }

                    v.$f(|a, b| a.cmp(b));
                    assert!(v.is_sorted());
                }
            }

            // Sort using a completely random comparison function.
            // This will reorder the elements *somehow*, but won't panic.
            let mut v: Vec<_> = (0..100).collect();
            v.$f(|_, _| *[Less, Equal, Greater].choose(&mut rand::rng()).unwrap());
            v.$f(|a, b| a.cmp(b));
            for i in 0..v.len() {
                assert_eq!(v[i], i);
            }

            // Should not panic.
            [0i32; 0].$f(|a, b| a.cmp(b));
            [(); 10].$f(|a, b| a.cmp(b));
            [(); 100].$f(|a, b| a.cmp(b));

            let mut v = [0xDEAD_BEEFu64];
            v.$f(|a, b| a.cmp(b));
            assert!(v == [0xDEAD_BEEF]);
        }
    };
}

sort!(par_sort_by, test_par_sort);
sort!(par_sort_unstable_by, test_par_sort_unstable);

#[test]
fn test_par_sort_stability() {
    for len in (2..25).chain(500..510).chain(50_000..50_010) {
        for _ in 0..10 {
            let mut counts = [0; 10];

            // Create a vector like [(6, 1), (5, 1), (6, 2), ...],
            // where the first item of each tuple is random, but
            // the second item represents which occurrence of that
            // number this element is, i.e. the second elements
            // will occur in sorted order.
            let mut rng = rng();
            let mut v: Vec<_> = (0..len)
                .map(|_| {
                    let n: usize = rng.random_range(0..10);
                    counts[n] += 1;
                    (n, counts[n])
                })
                .collect();

            // Only sort on the first element, so an unstable sort
            // may mix up the counts.
            v.par_sort_by(|&(a, _), &(b, _)| a.cmp(&b));

            // This comparison includes the count (the second item
            // of the tuple), so elements with equal first items
            // will need to be ordered with increasing
            // counts... i.e. exactly asserting that this sort is
            // stable.
            assert!(v.is_sorted());
        }
    }
}

#[test]
fn test_par_chunks_exact_remainder() {
    let v: &[i32] = &[0, 1, 2, 3, 4];
    let c = v.par_chunks_exact(2);
    assert_eq!(c.remainder(), &[4]);
    assert_eq!(c.len(), 2);
}

#[test]
fn test_par_chunks_exact_mut_remainder() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let mut c = v.par_chunks_exact_mut(2);
    assert_eq!(c.remainder(), &[4]);
    assert_eq!(c.len(), 2);
    assert_eq!(c.into_remainder(), &[4]);

    let mut c = v.par_chunks_exact_mut(2);
    assert_eq!(c.take_remainder(), &[4]);
    assert_eq!(c.take_remainder(), &[]);
    assert_eq!(c.len(), 2);
}

#[test]
fn test_par_rchunks_exact_remainder() {
    let v: &[i32] = &[0, 1, 2, 3, 4];
    let c = v.par_rchunks_exact(2);
    assert_eq!(c.remainder(), &[0]);
    assert_eq!(c.len(), 2);
}

#[test]
fn test_par_rchunks_exact_mut_remainder() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let mut c = v.par_rchunks_exact_mut(2);
    assert_eq!(c.remainder(), &[0]);
    assert_eq!(c.len(), 2);
    assert_eq!(c.into_remainder(), &[0]);

    let mut c = v.par_rchunks_exact_mut(2);
    assert_eq!(c.take_remainder(), &[0]);
    assert_eq!(c.take_remainder(), &[]);
    assert_eq!(c.len(), 2);
}

#[test]
fn slice_chunk_by() {
    let v: Vec<_> = (0..1000).collect();
    assert_eq!(v[..0].par_chunk_by(|_, _| todo!()).count(), 0);
    assert_eq!(v[..1].par_chunk_by(|_, _| todo!()).count(), 1);
    assert_eq!(v[..2].par_chunk_by(|_, _| true).count(), 1);
    assert_eq!(v[..2].par_chunk_by(|_, _| false).count(), 2);

    let count = AtomicUsize::new(0);
    let par: Vec<_> = v
        .par_chunk_by(|x, y| {
            count.fetch_add(1, Relaxed);
            (x % 10 < 3) == (y % 10 < 3)
        })
        .collect();
    assert_eq!(count.into_inner(), v.len() - 1);

    let seq: Vec<_> = v.chunk_by(|x, y| (x % 10 < 3) == (y % 10 < 3)).collect();
    assert_eq!(par, seq);
}

#[test]
fn slice_chunk_by_mut() {
    let mut v: Vec<_> = (0..1000).collect();
    assert_eq!(v[..0].par_chunk_by_mut(|_, _| todo!()).count(), 0);
    assert_eq!(v[..1].par_chunk_by_mut(|_, _| todo!()).count(), 1);
    assert_eq!(v[..2].par_chunk_by_mut(|_, _| true).count(), 1);
    assert_eq!(v[..2].par_chunk_by_mut(|_, _| false).count(), 2);

    let mut v2 = v.clone();
    let count = AtomicUsize::new(0);
    let par: Vec<_> = v
        .par_chunk_by_mut(|x, y| {
            count.fetch_add(1, Relaxed);
            (x % 10 < 3) == (y % 10 < 3)
        })
        .collect();
    assert_eq!(count.into_inner(), v2.len() - 1);

    let seq: Vec<_> = v2
        .chunk_by_mut(|x, y| (x % 10 < 3) == (y % 10 < 3))
        .collect();
    assert_eq!(par, seq);
}

use crate::slice::WindowBoundary;

/// Sequential reference for par_map_windows, returning for each produced
/// center its original index, its position in the window, the window
/// contents, and the window sum.
fn map_windows_reference(
    data: &[i32],
    left: usize,
    right: usize,
    boundary: WindowBoundary,
) -> Vec<(usize, usize, Vec<i32>, i32)> {
    let n = data.len();
    let centers: Vec<usize> = match boundary {
        WindowBoundary::Truncate => (0..n).collect(),
        WindowBoundary::Skip => (left..n.saturating_sub(right)).collect(),
    };
    centers
        .into_iter()
        .map(|c| {
            let start = c.saturating_sub(left);
            let end = Ord::min(n, c + right + 1);
            let window: Vec<i32> = data[start..end].to_vec();
            let sum = window.iter().sum();
            (c, c - start, window, sum)
        })
        .collect()
}

#[test]
fn par_map_windows_truncate() {
    let data = [1, 2, 3, 4];
    let sums: Vec<i32> = data
        .par_map_windows(1, 1, WindowBoundary::Truncate, |w, _| w.iter().sum())
        .collect();
    assert_eq!(sums, [3, 6, 9, 7]);

    let details: Vec<(Vec<i32>, usize, usize)> = data
        .par_map_windows(1, 1, WindowBoundary::Truncate, |w, c| {
            (w.to_vec(), c.index, c.position)
        })
        .collect();
    assert_eq!(
        details,
        [
            (vec![1, 2], 0, 0),
            (vec![1, 2, 3], 1, 1),
            (vec![2, 3, 4], 2, 1),
            (vec![3, 4], 3, 1),
        ]
    );
}

#[test]
fn par_map_windows_skip() {
    let data = [1, 2, 3, 4];
    let sums: Vec<i32> = data
        .par_map_windows(1, 1, WindowBoundary::Skip, |w, _| w.iter().sum())
        .collect();
    assert_eq!(sums, [6, 9]);

    // Center indices come from the original slice and are not renumbered.
    let indices: Vec<usize> = data
        .par_map_windows(1, 1, WindowBoundary::Skip, |_, c| c.index)
        .collect();
    assert_eq!(indices, [1, 2]);

    let details: Vec<(Vec<i32>, usize)> = data
        .par_map_windows(1, 1, WindowBoundary::Skip, |w, c| (w.to_vec(), c.position))
        .collect();
    assert_eq!(details, [(vec![1, 2, 3], 1), (vec![2, 3, 4], 1)]);
}

#[test]
fn par_map_windows_empty() {
    let data: [i32; 0] = [];
    for boundary in [WindowBoundary::Truncate, WindowBoundary::Skip] {
        let calls = AtomicUsize::new(0);
        let out: Vec<i32> = data
            .par_map_windows(1, 1, boundary, |w: &[i32], _| {
                calls.fetch_add(1, Relaxed);
                w.iter().sum()
            })
            .collect();
        assert!(out.is_empty());
        assert_eq!(calls.load(Relaxed), 0);
    }
}

#[test]
fn par_map_windows_matches_sequential_across_threads() {
    let data: Vec<i32> = (0..1000).map(|i| i * 7 % 13 - 6).collect();
    for threads in [1, 2, 4] {
        let pool = crate::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for boundary in [WindowBoundary::Truncate, WindowBoundary::Skip] {
            for &(left, right) in &[(0usize, 0usize), (1, 1), (3, 2), (2000, 2000)] {
                let expected = map_windows_reference(&data, left, right, boundary);
                let calls = AtomicUsize::new(0);
                let seen = std::sync::Mutex::new(Vec::new());
                let actual: Vec<(usize, usize, Vec<i32>, i32)> = pool.install(|| {
                    data.par_map_windows(left, right, boundary, |window, center| {
                        calls.fetch_add(1, Relaxed);
                        seen.lock().unwrap().push(center.index);
                        assert_eq!(window[center.position], data[center.index]);
                        (
                            center.index,
                            center.position,
                            window.to_vec(),
                            window.iter().sum(),
                        )
                    })
                    // Force fine-grained splits to exercise halo exchange
                    // across chunk boundaries.
                    .with_max_len(1)
                    .collect()
                });
                // Exactly one call per center, in stable center-index order.
                assert_eq!(calls.load(Relaxed), expected.len());
                assert_eq!(actual, expected);
                let mut seen = seen.into_inner().unwrap();
                seen.sort_unstable();
                seen.dedup();
                assert_eq!(seen.len(), expected.len());
            }
        }
    }
}

#[test]
fn par_map_windows_non_clone_elements() {
    struct NoClone(i32);

    let data: Vec<NoClone> = (0..10).map(NoClone).collect();
    let sums: Vec<i32> = data
        .par_map_windows(2, 2, WindowBoundary::Truncate, |w, _| {
            w.iter().map(|x| x.0).sum()
        })
        .collect();
    let expected: Vec<i32> = (0..10i32)
        .map(|c| {
            let start = Ord::max(c - 2, 0);
            let end = Ord::min(c + 3, 10);
            (start..end).sum()
        })
        .collect();
    assert_eq!(sums, expected);
}

#[test]
#[should_panic(expected = "left + right neighborhood width overflowed")]
fn par_map_windows_width_overflow() {
    let data = [1, 2, 3];
    let _ = data.par_map_windows(usize::MAX, 1, WindowBoundary::Truncate, |w, _| w.len());
}

#[test]
#[should_panic(expected = "boom")]
fn par_map_windows_panic_propagates() {
    let data: Vec<i32> = (0..1000).collect();
    data.par_map_windows(1, 1, WindowBoundary::Truncate, |_, c| {
        if c.index == 500 {
            panic!("boom")
        }
        c.index
    })
    .for_each(|_| {});
}
