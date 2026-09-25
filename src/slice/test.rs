#![cfg(test)]

use crate::ThreadPoolBuilder;
use crate::prelude::*;
use crate::slice::WindowBoundary;
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

#[test]
fn map_windows_truncate() {
    let sums: Vec<i32> = [1, 2, 3, 4]
        .par_map_windows(1, 1, WindowBoundary::Truncate, |window, _pos, _index| {
            window.iter().sum()
        })
        .collect();
    assert_eq!(sums, [3, 6, 9, 7]);
}

#[test]
fn map_windows_skip() {
    let sums: Vec<i32> = [1, 2, 3, 4]
        .par_map_windows(1, 1, WindowBoundary::Skip, |window, pos, _index| {
            assert_eq!(pos, 1);
            assert_eq!(window.len(), 3);
            window.iter().sum()
        })
        .collect();
    assert_eq!(sums, [6, 9]);
}

/// Checks every window against a sequential reference, including windows
/// that straddle chunk boundaries, and verifies that each center produces
/// exactly one output with its absolute slice index.
fn check_map_windows(threads: usize) {
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let v: Vec<i32> = (0..10_000).collect();
        for &(left, right) in &[(0usize, 0usize), (1, 1), (2, 0), (0, 3), (3, 5), (7, 2)] {
            let calls = AtomicUsize::new(0);
            let out: Vec<(usize, i64)> = v
                .par_map_windows(left, right, WindowBoundary::Truncate, |w, pos, index| {
                    calls.fetch_add(1, Relaxed);
                    assert_eq!(w[pos], v[index]);
                    (index, w.iter().map(|&x| x as i64).sum())
                })
                .collect();
            assert_eq!(calls.load(Relaxed), v.len());
            assert_eq!(out.len(), v.len());
            for (i, &(index, sum)) in out.iter().enumerate() {
                assert_eq!(index, i);
                let start = i.saturating_sub(left);
                let end = (i + right + 1).min(v.len());
                let expect: i64 = v[start..end].iter().map(|&x| x as i64).sum();
                assert_eq!(sum, expect, "truncate window at center {i}");
            }

            let calls = AtomicUsize::new(0);
            let out: Vec<(usize, i64)> = v
                .par_map_windows(left, right, WindowBoundary::Skip, |w, pos, index| {
                    calls.fetch_add(1, Relaxed);
                    assert_eq!(pos, left);
                    assert_eq!(w.len(), left + 1 + right);
                    assert_eq!(w[pos], v[index]);
                    (index, w.iter().map(|&x| x as i64).sum())
                })
                .collect();
            let expected_len = v.len() - left - right;
            assert_eq!(calls.load(Relaxed), expected_len);
            assert_eq!(out.len(), expected_len);
            for (k, &(index, sum)) in out.iter().enumerate() {
                let i = k + left;
                // The center index is the absolute slice index: skipping the
                // edges must not renumber the remaining centers from 0.
                assert_eq!(index, i);
                let expect: i64 = v[i - left..=i + right].iter().map(|&x| x as i64).sum();
                assert_eq!(sum, expect, "skip window at center {i}");
            }
        }
    });
}

#[test]
fn map_windows_cross_chunk_1_thread() {
    check_map_windows(1);
}

#[test]
fn map_windows_cross_chunk_2_threads() {
    check_map_windows(2);
}

#[test]
fn map_windows_cross_chunk_4_threads() {
    check_map_windows(4);
}

#[test]
fn map_windows_empty() {
    let v: Vec<i32> = Vec::new();
    for boundary in [WindowBoundary::Truncate, WindowBoundary::Skip] {
        let calls = AtomicUsize::new(0);
        let out: Vec<i32> = v
            .par_map_windows(1, 1, boundary, |w, _pos, _index| {
                calls.fetch_add(1, Relaxed);
                w.iter().sum()
            })
            .collect();
        assert!(out.is_empty());
        assert_eq!(calls.load(Relaxed), 0);
    }
}

#[test]
fn map_windows_oversized_neighborhood() {
    let v: Vec<i32> = (0..10).collect();

    // Truncation clamps every window to the whole slice.
    let out: Vec<usize> = v
        .par_map_windows(1000, 1000, WindowBoundary::Truncate, |w, pos, index| {
            assert_eq!(w, &v[..]);
            assert_eq!(pos, index);
            w.len()
        })
        .collect();
    assert_eq!(out, vec![10; 10]);

    // Skipping drops every center when no complete window fits.
    let calls = AtomicUsize::new(0);
    let out: Vec<i32> = v
        .par_map_windows(1000, 1000, WindowBoundary::Skip, |w, _pos, _index| {
            calls.fetch_add(1, Relaxed);
            w.iter().sum()
        })
        .collect();
    assert!(out.is_empty());
    assert_eq!(calls.load(Relaxed), 0);

    // A zero-width neighborhood yields single-element windows.
    let out: Vec<i32> = v
        .par_map_windows(0, 0, WindowBoundary::Truncate, |w, pos, _index| {
            assert_eq!(pos, 0);
            assert_eq!(w.len(), 1);
            w[0]
        })
        .collect();
    assert_eq!(out, v);
}

#[test]
fn map_windows_non_clone() {
    struct NoClone(i32);

    let v: Vec<NoClone> = (0..1000).map(NoClone).collect();
    let out: Vec<i32> = v
        .par_map_windows(2, 3, WindowBoundary::Truncate, |w, pos, index| {
            assert_eq!(w[pos].0, v[index].0);
            w.iter().map(|x| x.0).sum()
        })
        .collect();
    assert_eq!(out.len(), v.len());
    for (i, sum) in out.iter().enumerate() {
        let start = i.saturating_sub(2);
        let end = (i + 4).min(v.len());
        let expect: i32 = (start as i32..end as i32).sum();
        assert_eq!(*sum, expect);
    }
}

#[test]
#[should_panic(expected = "overflows")]
fn map_windows_overflow_rejected() {
    let v = [1, 2, 3];
    let _ = v.par_map_windows(usize::MAX, 1, WindowBoundary::Truncate, |_, _, _| ());
}

#[test]
#[should_panic(expected = "map_windows boom")]
fn map_windows_panic_propagates() {
    let v: Vec<i32> = (0..10_000).collect();
    let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    pool.install(|| {
        let _: Vec<i32> = v
            .par_map_windows(1, 1, WindowBoundary::Truncate, |w, _pos, index| {
                if index == 5_000 {
                    panic!("map_windows boom");
                }
                w[0]
            })
            .collect();
    });
}
