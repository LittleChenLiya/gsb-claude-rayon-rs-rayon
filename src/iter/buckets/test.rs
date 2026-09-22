use super::*;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn run_with_threads<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    crate::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(f)
}

#[test]
fn zero_buckets_is_an_error() {
    let result = (0..10i32).into_par_iter().partition_into_buckets(0, |_| 0);
    assert_eq!(result, Err(PartitionBucketsError::ZeroBuckets));
}

#[test]
fn mod3_matches_sequential_for_various_thread_counts() {
    for threads in 1..=4usize {
        run_with_threads(threads, || {
            let n = 10_000i32;
            let buckets = (0..n)
                .into_par_iter()
                .enumerate()
                .partition_into_buckets(3, |&(_, x)| (x.rem_euclid(3)) as usize)
                .unwrap();

            assert_eq!(buckets.len(), 3);

            let mut expected: Vec<Vec<(usize, i32)>> = vec![Vec::new(); 3];
            for item in (0..n).enumerate() {
                expected[(item.1.rem_euclid(3)) as usize].push(item);
            }
            assert_eq!(buckets, expected);

            // Every bucket must preserve the original index order.
            for bucket in &buckets {
                assert!(bucket.windows(2).all(|w| w[0].0 < w[1].0));
            }
            assert_eq!(buckets.iter().map(|b| b.len()).sum::<usize>(), n as usize);
        });
    }
}

#[test]
fn split_granularity_does_not_change_the_result() {
    let reference = (0..1_000i32)
        .into_par_iter()
        .with_min_len(1)
        .partition_into_buckets(4, |x| (x % 4) as usize)
        .unwrap();

    for min_len in [1, 2, 7, 13, 64, 1_000] {
        let buckets = (0..1_000)
            .into_par_iter()
            .with_min_len(min_len)
            .partition_into_buckets(4, |x| (x % 4) as usize)
            .unwrap();
        assert_eq!(buckets, reference);
    }

    for max_len in [1, 3, 16, 500] {
        let buckets = (0..1_000)
            .into_par_iter()
            .with_max_len(max_len)
            .partition_into_buckets(4, |x| (x % 4) as usize)
            .unwrap();
        assert_eq!(buckets, reference);
    }
}

#[test]
fn more_buckets_than_items() {
    for threads in [1, 2, 4] {
        run_with_threads(threads, || {
            let buckets = (0..3)
                .into_par_iter()
                .partition_into_buckets(10, |&x| x as usize)
                .unwrap();
            assert_eq!(buckets.len(), 10);
            assert_eq!(buckets[0], [0]);
            assert_eq!(buckets[1], [1]);
            assert_eq!(buckets[2], [2]);
            for bucket in &buckets[3..] {
                assert!(bucket.is_empty());
            }
        });
    }
}

#[test]
fn empty_input() {
    let buckets: Vec<Vec<i32>> = Vec::<i32>::new()
        .into_par_iter()
        .partition_into_buckets(3, |_| 0)
        .unwrap();
    assert_eq!(buckets, vec![Vec::<i32>::new(), Vec::new(), Vec::new()]);
}

#[test]
fn all_items_in_one_bucket_keep_their_order() {
    for threads in [1, 2, 4] {
        run_with_threads(threads, || {
            let buckets = (0..5_000)
                .into_par_iter()
                .partition_into_buckets(3, |_| 0)
                .unwrap();
            assert_eq!(buckets[0], (0..5_000).collect::<Vec<_>>());
            assert!(buckets[1].is_empty());
            assert!(buckets[2].is_empty());
        });
    }
}

#[test]
fn classification_runs_exactly_once_per_item() {
    for threads in [1, 2, 4] {
        run_with_threads(threads, || {
            let n = 5_000;
            let calls = AtomicUsize::new(0);
            let buckets = (0..n)
                .into_par_iter()
                .partition_into_buckets(5, |&x| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    x % 5
                })
                .unwrap();
            assert_eq!(calls.load(Ordering::Relaxed), n);
            assert_eq!(buckets.iter().map(|b| b.len()).sum::<usize>(), n);
        });
    }
}

#[derive(Debug)]
struct Tracked<'a> {
    drops: &'a AtomicUsize,
    id: usize,
}

impl Drop for Tracked<'_> {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn non_clone_items_are_dropped_once_on_success() {
    for threads in [1, 2, 4] {
        run_with_threads(threads, || {
            let n = 3_000;
            let drops = AtomicUsize::new(0);
            let items: Vec<Tracked<'_>> = (0..n).map(|id| Tracked { drops: &drops, id }).collect();

            let buckets = items
                .into_par_iter()
                .partition_into_buckets(3, |item| item.id % 3)
                .unwrap();

            // Everything was moved into the buckets, nothing dropped yet.
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            assert_eq!(buckets.iter().map(|b| b.len()).sum::<usize>(), n,);

            // Within-bucket order is the original order.
            for bucket in &buckets {
                assert!(bucket.windows(2).all(|w| w[0].id < w[1].id));
            }

            drop(buckets);
            assert_eq!(drops.load(Ordering::SeqCst), n);
        });
    }
}

#[test]
fn non_clone_items_are_dropped_once_on_invalid_bucket() {
    for threads in [1, 2, 4] {
        for bad in [0, 1, 1_500, 2_999] {
            run_with_threads(threads, || {
                let n = 3_000;
                let drops = AtomicUsize::new(0);
                let items: Vec<Tracked<'_>> =
                    (0..n).map(|id| Tracked { drops: &drops, id }).collect();

                let result = items
                    .into_par_iter()
                    .partition_into_buckets(3, |item| if item.id == bad { 3 } else { item.id % 3 });

                match result {
                    Err(PartitionBucketsError::InvalidBucket {
                        bucket: 3,
                        num_buckets: 3,
                    }) => {}
                    other => panic!("unexpected result: {other:?}"),
                }
                // No partial result: every item dropped exactly once.
                assert_eq!(drops.load(Ordering::SeqCst), n);
            });
        }
    }
}

#[test]
fn non_clone_items_are_dropped_once_on_panic() {
    for threads in [1, 2, 4] {
        for boom in [0, 1, 1_500, 2_999] {
            run_with_threads(threads, || {
                let n = 3_000;
                let drops = Arc::new(AtomicUsize::new(0));
                let items: Vec<Tracked<'_>> =
                    (0..n).map(|id| Tracked { drops: &drops, id }).collect();

                let panicked = catch_unwind(AssertUnwindSafe(|| {
                    items.into_par_iter().partition_into_buckets(3, |item| {
                        if item.id == boom {
                            panic!("boom at {}", item.id);
                        }
                        item.id % 3
                    })
                }));
                assert!(panicked.is_err());
                // Every item pulled from the producer was dropped exactly once.
                assert_eq!(drops.load(Ordering::SeqCst), n);
            });
        }
    }
}

#[test]
fn invalid_bucket_in_the_middle_is_reported() {
    let result = (0..100)
        .into_par_iter()
        .partition_into_buckets(4, |&x| if x == 42 { 4 } else { x % 4 });
    assert_eq!(
        result,
        Err(PartitionBucketsError::InvalidBucket {
            bucket: 4,
            num_buckets: 4,
        })
    );
}

#[test]
fn doc_example_shape() {
    let buckets = (0..10)
        .into_par_iter()
        .partition_into_buckets(3, |x| x % 3)
        .unwrap();
    assert_eq!(
        buckets,
        vec![vec![0, 3, 6, 9], vec![1, 4, 7], vec![2, 5, 8]]
    );
}
