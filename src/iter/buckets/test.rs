use super::BucketError;
use crate::ThreadPoolBuilder;
use crate::prelude::*;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A non-`Clone` item that records every drop of itself.
#[derive(Debug)]
struct DropTracker {
    id: usize,
    drops: Arc<Vec<AtomicUsize>>,
}

impl Drop for DropTracker {
    fn drop(&mut self) {
        self.drops[self.id].fetch_add(1, Ordering::SeqCst);
    }
}

fn make_trackers(n: usize) -> (Vec<DropTracker>, Arc<Vec<AtomicUsize>>) {
    let drops = Arc::new((0..n).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>());
    let items = (0..n)
        .map(|id| DropTracker {
            id,
            drops: drops.clone(),
        })
        .collect();
    (items, drops)
}

fn assert_dropped_exactly_once(drops: &[AtomicUsize]) {
    for (id, slot) in drops.iter().enumerate() {
        assert_eq!(
            slot.load(Ordering::SeqCst),
            1,
            "item {id} was not dropped exactly once"
        );
    }
}

/// Integers tagged with their original index, bucketed by value mod 3,
/// checked against a sequential filter, with 1/2/4 threads. Also verifies
/// that the classifier runs exactly once per item and that the result does
/// not depend on the split granularity.
#[test]
fn mod3_matches_sequential() {
    for threads in [1, 2, 4] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let n = 10_000usize;
            let classify_calls = AtomicUsize::new(0);
            let classify = |&(_, value): &(usize, usize)| {
                classify_calls.fetch_add(1, Ordering::SeqCst);
                value % 3
            };

            let buckets = (0..n)
                .into_par_iter()
                .map(|i| (i, i * 7 + 3))
                .partition_into_buckets(3, classify)
                .unwrap();
            assert_eq!(classify_calls.load(Ordering::SeqCst), n);

            let expected: Vec<Vec<usize>> = (0..3)
                .map(|b| {
                    (0..n)
                        .map(|i| (i, i * 7 + 3))
                        .filter(|&(_, value)| value % 3 == b)
                        .map(|(i, _)| i)
                        .collect()
                })
                .collect();
            let actual: Vec<Vec<usize>> = buckets
                .iter()
                .map(|bucket| bucket.iter().map(|&(i, _)| i).collect())
                .collect();
            assert_eq!(actual, expected, "threads = {threads}");

            // A different split granularity must give the same result.
            let grained: Vec<Vec<usize>> = (0..n)
                .into_par_iter()
                .with_max_len(1)
                .map(|i| (i, i * 7 + 3))
                .partition_into_buckets(3, |&(_, value)| value % 3)
                .unwrap()
                .iter()
                .map(|bucket| bucket.iter().map(|&(i, _)| i).collect())
                .collect();
            assert_eq!(grained, expected, "with_max_len(1), threads = {threads}");
        });
    }
}

#[test]
fn drops_exactly_once_on_success() {
    let n = 1_000;
    let (items, drops) = make_trackers(n);
    let buckets = items
        .into_par_iter()
        .partition_into_buckets(4, |t| t.id % 4)
        .unwrap();
    for (b, bucket) in buckets.iter().enumerate() {
        let actual: Vec<usize> = bucket.iter().map(|t| t.id).collect();
        let expected: Vec<usize> = (0..n).filter(|id| id % 4 == b).collect();
        assert_eq!(actual, expected);
    }
    drop(buckets);
    assert_dropped_exactly_once(&drops);
}

#[test]
fn drops_exactly_once_on_invalid_bucket() {
    let n = 1_000;
    let (items, drops) = make_trackers(n);
    let err = items
        .into_par_iter()
        .partition_into_buckets(4, |t| if t.id == 777 { 4 } else { t.id % 4 })
        .unwrap_err();
    assert_eq!(
        err,
        BucketError::BucketOutOfRange {
            bucket: 4,
            num_buckets: 4,
        }
    );
    assert_dropped_exactly_once(&drops);
}

#[test]
fn drops_exactly_once_on_classify_panic() {
    let n = 1_000;
    let (items, drops) = make_trackers(n);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _ = items.into_par_iter().partition_into_buckets(4, |t| {
            if t.id == 500 {
                panic!("boom");
            }
            t.id % 4
        });
    }));
    assert!(result.is_err());
    assert_dropped_exactly_once(&drops);
}

#[test]
fn zero_buckets_is_an_error() {
    let err = (0..10)
        .into_par_iter()
        .partition_into_buckets(0, |_| 0)
        .unwrap_err();
    assert_eq!(err, BucketError::NoBuckets);
}

#[test]
fn more_buckets_than_items() {
    let buckets = (0..5)
        .into_par_iter()
        .partition_into_buckets(10, |x| x % 10)
        .unwrap();
    assert_eq!(buckets.len(), 10);
    for (i, bucket) in buckets.iter().enumerate() {
        if i < 5 {
            assert_eq!(*bucket, vec![i]);
        } else {
            assert!(bucket.is_empty());
        }
    }
}

#[test]
fn empty_input() {
    let buckets = Vec::<i32>::new()
        .into_par_iter()
        .partition_into_buckets(3, |x| (x % 3) as usize)
        .unwrap();
    assert_eq!(buckets.len(), 3);
    assert!(buckets.iter().all(Vec::is_empty));
}

#[test]
fn all_items_in_one_bucket() {
    let buckets = (0..1_000)
        .into_par_iter()
        .partition_into_buckets(3, |_| 1)
        .unwrap();
    assert!(buckets[0].is_empty());
    assert!(buckets[2].is_empty());
    assert_eq!(buckets[1], (0..1_000).collect::<Vec<_>>());
}
