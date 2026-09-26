use super::{IndexedParallelIterator, ParallelIterator};

use std::cmp::Ordering;

/// Implementation of top_k for indexed parallel iterators.
///
/// Each fold segment keeps a bounded candidate set of at most k entries,
/// sorted by rank, and the reducer merges two sorted candidate sets while
/// keeping only the k best. Because ties are broken by the original
/// index, the rank order is total and the final result does not depend on
/// how the input was split across threads.
pub(super) fn top_k<PI, F>(pi: PI, k: usize, compare: F) -> Vec<PI::Item>
where
    PI: IndexedParallelIterator,
    F: Fn(&PI::Item, &PI::Item) -> Ordering + Sync,
{
    // With k == 0 every candidate set would stay empty, so there is no
    // point in driving the iterator (and its upstream maps) at all.
    if k == 0 {
        return Vec::new();
    }

    let compare = &compare;
    pi.enumerate()
        .fold(Vec::new, move |mut best, (index, item)| {
            insert_bounded(&mut best, k, index, item, compare);
            best
        })
        .reduce(Vec::new, move |left, right| {
            merge_bounded(left, right, k, compare)
        })
        .into_iter()
        .map(|(_, item)| item)
        .collect()
}

/// The rank order: compare first, then the original index so that equal
/// elements keep their input order (stability).
fn rank<T, F>(compare: &F, a: &(usize, T), b: &(usize, T)) -> Ordering
where
    F: Fn(&T, &T) -> Ordering,
{
    compare(&a.1, &b.1).then(a.0.cmp(&b.0))
}

/// Inserts (index, item) into the sorted candidate set, keeping at most
/// the k best entries. Entries that do not rank among the current top k
/// are dropped immediately.
fn insert_bounded<T, F>(best: &mut Vec<(usize, T)>, k: usize, index: usize, item: T, compare: &F)
where
    F: Fn(&T, &T) -> Ordering,
{
    let pos =
        best.partition_point(|(i, t)| compare(t, &item).then(i.cmp(&index)) == Ordering::Less);
    if pos == k {
        // The set is full and item ranks after everything in it.
        return;
    }
    best.insert(pos, (index, item));
    best.truncate(k);
}

/// Merges two sorted candidate sets, keeping at most the k best entries.
/// Both inputs and the output are sorted by rank, so the merge is a plain
/// linear pass and its result is independent of the split/reduce order.
fn merge_bounded<T, F>(
    left: Vec<(usize, T)>,
    right: Vec<(usize, T)>,
    k: usize,
    compare: &F,
) -> Vec<(usize, T)>
where
    F: Fn(&T, &T) -> Ordering,
{
    let mut merged = Vec::with_capacity(k.min(left.len() + right.len()));
    let mut left = left.into_iter();
    let mut right = right.into_iter();
    while merged.len() < k {
        let take_left = match (left.as_slice().first(), right.as_slice().first()) {
            (Some(l), Some(r)) => rank(compare, l, r) != Ordering::Greater,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        let next = if take_left { left.next() } else { right.next() };
        merged.push(next.expect("candidate set unexpectedly exhausted"));
    }
    merged
}
