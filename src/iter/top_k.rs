//! Parallel stable top-k selection for indexed parallel iterators.
//!
//! Each work segment keeps a bounded candidate set of at most k
//! entries, tagged with the original index so ties resolve
//! deterministically. Merging two segments keeps the k best
//! candidates, so peak memory grows with the number of work segments
//! times k, not with the total input length.

use super::{IndexedParallelIterator, ParallelIterator};
use std::cmp::Ordering;

pub(super) fn top_k_by<PI, F>(pi: PI, k: usize, compare: F) -> Vec<PI::Item>
where
    PI: IndexedParallelIterator,
    F: Fn(&PI::Item, &PI::Item) -> Ordering + Sync + Send,
{
    // A zero-sized result must not drive the iterator at all, so any
    // upstream `map` closures are never executed.
    if k == 0 {
        return Vec::new();
    }

    let bound = k.min(pi.len());
    let compare = &compare;

    pi.enumerate()
        .fold(
            || Vec::with_capacity(bound),
            |mut acc, (index, item)| {
                insert_bounded(&mut acc, k, (index, item), compare);
                acc
            },
        )
        .reduce(Vec::new, |left, right| {
            merge_bounded(left, right, k, compare)
        })
        .into_iter()
        .map(|(_, item)| item)
        .collect()
}

/// Total order on candidates: by the user's comparison first, then by
/// the original index so equal elements keep their input order. Since
/// indices are unique, this never returns `Equal`.
fn entry_cmp<T, F>(a: &(usize, T), b: &(usize, T), compare: &F) -> Ordering
where
    F: Fn(&T, &T) -> Ordering,
{
    compare(&a.1, &b.1).then_with(|| a.0.cmp(&b.0))
}

/// Inserts `entry` into the sorted candidate set `acc`, keeping at
/// most the `k` best entries.
fn insert_bounded<T, F>(acc: &mut Vec<(usize, T)>, k: usize, entry: (usize, T), compare: &F)
where
    F: Fn(&T, &T) -> Ordering,
{
    if acc.len() == k && entry_cmp(&entry, acc.last().unwrap(), compare) != Ordering::Less {
        // The candidate set is full and the new entry is not better
        // than the worst entry already kept.
        return;
    }
    let pos = acc.partition_point(|e| entry_cmp(e, &entry, compare) == Ordering::Less);
    acc.insert(pos, entry);
    acc.truncate(k);
}

/// Merges two sorted candidate sets, keeping at most the `k` best
/// entries. Leftover entries in either input are dropped normally.
fn merge_bounded<T, F>(
    left: Vec<(usize, T)>,
    right: Vec<(usize, T)>,
    k: usize,
    compare: &F,
) -> Vec<(usize, T)>
where
    F: Fn(&T, &T) -> Ordering,
{
    let mut out = Vec::with_capacity(k.min(left.len() + right.len()));
    let mut left = left.into_iter();
    let mut right = right.into_iter();
    let mut l = left.next();
    let mut r = right.next();
    while out.len() < k {
        match (l, r) {
            (Some(a), Some(b)) => {
                if entry_cmp(&a, &b, compare) == Ordering::Less {
                    out.push(a);
                    l = left.next();
                    r = Some(b);
                } else {
                    out.push(b);
                    r = right.next();
                    l = Some(a);
                }
            }
            (Some(a), None) => {
                out.push(a);
                l = left.next();
                r = None;
            }
            (None, Some(b)) => {
                out.push(b);
                r = right.next();
                l = None;
            }
            (None, None) => break,
        }
    }
    out
}
