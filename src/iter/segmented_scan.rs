use super::*;
use crate::slice::ParallelSliceMut;

use std::fmt::{self, Debug};

/// `SegmentedScan` is an iterator that computes an inclusive prefix scan
/// which resets wherever a parallel "segment start" flag iterator yields
/// `true`. The very first element implicitly starts a new segment.
///
/// This struct is created by the [`segmented_scan()`] method on
/// [`IndexedParallelIterator`].
///
/// [`segmented_scan()`]: IndexedParallelIterator::segmented_scan()
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Clone)]
pub struct SegmentedScan<I, F, T, OP> {
    base: I,
    flags: F,
    identity: T,
    scan_op: OP,
}

impl<I: Debug, F: Debug, T: Debug, OP> Debug for SegmentedScan<I, F, T, OP> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SegmentedScan")
            .field("base", &self.base)
            .field("flags", &self.flags)
            .field("identity", &self.identity)
            .finish()
    }
}

impl<I, F, T, OP> SegmentedScan<I, F, T, OP> {
    /// Creates a new `SegmentedScan` iterator.
    pub(super) fn new(base: I, flags: F, identity: T, scan_op: OP) -> Self {
        SegmentedScan {
            base,
            flags,
            identity,
            scan_op,
        }
    }
}

/// A summary of one chunk of input, used to carry scan state across chunks.
///
/// The summaries form a monoid under `combine` (with the associative
/// `scan_op` provided by the caller), which is what allows the per-chunk
/// scans to be stitched together in the correct left-to-right order.
#[derive(Debug, Clone)]
struct Summary<T> {
    /// Whether the chunk contains any segment-start flag.
    has_flag: bool,
    /// The combined value of the leading partial segment: everything before
    /// the first flag of the chunk (or the whole chunk if it has no flag).
    head: T,
    /// The combined value of the trailing partial segment: everything from
    /// the last flag of the chunk onward (or the whole chunk if no flag).
    tail: T,
}

impl<T> Summary<T> {
    /// Combines two adjacent chunk summaries, preserving operand order.
    fn combine<OP>(self, right: &Summary<T>, scan_op: &OP) -> Summary<T>
    where
        T: Clone,
        OP: Fn(T, T) -> T,
    {
        Summary {
            has_flag: self.has_flag || right.has_flag,
            // The leading partial segment of the left chunk extends into the
            // right chunk only if the left chunk never started a new segment.
            head: if self.has_flag {
                self.head
            } else {
                scan_op(self.head, right.head.clone())
            },
            // The trailing partial segment runs into the right chunk only if
            // the right chunk never started a new segment of its own.
            tail: if right.has_flag {
                right.tail.clone()
            } else {
                scan_op(self.tail, right.tail.clone())
            },
        }
    }

    /// The value a following chunk should use as its incoming segment total.
    fn carry(&self) -> T
    where
        T: Clone,
    {
        if self.has_flag {
            self.tail.clone()
        } else {
            self.head.clone()
        }
    }
}

/// Performs a sequential inclusive segmented scan of one chunk in place and
/// returns the chunk [`Summary`]. Segment starts (including an implicit one
/// at the very first element) restart the accumulation from `identity`.
fn local_scan<T, OP>(chunk: &mut [(T, bool)], identity: &T, scan_op: &OP) -> Summary<T>
where
    T: Clone,
    OP: Fn(T, T) -> T,
{
    debug_assert!(!chunk.is_empty());
    let mut has_flag = false;
    let mut head = identity.clone();
    let mut acc = identity.clone();
    for (value, flag) in chunk.iter_mut() {
        if *flag {
            has_flag = true;
            acc = identity.clone();
        }
        acc = scan_op(acc, value.clone());
        *value = acc.clone();
        if !has_flag {
            head = acc.clone();
        }
    }
    Summary {
        has_flag,
        head,
        tail: acc,
    }
}

/// Computes an inclusive segmented scan of `data` in place, replacing each
/// value with the prefix total of its segment. The scan is parallel across
/// chunks of `chunk_size` elements; `chunk_size == 0` is treated as 1.
pub(crate) fn scan<T, OP>(data: &mut [(T, bool)], identity: &T, scan_op: &OP, chunk_size: usize)
where
    T: Clone + Send + Sync,
    OP: Fn(T, T) -> T + Sync + Send,
{
    if data.is_empty() {
        return;
    }
    let chunk_size = Ord::max(chunk_size, 1);

    // Phase 1: scan each chunk locally (in parallel) and summarize it.
    let summaries: Vec<Summary<T>> = data
        .par_chunks_mut(chunk_size)
        .map(|chunk| local_scan(chunk, identity, scan_op))
        .collect();

    // Phase 2: fold the chunk summaries left-to-right (one step per chunk, so
    // this is cheap) to find the carry-in value of every chunk. The first
    // chunk gets no carry: its first element implicitly starts a segment.
    let mut carries: Vec<Option<T>> = Vec::with_capacity(summaries.len());
    let mut acc: Option<Summary<T>> = None;
    for summary in &summaries {
        carries.push(acc.as_ref().map(Summary::carry));
        acc = Some(match acc {
            None => summary.clone(),
            Some(acc) => acc.combine(summary, scan_op),
        });
    }

    // Phase 3: apply each chunk carry to its leading partial segment (in
    // parallel), i.e. every element up to the first segment start of the
    // chunk. Because `scan_op` is associative, `carry` can simply be
    // prepended to each locally-scanned prefix of that leading segment.
    data.par_chunks_mut(chunk_size)
        .zip_eq(carries.into_par_iter())
        .for_each(|(chunk, carry)| {
            if let Some(carry) = carry {
                for (value, flag) in chunk.iter_mut() {
                    if *flag {
                        break;
                    }
                    *value = scan_op(carry.clone(), value.clone());
                }
            }
        });
}

/// Picks a default chunk size that gives every thread a few chunks to scan.
fn default_chunk_size(len: usize) -> usize {
    let threads = crate::current_num_threads();
    Ord::max(len / Ord::max(threads * 4, 1), 1)
}

impl<I, F, T, OP> SegmentedScan<I, F, T, OP>
where
    I: IndexedParallelIterator<Item = T>,
    F: IndexedParallelIterator<Item = bool>,
    T: Clone + Send + Sync,
    OP: Fn(T, T) -> T + Sync + Send,
{
    /// Eagerly computes the segmented scan into a `Vec`, in parallel.
    fn compute(self) -> Vec<T> {
        let SegmentedScan {
            base,
            flags,
            identity,
            scan_op,
        } = self;
        let mut data: Vec<(T, bool)> = base.zip_eq(flags).collect();
        let chunk_size = default_chunk_size(data.len());
        scan(&mut data, &identity, &scan_op, chunk_size);
        data.into_iter().map(|(value, _)| value).collect()
    }
}

impl<I, F, T, OP> ParallelIterator for SegmentedScan<I, F, T, OP>
where
    I: IndexedParallelIterator<Item = T>,
    F: IndexedParallelIterator<Item = bool>,
    T: Clone + Send + Sync,
    OP: Fn(T, T) -> T + Sync + Send,
{
    type Item = T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self.compute().into_par_iter(), consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.base.len())
    }
}

impl<I, F, T, OP> IndexedParallelIterator for SegmentedScan<I, F, T, OP>
where
    I: IndexedParallelIterator<Item = T>,
    F: IndexedParallelIterator<Item = bool>,
    T: Clone + Send + Sync,
    OP: Fn(T, T) -> T + Sync + Send,
{
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self.compute().into_par_iter(), consumer)
    }

    fn len(&self) -> usize {
        self.base.len()
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        self.compute().into_par_iter().with_producer(callback)
    }
}
