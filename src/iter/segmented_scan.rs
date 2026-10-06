use super::*;
use rayon_core::join;

/// `SegmentedScan` is an iterator that yields the inclusive prefix
/// combination of every segment of the input iterator.
///
/// This struct is created by the [`segmented_scan()`] method on
/// [`IndexedParallelIterator`].
///
/// [`segmented_scan()`]: IndexedParallelIterator::segmented_scan()
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Debug, Clone)]
pub struct SegmentedScan<I, T, F> {
    base: I,
    identity: T,
    op: F,
}

impl<I, T, F> SegmentedScan<I, T, F> {
    /// Creates a new `SegmentedScan` iterator.
    pub(super) fn new(base: I, identity: T, op: F) -> Self {
        SegmentedScan { base, identity, op }
    }
}

impl<I, T, F> ParallelIterator for SegmentedScan<I, T, F>
where
    I: IndexedParallelIterator<Item = (T, bool)>,
    T: Clone + Send,
    F: Fn(&T, &T) -> T + Sync + Send,
{
    type Item = T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.base.len())
    }
}

impl<I, T, F> IndexedParallelIterator for SegmentedScan<I, T, F>
where
    I: IndexedParallelIterator<Item = (T, bool)>,
    T: Clone + Send,
    F: Fn(&T, &T) -> T + Sync + Send,
{
    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.base.len()
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        let SegmentedScan { base, identity, op } = self;

        // Collect the `(value, is_segment_start)` pairs in parallel,
        // preserving their original order.
        let mut items: Vec<(T, bool)> = base.collect();

        // Compute the inclusive segmented scan in place, in parallel.
        scan_segments(&mut items, &identity, &op);

        let output: Vec<T> = items.into_iter().map(|(value, _)| value).collect();
        output.into_par_iter().with_producer(callback)
    }
}

/// A summary of one scanned block, used to stitch neighboring blocks
/// together without losing the left-to-right order of the operation.
struct Summary<T> {
    /// Whether the block contains any segment-start marker.
    has_flag: bool,
    /// The number of leading items before the first segment-start marker.
    lead_len: usize,
    /// The combined value of the trailing (still open) segment.
    tail: T,
}

/// Blocks at or below this length are scanned sequentially.
const GRANULARITY: usize = 1024;

/// Computes an inclusive segmented scan of `items` in place, replacing every
/// value with the combination of its segment up to and including itself.
fn scan_segments<T, F>(items: &mut [(T, bool)], identity: &T, op: &F)
where
    T: Clone + Send,
    F: Fn(&T, &T) -> T + Sync,
{
    if !items.is_empty() {
        scan_recursive(items, identity, op);
    }
}

fn scan_recursive<T, F>(items: &mut [(T, bool)], identity: &T, op: &F) -> Summary<T>
where
    T: Clone + Send,
    F: Fn(&T, &T) -> T + Sync,
{
    let len = items.len();
    if len <= GRANULARITY {
        return scan_sequential(items, identity, op);
    }

    let mid = len / 2;
    let (left, right) = items.split_at_mut(mid);
    // Clone the identity for both halves, so the closures only ever share
    // a reference to `op` (which is `Sync`) and `T: Sync` is not required.
    // The right closure hands its half of the slice back, so it can still
    // be used for the fix-up below.
    let identity_left = identity.clone();
    let identity_right = identity.clone();
    let (left_sum, (right_sum, right)) = join(
        move || scan_recursive(left, &identity_left, op),
        move || {
            let summary = scan_recursive(right, &identity_right, op);
            (summary, right)
        },
    );

    // The leading run of the right half (the items before its first segment
    // start) belongs to the segment that is still open at the end of the
    // left half, so the left half's trailing value must be prepended to
    // those results. Because the operation is associative with `identity`
    // as its unit, `op(left_tail, fold(identity, ..))` is the same as
    // folding that run with `left_tail` in the first place, and the
    // original left-to-right order is preserved.
    for (value, _) in &mut right[..right_sum.lead_len] {
        *value = op(&left_sum.tail, value);
    }

    Summary {
        has_flag: left_sum.has_flag || right_sum.has_flag,
        lead_len: if left_sum.has_flag {
            left_sum.lead_len
        } else {
            mid + right_sum.lead_len
        },
        tail: if right_sum.has_flag {
            right_sum.tail
        } else {
            op(&left_sum.tail, &right_sum.tail)
        },
    }
}

fn scan_sequential<T, F>(items: &mut [(T, bool)], identity: &T, op: &F) -> Summary<T>
where
    T: Clone,
    F: Fn(&T, &T) -> T,
{
    let mut carry = identity.clone();
    let mut has_flag = false;
    let mut lead_len = 0;
    for (index, (value, is_start)) in items.iter_mut().enumerate() {
        if *is_start {
            carry = identity.clone();
            has_flag = true;
        } else if !has_flag {
            lead_len = index + 1;
        }
        carry = op(&carry, value);
        *value = carry.clone();
    }
    Summary {
        has_flag,
        lead_len,
        tail: carry,
    }
}
