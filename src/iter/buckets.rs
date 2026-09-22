//! Stable parallel partitioning into numbered buckets.

use super::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use crate::SendPtr;
use std::error::Error;
use std::fmt;
use std::mem::ManuallyDrop;

/// The error type for [`IndexedParallelIterator::partition_into_buckets`].
///
/// [`IndexedParallelIterator::partition_into_buckets`]:
///     trait.IndexedParallelIterator.html#method.partition_into_buckets
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BucketError {
    /// The requested number of buckets was zero.
    NoBuckets,
    /// The classifier returned a bucket index outside of `0..num_buckets`.
    BucketOutOfRange {
        /// The offending bucket index.
        bucket: usize,
        /// The number of buckets that were requested.
        num_buckets: usize,
    },
    /// A capacity computation overflowed or an allocation failed.
    CapacityOverflow,
}

impl fmt::Display for BucketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BucketError::NoBuckets => write!(f, "the number of buckets must not be zero"),
            BucketError::BucketOutOfRange {
                bucket,
                num_buckets,
            } => write!(
                f,
                "classifier returned bucket {bucket}, which is out of range 0..{num_buckets}"
            ),
            BucketError::CapacityOverflow => {
                write!(f, "bucket capacity computation overflowed")
            }
        }
    }
}

impl Error for BucketError {}

/// Stable partition of an indexed parallel iterator into numbered buckets.
///
/// This is called by `IndexedParallelIterator::partition_into_buckets`.
pub(super) fn partition_into_buckets<I, F>(
    pi: I,
    num_buckets: usize,
    classify: F,
) -> Result<Vec<Vec<I::Item>>, BucketError>
where
    I: IndexedParallelIterator,
    F: Fn(&I::Item) -> usize + Sync + Send,
{
    if num_buckets == 0 {
        return Err(BucketError::NoBuckets);
    }

    // Pass 1 (parallel): classify every item exactly once and split the
    // `(bucket, item)` pairs into two vectors, both in the original order.
    //
    // `unzip_into_vecs` writes through raw pointers guarded by an
    // initialized-length counter, so if `classify` panics, every item that
    // was already written is dropped exactly once and uninitialized slots
    // are never read as values.
    let mut ids: Vec<usize> = Vec::new();
    let mut items: Vec<I::Item> = Vec::new();
    pi.map(|item| (classify(&item), item))
        .unzip_into_vecs(&mut ids, &mut items);

    // Reject out-of-range bucket indices before publishing anything.
    // Dropping `items` on this path drops every element exactly once.
    if let Some(&bucket) = ids.iter().find(|&&bucket| bucket >= num_buckets) {
        return Err(BucketError::BucketOutOfRange {
            bucket,
            num_buckets,
        });
    }

    let len = ids.len();
    if len == 0 {
        return Ok((0..num_buckets).map(|_| Vec::new()).collect());
    }

    // Choose a modest number of chunks so the per-chunk histograms and
    // offset tables stay small. The result is independent of this choice.
    let num_chunks = (crate::current_num_threads() * 4).clamp(1, len);
    let chunk_size = len.div_ceil(num_chunks);
    let num_chunks = len.div_ceil(chunk_size);

    // Pass 2 (parallel): count the items of each bucket within each chunk.
    let histograms: Vec<Vec<usize>> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk| {
            let start = chunk * chunk_size;
            let end = (start + chunk_size).min(len);
            let mut histogram = vec![0; num_buckets];
            for &id in &ids[start..end] {
                histogram[id] += 1;
            }
            histogram
        })
        .collect();

    // Exclusive prefix sums (sequential, O(num_chunks * num_buckets)):
    // `offsets[chunk][bucket]` is the position within `bucket` of the
    // first item of `chunk` that belongs to it. All arithmetic is checked
    // so a capacity overflow is reported instead of wrapping.
    let mut totals = vec![0usize; num_buckets];
    for histogram in &histograms {
        for (total, &count) in totals.iter_mut().zip(histogram) {
            *total = total
                .checked_add(count)
                .ok_or(BucketError::CapacityOverflow)?;
        }
    }
    let mut offsets: Vec<Vec<usize>> = Vec::with_capacity(num_chunks);
    let mut running = vec![0usize; num_buckets];
    for histogram in &histograms {
        offsets.push(running.clone());
        for (offset, &count) in running.iter_mut().zip(histogram) {
            *offset = offset
                .checked_add(count)
                .ok_or(BucketError::CapacityOverflow)?;
        }
    }

    // Reserve the exact final capacity of every bucket up front, so the
    // scatter below performs no allocation and cannot fail.
    let mut buckets: Vec<Vec<I::Item>> = Vec::with_capacity(num_buckets);
    for &total in &totals {
        let mut bucket = Vec::new();
        bucket
            .try_reserve_exact(total)
            .map_err(|_| BucketError::CapacityOverflow)?;
        buckets.push(bucket);
    }

    // Pass 3 (parallel scatter): move every item directly from the `items`
    // buffer to its final position. Each destination slot is written exactly
    // once, and the slots written by different chunks are disjoint, so no
    // locks are needed.
    //
    // Safety invariants:
    // * `items` holds `len` initialized elements. It is wrapped in
    //   `ManuallyDrop`, and each element is read out exactly once below
    //   (the chunk ranges partition `0..len`), so no element is dropped
    //   twice and none is leaked on the success path.
    // * For each bucket, `offsets` places every write within
    //   `0..totals[bucket]`, i.e. inside the reserved capacity, and every
    //   slot in that range is written exactly once, because the histograms
    //   and offsets come from the same counts.
    // * No user code runs in this pass, so it cannot panic. Even if it
    //   somehow did, `ManuallyDrop` and the not-yet-published bucket
    //   lengths would leak memory rather than double-drop or read
    //   uninitialized slots as values.
    let mut items = ManuallyDrop::new(items);
    let items_ptr = SendPtr(items.as_mut_ptr());
    let bucket_ptrs: Vec<SendPtr<I::Item>> = buckets
        .iter_mut()
        .map(|bucket| SendPtr(bucket.as_mut_ptr()))
        .collect();

    (0..num_chunks).into_par_iter().for_each(|chunk| {
        let start = chunk * chunk_size;
        let end = (start + chunk_size).min(len);
        let mut cursor = offsets[chunk].clone();
        for (i, &bucket) in ids[start..end].iter().enumerate() {
            // SAFETY: see the invariants above.
            unsafe {
                let item = items_ptr.get().add(start + i).read();
                bucket_ptrs[bucket].get().add(cursor[bucket]).write(item);
            }
            cursor[bucket] += 1;
        }
    });

    // Every slot of every bucket is now initialized; publish the lengths.
    for (bucket, &total) in buckets.iter_mut().zip(&totals) {
        // SAFETY: the scatter above initialized exactly `total` elements.
        unsafe { bucket.set_len(total) };
    }

    // Free the `items` buffer without dropping the moved-out elements.
    // SAFETY: all `len` elements were moved out by the scatter above.
    unsafe { ManuallyDrop::into_inner(items).set_len(0) };

    Ok(buckets)
}

#[cfg(test)]
mod test;
