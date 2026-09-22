//! Stable parallel partitioning of an indexed iterator into numbered buckets.
//!
//! This module implements [`IndexedParallelIterator::partition_into_buckets`].
//!
//! The algorithm runs in two phases:
//!
//! 1. **Classify and buffer (parallel).** The iterator is driven with a
//!    consumer whose folders buffer their items locally, invoking the
//!    classification function exactly once per item and counting how many
//!    items fall into each bucket. Each leaf folder owns a private
//!    [`Chunk`], so no element ever contends on a shared lock, and the
//!    input is never re-scanned per bucket.
//! 2. **Scatter (parallel).** The per-chunk counts are combined with a
//!    sequential prefix pass to compute, for every chunk and bucket, the
//!    exact output position of that chunk's run inside the bucket. Each
//!    chunk then moves its buffered items directly into the spare capacity
//!    of the destination bucket vectors, at disjoint positions, so the
//!    writes can happen in parallel without synchronization.
//!
//! Because chunk results are always reduced left-to-right, the bucket
//! contents preserve the original iterator order, and the result is
//! independent of the thread count and splitting granularity.
//!
//! # Ownership and panic safety
//!
//! Phase 1 is entirely safe code: every consumed item lives in an ordinary
//! [`Vec`] inside its chunk, so if the classification function panics, each
//! item that was taken from the producer is dropped exactly once by normal
//! [`Vec`] drop semantics, and items never taken remain owned (and dropped)
//! by the producer. No uninitialized memory is ever read.
//!
//! Phase 2 uses a small amount of `unsafe` to write into the spare
//! capacity of the output vectors. The invariants are:
//!
//! - each bucket vector is allocated with capacity equal to its final
//!   length before any write happens, and is never reallocated during the
//!   scatter, so the raw pointers stay valid;
//! - the per-chunk offsets partition each bucket's `0..len` range into
//!   disjoint runs, so every slot is written exactly once, by exactly one
//!   chunk, and no slot is ever read before it is initialized;
//! - `set_len` is only called after the scatter has completed, so every
//!   slot below the new length is initialized;
//! - the scatter runs no user code, so it cannot panic in practice; if an
//!   internal assertion ever did panic, the still-zero-length vectors would
//!   simply leak the already-written items rather than double-drop them or
//!   expose uninitialized memory.

use super::plumbing::*;
use super::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use crate::SendPtr;
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Sentinel for the shared "first invalid bucket" slot: no error recorded.
const NO_INVALID: usize = usize::MAX;

/// The error type returned by
/// [`IndexedParallelIterator::partition_into_buckets`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PartitionBucketsError {
    /// The requested number of buckets was zero.
    ZeroBuckets,
    /// The classification function returned a bucket index that was not in
    /// `0..num_buckets`.
    InvalidBucket {
        /// The bucket index that was returned.
        bucket: usize,
        /// The number of buckets that were requested.
        num_buckets: usize,
    },
    /// A capacity computation overflowed `usize`.
    CapacityOverflow,
}

impl fmt::Display for PartitionBucketsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            PartitionBucketsError::ZeroBuckets => {
                write!(f, "the number of buckets must be greater than zero")
            }
            PartitionBucketsError::InvalidBucket {
                bucket,
                num_buckets,
            } => write!(
                f,
                "classification returned bucket {bucket}, which is out of range 0..{num_buckets}"
            ),
            PartitionBucketsError::CapacityOverflow => {
                write!(f, "capacity overflow while sizing the buckets")
            }
        }
    }
}

impl Error for PartitionBucketsError {}

/// The items of one leaf task, buffered in iterator order, along with the
/// bucket each item was assigned to and per-bucket totals.
struct Chunk<T> {
    items: Vec<T>,
    buckets: Vec<usize>,
    counts: Vec<usize>,
}

/// Partitions the items of `pi` into `num_buckets` buckets according to
/// `classify`, preserving the relative order of items within each bucket.
pub(super) fn partition_into_buckets<I, F>(
    pi: I,
    num_buckets: usize,
    classify: F,
) -> Result<Vec<Vec<I::Item>>, PartitionBucketsError>
where
    I: IndexedParallelIterator,
    F: Fn(&I::Item) -> usize + Sync + Send,
{
    if num_buckets == 0 {
        return Err(PartitionBucketsError::ZeroBuckets);
    }

    let len = pi.len();

    // Reject sizes that cannot possibly be allocated, instead of panicking
    // later inside `Vec`. This covers the output items, the bucket vector
    // itself, and the per-bucket count/offset tables.
    if std::alloc::Layout::array::<I::Item>(len).is_err()
        || std::alloc::Layout::array::<Vec<I::Item>>(num_buckets).is_err()
        || std::alloc::Layout::array::<usize>(num_buckets).is_err()
    {
        return Err(PartitionBucketsError::CapacityOverflow);
    }

    if len == 0 {
        return Ok((0..num_buckets).map(|_| Vec::new()).collect());
    }

    // Phase 1: classify and buffer in parallel. `classify` is called exactly
    // once per item, and each folder works on private buffers, so there is
    // no shared lock to contend on.
    let invalid = AtomicUsize::new(NO_INVALID);
    let consumer = BucketConsumer {
        classify: &classify,
        num_buckets,
        invalid: &invalid,
    };
    let chunks: Vec<Chunk<I::Item>> = pi.drive(consumer);

    let bucket = invalid.load(Ordering::Relaxed);
    if bucket != NO_INVALID {
        // `chunks` drops here, releasing every buffered item exactly once;
        // no partial result is published.
        return Err(PartitionBucketsError::InvalidBucket {
            bucket,
            num_buckets,
        });
    }

    // Total length of each bucket.
    let mut totals = vec![0usize; num_buckets];
    for chunk in &chunks {
        for (b, &count) in chunk.counts.iter().enumerate() {
            totals[b] += count;
        }
    }

    // Allocate one vector per bucket, sized exactly to its final length.
    let mut buckets: Vec<Vec<I::Item>> = totals
        .iter()
        .map(|&count| Vec::with_capacity(count))
        .collect();
    let bucket_ptrs: Vec<SendPtr<I::Item>> = buckets
        .iter_mut()
        .map(|bucket| SendPtr(bucket.as_mut_ptr()))
        .collect();

    // Prefix pass over the chunks (in iterator order): `offsets[i][b]` is
    // the position within bucket `b` where chunk `i`'s run begins.
    let mut offsets: Vec<Vec<usize>> = Vec::with_capacity(chunks.len());
    let mut running = vec![0usize; num_buckets];
    for chunk in &chunks {
        offsets.push(running.clone());
        for (b, &count) in chunk.counts.iter().enumerate() {
            running[b] += count;
        }
    }

    // Phase 2: scatter the buffered items into their final positions.
    // Distinct chunks write to disjoint slots of every bucket, so this can
    // run in parallel without any synchronization.
    let bucket_ptrs = &bucket_ptrs;
    chunks
        .into_par_iter()
        .zip(offsets)
        .for_each(|(chunk, mut pos)| {
            let Chunk { items, buckets, .. } = chunk;
            for (item, b) in items.into_iter().zip(buckets) {
                // SAFETY: `bucket_ptrs[b]` points into the allocation of
                // output bucket `b`, which has capacity `totals[b]`. The
                // prefix pass guarantees that this chunk owns the slot range
                // `offsets[i][b]..offsets[i][b] + counts[i][b]` of bucket
                // `b`, and `pos[b]` walks exactly that range, so this write
                // is in bounds and no other chunk touches the same slot.
                // `ptr::write` moves `item` in without reading or dropping
                // the uninitialized slot.
                unsafe {
                    bucket_ptrs[b].0.add(pos[b]).write(item);
                }
                pos[b] += 1;
            }
        });

    // SAFETY: every bucket received exactly `totals[b]` writes, covering
    // each slot in `0..totals[b]` exactly once, so the prefix of that
    // length is fully initialized.
    for (bucket, &count) in buckets.iter_mut().zip(&totals) {
        unsafe {
            bucket.set_len(count);
        }
    }

    Ok(buckets)
}

/// Consumer for phase 1: splits freely, sharing the classification
/// function and the "first invalid bucket" slot.
struct BucketConsumer<'a, F> {
    classify: &'a F,
    num_buckets: usize,
    invalid: &'a AtomicUsize,
}

impl<'a, T, F> Consumer<T> for BucketConsumer<'a, F>
where
    T: Send,
    F: Fn(&T) -> usize + Sync + Send,
{
    type Folder = BucketFolder<'a, F, T>;
    type Reducer = AppendChunks;
    type Result = Vec<Chunk<T>>;

    fn split_at(self, _index: usize) -> (Self, Self, AppendChunks) {
        let left = BucketConsumer { ..self };
        (left, self, AppendChunks)
    }

    fn into_folder(self) -> Self::Folder {
        BucketFolder {
            classify: self.classify,
            num_buckets: self.num_buckets,
            invalid: self.invalid,
            items: Vec::new(),
            buckets: Vec::new(),
            counts: vec![0; self.num_buckets],
        }
    }

    fn full(&self) -> bool {
        self.invalid.load(Ordering::Relaxed) != NO_INVALID
    }
}

/// Folder for phase 1: buffers items and records their buckets.
struct BucketFolder<'a, F, T> {
    classify: &'a F,
    num_buckets: usize,
    invalid: &'a AtomicUsize,
    items: Vec<T>,
    buckets: Vec<usize>,
    counts: Vec<usize>,
}

impl<T, F> Folder<T> for BucketFolder<'_, F, T>
where
    F: Fn(&T) -> usize + Sync + Send,
{
    type Result = Vec<Chunk<T>>;

    fn consume(mut self, item: T) -> Self {
        let bucket = (self.classify)(&item);
        if bucket >= self.num_buckets {
            // Record the first invalid bucket seen and stop early. `item`
            // is dropped here, exactly once, like any other buffered item
            // will be when its chunk is dropped.
            let _ = self.invalid.compare_exchange(
                NO_INVALID,
                bucket,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        } else {
            self.counts[bucket] += 1;
            self.buckets.push(bucket);
            self.items.push(item);
        }
        self
    }

    fn complete(self) -> Self::Result {
        vec![Chunk {
            items: self.items,
            buckets: self.buckets,
            counts: self.counts,
        }]
    }

    fn full(&self) -> bool {
        self.invalid.load(Ordering::Relaxed) != NO_INVALID
    }
}

/// Reducer for phase 1: concatenates the chunk lists, keeping left chunks
/// before right chunks so the global order matches the iterator order.
struct AppendChunks;

impl<T> Reducer<Vec<Chunk<T>>> for AppendChunks {
    fn reduce(self, mut left: Vec<Chunk<T>>, mut right: Vec<Chunk<T>>) -> Vec<Chunk<T>> {
        left.append(&mut right);
        left
    }
}

#[cfg(test)]
mod test;
