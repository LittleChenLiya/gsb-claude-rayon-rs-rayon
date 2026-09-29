use super::IndexedParallelIterator;
use super::plumbing::{Producer, ProducerCallback};
use crate::join;

/// Reduces the items of an indexed parallel iterator with a fixed binary
/// reduction tree, returning `None` for an empty iterator.
///
/// The tree depends only on the item indices: every segment is split at its
/// midpoint (`len / 2`), and the results of the left and right halves are
/// combined left-to-right. The number of worker threads therefore only
/// affects which tasks run where, never the order in which `op` is applied.
pub(super) fn tree_reduce<PI, OP, T>(pi: PI, op: OP) -> Option<T>
where
    PI: IndexedParallelIterator<Item = T>,
    OP: Fn(T, T) -> T + Sync,
    T: Send,
{
    let len = pi.len();
    if len == 0 {
        return None;
    }
    pi.with_producer(TreeReduceCallback { len, op: &op })
}

struct TreeReduceCallback<'op, OP> {
    len: usize,
    op: &'op OP,
}

impl<'op, OP, T> ProducerCallback<T> for TreeReduceCallback<'op, OP>
where
    OP: Fn(T, T) -> T + Sync,
    T: Send,
{
    type Output = Option<T>;

    fn callback<P>(self, producer: P) -> Self::Output
    where
        P: Producer<Item = T>,
    {
        Some(reduce(self.len, producer, self.op))
    }
}

/// Splits `producer` at fixed midpoints until each subtree owns a single
/// item. The recursion shape is driven entirely by `len`, so the join tree
/// (and hence the order of `op`) is independent of the runtime scheduling.
fn reduce<P, OP, T>(len: usize, producer: P, op: &OP) -> T
where
    P: Producer<Item = T>,
    OP: Fn(T, T) -> T + Sync,
    T: Send,
{
    debug_assert!(len > 0);
    if len == 1 {
        producer
            .into_iter()
            .next()
            .expect("exact-size producer yielded no item")
    } else {
        let mid = len / 2;
        let (left_producer, right_producer) = producer.split_at(mid);
        let (left, right) = join(
            move || reduce(mid, left_producer, op),
            move || reduce(len - mid, right_producer, op),
        );
        op(left, right)
    }
}

/// Reduces the items of an indexed parallel iterator with the same fixed
/// binary tree as [`tree_reduce`], using `identity` only for the empty case.
pub(super) fn tree_reduce_with_identity<PI, ID, OP, T>(pi: PI, identity: ID, op: OP) -> T
where
    PI: IndexedParallelIterator<Item = T>,
    ID: FnOnce() -> T,
    OP: Fn(T, T) -> T + Sync,
    T: Send,
{
    tree_reduce(pi, op).unwrap_or_else(identity)
}
