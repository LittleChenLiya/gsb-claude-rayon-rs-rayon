use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::IndexedParallelIterator;
use super::noop::NoopReducer;
use super::plumbing::{Consumer, Folder};

/// Collects the `Ok` values of a fallible indexed iterator into a vector.
///
/// On success this returns every value in input order. On failure it returns
/// the error with the smallest input index, along with that index, no matter
/// which error happened to complete first.
///
/// This is called by `IndexedParallelIterator::try_collect`.
pub(super) fn try_collect<I, T, E>(pi: I) -> Result<Vec<T>, (usize, E)>
where
    I: IndexedParallelIterator<Item = Result<T, E>>,
    T: Send,
    E: Send,
{
    let len = pi.len();
    let shared = Shared::new(len);
    let consumer = TryCollectConsumer {
        shared: &shared,
        base: 0,
        len,
    };
    pi.drive(consumer);
    shared.into_result()
}

/// State shared by all of the split consumers.
///
/// Each input index owns exactly one slot, written at most once by the folder
/// that produced it, so no synchronization is needed on the slots themselves
/// beyond the `written` flags that publish them.
struct Shared<T, E> {
    slots: Vec<UnsafeCell<MaybeUninit<Result<T, E>>>>,
    written: Vec<AtomicBool>,
    /// The smallest index known to have failed, or `len` while none has.
    min_err: AtomicUsize,
}

// Each slot is written by exactly one folder and only read back after the
// driver has joined all of them, so shared references are safe to send.
unsafe impl<T: Send, E: Send> Send for Shared<T, E> {}
unsafe impl<T: Send, E: Send> Sync for Shared<T, E> {}

impl<T, E> Shared<T, E> {
    fn new(len: usize) -> Self {
        Shared {
            slots: (0..len)
                .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
                .collect(),
            written: (0..len).map(|_| AtomicBool::new(false)).collect(),
            min_err: AtomicUsize::new(len),
        }
    }

    /// Store the item produced for `index`, recording a new smallest error.
    fn write(&self, index: usize, item: Result<T, E>) {
        if item.is_err() {
            self.min_err.fetch_min(index, Ordering::Relaxed);
        }
        // SAFETY: this index is only ever written by this folder.
        unsafe {
            (*self.slots[index].get()).write(item);
        }
        self.written[index].store(true, Ordering::Release);
    }

    /// Best-effort check whether `index` is already dominated by an error at
    /// a smaller index, making its evaluation avoidable. Indices smaller
    /// than the current error are never cancelled.
    fn cancelled(&self, index: usize) -> bool {
        self.min_err.load(Ordering::Relaxed) <= index
    }

    /// Take the item out of a written slot, marking the slot empty again so
    /// the `Drop` impl will not touch it.
    fn take(&mut self, index: usize) -> Result<T, E> {
        debug_assert!(self.written[index].load(Ordering::Relaxed));
        *self.written[index].get_mut() = false;
        // SAFETY: the slot is written and we just claimed ownership of it.
        unsafe { self.slots[index].get_mut().assume_init_read() }
    }

    fn into_result(mut self) -> Result<Vec<T>, (usize, E)> {
        let len = self.slots.len();
        let min_err = *self.min_err.get_mut();
        if min_err < len {
            // Every index below `min_err` was necessarily evaluated, and the
            // slot at `min_err` holds the earliest error by input position.
            return match self.take(min_err) {
                Err(error) => Err((min_err, error)),
                Ok(_) => unreachable!("min_err always points at a written error"),
            };
        }
        // No error was ever recorded, so nothing was ever cancelled and every
        // slot holds an `Ok` value in input order.
        let mut vec = Vec::with_capacity(len);
        for index in 0..len {
            match self.take(index) {
                Ok(value) => vec.push(value),
                Err(_) => unreachable!("no error was recorded"),
            }
        }
        Ok(vec)
    }
}

impl<T, E> Drop for Shared<T, E> {
    fn drop(&mut self) {
        // Drop every produced value that was not moved out by `into_result`,
        // including temporary successes before the error and anything written
        // before a panic unwound through the driver. Slots that were never
        // written (cancelled or never evaluated) hold nothing to drop.
        for (slot, written) in self.slots.iter_mut().zip(&mut self.written) {
            if *written.get_mut() {
                // SAFETY: the flag says this slot was initialized.
                unsafe { slot.get_mut().assume_init_drop() };
            }
        }
    }
}

struct TryCollectConsumer<'a, T, E> {
    shared: &'a Shared<T, E>,
    base: usize,
    len: usize,
}

impl<'a, T, E> Consumer<Result<T, E>> for TryCollectConsumer<'a, T, E>
where
    T: Send,
    E: Send,
{
    type Folder = TryCollectFolder<'a, T, E>;
    type Reducer = NoopReducer;
    type Result = ();

    fn split_at(self, index: usize) -> (Self, Self, NoopReducer) {
        let TryCollectConsumer { shared, base, len } = self;
        debug_assert!(index < len);
        (
            TryCollectConsumer {
                shared,
                base,
                len: index,
            },
            TryCollectConsumer {
                shared,
                base: base + index,
                len: len - index,
            },
            NoopReducer,
        )
    }

    fn into_folder(self) -> Self::Folder {
        TryCollectFolder {
            shared: self.shared,
            base: self.base,
        }
    }

    fn full(&self) -> bool {
        // If this whole range is already dominated by an earlier error, stop
        // splitting; the folder will skip it without evaluating anything.
        self.shared.cancelled(self.base)
    }
}

struct TryCollectFolder<'a, T, E> {
    shared: &'a Shared<T, E>,
    base: usize,
}

impl<'a, T, E> Folder<Result<T, E>> for TryCollectFolder<'a, T, E> {
    type Result = ();

    fn consume(self, item: Result<T, E>) -> Self {
        let TryCollectFolder { shared, base } = self;
        shared.write(base, item);
        TryCollectFolder {
            shared,
            base: base + 1,
        }
    }

    fn consume_iter<I>(self, iter: I) -> Self
    where
        I: IntoIterator<Item = Result<T, E>>,
    {
        let TryCollectFolder { shared, mut base } = self;
        let mut iter = iter.into_iter();
        // Check *before* pulling each item so that indices dominated by an
        // earlier error are never evaluated. Items already produced by the
        // iterator adaptor are consumed normally; we simply stop asking for
        // more. Indices below the current error are never skipped this way.
        while !shared.cancelled(base) {
            match iter.next() {
                Some(item) => {
                    shared.write(base, item);
                    base += 1;
                }
                None => break,
            }
        }
        TryCollectFolder { shared, base }
    }

    fn complete(self) {}

    fn full(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{Duration, Instant};

    /// Wait for `flag` without ever blocking a thread pool worker forever:
    /// gives up after a bounded timeout so single-threaded pools cannot
    /// deadlock no matter which index happens to run first.
    fn wait_for(flag: &AtomicBool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !flag.load(Ordering::Relaxed) && Instant::now() < deadline {
            thread::yield_now();
        }
    }

    fn run_two_and_seven(threads: usize) -> Result<Vec<i32>, (usize, String)> {
        let pool = crate::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let seven_done = AtomicBool::new(false);
        pool.install(|| {
            (0..8i32)
                .into_par_iter()
                .with_max_len(1)
                .map(|i| {
                    if i == 7 {
                        // Index 7 fails and finishes first.
                        seven_done.store(true, Ordering::Relaxed);
                        Err(format!("error at {i}"))
                    } else if i == 2 {
                        // Index 2 fails too, but only completes after 7.
                        wait_for(&seven_done);
                        Err(format!("error at {i}"))
                    } else {
                        Ok(i)
                    }
                })
                .try_collect()
        })
    }

    #[test]
    fn smallest_error_index_wins_multithreaded() {
        // Index 7 records its error before index 2, yet the answer must be 2.
        assert_eq!(run_two_and_seven(4), Err((2, String::from("error at 2"))));
    }

    #[test]
    fn smallest_error_index_wins_single_threaded() {
        // The timeout-based wait keeps a one-thread pool from deadlocking.
        assert_eq!(run_two_and_seven(1), Err((2, String::from("error at 2"))));
    }

    #[test]
    fn smallest_error_index_wins_barrier_race() {
        // Two threads rendezvous so indices 2 and 15 start together, then 15
        // fails immediately while 2 sleeps: 15's error is recorded first, but
        // index 2 is still the answer.
        let pool = crate::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let barrier = Barrier::new(2);
        let result = pool.install(|| {
            (0..16i32)
                .into_par_iter()
                .with_max_len(1)
                .map(|i| {
                    if i == 2 || i == 15 {
                        barrier.wait();
                        if i == 2 {
                            thread::sleep(Duration::from_millis(100));
                        }
                        Err(format!("error at {i}"))
                    } else {
                        Ok(i)
                    }
                })
                .try_collect()
        });
        assert_eq!(result, Err((2, String::from("error at 2"))));
    }

    #[test]
    fn success_preserves_input_order() {
        let vec = (0..997i32)
            .into_par_iter()
            .map(|i| Ok::<i32, String>(i * 2))
            .try_collect()
            .unwrap();
        assert_eq!(vec.len(), 997);
        for (i, v) in vec.into_iter().enumerate() {
            assert_eq!(v, i as i32 * 2);
        }
    }

    #[test]
    fn empty_input_is_ok() {
        let vec = (0..0i32)
            .into_par_iter()
            .map(Ok::<i32, String>)
            .try_collect()
            .unwrap();
        assert!(vec.is_empty());
    }

    #[test]
    fn each_input_evaluated_at_most_once() {
        let counts: Vec<AtomicUsize> = (0..1000).map(|_| AtomicUsize::new(0)).collect();
        let vec = (0..1000i32)
            .into_par_iter()
            .with_max_len(3)
            .map(|i| {
                counts[i as usize].fetch_add(1, Ordering::Relaxed);
                Ok::<i32, String>(i)
            })
            .try_collect()
            .unwrap();
        assert_eq!(vec.len(), 1000);
        for count in &counts {
            assert_eq!(count.load(Ordering::Relaxed), 1);
        }
    }

    /// A value that counts how many were produced and how many were dropped,
    /// to prove nothing leaks or drops twice on any path.
    #[derive(Debug)]
    struct Counted {
        dropped: Arc<AtomicUsize>,
    }

    impl Counted {
        fn new(produced: &Arc<AtomicUsize>, dropped: &Arc<AtomicUsize>) -> Self {
            produced.fetch_add(1, Ordering::Relaxed);
            Counted {
                dropped: dropped.clone(),
            }
        }

        fn counters() -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
            (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)))
        }
    }

    impl Drop for Counted {
        fn drop(&mut self) {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn get(counter: &AtomicUsize) -> usize {
        counter.load(Ordering::Relaxed)
    }

    #[test]
    fn no_leaks_on_success() {
        let (produced, dropped) = Counted::counters();
        let vec = (0..500)
            .into_par_iter()
            .map(|_| Ok::<_, String>(Counted::new(&produced, &dropped)))
            .try_collect()
            .unwrap();
        assert_eq!(get(&produced), 500);
        drop(vec);
        assert_eq!(get(&dropped), 500);
    }

    #[test]
    fn no_leaks_on_error() {
        let (produced, dropped) = Counted::counters();
        let result = (0..500i32)
            .into_par_iter()
            .with_max_len(1)
            .map(|i| {
                if i == 100 {
                    Err(Counted::new(&produced, &dropped))
                } else {
                    Ok(Counted::new(&produced, &dropped))
                }
            })
            .try_collect();
        let (index, error) = result.unwrap_err();
        assert_eq!(index, 100);
        // Everything produced so far except the returned error is dropped.
        assert_eq!(get(&dropped), get(&produced) - 1);
        drop(error);
        assert_eq!(get(&dropped), get(&produced));
    }

    #[test]
    fn no_leaks_on_panic() {
        let (produced, dropped) = Counted::counters();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = (0..500i32)
                .into_par_iter()
                .with_max_len(1)
                .map(|i| {
                    if i == 250 {
                        panic!("boom at {i}");
                    }
                    Ok::<_, String>(Counted::new(&produced, &dropped))
                })
                .try_collect();
        }));
        assert!(result.is_err());
        // Every value produced before the panic was released during unwinding.
        assert_eq!(get(&dropped), get(&produced));
    }
}
