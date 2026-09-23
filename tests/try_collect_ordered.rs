use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex};

fn pool(threads: usize) -> rayon::ThreadPool {
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

/// Indices 2 and 7 both fail, and index 7 is forced to record its error
/// first, yet the result must still report index 2.
///
/// The waiting is done with `rayon::yield_now()`, which lets the current
/// worker run other pool tasks while it waits, so this cannot deadlock
/// even on a single-threaded pool.
#[test]
fn lowest_index_error_wins_single_thread() {
    let later_failed = AtomicBool::new(false);
    let calls = Mutex::new(Vec::new());

    let result: Result<Vec<i32>, (usize, i32)> = pool(1).install(|| {
        (0..10).into_par_iter().try_collect_ordered(|x| {
            calls.lock().unwrap().push(x);
            match x {
                7 => {
                    // The later error records itself first...
                    later_failed.store(true, Ordering::SeqCst);
                    Err(x)
                }
                2 => {
                    // ... and index 2 only fails afterwards. `yield_now`
                    // drives the pool (even the 1-thread pool) so the
                    // task evaluating index 7 can run while we wait.
                    while !later_failed.load(Ordering::SeqCst) {
                        if rayon::yield_now().is_none() {
                            std::thread::yield_now();
                        }
                    }
                    Err(x)
                }
                _ => Ok(x * 10),
            }
        })
    });

    // The error at index 2 wins even though index 7 failed first.
    assert_eq!(result, Err((2, 2)));

    let calls = calls.into_inner().unwrap();
    // Every input below the winning error was still evaluated, even though
    // some of them finished after the error at index 7 was already known.
    for i in 0..=2 {
        assert!(calls.contains(&i), "index {i} must still be evaluated");
    }
    assert!(calls.contains(&7));
    // No input was evaluated more than once.
    let dedup: BTreeSet<_> = calls.iter().collect();
    assert_eq!(
        dedup.len(),
        calls.len(),
        "each input evaluated at most once"
    );
}

/// The same guarantee under a real race: a barrier makes the errors at
/// indices 2 and 7 complete as concurrently as possible, many times.
#[test]
fn lowest_index_error_wins_under_barrier_race() {
    let pool = pool(4);
    for _ in 0..100 {
        let barrier = Barrier::new(2);
        let result: Result<Vec<i32>, (usize, i32)> = pool.install(|| {
            (0..10).into_par_iter().try_collect_ordered(|x| {
                if x == 2 || x == 7 {
                    // With 4 workers and small leaves, the two failing
                    // tasks always run on different workers, so this
                    // barrier cannot deadlock the pool.
                    barrier.wait();
                    Err(x)
                } else {
                    Ok(x)
                }
            })
        });
        assert_eq!(result, Err((2, 2)));
    }
}

#[test]
fn success_preserves_input_order() {
    // 997 is prime, so the splits are not neatly aligned.
    let result: Result<Vec<i32>, (usize, i32)> =
        pool(4).install(|| (0..997).into_par_iter().try_collect_ordered(|x| Ok(x * 2)));
    let expected: Vec<i32> = (0..997).map(|x| x * 2).collect();
    assert_eq!(result, Ok(expected));
}

#[test]
fn empty_input() {
    let calls = AtomicUsize::new(0);
    let result: Result<Vec<i32>, (usize, i32)> = pool(1).install(|| {
        (0..0).into_par_iter().try_collect_ordered(|x| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(x)
        })
    });
    assert_eq!(result, Ok(vec![]));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

/// Once an error is known, later inputs that have not started yet may be
/// skipped. On a single-threaded pool inputs are evaluated in ascending
/// order, so an error at index 0 makes everything else skippable.
#[test]
fn later_inputs_are_skipped_after_an_error() {
    let calls = AtomicUsize::new(0);
    let result: Result<Vec<i32>, (usize, i32)> = pool(1).install(|| {
        (0..10).into_par_iter().try_collect_ordered(|x| {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(x)
        })
    });
    assert_eq!(result, Err((0, 0)));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

static LIVE_VALUES: AtomicUsize = AtomicUsize::new(0);
static LIVE_ERRORS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
struct CountedValue(i32);

impl CountedValue {
    fn new(x: i32) -> Self {
        LIVE_VALUES.fetch_add(1, Ordering::SeqCst);
        CountedValue(x)
    }
}

impl Drop for CountedValue {
    fn drop(&mut self) {
        LIVE_VALUES.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Debug)]
struct CountedError(i32);

impl CountedError {
    fn new(x: i32) -> Self {
        LIVE_ERRORS.fetch_add(1, Ordering::SeqCst);
        CountedError(x)
    }
}

impl Drop for CountedError {
    fn drop(&mut self) {
        LIVE_ERRORS.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Successful temporaries, unreported errors, and skipped inputs must all
/// be freed exactly once, on the success, error, and panic paths alike.
#[test]
#[cfg_attr(not(panic = "unwind"), ignore)]
fn drop_accounting() {
    let pool = pool(4);

    // All successes: the values are handed to the caller and freed with it.
    let result = pool.install(|| {
        (0..100)
            .into_par_iter()
            .try_collect_ordered(|x| Ok::<_, CountedError>(CountedValue::new(x)))
    });
    assert_eq!(LIVE_VALUES.load(Ordering::SeqCst), 100);
    assert_eq!(LIVE_ERRORS.load(Ordering::SeqCst), 0);
    let values = result.unwrap();
    assert_eq!(values.iter().map(|v| v.0).sum::<i32>(), (0..100).sum());
    drop(values);
    assert_eq!(LIVE_VALUES.load(Ordering::SeqCst), 0);

    // Failure: successful temporaries and the unreported error are freed;
    // only the reported error stays alive. 13 is not a power of two.
    let result = pool.install(|| {
        (0..13).into_par_iter().try_collect_ordered(|x| {
            if x == 4 || x == 9 {
                Err(CountedError::new(x))
            } else {
                Ok(CountedValue::new(x))
            }
        })
    });
    assert_eq!(LIVE_VALUES.load(Ordering::SeqCst), 0);
    assert_eq!(LIVE_ERRORS.load(Ordering::SeqCst), 1);
    let (index, error) = result.unwrap_err();
    assert_eq!(index, 4);
    assert_eq!(error.0, 4);
    drop(error);
    assert_eq!(LIVE_ERRORS.load(Ordering::SeqCst), 0);

    // Panic: everything created before the panic is freed during unwind,
    // and the panic itself propagates instead of becoming an `Err`.
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        pool.install(|| {
            (0..100).into_par_iter().try_collect_ordered(
                |x| -> Result<CountedValue, CountedError> {
                    if x == 50 {
                        panic!("boom");
                    }
                    Ok(CountedValue::new(x))
                },
            )
        })
    }));
    assert!(result.is_err());
    assert_eq!(LIVE_VALUES.load(Ordering::SeqCst), 0);
    assert_eq!(LIVE_ERRORS.load(Ordering::SeqCst), 0);
}
