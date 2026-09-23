//! Demonstrates `IndexedParallelIterator::try_collect_ordered`.
//!
//! Sixteen inputs are processed on four threads. Later indices are given
//! less work, so they finish *earlier* than earlier indices: the errors at
//! indices 12, 9, and 3 complete in exactly that (descending) order. The
//! final result still reports index 3, the lowest failing input index,
//! rather than whichever error happened to finish first.
//!
//! Run with: cargo run --example try_collect_ordered

use rayon::prelude::*;
use std::collections::BTreeSet;
use std::sync::Mutex;
use std::time::{Duration, Instant};

fn main() {
    const N: i32 = 16;
    const ERRORS: [i32; 3] = [3, 9, 12];

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    let events = Mutex::new(Vec::new());
    let evaluated = Mutex::new(BTreeSet::new());
    let start = Instant::now();

    let result: Result<Vec<i32>, (usize, i32)> = pool.install(|| {
        (0..N).into_par_iter().try_collect_ordered(|x| {
            evaluated.lock().unwrap().insert(x);
            // Later indices do less work, so errors complete in the order
            // 12, then 9, then 3 -- the reverse of the input order.
            std::thread::sleep(Duration::from_millis((15 * (N - x)) as u64));
            let outcome = if ERRORS.contains(&x) {
                Err(x)
            } else {
                Ok(x * 10)
            };
            events
                .lock()
                .unwrap()
                .push((start.elapsed(), x, outcome.is_ok()));
            outcome
        })
    });

    let mut events = events.into_inner().unwrap();
    events.sort();
    println!("evaluated indices, in completion order:");
    for (t, x, ok) in events {
        let outcome = if ok { "Ok" } else { "Err" };
        println!("  t={t:>4.0?}  index {x:>2} -> {outcome}");
    }

    let evaluated = evaluated.into_inner().unwrap();
    let skipped: Vec<i32> = (0..N).filter(|x| !evaluated.contains(x)).collect();
    println!();
    println!("evaluated indices: {evaluated:?}");
    println!("never started (best-effort cancellation): {skipped:?}");

    match result {
        Ok(values) => println!("all inputs succeeded: {values:?}"),
        Err((index, error)) => {
            println!();
            println!("errors completed in the order 12, 9, 3, but the final choice is");
            println!("the lowest failing input index: Err(({index}, {error}))");
        }
    }
}
