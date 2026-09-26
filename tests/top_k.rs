use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Record {
    score: u32,
    id: usize,
}

/// Records with lots of duplicate scores, so stability is observable.
fn sample_records() -> Vec<Record> {
    (0..1000)
        .map(|i| Record {
            score: (i * 37 % 10) as u32,
            id: i,
        })
        .collect()
}

#[test]
fn top_k_matches_stable_sort_across_thread_counts() {
    let data = sample_records();

    let mut expected = data.clone();
    expected.sort_by_key(|r| r.score); // stable
    expected.truncate(3);

    let mut previous = None;
    for threads in [1, 2, 4] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let top = pool.install(|| {
            data.par_iter()
                .cloned()
                .top_k(3, |a, b| a.score.cmp(&b.score))
        });
        assert_eq!(top, expected, "threads = {threads}");
        if let Some(previous) = previous {
            assert_eq!(top, previous, "result differs between thread counts");
        }
        previous = Some(top);
    }
}

#[test]
fn top_k_zero_does_not_run_upstream() {
    let mapped = AtomicUsize::new(0);
    let top = (0..1000i32)
        .into_par_iter()
        .map(|x| {
            mapped.fetch_add(1, Relaxed);
            x
        })
        .top_k(0, |a, b| a.cmp(b));
    assert!(top.is_empty());
    assert_eq!(
        mapped.load(Relaxed),
        0,
        "upstream map must not run for k = 0"
    );
}

#[test]
fn top_k_empty_input() {
    let top = Vec::<i32>::new().into_par_iter().top_k(5, |a, b| a.cmp(b));
    assert!(top.is_empty());
}

#[test]
fn top_k_greater_than_len_returns_everything_ranked() {
    let data = [3, 1, 4, 1, 5, 9, 2, 6];
    let top = data.into_par_iter().top_k(100, |a, b| a.cmp(b));
    let mut expected = data.to_vec();
    expected.sort();
    assert_eq!(top, expected);
}

#[test]
fn top_k_reverse_comparator_keeps_greatest() {
    let data = [3, 1, 4, 1, 5, 9, 2, 6];
    let top = data.into_par_iter().top_k(3, |a, b| b.cmp(a));
    assert_eq!(top, [9, 6, 5]);
}

static CREATED: AtomicUsize = AtomicUsize::new(0);
static DROPPED: AtomicUsize = AtomicUsize::new(0);

/// An element that counts how many times it was created and dropped, so
/// leaks (DROPPED < CREATED) and double drops (DROPPED > CREATED) show up
/// as a mismatch.
struct Counted {
    key: i32,
}

impl Counted {
    fn new(key: i32) -> Self {
        CREATED.fetch_add(1, Relaxed);
        Counted { key }
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        DROPPED.fetch_add(1, Relaxed);
    }
}

static SILENCE_PANIC: AtomicBool = AtomicBool::new(false);

fn silence_expected_panics() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if !SILENCE_PANIC.load(Relaxed) {
            prev(info);
        }
    }));
}

#[test]
fn top_k_panic_drops_held_elements_exactly_once() {
    silence_expected_panics();

    for threads in [1, 2, 4] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();

        // Panic inside the comparison function.
        CREATED.store(0, Relaxed);
        DROPPED.store(0, Relaxed);
        let data: Vec<Counted> = (0..1000).map(|i| Counted::new(i % 7)).collect();
        SILENCE_PANIC.store(true, Relaxed);
        let result = catch_unwind(AssertUnwindSafe(|| {
            pool.install(|| {
                let countdown = AtomicUsize::new(50);
                data.into_par_iter().top_k(10, move |a, b| {
                    if countdown.fetch_sub(1, Relaxed) == 1 {
                        panic!("compare boom");
                    }
                    a.key.cmp(&b.key)
                })
            })
        }));
        SILENCE_PANIC.store(false, Relaxed);
        assert!(result.is_err(), "threads = {threads}");
        assert_eq!(
            CREATED.load(Relaxed),
            DROPPED.load(Relaxed),
            "compare panic with threads = {threads}"
        );

        // Panic inside the upstream map.
        CREATED.store(0, Relaxed);
        DROPPED.store(0, Relaxed);
        SILENCE_PANIC.store(true, Relaxed);
        let result = catch_unwind(AssertUnwindSafe(|| {
            pool.install(|| {
                let countdown = AtomicUsize::new(500);
                (0..2000i32)
                    .into_par_iter()
                    .map(move |i| {
                        if countdown.fetch_sub(1, Relaxed) == 1 {
                            panic!("map boom");
                        }
                        Counted::new(i)
                    })
                    .top_k(8, |a, b| a.key.cmp(&b.key))
            })
        }));
        SILENCE_PANIC.store(false, Relaxed);
        assert!(result.is_err(), "threads = {threads}");
        assert_eq!(
            CREATED.load(Relaxed),
            DROPPED.load(Relaxed),
            "map panic with threads = {threads}"
        );
    }
}
