use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Once;
use std::sync::atomic::{AtomicUsize, Ordering};

// ---------------------------------------------------------------------
// Tracking allocator for the memory-bound test
// ---------------------------------------------------------------------

struct TrackingAllocator;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

fn pool(threads: usize) -> rayon::ThreadPool {
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Record {
    id: usize,
    score: u32,
}

/// Deterministic records with many duplicate scores.
fn records(n: usize) -> Vec<Record> {
    (0..n)
        .map(|id| Record {
            id,
            score: (id as u32).wrapping_mul(2_654_435_761) % 50,
        })
        .collect()
}

fn by_score_asc(a: &Record, b: &Record) -> std::cmp::Ordering {
    a.score.cmp(&b.score)
}

// ---------------------------------------------------------------------
// Stability and cross-thread determinism
// ---------------------------------------------------------------------

#[test]
fn stable_top_k_matches_sequential_sort_for_1_2_4_threads() {
    let input = records(10_000);
    let mut reference = input.clone();
    reference.sort_by(by_score_asc); // Vec::sort_by is stable
    reference.truncate(3);

    for threads in [1, 2, 4] {
        let input = input.clone();
        let top = pool(threads).install(|| input.par_iter().cloned().top_k_by(3, by_score_asc));
        assert_eq!(top, reference, "mismatch with {threads} thread(s)");
    }
}

#[test]
fn results_identical_across_thread_counts() {
    let input = records(50_000);
    let mut results = Vec::new();
    for threads in [1, 2, 4] {
        let input = input.clone();
        results.push(pool(threads).install(|| {
            input
                .par_iter()
                .cloned()
                .top_k_by(10, |a, b| b.score.cmp(&a.score))
        }));
    }
    assert_eq!(results[0], results[1]);
    assert_eq!(results[1], results[2]);
}

// ---------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------

#[test]
fn k_zero_does_not_execute_upstream_map() {
    let calls = AtomicUsize::new(0);
    let out = pool(4).install(|| {
        (0..10_000)
            .into_par_iter()
            .map(|x| {
                calls.fetch_add(1, Ordering::SeqCst);
                x
            })
            .top_k_by(0, |a, b| a.cmp(b))
    });
    assert!(out.is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn empty_input_returns_empty() {
    let v: Vec<i32> = Vec::new();
    let out = v.par_iter().top_k(5);
    assert!(out.is_empty());
}

#[test]
fn k_greater_than_len_returns_everything_sorted() {
    let v: Vec<u32> = (0..100).map(|x| (x * 37) % 100).collect();
    let mut expected = v.clone();
    expected.sort();
    let out = v.par_iter().copied().top_k(1_000);
    assert_eq!(out, expected);
}

#[test]
fn reverse_comparator_selects_largest() {
    let v = vec![5, 1, 4, 2, 3, 9, 7];
    let top = v.par_iter().copied().top_k_by(3, |a, b| b.cmp(a));
    assert_eq!(top, [9, 7, 5]);
    let top = v.par_iter().copied().top_k(2);
    assert_eq!(top, [1, 2]);
}

// ---------------------------------------------------------------------
// Panic safety: no leaks, no double drops
// ---------------------------------------------------------------------

#[derive(Default)]
struct DropCounter {
    created: AtomicUsize,
    dropped: AtomicUsize,
}

struct Guard<'a> {
    counter: &'a DropCounter,
    value: u64,
}

impl DropCounter {
    fn make(&self, value: u64) -> Guard<'_> {
        self.created.fetch_add(1, Ordering::SeqCst);
        Guard {
            counter: self,
            value,
        }
    }
    fn assert_balanced(&self) {
        assert_eq!(
            self.created.load(Ordering::SeqCst),
            self.dropped.load(Ordering::SeqCst),
            "leaked or double-dropped elements"
        );
    }
}

impl Drop for Guard<'_> {
    fn drop(&mut self) {
        self.counter.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

static HOOK: Once = Once::new();

/// Keep the test output readable; the panics below are expected.
fn silence_panics() {
    HOOK.call_once(|| std::panic::set_hook(Box::new(|_| {})));
}

#[test]
fn panic_in_comparator_recovers_all_elements() {
    silence_panics();
    let counter = DropCounter::default();
    let result = {
        let v: Vec<_> = (0..1_000).map(|i| counter.make(i)).collect();
        catch_unwind(AssertUnwindSafe(|| {
            pool(4).install(|| {
                v.into_par_iter().top_k_by(4, |a, b| {
                    if a.value == 500 {
                        panic!("boom in comparator");
                    }
                    a.value.cmp(&b.value)
                })
            })
        }))
    };
    assert!(result.is_err());
    counter.assert_balanced();
}

#[test]
fn panic_in_upstream_map_recovers_all_elements() {
    silence_panics();
    let counter = DropCounter::default();
    let result = catch_unwind(AssertUnwindSafe(|| {
        pool(4).install(|| {
            (0..1_000usize)
                .into_par_iter()
                .map(|i| {
                    if i == 500 {
                        panic!("boom in map");
                    }
                    counter.make(i as u64)
                })
                .top_k_by(4, |a, b| a.value.cmp(&b.value))
        })
    }));
    assert!(result.is_err());
    counter.assert_balanced();
}

// ---------------------------------------------------------------------
// Memory: large input, small k
// ---------------------------------------------------------------------

#[test]
fn memory_grows_with_k_not_with_input_len() {
    let n = 2_000_000usize;
    let k = 16;
    let pool = pool(4);
    let input: Vec<u64> = (0..n as u64)
        .map(|x| x.wrapping_mul(0x9E37_79B9_7F4A_7C15).reverse_bits())
        .collect();
    let input_bytes = n * std::mem::size_of::<u64>();

    let baseline = CURRENT.load(Ordering::SeqCst);
    PEAK.store(baseline, Ordering::SeqCst);

    let top = pool.install(|| input.par_iter().copied().top_k_by(k, |a, b| b.cmp(a)));

    let extra = PEAK.load(Ordering::SeqCst) - baseline;
    assert_eq!(top.len(), k);
    assert!(
        extra < input_bytes / 8,
        "peak extra memory {extra} bytes should be far below the {input_bytes} byte input"
    );
}
