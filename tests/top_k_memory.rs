//! Verifies that top_k memory usage scales with the number of work
//! segments times k, not with the total input length: a large input
//! with a small k must not be collected in full anywhere.

use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Relaxed) + layout.size();
            PEAK.fetch_max(now, Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT.fetch_sub(layout.size(), Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

/// Big enough that collecting the whole iterator would be obvious.
#[derive(Clone)]
struct Large {
    key: u64,
    payload: [u8; 64],
}

#[test]
fn top_k_memory_is_bounded_by_k_not_len() {
    const N: usize = 1_000_000;
    const K: usize = 8;

    let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();

    // Warm up the pool (worker threads, stealers, ...) so their one-time
    // allocations do not count towards the measurement.
    pool.install(|| {
        (0..1024usize)
            .into_par_iter()
            .map(|i| Large {
                key: i as u64,
                payload: [0; 64],
            })
            .top_k(K, |a, b| a.key.cmp(&b.key))
    });

    let baseline = CURRENT.load(Relaxed);
    PEAK.store(baseline, Relaxed);

    let top = pool.install(|| {
        (0..N)
            .into_par_iter()
            .map(|i| Large {
                key: (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32,
                payload: [i as u8; 64],
            })
            .top_k(K, |a, b| a.key.cmp(&b.key))
    });

    assert_eq!(top.len(), K);
    assert!(top.windows(2).all(|w| w[0].key <= w[1].key));
    std::hint::black_box(top.iter().map(|l| l.payload[0] as u64).sum::<u64>());

    let peak_extra = PEAK.load(Relaxed) - baseline;
    let full_collect = N * size_of::<(usize, Large)>();
    assert!(
        peak_extra < full_collect / 10,
        "peak extra memory {peak_extra} bytes should be far below the \
         {full_collect} bytes needed to collect the whole input"
    );
    // Absolute sanity bound as well: a handful of candidate sets of 8
    // elements each needs nowhere near 8 MiB.
    assert!(
        peak_extra < 8 * 1024 * 1024,
        "peak extra memory {peak_extra} bytes exceeds 8 MiB"
    );
}
