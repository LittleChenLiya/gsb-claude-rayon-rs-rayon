use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::ops::Add;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Sequentially evaluate the exact same midpoint-split binary tree.
fn sequential_tree(values: &[f64]) -> f64 {
    fn reduce(values: &[f64]) -> f64 {
        match values {
            [] => 0.0,
            [single] => *single,
            _ => {
                let mid = values.len() / 2;
                reduce(&values[..mid]) + reduce(&values[mid..])
            }
        }
    }
    reduce(values)
}

fn make_input() -> Vec<f64> {
    let mut values = Vec::new();
    // Mix magnitudes that lose precision when folded in another order.
    for i in 0..4096usize {
        values.push(1.0e300 * ((i % 2) as f64 * 2.0 - 1.0));
        values.push(1.0e-300);
        values.push(f64::from((i % 7) as i32 - 3));
        values.push(f64::from_bits(1)); // smallest subnormal
    }
    values
}

fn bits_on_threads(threads: usize, values: &[f64]) -> u64 {
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| values.par_iter().copied().tree_reduce(|| 0.0, f64::add))
        .to_bits()
}

fn composed_bits_on_threads(threads: usize, values: &[f64]) -> u64 {
    let (left, right) = values.split_at(values.len() / 2 + 1);
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        left.par_iter()
            .chain(right)
            .enumerate()
            .map(|(_, &v)| v)
            .tree_reduce_with(Add::add)
            .unwrap()
    })
    .to_bits()
}

#[test]
fn deterministic_float_bits_regardless_of_threads() {
    let values = make_input();
    let bits1 = bits_on_threads(1, &values);
    let bits2 = bits_on_threads(2, &values);
    let bits4 = bits_on_threads(4, &values);

    assert_eq!(bits1, bits2, "1 and 2 threads must agree on bit pattern");
    assert_eq!(bits2, bits4, "2 and 4 threads must agree on bit pattern");

    let expected = sequential_tree(&values).to_bits();
    assert_eq!(bits4, expected, "must match the fixed midpoint tree");

    let composed1 = composed_bits_on_threads(1, &values);
    let composed2 = composed_bits_on_threads(2, &values);
    let composed4 = composed_bits_on_threads(4, &values);
    assert_eq!(composed1, expected);
    assert_eq!(composed2, expected);
    assert_eq!(composed4, expected);
}

#[test]
fn empty_iterator_uses_identity() {
    let values: Vec<f64> = Vec::new();
    let result = values.par_iter().copied().tree_reduce(|| 42.0, f64::add);
    assert_eq!(result, 42.0);
    assert_eq!([].par_iter().copied().tree_reduce_with(f64::add), None);
}

struct Recorder<'a> {
    merges: &'a Mutex<Vec<String>>,
}

impl Recorder<'_> {
    fn merge(&self, left: String, right: String) -> String {
        let combined = format!("({left},{right})");
        self.merges.lock().unwrap().push(combined.clone());
        combined
    }
}

fn fixed_tree(values: &[usize]) -> (Option<String>, Vec<String>) {
    let merges = Mutex::new(Vec::new());
    let result = values
        .par_iter()
        .map(|&i| i.to_string())
        .tree_reduce_with(|left, right| Recorder { merges: &merges }.merge(left, right));
    let mut recorded = merges.into_inner().unwrap();
    recorded.sort();
    (result, recorded)
}

#[test]
fn single_element_has_no_merges() {
    let (result, merges) = fixed_tree(&[0]);
    assert_eq!(result.as_deref(), Some("0"));
    assert!(merges.is_empty());
}

#[test]
fn odd_length_fixed_tree() {
    // len 3: split at 1 => (0,(1,2))
    let (result, mut merges) = fixed_tree(&[0, 1, 2]);
    assert_eq!(result.as_deref(), Some("(0,(1,2))"));
    merges.sort();
    assert_eq!(merges, vec!["(0,(1,2))".to_owned(), "(1,2)".to_owned()]);
}

#[test]
fn seven_elements_fixed_tree() {
    // len 7: split at 3 => ((0,(1,2)),((3,4),(5,6)))
    let (result, mut merges) = fixed_tree(&[0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(result.as_deref(), Some("((0,(1,2)),((3,4),(5,6)))"));
    let mut expected = vec![
        "(1,2)".to_owned(),
        "(0,(1,2))".to_owned(),
        "(3,4)".to_owned(),
        "(5,6)".to_owned(),
        "((3,4),(5,6))".to_owned(),
        "((0,(1,2)),((3,4),(5,6)))".to_owned(),
    ];
    expected.sort();
    merges.sort();
    assert_eq!(merges, expected);
}

#[test]
fn panic_unwinds_and_all_started_tasks_finish() {
    let started = AtomicUsize::new(0);
    let finished = AtomicUsize::new(0);

    let values: Vec<usize> = (0..512).collect();
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        values
            .par_iter()
            .map(|&i| {
                started.fetch_add(1, Ordering::SeqCst);
                if i == 400 {
                    panic!("boom at {i}");
                }
                finished.fetch_add(1, Ordering::SeqCst);
                i as f64
            })
            .tree_reduce(|| 0.0, f64::add)
    }));

    assert!(result.is_err(), "the panic must propagate");

    // After the unwound call, the pool must still service new work, which
    // means no worker task was left dangling.
    let pool_check: f64 = (0..10_000usize)
        .into_par_iter()
        .map(|i| i as f64)
        .tree_reduce(|| 0.0, f64::add);
    assert_eq!(pool_check, (0..10_000usize).sum::<usize>() as f64);

    // Every leaf closure that started also finished (apart from the panicking
    // one, which by definition unwinds through its own body).
    assert_eq!(
        started.load(Ordering::SeqCst) - finished.load(Ordering::SeqCst),
        1
    );
}

#[test]
fn panic_in_reduce_op_lets_started_tasks_finish() {
    let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let leaves_started = AtomicUsize::new(0);
    let leaves_finished = AtomicUsize::new(0);

    let values: Vec<usize> = (0..256).collect();
    let result = pool.install(|| {
        panic::catch_unwind(AssertUnwindSafe(|| {
            values
                .par_iter()
                .map(|&i| {
                    leaves_started.fetch_add(1, Ordering::SeqCst);
                    let v = i as f64;
                    leaves_finished.fetch_add(1, Ordering::SeqCst);
                    v
                })
                .tree_reduce_with(|left, right| {
                    if left > 200.0 {
                        panic!("boom while reducing");
                    }
                    left + right
                })
        }))
    });

    assert!(result.is_err(), "the panic must propagate");
    assert_eq!(
        leaves_started.load(Ordering::SeqCst),
        leaves_finished.load(Ordering::SeqCst),
        "every leaf task that started must finish",
    );

    // The dedicated pool is still healthy afterwards.
    let after = pool.install(|| {
        (0..1000usize)
            .into_par_iter()
            .map(|i| i * 2)
            .tree_reduce_with(usize::wrapping_add)
            .unwrap()
    });
    assert_eq!(after, (0..1000usize).map(|i| i * 2).sum::<usize>());
}
