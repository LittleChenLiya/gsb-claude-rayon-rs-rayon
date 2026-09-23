use rayon::prelude::*;
use std::sync::Mutex;
use std::time::Duration;

/// Demonstrates `try_collect`: indices 2 and 7 both fail, and the sleeps
/// force index 7 to record its error *before* index 2. The final result must
/// still pick index 2, the smallest failing input position.
fn main() {
    let executed = Mutex::new(Vec::new());

    let result: Result<Vec<i32>, (usize, String)> = (0..12)
        .into_par_iter()
        .with_max_len(1)
        .map(|i| {
            executed.lock().unwrap().push(i);
            println!("evaluating index {i}");
            if i == 2 {
                // Slow failure: completes after index 7's error.
                std::thread::sleep(Duration::from_millis(300));
                Err(format!("error at {i}"))
            } else if i == 7 {
                // Fast failure: recorded first, but not the smallest index.
                std::thread::sleep(Duration::from_millis(50));
                Err(format!("error at {i}"))
            } else {
                Ok(i)
            }
        })
        .try_collect();

    println!("executed indices: {:?}", *executed.lock().unwrap());
    match result {
        Ok(vec) => println!("all succeeded: {vec:?}"),
        Err((index, error)) => {
            println!("selected smallest failing index {index}: {error}");
        }
    }
}
