use crate::resolution::Resolution;
use std::time::Duration;

/// Number of untimed iterations before measurement begins.
pub const WARMUP: usize = 10;

/// Number of timed iterations.
pub const ITERATIONS: usize = 25;

/// Prints the average duration of a timed iteration.
pub fn report(backend_name: &str, res: &Resolution, total: Duration) {
    println!(
        "[{backend_name}/{}] Average: {}ms",
        res.name,
        total.as_millis() / ITERATIONS as u128
    );
}
