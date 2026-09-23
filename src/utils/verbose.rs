//! Terminal progress output (feature `verbose`, enabled by default): a progress bar that
//! redraws in place for the long loops of commit and prove, and a one-line tick for completed
//! steps. Purely cosmetic; nothing here affects the protocol.

use std::io::{Write, stdout};

/// Redraw a 100-cell progress bar on the current terminal line after step `cur` (0-based) of
/// `total`: prints a carriage return, `title: [`, then $p$ filled and $100 - p$ empty cells
/// with $p = \lfloor 100 (\mathrm{cur} + 1) / \mathrm{total} \rfloor$, and flushes `stdout`.
/// A newline is emitted when the bar reaches 100%, which happens exactly at the last step
/// `cur == total - 1`. Requires `total >= 1` and `cur < total`.
pub fn progress_bar(title: &str, cur: usize, total: usize) {
    let pc = ((cur + 1) * 100) / total;
    print!("\r{}: ", title);
    print!("[{:■<1$}", "", pc);
    print!("{:□<1$}]", "", 100-pc);

    if pc == 100 {
        print!("\n");
    }

    let _ = stdout().flush();
}

/// Print `title: ✓` on its own line, marking a completed step.
pub fn tick_item(title: &str) {
    println!("{}: ✓", title);
}
