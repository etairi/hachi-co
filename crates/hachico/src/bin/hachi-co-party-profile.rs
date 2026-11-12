//! `hachi-co-party-profile <config.json>`: one party as `hachi-co-party` runs it, under a counting
//! allocator, for the memory profile of the parties (`docs/optimizations.md`, item 87). It is built
//! only with the feature `alloc-profile`, with line tables for readable frames:
//!
//! ```text
//! CARGO_PROFILE_RELEASE_DEBUG=line-tables-only cargo build --release -p hachi-co \
//!   --features alloc-profile --bin hachi-co-party-profile
//! HCO_PROFILE_MIN_MIB=16 hachi-co-party-profile party-1.json 2> party-1.profile
//! ```
//!
//! It prints the party's times on stdout, as `hachi-co-party` does, and its heap on stderr, in
//! lines that start with `hco-profile`: the peak of the live bytes and when it was reached, the
//! phases of the run, every allocation of at least `HCO_PROFILE_MIN_MIB` MiB (16 by default) that
//! was live at the peak, with its size, its lifetime and the frames of this workspace that made it,
//! the same allocations grouped by site over the whole run, and a timeline of the live bytes every
//! 100 ms. The allocator only counts and forwards to the system allocator, so the party's
//! messages and proof are those of `hachi-co-party`.
#![deny(missing_docs)]
// The global allocator is the only unsafe code of this binary, and the only reason it is not in
// `hachi-co-party` (which forbids unsafe code).
#![deny(unsafe_code)]

use hachi_co::harness::{self, PartyConfig};
use std::alloc::{GlobalAlloc, Layout, System};
use std::backtrace::Backtrace;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The bytes currently allocated.
static LIVE: AtomicUsize = AtomicUsize::new(0);
/// Their largest value so far, and when it was reached (nanoseconds since [`START`]).
static PEAK: AtomicUsize = AtomicUsize::new(0);
static PEAK_NS: AtomicU64 = AtomicU64::new(0);
/// The size from which an allocation is recorded with its frames (`HCO_PROFILE_MIN_MIB`).
static MIN_BIG: AtomicUsize = AtomicUsize::new(usize::MAX);
/// Cleared before the report: resolving the frames takes the backtrace lock and allocates and frees
/// large buffers, which must be neither recorded (recording captures a backtrace, which needs that
/// lock) nor looked up (the report holds the lock of [`BIGS`]).
static RECORDING: AtomicBool = AtomicBool::new(true);
static START: OnceLock<Instant> = OnceLock::new();

/// One allocation of at least [`MIN_BIG`] bytes.
struct Big {
    ptr: usize,
    size: usize,
    from_ns: u64,
    to_ns: Option<u64>,
    trace: Backtrace,
}

static BIGS: Mutex<Vec<Big>> = Mutex::new(Vec::new());

thread_local! {
    /// Set while the allocator records a large allocation, whose own allocations (the frames, the
    /// record) are counted but not recorded.
    static INSIDE: Cell<bool> = const { Cell::new(false) };
}

fn now_ns() -> u64 {
    START.get().map_or(0, |s| {
        u64::try_from(s.elapsed().as_nanos()).unwrap_or(u64::MAX)
    })
}

fn counted(size: usize) {
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    let mut peak = PEAK.load(Ordering::Relaxed);
    while live > peak {
        match PEAK.compare_exchange_weak(peak, live, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => {
                PEAK_NS.store(now_ns(), Ordering::Relaxed);
                break;
            }
            Err(p) => peak = p,
        }
    }
}

fn record(ptr: *mut u8, size: usize) {
    if ptr.is_null()
        || size < MIN_BIG.load(Ordering::Relaxed)
        || INSIDE.get()
        || !RECORDING.load(Ordering::Relaxed)
    {
        return;
    }
    INSIDE.set(true);
    let big = Big {
        ptr: ptr as usize,
        size,
        from_ns: now_ns(),
        to_ns: None,
        trace: Backtrace::force_capture(),
    };
    if let Ok(mut bigs) = BIGS.lock() {
        bigs.push(big);
    }
    INSIDE.set(false);
}

fn released(ptr: *mut u8, size: usize) {
    if size < MIN_BIG.load(Ordering::Relaxed) || INSIDE.get() || !RECORDING.load(Ordering::Relaxed)
    {
        return;
    }
    INSIDE.set(true);
    let at = now_ns();
    if let Ok(mut bigs) = BIGS.lock()
        && let Some(b) = bigs
            .iter_mut()
            .rev()
            .find(|b| b.ptr == ptr as usize && b.to_ns.is_none())
    {
        b.to_ns = Some(at);
    }
    INSIDE.set(false);
}

/// The system allocator, counting the live bytes and recording the large allocations.
struct Counting;

#[allow(unsafe_code)]
// SAFETY: every method forwards to `System` with the same arguments and returns its result; the
// bookkeeping around it allocates only through this allocator, guarded by `INSIDE`.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as for `System::alloc`, with the caller's layout.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            counted(layout.size());
            record(ptr, layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as for `System::alloc_zeroed`, with the caller's layout.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            counted(layout.size());
            record(ptr, layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        released(ptr, layout.size());
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: `ptr` was allocated by this allocator, hence by `System`, with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: as for `System::realloc`, with the caller's arguments.
        let out = unsafe { System.realloc(ptr, layout, new_size) };
        if !out.is_null() {
            released(ptr, layout.size());
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            counted(new_size);
            record(out, new_size);
        }
        out
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const GIB: f64 = (1u64 << 30) as f64;
const MIB: f64 = (1u64 << 20) as f64;

/// The frames of this workspace in a backtrace, innermost first: the source file under `crates/`
/// and line, and the function's name without its generic arguments. Inlined frames carry short
/// names (`prove`, `reserve`), so the file and line are what identify a site.
fn frames(trace: &Backtrace, count: usize) -> String {
    let text = trace.to_string();
    let mut out: Vec<String> = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let Some((_, func)) = line.trim().split_once(": ") else {
            continue;
        };
        let Some(at) = lines
            .peek()
            .and_then(|next| next.trim().strip_prefix("at "))
        else {
            continue;
        };
        let Some(i) = at.find("crates/") else {
            continue;
        };
        let at = &at[i + "crates/".len()..];
        if at.contains("hachi-co-party-profile") {
            continue;
        }
        let at = at.replacen("/src/", "/", 1);
        let at = at.rsplit_once(':').map_or(at.as_str(), |(loc, _col)| loc);
        let short = func.split('<').next().unwrap_or(func);
        let short = short.rsplit("::").next().unwrap_or(short);
        out.push(format!("{at} {short}"));
        if out.len() == count {
            break;
        }
    }
    if out.is_empty() {
        "(no frame of this workspace)".into()
    } else {
        out.join(" < ")
    }
}

fn report(config: &PartyConfig, r: &harness::PartyReport, timeline: &[(f64, usize)]) {
    let o = &r.output;
    println!(
        "party {} of {}, preset {}",
        config.party, config.n, config.preset
    );
    println!("setup      {:>10.2?}", r.setup_time);
    println!("connect    {:>10.2?}", r.connect_time);
    println!("commit     {:>10.2?}", o.commit_time);
    println!("eval       {:>10.2?}", o.eval_time);
    println!("proof      {} bytes", r.proof_bytes);
    let peak = PEAK.load(Ordering::Relaxed);
    let peak_s = PEAK_NS.load(Ordering::Relaxed) as f64 / 1e9;
    eprintln!(
        "hco-profile peak {:.2} GiB at {peak_s:.1} s, {:.2} GiB live at the end",
        peak as f64 / GIB,
        LIVE.load(Ordering::Relaxed) as f64 / GIB
    );
    let (s, c, m, e) = (
        r.setup_time.as_secs_f64(),
        r.connect_time.as_secs_f64(),
        o.commit_time.as_secs_f64(),
        o.eval_time.as_secs_f64(),
    );
    eprintln!(
        "hco-profile phases (s): setup 0 to {s:.1}, connect to {:.1}, commit to {:.1}, eval to {:.1}",
        s + c,
        s + c + m,
        s + c + m + e
    );
    let Ok(bigs) = BIGS.lock() else { return };
    let peak_ns = PEAK_NS.load(Ordering::Relaxed);
    let mut at_peak: Vec<&Big> = bigs
        .iter()
        .filter(|b| b.from_ns <= peak_ns && b.to_ns.is_none_or(|t| t > peak_ns))
        .collect();
    at_peak.sort_by_key(|b| std::cmp::Reverse(b.size));
    let total: usize = at_peak.iter().map(|b| b.size).sum();
    eprintln!(
        "hco-profile live at the peak: {} allocations of at least {:.0} MiB, {:.2} GiB of the {:.2} GiB",
        at_peak.len(),
        MIN_BIG.load(Ordering::Relaxed) as f64 / MIB,
        total as f64 / GIB,
        peak as f64 / GIB
    );
    for b in &at_peak {
        eprintln!(
            "hco-profile   {:9.1} MiB  {:7.1} s to {}  {}",
            b.size as f64 / MIB,
            b.from_ns as f64 / 1e9,
            b.to_ns
                .map_or("end".to_owned(), |t| format!("{:.1} s", t as f64 / 1e9)),
            frames(&b.trace, 4)
        );
    }
    // Every large allocation of the run, grouped by its innermost workspace frames.
    let mut sites: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    for b in bigs.iter() {
        let e = sites.entry(frames(&b.trace, 2)).or_default();
        e.0 += 1;
        e.1 += b.size;
        e.2 = e.2.max(b.size);
    }
    let mut sites: Vec<_> = sites.into_iter().collect();
    sites.sort_by_key(|(_, (_, total, _))| std::cmp::Reverse(*total));
    eprintln!("hco-profile sites over the run (count, total GiB, largest MiB):");
    for (site, (count, total, largest)) in sites {
        eprintln!(
            "hco-profile   {count:5} {:8.2} {:9.1}  {site}",
            total as f64 / GIB,
            largest as f64 / MIB
        );
    }
    let line: Vec<String> = timeline
        .iter()
        .map(|(t, live)| format!("{t:.1}:{:.3}", *live as f64 / GIB))
        .collect();
    eprintln!("hco-profile timeline (s:GiB) {}", line.join(" "));
}

fn main() -> ExitCode {
    START.get_or_init(Instant::now);
    let mib: usize = std::env::var("HCO_PROFILE_MIN_MIB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    MIN_BIG.store(mib << 20, Ordering::Relaxed);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [path] = args.as_slice() else {
        eprintln!("usage: hachi-co-party-profile <config.json>");
        return ExitCode::from(2);
    };
    let config = match PartyConfig::load(Path::new(path)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("hachi-co-party-profile: {e}");
            return ExitCode::from(2);
        }
    };
    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let sampler = {
        let stop = stop.clone();
        std::thread::spawn(move || {
            let mut timeline = Vec::new();
            while !stop.load(Ordering::Relaxed) {
                timeline.push((now_ns() as f64 / 1e9, LIVE.load(Ordering::Relaxed)));
                std::thread::sleep(Duration::from_millis(100));
            }
            timeline
        })
    };
    let result = harness::run_party(&config);
    stop.store(true, Ordering::Relaxed);
    let timeline = sampler.join().unwrap_or_default();
    RECORDING.store(false, Ordering::Relaxed);
    match result {
        Ok(r) => {
            report(&config, &r, &timeline);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("hachi-co-party-profile: {e}");
            ExitCode::FAILURE
        }
    }
}
