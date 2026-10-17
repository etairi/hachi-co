//! Feature-gated utilities that are not part of the scheme: progress output on the terminal
//! (feature `verbose`, enabled by default) and generation of a random witness file (feature
//! `gen_file`).
//!
//! * `verbose::progress_bar` and `verbose::tick_item` are called from commit, prove and verify
//!   behind `#[cfg(feature = "verbose")]`; they print and nothing else.
//! * `gen_file::write_random_data` writes $2^\ell$ uniform residues in the file format read by
//!   [`crate::stream::file_stream::U64FileStream`]; `main` calls it before committing when the
//!   `gen_file` feature is on.

/// Terminal progress output (feature `verbose`).
#[cfg(feature = "verbose")]
pub mod verbose;

/// Random witness-file generation (feature `gen_file`).
#[cfg(feature = "gen_file")]
pub mod gen_file;
