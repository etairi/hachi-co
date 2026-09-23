//! Streaming access to the witness. The committed polynomial has $2^\ell$ coefficients in
//! $\mathbb Z_q$, and so that very large polynomials need not be held in memory, commit and
//! prove read the coefficient vector sequentially, chunk by chunk, through the [`Stream`] trait
//! and rewind it with [`Stream::reset`] before each pass: the commitment reads the witness once
//! (it expects a stream positioned at the start), and the prover reads it twice, once to form
//! $Y$ and $\mathbf{w}$ and once to form the response $\mathbf{z}$.
//!
//! The stream yields the coefficients in index order $p = 0, 1, \dots, 2^\ell - 1$; each block
//! of $d$ consecutive coefficients is one element of $\mathbf R_q$ (coefficient $j$ of the
//! element at position $j$ of the block) and $2^m$ consecutive elements form one chunk
//! $\mathbf f_i$, so reading $2^m d$ values into a `PVec` of $2^m$ elements fills coefficient
//! $j$ of element $i$ at index $i d + j$ (see [`crate::hachi::commit`]). The only
//! implementation is [`file_stream::U64FileStream`].

pub mod file_stream;

/// A sequential, rewindable source of elements of type `T`.
pub trait Stream<T> {
    /// Total number of elements the source holds (not the number still unread).
    fn length(&self) -> usize;

    /// Fill `arr` with the next `arr.len()` elements, in order, and advance past them.
    /// Implementations may panic if fewer elements remain.
    fn read(&mut self, arr: &mut [T]);

    /// Rewind to the first element, so that the next [`read`](Stream::read) starts over.
    fn reset(&mut self);
}
