//! [`Collectives`], the group operations a partition's truncation and layer loop issue besides the exchange (ARCHITECTURE.md §Partitioning).

/// Seals [`Collectives`] and, through it, [`Transport`](crate::Transport).
pub(crate) mod sealed {
    pub trait Sealed {}
}

/// The group operations a partition needs besides the exchange: its rank, the reductions, and a barrier.
///
/// Every partition must issue the identical sequence of collective and transport calls, and every reduction must return the identical value on every partition.
/// Sealed: implemented only by the in-process, solo and `mpi::MpiTransport` transports.
pub trait Collectives: sealed::Sealed + Send + Sync {
    /// This partition's index in the group, `0 <= rank < size`.
    fn rank(&self) -> u32;
    /// Number of partitions in the group.
    fn size(&self) -> u32;
    /// Maximum of `v` over the group.
    fn allreduce_max_u8(&self, v: u8) -> u8;
    /// Element-wise wrapping sum of `buffer` over the group, in place; every partition passes the same length.
    fn allreduce_sum_u64(&self, buffer: &mut [u64]);
    /// Element-wise sum of `buffer` over the group, in place, bitwise identical on every partition.
    ///
    /// The combination order is the implementation's, so the result may differ by rounding from a serial sum; a slot only one partition fills is exact.
    fn allreduce_sum_f64(&self, buffer: &mut [f64]);
    /// Block until every partition has arrived.
    fn barrier(&self);

    /// Panic unless every partition passed the same `fingerprint`; one collective.
    fn check_consistency(&self, fingerprint: u64) {
        // Per bit, how many partitions set it: an agreeing group answers 0 or `size` for every bit.
        let mut counts = [0u64; 64];
        for (i, count) in counts.iter_mut().enumerate() {
            *count = (fingerprint >> i) & 1;
        }
        self.allreduce_sum_u64(&mut counts);
        let size = u64::from(self.size());
        for (bit, &count) in counts.iter().enumerate() {
            assert!(
                count == 0 || count == size,
                "the partitions disagree about the run: partition {} offered fingerprint \
                 {fingerprint:#018x}, and {count} of {size} partitions set bit {bit} of theirs. \
                 Every partition must be driven through the same circuit, in the same direction, \
                 under the same policy and options — a group that is not stays in step only by \
                 luck.",
                self.rank(),
            );
        }
    }
}
