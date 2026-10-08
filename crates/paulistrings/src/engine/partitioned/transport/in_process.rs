//! [`InProcessTransport`], the shared-memory transport between the partitions of one process.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::{AlreadyHere, ChunkMap, ChunkWait, Collectives, Payload, Transport};

/// Which transport call a published generation word belongs to; never zero, the kind of a never-written slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum CallKind {
    Barrier = 1,
    MaxU8 = 2,
    SumU64 = 3,
    Exchange = 4,
    SumF64 = 5,
}

impl CallKind {
    fn name(self) -> &'static str {
        match self {
            CallKind::Barrier => "barrier",
            CallKind::MaxU8 => "allreduce_max_u8",
            CallKind::SumU64 => "allreduce_sum_u64",
            CallKind::Exchange => "exchange",
            CallKind::SumF64 => "allreduce_sum_f64",
        }
    }

    /// [`Self::name`] of a raw published kind byte, which may be unknown or 0.
    fn name_of(raw: u8) -> &'static str {
        match raw {
            1 => "barrier",
            2 => "allreduce_max_u8",
            3 => "allreduce_sum_u64",
            4 => "exchange",
            5 => "allreduce_sum_f64",
            _ => "no call",
        }
    }
}

/// Spin-wait tier of a waiting rank, before it yields.
const SPINS_BEFORE_YIELD: u32 = 1_000;

/// Yield tier, before it sleeps; this tier also keeps a group larger than the core count from livelocking.
const YIELDS_BEFORE_SLEEP: u32 = 100;

/// Sleep tier step, which keeps a far-behind waiter off the cores its partner's workers run on.
const SLEEP_STEP: Duration = Duration::from_micros(50);

/// Spin iterations between checks of the departure mask and the deadline while in the spin tier.
const CHECKS_EVERY: u32 = 64;

/// Backstop for a partner that is neither arriving nor departed; a panicking partner is reported through the departure mask instead.
#[cfg(any(test, feature = "test-utils"))]
pub(super) const WAIT_TIMEOUT: Duration = Duration::from_secs(10);

/// One rank's contribution to one generation, double-buffered by generation parity ([`GroupState`]).
#[derive(Default)]
struct ValueSlot {
    /// `(generation << 16) | (kind << 8) | u8 payload`; generation 0 means never written.
    tag: AtomicU64,
    len: AtomicUsize,
    /// A sum contribution (`f64`s as bits); written only by the owner while no partner reads it, atomic elements so a hypothetical overlap is a stale read rather than UB.
    buf: UnsafeCell<Vec<AtomicU64>>,
}

/// SAFETY: `buf`'s exclusivity is established by the generation protocol documented on [`GroupState`], not by Rust's borrow checker.
unsafe impl Sync for ValueSlot {}

impl ValueSlot {
    /// Store `values` as this generation's contribution.
    ///
    /// # Safety
    ///
    /// The caller must be the rank that owns this slot, and no partner may be reading it — [`GroupState`]'s parity argument.
    unsafe fn write_buf(&self, values: &[u64]) {
        let buf = &mut *self.buf.get();
        if buf.len() != values.len() {
            buf.clear();
            buf.resize_with(values.len(), AtomicU64::default);
        }
        for (slot, &v) in buf.iter().zip(values) {
            slot.store(v, Ordering::Relaxed);
        }
        self.len.store(values.len(), Ordering::Relaxed);
    }

    /// This generation's contribution.
    ///
    /// # Safety
    ///
    /// The caller must have observed the owner's `Release` of the generation it is reading (so the elements are visible) and must be inside the window in which the owner cannot be writing — [`GroupState`]'s parity argument.
    unsafe fn read_buf(&self) -> &[AtomicU64] {
        let buf: &Vec<AtomicU64> = &*self.buf.get();
        &buf[..]
    }
}

/// One rank's publication point, padded to 128 bytes (the adjacent-line prefetch pair) so polling ranks never share a line.
#[repr(align(128))]
#[derive(Default)]
struct RankSlot {
    /// `(generation << 8) | kind` of the last published call, monotone, so a partner that ran past a generation is a panic, not a hang.
    progress: AtomicU64,
    values: [ValueSlot; 2],
}

/// The collective state one [`InProcessTransport`] group shares.
///
/// Each rank numbers its calls and publishes into its own slot with a `Release` store to `progress` that partners `Acquire`; that pairing is the only synchronization, so everything else in the slot may be loaded `Relaxed` once `progress` is observed.
/// `values` is double-buffered by generation parity: a rank's own wait for generation `g` before starting `g + 1` guarantees every partner has finished reading the slot it is about to overwrite.
struct GroupState {
    size: u32,
    timeout: Duration,
    /// Bit `q` set once rank `q`'s transport has been dropped; `P ≤ 64` fits a `u64`.
    departed: AtomicU64,
    slots: Box<[RankSlot]>,
}

impl GroupState {
    fn new(size: u32, timeout: Duration) -> Self {
        Self {
            size,
            timeout,
            departed: AtomicU64::new(0),
            slots: (0..size).map(|_| RankSlot::default()).collect(),
        }
    }

    fn value_slot(&self, rank: u32, gen: u64) -> &ValueSlot {
        &self.slots[rank as usize].values[(gen & 1) as usize]
    }

    /// Publish generation `gen` of `kind` carrying `payload`; the caller has already written the buffer, if its call carries one.
    fn publish(&self, rank: u32, gen: u64, kind: CallKind, payload: u8) {
        let k = kind as u64;
        self.value_slot(rank, gen).tag.store(
            (gen << 16) | (k << 8) | u64::from(payload),
            Ordering::Release,
        );
        self.slots[rank as usize]
            .progress
            .store((gen << 8) | k, Ordering::Release);
    }

    /// Wait for rank `src` to publish generation `gen` of `kind` and return its tag word; panics on a different call, a departed partner, or the timeout.
    fn wait(&self, waiter: u32, src: u32, gen: u64, kind: CallKind) -> u64 {
        let slot = &self.slots[src as usize];
        let value = &slot.values[(gen & 1) as usize];
        let mut spins: u32 = 0;
        let mut waiting_since: Option<Instant> = None;
        loop {
            let progress = slot.progress.load(Ordering::Acquire);
            if progress >> 8 >= gen {
                let tag = value.tag.load(Ordering::Acquire);
                if tag >> 16 == gen && (tag >> 8) & 0xff == kind as u64 {
                    return tag;
                }
                panic!(
                    "collective order mismatch: partition {waiter} is at transport call {gen} \
                     ({}) but partition {src} published {} at call {} — every partition must \
                     issue the identical sequence of transport calls per layer",
                    kind.name(),
                    CallKind::name_of(((tag >> 8) & 0xff) as u8),
                    tag >> 16,
                );
            }

            spins = spins.saturating_add(1);
            if spins >= SPINS_BEFORE_YIELD || spins.is_multiple_of(CHECKS_EVERY) {
                if self.departed.load(Ordering::Acquire) & (1u64 << src) != 0
                    // Re-read: a partner that published `gen` and then left is not dead.
                    && slot.progress.load(Ordering::Acquire) >> 8 < gen
                {
                    panic!(
                        "partition {src} terminated before completing the {} (it panicked)",
                        kind.name(),
                    );
                }
                let since = waiting_since.get_or_insert_with(Instant::now);
                if since.elapsed() > self.timeout {
                    panic!(
                        "partition {src} terminated before completing the {}: no response in \
                         {} s (this partition is at transport call {gen}, that one at {})",
                        kind.name(),
                        self.timeout.as_secs(),
                        slot.progress.load(Ordering::Acquire) >> 8,
                    );
                }
            }
            if spins < SPINS_BEFORE_YIELD {
                std::hint::spin_loop();
            } else if spins < SPINS_BEFORE_YIELD + YIELDS_BEFORE_SLEEP {
                std::thread::yield_now();
            } else {
                std::thread::sleep(SLEEP_STEP);
            }
        }
    }

    /// [`Self::wait`] on every partner, folding their tag words in rank order.
    fn wait_all(&self, rank: u32, gen: u64, kind: CallKind, mut fold: impl FnMut(u32, u64)) {
        for src in 0..self.size {
            if src != rank {
                fold(src, self.wait(rank, src, gen, kind));
            }
        }
    }
}

/// One exchange message: the `Option<P>` payload, plus the sender's generation in debug builds.
struct Message {
    #[cfg(debug_assertions)]
    seq: u64,
    body: Box<dyn std::any::Any + Send>,
}

impl Message {
    fn new(seq: u64, body: Box<dyn std::any::Any + Send>) -> Self {
        let _ = seq;
        Self {
            #[cfg(debug_assertions)]
            seq,
            body,
        }
    }
}

/// Take a received message's body as `T`; a mismatch means the partitions issued different calls, so it panics.
fn downcast<T: 'static>(body: Box<dyn std::any::Any + Send>, from: usize, op: &str) -> T {
    match body.downcast::<T>() {
        Ok(value) => *value,
        Err(_) => panic!(
            "partition {from} sent a different kind of message during the {op}: the partitions \
             issued different transport calls (every partition must issue the identical sequence \
             of transport calls per layer)",
        ),
    }
}

/// The transport between the partitions of one process: shared-memory collectives and a `P × P` matrix of unbounded `mpsc` channels for the exchange (ARCHITECTURE.md §Transport composition).
///
/// Built as a group by [`group`](Self::group), one endpoint per partition thread, and called by one thread per rank, since the generation counter is that thread's call index.
pub struct InProcessTransport {
    rank: u32,
    size: u32,
    /// Outlives every endpoint, so a partner can read a slot while its owner is leaving.
    state: Arc<GroupState>,
    /// This rank's transport-call counter, touched only by its driving thread.
    gen: AtomicU64,
    /// Sender to partition `q`, `None` in the self slot.
    outbox: Vec<Option<std::sync::mpsc::Sender<Message>>>,
    /// Receiver from partition `q`; the uncontended `Mutex` only makes the transport `Sync`.
    inbox: Vec<Option<std::sync::Mutex<std::sync::mpsc::Receiver<Message>>>>,
}

/// Marks this rank departed so waiting partners fail fast; unconditional, since a rank that returns early is as dead to its partners as one that panicked.
impl Drop for InProcessTransport {
    fn drop(&mut self) {
        self.state
            .departed
            .fetch_or(1u64 << self.rank, Ordering::Release);
    }
}

impl InProcessTransport {
    /// A group of `size` endpoints wired to each other, in rank order.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn group(size: u32) -> Vec<InProcessTransport> {
        Self::group_with_timeout(size, WAIT_TIMEOUT)
    }

    /// [`Self::group`] with the collective wait's backstop set to `timeout`.
    pub(crate) fn group_with_timeout(size: u32, timeout: Duration) -> Vec<InProcessTransport> {
        assert!(size > 0, "a transport group needs at least one partition");
        let n = size as usize;
        let mut senders: Vec<Vec<Option<std::sync::mpsc::Sender<Message>>>> =
            (0..n).map(|_| (0..n).map(|_| None).collect()).collect();
        let mut receivers: Vec<Vec<Option<std::sync::mpsc::Receiver<Message>>>> =
            (0..n).map(|_| (0..n).map(|_| None).collect()).collect();
        for src in 0..n {
            for dst in 0..n {
                if src != dst {
                    let (tx, rx) = std::sync::mpsc::channel();
                    senders[src][dst] = Some(tx);
                    receivers[src][dst] = Some(rx);
                }
            }
        }

        let state = Arc::new(GroupState::new(size, timeout));
        (0..n)
            .map(|rank| InProcessTransport {
                rank: rank as u32,
                size,
                state: Arc::clone(&state),
                gen: AtomicU64::new(0),
                outbox: std::mem::take(&mut senders[rank]),
                inbox: (0..n)
                    .map(|src| receivers[src][rank].take().map(std::sync::Mutex::new))
                    .collect(),
            })
            .collect()
    }

    /// The generation of the call starting now, never zero.
    fn next_gen(&self) -> u64 {
        self.gen.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Pretend one extra collective was issued, to test the order check.
    #[cfg(test)]
    pub(super) fn skip_sequence_for_test(&self) {
        self.gen.fetch_add(1, Ordering::Relaxed);
    }

    fn send_to(&self, dst: usize, seq: u64, body: Box<dyn std::any::Any + Send>, op: &str) {
        let tx = self.outbox[dst]
            .as_ref()
            .expect("a partition has no channel to itself");
        if tx.send(Message::new(seq, body)).is_err() {
            panic!("partition {dst} terminated before completing the {op} (it panicked)");
        }
    }

    /// Receive this call's message from `src`, checking the generation stamp in debug builds.
    fn recv_from(&self, src: usize, seq: u64, op: &str) -> Box<dyn std::any::Any + Send> {
        let rx = self.inbox[src]
            .as_ref()
            .expect("a partition has no channel to itself")
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match rx.recv() {
            Ok(message) => {
                #[cfg(debug_assertions)]
                assert_eq!(
                    message.seq, seq,
                    "collective order mismatch: partition {} is at transport call {seq} but \
                     partition {src} sent its call {} — every partition must issue the identical \
                     sequence of transport calls per layer",
                    self.rank, message.seq,
                );
                let _ = seq;
                message.body
            }
            Err(_) => {
                panic!("partition {src} terminated before completing the {op} (it panicked)")
            }
        }
    }
}

impl super::sealed::Sealed for InProcessTransport {}

impl Collectives for InProcessTransport {
    fn rank(&self) -> u32 {
        self.rank
    }

    fn size(&self) -> u32 {
        self.size
    }

    fn allreduce_max_u8(&self, v: u8) -> u8 {
        if self.size == 1 {
            return v;
        }
        let gen = self.next_gen();
        self.state.publish(self.rank, gen, CallKind::MaxU8, v);

        let mut acc = v;
        self.state
            .wait_all(self.rank, gen, CallKind::MaxU8, |_, tag| {
                acc = acc.max((tag & 0xff) as u8);
            });
        acc
    }

    fn allreduce_sum_u64(&self, buf: &mut [u64]) {
        if self.size == 1 {
            return;
        }
        let gen = self.next_gen();
        let slot = self.state.value_slot(self.rank, gen);
        // SAFETY: this rank owns the slot and the generation parity keeps every partner out of it (`GroupState`); the publish below makes the write visible.
        unsafe { slot.write_buf(buf) };
        self.state.publish(self.rank, gen, CallKind::SumU64, 0);

        let n = self.size;
        for src in 0..n {
            if src == self.rank {
                continue;
            }
            self.state.wait(self.rank, src, gen, CallKind::SumU64);
            let theirs = self.state.value_slot(src, gen);
            // SAFETY: `wait` returned, so this rank has observed `src`'s `Release` of this generation — its buffer and length are visible — and by the parity argument `src` cannot write the slot again before this rank publishes its next generation.
            let values = unsafe { theirs.read_buf() };
            assert_eq!(
                values.len(),
                buf.len(),
                "allreduce_sum_u64: partition {src} contributed {} values, this partition {}",
                values.len(),
                buf.len(),
            );
            for (a, v) in buf.iter_mut().zip(values) {
                *a = a.wrapping_add(v.load(Ordering::Relaxed));
            }
        }
    }

    /// Every partition folds all contributions, its own included, in rank order from zero, so all get the same bits.
    fn allreduce_sum_f64(&self, buf: &mut [f64]) {
        if self.size == 1 {
            return;
        }
        let gen = self.next_gen();
        let bits: Vec<u64> = buf.iter().map(|v| v.to_bits()).collect();
        let slot = self.state.value_slot(self.rank, gen);
        // SAFETY: as in `allreduce_sum_u64`.
        unsafe { slot.write_buf(&bits) };
        self.state.publish(self.rank, gen, CallKind::SumF64, 0);

        buf.fill(0.0);
        for src in 0..self.size {
            if src != self.rank {
                self.state.wait(self.rank, src, gen, CallKind::SumF64);
            }
            // SAFETY: as in `allreduce_sum_u64`; this rank's own slot is its own to read.
            let values = unsafe { self.state.value_slot(src, gen).read_buf() };
            assert_eq!(
                values.len(),
                buf.len(),
                "allreduce_sum_f64: partition {src} contributed {} values, this partition {}",
                values.len(),
                buf.len(),
            );
            for (a, v) in buf.iter_mut().zip(values) {
                *a += f64::from_bits(v.load(Ordering::Relaxed));
            }
        }
    }

    fn barrier(&self) {
        if self.size == 1 {
            return;
        }
        let gen = self.next_gen();
        self.state.publish(self.rank, gen, CallKind::Barrier, 0);
        self.state
            .wait_all(self.rank, gen, CallKind::Barrier, |_, _| {});
    }
}

impl Transport for InProcessTransport {
    /// Moves each payload to its partner, so the transfer is complete before `body` runs and `map` and `spare` go unused.
    fn exchange_layer<P, F, R>(
        &self,
        send: Vec<Option<P>>,
        _spare: &mut Vec<P>,
        _map: &ChunkMap,
        body: F,
    ) -> (Vec<Option<P>>, R)
    where
        P: Payload,
        F: FnOnce(&[Option<P>], &dyn ChunkWait) -> R,
    {
        let n = self.size as usize;
        assert_eq!(
            send.len(),
            n,
            "exchange: send has {} entries, expected one entry per partition ({n})",
            send.len(),
        );
        assert!(
            send[self.rank as usize].is_none(),
            "exchange: send[{}] is this partition's own slot and must be None",
            self.rank,
        );
        let recv: Vec<Option<P>> = if n == 1 {
            vec![None]
        } else {
            let seq = self.next_gen();
            // Published although it waits on the channels, so a partner in a collective at this generation sees the kind mismatch.
            self.state.publish(self.rank, seq, CallKind::Exchange, 0);
            for (dst, payload) in send.into_iter().enumerate() {
                if dst != self.rank as usize {
                    self.send_to(dst, seq, Box::new(payload), "exchange");
                }
            }
            (0..n)
                .map(|src| {
                    if src == self.rank as usize {
                        None
                    } else {
                        downcast::<Option<P>>(self.recv_from(src, seq, "exchange"), src, "exchange")
                    }
                })
                .collect()
        };

        let out = body(&recv, &AlreadyHere);
        (recv, out)
    }
}
