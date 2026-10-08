//! [`InProcessTransport`], the shared-memory transport between the partitions of one process.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::{AlreadyHere, ChunkMap, ChunkWait, Collectives, Payload, Transport};

// ---- the shared-memory collective state ------------------------------------

/// Which transport call a published generation word belongs to.
///
/// Stamped into every word a rank publishes, so a rank waiting for generation
/// `g` can tell "my partner has not arrived yet" from "my partner issued a
/// *different* call at `g`" — the collective-order invariant (module docs).
/// Never zero: a slot that was never written reads as generation 0, kind 0.
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
    /// The name used in a mismatch panic. Also the `op` string a partner-death
    /// panic names, so the two messages agree on what a call is called.
    fn name(self) -> &'static str {
        match self {
            CallKind::Barrier => "barrier",
            CallKind::MaxU8 => "allreduce_max_u8",
            CallKind::SumU64 => "allreduce_sum_u64",
            CallKind::Exchange => "exchange",
            CallKind::SumF64 => "allreduce_sum_f64",
        }
    }

    /// The name of a raw kind byte off a published word, which may be a kind
    /// this build does not know (or the 0 of a never-written slot).
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

/// `spin_loop` hints a waiting rank issues before it starts yielding instead.
///
/// The mechanism is a few hundred nanoseconds between dedicated threads, so the fast path — the partner is already here, or arrives within a couple of microseconds — never leaves this tier.
const SPINS_BEFORE_YIELD: u32 = 1_000;

/// `yield_now` calls after the spin tier before the waiter starts sleeping.
///
/// Covers the ordinary case the spins do not: the partner is a few tens of microseconds behind because its share of the layer was bigger.
/// A `P = 16` group on a smaller box (the test suite runs one) also makes progress here rather than livelocking.
const YIELDS_BEFORE_SLEEP: u32 = 100;

/// How long a waiter sleeps per iteration once even yielding has not helped.
///
/// `sched_yield` in a loop is not free to the rest of the machine: it re-enters the run queue and takes its fair share of the CPU, which on a partitioned run is a share of the CPUs the partner's *own* workers are trying to finish the layer on.
/// Past a wait this long the partner is not close, so paying up to one step of extra latency to stay off its cores is the right trade.
const SLEEP_STEP: Duration = Duration::from_micros(50);

/// Spin iterations between two checks of the departure mask and the deadline while still in the spin tier.
/// Past it, both are checked every iteration — a yield or a sleep dwarfs two loads.
///
/// Both live on lines nobody writes in steady state, but keeping them out of the tight loop leaves the fast path a single load.
const CHECKS_EVERY: u32 = 64;

/// How long a rank waits for a partner before declaring it dead.
///
/// The backstop, not the mechanism: a partner that panics drops its transport and is reported within `CHECKS_EVERY` spins ([`InProcessTransport::drop`]).
/// This bound only catches a partner that is neither dead nor arriving — a deadlock elsewhere in the process — so it is generous.
#[cfg(any(test, feature = "test-utils"))]
pub(super) const WAIT_TIMEOUT: Duration = Duration::from_secs(10);

/// One rank's contribution to one generation, double-buffered by generation parity (see [`GroupState`] for why two are enough).
#[derive(Default)]
struct ValueSlot {
    /// `(generation << 16) | (kind << 8) | u8 payload`, published last-but-one with `Release`.
    /// Generation 0 means "never written".
    tag: AtomicU64,
    /// Elements of `buf` that belong to this generation.
    len: AtomicUsize,
    /// The `allreduce_sum_u64` contribution, or an `allreduce_sum_f64` one as `f64` bits.
    ///
    /// Written only by the owning rank and only while no partner can be reading it, which is what the parity split buys (see [`GroupState`]); the elements are atomics so that a *hypothetical* overlap is a stale read rather than undefined behaviour, and the `UnsafeCell` is there for the resize, which needs `&mut`.
    /// Allocated on first use and reused at the same length ever after.
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

/// One rank's publication point: everything a partner reads to learn where that rank is and what it contributed.
///
/// Padded to 128 bytes — two x86 cache lines, the granularity the hardware prefetcher pairs — so `P` ranks polling each other never share a line.
#[repr(align(128))]
#[derive(Default)]
struct RankSlot {
    /// `(generation << 8) | kind` of the last call this rank published: **monotone**, overwritten every call, both parities.
    ///
    /// It is what makes a desynchronized partner a panic instead of a hang: a rank waiting for generation `g` sees a partner that ran *past* `g` here, even though the partner's `values` slot for `g`'s parity never got `g`'s tag.
    progress: AtomicU64,
    /// The value published at each generation parity.
    values: [ValueSlot; 2],
}

/// The collective state one [`InProcessTransport`] group shares.
///
/// Each rank numbers its own calls (generation 1, 2, 3, …) and publishes `(generation, kind, payload)` into its own slot with a `Release` store to `progress`, which every partner spins on with `Acquire`; that pairing is the only synchronization, so the tag, `len` and buffer elements may be loaded `Relaxed` once `progress` is observed.
/// `values` is double-buffered by generation parity: a rank only overwrites `values[p]` once every partner has observed its previous use, which a rank's own wait for generation `g` before starting `g + 1` guarantees.
struct GroupState {
    /// Ranks in the group.
    size: u32,
    /// The backstop wait for a partner's publication.
    timeout: Duration,
    /// Bit `q` set once rank `q`'s transport has been dropped — it will publish nothing further.
    /// `P ≤ 64` (`P_MAX_BITS`), so a `u64` mask is ample.
    departed: AtomicU64,
    /// One publication point per rank, in rank order.
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

    /// The slot rank `rank` publishes generation `gen` into.
    fn value_slot(&self, rank: u32, gen: u64) -> &ValueSlot {
        &self.slots[rank as usize].values[(gen & 1) as usize]
    }

    /// Publish generation `gen` of kind `kind` carrying `payload`, storing `tag` then `progress` (see [`GroupState`]); the caller has already written the buffer, if its call carries one.
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

    /// Wait for rank `src` to publish generation `gen` of kind `kind`, and
    /// return its tag word (whose low byte is the `u8` payload).
    ///
    /// `waiter` is only used to name this rank in a mismatch panic.
    ///
    /// # Panics
    ///
    /// If `src` published a different call at `gen`, or ran past `gen` without
    /// publishing it (the collective-order invariant); if `src`'s transport
    /// has been dropped without publishing `gen` (it panicked); or if `src`
    /// has neither arrived nor died within [`WAIT_TIMEOUT`].
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
                    // A rank that published this generation and then left the group is not a dead partner: re-read before condemning it, so the last collective of a call cannot race the partner's return.
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
            // Three tiers, cheapest first: burn a few microseconds where the partner is about to arrive, hand the core over where it is a layer's skew behind, and get off the machine entirely where it is further than that.
            if spins < SPINS_BEFORE_YIELD {
                std::hint::spin_loop();
            } else if spins < SPINS_BEFORE_YIELD + YIELDS_BEFORE_SLEEP {
                std::thread::yield_now();
            } else {
                std::thread::sleep(SLEEP_STEP);
            }
        }
    }

    /// Wait for every partner of `rank` to publish generation `gen` of kind `kind`, folding their tag words in **rank order** through `fold`.
    fn wait_all(&self, rank: u32, gen: u64, kind: CallKind, mut fold: impl FnMut(u32, u64)) {
        for src in 0..self.size {
            if src != rank {
                fold(src, self.wait(rank, src, gen, kind));
            }
        }
    }
}

/// One message on an in-process channel: the payload, plus (debug builds) the
/// sender's collective sequence number.
struct Message {
    /// The sender's collective counter at the time of the send. Debug only — the check it feeds is a development tripwire, not a wire field.
    #[cfg(debug_assertions)]
    seq: u64,
    /// `Option<P>` for an exchange, the contribution for a reduction, `()` for a barrier. Typed on receive by [`downcast`].
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

/// Take a received message's body as `T`.
///
/// A failure means two partitions ran different transport calls at the same step — the collective-order invariant (module docs) — so it is a panic, not an error return.
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

/// In-process transport: `P` partitions sharing one `GroupState` for the collectives, and wired as a `P × P` matrix of unbounded `std::sync::mpsc` channels for the exchange.
///
/// Built as a group by [`group`](Self::group) and moved one per partition thread.
/// Deliberately Rayon-free: it is called from the partition's driving thread between layers, never from inside a parallel region — and by *one* thread per rank, since the generation counter is that thread's call index.
///
/// The three [`Collectives`] operations spin on shared atomics (module docs and `GroupState`); the exchange keeps the channels — it moves a payload, and only on the layers that have one.
/// Channels are unbounded, so a send never blocks and the "send everything, then receive in rank order" shape cannot deadlock.
/// Its receives block; a partner that died is reported by name rather than waited on forever (its dropped sender disconnects the channel).
///
/// `size == 1` is a no-op path: no channels exist, no generation is consumed, the exchange returns one `None`, and the reductions return their input.
pub struct InProcessTransport {
    /// This partition's index.
    rank: u32,
    /// Partitions in the group.
    size: u32,
    /// The group's shared collective state, one `Arc` per rank.
    /// It outlives every rank's endpoint, which is what lets a partner read a slot's buffer while its owner is on its way out.
    state: Arc<GroupState>,
    /// This rank's transport-call counter: one increment per call, the generation published to [`GroupState`] and (in debug builds) the stamp on every exchange message.
    /// Atomic only because the methods take `&self`; nothing but this rank's driving thread touches it.
    gen: AtomicU64,
    /// Sender to partition `q`, `None` in the self slot.
    outbox: Vec<Option<std::sync::mpsc::Sender<Message>>>,
    /// Receiver of what partition `q` sends here, `None` in the self slot.
    /// `Mutex` only to make the transport `Sync` — `Receiver` is `Send` but not `Sync`, and nothing here contends for it.
    inbox: Vec<Option<std::sync::Mutex<std::sync::mpsc::Receiver<Message>>>>,
}

/// Leaving the group marks this rank departed, so a partner spinning for a generation this rank will never publish fails fast instead of waiting out its `WAIT_TIMEOUT` backstop.
///
/// Unconditional rather than `if std::thread::panicking()`: a rank that returns from its partition body with fewer collectives than its partners is exactly as dead to them as one that panicked, and the panic message they raise says so.
/// It cannot fire spuriously at the end of a healthy call — `GroupState::wait` re-reads the partner's progress before condemning it, and a rank only leaves after publishing the group's last generation.
impl Drop for InProcessTransport {
    fn drop(&mut self) {
        self.state
            .departed
            .fetch_or(1u64 << self.rank, Ordering::Release);
    }
}

impl InProcessTransport {
    /// Build a group of `size` transports wired to each other, one per partition, in rank order.
    ///
    /// # Panics
    ///
    /// If `size` is zero.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn group(size: u32) -> Vec<InProcessTransport> {
        Self::group_with_timeout(size, WAIT_TIMEOUT)
    }

    /// [`Self::group`] with the collective wait's backstop set to `timeout`; a group sharing one device queue needs more than the default.
    pub fn group_with_timeout(size: u32, timeout: Duration) -> Vec<InProcessTransport> {
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

    /// The generation of the transport call starting now: one more than the
    /// last, and never zero (an unwritten slot reads as generation 0).
    fn next_gen(&self) -> u64 {
        self.gen.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Pretend one extra collective was issued, to test the order check.
    #[cfg(test)]
    pub(super) fn skip_sequence_for_test(&self) {
        self.gen.fetch_add(1, Ordering::Relaxed);
    }

    /// Send one message to partition `dst`.
    fn send_to(&self, dst: usize, seq: u64, body: Box<dyn std::any::Any + Send>, op: &str) {
        let tx = self.outbox[dst]
            .as_ref()
            .expect("a partition has no channel to itself");
        if tx.send(Message::new(seq, body)).is_err() {
            panic!("partition {dst} terminated before completing the {op} (it panicked)");
        }
    }

    /// Receive this call's message from partition `src`, checking the sequence stamp in debug builds.
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

    /// The maximum rides in the published word's payload byte, so the whole reduction is one store and `P − 1` loads.
    /// `max` is order-independent, so every partition returns the same byte however the arrivals interleave.
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

    /// Each partition publishes its own contribution once and folds its
    /// partners' in place.
    ///
    /// The local sum runs in rank order, but it would not have to: wrapping `u64` addition is exact, associative and commutative, so every partition ends with the identical bits whatever order it folds in.
    fn allreduce_sum_u64(&self, buf: &mut [u64]) {
        if self.size == 1 {
            return;
        }
        let gen = self.next_gen();
        let slot = self.state.value_slot(self.rank, gen);
        // SAFETY: this rank owns the slot, and the generation parity keeps every partner out of it (see `GroupState`). The write must precede the publish below, which is what makes it visible to the partners at all.
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

    /// Every partition publishes its contribution as bits and then folds **all** of them, its own included, in rank order from zero, so each computes the same additions in the same order and gets the same bits.
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
    /// A payload is *moved* to its partner rather than copied, so there is nothing for the two-phase shape to overlap: the transfer is complete before `body` runs and its [`ChunkWait`] is a no-op.
    /// `map` therefore goes unread, and `spare` is neither drawn from nor added to — a sender's blocks become the receiver's, and the pool circulates through the group rather than through the transport.
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
            // The exchange consumes a generation like any other call and publishes it before sending, even though it waits on the channels rather than on the slots: that is what lets a partner spinning in a *collective* at the same generation see the kind mismatch and panic, instead of the pair hanging on each other's different call.
            self.state.publish(self.rank, seq, CallKind::Exchange, 0);
            // Unbounded channels: every send completes before the first receive, so no pair of partitions can block on each other.
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
