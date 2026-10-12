//! [`InProcessTransport`], the shared-memory transport between the partitions of one process.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use super::{AlreadyHere, ChunkMap, ChunkWait, Collectives, Payload, Transport};

/// Which transport call an arrival belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CallKind {
    Barrier,
    MaxU8,
    SumU64,
    Exchange,
    SumF64,
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
}

/// Backstop for a partner that is neither arriving nor departed; a panicking partner is reported through its departure instead.
pub(crate) const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(10);

/// One rank's latest transport call.
#[derive(Default)]
struct Arrival {
    /// The rank's call index, 0 before its first call.
    generation: u64,
    kind: Option<CallKind>,
    /// The contribution to a reduction (`f64`s as bits).
    values: Vec<u64>,
    /// Set once the rank's endpoint is dropped.
    departed: bool,
}

/// The group's shared state, behind [`GroupState::rendezvous`].
struct Rendezvous {
    arrivals: Vec<Arrival>,
    /// The last generation every rank arrived at.
    completed: u64,
    /// The combined values of `completed`; overwritten only once every rank, including any still reading it, has arrived at a later call.
    result: Vec<u64>,
}

impl Rendezvous {
    /// Why `rank`, at `generation` of `kind` with `len` values, cannot complete it in step with its partners, if it cannot.
    fn mismatch(&self, rank: u32, generation: u64, kind: CallKind, len: usize) -> Option<String> {
        let (partner, theirs) = self
            .arrivals
            .iter()
            .enumerate()
            .find(|&(partner, theirs)| {
                partner != rank as usize
                    && (theirs.generation > generation
                        || theirs.generation == generation
                            && (theirs.kind != Some(kind)
                                || kind != CallKind::Exchange && theirs.values.len() != len))
            })?;
        if theirs.generation == generation && theirs.kind == Some(kind) {
            return Some(format!(
                "{}: partition {partner} contributed {} values, this partition {len}",
                kind.name(),
                theirs.values.len(),
            ));
        }
        Some(format!(
            "collective order mismatch: partition {rank} is at transport call {generation} ({}) \
             but partition {partner} published {} at call {} — every partition must issue the \
             identical sequence of transport calls per layer",
            kind.name(),
            theirs.kind.map_or("no call", CallKind::name),
            theirs.generation,
        ))
    }

    /// Whether some partner waits on a call other than `generation` of `kind`, and must be woken to see the mismatch.
    fn partner_waits_elsewhere(&self, rank: u32, generation: u64, kind: CallKind) -> bool {
        self.arrivals.iter().enumerate().any(|(partner, theirs)| {
            partner != rank as usize
                && theirs.generation > self.completed
                && (theirs.generation != generation || theirs.kind != Some(kind))
        })
    }

    /// Fold every rank's contribution to `kind` into `result`, in rank order from zero.
    fn combine(&mut self, kind: CallKind) {
        let len = self.arrivals[0].values.len();
        self.result.clear();
        self.result.resize(len, 0);
        match kind {
            CallKind::Barrier | CallKind::Exchange => {}
            CallKind::MaxU8 => {
                self.result[0] = self.arrivals.iter().map(|a| a.values[0]).max().unwrap_or(0);
            }
            CallKind::SumU64 => {
                for arrival in &self.arrivals {
                    for (total, &v) in self.result.iter_mut().zip(&arrival.values) {
                        *total = total.wrapping_add(v);
                    }
                }
            }
            CallKind::SumF64 => {
                // `result` starts at the bits of `0.0`.
                for arrival in &self.arrivals {
                    for (total, &v) in self.result.iter_mut().zip(&arrival.values) {
                        *total = (f64::from_bits(*total) + f64::from_bits(v)).to_bits();
                    }
                }
            }
        }
    }
}

/// The collective state one [`InProcessTransport`] group shares: a generation-counted rendezvous whose last arriver combines the contributions.
struct GroupState {
    timeout: Duration,
    rendezvous: Mutex<Rendezvous>,
    /// Signalled when a generation completes, a rank departs, or an arrival leaves a waiting partner out of step.
    changed: Condvar,
}

impl GroupState {
    fn new(size: u32, timeout: Duration) -> Self {
        Self {
            timeout,
            rendezvous: Mutex::new(Rendezvous {
                arrivals: (0..size).map(|_| Arrival::default()).collect(),
                completed: 0,
                result: Vec::new(),
            }),
            changed: Condvar::new(),
        }
    }

    /// The state, ignoring poison: every panic below is raised after the guard is dropped, so the state is consistent.
    fn lock(&self) -> MutexGuard<'_, Rendezvous> {
        self.rendezvous
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Wake every waiter, then panic with `message`, so a partner out of step learns of it without waiting for this endpoint's drop.
    fn fail(&self, state: MutexGuard<'_, Rendezvous>, message: String) -> ! {
        drop(state);
        self.changed.notify_all();
        panic!("{message}");
    }

    /// Record `rank`'s arrival at `generation` of `kind` with `values`, completing the generation if it is the last; returns the locked state.
    fn arrive(
        &self,
        rank: u32,
        generation: u64,
        kind: CallKind,
        values: &[u64],
    ) -> MutexGuard<'_, Rendezvous> {
        let mut state = self.lock();
        let arrival = &mut state.arrivals[rank as usize];
        arrival.generation = generation;
        arrival.kind = Some(kind);
        arrival.values.clear();
        arrival.values.extend_from_slice(values);
        if let Some(message) = state.mismatch(rank, generation, kind, values.len()) {
            self.fail(state, message);
        }
        if state.arrivals.iter().all(|a| a.generation == generation) {
            state.combine(kind);
            state.completed = generation;
            self.changed.notify_all();
        } else if state.partner_waits_elsewhere(rank, generation, kind) {
            self.changed.notify_all();
        }
        state
    }

    /// Contribute `buffer` to `generation` of `kind`, wait for every partner, and overwrite `buffer` with the combined result; panics on a mismatched call, a departed partner, or the timeout.
    fn collective(&self, rank: u32, generation: u64, kind: CallKind, buffer: &mut [u64]) {
        let mut state = self.arrive(rank, generation, kind, buffer);
        let mut deadline: Option<Instant> = None;
        loop {
            if state.completed >= generation {
                buffer.copy_from_slice(&state.result);
                return;
            }
            if let Some(message) = state.mismatch(rank, generation, kind, buffer.len()) {
                self.fail(state, message);
            }
            let behind = state
                .arrivals
                .iter()
                .enumerate()
                .filter(|(_, a)| a.generation < generation);
            if let Some((partner, _)) = behind.clone().find(|(_, a)| a.departed) {
                let message = format!(
                    "partition {partner} terminated before completing the {} (it panicked)",
                    kind.name(),
                );
                self.fail(state, message);
            }
            let now = Instant::now();
            let deadline = *deadline.get_or_insert(now + self.timeout);
            if now >= deadline {
                let (partner, theirs) = behind
                    .min_by_key(|(_, a)| a.generation)
                    .expect("an incomplete generation has a partner behind it");
                let message = format!(
                    "partition {partner} terminated before completing the {}: no response in \
                     {:?} (this partition is at transport call {generation}, that one at {})",
                    kind.name(),
                    self.timeout,
                    theirs.generation,
                );
                self.fail(state, message);
            }
            state = self
                .changed
                .wait_timeout(state, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }
}

/// One exchange message: the `Option<P>` payload, plus the sender's generation in debug builds.
struct Message {
    #[cfg(debug_assertions)]
    generation: u64,
    body: Box<dyn std::any::Any + Send>,
}

impl Message {
    fn new(generation: u64, body: Box<dyn std::any::Any + Send>) -> Self {
        let _ = generation;
        Self {
            #[cfg(debug_assertions)]
            generation,
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

/// The transport between the partitions of one process: a mutex-and-condvar rendezvous for the collectives and a `P × P` matrix of unbounded `mpsc` channels for the exchange (ARCHITECTURE.md §Transport composition).
///
/// Built as a group by [`group`](Self::group), one endpoint per partition thread, and called by one thread per rank, since the generation counter is that thread's call index.
pub struct InProcessTransport {
    rank: u32,
    size: u32,
    /// Outlives every endpoint, so a partner can still read the state once its owner has left.
    state: Arc<GroupState>,
    /// This rank's transport-call counter, touched only by its driving thread.
    generation: AtomicU64,
    /// Sender to partition `q`, `None` in the self slot.
    outbox: Vec<Option<std::sync::mpsc::Sender<Message>>>,
    /// Receiver from partition `q`; the uncontended `Mutex` only makes the transport `Sync`.
    inbox: Vec<Option<std::sync::Mutex<std::sync::mpsc::Receiver<Message>>>>,
}

/// Marks this rank departed so waiting partners fail fast; unconditional, since a rank that returns early is as dead to its partners as one that panicked.
impl Drop for InProcessTransport {
    fn drop(&mut self) {
        self.state.lock().arrivals[self.rank as usize].departed = true;
        self.state.changed.notify_all();
    }
}

impl InProcessTransport {
    /// A group of `size` endpoints wired to each other, in rank order.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn group(size: u32) -> Vec<InProcessTransport> {
        Self::group_with_timeout(size, DEFAULT_WAIT_TIMEOUT)
    }

    /// [`Self::group`] with the collective wait's backstop set to `timeout`.
    pub(crate) fn group_with_timeout(size: u32, timeout: Duration) -> Vec<InProcessTransport> {
        assert!(size > 0, "a transport group needs at least one partition");
        let n = size as usize;
        let mut senders: Vec<Vec<Option<std::sync::mpsc::Sender<Message>>>> =
            (0..n).map(|_| (0..n).map(|_| None).collect()).collect();
        let mut receivers: Vec<Vec<Option<std::sync::mpsc::Receiver<Message>>>> =
            (0..n).map(|_| (0..n).map(|_| None).collect()).collect();
        for source in 0..n {
            for destination in 0..n {
                if source != destination {
                    let (sender, receiver) = std::sync::mpsc::channel();
                    senders[source][destination] = Some(sender);
                    receivers[source][destination] = Some(receiver);
                }
            }
        }

        let state = Arc::new(GroupState::new(size, timeout));
        (0..n)
            .map(|rank| InProcessTransport {
                rank: rank as u32,
                size,
                state: Arc::clone(&state),
                generation: AtomicU64::new(0),
                outbox: std::mem::take(&mut senders[rank]),
                inbox: (0..n)
                    .map(|source| receivers[source][rank].take().map(std::sync::Mutex::new))
                    .collect(),
            })
            .collect()
    }

    /// The generation of the call starting now, never zero.
    fn next_generation(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Pretend one extra collective was issued, to test the order check.
    #[cfg(test)]
    pub(super) fn skip_sequence_for_test(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    fn send_to(
        &self,
        destination: usize,
        generation: u64,
        body: Box<dyn std::any::Any + Send>,
        op: &str,
    ) {
        let sender = self.outbox[destination]
            .as_ref()
            .expect("a partition has no channel to itself");
        if sender.send(Message::new(generation, body)).is_err() {
            panic!("partition {destination} terminated before completing the {op} (it panicked)");
        }
    }

    /// Receive this call's message from `source`, checking the generation stamp in debug builds.
    fn receive_from(
        &self,
        source: usize,
        generation: u64,
        op: &str,
    ) -> Box<dyn std::any::Any + Send> {
        let receiver = self.inbox[source]
            .as_ref()
            .expect("a partition has no channel to itself")
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match receiver.recv() {
            Ok(message) => {
                #[cfg(debug_assertions)]
                assert_eq!(
                    message.generation, generation,
                    "collective order mismatch: partition {} is at transport call {generation} but \
                     partition {source} sent its call {} — every partition must issue the identical \
                     sequence of transport calls per layer",
                    self.rank, message.generation,
                );
                let _ = generation;
                message.body
            }
            Err(_) => {
                panic!("partition {source} terminated before completing the {op} (it panicked)")
            }
        }
    }
}

impl crate::collectives::sealed::Sealed for InProcessTransport {}

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
        let mut maximum = [u64::from(v)];
        let generation = self.next_generation();
        self.state
            .collective(self.rank, generation, CallKind::MaxU8, &mut maximum);
        maximum[0] as u8
    }

    fn allreduce_sum_u64(&self, buffer: &mut [u64]) {
        if self.size == 1 {
            return;
        }
        let generation = self.next_generation();
        self.state
            .collective(self.rank, generation, CallKind::SumU64, buffer);
    }

    /// The last arriver folds all contributions in rank order from zero and every partition copies that one result, so all get the same bits.
    fn allreduce_sum_f64(&self, buffer: &mut [f64]) {
        if self.size == 1 {
            return;
        }
        let generation = self.next_generation();
        // The `f64`s travel as their bit patterns.
        self.state.collective(
            self.rank,
            generation,
            CallKind::SumF64,
            bytemuck::cast_slice_mut(buffer),
        );
    }

    fn barrier(&self) {
        if self.size == 1 {
            return;
        }
        let generation = self.next_generation();
        self.state
            .collective(self.rank, generation, CallKind::Barrier, &mut []);
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
        let received: Vec<Option<P>> = if n == 1 {
            vec![None]
        } else {
            let generation = self.next_generation();
            // Recorded although it waits on the channels, so a partner in a collective at this generation sees the kind mismatch.
            drop(
                self.state
                    .arrive(self.rank, generation, CallKind::Exchange, &[]),
            );
            for (destination, payload) in send.into_iter().enumerate() {
                if destination != self.rank as usize {
                    self.send_to(destination, generation, Box::new(payload), "exchange");
                }
            }
            (0..n)
                .map(|source| {
                    if source == self.rank as usize {
                        None
                    } else {
                        downcast::<Option<P>>(
                            self.receive_from(source, generation, "exchange"),
                            source,
                            "exchange",
                        )
                    }
                })
                .collect()
        };

        let out = body(&received, &AlreadyHere);
        (received, out)
    }
}
