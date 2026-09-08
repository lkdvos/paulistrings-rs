# Title

**Pauli propagation at scale: "Wie niet sterk is, moet slim zijn"**

# Abstract (announcement version)

Pauli propagation simulates quantum circuits by evolving an observable in the Pauli basis and truncating the
sum, trading exactness for sparsity. It can reach wide and deep circuits where state vectors and tensor
networks give up, but only by leveraging efficient bitwise operations and pushing billions of term updates
per run. In this talk I present an approach that outperforms existing implementations, using the 127-qubit
heavy-hex kicked Ising circuit as the running example through every attempt to make it fast.

I will start with the general concepts and the algorithmic approaches currently in use. Then I will take you
through my attempts at improving on them, and why naive optimizations make little headway once memory, not
arithmetic, is the bottleneck. Finally, I will explain a new approach that sidesteps these limits: a
GF(2)-linear hash that partitions the problem so that every layer becomes embarrassingly parallel, while also
speeding up the single-threaded case, and that extends naturally towards distributed memory.

# Abstract (results-forward alternative)

Pauli propagation simulates quantum circuits by evolving an observable in the Pauli basis and truncating the
sum, trading exactness for sparsity. It reaches 127-qubit, deep circuits where state vectors and tensor networks
give up, but only by pushing billions of term updates per run. This talk follows one benchmark, the 127-qubit
heavy-hex kicked Ising circuit, through every attempt to make that fast.

The first attempts are about being strong. A tuned hash-map kernel with vectorization flags gains a few
percent. Parallelizing it with per-thread dictionaries or a parallel mergesort never beats a factor of two on
sixteen cores, because both keep one global order and pay for it with one global merge. Counting bytes shows
why: at 48 bytes per term the algorithm is bound by memory latency and bandwidth, not arithmetic.

The turn comes from an algebraic observation. Partitioning the sum by a random GF(2)-linear hash makes a gate's
output buckets statically predictable and duplicates bucket-local, so a layer decomposes into write-disjoint
tasks with no locks, no atomics and no global sort. The same engine runs three times faster on one thread and
scales to more than ten times on sixteen cores, with the bucket size deciding whether the cores help at all:
buckets that fit the private L2 caches give 11×, buckets that do not give 2×. The structure keeps giving:
smaller footprint, GPU-ready buffers, and a statically known communication plan for distributed memory.
