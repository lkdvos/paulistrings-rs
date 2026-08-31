#!/usr/bin/env julia
#
# Semantics probes for the PauliPropagation.jl baseline
# =====================================================
#
# Every claim in `benchmarks/julia/README.md` about PauliPropagation.jl's
# semantics is produced by this script. Run it and paste/compare the output;
# it takes no arguments and writes only to stdout.
#
#     julia --project=benchmarks/julia benchmarks/julia/probes.jl
#
# The probes are deliberately hand-computable: every expected value below is
# derived in a comment, never read back from the library.

using PauliPropagation

const RULE = "=" ^ 78

section(title) = (println(); println(RULE); println(title); println(RULE))

"""Render a PauliSum as a sorted Vector of (string, coeff) for stable printing."""
function dump(psum)
    n = nqubits(psum)
    out = [(inttostring(p, n), c) for (p, c) in zip(paulis(psum), coefficients(psum))]
    sort!(out; by = first)
    return out
end

# ---------------------------------------------------------------------------
section("P0. Versions")
# ---------------------------------------------------------------------------

println("julia              = ", string(VERSION))
println("PauliPropagation   = ", string(pkgversion(PauliPropagation)))
println("Threads.nthreads() = ", Threads.nthreads())

# ---------------------------------------------------------------------------
section("P1. String / qubit-index convention")
# ---------------------------------------------------------------------------
#
# `symboltoint` enumerates the symbol vector with `enumerate`, so symbol
# position 1 lands on qubit index 1, and `inttostring(pstr, n)` walks
# `ii in 1:n` left to right. Prediction: qubit 1 is the LEFTMOST character,
# and qubit indices are 1-based.

for q in 1:3
    pstr = PauliString(3, :Z, q, 1.0)
    println("PauliString(3, :Z, $q) -> \"", inttostring(pstr.term, 3), "\"")
end
println("getpauli codes: I=", symboltoint(:I), " X=", symboltoint(:X),
        " Y=", symboltoint(:Y), " Z=", symboltoint(:Z))

# ---------------------------------------------------------------------------
section("P2. Pauli-Y / coefficient convention (S X S† = -Y ... which sign, where?)")
# ---------------------------------------------------------------------------
#
# S = diag(1, i).  By hand:
#   S  X S† = diag(1,i)·[[0,1],[1,0]]·diag(1,-i) = [[0,-i],[i,0]]  = +Y
#   S† X S  = diag(1,-i)·[[0,1],[1,0]]·diag(1,i) = [[0, i],[-i,0]] = -Y
# with the Hermitian Y = [[0,-i],[i,0]].
#
# So a HERMITIAN-Y convention (coefficient multiplies the literal Pauli
# string, Y carries no phase of its own) must give:
#   heisenberg=true  (U† O U):  X -> -1.0 * Y
#   heisenberg=false (U O U†):  X -> +1.0 * Y
# and both coefficients must stay REAL. A "phaseless-iY" convention would
# instead show a ±i somewhere.

psum_x = PauliSum(PauliString(1, :X, 1, 1.0))
s_gate = CliffordGate(:S, [1])

for heis in (true, false)
    out = propagate(s_gate, psum_x; heisenberg = heis, min_abs_coeff = 0.0)
    println("S, heisenberg=$heis : X -> ", dump(out), "   coefftype=", coefftype(out))
end

# Same question for a rotation: rz(θ) = exp(-iθZ/2) acting on X.
#   U† X U = cos(θ) X - sin(θ) Y      (heisenberg=true)
#   U  X U† = cos(θ) X + sin(θ) Y     (heisenberg=false)
let theta = 0.3
    for heis in (true, false)
        out = propagate([PauliRotation(:Z, 1)], psum_x, [theta];
                        heisenberg = heis, min_abs_coeff = 0.0)
        println("rz($theta), heisenberg=$heis : X -> ", dump(out),
                "   [cos=", cos(theta), ", sin=", sin(theta), "]")
    end
end

# ---------------------------------------------------------------------------
section("P3. min_abs_coeff boundary: is |c| == threshold dropped or kept?")
# ---------------------------------------------------------------------------
#
# `truncatemincoeff(coeff, min_abs_coeff) = abs(coeff) < min_abs_coeff`
# (Base/truncate.jl) predicts STRICT: a coefficient exactly equal to the
# threshold is KEPT. This repo's `CoefficientThreshold` keeps `|c| > eps`,
# i.e. it DROPS the boundary. Probe with an exactly-representable dyadic.
#
# Circuit: CliffordGate(:Z) on a Z string is the identity map with sign +1,
# so the coefficient is bit-exactly preserved and the only thing that can
# remove the term is the truncation.

for c in (0.25, 0.25 - eps(0.25), 0.25 + eps(0.25))
    psum = PauliSum(PauliString(1, :Z, 1, c))
    out = propagate(CliffordGate(:Z, [1]), psum; min_abs_coeff = 0.25)
    println("coeff=", repr(c), " (== 0.25: ", c == 0.25, ")  min_abs_coeff=0.25 -> ",
            length(out), " term(s) ", dump(out))
end

# and via the standalone truncate! entry point, to show it is the same predicate
let psum = PauliSum(1, Dict(symboltoint([:Z]) => 0.25))
    kept = length(truncate!(psum; min_abs_coeff = 0.25))
    println("truncate!(psum with |c|=0.25; min_abs_coeff=0.25) -> ", kept, " term(s)")
end

# ---------------------------------------------------------------------------
section("P4. max_weight boundary: is weight == max_weight dropped or kept?")
# ---------------------------------------------------------------------------
#
# `truncateweight(pstr, max_weight) = countweight(pstr) > max_weight`
# predicts weight == max_weight is KEPT. This repo's `WeightCutoff` keeps
# `weight <= k`. Prediction: the two AGREE.

let psum = PauliSum([PauliString(3, [:Z], [1], 1.0),
                     PauliString(3, [:Z, :Z], [1, 2], 1.0),
                     PauliString(3, [:Z, :Z, :Z], [1, 2, 3], 1.0)])
    out = propagate(CliffordGate(:Z, [1]), psum; min_abs_coeff = 0.0, max_weight = 2)
    println("weights {1,2,3}, max_weight=2 -> ", dump(out))
end

# ---------------------------------------------------------------------------
section("P5. When is truncation applied — per gate, or per layer?")
# ---------------------------------------------------------------------------
#
# `_applymergetruncate!` calls `truncate!(prop_cache)` after EVERY gate
# (Base/propagate.jl), and `_propagate!` has no notion of a layer at all.
# Probe: rz(θ) on X splits into cos(θ)·X + sin(θ)·Y. Pick θ and a threshold
# with sin(θ) < eps < cos(θ), so the Y branch dies immediately. If truncation
# were deferred to the end of a "layer" of two rz gates, the second gate
# would still see 2 terms and the count after gate 2 would be larger.
#
# θ = 0.05: cos = 0.99875, sin = 0.049979. eps = 0.1 kills only the Y branch.

let theta = 0.05, thresh = 0.1
    circ = [PauliRotation(:Z, 1), PauliRotation(:X, 1)]
    counts = @countpaulis propagate(circ, PauliSum(PauliString(1, :X, 1, 1.0)),
                                    [theta, theta];
                                    heisenberg = true, min_abs_coeff = thresh)
    println("θ=$theta, min_abs_coeff=$thresh, 2 gates -> per-gate counts = ", counts)

    counts_loose = @countpaulis propagate(circ, PauliSum(PauliString(1, :X, 1, 1.0)),
                                          [theta, theta];
                                          heisenberg = true, min_abs_coeff = 0.0)
    println("same circuit, min_abs_coeff=0.0     -> per-gate counts = ", counts_loose)
end

# `@countpaulis` ordering: counts are in APPLICATION order, which for
# heisenberg=true is the reverse of the circuit as written. Probe with two
# distinguishable gates: a CNOT that fans nothing out, then a rotation.
let
    # written order: [rz(1), cnot(1->2)];  heisenberg applies cnot first.
    circ = [PauliRotation(:Z, 1), CliffordGate(:CNOT, [1, 2])]
    obs = PauliSum(PauliString(2, :X, 1, 1.0))
    counts = @countpaulis propagate(circ, obs, [0.7]; heisenberg = true, min_abs_coeff = 0.0)
    println("counts for [rz, cnot] heisenberg=true (applied cnot-then-rz) = ", counts,
            "   (1 then 2 => reverse-of-written application order)")
    counts_f = @countpaulis propagate(circ, obs, [0.7]; heisenberg = false, min_abs_coeff = 0.0)
    println("counts for [rz, cnot] heisenberg=false (applied rz-then-cnot) = ", counts_f)
end

# ---------------------------------------------------------------------------
section("P6. Noise-channel parameter mapping")
# ---------------------------------------------------------------------------
#
# This repo: depolarize(p) scales every non-identity Pauli on the support by
# 1 - 4p/3; dephase(p) scales X and Y by 1 - 2p; amplitude_damping(γ) gives
# I->I, X,Y->√(1-γ)·same, Z->(1-γ)Z + γI (Heisenberg/`apply`).
#
# jl: DepolarizingNoise(q, λ) scales X,Y,Z by 1-λ  =>  λ = 4p/3
#     DephasingNoise(q, λ)    scales X,Y   by 1-λ  =>  λ = 2p
#     AmplitudeDampingNoise(q, γ) should match 1:1.

let p = 0.15
    for (sym, gate) in ((:X, PauliString(1, :X, 1, 1.0)),
                        (:Y, PauliString(1, :Y, 1, 1.0)),
                        (:Z, PauliString(1, :Z, 1, 1.0)))
        dep = propagate([DepolarizingNoise(1)], PauliSum(gate), [4p / 3];
                        min_abs_coeff = 0.0)
        deph = propagate([DephasingNoise(1)], PauliSum(gate), [2p]; min_abs_coeff = 0.0)
        println("$sym : Depolarizing(λ=4p/3=", 4p / 3, ") -> ", dump(dep),
                " | expect 1-4p/3=", 1 - 4p / 3)
        println("$sym : Dephasing(λ=2p=", 2p, ")     -> ", dump(deph),
                " | expect 1-2p=", 1 - 2p, " on X,Y and 1.0 on Z")
    end
end

let gamma = 0.3
    for sym in (:X, :Y, :Z)
        out = propagate([AmplitudeDampingNoise(1)], PauliSum(PauliString(1, sym, 1, 1.0)),
                        [gamma]; heisenberg = true, min_abs_coeff = 0.0)
        println("$sym : AmplitudeDamping(γ=$gamma) -> ", dump(out),
                " | expect √(1-γ)=", sqrt(1 - gamma), " for X,Y; (1-γ)Z + γI for Z")
    end
end

# ---------------------------------------------------------------------------
section("P7. TransferMapGate matrix ordering (for unitary_1q / unitary_2q)")
# ---------------------------------------------------------------------------
#
# `TransferMapGate(mat, qinds)` accepts a 2^n x 2^n unitary in the 0/1 basis.
# For n = 2 the Kronecker order is not documented, so pin it by comparing
# against `CliffordGate(:CNOT, [1, 2])`, whose action is known.

let
    H = [1 1; 1 -1] / sqrt(2)
    obs = PauliSum(PauliString(1, :X, 1, 1.0))
    a = propagate(CliffordGate(:H, [1]), obs; min_abs_coeff = 0.0)
    b = propagate(TransferMapGate(Matrix{ComplexF64}(H), [1]), obs; min_abs_coeff = 0.0)
    println("1q: CliffordGate(:H) -> ", dump(a))
    println("1q: TransferMapGate(H) -> ", dump(b))
end

let
    # CNOT with control = first tensor factor, target = second:
    #   |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>
    cnot_ct = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
    # CNOT with control = second tensor factor, target = first:
    cnot_tc = ComplexF64[1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0]

    for obs_sym in (:X, :Z)
        for q in (1, 2)
            obs = PauliSum(PauliString(2, obs_sym, q, 1.0))
            ref = propagate(CliffordGate(:CNOT, [1, 2]), obs; min_abs_coeff = 0.0)
            m1 = propagate(TransferMapGate(cnot_ct, [1, 2]), obs; min_abs_coeff = 0.0)
            m2 = propagate(TransferMapGate(cnot_tc, [1, 2]), obs; min_abs_coeff = 0.0)
            println("2q $obs_sym on q$q: Clifford=", dump(ref),
                    "  ctrl-first-matrix=", dump(m1),
                    "  ctrl-second-matrix=", dump(m2))
        end
    end
end

# ---------------------------------------------------------------------------
section("P8. Diagonal-PTM ordering (for pauli_channel / depolarize2)")
# ---------------------------------------------------------------------------
#
# A 4^n x 4^n matrix passed to `TransferMapGate` is taken as a PTM verbatim
# (no Heisenberg conjugation). A single-qubit Pauli channel and 2q
# depolarizing are diagonal PTMs, so they can be ONE gate — provided the
# basis index order is the `symboltoint` order I=0, X=1, Y=2, Z=3. Pin it
# with three distinct diagonal entries.

let
    ptm = zeros(Float64, 4, 4)
    ptm[1, 1] = 1.0
    ptm[2, 2] = 0.2   # if the order is (I, X, Y, Z), this hits X
    ptm[3, 3] = 0.3   # ... Y
    ptm[4, 4] = 0.4   # ... Z
    for sym in (:I, :X, :Y, :Z)
        obs = PauliSum(PauliString(1, sym, 1, 1.0))
        out = propagate(TransferMapGate(ptm, [1]), obs; min_abs_coeff = 0.0)
        println("diag PTM (1, .2, .3, .4) on $sym -> ", dump(out))
    end
end

let
    # 2q: index = 4*(pauli on qinds[2]) + (pauli on qinds[1])?  or the other
    # way round?  Mark only the (X on one site, I on the other) columns.
    ptm = zeros(Float64, 16, 16)
    for i in 1:16
        ptm[i, i] = 1.0
    end
    ptm[2, 2] = 0.5     # basis index 1
    ptm[5, 5] = 0.25    # basis index 4
    for (sym, q) in ((:X, 1), (:X, 2))
        obs = PauliSum(PauliString(2, sym, q, 1.0))
        out = propagate(TransferMapGate(ptm, [1, 2]), obs; min_abs_coeff = 0.0)
        println("2q diag PTM (idx1=0.5, idx4=0.25) on $sym at q$q -> ", dump(out))
    end
end

# ---------------------------------------------------------------------------
section("P9. Exact-zero coefficients: kept or dropped at min_abs_coeff=0?")
# ---------------------------------------------------------------------------
#
# With min_abs_coeff = 0.0, `abs(c) < 0` is never true, so an exactly-zero
# coefficient survives in jl. The Rust merge kernels drop exact zeros
# unconditionally. This is a term-count divergence that only fires when a
# merge cancels exactly, so parity runs should use a strictly positive
# min_abs_coeff (or avoid exactly-cancelling circuits).

let
    # rz(π) on X: cos(π) = -1, sin(π) = 1.2246e-16 -- not an exact zero.
    # Force an exact cancellation instead: X + X with opposite signs.
    psum = PauliSum(1, Dict(symboltoint([:X]) => 0.0))
    out = propagate(CliffordGate(:Z, [1]), psum; min_abs_coeff = 0.0)
    println("single term with coeff 0.0, min_abs_coeff=0.0 -> ", length(out), " term(s)")
end

println()
println(RULE)
println("probes complete")
println(RULE)
