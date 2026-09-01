#!/usr/bin/env julia
#
# PauliPropagation.jl baseline runner
# ===================================
#
# Reads a task-JSON file (schema v1, frozen in
# research/notes/2026-09-01-python-api-extensions.md §A5), builds the circuit
# and observable in PauliPropagation.jl, propagates, and emits a result JSON.
#
#     julia --project=benchmarks/julia benchmarks/julia/runner.jl task.json
#     julia --project=benchmarks/julia benchmarks/julia/runner.jl task.json -o out.json
#
# stdout carries ONLY the result JSON (one line); progress and diagnostics go
# to stderr, so the Python wrapper can parse stdout unconditionally.
#
# The schema is versioned rather than tolerant: unknown top-level keys,
# unknown gate names, unknown gate fields and missing required keys are hard
# errors on both sides. Nothing is defaulted that the schema does not default.
#
# Semantic mapping to PauliPropagation.jl (every claim here is produced by
# benchmarks/julia/probes.jl and recorded in benchmarks/julia/README.md):
#
#   * Qubit indices: the schema is 0-based (this repo's convention), jl is
#     1-based => +1 everywhere.
#   * Pauli-string keys: leftmost character is qubit 0 in the schema and
#     qubit 1 in jl -- the same left-to-right order, so keys map verbatim.
#   * Hermitian-Y on both sides: a real coefficient multiplies the literal
#     Pauli string, `Y` carries no phase of its own.
#   * direction "heisenberg" -> jl `heisenberg=true`  (reverse order, U'PU)
#     direction "forward"    -> jl `heisenberg=false` (written order, UPU')
#   * min_abs_coeff: jl drops `|c| < eps`; this repo drops `|c| <= eps`. The
#     two disagree exactly on the boundary. NOT papered over here -- the
#     value is passed through unchanged and the divergence is reported in the
#     output's `notes`.
#   * When `truncation.min_abs_coeff` is absent the runner passes 0.0, because
#     jl's own default is 1e-10 and silently truncating would be wrong.
#   * Truncation runs after every gate in jl (`_applymergetruncate!`), which is
#     why the schema is one gate object = one channel.
#
# Non-schema knobs (environment variables, for benchmarking only -- they never
# change what the task means):
#
#   PP_BACKEND=dict|vector   storage backend (default dict, jl's `PauliSum`)
#   PP_WARM_REPEATS=N        timed warm propagations after the cold one (default 3)
#   PP_LAYER_COUNTS=0|1      collect per-gate term counts in an extra,
#                            untimed propagation (default 1)
#   PP_FUSED=0|1             experimental fused rotation kernel (default 0);
#                            term-count parity is only established for 0
#   PP_EMIT_TERMS=N          also emit the evolved sum term-by-term when it has
#                            at most N terms (default 0 = never). For parity
#                            debugging on small cases only.

using PauliPropagation
using JSON3

const SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

struct TaskError <: Exception
    msg::String
end
Base.showerror(io::IO, e::TaskError) = print(io, "task error: ", e.msg)

fail(msg) = throw(TaskError(msg))

# ---------------------------------------------------------------------------
# Strict JSON access helpers
# ---------------------------------------------------------------------------

_keystrings(obj) = sort!([String(k) for k in keys(obj)])

function require_keys(obj, required, allowed, what)
    isa(obj, JSON3.Object) || fail("$what must be a JSON object, got $(typeof(obj))")
    present = Set(String(k) for k in keys(obj))
    for k in required
        k in present || fail("$what is missing required key \"$k\" (present: $(_keystrings(obj)))")
    end
    for k in present
        k in allowed || fail("$what has unknown key \"$k\" (allowed: $(sort(collect(allowed))))")
    end
    return obj
end

function getint(obj, key, what)
    v = obj[Symbol(key)]
    isa(v, Integer) || fail("$what field \"$key\" must be an integer, got $(repr(v))")
    return Int(v)
end

function getfloat(obj, key, what)
    v = obj[Symbol(key)]
    isa(v, Real) || fail("$what field \"$key\" must be a number, got $(repr(v))")
    return Float64(v)
end

function getstr(obj, key, what)
    v = obj[Symbol(key)]
    isa(v, AbstractString) || fail("$what field \"$key\" must be a string, got $(repr(v))")
    return String(v)
end

function getqubits(obj, what, n_qubits, expected_len)
    v = obj[:qubits]
    isa(v, JSON3.Array) || fail("$what field \"qubits\" must be a list, got $(repr(v))")
    qs = Int[]
    for q in v
        isa(q, Integer) || fail("$what field \"qubits\" must contain integers, got $(repr(q))")
        0 <= q < n_qubits || fail("$what qubit index $q out of range for n_qubits=$n_qubits")
        push!(qs, Int(q))
    end
    if expected_len !== nothing && length(qs) != expected_len
        fail("$what expects $expected_len qubit(s), got $(length(qs)): $qs")
    end
    length(qs) == length(Set(qs)) || fail("$what qubit indices must be distinct, got $qs")
    return qs
end

# Parse a JSON coefficient: a bare number, or [re, im].
function getcomplex(v, what)
    if isa(v, Real)
        return ComplexF64(Float64(v), 0.0)
    elseif isa(v, JSON3.Array) && length(v) == 2 && all(x -> isa(x, Real), v)
        return ComplexF64(Float64(v[1]), Float64(v[2]))
    end
    fail("$what must be a number or a [re, im] pair, got $(repr(v))")
end

# Parse a JSON matrix of [re, im] entries (or bare reals) into a dense matrix.
function getmatrix(obj, what, dim)
    v = obj[:matrix]
    isa(v, JSON3.Array) || fail("$what field \"matrix\" must be a list of rows")
    length(v) == dim || fail("$what field \"matrix\" must have $dim rows, got $(length(v))")
    mat = zeros(ComplexF64, dim, dim)
    for (i, row) in enumerate(v)
        isa(row, JSON3.Array) || fail("$what matrix row $i must be a list")
        length(row) == dim || fail("$what matrix row $i must have $dim entries, got $(length(row))")
        for (j, entry) in enumerate(row)
            mat[i, j] = getcomplex(entry, "$what matrix entry [$i][$j]")
        end
    end
    return mat
end

# ---------------------------------------------------------------------------
# Gate vocabulary (schema v1, verbatim)
# ---------------------------------------------------------------------------

const PAULI_SYMBOL = Dict('X' => :X, 'Y' => :Y, 'Z' => :Z)

# Diagonal PTM helper: `factors[p]` scales the local Pauli whose per-site
# codes are `p` (a vector of symbols, one per qubit in `qinds` order).
# Basis index order is `symboltoint`'s -- verified by probes.jl P8:
# 1q order is (I, X, Y, Z); 2q index = code(q1) + 4*code(q2).
function diagonal_ptm(factors::Dict{Vector{Symbol},Float64}, nq::Int)
    dim = 4^nq
    ptm = zeros(Float64, dim, dim)
    for (syms, f) in factors
        length(syms) == nq || fail("internal: PTM key $syms does not have $nq entries")
        idx = Int(symboltoint(syms)) + 1
        ptm[idx, idx] = f
    end
    return ptm
end

const _ALL_SYMS = (:I, :X, :Y, :Z)

"""
Single-qubit Pauli channel `E(rho) = (1-px-py-pz) rho + px XrhoX + py YrhoY + pz ZrhoZ`.
Heisenberg dual (self-adjoint, diagonal): I -> 1, X -> 1-2(py+pz), Y -> 1-2(px+pz),
Z -> 1-2(px+py). Built as a one-gate diagonal PTM so the per-gate truncation
timing matches a single `paulistrings` channel.
"""
function pauli_channel_gate(px, py, pz, qind)
    factors = Dict{Vector{Symbol},Float64}(
        [:I] => 1.0,
        [:X] => 1 - 2 * (py + pz),
        [:Y] => 1 - 2 * (px + pz),
        [:Z] => 1 - 2 * (px + py),
    )
    return TransferMapGate(diagonal_ptm(factors, 1), [qind])
end

"""
Two-qubit uniform depolarizing with total error probability `p` spread over the
15 non-identity 2q Paulis. Dual: II -> 1, everything else -> 1 - 16p/15.
"""
function depolarize2_gate(p, q0, q1)
    scale = 1 - 16 * p / 15
    factors = Dict{Vector{Symbol},Float64}()
    for a in _ALL_SYMS, b in _ALL_SYMS
        factors[[a, b]] = (a === :I && b === :I) ? 1.0 : scale
    end
    return TransferMapGate(diagonal_ptm(factors, 2), [q0, q1])
end

# Gates that jl cannot put into the Schrodinger picture: `_toschrodinger` has
# no method for `TransferMapGate` or `AmplitudeDampingNoise`, so a
# direction="forward" task containing one of these is rejected up front with a
# message naming the gap, rather than dying inside `propagate`.
const NO_FORWARD = Set(["unitary_1q", "unitary_2q", "amplitude_damping",
                        "pauli_channel", "depolarize2"])

"""
    build_gate(gobj, n_qubits, direction) -> Gate

Translate one schema-v1 gate object into a frozen PauliPropagation.jl gate.
Everything is frozen (`FrozenGate`), so `propagate` needs no parameter vector
and no parameter-ordering assumption can go wrong.
"""
function build_gate(gobj, n_qubits::Int, direction::String)
    isa(gobj, JSON3.Object) || fail("each entry of circuit.gates must be an object, got $(repr(gobj))")
    haskey(gobj, :name) || fail("gate object is missing required key \"name\": $(_keystrings(gobj))")
    name = getstr(gobj, "name", "gate")

    if direction == "forward" && name in NO_FORWARD
        fail("gate \"$name\" cannot be propagated with direction=\"forward\": " *
             "PauliPropagation.jl 0.8.2 defines no `_toschrodinger` method for the gate type it " *
             "maps to (TransferMapGate / AmplitudeDampingNoise), so the Schrodinger picture is " *
             "unavailable. Use direction=\"heisenberg\", or see benchmarks/julia/README.md " *
             "(\"Known gaps\").")
    end

    what = "gate \"$name\""

    if name in ("h", "s", "x", "y", "z")
        require_keys(gobj, ["name", "qubits"], Set(["name", "qubits"]), what)
        q = getqubits(gobj, what, n_qubits, 1)[1]
        sym = name == "h" ? :H : name == "s" ? :S : name == "x" ? :X : name == "y" ? :Y : :Z
        return CliffordGate(sym, [q + 1])

    elseif name in ("cnot", "cz", "swap")
        require_keys(gobj, ["name", "qubits"], Set(["name", "qubits"]), what)
        qs = getqubits(gobj, what, n_qubits, 2)
        sym = name == "cnot" ? :CNOT : name == "cz" ? :CZ : :SWAP
        # jl's :CNOT map takes qinds[1] as the control, matching the schema's
        # qubits = [control, target].
        return CliffordGate(sym, [qs[1] + 1, qs[2] + 1])

    elseif name in ("rz", "rx", "ry")
        require_keys(gobj, ["name", "qubits", "theta"], Set(["name", "qubits", "theta"]), what)
        q = getqubits(gobj, what, n_qubits, 1)[1]
        theta = getfloat(gobj, "theta", what)
        sym = name == "rz" ? :Z : name == "rx" ? :X : :Y
        return PauliRotation(sym, q + 1, theta)

    elseif name == "pauli_rotation"
        require_keys(gobj, ["name", "pauli", "qubits", "theta"],
                     Set(["name", "pauli", "qubits", "theta"]), what)
        pauli = getstr(gobj, "pauli", what)
        isempty(pauli) && fail("$what field \"pauli\" must be non-empty")
        syms = Symbol[]
        for ch in pauli
            haskey(PAULI_SYMBOL, ch) ||
                fail("$what field \"pauli\" must contain only X, Y, Z (identity positions are " *
                     "expressed by omission); got \"$pauli\"")
            push!(syms, PAULI_SYMBOL[ch])
        end
        qs = getqubits(gobj, what, n_qubits, length(syms))
        theta = getfloat(gobj, "theta", what)
        return PauliRotation(syms, qs .+ 1, theta)

    elseif name == "depolarize"
        require_keys(gobj, ["name", "qubits", "p"], Set(["name", "qubits", "p"]), what)
        q = getqubits(gobj, what, n_qubits, 1)[1]
        p = getfloat(gobj, "p", what)
        # paulistrings scales X, Y, Z by 1 - 4p/3; jl's DepolarizingNoise
        # scales them by 1 - lambda  =>  lambda = 4p/3  (probes.jl P6).
        lambda = 4 * p / 3
        0 <= lambda <= 1 ||
            fail("$what: p=$p maps to jl noise strength lambda=4p/3=$lambda, outside [0, 1]")
        return DepolarizingNoise(q + 1, lambda)

    elseif name == "dephase"
        require_keys(gobj, ["name", "qubits", "p"], Set(["name", "qubits", "p"]), what)
        q = getqubits(gobj, what, n_qubits, 1)[1]
        p = getfloat(gobj, "p", what)
        # paulistrings scales X, Y by 1 - 2p; jl's DephasingNoise (== PauliZNoise)
        # scales them by 1 - lambda  =>  lambda = 2p  (probes.jl P6).
        lambda = 2 * p
        0 <= lambda <= 1 ||
            fail("$what: p=$p maps to jl noise strength lambda=2p=$lambda, outside [0, 1]")
        return DephasingNoise(q + 1, lambda)

    elseif name == "amplitude_damping"
        require_keys(gobj, ["name", "qubits", "gamma"], Set(["name", "qubits", "gamma"]), what)
        q = getqubits(gobj, what, n_qubits, 1)[1]
        gamma = getfloat(gobj, "gamma", what)
        return AmplitudeDampingNoise(q + 1, gamma)

    elseif name == "pauli_channel"
        require_keys(gobj, ["name", "qubits", "px", "py", "pz"],
                     Set(["name", "qubits", "px", "py", "pz"]), what)
        q = getqubits(gobj, what, n_qubits, 1)[1]
        px = getfloat(gobj, "px", what)
        py = getfloat(gobj, "py", what)
        pz = getfloat(gobj, "pz", what)
        (px >= 0 && py >= 0 && pz >= 0) || fail("$what: px, py, pz must be non-negative")
        px + py + pz <= 1 || fail("$what: px + py + pz must be <= 1, got $(px + py + pz)")
        return pauli_channel_gate(px, py, pz, q + 1)

    elseif name == "depolarize2"
        require_keys(gobj, ["name", "qubits", "p"], Set(["name", "qubits", "p"]), what)
        qs = getqubits(gobj, what, n_qubits, 2)
        p = getfloat(gobj, "p", what)
        0 <= p <= 1 || fail("$what: p must be in [0, 1], got $p")
        return depolarize2_gate(p, qs[1] + 1, qs[2] + 1)

    elseif name == "unitary_1q"
        require_keys(gobj, ["name", "qubits", "matrix"], Set(["name", "qubits", "matrix"]), what)
        q = getqubits(gobj, what, n_qubits, 1)[1]
        mat = getmatrix(gobj, what, 2)
        check_unitary(mat, what)
        return TransferMapGate(mat, [q + 1])

    elseif name == "unitary_2q"
        require_keys(gobj, ["name", "qubits", "matrix"], Set(["name", "qubits", "matrix"]), what)
        qs = getqubits(gobj, what, n_qubits, 2)
        mat = getmatrix(gobj, what, 4)
        check_unitary(mat, what)
        # probes.jl P7: TransferMapGate takes qinds[1] as the FIRST (most
        # significant) tensor factor of the matrix, so a matrix acting on
        # |q0 q1> maps to qinds = [q0+1, q1+1] verbatim.
        return TransferMapGate(mat, [qs[1] + 1, qs[2] + 1])
    end

    fail("unknown gate name \"$name\". Schema v1 vocabulary: h s x y z cnot cz swap rz rx ry " *
         "pauli_rotation depolarize dephase amplitude_damping pauli_channel depolarize2 " *
         "unitary_1q unitary_2q")
end

function check_unitary(mat, what)
    dev = maximum(abs.(mat' * mat - one(mat)))
    dev < 1e-10 || fail("$what matrix is not unitary (max |U'U - I| = $dev)")
    return nothing
end

# ---------------------------------------------------------------------------
# Observable
# ---------------------------------------------------------------------------

const CHAR_SYMBOL = Dict('I' => :I, 'X' => :X, 'Y' => :Y, 'Z' => :Z)

"""
    build_observable(obs_obj, n_qubits) -> (PauliSum, coeff_type_name)

Build the observable from full-length Pauli-string keys with the Hermitian-Y
convention. Real coefficients produce a `Float64` sum (jl's natural, and
fastest, coefficient type); any non-zero imaginary part promotes the whole sum
to `ComplexF64`.
"""
function build_observable(obs_obj, n_qubits::Int)
    isa(obs_obj, JSON3.Object) || fail("\"observable\" must be a JSON object of {pauli_string: coeff}")
    isempty(keys(obs_obj)) && fail("\"observable\" must contain at least one term")

    keysyms = Vector{Vector{Symbol}}()
    coeffs = ComplexF64[]
    for k in keys(obs_obj)
        label = String(k)
        length(label) == n_qubits ||
            fail("observable key \"$label\" has length $(length(label)), expected n_qubits=$n_qubits")
        syms = Symbol[]
        for ch in label
            haskey(CHAR_SYMBOL, ch) ||
                fail("observable key \"$label\" contains \"$ch\"; only I, X, Y, Z are allowed")
            push!(syms, CHAR_SYMBOL[ch])
        end
        push!(keysyms, syms)
        push!(coeffs, getcomplex(obs_obj[k], "observable value for \"$label\""))
    end

    all_real = all(iszero(imag(c)) for c in coeffs)
    CT = all_real ? Float64 : ComplexF64
    psum = PauliSum(CT, n_qubits)
    for (syms, c) in zip(keysyms, coeffs)
        # `symboltoint` maps symbol position 1 to qubit 1 (probes.jl P1), i.e.
        # the same left-to-right order the schema uses with qubit 0 first.
        add!(psum, symboltoint(syms), convert(CT, all_real ? real(c) : c))
    end
    return psum, string(CT)
end

# ---------------------------------------------------------------------------
# Expectation against the requested product state
# ---------------------------------------------------------------------------

"""
    expectation(psum, state) -> (value, method)

`state` follows the schema: the uniform names "x+" / "y+" / "z+", or a
per-qubit label string. jl provides |0...0> (`overlapwithzero`), |+...+>
(`overlapwithplus`) and computational basis states
(`overlapwithcomputational`); "y+" and any label containing +, -, r or l have
no jl counterpart and are hard errors.
"""
function expectation(psum, state::String)
    lowered = lowercase(state)
    if lowered == "z+"
        return overlapwithzero(psum), "overlapwithzero"
    elseif lowered == "x+"
        return overlapwithplus(psum), "overlapwithplus"
    elseif lowered == "y+"
        fail("run.state=\"y+\" has no PauliPropagation.jl counterpart: stateoverlap.jl provides " *
             "|0>, |+> and computational basis states only (\"eval against |±i> not implemented\"). " *
             "Use \"x+\" or \"z+\", or an all-0/1 label string.")
    end

    # per-qubit label string (A4): only 0/1 are expressible in jl.
    n = nqubits(psum)
    length(state) == n ||
        fail("run.state=\"$state\" is neither a uniform name (x+/y+/z+) nor a label string of " *
             "length n_qubits=$n")
    onebits = Int[]
    for (i, ch) in enumerate(state)
        if ch == '1'
            push!(onebits, i)
        elseif ch != '0'
            fail("run.state=\"$state\" contains \"$ch\": PauliPropagation.jl can only contract " *
                 "against computational basis states, so per-qubit labels must be 0 or 1 " *
                 "(+, -, r, l have no counterpart).")
        end
    end
    return overlapwithcomputational(psum, onebits), "overlapwithcomputational"
end

# ---------------------------------------------------------------------------
# Task parsing
# ---------------------------------------------------------------------------

struct Task
    n_qubits::Int
    circuit::Vector{Gate}
    observable::Any
    coeff_type::String
    max_weight::Float64
    min_abs_coeff::Float64
    truncation_given::Dict{String,Any}
    direction::String
    threads::Int
    state::Union{String,Nothing}
end

function parse_task(path::String)
    isfile(path) || fail("task file not found: $path")
    obj = JSON3.read(read(path, String))
    isa(obj, JSON3.Object) || fail("task file must contain a JSON object at the top level")

    require_keys(obj,
                 ["version", "n_qubits", "circuit", "run"],
                 Set(["version", "n_qubits", "circuit", "observable", "truncation", "run"]),
                 "task")

    version = getint(obj, "version", "task")
    version == SCHEMA_VERSION ||
        fail("task \"version\" must be $SCHEMA_VERSION, got $version")

    n_qubits = getint(obj, "n_qubits", "task")
    n_qubits > 0 || fail("task \"n_qubits\" must be positive, got $n_qubits")

    # --- run block (direction is required; never defaulted) ---
    run = require_keys(obj[:run], ["direction"], Set(["direction", "threads", "state"]), "task.run")
    direction = getstr(run, "direction", "task.run")
    direction in ("forward", "heisenberg") ||
        fail("task.run \"direction\" must be \"forward\" or \"heisenberg\", got \"$direction\"")
    threads = haskey(run, :threads) ? getint(run, "threads", "task.run") : 1
    threads >= 1 || fail("task.run \"threads\" must be >= 1, got $threads")
    state = haskey(run, :state) ? getstr(run, "state", "task.run") : nothing

    # --- circuit ---
    circ_obj = obj[:circuit]
    isa(circ_obj, JSON3.Object) || fail("task \"circuit\" must be a JSON object")
    if haskey(circ_obj, :stim_file)
        fail("task circuit uses \"stim_file\", which this runner cannot read: " *
             "PauliPropagation.jl has no Stim parser. Convert to an inline \"gates\" list on " *
             "the Python side first (schema v1 allows both spellings).")
    end
    require_keys(circ_obj, ["gates"], Set(["gates"]), "task.circuit")
    gates_arr = circ_obj[:gates]
    isa(gates_arr, JSON3.Array) || fail("task.circuit \"gates\" must be a list")
    circuit = Vector{Gate}(undef, length(gates_arr))
    for (i, gobj) in enumerate(gates_arr)
        circuit[i] = build_gate(gobj, n_qubits, direction)
    end

    # --- observable ---
    haskey(obj, :observable) ||
        fail("task is missing \"observable\": the runner has no Stim path, so the observable " *
             "must be given inline")
    observable, coeff_type = build_observable(obj[:observable], n_qubits)

    # --- truncation ---
    # jl's own default is min_abs_coeff=1e-10; an absent key must mean NO
    # truncation, so pass 0.0 explicitly.
    max_weight = Inf
    min_abs_coeff = 0.0
    given = Dict{String,Any}()
    if haskey(obj, :truncation)
        tr = require_keys(obj[:truncation], String[], Set(["max_weight", "min_abs_coeff"]),
                          "task.truncation")
        if haskey(tr, :max_weight)
            w = getint(tr, "max_weight", "task.truncation")
            w >= 0 || fail("task.truncation \"max_weight\" must be >= 0, got $w")
            max_weight = Float64(w)
            given["max_weight"] = w
        end
        if haskey(tr, :min_abs_coeff)
            eps = getfloat(tr, "min_abs_coeff", "task.truncation")
            eps >= 0 || fail("task.truncation \"min_abs_coeff\" must be >= 0, got $eps")
            min_abs_coeff = eps
            given["min_abs_coeff"] = eps
        end
    end

    return Task(n_qubits, circuit, observable, coeff_type, max_weight, min_abs_coeff,
                given, direction, threads, state)
end

# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

envflag(name, default) = get(ENV, name, default ? "1" : "0") in ("1", "true", "yes")
envint(name, default) = haskey(ENV, name) ? parse(Int, ENV[name]) : default

function make_input(task::Task, backend::String)
    psum = deepcopy(task.observable)
    if backend == "vector"
        return VectorPauliSum(psum)
    elseif backend == "dict"
        return psum
    end
    fail("PP_BACKEND must be \"dict\" or \"vector\", got \"$backend\"")
end

function run_once(task::Task, backend::String, fused::Bool)
    psum = make_input(task, backend)
    # `fused` is only understood by the experimental Performance overload for
    # PauliRotation; it is passed only when asked for, so the default path sees
    # exactly the kwargs a plain jl user would pass.
    extra = fused ? (; fused = true) : (;)
    return propagate(task.circuit, psum;
                     heisenberg = (task.direction == "heisenberg"),
                     min_abs_coeff = task.min_abs_coeff,
                     max_weight = task.max_weight,
                     thread = task.threads > 1,
                     extra...)
end

function main(args)
    if isempty(args) || args[1] in ("-h", "--help")
        println(stderr, "usage: julia --project=benchmarks/julia runner.jl <task.json> [-o out.json]")
        return isempty(args) ? 2 : 0
    end
    task_path = args[1]
    out_path = nothing
    i = 2
    while i <= length(args)
        if args[i] in ("-o", "--output")
            i + 1 <= length(args) || fail("$(args[i]) needs a path")
            out_path = args[i+1]
            i += 2
        else
            fail("unexpected argument \"$(args[i])\"")
        end
    end

    task = parse_task(task_path)

    backend = get(ENV, "PP_BACKEND", "dict")
    warm_repeats = envint("PP_WARM_REPEATS", 3)
    warm_repeats >= 0 || fail("PP_WARM_REPEATS must be >= 0")
    want_layer_counts = envflag("PP_LAYER_COUNTS", true)
    fused = envflag("PP_FUSED", false)

    if Threads.nthreads() != task.threads
        println(stderr, "runner.jl: warning: task.run.threads=$(task.threads) but " *
                        "Threads.nthreads()=$(Threads.nthreads()); pass `-t $(task.threads)` " *
                        "to julia to match.")
    end

    input_terms = length(task.observable)

    # Cold run: includes JIT compilation of the specialized propagate path.
    cold = @timed run_once(task, backend, fused)
    psum_out = cold.value

    # Warm runs: same work, compiled. `wall_warm_s` is the minimum.
    warm_times = Float64[]
    warm_gc = Float64[]
    warm_bytes = Int[]
    for _ in 1:warm_repeats
        t = @timed run_once(task, backend, fused)
        push!(warm_times, t.time)
        push!(warm_gc, t.gctime)
        push!(warm_bytes, t.bytes)
        psum_out = t.value
    end

    # Per-gate term counts: a separate, untimed propagation, because
    # `@countpaulis` installs a global counter that locks once per gate.
    layer_counts = nothing
    if want_layer_counts
        counts = @countpaulis run_once(task, backend, fused)
        layer_counts = collect(counts)
    end

    # Optional term-by-term dump, for parity debugging on small cases. Keys are
    # full-length Pauli strings with qubit 0 leftmost -- the same spelling the
    # task file's observable uses.
    emit_limit = envint("PP_EMIT_TERMS", 0)
    terms_out = nothing
    if emit_limit > 0 && length(psum_out) <= emit_limit
        terms_out = Dict{String,Vector{Float64}}()
        for (p, c) in zip(paulis(psum_out), coefficients(psum_out))
            cc = ComplexF64(c)
            terms_out[inttostring(p, task.n_qubits)] = [real(cc), imag(cc)]
        end
    end

    exp_value = nothing
    exp_method = nothing
    if task.state !== nothing
        v, m = expectation(psum_out, task.state)
        exp_value = ComplexF64(v)
        exp_method = m
    end

    notes = String[
        "min_abs_coeff boundary: PauliPropagation.jl drops |c| < eps (strict), paulistrings " *
        "drops |c| <= eps (inclusive); the two disagree only when a coefficient lands exactly " *
        "on the threshold.",
        "max_weight boundary: both engines keep weight == max_weight.",
        "exact-zero coefficients survive in PauliPropagation.jl but are dropped by the " *
        "paulistrings merge, so an exactly-cancelling circuit can differ by term count.",
        "truncation is applied after every gate (no layer concept), so one gate object must " *
        "equal one paulistrings channel.",
    ]
    if layer_counts !== nothing
        push!(notes,
              "per_layer_terms is in APPLICATION order, which for direction=\"heisenberg\" is " *
              "the reverse of the order gates appear in the task file.")
    end
    if fused
        push!(notes, "PP_FUSED=1: the experimental fused rotation kernel truncates during gate " *
                     "application; term-count parity with paulistrings is not established for it.")
    end

    result = (
        runner = "benchmarks/julia/runner.jl",
        schema_version = SCHEMA_VERSION,
        task_file = abspath(task_path),
        versions = (
            julia = string(VERSION),
            PauliPropagation = string(pkgversion(PauliPropagation)),
            JSON3 = string(pkgversion(JSON3)),
        ),
        task = (
            n_qubits = task.n_qubits,
            n_gates = length(task.circuit),
            direction = task.direction,
            truncation = task.truncation_given,
            requested_threads = task.threads,
            state = task.state,
        ),
        config = (
            backend = backend,
            fused = fused,
            warm_repeats = warm_repeats,
            julia_threads = Threads.nthreads(),
            coeff_type = task.coeff_type,
            min_abs_coeff_passed = task.min_abs_coeff,
            max_weight_passed = isinf(task.max_weight) ? nothing : Int(task.max_weight),
        ),
        result = (
            expectation = exp_value === nothing ? nothing :
                          (re = real(exp_value), im = imag(exp_value)),
            expectation_method = exp_method,
            input_terms = input_terms,
            final_terms = length(psum_out),
            per_layer_terms = layer_counts,
            peak_terms = layer_counts === nothing ? nothing :
                         maximum(vcat(input_terms, layer_counts)),
            terms = terms_out,
        ),
        timing = (
            wall_cold_s = cold.time,
            wall_warm_s = isempty(warm_times) ? nothing : minimum(warm_times),
            wall_warm_all_s = warm_times,
            gc_warm_s = isempty(warm_gc) ? nothing : minimum(warm_gc),
            bytes_warm = isempty(warm_bytes) ? nothing : minimum(warm_bytes),
        ),
        host = (
            hostname = gethostname(),
            cpu = length(Sys.cpu_info()) > 0 ? Sys.cpu_info()[1].model : "unknown",
            ncores = length(Sys.cpu_info()),
        ),
        notes = notes,
    )

    payload = JSON3.write(result)
    if out_path === nothing
        println(payload)
    else
        open(out_path, "w") do io
            println(io, payload)
        end
        println(stderr, "runner.jl: wrote $out_path")
    end
    return 0
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    code = try
        main(ARGS)
    catch e
        if isa(e, TaskError)
            println(stderr, "runner.jl: ", e.msg)
            1
        else
            rethrow()
        end
    end
    exit(code)
end
