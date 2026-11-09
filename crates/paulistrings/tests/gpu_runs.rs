//! Device runs against the host: permutation runs walked in one pass and rotation runs over joined one-qubit tables agree with `propagate` to tolerance, on one device and in a group of device partitions.
//! Every case returns early without a device.

use num_complex::Complex64;
use paulistrings::gpu::{GpuError, GpuLayerOptions, GpuPartitionedSum, GpuScatterOptions};
use paulistrings::require_cuda;
use paulistrings::test_support::{
    assert_runs_agree, assert_terms_close, assert_terms_close_up_to_residues,
    coefficient_tolerance, collapsing_circuit, device_run_as, low_weight_sum, propagate_discarding,
    rand_sum_real, set_fault_plan, z0_sum, zz_rotation, FaultPlan, KeepAll,
};
use paulistrings::{
    propagate, ApproxTopN, BuildAccumulator, BuiltinTruncation, Circuit, Coefficient,
    CoefficientThreshold, CollapseSample, Direction, PartitionConfig, PartitionRowPolicy,
    PartitionRuntime, PartitionTrace, PauliAxis, PauliRotation, PauliString, PauliSum, Placement,
    ProductBasis, PtmChannel, ScatterRows, WeightCutoff,
};

/// A kicked-Ising step on a chain, `R_X(θ)` on every qubit and a quarter-turn `ZZ` on every bond, then `R_X(θ)` again on the first half: a rotation run, a Clifford run, a rotation run.
fn kicked_chain<const W: usize>(num_qubits: usize, theta: f64) -> Circuit<W> {
    let mut circuit = Circuit::<W>::new(num_qubits);
    for q in 0..num_qubits {
        circuit.push(PauliRotation::new(PauliString::<W>::x(q), theta));
    }
    for q in 0..num_qubits - 1 {
        circuit.push(zz_rotation::<W>(q, q + 1, std::f64::consts::FRAC_PI_2));
    }
    for q in 0..num_qubits / 2 {
        circuit.push(PauliRotation::new(PauliString::<W>::x(q), theta));
    }
    circuit
}

/// Weight-two strings with real coefficients, so the untruncated runs stay small.
fn sparse_input<const W: usize>(num_qubits: usize, seed: u64) -> PauliSum<W> {
    let mut acc = BuildAccumulator::<W>::new(num_qubits);
    for (x, z, c) in low_weight_sum::<W>(60, num_qubits, 2, seed).iter() {
        acc.add_term(PauliString::<W> { x: *x, z: *z }, Complex64::new(c.re, 0.0));
    }
    acc.finalize()
}

/// Cliffords alone: Hadamards, CNOTs and quarter-turn `ZZ`s, one run per direction.
fn clifford_block<const W: usize>(num_qubits: usize) -> Circuit<W> {
    let mut circuit = Circuit::<W>::new(num_qubits);
    for q in (0..num_qubits).step_by(3) {
        circuit.push(PtmChannel::hadamard(q));
    }
    for q in 0..num_qubits - 1 {
        circuit.push(PtmChannel::cnot(q, q + 1));
        circuit.push(zz_rotation::<W>(q, (q + 5) % num_qubits, -std::f64::consts::FRAC_PI_2));
    }
    circuit
}

/// `circuit` on `sum` narrowed to `C` on the device, widened back, and how many layers it applied inside a run past the run's first.
fn device_as<const W: usize, C: Coefficient>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: &BuiltinTruncation,
    direction: Direction,
) -> (PauliSum<W>, usize) {
    let options = GpuLayerOptions::default();
    let (got, trace) = device_run_as::<W, C>(circuit, sum, policy, direction, options, true);
    let joined = trace.expect("tracing").layers.iter().filter(|l| l.nanos[0] == 0).count();
    (got, joined)
}

/// The device in `f64` and `c128` against the host `propagate`, both directions; returns the fewest layers any device run joined into a run.
fn agrees<const W: usize>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: impl Into<BuiltinTruncation>,
    name: &str,
) -> usize {
    let policy: BuiltinTruncation = policy.into();
    let mut joined = usize::MAX;
    for direction in [Direction::Forward, Direction::Heisenberg] {
        let want = propagate(circuit, sum.clone(), &policy, direction);
        assert!(want.len() > 1, "{name}: the fixture must keep terms");
        let (got, f64_joined) = device_as::<W, f64>(circuit, sum, &policy, direction);
        assert_terms_close_up_to_residues(
            &got,
            &want,
            1e3 * coefficient_tolerance::<f64>(),
            &format!("{name} f64 {direction:?}"),
        );
        let (got, c128_joined) = device_as::<W, Complex64>(circuit, sum, &policy, direction);
        assert_terms_close_up_to_residues(
            &got,
            &want,
            1e3 * coefficient_tolerance::<Complex64>(),
            &format!("{name} c128 {direction:?}"),
        );
        joined = joined.min(f64_joined).min(c128_joined);
    }
    joined
}

fn runs_agree<const W: usize>(num_qubits: usize, seed: u64) {
    let input = sparse_input::<W>(num_qubits, seed);
    let cliffords = clifford_block::<W>(num_qubits);
    let kicked = kicked_chain::<W>(num_qubits, 0.6);
    let mut rotations = Circuit::<W>::new(num_qubits);
    for q in 0..num_qubits {
        rotations.push(PauliRotation::new(PauliString::<W>::x(q), 0.6));
    }
    for (name, circuit) in
        [("clifford", &cliffords), ("kicked", &kicked), ("rotations", &rotations)]
    {
        let name = format!("{name} W={W}");
        let joined = agrees(circuit, &input, KeepAll, &format!("{name} keep"));
        assert!(joined > 0, "{name}: the device took no run");
        agrees(circuit, &input, CoefficientThreshold(2e-3), &format!("{name} threshold"));
        agrees(circuit, &input, WeightCutoff((num_qubits / 2) as _), &format!("{name} weight"));
    }
}

/// Strings with `Z` on all of qubits `0..8` and distinct elsewhere, under `R_X` on those eight: each grows 256-fold within its class, so the one device block outgrows its rows and the run falls back to its layers one by one.
fn crowded_classes<const W: usize>(num_qubits: usize, strings: usize) {
    let mut acc = BuildAccumulator::<W>::new(num_qubits);
    for i in 0..strings {
        let mut p = PauliString::<W>::z(0);
        for q in 1..8 {
            p.z[q / 64] |= 1 << (q % 64);
        }
        let q = 8 + i % (num_qubits - 8);
        p.x[q / 64] |= 1 << (q % 64);
        if i >= num_qubits - 8 {
            p.z[q / 64] |= 1 << (q % 64);
        }
        acc.add_term(p, Complex64::new(1.0 + 0.1 * i as f64, 0.0));
    }
    let input = acc.finalize();
    let mut circuit = Circuit::<W>::new(num_qubits);
    for q in 0..8 {
        circuit.push(PauliRotation::new(PauliString::<W>::x(q), 0.4 + 0.1 * q as f64));
    }
    let name = format!("crowded W={W}");
    agrees(&circuit, &input, CoefficientThreshold(1e-3), &name);
    agrees(&circuit, &input, CoefficientThreshold(1e-9), &name);
}

#[test]
fn crowded_rotation_classes_agree_with_the_host() {
    require_cuda!();
    crowded_classes::<1>(20, 12);
    crowded_classes::<2>(90, 24);
}

#[test]
fn device_runs_agree_with_the_host_w1() {
    require_cuda!();
    runs_agree::<1>(12, 0x5EED1);
}

#[test]
fn device_runs_agree_with_the_host_w2() {
    require_cuda!();
    runs_agree::<2>(90, 0x5EED2);
}

/// `sum` over `p` partitions of device 0, cut into equal contiguous qubit blocks, so every `R_X` is local and only a bond across blocks exchanges.
fn cut_split<const W: usize>(
    sum: &PauliSum<W>,
    p: usize,
    layer: GpuLayerOptions,
) -> GpuPartitionedSum<W> {
    let config = PartitionConfig {
        placement: Placement::Devices { devices: vec![0], per_device: p },
        bind_memory: false,
        threads_per_partition: None,
    };
    let per = sum.num_qubits() / p;
    let blocks = (0..p).map(|b| (b * per..(b + 1) * per).collect()).collect();
    let options = GpuScatterOptions {
        devices: PartitionRuntime::new(&config).expect("placement"),
        rows: ScatterRows::Policy(PartitionRowPolicy::Cut(blocks)),
        layer,
    };
    let mut split = GpuPartitionedSum::scatter_to_devices_with(sum, options).expect("scatter");
    split.enable_trace();
    split
}

/// The layers each rank applied inside a run past its first, which must be the same on every rank, and never a remote layer.
fn joined_layers(trace: &PartitionTrace, what: &str) -> Vec<bool> {
    trace
        .layers
        .iter()
        .map(|layer| {
            let joined = layer.nanos[0] == 0;
            assert!(
                layer.nanos.iter().all(|&nanos| (nanos == 0) == joined),
                "{what}: ranks took different runs at layer {}",
                layer.application_index,
            );
            assert!(!(joined && layer.remote_deltas > 0), "{what}: a run took a remote layer");
            joined
        })
        .collect()
}

/// [`kicked_chain`] over a cut group of `p` partitions against the host, both directions; returns, per direction, the trace of the device run.
fn group_agrees<const W: usize>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy_of: &dyn Fn(Direction) -> BuiltinTruncation,
    p: usize,
    what: &str,
) -> Vec<PartitionTrace> {
    let mut traces = Vec::new();
    for direction in [Direction::Forward, Direction::Heisenberg] {
        let policy = policy_of(direction);
        let what = format!("{what} P={p} {direction:?}");
        let (want, discarded) = propagate_discarding(circuit, sum.clone(), &policy, direction);
        let mut split = cut_split(sum, p, GpuLayerOptions::default());
        split.propagate(circuit, policy.clone(), direction).expect("device propagate");
        let got = split.gather().expect("gather");
        assert_runs_agree(
            policy.clone(),
            discarded,
            &got,
            &want,
            1e3 * coefficient_tolerance::<f64>(),
            &what,
        );
        let trace = split.take_trace().expect("tracing");
        joined_layers(&trace, &what);
        traces.push(trace);
    }
    traces
}

/// Rotation layers of [`kicked_chain`] are its first `num_qubits` and its last `num_qubits / 2`.
fn is_rotation(num_qubits: usize, circuit_index: u32) -> bool {
    let index = circuit_index as usize;
    index < num_qubits || index >= 2 * num_qubits - 1
}

/// The largest `ApproxTopN` cap that still cuts a `direction` run, so the lightest cut, which the operator check can compare.
fn light_cut<const W: usize>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    direction: Direction,
) -> usize {
    let cuts =
        |n: usize| propagate_discarding(circuit, sum.clone(), &ApproxTopN(n), direction).1 > 0.0;
    let (mut cutting, mut keeping) = (1usize, 1usize << 40);
    while keeping - cutting > 1 {
        let middle = cutting + (keeping - cutting) / 2;
        if cuts(middle) {
            cutting = middle;
        } else {
            keeping = middle;
        }
    }
    cutting
}

fn group_runs_agree<const W: usize>(num_qubits: usize, seed: u64) {
    let input = sparse_input::<W>(num_qubits, seed);
    let circuit = kicked_chain::<W>(num_qubits, 0.6);
    let caps = [Direction::Forward, Direction::Heisenberg].map(|d| light_cut(&circuit, &input, d));
    let cutting = move |direction: Direction| {
        BuiltinTruncation::from(ApproxTopN(caps[usize::from(direction == Direction::Heisenberg)]))
    };
    type PolicyOf<'a> = Box<dyn Fn(Direction) -> BuiltinTruncation + 'a>;
    let policies: [(&str, PolicyOf); 4] = [
        ("keep", Box::new(|_| KeepAll.into())),
        ("threshold", Box::new(|_| CoefficientThreshold(2e-3).into())),
        ("approx-topn above every bound", Box::new(|_| ApproxTopN(1 << 40).into())),
        ("cutting approx-topn", Box::new(cutting)),
    ];
    for p in [2usize, 4] {
        for (name, policy) in &policies {
            let what = format!("{name} W={W}");
            for trace in group_agrees(&circuit, &input, policy, p, &what) {
                let joined = joined_layers(&trace, &what);
                let of = |rotation: bool| {
                    trace.layers.iter().zip(&joined).filter(move |(layer, &joined)| {
                        joined && is_rotation(num_qubits, layer.circuit_index) == rotation
                    })
                };
                if !name.starts_with("cutting") {
                    assert!(of(true).count() > 0, "{what} P={p}: no rotation run");
                    assert!(of(false).count() > 0, "{what} P={p}: no Clifford run");
                }
                if name.starts_with("approx-topn above") && num_qubits <= 16 {
                    // The bound agreed at the call's start, times every fanout, stays below `n`, so only the bucket counts of the remote layers and that agreement are collective.
                    assert_eq!(
                        trace.total_collectives(),
                        trace.remote_layers() as u64 + 1,
                        "{what} P={p}: collectives",
                    );
                }
            }
        }
    }
}

#[test]
fn device_group_runs_agree_with_the_host_w1() {
    require_cuda!();
    group_runs_agree::<1>(16, 0x6A1);
}

#[test]
fn device_group_runs_agree_with_the_host_w2() {
    require_cuda!();
    group_runs_agree::<2>(96, 0x6A2);
}

/// Rotations growing a dense sum refine the partitions, in the run or at the crossing bond after it, which runs at one agreed count.
fn group_run_refines<const W: usize>(num_qubits: usize) {
    let input = rand_sum_real::<W>(3_000, num_qubits, 0x6A3);
    let half = num_qubits / 2;
    let mut circuit = Circuit::<W>::new(num_qubits);
    for q in 0..8 {
        circuit.push(PauliRotation::new(PauliString::<W>::x(q), 0.3 + 0.05 * q as f64));
    }
    circuit.push(zz_rotation::<W>(half - 1, half, 0.4));
    let small = GpuLayerOptions { records_per_block: 256, ..GpuLayerOptions::default() };
    let want = propagate(&circuit, input.clone(), &KeepAll, Direction::Forward);
    let mut split = cut_split(&input, 2, small);
    split.propagate(&circuit, KeepAll, Direction::Forward).expect("device propagate");
    let what = format!("refining run W={W}");
    assert_terms_close(
        &split.gather().unwrap(),
        &want,
        1e3 * coefficient_tolerance::<f64>(),
        &what,
    );
    let trace = split.take_trace().unwrap();
    let joined = joined_layers(&trace, &what);
    assert!(joined[1..8].iter().all(|&j| j), "{what}: the rotations ran as one run");
    let crossing = trace.layers.last().unwrap();
    assert!(crossing.remote_deltas > 0, "{what}: the bond crosses");
    assert!(crossing.bits.iter().all(|&b| b == crossing.bits[0]), "{what}: {:?}", crossing.bits);
    let refined =
        |layer: &paulistrings::PartitionLayerRecord| layer.refine_passes.iter().any(|&r| r > 0);
    assert!(refined(&trace.layers[0]) || refined(crossing), "{what}: the call refined");
}

#[test]
fn a_device_group_refines_around_a_run_and_meets_a_remote_layer() {
    require_cuda!();
    group_run_refines::<1>(16);
    group_run_refines::<2>(96);
}

/// A failure injected inside a run on one rank, under a layer pass whose collectives follow the run, fails the call and poisons the split while the partners finish.
fn group_run_fails<const W: usize>(num_qubits: usize) {
    let input = sparse_input::<W>(num_qubits, 0x6A4);
    let circuit = kicked_chain::<W>(num_qubits, 0.6);
    for policy in [ApproxTopN(1 << 40), ApproxTopN(40)] {
        let mut split = cut_split(&input, 2, GpuLayerOptions::default());
        set_fault_plan(&mut split, 1, FaultPlan { at_layer: Some(3), ..FaultPlan::default() });
        let first = split.propagate(&circuit, policy, Direction::Forward);
        assert!(matches!(first, Err(GpuError::Unsupported(_))), "W={W} {policy:?}: {first:?}");
        let again = split.propagate(&circuit, policy, Direction::Forward);
        assert!(
            matches!(again, Err(GpuError::Poisoned { rank: 1, .. })),
            "W={W} {policy:?}: {again:?}"
        );
        // The crossing quarter-turn bonds scatter their received rows; a receive failing there fails the same way.
        let mut split = cut_split(&input, 2, GpuLayerOptions::default());
        set_fault_plan(&mut split, 1, FaultPlan { after_chunk: Some(0), ..FaultPlan::default() });
        let first = split.propagate(&circuit, policy, Direction::Forward);
        assert!(matches!(first, Err(GpuError::OutOfMemory { .. })), "W={W} {policy:?}: {first:?}");
        let trace = split.take_trace().unwrap();
        let failing =
            trace.layers.iter().find(|layer| layer.remote_deltas > 0).expect("a remote layer");
        assert!(failing.device.iter().flatten().any(|d| d.permuted), "W={W}: {failing:?}");
        let again = split.propagate(&circuit, policy, Direction::Forward);
        assert!(matches!(again, Err(GpuError::Poisoned { rank: 1, .. })), "W={W}: {again:?}");
    }
}

#[test]
fn a_failure_inside_a_device_group_run_poisons_the_split() {
    require_cuda!();
    group_run_fails::<1>(16);
    group_run_fails::<2>(96);
}

/// Above its cache a group's `CollapseSample` follows its runs: a cache no run reaches is the untruncated run with no pass collective, and a small one averages to the exact readout within four standard errors.
#[test]
fn collapse_sample_in_a_device_group_takes_runs_and_stays_unbiased() {
    require_cuda!();
    let (circuit, input) = (collapsing_circuit(), z0_sum());
    let want = propagate(&circuit, input.clone(), &KeepAll, Direction::Heisenberg);
    let mut split = cut_split(&input, 2, GpuLayerOptions::default());
    let roomy = CollapseSample::new(1 << 60, 1 << 20, 3);
    split.propagate(&circuit, roomy, Direction::Heisenberg).expect("device propagate");
    assert_terms_close(&split.gather().unwrap(), &want, 1e-10, "roomy collapse");
    let trace = split.take_trace().unwrap();
    assert!(joined_layers(&trace, "roomy collapse").contains(&true), "no run");
    assert_eq!(trace.total_collectives(), trace.remote_layers() as u64 + 1, "roomy collectives");

    let zero = ProductBasis::uniform(PauliAxis::Z);
    let exact = want.expectation_product(&zero).re;
    let seeds = 200u64;
    let (mut sum, mut squares, mut runs) = (0.0f64, 0.0f64, 0usize);
    for seed in 0..seeds {
        let policy = std::sync::Arc::new(CollapseSample::new(6, 4, seed));
        let mut split = cut_split(&input, 2, GpuLayerOptions::default());
        split.propagate(&circuit, policy.clone(), Direction::Heisenberg).expect("propagate");
        let trace = split.take_trace().unwrap();
        runs += joined_layers(&trace, "collapse").iter().filter(|&&j| j).count();
        let readout = split.gather().unwrap().expectation_product(&zero).re;
        sum += readout;
        squares += readout * readout;
    }
    let n = seeds as f64;
    let mean = sum / n;
    let error = ((squares / n - mean * mean) / (n - 1.0)).sqrt();
    assert!(runs > 0, "no seed took a run");
    assert!(
        (mean - exact).abs() < 4.0 * error,
        "mean {mean} vs exact {exact}, standard error {error}"
    );
}

/// A traced run reads exact counts at its end, so every layer's output is the next layer's input, a run's last row included.
#[test]
fn a_traced_device_run_ends_on_exact_counts() {
    require_cuda!();
    let input = sparse_input::<1>(16, 0x6A5);
    let circuit = kicked_chain::<1>(16, 0.6);
    let policy = BuiltinTruncation::from(KeepAll);
    let options = GpuLayerOptions::default();
    let (_, trace) =
        device_run_as::<1, f64>(&circuit, &input, &policy, Direction::Heisenberg, options, true);
    let layers = trace.expect("tracing").layers;
    assert!(layers.iter().any(|layer| layer.nanos[0] == 0), "the device took no run");
    for (layer, next) in layers.iter().zip(&layers[1..]) {
        assert_eq!(layer.terms_out, next.terms_in, "layer {}", layer.application_index);
    }
}
