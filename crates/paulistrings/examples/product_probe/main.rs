use paulistrings::{
    BuildAccumulator, CoefficientThreshold, Gf2Hash, PartitionConfig, PartitionRows,
    PartitionRuntime, PartitionedSum, PauliString, PauliSum, Placement, ScatterOptions,
    ScatterRows,
};
use std::time::Instant;

const L: usize = 40;

fn heisenberg() -> PauliSum<1, f64> {
    let mut acc = BuildAccumulator::<1, f64>::new(L);
    for j in 0..L - 1 {
        for f in [PauliString::<1>::x, PauliString::<1>::y, PauliString::<1>::z] {
            let mut p = f(j);
            p.mul_assign(&f(j + 1));
            acc.add_term(p, 1.0);
        }
    }
    for j in 0..L {
        acc.add_term(PauliString::<1>::z(j), 0.3);
    }
    acc.finalize()
}

fn main() {
    let steps: usize = std::env::args().nth(1).map_or(13, |s| s.parse().unwrap());
    let h = heisenberg();
    let mut acc = BuildAccumulator::<1, f64>::new(L);
    acc.add_term(PauliString::<1>::z(L / 2), 1.0);
    let mut o = acc.finalize();
    for _ in 0..steps {
        let mut next = h.liouvillian(&o);
        next.scale(1.0 / next.overlap(&next).sqrt());
        next.retain(|_, _, c| c.abs() > 1e-12);
        o = next;
    }
    if std::env::args().nth(2).as_deref() == Some("layout") && std::env::var("CASE").is_ok() {
        layout(&h, &o);
        return;
    }
    let max = o.iter().map(|(_, _, c)| c.abs()).fold(0.0, f64::max);
    for cut in [1e-14, 1e-12, 1e-10] {
        println!(
            "terms below {cut:e} x max: {}",
            o.iter().filter(|(_, _, c)| c.abs() < cut * max).count()
        );
    }
    let (mut overlapping, mut anticommuting) = (0u64, 0u64);
    for (ox, oz, _) in o.iter() {
        for (hx, hz, _) in h.iter() {
            overlapping += (((ox[0] | oz[0]) & (hx[0] | hz[0])) != 0) as u64;
            anticommuting += (((ox[0] & hz[0]) ^ (oz[0] & hx[0])).count_ones() & 1) as u64;
        }
    }
    let pairs = (o.len() * h.len()) as f64;
    let weight: f64 =
        o.iter().map(|(x, z, _)| (x[0] | z[0]).count_ones() as f64).sum::<f64>() / o.len() as f64;
    println!(
        "len(O) {} mean weight {weight:.1} pairs {pairs:.3e} overlapping {:.3} anticommuting {:.3}",
        o.len(),
        overlapping as f64 / pairs,
        anticommuting as f64 / pairs
    );
    if std::env::args().nth(2).as_deref() == Some("layout") {
        layout(&h, &o);
        return;
    }
    if std::env::args().nth(2).as_deref() == Some("split") {
        split(&h, &o);
        return;
    }
    for _ in 0..3 {
        let t = Instant::now();
        let out = h.liouvillian(&o);
        let dt = t.elapsed().as_secs_f64();
        println!("liouvillian: out {} {:.1} ms {:.2} ns/pair {:.2} ns/anticommuting pair, generated/unique {:.1}", out.len(), dt * 1e3, dt / pairs * 1e9, dt / anticommuting as f64 * 1e9, anticommuting as f64 / out.len() as f64);
    }
}

fn split(h: &PauliSum<1, f64>, o: &PauliSum<1, f64>) {
    let policy = CoefficientThreshold(1e-12);
    for _ in 0..2 {
        let t = Instant::now();
        let want = h.liouvillian_with(o, &policy);
        println!("one process: {} terms {:.1} ms", want.len(), t.elapsed().as_secs_f64() * 1e3);
    }
    let block = |lo: usize, hi: usize| -> [u64; 1] { [((1u64 << hi) - 1) & !((1u64 << lo) - 1)] };
    // Blocks of 10 qubits labelled 0, 1, 3, 2 along the chain, so neighbours differ in one row (Gray order).
    let gray_rows = |x_rows: bool| {
        let r0 = [block(10, 20)[0] | block(30, 40)[0]];
        let r1 = block(20, 40);
        if x_rows {
            PartitionRows::<1>::from_rows(L, vec![r0, r1], vec![[0], [0]])
        } else {
            PartitionRows::<1>::from_rows(L, vec![[0], [0]], vec![r0, r1])
        }
    };
    let cases: Vec<(&str, usize, PartitionRows<1>)> = vec![
        ("P=2 seeded", 2, PartitionRows::<1>::from_seed(L, 1, 7)),
        ("P=2 x cut", 2, PartitionRows::<1>::from_rows(L, vec![block(20, 40)], vec![[0]])),
        ("P=4 seeded", 4, PartitionRows::<1>::from_seed(L, 2, 7)),
        ("P=4 z cut", 4, gray_rows(false)),
        ("P=4 x cut", 4, gray_rows(true)),
    ];
    for (name, partitions, rows) in cases {
        let config = PartitionConfig {
            placement: Placement::Unpinned { partitions },
            bind_memory: false,
            threads_per_partition: Some(32 / partitions),
        };
        let options = ScatterOptions {
            runtime: PartitionRuntime::new(&config).unwrap(),
            rows: ScatterRows::Explicit(rows),
        };
        let split = PartitionedSum::scatter_with(o.clone(), options);
        let mut by_delta: std::collections::BTreeMap<u32, Vec<PauliString<1>>> = Default::default();
        for (x, z, _) in h.iter() {
            by_delta
                .entry(split.rows().partition_of(x, z))
                .or_default()
                .push(PauliString { x: *x, z: *z });
        }
        let mut sent = 0usize;
        let mut partners = 0usize;
        for r in 0..partitions {
            for (_, terms) in by_delta.iter().filter(|(d, _)| **d != 0) {
                partners += 1;
                sent += split
                    .partition(r)
                    .iter()
                    .filter(|(x, z, _)| {
                        let key = PauliString { x: **x, z: **z };
                        terms.iter().any(|g| !g.commutes_with(&key))
                    })
                    .count();
            }
        }
        let shares: Vec<usize> = (0..partitions).map(|r| split.partition(r).len()).collect();
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            let _ = split.liouvillian_with(h, &policy);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        println!("{name}: partners/rank {}, sent {:.2} x operand, max share {:.2} x mean, best {best:.1} ms", partners / partitions, sent as f64 / o.len() as f64, *shares.iter().max().unwrap() as f64 * partitions as f64 / o.len() as f64);
    }
}

/// Coordinate rows `x_q`, `z_q` for qubits from the centre outward, `rows` of them.
fn coordinate_hash(bits: u8, rows: usize) -> Gf2Hash<1> {
    let mut order = vec![L / 2];
    for d in 1..L {
        if L / 2 >= d {
            order.push(L / 2 - d);
        }
        if L / 2 + d < L {
            order.push(L / 2 + d);
        }
    }
    let (mut rx, mut rz) = (Vec::new(), Vec::new());
    for &q in order.iter() {
        for coord in 0..2 {
            if rx.len() == rows {
                break;
            }
            rx.push(if coord == 0 { [1u64 << q] } else { [0] });
            rz.push(if coord == 1 { [1u64 << q] } else { [0] });
        }
    }
    Gf2Hash::from_rows(L, bits, 0x9E37_79B9_7F4A_7C15, rx, rz)
}

fn layout(h: &PauliSum<1, f64>, o: &PauliSum<1, f64>) {
    if std::env::var_os("PROBE_ABLATE").is_some() {
        unsafe { std::env::set_var("ABLATE", "1") };
    }
    let policy = CoefficientThreshold(1e-12);
    let bits = o.hash().bits();
    let cases: Vec<(&str, PauliSum<1, f64>)> = vec![
        ("dense rows", o.clone()),
        ("coordinate rows", o.clone().with_hash(coordinate_hash(bits, 31))),
    ];
    for (name, operand) in cases {
        if std::env::var("CASE").is_ok_and(|c| !name.starts_with(&c)) {
            continue;
        }
        let hash = operand.hash();
        let deltas: std::collections::BTreeSet<u32> =
            h.iter().map(|(x, z, _)| hash.bucket_of(x, z)).collect();
        let zero = h.iter().filter(|(x, z, _)| hash.bucket_of(x, z) == 0).count();
        let lens = operand.bucket_lengths();
        let mean = operand.len() as f64 / lens.len() as f64;
        let max = *lens.iter().max().unwrap() as f64;
        let empty = lens.iter().filter(|&&l| l == 0).count();
        let mut best = f64::MAX;
        let mut out_len = 0;
        for _ in 0..3 {
            let t = Instant::now();
            let out = h.liouvillian_with(&operand, &policy);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
            out_len = out.len();
        }
        println!("{name}: {} buckets, {} distinct deltas, {zero} terms at delta 0, max bucket {:.1} x mean, {empty} empty, out {out_len}, best {best:.1} ms", lens.len(), deltas.len(), max / mean);
    }
}
