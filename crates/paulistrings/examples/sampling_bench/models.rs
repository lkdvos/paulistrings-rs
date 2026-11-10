//! The two workloads: the kicked-Ising quench on a heavy-hex lattice and the transverse-field Ising quench on an open square lattice, as paulistrings-sim builds them.

use paulistrings::test_support::HEAVY_HEX_127_EDGES;
use paulistrings::{BuildAccumulator, Circuit, Coefficient, PauliRotation, PauliString, PauliSum};

/// A lattice model `H = -J Σ Z_i Z_j - h Σ X_i` on `bonds`, Trotterised at step `dt`.
pub(crate) struct IsingQuench {
    pub(crate) name: &'static str,
    pub(crate) num_qubits: usize,
    pub(crate) bonds: Vec<(usize, usize)>,
    /// Each qubit's lattice position `(y, x)`; a heavy-hex bridge sits at the odd `y` between its rows.
    pub(crate) positions: Vec<(i32, i32)>,
    pub(crate) j: f64,
    pub(crate) h: f64,
    pub(crate) dt: f64,
    /// 1: `R_X(-2h dt)` then `R_ZZ(-2J dt)`; 2: `R_X(-h dt)`, `R_ZZ(-2J dt)`, `R_X(-h dt)`.
    pub(crate) order: u8,
    pub(crate) observable: Observable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Observable {
    Site(usize),
    /// `(1/n) Σ_i Z_i`.
    Magnetization,
}

/// What a channel of the step circuit is, for the per-kind timing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GateKind {
    Rx,
    Rzz,
}

impl GateKind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            GateKind::Rx => "rx",
            GateKind::Rzz => "rzz",
        }
    }
}

impl IsingQuench {
    /// paulistrings-sim's `heavy_hex_kicked_ising` at `configs/heavy_hex_eagle.json`: `J dt = π/4` (Clifford `ZZ`), `h dt = π/8` (`θ_h = π/4`), first order, `Z` on `site`, by default `Z_62` (Kim et al. Fig. 4b) on the Eagle and the middle of the central row elsewhere.
    pub(crate) fn kicked_ising(rows: usize, width: usize, site: Option<usize>) -> Self {
        let (num_qubits, bonds, positions) = heavy_hex(rows, width);
        let row_of: Vec<usize> = positions.iter().map(|&(y, _)| y as usize / 2).collect();
        let eagle = (rows, width) == (7, 15);
        let site = site.or(eagle.then_some(62)).unwrap_or_else(|| {
            let middle: Vec<usize> = (0..num_qubits).filter(|&q| row_of[q] == rows / 2).collect();
            middle[middle.len() / 2]
        });
        Self {
            name: "kicked_ising",
            num_qubits,
            bonds,
            positions,
            j: std::f64::consts::FRAC_PI_4,
            h: std::f64::consts::FRAC_PI_8,
            dt: 1.0,
            order: 1,
            observable: Observable::Site(site),
        }
    }

    /// paulistrings-sim's `square_tfim` at `experiments/truncation_equivalence/square_tfim.json`: open boundaries, `J = 1`, `h = 3.04`, `dt = 0.1`, second order.
    pub(crate) fn square_tfim(lx: usize, ly: usize, observable: Observable) -> Self {
        Self {
            name: "tfim",
            num_qubits: lx * ly,
            bonds: square(lx, ly),
            positions: (0..lx * ly).map(|q| ((q / lx) as i32, (q % lx) as i32)).collect(),
            j: 1.0,
            h: 3.04,
            dt: 0.1,
            order: 2,
            observable,
        }
    }

    /// `steps` Trotter steps in circuit (Schrödinger) order, and each channel's kind.
    pub(crate) fn circuit<const W: usize>(&self, steps: usize) -> (Circuit<W>, Vec<GateKind>) {
        let mut circuit = Circuit::<W>::new(self.num_qubits);
        let mut kinds = Vec::new();
        let kick_angle = if self.order == 1 { -2.0 } else { -1.0 } * self.h * self.dt;
        for _ in 0..steps {
            self.push_kick(&mut circuit, &mut kinds, kick_angle);
            self.push_bonds(&mut circuit, &mut kinds);
            if self.order == 2 {
                self.push_kick(&mut circuit, &mut kinds, kick_angle);
            }
        }
        (circuit, kinds)
    }

    fn push_kick<const W: usize>(
        &self,
        circuit: &mut Circuit<W>,
        kinds: &mut Vec<GateKind>,
        angle: f64,
    ) {
        for q in 0..self.num_qubits {
            circuit.push(PauliRotation::new(PauliString::<W>::x(q), angle));
            kinds.push(GateKind::Rx);
        }
    }

    fn push_bonds<const W: usize>(&self, circuit: &mut Circuit<W>, kinds: &mut Vec<GateKind>) {
        for &(i, j) in &self.bonds {
            let mut generator = PauliString::<W>::z(i);
            generator.z[j / 64] |= 1u64 << (j % 64);
            circuit.push(PauliRotation::new(generator, -2.0 * self.j * self.dt));
            kinds.push(GateKind::Rzz);
        }
    }

    /// The observable at coefficient type `C`.
    pub(crate) fn observable_sum<const W: usize, C: Coefficient>(&self) -> PauliSum<W, C> {
        let n = self.num_qubits;
        let mut accumulator = BuildAccumulator::<W, C>::with_capacity(n, n);
        match self.observable {
            Observable::Site(q) => accumulator.add_term(PauliString::<W>::z(q), C::ONE),
            Observable::Magnetization => {
                for q in 0..n {
                    accumulator.add_term(PauliString::<W>::z(q), C::ONE * C::real(1.0 / n as f64));
                }
            }
        }
        accumulator.finalize()
    }

    pub(crate) fn observable_label(&self) -> String {
        match self.observable {
            Observable::Site(q) => format!("Z_{q}"),
            Observable::Magnetization => "m_z".to_string(),
        }
    }
}

/// A lattice as `(n, bonds, positions)`.
type Lattice = (usize, Vec<(usize, usize)>, Vec<(i32, i32)>);

/// paulistrings-sim's `heavy_hex(rows, width)`: IBM numbering, each row followed by its bridges, row `r` at `y = 2r`.
pub(crate) fn heavy_hex(rows: usize, width: usize) -> Lattice {
    assert!(rows % 2 == 1 && width % 4 == 3, "heavy_hex needs odd rows and width = 3 (mod 4)");
    let mut qubit = std::collections::BTreeMap::new();
    let mut bridges = Vec::new();
    let mut positions = Vec::new();
    for r in 0..rows {
        for c in 0..width {
            if (r, c) != (0, width - 1) && (r, c) != (rows - 1, 0) {
                qubit.insert((r, c), positions.len());
                positions.push((2 * r as i32, c as i32));
            }
        }
        if r + 1 < rows {
            for c in (2 * (r % 2)..width).step_by(4) {
                bridges.push(((r, c), positions.len()));
                positions.push((2 * r as i32 + 1, c as i32));
            }
        }
    }
    let mut bonds: Vec<(usize, usize)> = qubit
        .iter()
        .filter_map(|(&(r, c), &q)| qubit.get(&(r, c + 1)).map(|&next| (q, next)))
        .collect();
    for &((r, c), b) in &bridges {
        bonds.push((qubit[&(r, c)], b));
        bonds.push((b, qubit[&(r + 1, c)]));
    }
    // The 127-qubit Eagle layout must be the qiskit-derived coupling map the repo pins.
    if (rows, width) == (7, 15) {
        let mut sorted: Vec<(usize, usize)> =
            bonds.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
        sorted.sort_unstable();
        assert_eq!(sorted, HEAVY_HEX_127_EDGES, "heavy_hex(7, 15) is not the Eagle coupling map");
    }
    (positions.len(), bonds, positions)
}

/// paulistrings-sim's open `square(lx, ly)`: qubit `(x, y)` is `y·lx + x`.
pub(crate) fn square(lx: usize, ly: usize) -> Vec<(usize, usize)> {
    let mut bonds = Vec::new();
    for y in 0..ly {
        for x in 0..lx {
            let q = y * lx + x;
            if x + 1 < lx {
                bonds.push((q, q + 1));
            }
            if y + 1 < ly {
                bonds.push((q, q + lx));
            }
        }
    }
    bonds
}
