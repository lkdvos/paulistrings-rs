// Section 3 — Being smart: structured partitioning (outline ids 15-23).
// bit-flip + two-sorted-streams are prototype P3; rotation-bucket-pairs +
// coset-closure are prototype P4. Both pairs share one running example and
// keep the same integer positions/colors across their build stages.

#import "../theme/organic.typ": *
#import "../theme/extensions.typ": *
#import "../components.typ": *

// The checked bit-flip example (handoff, "P3. Bit-flip example"), in binary
// so the group-exchange story stays legible as bit patterns, not decimal
// integers. Colored by GROUP (each element's bit-1 value before the flip),
// not by position — so the same color travels with an element across every
// slide even after XOR moves it to the opposite bit-value slot. This is one
// source of truth for bit-flip, two-sorted-streams and whole-bucket-movement.
#let bits = 3
#let group-a = (0, 1, 4, 5) // bit 1 == 0
#let group-b = (2, 7) // bit 1 == 1
#let sorted-in = (0, 1, 2, 4, 5, 7)
#let sorted-colors = (pair-a-fill, pair-a-fill, pair-b-fill, pair-a-fill, pair-a-fill, pair-b-fill)
#let sorted-tcols = (pair-a-text, pair-a-text, pair-b-text, pair-a-text, pair-a-text, pair-b-text)
#let xor-out = sorted-in.map(v => xor-int(v, 2, bits))
#let merged-out = (0, 2, 3, 5, 6, 7)
#let merged-colors = (pair-b-fill, pair-a-fill, pair-a-fill, pair-b-fill, pair-a-fill, pair-a-fill)
#let merged-tcols = (pair-b-text, pair-a-text, pair-a-text, pair-b-text, pair-a-text, pair-a-text)

#let colored-row(values, colors, tcols) = stack(
  dir: ltr, spacing: 14pt,
  ..values.enumerate().map(((i, v)) => bit-chip(v, bits, fill: colors.at(i), tcol: tcols.at(i))),
)

#let render(mode) = {
  slide("memory-opportunities", mode, () => content-slide("Opportunities in memory use")[
    #bullets((
      [A smaller representation],
      [Fewer transfers],
      [More reuse while resident],
      [Access to more aggregate bandwidth],
    ))
    #v(baseline)
    #text(size: t-small, fill: ink-70, "These are related but distinct — the measurements will tell us how much comes from locality and how much from parallel execution. Capacity limits the largest calculation separately from all four.")
  ])

  // P3, stage 1/2 — flipping one bit: the trivial operation, the ordering
  // problem it creates. Color follows the element (its group), so stage 2
  // visibly shows the two colors trading bit-1 slots, not just new numbers.
  progressive-slide("bit-flip", mode, 2, stage => content-slide("Flipping one bit", align-center: true)[
    #set par(spacing: 0pt)
    #v(2 * baseline)
    #text(size: t-small, fill: ink-70, "Sorted input, colored by bit 1:")
    #v(baseline - 30pt)
    #colored-row(sorted-in, sorted-colors, sorted-tcols)
    #v(baseline)
    #reveal(stage >= 2)[
      #text(size: t-small, fill: ink-70, "XOR with 010:")
      #v(baseline - 30pt)
      #colored-row(xor-out, sorted-colors, sorted-tcols)
      #v(baseline)
      #text(size: t-small, fill: accent-700, "Each color moved wholesale to the other bit-1 slot. The list is no longer sorted — do we need to sort everything again?")
    ]
  ])

  // P3, stage 1/2 — the two subsequences that each stay sorted, then the merge.
  progressive-slide("two-sorted-streams", mode, 3, stage => content-slide("Two sorted streams")[
    #set par(spacing: 0pt)
    #data-table(
      header: ("Group (bit 1)", "Input subsequence", "After XOR with 010"),
      rows: (
        (
          box(fill: pair-a-fill, radius: 999pt, inset: (x: 16pt, y: 8pt), text(fill: pair-a-text, "0")),
          bit-chip-row(group-a, bits, fill: pair-a-fill, tcol: pair-a-text),
          bit-chip-row(group-a.map(v => xor-int(v, 2, bits)), bits, fill: pair-a-fill, tcol: pair-a-text),
        ),
        (
          box(fill: pair-b-fill, radius: 999pt, inset: (x: 16pt, y: 8pt), text(fill: pair-b-text, "1")),
          bit-chip-row(group-b, bits, fill: pair-b-fill, tcol: pair-b-text),
          bit-chip-row(group-b.map(v => xor-int(v, 2, bits)), bits, fill: pair-b-fill, tcol: pair-b-text),
        ),
      ),
      numeric-from: 3, cols: (auto, auto, auto),
    )
    #v(baseline)
    #reveal(stage >= 2)[
      #text(size: t-small, fill: ink-70, "Within each subsequence the operation adds or subtracts the same value, so its order survives.")
    ]
    #v(baseline - 20pt)
    #reveal(stage >= 3)[
      #text(size: t-body, fill: ink)[Merged result: #colored-row(merged-out, merged-colors, merged-tcols)]
      #v(baseline - 20pt)
      #text(size: t-kicker, fill: ink-55, "O(N) linear merge — not the O(N log(N/b)) cost of independently sorting balanced buckets.")
    ]
  ])

  slide("whole-bucket-movement", mode, () => content-slide("Whole-bucket movement")[
    #set par(spacing: 0pt)
    #grid(columns: (1fr, 1fr), column-gutter: 96pt,
      block(fill: pair-a-fill, radius: radius-lg, inset: 36pt, width: 100%)[
        #align(center)[#text(size: t-small, fill: pair-a-text, "bucket 0 (bit 1 = 0)")
        #v(baseline - 30pt)
        #bit-chip-row(group-a, bits, fill: bg, tcol: ink)]
      ],
      block(fill: pair-b-fill, radius: radius-lg, inset: 36pt, width: 100%)[
        #align(center)[#text(size: t-small, fill: pair-b-text, "bucket 1 (bit 1 = 1)")
        #v(baseline - 30pt)
        #bit-chip-row(group-b, bits, fill: bg, tcol: ink)]
      ],
    )
    #v(baseline)
    #text(size: t-small, fill: ink-70, "XOR with 010 either preserves each bucket or swaps the two buckets — we can reason about the movement of a whole collection without inspecting each label's destination.")
  ])

  slide("predictable-destination-rule", mode, () => content-slide("A predictable destination rule", align-center: true)[
    #math-setup[
      $ h(p plus.o g) = h(p) plus.o h(g) $
    ]
    #v(baseline)
    #text(size: t-body, fill: ink-70, "For a fixed generator, one bucket-level rule should work for every label. Ordinary hashing does not generally give this relationship — the bucket labels themselves must respect the XOR structure.")
  ])

  slide("gf2-linear-hash", mode, () => content-slide("A GF(2)-linear hash")[
    #set par(spacing: 0pt)
    #math-setup[
      $ h(p) = A p, quad A in bb(F)_2^(k times 2n), quad A(p plus.o g) = A p plus.o A g $
    ]
    #v(baseline)
    #text(size: t-body, fill: ink-70, "Each output bit is a parity of selected label bits. A rank-k matrix gives 2^k bucket IDs; buckets are cosets of ker A.")
    #v(baseline)
    #block(width: 100%, fill: surface, radius: radius-lg, inset: (x: 42pt, y: 30pt))[
      #text(size: t-small, fill: ink)[
        *Worked example*, $n=3$ qubits (6 label bits), $k=2$: row 1 reads the parity of ${x_1, z_2}$, row 2 the parity of ${x_3, z_1, z_3}$.
      ]
      #v(baseline - 30pt)
      #text(size: t-small, fill: ink)[
        For $p$ with $(x_1,x_2,x_3,z_1,z_2,z_3) = (1,0,1,0,1,0)$: row 1 $= x_1 plus.o z_2 = 1 plus.o 1 = 0$, row 2 $= x_3 plus.o z_1 plus.o z_3 = 1 plus.o 0 plus.o 0 = 1$ — bucket $h(p) = 10$.
      ]
    ]
    #v(baseline - 20pt)
    #text(size: t-kicker, fill: ink-55)[
      $ker A$ has dimension $2n-k=4$: the $2^k=4$ buckets are its cosets, each of size $2^4=16$ label vectors — equal coset sizes in the full vector space do not guarantee equal occupancy in the operator being propagated.
    ]
  ])

  // P4, stage 1 — the rotation example h(g) = 0101 on the full partition of
  // 4-bit bucket IDs into its 8 cosets, each colored to its own coset.
  progressive-slide("rotation-bucket-pairs", mode, 3, stage => content-slide("Rotations couple bucket pairs")[
    #set par(spacing: 0pt)
    #text(size: t-body, fill: ink, [Let $d = h(g) = 0101$ on 4-bit bucket IDs — the full partition into 8 pairs:])
    #v(baseline - 10pt)
    #let pairs4 = ((0, 5), (1, 4), (2, 7), (3, 6), (8, 13), (9, 12), (10, 15), (11, 14))
    #grid(
      columns: (1fr, 1fr, 1fr, 1fr), rows: (auto, auto), column-gutter: 20pt, row-gutter: 20pt,
      ..pairs4.enumerate().map(((i, pair)) => {
        let c = coset-palette.at(i)
        block(fill: c.fill, radius: radius-lg, inset: 16pt, width: 100%)[
          #align(center, stack(
            dir: ttb, spacing: 6pt,
            bit-chip(pair.at(0), 4, fill: bg, tcol: ink),
            text(fill: c.text, size: t-small, "\u{2195}"),
            bit-chip(pair.at(1), 4, fill: bg, tcol: ink),
          ))
        ]
      }),
    )
    #v(baseline - 10pt)
    #reveal(stage >= 2)[
      #text(size: t-small, fill: ink-70, "A term stays in its bucket, or contributes to its partner at b XOR d. Applying the offset twice returns to b — the bucket graph splits into disjoint pairs, one color each.")
    ]
    #v(baseline - 30pt)
    #reveal(stage >= 3)[
      #text(size: t-kicker, fill: ink-55, "d = 0000 would be the singleton case: a bucket pairs with itself.")
    ]
  ])

  // P4, stage 2 — the general construction, then explicitly a special case.
  progressive-slide("coset-closure", mode, 2, stage => content-slide("Closed cosets and in-place gate application")[
    #set par(spacing: 0pt)
    #math-setup[
      $ V = "span"(h(D)), quad "work units" = b + V $
    ]
    #v(baseline - 20pt)
    #code-block(size: t-kicker)[```rust
for gate in circuit {                    // Clifford or Rotation
    let cosets = closed_partition_cosets(gate);
    exchange_and_coordinate(&cosets);
    par_for coset in cosets {
        let local = gather(coset);
        let delta = apply_gate(gate, local);
        sort(&delta);
        merge_reduce(&mut buckets[coset], delta, cutoff);
    }
    if let Some(hist) = histogram_policy {
        finalize_total_count(hist);
    }
    synchronize();
}
```]
    #v(baseline - 20pt)
    #reveal(stage >= 2)[
      #text(size: t-kicker, fill: ink-55, "The rotation pairs above are the special case D = {g}. A task owns every partition its coset can exchange contributions with, so it reads and writes them in place; a Clifford can still split one partition across destinations while remaining inside a closed coset.")
    ]
  ])

  slide("sort-merge-work-unit", mode, () => content-slide("Sort-merge inside the work unit")[
    #set par(spacing: 0pt)
    #bullets((
      [#raw("fill_coset") swaps source bucket columns into scratch before using bucket slots as destinations],
      [The identity-delta stream stays sorted and skips sorting; other contributions join the rest stream, sorted before #raw("merge2_into")],
      [Dense identity plans can borrow original key columns instead of materializing them again],
      [A layer-level choice selects adaptive comparison or radix sorting],
    ))
    #v(baseline)
    #text(size: t-small, fill: ink-70, "Algebraic ownership makes local sort-merge and in-place updates possible; measured traffic and cache evidence — not code structure alone — explain the resulting gain.")
  ])
}
