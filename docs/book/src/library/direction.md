# Direction semantics

`direction=` selects which conjugation `propagate`/`propagate_with_stats` performs.

| `direction` | Picture | What the engine does |
|---|---|---|
| `"heisenberg"` | `U† O U` | walks the channel list **in reverse**, applying each channel's adjoint |
| `"forward"` (default when `None`) | `U O U†` | walks the channel list **as written**, applying each channel |

`None` selects `"forward"`, which is **not** the direction most examples on this site use; pass `direction` explicitly.

## What the state label then means

`direction` also fixes which state a following [`expectation(state)`](measurement.md) call is read against, because `expectation` evaluates `⟨s|A|s⟩` for whatever sum `A` it is given.

| `direction` | `evolved.expectation(s)` equals | so `s` is |
|---|---|---|
| `"heisenberg"` | `⟨s\|U†OU\|s⟩`, i.e. `O`'s expectation in `U\|s⟩` | the **input** state, the one the circuit acts on |
| `"forward"` | `⟨s\|UOU†\|s⟩`, i.e. `O`'s expectation in `U†\|s⟩` | the state at the circuit's **output** end |

Swapping the two is silent: both give a number of the right magnitude and type, and neither raises.
[Direction](../manual/propagation/direction.md) has the same circuit and observable run both ways, with the two different numbers.

## Push order

Push order interacts with direction. Under `"heisenberg"` the engine iterates channels in reverse, so pushing `ZZ` rotations before `X` rotations gives a step operator `U = U_X · U_ZZ`, and Heisenberg evolution computes `U_ZZ† U_X† O U_X U_ZZ`.
Build the circuit in the order you want applied under `"forward"`; the direction flag handles the reversal.

See [Direction](../manual/propagation/direction.md) for guidance on which picture answers which question.
