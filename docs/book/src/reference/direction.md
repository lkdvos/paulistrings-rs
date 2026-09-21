# Direction semantics

`direction=` selects which conjugation `propagate`/`propagate_with_stats` performs.

| `direction` | Picture | What the engine does |
|---|---|---|
| `"heisenberg"` | `U† O U` | walks the channel list **in reverse**, applying each channel's adjoint |
| `"forward"` (default when `None`) | `U O U†` | walks the channel list **as written**, applying each channel |

## Push order

Push order interacts with direction. Under `"heisenberg"` the engine iterates channels in reverse, so pushing `ZZ` rotations before `X` rotations gives a step operator `U = U_X · U_ZZ`, and Heisenberg evolution computes `U_ZZ† U_X† O U_X U_ZZ`.
Build the circuit in the order you want applied under `"forward"`; the direction flag handles the reversal.

See [Choose a propagation direction](../how-to/choose-propagation-direction.md) for guidance on which picture answers which question.
