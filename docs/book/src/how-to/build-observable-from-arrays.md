# Build an observable from raw arrays

```python
import numpy as np
from paulistrings import PauliSum

observable = PauliSum.from_strings({"XI": 1.0, "ZI": 0.5}, num_qubits=2)

x = observable.x_array()
z = observable.z_array()
c = observable.coefficients_array()

rebuilt = PauliSum.from_arrays(x, z, c, num_qubits=2)
```

`x_array()`/`z_array()`/`coefficients_array()` are zero-copy numpy views of the symplectic bit columns and coefficients — the shape an analysis pass would want to work in directly, rather than going through strings.
`from_arrays` is the exact inverse: it hands the same columns back to the engine as a `PauliSum`, so an analysis pass can round-trip without re-parsing strings.
See [PauliSum reference](../reference/pauli-sum.md) for the array dtypes and layout.
