# target-cpu=native vs default, paired per run (1 thread)

| variant | pairs | delta % per pair | same sign | median delta % | verdict |
|---|---|---|---|---|---|
| bucketed | 10 | -9.4, -9.7, -9.2, -9.4, -7.1, -9.0, -9.1, -9.6, -9.0, -9.3 | yes | -9.2 | consistent |
| naive | 10 | -3.0, -2.3, -1.9, -0.9, -0.5, -3.9, -4.7, -4.1, -1.5, -3.2 | yes | -2.7 | consistent |
