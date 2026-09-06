# target-cpu=native vs default, paired per run (1 thread)

| variant | pairs | delta % per pair | same sign | median delta % | verdict |
|---|---|---|---|---|---|
| bucketed | 10 | -8.7, -7.9, -7.5, -8.3, -7.4, -8.3, -8.0, -7.5, -8.1, -8.7 | yes | -8.1 | consistent |
| naive | 10 | -3.6, -3.4, -1.4, -2.7, -0.2, -1.4, -2.3, -0.9, -1.9, -2.6 | yes | -2.1 | consistent |
