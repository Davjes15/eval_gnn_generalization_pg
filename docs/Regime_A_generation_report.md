# Regime A (fixed topology, k=0) -- generation report

Generated from `/home/ubuntu/repos/eval_gnn_generalization_pg/data_a`, compared against `/home/ubuntu/repos/eval_gnn_generalization_pg/full_run/data`.

## IEEE24
### Splits
| split | samples | distinct demand snapshots | k values |
|---|---|---|---|
| train | 800 | 800 | [0] |
| val | 100 | 100 | [0] |
| test | 100 | 100 | [0] |


Demand-snapshot overlap between splits: `{'train n val': 0, 'train n test': 0, 'val n test': 0}`

Buses: 24. Directed edges (2E) observed: `{76: 1000}`

### Target ranges
| regime | quantity | min | max | mean | std |
|---|---|---|---|---|---|
| data | p_mw | -660 | 1633 | -4.929 | 289.6 |
| data_a | p_mw | -660 | 1625 | -4.522 | 288.6 |
| data | q_mvar | -860.4 | 167.9 | -15.46 | 98.32 |
| data_a | q_mvar | -610.4 | 128.1 | -11.86 | 92.65 |
| data | va_degree | -19.94 | 90.55 | 20.09 | 15 |
| data_a | va_degree | -12.72 | 56.49 | 18.29 | 13.19 |
| data | vm_pu | 0.8305 | 1.051 | 1.011 | 0.03054 |
| data_a | vm_pu | 0.9363 | 1.05 | 1.012 | 0.0284 |


## IEEE39
### Splits
| split | samples | distinct demand snapshots | k values |
|---|---|---|---|
| train | 800 | 800 | [0] |
| val | 100 | 100 | [0] |
| test | 100 | 100 | [0] |


Demand-snapshot overlap between splits: `{'train n val': 0, 'train n test': 0, 'val n test': 0}`

Buses: 39. Directed edges (2E) observed: `{92: 1000}`

### Target ranges
| regime | quantity | min | max | mean | std |
|---|---|---|---|---|---|
| data | p_mw | -830 | 2111 | -3.448 | 402.6 |
| data_a | p_mw | -830 | 2110 | -3.297 | 409.4 |
| data | q_mvar | -1803 | 271.2 | -51.35 | 229.2 |
| data_a | q_mvar | -1800 | 184 | -51.05 | 232.8 |
| data | va_degree | -12.2 | 125.5 | 50.62 | 22.82 |
| data_a | va_degree | -12.2 | 99.6 | 50.87 | 21.89 |
| data | vm_pu | 0.8001 | 1.082 | 0.9851 | 0.05772 |
| data_a | vm_pu | 0.8006 | 1.064 | 0.9869 | 0.05673 |


## IEEE118
### Splits
| split | samples | distinct demand snapshots | k values |
|---|---|---|---|
| train | 800 | 800 | [0] |
| val | 100 | 100 | [0] |
| test | 100 | 100 | [0] |


Demand-snapshot overlap between splits: `{'train n val': 0, 'train n test': 0, 'val n test': 0}`

Buses: 118. Directed edges (2E) observed: `{368: 1000}`

### Target ranges
| regime | quantity | min | max | mean | std |
|---|---|---|---|---|---|
| data | p_mw | -5979 | 295.6 | -9.674 | 430.5 |
| data_a | p_mw | -6081 | 296.3 | -9.657 | 431 |
| data | q_mvar | -3334 | 102.7 | -48.07 | 204 |
| data_a | q_mvar | -3526 | 102.3 | -47.75 | 203.3 |
| data | va_degree | -180 | 179.9 | -91.27 | 33.11 |
| data_a | va_degree | -179.9 | 180 | -90.7 | 32.83 |
| data | vm_pu | 0.8057 | 1.052 | 0.9826 | 0.02681 |
| data_a | vm_pu | 0.8303 | 1.052 | 0.9827 | 0.0266 |


## UK
### Splits
| split | samples | distinct demand snapshots | k values |
|---|---|---|---|
| train | 800 | 800 | [0] |
| val | 100 | 100 | [0] |
| test | 100 | 100 | [0] |


Demand-snapshot overlap between splits: `{'train n val': 0, 'train n test': 0, 'val n test': 0}`

Buses: 29. Directed edges (2E) observed: `{180: 1000}`

### Target ranges
| regime | quantity | min | max | mean | std |
|---|---|---|---|---|---|
| data | p_mw | -1.134e+04 | 3.759e+04 | -94.99 | 5421 |
| data_a | p_mw | -1.133e+04 | 3.724e+04 | -93.25 | 5435 |
| data | q_mvar | -1.954e+04 | 766 | -992.3 | 1759 |
| data_a | q_mvar | -1.331e+04 | 766 | -971.6 | 1718 |
| data | va_degree | -178.9 | 178.7 | 59.16 | 36.93 |
| data_a | va_degree | -6.29 | 167.9 | 58.16 | 36.14 |
| data | vm_pu | 0.9244 | 1 | 0.9979 | 0.005221 |
| data_a | vm_pu | 0.9703 | 1 | 0.998 | 0.004933 |

