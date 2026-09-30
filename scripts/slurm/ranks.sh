# Sourced by mpi-ranks.sbatch and ole-mpi.sbatch, after `cd` to the repo root.
# One MPI rank per NUMA domain, rounded down to a power of two -- a partition is named by
# log2(P) GF(2) rows -- spread evenly over the allocation's nodes.
numa=$(ls -d /sys/devices/system/node/node* | wc -l)
cpus_per_numa=$(( SLURM_CPUS_ON_NODE / numa ))
nodes=${SLURM_JOB_NUM_NODES:-1}
total=$(( nodes * numa ))
ranks=1
while [ $(( ranks * 2 )) -le "$total" ]; do ranks=$(( ranks * 2 )); done
per_node=$(( ranks / nodes ))
[ "$per_node" -lt 1 ] && { per_node=1; ranks=$nodes; }
echo "numa domains per node: $numa, cpus per domain: $cpus_per_numa, nodes: $nodes"
echo "== $ranks ranks ($per_node per node), from $total available domains"
