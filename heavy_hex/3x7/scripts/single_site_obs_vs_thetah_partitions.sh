#!/bin/bash
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --constraint=icelake
#SBATCH --mem=0
#SBATCH --job-name=partitions_3x7
#SBATCH --time=01:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=...

CONFIG="heavy_hex/3x7/3x7_partitions.json"
OUTPUT_DIR="/.../heavy_hex/3x7/data"

mkdir -p $OUTPUT_DIR

module load python
source ~/venvs/paulistrings/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
unset RAYON_NUM_THREADS

lscpu | grep "NUMA node"
srun --cpus-per-task=$SLURM_CPUS_ON_NODE python3 -c "import paulistrings; print(len(paulistrings.numa_nodes()), 'NUMA nodes visible')"

srun --cpus-per-task=$SLURM_CPUS_ON_NODE python3 heavy_hex/3x7/scripts/single_site_obs_vs_thetah_partitions.py --config $CONFIG --output-dir $OUTPUT_DIR