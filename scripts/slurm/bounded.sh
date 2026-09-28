# Sourced by the Slurm templates: `bounded <secs> <cmd...>` runs one step under a hard bound, `leftover_gpu_processes` lists what a failed step left on the node's GPUs.
# A step that hangs (a rank stuck in a device wait, an `MPI_Abort` that never returns) would otherwise hold the allocation to the job's time limit.

# Run the command under `secs` seconds: TERM at the bound, KILL 30 s later.
# Returns the command's status, 124 when the bound ended it and 137 when it needed the KILL.
bounded() {
    local secs=$1 t0=$SECONDS status=0
    shift
    timeout --kill-after=30 "$secs" "$@" || status=$?
    if [ "$status" = 124 ] || [ "$status" = 137 ]; then
        echo "== step ended by its ${secs}s bound (status $status) after $((SECONDS - t0))s: $*" >&2
    fi
    return $status
}

# The compute processes still on the node's GPUs, from a step that shares the allocation with whatever is stuck (`--overlap`), itself bounded.
leftover_gpu_processes() {
    echo "== GPU processes left on ${SLURM_JOB_NODELIST:-this host}:"
    bounded 60 srun --overlap --ntasks-per-node=1 --ntasks="${SLURM_JOB_NUM_NODES:-1}" bash -c \
        'echo "-- $(hostname -s)"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true' || true
}
