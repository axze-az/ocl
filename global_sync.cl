
static void
global_sync(volatile __global int *flags)
{
        const size_t thread_id = get_local_id(0);
        const size_t workgroup_id = get_group_id(0);

        if (thread_id == 0) {
                flags[workgroup_id] = 1;
        }

        if (workgroup_id == 0) {
                if (thread_id < get_num_groups(0)) {
                        while (flags[thread_id] != 1) ;
                }
                barrier(CLK_GLOBAL_MEM_FENCE);

                if (thread_id < get_num_groups(0)) {
                        flags[thread_id] = 0;
                }
        }

        if (thread_id == 0) {
                while (flags[workgroup_id] != 0) ;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
}
