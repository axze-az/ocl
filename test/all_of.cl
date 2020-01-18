
kernel
void all_of(ulong n,
            __global int* d,
            __global int* dcnt,
            __global const int* s,
            __local int* t)
{
    uint lid= get_local_id(0);
    uint gid= get_global_id(0);
    uint lsz= get_local_size(0);
    // copy s[gid] into t[lid]
    int v= gid < n ? s[gid] : 1;
    t[lid]=v != 0 ? 1: 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    // loop over t[0, lsz)
    for (uint stride=lsz>>1; stride>0; stride >>=1) {
        if (lid < stride) {
            uint pos=lid + stride;
            int vi= pos < lsz ? t[pos] : 1;
            t[lid] &= vi;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        uint grp_id=get_group_id(0);
        d[grp_id]=t[0];
    }
    if (gid == 0) {
        uint grps=get_num_groups(0);
        dcnt[0]=grps;
    }
}
