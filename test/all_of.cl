
__kernel void all_of(ulong n,
                     __global ulong* dcnt,
                     __global TYPE* ds,
                     __local TYPE* t)
{
    ulong gid= get_global_id(0);
    uint lid= get_local_id(0);
    uint lsz= get_local_size(0);
    // copy s[gid] into t[lid]
    TYPE v= gid < n ? ds[gid] : 1;
    t[lid]=v != 0 ? 1: 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    // loop over t[0, lsz)
    for (uint stride=lsz>>1; stride>0; stride >>=1) {
#if 1
        uint pos= lid + stride;
        TYPE vi= (lid < stride) & (pos < lsz) ? t[pos] : 1;
        t[lid] &= vi;
#else
        if (lid < stride) {
            uint pos=lid + stride;
            int vi= pos < lsz ? t[pos] : 1;
            t[lid] &= vi;
        }
#endif
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        ulong grp_id=get_group_id(0);
        ds[grp_id]=t[0];
    }
    if (gid == 0) {
        ulong grps=get_num_groups(0);
        dcnt[0]=grps;
    }
}

/*
 * local variables:
 * mode: c++
 * end:
 */
