
kernel
void all_of(ulong n, 
            __global int* d,
            __global const int* s,
            __local int* t)
{
    uint lid= get_local_id(0);
    uint gid= get_global_id(0);
    uint lsz= get_local_size(0);
    // copy s[gid] into t[lid]
    int v=0;
    if (gid < n) {
        v=s[gid]
    }
    t[lid]=v;
    barrier(CLK_LOCAL_MEM_FENCE);
    // loop over t[0, lsz)
    int sv=1;
    if (lid == 0) {
        uint grp_id=get_group_id(0);
        for (uint i=0; i<lsz; ++i) {
            sv &= t[i] != 0;
        }
        d[grp_id] = sv;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    // loop over d[0, grps]
    if (gid == 0) {
        uint grps=get_groups(0);
        sv = 1;
        for (uint i=0; i<grps; ++i) {
            sv &= d[i] != 0;
        }
        d[0] = sv;
    }
}
