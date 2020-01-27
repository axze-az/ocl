
void
rand_next(ulong n, __global int* r, __global uint* s)
{
    ulong gid=get_global_id(0);
    if (gid < n) {
        uint next = s[gid];
        next *= 1103515245;
        next += 12345;
        int result = (int)((next >> 16) & 2047);
        next *= 1103515245;
        next += 12345;
        result <<= 10;
        result ^= (int)((next >> 16) & 1023);
        next *= 1103515245;
        next += 12345;
        result <<= 10;
        result ^= (int) ((next >> 16) & 1023);
        s[gid] = next;
        r[gid] = result;
    }
}

void
rand_nextf(ulong n, __global float* r, __global uint* s)
{
    ulong gid=get_global_id(0);
    if (gid < n) {
        uint next = s[gid];
        next *= 1103515245;
        next += 12345;
        int result = (int)((next >> 16) & 2047);
        next *= 1103515245;
        next += 12345;
        result <<= 10;
        result ^= (int)((next >> 16) & 1023);
        next *= 1103515245;
        next += 12345;
        result <<= 10;
        result ^= (int) ((next >> 16) & 1023);
        result &= 0xffffff; 
        float fresult= convert_float_rtz(result)* 0x1p-24f;
        s[gid] = next;
        r[gid] = fresult;
    }
}

void
rand_nextf64(ulong n, __global float* r, __global uint* s)
{
    ulong gid=get_global_id(0);
    if (gid < n) {
        uint next = s[gid];
        next *= 1103515245;
        next += 12345;
        // 11
        long result = (int)((next >> 16) & 2047);
        // 22
        next *= 1103515245;
        next += 12345;
        result <<= 11;
        result ^= (int)((next >> 16) & 2047);
        // 33
        next *= 1103515245;
        next += 12345;
        result <<= 11;
        result ^= (int) ((next >> 16) & 2047);
        // 44
        next *= 1103515245;
        next += 12345;
        result <<= 11;
        result ^= (int) ((next >> 16) & 2047);
        // 55
        next *= 1103515245;
        next += 12345;
        result <<= 11;
        result ^= (int) ((next >> 16) & 2047);
        
        result &= 0xffffff; 
        float fresult= convert_float_rtz(result)* 0x1p-24f;
        s[gid] = next;
        r[gid] = fresult;
    }
}


void
update_histogram_float(ulong n,
                       __global int* h,
                       int entries,
                       float min_val,
                       float max_val,
                       float rec_interval,
                       __global const float* s)
{
    ulong gid=get_global_id(0);
    if (gid < n) {
        float v=s[gid];
        float offset = (v - min_val) * rec_interval * entries;
        uint o= offset;
        if (v > max_val) {
            o=entries + 1;
        }
        if (v < min_val) {
            o=entries;
        }
        atomic_add(h+o, 1);
    }
}
