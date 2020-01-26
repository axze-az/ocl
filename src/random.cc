#include "ocl/random.h"

ocl::dvec<std::uint32_t>
ocl::rand::vgid(std::size_t s, be::data_ptr p)
{
    ocl::dvec<std::uint32_t> r(p, s);
    if (s) {
        const char* kname="get_gid";
        const char* ksrc=
            "void get_gid(ulong n, __global uint* d)\n"
            "{\n"
            "    ulong gid=get_global_id(0);\n"
            "    if (gid < n) {\n"
            "        d[gid]=gid;\n"
            "    }\n"
            "}";
        auto ck=custom_kernel<std::uint32_t>(kname, ksrc, r);
        execute_custom(ck, s, p);
    }
    return r;
}

ocl::rand::rand(std::size_t s)
    : _next(vgid(s, be::data::instance()))
{
}

ocl::rand::rand(std::size_t s, be::data_ptr p)
    : _next(vgid(s, p))
{
}

ocl::dvec<int32_t>
ocl::rand::next()
{
    const char* kname="rand_next";
    const char* ksrc=
        "void\n"
        "rand_next(ulong n, __global int* r, __global uint* s)\n"
        "{\n"
        "    ulong gid=get_global_id(0);\n"
        "    if (gid < n) {\n"
        "        uint next = s[gid];\n"
        "        next *= 1103515245;\n"
        "        next += 12345;\n"
        "        int result = (int)((next >> 16) & 2047);\n"
        "        next *= 1103515245;\n"
        "        next += 12345;\n"
        "        result <<= 10;\n"
        "        result ^= (int)((next >> 16) & 1023);\n"
        "        next *= 1103515245;\n"
        "        next += 12345;\n"
        "        result <<= 10;\n"
        "        result ^= (int) ((next >> 16) & 1023);\n"
        "        s[gid] = next;\n"
        "        r[gid] = result;\n"
        "    }\n"
        "}\n";
    auto ck=custom_kernel<std::int32_t>(kname, ksrc, _next);
    dvec<std::int32_t> r(ck);
    return r;
}

ocl::dvec<float>
ocl::rand::nextf()
{
    const char* kname="rand_nextf";
    const char* ksrc=
        "void\n"
        "rand_nextf(ulong n, __global float* r, __global uint* s)\n"
        "{\n"
        "    ulong gid=get_global_id(0);\n"
        "    if (gid < n) {\n"
        "        uint next = s[gid];\n"
        "        next *= 1103515245;\n"
        "        next += 12345;\n"
        "        int result = (int)((next >> 16) & 2047);\n"
        "        next *= 1103515245;\n"
        "        next += 12345;\n"
        "        result <<= 10;\n"
        "        result ^= (int)((next >> 16) & 1023);\n"
        "        next *= 1103515245;\n"
        "        next += 12345;\n"
        "        result <<= 10;\n"
        "        result ^= (int) ((next >> 16) & 1023);\n"
        "        result &= 0xffffff; \n"
        "        float fresult= convert_float_rtz(result)* 0x1p-24f;\n"
        "        s[gid] = next;\n"
        "        r[gid] = fresult;\n"
        "    }\n"
        "}\n";
    auto ck=custom_kernel<float>(kname, ksrc, _next);
    dvec<float> r(ck);
    return r;
}
