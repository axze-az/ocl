#include "ocl/be/types.h"

std::vector<ocl::cl::platform>
ocl::cl::system::
platforms()
{
    cl_uint count = 0;
    cl_int err=clGetPlatformIDs(0, 0, &count);
    error::throw_on(err, __FILE__, __LINE__);
    std::vector<platform> platforms;
    if(count > 0) {
        cl_platform_id* platform_ids=
            static_cast<cl_platform_id*>(alloca(count*sizeof(cl_platform_id)));
        err=clGetPlatformIDs(count, &platform_ids[0], 0);
        error::throw_on(err, __FILE__, __LINE__);
        for(cl_uint i=0; i<count; ++i){
            platforms.push_back(platform(platform_ids[i]));
        }
    }
    return platforms;
}

std::vector<ocl::cl::device>
ocl::cl::system::
devices()
{
    std::vector<device> vd;
    auto vp = platforms();
    for (const auto& p : vp) {
        const auto vpd = p.devices();
        vd.insert(vd.end(), vpd.begin(), vpd.end());
    }
    return vd;
}

