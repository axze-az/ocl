#include "ocl/expr_kernel.h"
#include <iterator>

namespace ocl {
    namespace impl {

        void gen_byte_swaps(std::ostream& s);

        void
        insert_swap_if(std::ostream& s, bool dev_is_little_endian);
    }
}

void
ocl::impl::gen_byte_swaps(std::ostream& s)
{

    static
    const char* vec_types[]={
        "", "2", "3", "4", "8", "16"
    };
    auto e=std::end(vec_types);
    for (auto b=std::begin(vec_types); b!= e; ++b) {
        const char* p=*b;
        s << "uchar" << p <<
             " __swap_bytes_uchar"<< p << "(uchar" << p << " x)\n"
             "{\n"
             "    return x;\n"
             "}\n\n";

        s << "char" << p <<
             " __swap_bytes_char"<< p << "(char" << p << " x)\n"
             "{\n"
             "    return x;\n"
             "}\n\n";

        s << "ushort" << p <<
             " __swap_bytes_ushort" << p << "(ushort"<< p <<" x)\n"
             "{\n"
             "    ushort" << p << " l= x & ((ushort)0xff);\n"
             "    ushort" << p << " h= x >> 8;\n"
             "    ushort" << p << " r= h|l;\n"
             "    return r;\n"
             "}\n\n";

        s << "short" << p <<
             " __swap_bytes_short" << p << "(short"<< p <<" x)\n"
             "{\n"
             "    ushort" << p << " ux=as_ushort"<< p <<"(x);\n"
             "    ushort" << p << " ur=__swap_bytes_ushort"<< p <<"(ux);\n"
             "    short" << p << " r= as_short"<<p<<"(ux);\n"
             "    return r;\n"
             "}\n\n";

        s << "uint" << p <<
             " __swap_bytes_uint" << p << "(uint"<< p <<" x)\n"
             "{\n"
             "    uint" << p << " b03= x & ((uint)0x0000000ff);\n"
             "    uint" << p << " b12= x & ((uint)0x00000ff00);\n"
             "    uint" << p << " b21= x & ((uint)0x0000000ff);\n"
             "    uint" << p << " b30= x & ((uint)0x00000ff00);\n"
             "    uint" << p << " r= h|l;\n"
             "    return r;\n"
             "}\n\n";

    }
}

void
ocl::impl::
insert_swap_if(std::ostream& s, bool dev_is_little_endian)
{
    enum endian {
#ifdef _WIN32
        little = 0,
        big    = 1,
        native = little
#else
        little = __ORDER_LITTLE_ENDIAN__,
        big    = __ORDER_BIG_ENDIAN__,
        native = __BYTE_ORDER__
#endif
    };
    const endian device_endian= dev_is_little_endian ?
        endian::little : endian::big;
    const endian host_endian= endian::native;
    bool do_swap= device_endian != host_endian;

    if (do_swap == false) {
    } else {
    }
}

void
ocl::impl::insert_headers(std::ostream& s, size_t lmem_size)
{
    // fp64 extension
    s << "#if defined (cl_khr_fp64)\n"
         "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
         "#elif defined (cl_amd_fp64)\n"
         "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
         "#endif\n";
    // fp16
    s << "#if defined (cl_khr_fp16)\n"
         "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
         "#endif\n\n";
    // gen_byte_swaps(s);
    if (lmem_size==0) {
        s << "#define __arg_local __global\n\n";
    } else {
        s << "#define __arg_local __local\n\n";
    }
}

void
ocl::impl::missing_backend_data()
{
    throw std::runtime_error("missing backend data");
}
