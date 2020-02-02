#include "ocl/expr_kernel.h"
#include <iterator>

#if 0
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
#endif

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

ocl::be::kernel_handle
ocl::impl::
compile(const std::string& s, const std::string& k_name,
        be::data_ptr& b)
{
    using namespace impl;
    if (b->debug() != 0) {
        std::ostringstream st;
        st << std::this_thread::get_id() << ": "
           << k_name << ": --- source code ------------------\n"
           << s;
        be::data::debug_print(st.str());
    }
    be::program pgm=be::program::create_with_source(s, b->dcq().c());
    try {
        // pgm=program::build_with_source(ss, d->c(), "-cl-std=clc++");
        // pgm=program::build_with_source(ss, d->c(), "-cl-std=CL1.1");
        pgm.build( "-cl-std=CL1.1 "
                   "-cl-mad-enable "
                   "-cl-fp32-correctly-rounded-divide-sqrt");
    }
    catch (const be::error& e) {
        std::cerr << "error info: " << e.what() << '\n';
        std::cerr << pgm.build_log() << std::endl;
        throw;
    }
    be::kernel k(pgm, k_name);
    if (b->debug() != 0) {
        std::ostringstream st;
        st << std::this_thread::get_id() << ": "
           << k_name << ": --- compiled with success --------\n";
        be::data::debug_print(st.str());
    }
    be::kernel_handle pkl(pgm, k);
    return pkl;
}

void
ocl::impl::missing_backend_data()
{
    throw std::runtime_error("missing backend data");
}

void
ocl::impl::
print_arg_buffer_info(be::kernel& k, size_t ab_size)
{
    std::string kn=k.name();
    std::ostringstream st;
    st << std::this_thread::get_id() << ": "
        << kn << ": binding argument buffer of size "
        << ab_size << '\n';
    be::data::debug_print(st.str());
}

void
ocl::impl::
print_cached_kernel_info(be::kernel_cache::iterator f,
                         const be::kernel_key& kk)
{
    std::string kn=f->second.k().name();
    std::ostringstream s;
    s << std::this_thread::get_id() << ": "
        << kn << ": using cached kernel " << kk
        << '\n';
    be::data::debug_print(s.str());
}

