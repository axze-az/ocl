#include "ocl/expr_custom.h"


std::string
ocl::impl::
decl_buffer_args_local_mem(const std::string_view& tname,
                           unsigned& arg_num,
                           bool ro)
{
    static_cast<void>(ro);
    std::ostringstream s;
    s << "    __local "
      << tname
      << "* arg" << arg_num << ",\n";
    ++arg_num;
    return s.str();
}

std::string
ocl::impl::
decl_buffer_args_local_mem(const char* tname,
                            unsigned& arg_num,
                            bool ro)
{
    return decl_buffer_args_local_mem(std::string_view(tname), arg_num, ro);
}

std::string
ocl::impl::
decl_buffer_args_local_mem(const std::string& tname,
                           unsigned& arg_num,
                           bool ro)
{
    return decl_buffer_args_local_mem(std::string_view(tname), arg_num, ro);
}

void
ocl::impl::
bind_buffer_args_local_mem(const std::string_view& tname,
                           size_t bytes,
                           size_t elements,
                           size_t wg_size,
                           unsigned& buf_num,
                           be::kernel& k)
{
    // bind p.bytes() * wgs bytes local memory to k
    if (be::data::instance()->debug() != 0) {
        std::string kn=k.name();
        std::ostringstream s;
        s << std::this_thread::get_id() << ": "
          << kn << ": " << " binding local_mem_per_workitem<"
          << tname << "> with "
          << elements
          << " elements and "
          << wg_size << " workitems "
          << "to arg " << buf_num << '\n';
        be::data::debug_print(s.str());
    }
    k.set_arg(buf_num, bytes, static_cast<const void*>(0));
    ++buf_num;
}

void
ocl::impl::
bind_buffer_args_local_mem(const char* tname,
                           size_t bytes,
                           size_t elements,
                           size_t wg_size,
                           unsigned& buf_num,
                           be::kernel& k)
{
    return bind_buffer_args_local_mem(std::string_view(tname),
                                      bytes, elements, wg_size, buf_num, k);
}

void
ocl::impl::
bind_buffer_args_local_mem(const std::string& tname,
                           size_t bytes,
                           size_t elements,
                           size_t wg_size,
                           unsigned& buf_num,
                           be::kernel& k)
{
    return bind_buffer_args_local_mem(std::string_view(tname),
                                      bytes, elements, wg_size, buf_num, k);
}


std::string
ocl::impl::concat_args_local_mem(var_counters& c)
{
    std::ostringstream s;
    s << "arg" << c._buf_num;
    ++c._var_num;
    ++c._buf_num;
    return s.str();
}
