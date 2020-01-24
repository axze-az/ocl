#include "ocl/expr.h"
namespace ocl {
    namespace impl {
    }
}

std::string
ocl::impl::
decl_non_buffer_args_t(const std::string_view& tname,
                       size_t alignment,
                       unsigned& arg_num)
{
    std::ostringstream s;
    s << spaces(4) << tname
      << " _a" << arg_num
      << " __attribute__((aligned(" << alignment << ")));\n";
    ++arg_num;
    return s.str();
}

std::string
ocl::impl::
decl_non_buffer_args_t(const char* tname,
                       size_t alignment,
                       unsigned& arg_num)
{
    std::string_view tn(tname);
    return decl_non_buffer_args_t(tn,
                                  alignment,
                                  arg_num);
}

std::string
ocl::impl::
decl_non_buffer_args_t(const std::string& tname,
                       size_t alignment,
                       unsigned& arg_num)
{
    std::string_view tn(tname);
    return decl_non_buffer_args_t(tn,
                                  alignment,
                                  arg_num);
}

std::string
ocl::impl::
decl_non_buffer_args_array_ptr(const std::string_view& tname,
                               size_t n,
                               size_t alignment,
                               unsigned& arg_num)
{
    std::ostringstream s;
    s << spaces(4) << tname
      << " _a" << arg_num
      << "[" << n
      << "] __attribute__((aligned(" << alignment << ")));\n";
    ++arg_num;
    return s.str();
}

std::string
ocl::impl::
decl_non_buffer_args_array_ptr(const char* tname,
                               size_t n,
                               size_t alignment,
                               unsigned& arg_num)
{
    std::string_view tn(tname);
    return decl_non_buffer_args_array_ptr(tn,
                                          n,
                                          alignment,
                                          arg_num);
}

std::string
ocl::impl::
decl_non_buffer_args_array_ptr(const std::string& tname,
                               size_t n,
                               size_t alignment,
                               unsigned& arg_num)
{
    std::string_view tn(tname);
    return decl_non_buffer_args_array_ptr(tn,
                                          n,
                                          alignment,
                                          arg_num);
}

std::string
ocl::impl::
concat_args_t(var_counters& c)
{
    std::ostringstream s;
    s << "pa->_a" << c._scalar_num;
    ++c._var_num;
    ++c._scalar_num;
    return s.str();
}

std::string
ocl::impl::
eval_args_t(const std::string_view& tname,
            unsigned& arg_num,
            bool ro)
{
    static_cast<void>(ro);
    std::ostringstream s;
    s << spaces(4) ;
    s << tname
      << " arg"  << arg_num;
    ++arg_num;
    return s.str();
}

std::string
ocl::impl::
eval_args_t(const char* tname,
            unsigned& arg_num,
            bool ro)
{
    return eval_args_t(std::string_view(tname), arg_num, ro);
}

std::string
ocl::impl::
eval_args_t(const std::string& tname,
            unsigned& arg_num,
            bool ro)
{
    return eval_args_t(std::string_view(tname), arg_num, ro);
}


std::string
ocl::impl::
eval_args_array_ptr(const std::string_view& tname,
                    unsigned& arg_num,
                    bool ro)
{
    std::ostringstream s;
    s << spaces(4) ;
    s << "__arg_local ";
    if (ro)
        s << "const ";
    s << tname
      << "* arg"  << arg_num;
    ++arg_num;
    return s.str();
}

std::string
ocl::impl::
eval_args_array_ptr(const char* tname,
                    unsigned& arg_num,
                    bool ro)
{
    return eval_args_array_ptr(std::string_view(tname), arg_num, ro);
}

std::string
ocl::impl::
eval_args_array_ptr(const std::string& tname,
                    unsigned& arg_num,
                    bool ro)
{
    return eval_args_array_ptr(std::string_view(tname), arg_num, ro);
}

std::string
ocl::impl::
eval_vars_t(const std::string_view& tname,
            unsigned& arg_num,
            bool ro)
{
    std::ostringstream s;
    s << spaces(8) << tname
      << " v" << arg_num;
    if (ro== true) {
        s << " = arg"
          << arg_num << ";";
    }
    std::string a(s.str());
    ++arg_num;
    return a;
}

std::string
ocl::impl::
eval_vars_t(const char* tname,
            unsigned& arg_num,
            bool ro)
{
    return eval_vars_t(std::string_view(tname), arg_num, ro);
}

std::string
ocl::impl::
eval_vars_t(const std::string& tname,
            unsigned& arg_num,
            bool ro)
{
    return eval_vars_t(std::string_view(tname), arg_num, ro);
}


std::string
ocl::impl::
eval_vars_array_ptr(const std::string_view& tname,
                    unsigned& arg_num,
                    bool ro)
{
    std::ostringstream s;
    s << spaces(8)
      << "__arg_local ";
    if (ro == true)
        s << "const ";
    s << tname
      << "* v" << arg_num;
    if (ro == true) {
        s << " = arg"
          << arg_num << ";";
    }
    std::string a(s.str());
    ++arg_num;
    return a;
}

std::string
ocl::impl::
eval_vars_array_ptr(const char* tname,
                    unsigned& arg_num,
                    bool ro)
{
    return eval_vars_array_ptr(std::string_view(tname), arg_num, ro);
}

std::string
ocl::impl::
eval_vars_array_ptr(const std::string& tname,
                    unsigned& arg_num,
                    bool ro)
{
    return eval_vars_array_ptr(std::string_view(tname), arg_num, ro);
}

std::string
ocl::impl::
eval_ops_t(unsigned& arg_num)
{
    std::ostringstream s;
    s << "v" << arg_num;
    std::string a(s.str());
    ++arg_num;
    return a;
}
