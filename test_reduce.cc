#include "ocl.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <atomic>
#include <mutex>
#include <memory> // for shared_ptr
#include <cmath>

namespace ocl {


    class custom_kernel {
    };

    bool any_of(const vector<int32_t>& v);
    bool all_of(const vector<int32_t>& v);
    bool none_of(const vector<int32_t>& v);
}

bool
ocl::any_of(const vector<int32_t>& v)
{
    return false;
}

bool
ocl::all_of(const vector<int32_t>& v)
{
    return false;
}

bool
ocl::none_of(const vector<int32_t>& v)
{
    return false;
}

int main()
{
    try {

        using namespace ocl;

        using cftal::v8f32;

        const unsigned SIZE=4096;
        std::cout << "using buffers of "
                  << double(SIZE*sizeof(float))/(1024*1024)
                  << "MiB\n";
        float a(2.0f), b(3.0f);

        vector<float> va(SIZE, a);
        vector<float> vb(SIZE, b);
        vector<int32_t> tgt= va == vb;

        bool ao = all_of(tgt);
        bool no = none_of(tgt);
        bool so = any_of(tgt);

        if (ao != true || no != false || so != true) {
            throw std::runtime_error("xxx_of failed.");
        }
        
        impl::be_data::instance()->clear();
    }
    catch (const ocl::impl::error& e) {
        std::cout << "caught exception: " << e.what()
                  << '\n'
                  << ocl::impl::err2str(e)
                  << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "caught exception: " << e.what()
                  << std::endl;
    }
    return 0;
}
