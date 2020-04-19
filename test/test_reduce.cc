#include <ocl/ocl.h>
#include <ocl/test/tools.h>
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

}


int main()
{
    try {

        using namespace ocl;

        using cftal::v8f32;

        const unsigned SIZE=512;
        std::cout << "using buffers of "
                  << double(SIZE*sizeof(float))/(1024*1024)
                  << "MiB\n";
        float a(2.0f), b(3.0f);

        dvec<float> va(SIZE, a);
        dvec<float> vb(SIZE, b);
        dvec<int32_t> tgt= va == vb;        

        bool ao = all_of(tgt);
        bool no = none_of(tgt);
        bool so = any_of(tgt);

        if (ao != true || no != false || so != true) {
            throw std::runtime_error("xxx_of failed.");
        }
    }
    catch (const ocl::be::error& e) {
        std::cout << "caught exception: " << e.what()
                  << '\n'
                  << e.error_string()
                  << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "caught exception: " << e.what()
                  << std::endl;
    }
    return 0;
}
