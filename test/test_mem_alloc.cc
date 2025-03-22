#include <cftal/vec.h>
#include <ocl/ocl.h>
#include <iostream>
#include <iomanip>


int main()
{
    try {
        using namespace ocl;

        using ftype = float;

        constexpr const std::size_t elem_count=(64*1024*1024ULL)/sizeof(ftype);
        std::vector<dvec<ftype> > vv;

        constexpr const std::size_t max_count=
            (8*1024*1024ULL*1024ULL)/sizeof(ftype)/elem_count;
        std::cout << std::fixed << std::setprecision(2);
        for (std::size_t i=0; i<max_count; ++i) {
            double mb=(double(vv.size())*elem_count*sizeof(ftype))/
                double(1024*1024);
            std::cout << "total allocated memory: "
                      << std::setw(7) << mb << " MB\n";
            auto init_val=static_cast<ftype>(i+1);
            dvec<ftype> vi(init_val, elem_count);
            vv.emplace_back(std::move(vi));
            for (size_t j=0; j<vv.size(); ++j) {
                vv[j] += init_val;
            }
        }
        std::cout << "test passed\n";
    }
    catch (const ocl::be::error& e) {
        std::cout << "caught exception: ocl::be::error: " << e.what()
                  << '\n'
                  << e.error_string()
                  << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cout << "caught exception: runtime error: " << e.what()
                  << std::endl;
    }
    return 0;
}
