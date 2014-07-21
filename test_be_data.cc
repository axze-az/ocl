#include <ocl/ocl.h>

int main()
{
        int r;
        try {
                std::vector<cl::Device> v(ocl::impl::devices());
                std::cout << "count of devices: " << v.size() << std::endl;
                for (std::size_t i = 0; i< v.size(); ++i) {
                        std::cout << ocl::impl::device_info(v[i]);
                }
                ocl::impl::device dd(ocl::impl::default_device());
                std::cout << "selected device: \n";
                std::cout << ocl::impl::device_info(dd);
                
                const ocl::impl::device& bed =
                        ocl::impl::be_data::instance()->d();
                std::cout << "\nselected backend device: \n";
                std::cout << ocl::impl::device_info(bed);
                r = 0;
        }
        catch (const ocl::impl::error& e) {
                std::cout << "caught exception: " << e.what()
                          << '\n'
                          << ocl::impl::err2str(e)
                          << std::endl;
                r = 3;
        }
        return r;
}
