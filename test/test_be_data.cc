#include "ocl/ocl.h"
#include "ocl/be/devices.h"
#include "ocl/be/data.h"

int main()
{
    int r;
    try {

        std::vector<ocl::be::device> v(ocl::be::devices());
        std::cout << std::string(60, '-') << std::endl;
        std::cout << "count of devices: " << v.size() << std::endl;
        for (std::size_t i = 0; i< v.size(); ++i) {
            std::cout << std::string(60, '-') << std::endl;
            std::cout << ocl::be::device_info(v[i]);
        }
        ocl::be::device dd(ocl::be::default_device());
        std::cout << std::string(60, '-') << std::endl;
        std::cout << "selected device: \n";
        std::cout << ocl::be::device_info(dd);

        const ocl::be::device& bed =
            ocl::be::data::instance()->dcq().d();
        std::cout << std::string(60, '-') << std::endl;
        std::cout << "selected backend device: \n";
        std::cout << ocl::be::device_info(bed);
        r = 0;
    }
    catch (const ocl::be::error& e) {
        std::cout << "caught exception: " << e.what()
                  << '\n'
                  << e.error_string()
                  << std::endl;
        r = 3;
    }
    return r;
}
