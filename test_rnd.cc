#include "ocl.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <atomic>
#include <mutex>
#include <memory> // for shared_ptr
#include <cmath>
#include <cstdint>


namespace ocl {

    // template <class _T>
    class mrand48 {
        vector<std::uint64_t> _state;

        static const std::uint64_t A;
        static const std::uint32_t C;
        static const std::uint64_t M;
        static const std::uint64_t MM;
        static const float REC;
    public:
        inline
        vector<std::uint32_t>
        next() {
            // vector<std::uint64_t> nv= ((m_state * A + C) & MM);
            _state = (_state * A + C) /* & MM */;
            // nv >>= 16;
            return cvt_to<vector<std::uint32_t> >(_state);
        }

        void 
        seed(const vector<uint64_t>& gid){
            // m_state = (gid * 65536) | 0x330E;
            _state = (7*gid) ^ 0x330E;
        }

        vector<float>
        nextf() {
            vector<float> r=max(
                min(cvt_to<vector<float> >(next()) * REC, 1.0f), 0.0f);
            return r;
        }
    };
}

const std::uint64_t ocl::mrand48::A=0x5DEECE66Dul;
const std::uint32_t ocl::mrand48::C=0xBL;
const std::uint64_t ocl::mrand48::M=(1ULL<<48);
const std::uint64_t ocl::mrand48::MM=(M-1);
const float ocl::mrand48::REC= 1.0f/uint32_t(-1);


int main()
{
    try {

        using namespace ocl;

        const unsigned _N = 1024;
        
        std::vector<std::uint64_t> gid(_N, 0ull);
        for (std::size_t i=0; i<gid.size(); ++i)
            gid[i] = i;
        vector<std::uint64_t> dg=gid;
        ocl::mrand48 t;
        t.seed(dg);

        for (int i=0; i<50; ++i) {
            vector<float> f= t.nextf();
            std::cout << "iteration " << i << std::endl;
            if ((i & (32-1)) == (32-1)) {
                std::vector<float> fh(f);
                for (std::size_t j=0; j<fh.size(); ++j) {
                    std::cout << std::setw(2) << j
                              << ": " << fh[j] << std::endl;
                }
            }
        }

    }
    catch (const ocl::impl::error& e) {
        std::cout << "caught exception: " << e.what()
                  << '\n'
                  << ocl::impl::err2str(e)
                  << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cout << "caught exception: " << e.what()
                  << std::endl;
    }
    return 0;
}
