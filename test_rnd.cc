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
    struct mrand48 {

        vector<std::uint64_t> m_state;

        static const std::uint64_t A;
        static const std::uint32_t C;
        static const std::uint64_t M;
        static const std::uint64_t MM;
        static const float REC;
        
        vector<std::int32_t>
        inline
        next() {
            vector<std::uint64_t> nv= (m_state * A + C) & MM;
            m_state = nv;
            // nv >>= 16;
            return cvt_to<vector<std::int32_t> >(nv);
        }

        void 
        seed(const vector<uint64_t>& gid){
            // m_state= (gid << 16u) | 0x330E;
            // m_state = (gid * 65536ll) | 0x330E;
            m_state = gid;
            m_state *= 65536lu;
            m_state |= 0x330Elu;
        }

        vector<float>
        nextf() {
            // vector<float> t= cvt_to<vector<float> >(next());
            vector<float> r=max(min(cvt_to<vector<float> >(next()) * REC, 1.0f), -1.0f);
            // vector<float> r= max(r1, -1.0f);
            return r;
        }
    };
}

const std::uint64_t ocl::mrand48::A=0x5DEECE66Du;
const std::uint32_t ocl::mrand48::C=0xBL;
const std::uint64_t ocl::mrand48::M=(1LL<<48);
const std::uint64_t ocl::mrand48::MM=(M-1);
const float ocl::mrand48::REC= 1.0f / 2147483647.0f;


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
