//
// Copyright (C) 2010-2025 Axel Zeuner
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
//
#include <cftal/vsvec.h>
#include <ocl/ocl.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace ocl {

    namespace test {
        void
	transfer_bw(be::data_ptr bedp);

        void
	transfer_bw(int argc, char** argv);

    }
}

void
ocl::test::
transfer_bw(be::data_ptr bedp)
{
    constexpr const size_t BUF_SIZE=1024*1024*128;
    constexpr const size_t ELEM_CNT=BUF_SIZE/sizeof(float);
    std::cout << "testing\n"  << bedp->dcq().d().name() << '\n'
              << "transfer bandwidth using "
	      << std::fixed << std::setprecision(4) 
              << double(BUF_SIZE)/(1024*1024*1024) << " GiB buffers\n";

    double to_device(0.0), from_device(0.0);
    double data_size(0.0);

    std::cout << std::fixed << std::setprecision(1);
    try {
	using d_vec_t = dvec<float>;
	using h_vec_t = cftal::vsvec<float>;
	h_vec_t h_sv(1.0f, ELEM_CNT);
	h_vec_t h_rv(1.0f, ELEM_CNT);
	d_vec_t d_sv(bedp, 0.0f, ELEM_CNT);
	d_vec_t d_rv(bedp, 0.0f, ELEM_CNT);
	
	const size_t _N = 72;
        for (size_t i=0; i<_N*4; ++i) {
            auto start0 = std::chrono::steady_clock::now();
	    d_sv.copy_from_host(h_sv.cbegin());
            auto end0 = std::chrono::steady_clock::now();
            auto ns_elapsed0=(end0 - start0).count();
	    to_device += double(ns_elapsed0);

	    d_rv = 2.0f * d_sv;

            auto start1 = std::chrono::steady_clock::now();
	    d_rv.copy_to_host(h_rv.begin());
            auto end1 = std::chrono::steady_clock::now();
            auto ns_elapsed1=(end1 - start1).count();

	    from_device += double(ns_elapsed1);
	    data_size += BUF_SIZE;
	    if ((i & 3)==3)
		std::cout << '.' << std::flush;
        }
	to_device = data_size/to_device;
	from_device = data_size/from_device;
	// to_device and from_device contain now bytes per ns
	// GiB/ns -> divide by (1024*1024*1024)
	// GiB/s -> multiply by 1e9
	const double factor= 1e9/(double(1024)*double(1024)*double(1024));
	to_device *= factor;
	from_device *= factor;
	std::cout << std::setprecision(2)
		  << "\nbandwith to device: " << to_device
		  << " GiB/s\n"
		  << "bandwith from device: " << to_device
		  << " GiB/s\n";
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
}

void
ocl::test::transfer_bw(int argc, char** argv)
{
    int device=-1;
    using std::string_view;
    int err=0;
    for (int i=1; i<argc; ++i) {
        string_view argi(argv[i]);
        string_view::size_type eq_pos=argi.find('=');
        string_view ai(argi.substr(0, eq_pos));
        if (ai == "--device") {
            if (argi.length()>eq_pos) {
                string_view pl=argi.substr(eq_pos+1);
                std::istringstream is(std::string(pl.data(), pl.length()));
                int32_t d=0;
                is >> d;
                if (is.fail() || !is.eof()) {
                    if (pl.size()==0) {
                        std::cerr << "device number is missing\n";
                    } else {
                        std::cerr << "invalid device number " << pl << '\n';
                    }
                    ++err;
                } else {
                    device=d;
                }
            } else {
                std::cerr << "device number is missing\n";
                ++err;
            }
        } else {
            std::cerr << "invalid argument " << argi << '\n';
            ++err;
        }
    }
    if (err) {
        std::cout << "usage: "  << argv[0]  << " [--device=X]\n";
        return;
    }
    try {
        std::vector<be::device> v(be::devices());
        if (device >= int(v.size())) {
            std::cerr << "device number " << device << "is undefined:\n";
            for (size_t i=0; i<v.size(); ++i) {
                std::cerr << i << ": " << v[i].name() << '\n';
            }
            return;
        }
        if (device > -1) {
            auto bedp=be::data::create(v[device]);
	    transfer_bw(bedp);
        }  else {
            for (auto& d : v) {
                auto bedp=be::data::create(d);
		transfer_bw(bedp);
            }
        }
    }
    catch (const be::error& e) {
        std::cout << "caught exception: be::error: " << e.what()
                  << '\n'
                  << e.error_string()
                  << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cout << "caught exception: runtime error: " << e.what()
                  << std::endl;
    }
}

int main(int argc, char** argv)
{
    ocl::test::transfer_bw(argc, argv);
    return 0;
}

