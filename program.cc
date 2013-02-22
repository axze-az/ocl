#include "ocl/ocl.h"

ocl::ll::program::program(const context& ctx,
			  const std::string& src,
			  bool build)
	: base_type(nullptr)
{
        const char * s=src.c_str();
        const std::size_t l(src.length());
	cl_int error;
	cl_program p(clCreateProgramWithSource(
			     ctx(), 1, &s, &l, &error));
	check_err(error);
	h(p);
	if (build) {
		error = clBuildProgram(
			p,
			0,
			nullptr,
			"",
			nullptr,
			nullptr);
		check_err(error);
        }
}

ocl::ll::program::program(const context& ctx,
			  const std::vector<std::string>& src,
			  bool build)
	: base_type(nullptr)
{
	std::vector<const char*> sv(src.size(), nullptr);
	std::vector<std::size_t> lv(src.size(), std::size_t(0));
	for (std::size_t i=0; i< src.size(); ++i) {
		sv[i] = src[i].c_str();
		lv[i] = src[i].length();
	}
	cl_int error;
	cl_program p(clCreateProgramWithSource(
			     ctx(), src.size(), 
			     &sv[0], &lv[0], &error));
	check_err(error);
	h(p);
	if (build) {
		error = clBuildProgram(
			p,
			0,
			nullptr,
			"",
			nullptr,
			nullptr);
		check_err(error);
        }
}
