#include "ocl/ocl.h"
#include <sstream>

std::string 
ocl::error::fmt_msg(cl_uint rc, const std::string& msg)
{
	std::stringstream s;
	s << msg << " error code " << rc;
	return s.str();
}

ocl::error::error(const std::string& msg)
	: runtime_error(msg)
{
}

ocl::error::error(cl_uint rc, const std::string& msg)
	: runtime_error(fmt_msg(rc, msg))
{
}

ocl::error::~error()
{
}
