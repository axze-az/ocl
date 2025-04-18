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
#include "ocl/be/types.h"
#include <sstream>

std::string
ocl::cl::error::to_string(cl_int code)
{
    switch(code){
    case CL_SUCCESS:
        return "Success";
    case CL_DEVICE_NOT_FOUND:
        return "Device Not Found";
    case CL_DEVICE_NOT_AVAILABLE:
        return "Device Not Available";
    case CL_COMPILER_NOT_AVAILABLE:
        return "Compiler Not Available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "Memory Object Allocation Failure";
    case CL_OUT_OF_RESOURCES:
        return "Out of Resources";
    case CL_OUT_OF_HOST_MEMORY:
        return "Out of Host Memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        return "Profiling Information Not Available";
    case CL_MEM_COPY_OVERLAP:
        return "Memory Copy Overlap";
    case CL_IMAGE_FORMAT_MISMATCH:
        return "Image Format Mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return "Image Format Not Supported";
    case CL_BUILD_PROGRAM_FAILURE:
        return "Build Program Failure";
    case CL_MAP_FAILURE:
        return "Map Failure";
    case CL_INVALID_VALUE:
        return "Invalid Value";
    case CL_INVALID_DEVICE_TYPE:
        return "Invalid Device Type";
    case CL_INVALID_PLATFORM:
        return "Invalid Platform";
    case CL_INVALID_DEVICE:
        return "Invalid Device";
    case CL_INVALID_CONTEXT:
        return "Invalid Context";
    case CL_INVALID_QUEUE_PROPERTIES:
        return "Invalid Queue Properties";
    case CL_INVALID_COMMAND_QUEUE:
        return "Invalid Command Queue";
    case CL_INVALID_HOST_PTR:
        return "Invalid Host Pointer";
    case CL_INVALID_MEM_OBJECT:
        return "Invalid Memory Object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return "Invalid Image Format Descriptor";
    case CL_INVALID_IMAGE_SIZE:
        return "Invalid Image Size";
    case CL_INVALID_SAMPLER:
        return "Invalid Sampler";
    case CL_INVALID_BINARY:
        return "Invalid Binary";
    case CL_INVALID_BUILD_OPTIONS:
        return "Invalid Build Options";
    case CL_INVALID_PROGRAM:
        return "Invalid Program";
    case CL_INVALID_PROGRAM_EXECUTABLE:
        return "Invalid Program Executable";
    case CL_INVALID_KERNEL_NAME:
        return "Invalid Kernel Name";
    case CL_INVALID_KERNEL_DEFINITION:
        return "Invalid Kernel Definition";
    case CL_INVALID_KERNEL:
        return "Invalid Kernel";
    case CL_INVALID_ARG_INDEX:
        return "Invalid Argument Index";
    case CL_INVALID_ARG_VALUE:
        return "Invalid Argument Value";
    case CL_INVALID_ARG_SIZE:
        return "Invalid Argument Size";
    case CL_INVALID_KERNEL_ARGS:
        return "Invalid Kernel Arguments";
    case CL_INVALID_WORK_DIMENSION:
        return "Invalid Work Dimension";
    case CL_INVALID_WORK_GROUP_SIZE:
        return "Invalid Work Group Size";
    case CL_INVALID_WORK_ITEM_SIZE:
        return "Invalid Work Item Size";
    case CL_INVALID_GLOBAL_OFFSET:
        return "Invalid Global Offset";
    case CL_INVALID_EVENT_WAIT_LIST:
        return "Invalid Event Wait List";
    case CL_INVALID_EVENT:
        return "Invalid Event";
    case CL_INVALID_OPERATION:
        return "Invalid Operation";
    case CL_INVALID_GL_OBJECT:
        return "Invalid GL Object";
    case CL_INVALID_BUFFER_SIZE:
        return "Invalid Buffer Size";
    case CL_INVALID_MIP_LEVEL:
        return "Invalid MIP Level";
    case CL_INVALID_GLOBAL_WORK_SIZE:
        return "Invalid Global Work Size";
#if CL_TARGET_OPENCL_VERSION>=120
    case CL_COMPILE_PROGRAM_FAILURE:
        return "Compile Program Failure";
    case CL_LINKER_NOT_AVAILABLE:
        return "Linker Not Available";
    case CL_LINK_PROGRAM_FAILURE:
        return "Link Program Failure";
    case CL_DEVICE_PARTITION_FAILED:
        return "Device Partition Failed";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
        return "Kernel Argument Info Not Available";
    case CL_INVALID_PROPERTY:
        return "Invalid Property";
    case CL_INVALID_IMAGE_DESCRIPTOR:
        return "Invalid Image Descriptor";
    case CL_INVALID_COMPILER_OPTIONS:
        return "Invalid Compiler Options";
    case CL_INVALID_LINKER_OPTIONS:
        return "Invalid Linker Options";
    case CL_INVALID_DEVICE_PARTITION_COUNT:
        return "Invalid Device Partition Count";
#endif
#if CL_TARGET_OPENCL_VERSION>=200
    case CL_INVALID_PIPE_SIZE:
        return "Invalid Pipe Size";
    case CL_INVALID_DEVICE_QUEUE:
        return "Invalid Device Queue";
#endif
    default: {
        std::ostringstream s;
        s << "Unknown OpenCL Error (" << code << ")";
        return s.str();
    }}
}

std::string
ocl::cl::error::to_string(cl_int code, const char* file, unsigned line)
{
    std::ostringstream s;
    s << to_string(code) << ' ' << file << ':' << line;
    return s.str();
}

ocl::cl::error::error(cl_int code)
    : base_type(to_string(code)), _code(code)
{
}

ocl::cl::error::error(cl_int code, const char* file, unsigned line)
    : base_type(to_string(code, file, line)), _code(code)
{
}


ocl::cl::error::error(const error& r)
    : base_type(r), _code(r._code)
{
}

ocl::cl::error::error(error&& r)
    : base_type(std::move(r)), _code(std::move(r._code))
{
}

ocl::cl::error&
ocl::cl::error::operator=(const error& r)
{
    if (&r != this) {
        base_type::operator=(r);
        _code = r._code;
    }
    return *this;
}

ocl::cl::error&
ocl::cl::error::operator=(error&& r)
{
    base_type::operator=(std::move(r));
    _code = std::move(r._code);
    return *this;
}

ocl::cl::error::~error()
{
}

std::string
ocl::cl::error::error_string() 
    const 
{
        return what();
}
            
void
ocl::cl::error::
_throw_on(cl_int code)
{
    if (code != CL_SUCCESS)
        throw error(code);
}

void
ocl::cl::error::
_throw_on(cl_int code, const char* file, unsigned line)
{
    if (code != CL_SUCCESS)
        throw error(code, file, line);
}

