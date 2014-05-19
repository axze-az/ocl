#include "ocl.h"
#include <iostream>

const char* ocl::impl::err2str(const error& e)
{
        return err2str(e.err());
}

const char* ocl::impl::err2str(int err) 
{
        if (err > 0) 
                return strerror(err);
        switch (err) {
        case CL_SUCCESS:                            
                return "Success!";
        case CL_DEVICE_NOT_FOUND:                   
                return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               
                return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             
                return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      
                return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   
                return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 
                return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       
                return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   
                return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              
                return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         
                return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              
                return "Program build failure";
        case CL_MAP_FAILURE:                        
                return "Map failure";
        case CL_INVALID_VALUE:                      
                return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                
                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   
                return "Invalid platform";
        case CL_INVALID_DEVICE:                     
                return "Invalid device";
        case CL_INVALID_CONTEXT:                    
                return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           
                return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              
                return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   
                return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 
                return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    
                return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 
                return "Invalid image size";
        case CL_INVALID_SAMPLER:                    
                return "Invalid sampler";
        case CL_INVALID_BINARY:                     
                return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              
                return "Invalid build options";
        case CL_INVALID_PROGRAM:                    
                return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         
                return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                
                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          
                return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     
                return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  
                return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  
                return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   
                return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                
                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             
                return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            
                return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             
                return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              
                return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            
                return "Invalid event wait list";
        case CL_INVALID_EVENT:                      
                return "Invalid event";
        case CL_INVALID_OPERATION:                  
                return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  
                return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                
                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  
                return "Invalid mip-map level";
        default:
                return "Unknown";
        }
        // return p;
}

std::ostream&
ocl::impl::operator<<(std::ostream& s, const device_info& dd)
{
        const device& d= dd._d;
        std::string n(d.getInfo<CL_DEVICE_NAME>());
        s << "device name: " << n << '\n';
        n = d.getInfo<CL_DEVICE_VENDOR>();
        s << "device vendeor: " << n << '\n';
        
        cl_device_type dt(d.getInfo<CL_DEVICE_TYPE>());
        s << "device type: ";
        switch (dt) {
        case CL_DEVICE_TYPE_CPU:
                s << "cpu\n" ;
                break;
        case CL_DEVICE_TYPE_GPU:
                s << "gpu\n";
                break;
        default:
                s << "unknown\n";
                break;
        }
        cl_uint t(d.getInfo<CL_DEVICE_VENDOR_ID>());
        s << "vendor id: " << t << '\n';
        
        return s;
}

std::vector<ocl::impl::device>
ocl::impl::devices()
{
        std::vector<device> r;
        // loop over all platforms and get all devices
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        std::vector<cl::Platform>::iterator i;
        std::vector<cl::Platform>::iterator pe(platforms.end());
        ctx_prop_vec_type props;
        props.push_back(CL_CONTEXT_PLATFORM);
        props.push_back(0);
        props.push_back(0);
        for(i = platforms.begin(); i != pe; ++i) {
                // std::string n(i->getInfo<CL_PLATFORM_NAME>());
                // std::cerr << "platform name: " << n << '\n';
                // cl_platform_id pi((*i)());
                try {
                        std::vector<device> dc;
                        i->getDevices(CL_DEVICE_TYPE_ALL, &dc);
                        std::copy(dc.begin(), dc.end(), std::back_inserter(r));
                }
                catch (const error& ex) {
                        std::cerr << "exception caught: "
                                  << ex.what()
                                  << '\n'
                                  << err2str(ex.err())
                                  << '\n';
                }
                catch (...) {
                        std::cerr << "oops" << std::endl;
                }
        }
        return r;
}

std::vector<ocl::impl::device>
ocl::impl::filter_devices(const std::vector<device>& v, 
	                  device_type::type dt)
{
	std::vector<device> r;
	for (std::size_t i=0; i< v.size(); ++i) {
		const device& d= v[i];
		cl_device_type t(d.getInfo<CL_DEVICE_TYPE>());
		if (t == static_cast<cl_device_type>(dt))
			r.push_back(d);
	}
	return r;
}

std::vector<ocl::impl::device>
ocl::impl::gpu_devices(const std::vector<device>& v)
{
	return filter_devices(v, device_type::gpu);
}

std::vector<ocl::impl::device>
ocl::impl::gpu_devices()
{
        std::vector<device> all_devs(devices());
	return gpu_devices(all_devs);
}

std::vector<ocl::impl::device>
ocl::impl::cpu_devices(const std::vector<device>& v)
{
	return filter_devices(v, device_type::cpu);
}

std::vector<ocl::impl::device>
ocl::impl::cpu_devices()
{
        std::vector<device> all_devs(devices());
	return cpu_devices(all_devs);
}

ocl::impl::device
ocl::impl::default_gpu_device()
{
	std::vector<device> gpu_devs(gpu_devices());	
        device r;
        return r;
}

