#include <iostream>
#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif
using namespace std;

#define NUM_GLOBAL_WITEMS 1024              // number of threads


string read_kernel()
{
    FILE *fp;
    char *source_str;
    size_t source_size, program_size;
    // get file name
    string path = __FILE__;
    size_t last_slash = path.find_last_of("/");
    path = path.substr(0, last_slash) + "/kernel.cl";

    fp = fopen(path.c_str(), "rb");
    if (!fp) {
        printf("Failed to load kernel...\n");
        exit(1);
    }

    fseek(fp, 0, SEEK_END);
    program_size = ftell(fp);
    rewind(fp);
    source_str = (char*)malloc(program_size + 1);
    source_str[program_size] = '\0';
    source_size = fread(source_str, sizeof(char), program_size, fp);
    fclose(fp);

    return string(source_str, source_size);
}

void create_context(cl::Context &context, cl::CommandQueue &data_queue, cl::CommandQueue &compute_queue,
                    cl::Kernel &euler)
{
    // get all platforms
    vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    if (all_platforms.size()==0)
    {
        printf("No platforms found...\n");
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>() << "...\n";

    // get default device (CPUs, GPUs) of the default platform
    vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0)
    {
        printf("No devices found...\n");
        exit(1);
    }

    // use device[0] for now
    cl::Device default_device=all_devices[0];
    cout << "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>() << "...\n";

    // create context
    cl::Context _context({default_device});
    context = _context;

    // read in kernel code
    cl::Program::Sources sources;
    string kernel_code = read_kernel();
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    // create program and build code
    cl::Program program(context, sources);
    if (program.build({default_device}) != CL_SUCCESS)
    {
        cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "...\n";
        exit(1);
    }

    // set up queues
    cl::CommandQueue _data_queue(context, default_device);
    cl::CommandQueue _compute_queue(context, default_device);
    data_queue = _data_queue;
    compute_queue = _compute_queue;
    // // create kernels
    cl::Kernel _euler = cl::Kernel(program, "euler");
    euler = _euler;

    // // allocate space
    // cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    // cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    // cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * n);

    // // push write commands to queue
    // queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int)*n, A);
    // queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*n, B);

    // // RUN ZE KERNEL
    // add.setArg(0, buffer_A);
    // add.setArg(1, buffer_B);
    // add.setArg(2, buffer_C);
    // add.setArg(3, n);
    // add.setArg(4, k);
    // compute_queue.enqueueNDRangeKernel(add, cl::NullRange,  // kernel, offset
    //         cl::NDRange(NUM_GLOBAL_WITEMS), // global number of work items
    //         cl::NDRange(32));               // local number (per group)

    // // read result from GPU to here; including for the sake of timing
    // queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int)*n, C);
    // // wait for everything to finish
    // queue.finish();
}
