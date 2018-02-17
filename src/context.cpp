#include "context.h"

// reads in kernel from file
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
// creates context and queues
void create_context(cl::Context &context, cl::CommandQueue &queue, cl::Program &program)
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
    // printf("max group size: %d threads\n", (int)default_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
    // printf("Local mem size: %d moments\n", (int)default_device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()/4/10);

    // create context
    cl::Context _context({default_device});
    context = _context;

    // read in kernel code
    cl::Program::Sources sources;
    string kernel_code = read_kernel();
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    // create program and build code
    program = cl::Program(context, sources);
    if (program.build({default_device}) != CL_SUCCESS)
    {
        cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "...\n";
        exit(1);
    }

    // set up queue
    queue = cl::CommandQueue(context, default_device);
    // create kernels
    // euler = cl::Kernel(program, "euler");
}
