#ifndef CONTEXT_H
#define CONTEXT_H
#include <iostream>
#include <vector>
#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif
using namespace std;

#define NUM_GLOBAL_WITEMS 1024              // number of threads


// reads in kernel from file
string read_kernel();
// creates context and queues
void create_context(cl::Context &context, cl::CommandQueue &queue, cl::Program &program);

#endif
