// Calculates total energy of every particle
__kernel
void energy(__global float* pos, __global float* vel, __global float* energy, 
             __local float4* block)
    {
    const int n = get_global_size(0);
    const int nt = get_local_size(0);

    const int gId = get_global_id(0);
    const int lId = get_local_id(0);    

    const int nb = n / nt;
       
    float4 p = vload4(gId, pos);
    float4 v = vload4(gId, vel);
    
    float m1 = p.w;
    float T = 0.5f * m1 * dot(v,v);
    float V = 0;

    for(int jb = 0; jb < nb; jb++)
        {        
        block[lId] = vload4(jb * nt + lId, pos);
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for(int j = 0; j < nt; j++)
            {
            if(jb * nt + j != gId)
                {
                float4 p2 = block[j];
                float4 d = p2 - p;
                float f = d.x*d.x + d.y*d.y + d.z*d.z;
                f = rsqrt(f);
                V -= m1 * p2.w * f;
                }
            }

        barrier(CLK_LOCAL_MEM_FENCE);
        }

    float E = T + 0.5f * V;
    energy[gId] = E;
    }


// Calculates interagtion with semi-implicit Euler integration
__kernel
void euler_local(const float dt, const float eps2, __global float* pos0, 
           __global float* pos1, __global float* vel, __local float4* block)
    {
    const int n = get_global_size(0);
    const int nt = get_local_size(0);

    const int gId = get_global_id(0);
    const int lId = get_local_id(0);    

    const int nb = n / nt;
    
    float4 dtv = (float4)(dt, dt, dt, 0);
    
    float4 p = vload4(gId, pos0);
    float4 v = vload4(gId, vel);
    float4 a = (float4)(0, 0, 0, 0);

    for(int jb = 0; jb < nb; jb++)
        {
        block[lId] = vload4(jb * nt + lId, pos0);
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int j = 0; j < nt; j++)
            {
            float4 p2 = block[j];
            float4 d = p2 - p;
            float f = d.x*d.x + d.y*d.y + d.z*d.z + eps2;
            f = rsqrt(f);
            f = p2.w * f * f * f;
            a += f * d;
            }

        barrier(CLK_LOCAL_MEM_FENCE);
        }    

    v += dtv * a;
    p += dtv * v;

    vstore4(p, gId, pos1);
    vstore4(v, gId, vel);
    }

__kernel
void euler(const float dt, const float eps2, __global float* pos, 
           __global float* vel, __global float* leaves, int n_leaves)
    {
    const int gid = get_global_id(0);
    
    float4 dtv = (float4)(dt, dt, dt, 0);
    
    float4 p = vload4(gid, pos);
    float4 v = vload4(gid, vel);
    float4 a = (float4)(0, 0, 0, 0);
    
    for(int j = 0; j < n_leaves; j++)
        {
        float4 p2 = vload4(j, leaves);
        float4 d = p2 - p;
        float f = d.x*d.x + d.y*d.y + d.z*d.z + eps2;
        f = rsqrt(f);
        f = p2.w * f * f * f;
        a += f * d;
        }
    
    v += dtv * a;
    p += dtv * v;

    vstore4(p, gid, pos);
    vstore4(v, gid, vel);
    }
