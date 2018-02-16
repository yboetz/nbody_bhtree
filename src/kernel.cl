
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
void euler(const float dt, const float eps2, __global float* pos, __global float* vel, const int n,
           __global float* leaves, const int n_leaves, __global float* cells, const int n_cells,
           __local float* block)
{
    const int SIZEOF_COM = 4;                           // SIZEOF_COM
    const int SIZEOF_TOT = 10;                          // SIZEOF_COM + SIZEOF_MOM
    const int lsize = get_local_size(0);
    const int gid = get_global_id(0);
    const int lid = get_local_id(0);

    const int nb_l = n_leaves / lsize;                     // block size for leaves
    const int nb_c = n_cells / lsize;                      // block size for cells

    float4 dtv = (float4)(dt, dt, dt, 0);
    float4 p, v;

    if(gid < n)                                         // if global id >= n, don't do any work
    {
        p = vload4(gid, pos);
        v = vload4(gid, vel);
    }
    else
    {
        p = (float4)(0, 0, 0, 0);
        v = (float4)(0, 0, 0, 0);
    }
    float4 a = (float4)(0, 0, 0, 0);
    
    for(int jb = 0; jb < nb_l; jb++)
    {
        vstore4(vload4(jb*lsize + lid, leaves), lid, block);    // write to local memory
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int j = 0; j < lsize; j++)
        {
            float4 p2 = vload4(j, block);
            float4 d = p2 - p;
            float invr = d.x*d.x + d.y*d.y + d.z*d.z + eps2;
            invr = rsqrt(invr);
            invr = p2.w * invr * invr * invr;
            a += invr * d;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for(int jb = 0; jb < nb_c; jb++)
    {
        int idx = SIZEOF_TOT*(jb*lsize + lid);
        int bidx = SIZEOF_TOT*lid;
        block[bidx] = cells[idx];       // write to local memory
        block[bidx+1] = cells[idx+1];
        block[bidx+2] = cells[idx+2];
        block[bidx+3] = cells[idx+3];
        block[bidx+4] = cells[idx+4];
        block[bidx+5] = cells[idx+5];
        block[bidx+6] = cells[idx+6];
        block[bidx+7] = cells[idx+7];
        block[bidx+8] = cells[idx+8];
        block[bidx+9] = cells[idx+9];
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int j = 0; j < lsize; j++)
        {
            int idx = SIZEOF_TOT*j;         // SIZEOF_TOT * j
            float m, xx, xy, xz, yy, yz, zz;
            float4 p2 = (float4)(block[idx], block[idx+1], block[idx+2], block[idx+3]);
            m = p2.w;
            xx = block[idx+4]; xy = block[idx+5]; xz = block[idx+6];
            yy = block[idx+7]; yz = block[idx+8]; zz = block[idx+9];

            float4 d = p2 - p;
            float invr = d.x*d.x + d.y*d.y + d.z*d.z + eps2;
            invr = rsqrt(invr);
            d = d*invr;
            invr = invr*invr;

            // monopole
            a += m * invr * d;

            // quadrupole
            invr = 3.0*invr*invr;                    // invr = 3/r^4
            float q1 = (xx*d.x + xy*d.y + xz*d.z)*invr;
            float q2 = (xy*d.x + yy*d.y + yz*d.z)*invr;
            float q3 = (xz*d.x + yz*d.y + zz*d.z)*invr;

            a -= (float4)(q1, q2, q3, 0.0);

            invr = 5.0/2.0 * (q1*d.x + q2*d.y + q3*d.z);
            a += invr * d;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    v += dtv * a;
    p += dtv * v;

    if(gid < n)
    {
        vstore4(p, gid, pos);
        vstore4(v, gid, vel);
    }
}
