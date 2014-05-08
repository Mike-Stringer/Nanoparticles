#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <fstream>
#include <cuda.h>
#include <map>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand_kernel.h>
#include <curand.h>

using namespace std;

//gpu stuff

#define NoPOLY 896

//Timing stuff

#define TIMESTEPS 15000    // can be put down to 10000 for testing purposes with a change to the y bits and datapoints to %10 (already there just commented out)
#define SUBDIFFUSE 7500         //21, do y=100000 41, do y=200000 61, do y=500000 81, do y=750000 121, do y=2000000 101, do y=1600000
#define DIFFUSE 12500        //81, do y=1250000 101, do y=2000000
#define DATAPOINTS (((((SUBDIFFUSE+(SUBDIFFUSE/10)) - (SUBDIFFUSE))/10) + (((DIFFUSE)  - (SUBDIFFUSE+(SUBDIFFUSE/10)))/250) + (((DIFFUSE*5) - (DIFFUSE))/5000) + (((TIMESTEPS) - (DIFFUSE*5))/30000))+10)
//#define DATAPOINTS ((TIMESTEPS/10)+10)

//(((SUBDIFFUSE+(SUBDIFFUSE/10)) - (SUBDIFFUSE))/10)
//(((DIFFUSE)  - (SUBDIFFUSE+(SUBDIFFUSE/10)))/250)
//(((DIFFUSE*5) - (DIFFUSE))/5000)
//(((TIMESTEPS) - (DIFFUSE*5))/30000)

//length of polymer

#define POLYLENGTH 81
#define RESOLUTION 1                  //segment length //carefull drastically reduces number of conformations
#define NANOSIZE 2             //size of nanoparticle
#define DENSITY 5               //NANOSIZE:DENSITY makes the ratio, with 1, 2, 3 .......DENSITY    where NANOSIZE fills 1 if 1,  1,2 if 2 1,2,3 if 3 etc etc.

class statistics{

private:
	int n;
	float sum;
	float sumsq;

public:
	statistics();
	int getNumber() const;
	float getAverage() const;
	float getSqAverage() const;
	void add(float x);
};


__global__ void cudarandomwalk(float* placeend, float* d_endtoends, float* placebead, float* d_beadtomids, float* placerad, float* d_radofgys, size_t pitch, curandState *randStates, unsigned long seed) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < NoPOLY)
	{

		curandState& randState = randStates[idx];
		curand_init(seed, idx, 0, &randState);
		int a=0;
		int datapointindex = 0;

		int currentnode = 0;
		int upnode;
		int downnode;

		int randomdir = 0;
		float endtoend;
		float beadtomid;
		float radofgy;
		int block;
		int resloop;
		int nloopx;
		int nloopy;
		int nloopz;

		float smidx = 0;
		float smidy = 0;
		float smidz = 0;
		float xSum = 0;
		float ySum = 0;
		float zSum = 0;
		float vrad = 0;
		int Gridx[POLYLENGTH];
		int Gridy[POLYLENGTH];
		int Gridz[POLYLENGTH];

		for (a=0; a < POLYLENGTH; a++)
		{
			Gridx[a] = NANOSIZE;
			Gridy[a] = NANOSIZE;
			Gridz[a] = NANOSIZE;
		}


		for(a=0; a < POLYLENGTH; a++)
		{
			randomdir = (int)(curand_uniform(&randState)*3);
			block = 0;

			if(randomdir==2)
			{
				for (resloop = 1; resloop <= RESOLUTION; resloop++)
				{
					for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
					{
						for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
						{
							for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
							{
								if (((((Gridx[POLYLENGTH-1]+resloop)%DENSITY) == nloopx)) && ((((Gridy[POLYLENGTH-1])%DENSITY) == nloopy)) && ((((Gridz[POLYLENGTH-1])%DENSITY) == nloopz))) block = 1;
							}
						}
					}
				}
				if (block == 0) Gridx[POLYLENGTH-1]++;
			}

			if(randomdir==1)
			{
				for (resloop = 1; resloop <= RESOLUTION; resloop++)
				{
					for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
					{
						for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
						{
							for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
							{
								if (((((Gridx[POLYLENGTH-1])%DENSITY) == nloopx)) && ((((Gridy[POLYLENGTH-1]+resloop)%DENSITY) == nloopy)) && ((((Gridz[POLYLENGTH-1])%DENSITY) == nloopz))) block = 1;
							}
						}
					}
				}
				if (block == 0) Gridy[POLYLENGTH-1]++;
			}

			if(randomdir==0)
			{
				for (resloop = 1; resloop <= RESOLUTION; resloop++)
				{
					for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
					{
						for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
						{
							for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
							{
								if (((((Gridx[POLYLENGTH-1])%DENSITY) == nloopx)) && ((((Gridy[POLYLENGTH-1])%DENSITY) == nloopy)) && ((((Gridz[POLYLENGTH-1]+resloop)%DENSITY) == nloopz))) block = 1;
							}
						}
					}
				}
				if (block == 0) Gridz[POLYLENGTH-1]++;
			}

			if (block == 0)
			{
				Gridx[a]=Gridx[POLYLENGTH-1];
				Gridy[a]=Gridy[POLYLENGTH-1];
				Gridz[a]=Gridz[POLYLENGTH-1];
			}
			if (block == 1) a--;
		}

		for (int y=0; y<=TIMESTEPS; y++)
		{

			for (int z=0; z<(POLYLENGTH); z++)
			{
				block = 0;

				currentnode = (int)(curand_uniform(&randState)*POLYLENGTH);
				randomdir = (int)(curand_uniform(&randState)*6);

				upnode=(currentnode+1);
				downnode=(currentnode-1);

				//-------------------------------------------------------------------block changes ------------------------------------------------------//
				if (currentnode == 0)
				{
					if (randomdir == 0)
					{  
						if (NANOSIZE != 0)
						{
							for (resloop = 1; resloop <= RESOLUTION; resloop++)
							{
								for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
								{
									for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
									{
										for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
										{
											if ((((((Gridx[upnode]-resloop)%DENSITY)) == nloopx) || ((((Gridx[upnode]-resloop)%DENSITY)) == (nloopx-DENSITY))) &&
												(((((Gridy[upnode])%DENSITY)) == nloopy) || ((((Gridy[upnode])%DENSITY)) == (nloopy-DENSITY))) &&
												(((((Gridz[upnode])%DENSITY)) == nloopz) || ((((Gridz[upnode])%DENSITY)) == (nloopz-DENSITY)))) block = 1;
										}
									}
								}
							}
						}
						if (block == 0)
						{
							Gridx[currentnode] = Gridx[upnode];
							Gridy[currentnode] = Gridy[upnode];
							Gridz[currentnode] = Gridz[upnode];
							Gridx[currentnode]--;
						}
					}
					if (randomdir == 1)
					{
						if (NANOSIZE != 0)
						{
							for (resloop = 1; resloop <= RESOLUTION; resloop++)
							{
								for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
								{
									for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
									{
										for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
										{
											if ((((((Gridx[upnode]+resloop)%DENSITY)) == nloopx) || ((((Gridx[upnode]+resloop)%DENSITY)) == (nloopx-DENSITY))) &&
												(((((Gridy[upnode])%DENSITY)) == nloopy) || ((((Gridy[upnode])%DENSITY)) == (nloopy-DENSITY))) &&
												(((((Gridz[upnode])%DENSITY)) == nloopz) || ((((Gridz[upnode])%DENSITY)) == (nloopz-DENSITY)))) block = 1;
										}
									}
								}
							}
						}
						if (block == 0)
						{
							Gridx[currentnode] = Gridx[upnode];
							Gridy[currentnode] = Gridy[upnode];
							Gridz[currentnode] = Gridz[upnode];
							Gridx[currentnode]++;
						}
					}
					if (randomdir == 2)
					{
						if (NANOSIZE != 0)
						{
							for (resloop = 1; resloop <= RESOLUTION; resloop++)
							{
								for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
								{
									for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
									{
										for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
										{
											if ((((((Gridx[upnode])%DENSITY)) == nloopx) || ((((Gridx[upnode])%DENSITY)) == (nloopx-DENSITY))) &&
												(((((Gridy[upnode]-resloop)%DENSITY)) == nloopy) || ((((Gridy[upnode]-resloop)%DENSITY)) == (nloopy-DENSITY))) &&
												(((((Gridz[upnode])%DENSITY)) == nloopz) || ((((Gridz[upnode])%DENSITY)) == (nloopz-DENSITY)))) block = 1;
										}
									}
								}
							}
						}
						if (block == 0)
						{
							Gridx[currentnode] = Gridx[upnode];
							Gridy[currentnode] = Gridy[upnode];
							Gridz[currentnode] = Gridz[upnode];
							Gridy[currentnode]--;
						}
					}
					if (randomdir == 3)
					{
						if (NANOSIZE != 0)
						{
							for (resloop = 1; resloop <= RESOLUTION; resloop++)
							{
								for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
								{
									for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
									{
										for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
										{
											if ((((((Gridx[upnode])%DENSITY)) == nloopx) || ((((Gridx[upnode])%DENSITY)) == (nloopx-DENSITY))) &&
												(((((Gridy[upnode]+resloop)%DENSITY)) == nloopy) || ((((Gridy[upnode]+resloop)%DENSITY)) == (nloopy-DENSITY))) &&
												(((((Gridz[upnode])%DENSITY)) == nloopz) || ((((Gridz[upnode])%DENSITY)) == (nloopz-DENSITY)))) block = 1;
										}
									}
								}
							}
						}
						if (block == 0)
						{
							Gridx[currentnode] = Gridx[upnode];
							Gridy[currentnode] = Gridy[upnode];
							Gridz[currentnode] = Gridz[upnode];
							Gridy[currentnode]++;
						}
					}
					if (randomdir == 4)
					{
						if (NANOSIZE != 0)
						{
							for (resloop = 1; resloop <= RESOLUTION; resloop++)
							{
								for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
								{
									for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
									{
										for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
										{
											if ((((((Gridx[upnode])%DENSITY)) == nloopx) || ((((Gridx[upnode])%DENSITY)) == (nloopx-DENSITY))) &&
												(((((Gridy[upnode])%DENSITY)) == nloopy) || ((((Gridy[upnode])%DENSITY)) == (nloopy-DENSITY))) &&
												(((((Gridz[upnode]-resloop)%DENSITY)) == nloopz) || ((((Gridz[upnode]-resloop)%DENSITY)) == (nloopz-DENSITY)))) block = 1;
										}
									}
								}
							}
						}
						if (block == 0)
						{
							Gridx[currentnode] = Gridx[upnode];
							Gridy[currentnode] = Gridy[upnode];
							Gridz[currentnode] = Gridz[upnode];
							Gridz[currentnode]--;
						}
					}
					if (randomdir == 5)
					{
						if (NANOSIZE != 0)
						{
							for (resloop = 1; resloop <= RESOLUTION; resloop++)
							{
								for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
								{
									for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
									{
										for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
										{
											if ((((((Gridx[upnode])%DENSITY)) == nloopx) || ((((Gridx[upnode])%DENSITY)) == (nloopx-DENSITY))) &&
												(((((Gridy[upnode])%DENSITY)) == nloopy) || ((((Gridy[upnode])%DENSITY)) == (nloopy-DENSITY))) &&
												(((((Gridz[upnode]+resloop)%DENSITY)) == nloopz) || ((((Gridz[upnode]+resloop)%DENSITY)) == (nloopz-DENSITY)))) block = 1;
										}
									}
								}
							}
						}
						if (block == 0)
						{
							Gridx[currentnode] = Gridx[upnode];
							Gridy[currentnode] = Gridy[upnode];
							Gridz[currentnode] = Gridz[upnode];
							Gridz[currentnode]++;
						}
					}
				} 

				if (currentnode == (POLYLENGTH-1))
				{
					if (randomdir == 0)
					{
						if (NANOSIZE != 0)
						{
							for (resloop = 1; resloop <= RESOLUTION; resloop++)
							{
								for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
								{
									for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
									{
										for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
										{
											if ((((((Gridx[downnode]-resloop)%DENSITY)) == nloopx) || ((((Gridx[downnode]-resloop)%DENSITY)) == (nloopx-DENSITY))) &&
												(((((Gridy[downnode])%DENSITY)) == nloopy) || ((((Gridy[downnode])%DENSITY)) == (nloopy-DENSITY))) &&
												(((((Gridz[downnode])%DENSITY)) == nloopz) || ((((Gridz[downnode])%DENSITY)) == (nloopz-DENSITY)))) block = 1;
										}
									}
								}
							}
						}
						if (block == 0)
						{
							Gridx[currentnode] = Gridx[downnode];
							Gridy[currentnode] = Gridy[downnode];
							Gridz[currentnode] = Gridz[downnode];
							Gridx[currentnode]--;
						}
					}
					if (randomdir == 1)
					{
						if (NANOSIZE != 0)
						{
							for (resloop = 1; resloop <= RESOLUTION; resloop++)
							{
								for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
								{
									for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
									{
										for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
										{
											if ((((((Gridx[downnode]+resloop)%DENSITY)) == nloopx) || ((((Gridx[downnode]+resloop)%DENSITY)) == (nloopx-DENSITY))) &&
												(((((Gridy[downnode])%DENSITY)) == nloopy) || ((((Gridy[downnode])%DENSITY)) == (nloopy-DENSITY))) &&
												(((((Gridz[downnode])%DENSITY)) == nloopz) || ((((Gridz[downnode])%DENSITY)) == (nloopz-DENSITY)))) block = 1;
										}
									}
								}
							}
						}
						if (block == 0)
						{
							Gridx[currentnode] = Gridx[downnode];
							Gridy[currentnode] = Gridy[downnode];
							Gridz[currentnode] = Gridz[downnode];
							Gridx[currentnode]++;
						}
					}
					if (randomdir == 2)
					{
						if (NANOSIZE != 0)
						{
							for (resloop = 1; resloop <= RESOLUTION; resloop++)
							{
								for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
								{
									for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
									{
										for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
										{
											if ((((((Gridx[downnode])%DENSITY)) == nloopx) || ((((Gridx[downnode])%DENSITY)) == (nloopx-DENSITY))) &&
												(((((Gridy[downnode]-resloop)%DENSITY)) == nloopy) || ((((Gridy[downnode]-resloop)%DENSITY)) == (nloopy-DENSITY))) &&
												(((((Gridz[downnode])%DENSITY)) == nloopz) || ((((Gridz[downnode])%DENSITY)) == (nloopz-DENSITY)))) block = 1;
										}
									}
								}
							}
						}
						if (block == 0)
						{
							Gridx[currentnode] = Gridx[downnode];
							Gridy[currentnode] = Gridy[downnode];
							Gridz[currentnode] = Gridz[downnode];
							Gridy[currentnode]--;
						}
					}
					if (randomdir == 3)
					{
						if (NANOSIZE != 0)
						{
							for (resloop = 1; resloop <= RESOLUTION; resloop++)
							{
								for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
								{
									for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
									{
										for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
										{
											if ((((((Gridx[downnode])%DENSITY)) == nloopx) || ((((Gridx[downnode])%DENSITY)) == (nloopx-DENSITY))) &&
												(((((Gridy[downnode]+resloop)%DENSITY)) == nloopy) || ((((Gridy[downnode]+resloop)%DENSITY)) == (nloopy-DENSITY))) &&
												(((((Gridz[downnode])%DENSITY)) == nloopz) || ((((Gridz[downnode])%DENSITY)) == (nloopz-DENSITY)))) block = 1;
										}
									}
								}
							}
						}
						if (block == 0)
						{
							Gridx[currentnode] = Gridx[downnode];
							Gridy[currentnode] = Gridy[downnode];
							Gridz[currentnode] = Gridz[downnode];
							Gridy[currentnode]++;
						}
					}
					if (randomdir == 4)
					{
						if (NANOSIZE != 0)
						{
							for (resloop = 1; resloop <= RESOLUTION; resloop++)
							{
								for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
								{
									for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
									{
										for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
										{
											if ((((((Gridx[downnode])%DENSITY)) == nloopx) || ((((Gridx[downnode])%DENSITY)) == (nloopx-DENSITY))) &&
												(((((Gridy[downnode])%DENSITY)) == nloopy) || ((((Gridy[downnode])%DENSITY)) == (nloopy-DENSITY))) &&
												(((((Gridz[downnode]-resloop)%DENSITY)) == nloopz) || ((((Gridz[downnode]-resloop)%DENSITY)) == (nloopz-DENSITY)))) block = 1;
										}
									}
								}
							}
						}
						if (block == 0)
						{
							Gridx[currentnode] = Gridx[downnode];
							Gridy[currentnode] = Gridy[downnode];
							Gridz[currentnode] = Gridz[downnode];
							Gridz[currentnode]--;
						}
					}
					if (randomdir == 5)
					{
						if (NANOSIZE != 0)
						{
							for (resloop = 1; resloop <= RESOLUTION; resloop++)
							{
								for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
								{
									for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
									{
										for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
										{
											if ((((((Gridx[downnode])%DENSITY)) == nloopx) || ((((Gridx[downnode])%DENSITY)) == (nloopx-DENSITY))) &&
												(((((Gridy[downnode])%DENSITY)) == nloopy) || ((((Gridy[downnode])%DENSITY)) == (nloopy-DENSITY))) &&
												(((((Gridz[downnode]+resloop)%DENSITY)) == nloopz) || ((((Gridz[downnode]+resloop)%DENSITY)) == (nloopz-DENSITY)))) block = 1;
										}
									}
								}
							}
						}
						if (block == 0)
						{
							Gridx[currentnode] = Gridx[downnode];
							Gridy[currentnode] = Gridy[downnode];
							Gridz[currentnode] = Gridz[downnode];
							Gridz[currentnode]++;
						}
					}
				}

				if ((0 < currentnode) && (currentnode < (POLYLENGTH-1)))
				{
					if ((Gridx[downnode] == Gridx[upnode]) && (Gridy[downnode] == Gridy[upnode]) && (Gridz[downnode] == Gridz[upnode]) )
					{
						if (randomdir == 0)
						{
							if (NANOSIZE != 0)
							{
								for (resloop = 1; resloop <= RESOLUTION; resloop++)
								{
									for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
									{
										for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
										{
											for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
											{
												if ((((((Gridx[upnode]-resloop)%DENSITY)) == nloopx) || ((((Gridx[upnode]-resloop)%DENSITY)) == (nloopx-DENSITY))) &&
													(((((Gridy[upnode])%DENSITY)) == nloopy) || ((((Gridy[upnode])%DENSITY)) == (nloopy-DENSITY))) &&
													(((((Gridz[upnode])%DENSITY)) == nloopz) || ((((Gridz[upnode])%DENSITY)) == (nloopz-DENSITY)))) block = 1;
											}
										}
									}
								}
							}
							if (block == 0)
							{
								Gridx[currentnode] = Gridx[upnode];
								Gridy[currentnode] = Gridy[upnode];
								Gridz[currentnode] = Gridz[upnode];
								Gridx[currentnode]--;
							}
						}
						if (randomdir == 1)
						{
							if (NANOSIZE != 0)
							{
								for (resloop = 1; resloop <= RESOLUTION; resloop++)
								{
									for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
									{
										for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
										{
											for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
											{
												if ((((((Gridx[upnode]+resloop)%DENSITY)) == nloopx) || ((((Gridx[upnode]+resloop)%DENSITY)) == (nloopx-DENSITY))) &&
													(((((Gridy[upnode])%DENSITY)) == nloopy) || ((((Gridy[upnode])%DENSITY)) == (nloopy-DENSITY))) &&
													(((((Gridz[upnode])%DENSITY)) == nloopz) || ((((Gridz[upnode])%DENSITY)) == (nloopz-DENSITY)))) block = 1;
											}
										}
									}
								}
							}
							if (block == 0)
							{
								Gridx[currentnode] = Gridx[upnode];
								Gridy[currentnode] = Gridy[upnode];
								Gridz[currentnode] = Gridz[upnode];
								Gridx[currentnode]++;
							}
						}
						if (randomdir == 2)
						{
							if (NANOSIZE != 0)
							{
								for (resloop = 1; resloop <= RESOLUTION; resloop++)
								{
									for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
									{
										for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
										{
											for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
											{
												if ((((((Gridx[upnode])%DENSITY)) == nloopx) || ((((Gridx[upnode])%DENSITY)) == (nloopx-DENSITY))) &&
													(((((Gridy[upnode]-resloop)%DENSITY)) == nloopy) || ((((Gridy[upnode]-resloop)%DENSITY)) == (nloopy-DENSITY))) &&
													(((((Gridz[upnode])%DENSITY)) == nloopz) || ((((Gridz[upnode])%DENSITY)) == (nloopz-DENSITY)))) block = 1;
											}
										}
									}
								}
							}
							if (block == 0)
							{
								Gridx[currentnode] = Gridx[upnode];
								Gridy[currentnode] = Gridy[upnode];
								Gridz[currentnode] = Gridz[upnode];
								Gridy[currentnode]--;
							}
						}
						if (randomdir == 3)
						{
							if (NANOSIZE != 0)
							{
								for (resloop = 1; resloop <= RESOLUTION; resloop++)
								{
									for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
									{
										for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
										{
											for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
											{
												if ((((((Gridx[upnode])%DENSITY)) == nloopx) || ((((Gridx[upnode])%DENSITY)) == (nloopx-DENSITY))) &&
													(((((Gridy[upnode]+resloop)%DENSITY)) == nloopy) || ((((Gridy[upnode]+resloop)%DENSITY)) == (nloopy-DENSITY))) &&
													(((((Gridz[upnode])%DENSITY)) == nloopz) || ((((Gridz[upnode])%DENSITY)) == (nloopz-DENSITY)))) block = 1;
											}
										}
									}
								}
							}
							if (block == 0)
							{
								Gridx[currentnode] = Gridx[upnode];
								Gridy[currentnode] = Gridy[upnode];
								Gridz[currentnode] = Gridz[upnode];
								Gridy[currentnode]++;
							}
						}
						if (randomdir == 4)
						{
							if (NANOSIZE != 0)
							{
								for (resloop = 1; resloop <= RESOLUTION; resloop++)
								{
									for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
									{
										for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
										{
											for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
											{
												if ((((((Gridx[upnode])%DENSITY)) == nloopx) || ((((Gridx[upnode])%DENSITY)) == (nloopx-DENSITY))) &&
													(((((Gridy[upnode])%DENSITY)) == nloopy) || ((((Gridy[upnode])%DENSITY)) == (nloopy-DENSITY))) &&
													(((((Gridz[upnode]-resloop)%DENSITY)) == nloopz) || ((((Gridz[upnode]-resloop)%DENSITY)) == (nloopz-DENSITY)))) block = 1;
											}
										}
									}
								}
							}
							if (block == 0)
							{
								Gridx[currentnode] = Gridx[upnode];
								Gridy[currentnode] = Gridy[upnode];
								Gridz[currentnode] = Gridz[upnode];
								Gridz[currentnode]--;
							}
						}
						if (randomdir == 5)
						{
							if (NANOSIZE != 0)
							{
								for (resloop = 1; resloop <= RESOLUTION; resloop++)
								{
									for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
									{
										for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
										{
											for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
											{
												if ((((((Gridx[upnode])%DENSITY)) == nloopx) || ((((Gridx[upnode])%DENSITY)) == (nloopx-DENSITY))) &&
													(((((Gridy[upnode])%DENSITY)) == nloopy) || ((((Gridy[upnode])%DENSITY)) == (nloopy-DENSITY))) &&
													(((((Gridz[upnode]+resloop)%DENSITY)) == nloopz) || ((((Gridz[upnode]+resloop)%DENSITY)) == (nloopz-DENSITY)))) block = 1;
											}
										}
									}
								}
							}
							if (block == 0)
							{
								Gridx[currentnode] = Gridx[upnode];
								Gridy[currentnode] = Gridy[upnode];
								Gridz[currentnode] = Gridz[upnode];
								Gridz[currentnode]++;
							}
						}
					}
				}
				//-------------------------------------------------------------------block changes ------------------------------------------------------//
			}
			if (y == SUBDIFFUSE)
			{
				for (a=0; a < POLYLENGTH; a++)
				{
					smidx = smidx + (float)Gridx[a];
					smidy = smidy + (float)Gridy[a];
					smidz = smidz + (float)Gridz[a];
				}
			}
			//if ((y % 10) == 0 )
			if ((((y % 10) == 0) && (y >= (SUBDIFFUSE+10)) && (y <= (SUBDIFFUSE+(SUBDIFFUSE/10)))) ||
				((y % 250) == 0 && (y <= (DIFFUSE)) && (y > (SUBDIFFUSE+(SUBDIFFUSE/10)))) ||
				((y % 5000) == 0 && (y <= (DIFFUSE*5)) && (y > DIFFUSE)) || ((y % 30000) == 0 && (y > (DIFFUSE*5))))
			{
				endtoend = sqrt((float)((Gridx[POLYLENGTH - 1] - Gridx[0]) * (Gridx[POLYLENGTH - 1] - Gridx[0]) +
					(Gridy[POLYLENGTH - 1] - Gridy[0]) * (Gridy[POLYLENGTH - 1] - Gridy[0]) +
					(Gridz[POLYLENGTH - 1] - Gridz[0]) * (Gridz[POLYLENGTH - 1] - Gridz[0])));

				xSum = 0;
				ySum = 0;
				zSum = 0;
				vrad = 0;

				for (a=0; a < POLYLENGTH; a++)
				{
					xSum = xSum + (float)Gridx[a];
					ySum = ySum + (float)Gridy[a];
					zSum = zSum + (float)Gridz[a];
				}
				xSum = (xSum)/float(POLYLENGTH);
				ySum = (ySum)/float(POLYLENGTH);
				zSum = (zSum)/float(POLYLENGTH);
				for (a=0;a<POLYLENGTH;a++)
				{
					vrad = vrad + (((float(Gridx[a])) - xSum)*((float(Gridx[a])) - xSum)) +
						(((float(Gridy[a])) - ySum)*((float(Gridy[a])) - ySum)) +
						(((float(Gridz[a])) - zSum)*((float(Gridz[a])) - zSum));
				}

				radofgy = sqrt(vrad/(float(POLYLENGTH)));

				xSum = xSum - (smidx/float(POLYLENGTH));
				ySum = ySum - (smidy/float(POLYLENGTH));
				zSum = zSum - (smidz/float(POLYLENGTH));

				beadtomid = ((xSum*xSum)+(ySum*ySum)+(zSum*zSum));

				float* datapointdata1 = (float*)(((char*)placeend) + (datapointindex*pitch));
				float* datapointdata2 = (float*)(((char*)placebead) + (datapointindex*pitch));
				float* datapointdata3 = (float*)(((char*)placerad) + (datapointindex*pitch));

				datapointdata1[idx] = endtoend;
				datapointdata2[idx] = beadtomid;
				datapointdata3[idx] = radofgy;

				d_endtoends[(datapointindex*NoPOLY) + idx] = datapointdata1[idx];
				d_beadtomids[(datapointindex*NoPOLY) + idx] = datapointdata2[idx];
				d_radofgys[(datapointindex*NoPOLY) + idx] = datapointdata3[idx];

				datapointindex++;
			}
		}
	}
}

statistics::statistics() {
	n=0;
	sum=0.0;
	sumsq=0.0;
}

int statistics::getNumber() const{
	return n;
}

float statistics::getAverage() const {
	if(n==0) return 0.;
	return sum/n;
}

float statistics::getSqAverage() const {
	if(n==0) return -1;
	return sumsq/n;
}

void statistics::add(float x) {
	n++;
	sum += x;
	sumsq += x*x;
}

//-----------------------------------------------------------MAIN PROGRAM------------------------------------------------------------------------------------//

int main()
{
	long startTime = clock();
	long starttime2;
	long finishtime2;
	int y = 0;
	int ig = 0;
	int jg = 0;
	int check1 = (((SUBDIFFUSE+(SUBDIFFUSE/10)) - (SUBDIFFUSE))/10);
	int check2 = (((DIFFUSE)  - (SUBDIFFUSE+(SUBDIFFUSE/10)))/250);
	int check3 = (((DIFFUSE*5) - (DIFFUSE))/5000);
	int check4 = (((TIMESTEPS) - (DIFFUSE*5))/30000);

	float *placeend;
	float *placebead;
	float *placerad;
	float *d_endtoends;
	float *d_beadtomids;
	float *d_radofgys;
	float *h_endtoends = new float[DATAPOINTS*NoPOLY];
	float *h_beadtomids = new float[DATAPOINTS*NoPOLY];
	float *h_radofgys = new float[DATAPOINTS*NoPOLY];
	curandState* randStates;
	size_t pitch;

	cudaMallocPitch(&placeend, &pitch, sizeof(float)*NoPOLY, DATAPOINTS);
	cudaMalloc(&d_endtoends, sizeof(float)*NoPOLY*DATAPOINTS);

	cudaMallocPitch(&placebead, &pitch, sizeof(float)*NoPOLY, DATAPOINTS);
	cudaMalloc(&d_beadtomids, sizeof(float)*NoPOLY*DATAPOINTS);

	cudaMallocPitch(&placerad, &pitch, sizeof(float)*NoPOLY, DATAPOINTS);
	cudaMalloc(&d_radofgys, sizeof(float)*NoPOLY*DATAPOINTS);

	cudaMalloc(&randStates, NoPOLY*sizeof(curandState));

	cout << "Should be integers:" << check1 << "," << check2 << "," << check3 << "," << check4 << endl;

	ofstream outfile;
	outfile.open ("81-(XX---).txt"); //*************************************************************************************************************************************************PROGRAM NAME
	if (!outfile.is_open())
	{
		cout << "file not open" << endl;
		return 666;
	}

	outfile << "TimeStep " << "E2EDistance " << "RadofGy " << "R^2 " << "log10(TimeStep) " << "log10(R^2) " << endl;

	//-----------------------------------------------------------KERNAL CALL------------------------------------------------------------------------------------//

	starttime2 = clock();

	cudarandomwalk<<<112, 8>>>(placeend, d_endtoends, placebead, d_beadtomids, placerad, d_radofgys, pitch, randStates, time(NULL));

	cudaMemcpy(h_endtoends, d_endtoends, sizeof(float)*NoPOLY*DATAPOINTS, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_beadtomids, d_beadtomids, sizeof(float)*NoPOLY*DATAPOINTS, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_radofgys, d_radofgys, sizeof(float)*NoPOLY*DATAPOINTS, cudaMemcpyDeviceToHost);

	finishtime2 = clock();
	cout<<endl<<"Kernal Run time is "<<((finishtime2 - starttime2)/double(CLOCKS_PER_SEC))<<" seconds"<<endl<<endl;

	for(y = (SUBDIFFUSE - 10) ; y < TIMESTEPS; y++)  
    //for(y = (0) ; y < TIMESTEPS; y++)  
	{ 
		statistics rsq;
		statistics flength;
		statistics rgst;
		if ((((y % 10) == 0) && (y >= (SUBDIFFUSE+10)) && (y <= (SUBDIFFUSE+(SUBDIFFUSE/10)))) ||
			((y % 250) == 0 && (y <= (DIFFUSE)) && (y > (SUBDIFFUSE+(SUBDIFFUSE/10)))) ||
			((y % 5000) == 0 && (y <= (DIFFUSE*5)) && (y > DIFFUSE)) || ((y % 30000) == 0 && (y > (DIFFUSE*5))))
		//if ((y % 10) == 0)
		{
			for(ig = 0 ; ig < NoPOLY; ig++)  
			{ 
				flength.add(h_endtoends[(jg*NoPOLY)+ig]);
				rsq.add(h_beadtomids[(jg*NoPOLY)+ig]);
				rgst.add(h_radofgys[(jg*NoPOLY)+ig]);
			}
			jg++;
			outfile << y-SUBDIFFUSE << " " <<  flength.getAverage() << " " <<  rgst.getAverage()  << " "  << rsq.getAverage() << " " << log10((float)(y-SUBDIFFUSE)) << " " << log10(rsq.getAverage()) << endl;
			//outfile << y << " " <<  flength.getAverage() << " " <<  rgst.getAverage()  << " "  << rsq.getAverage() << " " << log10((float)(y)) << " " << log10(rsq.getAverage()) << endl;
		}
	} 

	cout << "_____________" << endl << "END OF RANDOMWALK" << endl;

	cudaFree(d_endtoends);
	cudaFree(d_beadtomids);
	cudaFree(d_radofgys);
	cudaFree(placeend);
	cudaFree(placebead);
	cudaFree(placerad);
	cudaFree(randStates);

	cudaDeviceReset();
	outfile.close();
	long finishTime = clock();
	cout<<endl<<"Program Run time is "<<((finishTime - startTime)/double(CLOCKS_PER_SEC))<<" seconds"<<endl<<endl;
	//system("PAUSE");
	return 0;
}
