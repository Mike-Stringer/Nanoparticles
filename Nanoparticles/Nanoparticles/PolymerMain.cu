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

#define TIMESTEPS 15000000      // can be put down to 10000 for testing purposes with a change to the y bits and datapoints to %10 (already there just commented out)
#define SUBDIFFUSE 500000         //21, do y=100000 41, do y=200000 61, do y=500000 81, do y=750000 121, do y=2000000 101, do y=1600000
#define DIFFUSE 1000000        //81, do y=1250000 101, do y=2000000
//#define TIMESTEPS 1500000    // can be put down to 10000 for testing purposes with a change to the y bits and datapoints to %10 (already there just commented out)
//#define SUBDIFFUSE 500        //21, do y=100000 41, do y=200000 61, do y=500000 81, do y=750000 121, do y=2000000 101, do y=1600000
//#define DIFFUSE 1000     //81, do y=1250000 101, do y=2000000
#define DATAPOINTS (((((SUBDIFFUSE+(SUBDIFFUSE/10)) - (SUBDIFFUSE))/10) + (((DIFFUSE)  - (SUBDIFFUSE+(SUBDIFFUSE/10)))/250) + (((DIFFUSE*5) - (DIFFUSE))/5000) + (((TIMESTEPS) - (DIFFUSE*5))/30000))+10)
//#define DATAPOINTS ((TIMESTEPS/10)+10)

//(((SUBDIFFUSE+(SUBDIFFUSE/10)) - (SUBDIFFUSE))/10)
//(((DIFFUSE)  - (SUBDIFFUSE+(SUBDIFFUSE/10)))/250)
//(((DIFFUSE*5) - (DIFFUSE))/5000)
//(((TIMESTEPS) - (DIFFUSE*5))/30000)

//length of polymer

#define POLYLENGTH 21
//#define RESOLUTION 1                  //segment length //carefull drastically reduces number of conformations
#define NANOSIZE 2              //size of nanoparticle
#define DENSITY 10               //NANOSIZE:DENSITY makes the ratio, with 1, 2, 3 .......DENSITY    where NANOSIZE fills 1 if 1,  1,2 if 2 1,2,3 if 3 etc etc.

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

		int Randomx[NANOSIZE];
		int Randomy[NANOSIZE];
		int Randomz[NANOSIZE];
		int b;

		int startx = (int)(curand_uniform(&randState)*1000) + 1000;
		int starty = (int)(curand_uniform(&randState)*1000) + 1000;
		int startz = (int)(curand_uniform(&randState)*1000) + 1000;


		for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
		{
			Randomx[nloopx] = 0;
			Randomy[nloopx] = 0;
			Randomz[nloopx] = 0;
		}

		for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
		{
			randomdir = (int)(curand_uniform(&randState)*DENSITY);

			Randomx[nloopx] = randomdir;

			if (nloopx>0)
			{
				for (nloopy = 0; nloopy < (nloopx); nloopy++)
				{
					if (Randomx[nloopx] == Randomx[nloopy]) nloopx--;
				}
			}
		}

		for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
		{
			randomdir = (int)(curand_uniform(&randState)*DENSITY);

			Randomy[nloopx] = randomdir;

			if (nloopx>0)
			{
				for (nloopy = 0; nloopy < (nloopx); nloopy++)
				{
					if (Randomy[nloopx] == Randomy[nloopy]) nloopx--;
				}
			}
		}

		for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
		{
			randomdir = (int)(curand_uniform(&randState)*DENSITY);

			Randomz[nloopx] = randomdir;

			if (nloopx>0)
			{
				for (nloopy = 0; nloopy < (nloopx); nloopy++)
				{
					if (Randomz[nloopx] == Randomz[nloopy]) nloopx--;
				}
			}
		}

		for (a=0; a < POLYLENGTH; a++)
		{
			Gridx[a] = startx;
			Gridy[a] = starty;
			Gridz[a] = startz;
		}


		block=0;
		resloop=0;
		for(b=1; b < 2; b++)
		{
			block=0;
			for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
			{
				for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
				{
					for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
					{
						if (((((Gridx[a]+resloop)%DENSITY) == nloopx))
							&& ((((Gridy[a]+resloop)%DENSITY) == nloopy))
							&& ((((Gridz[a]+resloop)%DENSITY) == nloopz))) block = 1;
					}
				}
			}
			if (block == 1)
			{
				b--;
				resloop++;
			}
			if (block == 0)
			{
				for (a=0; a < POLYLENGTH; a++)
				{
					Gridx[a]= Gridx[a]+resloop;
					Gridy[a]= Gridy[a]+resloop;
					Gridz[a]= Gridz[a]+resloop;
				}
				b++;
			}
		}


		int incx[3] = { 1, 0, 0 };
		int incy[3] = { 0, 1, 0 };
		int incz[3] = { 0, 0, 1 };
		int inc2x[6] = { -1, 1, 0, 0, 0, 0 };
		int inc2y[6] = { 0, 0, -1, 1, 0, 0 };
		int inc2z[6] = { 0, 0, 0, 0, -1, 1 };

		for(a=0; a < POLYLENGTH; a++)
		{

			randomdir = (int)(curand_uniform(&randState)*3);
			block = 0;


			for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
			{
				for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
				{
					for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
					{
						if (((((Gridx[POLYLENGTH-1]+incx[randomdir])%DENSITY) == Randomx[nloopx])) && ((((Gridy[POLYLENGTH-1]+incy[randomdir])%DENSITY) == Randomy[nloopy])) && ((((Gridz[POLYLENGTH-1]+incz[randomdir])%DENSITY) == Randomz[nloopz]))) block = 1;
					}
				}
			}

			if (block == 0)
			{
				Gridx[POLYLENGTH-1]+=incx[randomdir];
				Gridy[POLYLENGTH-1]+=incy[randomdir];
				Gridz[POLYLENGTH-1]+=incz[randomdir];
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
					for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
					{
						for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
						{
							for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
							{
								if ((((((Gridx[upnode]+inc2x[randomdir])%DENSITY)) == Randomx[nloopx]) || ((((Gridx[upnode]+inc2x[randomdir])%DENSITY)) == ( Randomx[nloopx]-DENSITY))) &&
									(((((Gridy[upnode]+inc2y[randomdir])%DENSITY)) == Randomy[nloopy]) || ((((Gridy[upnode]+inc2y[randomdir])%DENSITY)) == ( Randomy[nloopy]-DENSITY))) &&
									(((((Gridz[upnode]+inc2z[randomdir])%DENSITY)) == Randomz[nloopz]) || ((((Gridz[upnode]+inc2z[randomdir])%DENSITY)) == ( Randomz[nloopz]-DENSITY)))) block = 1;
							}
						}
					}

					if (block == 0)
					{
						Gridx[currentnode] = Gridx[upnode];
						Gridy[currentnode] = Gridy[upnode];
						Gridz[currentnode] = Gridz[upnode];
						Gridx[currentnode]+=inc2x[randomdir];
						Gridy[currentnode]+=inc2y[randomdir];
						Gridz[currentnode]+=inc2z[randomdir];
					}

				} 

				if (currentnode == (POLYLENGTH-1))
				{
					for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
					{
						for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
						{
							for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
							{
								if ((((((Gridx[downnode]+inc2x[randomdir])%DENSITY)) == Randomx[nloopx]) || ((((Gridx[downnode]+inc2x[randomdir])%DENSITY)) == ( Randomx[nloopx]-DENSITY))) &&
									(((((Gridy[downnode]+inc2y[randomdir])%DENSITY)) == Randomy[nloopy]) || ((((Gridy[downnode]+inc2y[randomdir])%DENSITY)) == ( Randomy[nloopy]-DENSITY))) &&
									(((((Gridz[downnode]+inc2z[randomdir])%DENSITY)) == Randomz[nloopz]) || ((((Gridz[downnode]+inc2z[randomdir])%DENSITY)) == ( Randomz[nloopz]-DENSITY)))) block = 1;
							}
						}
					}

					if (block == 0)
					{
						Gridx[currentnode] = Gridx[downnode];
						Gridy[currentnode] = Gridy[downnode];
						Gridz[currentnode] = Gridz[downnode];
						Gridx[currentnode]+=inc2x[randomdir];
						Gridy[currentnode]+=inc2y[randomdir];
						Gridz[currentnode]+=inc2z[randomdir];
					}
				}

				if ((0 < currentnode) && (currentnode < (POLYLENGTH-1)))
				{
					if ((Gridx[downnode] == Gridx[upnode]) && (Gridy[downnode] == Gridy[upnode]) && (Gridz[downnode] == Gridz[upnode]) )
					{
						for (nloopx = 0; nloopx < NANOSIZE; nloopx++)
						{
							for (nloopy = 0; nloopy < NANOSIZE; nloopy++)
							{
								for (nloopz = 0; nloopz < NANOSIZE; nloopz++)
								{
									if ((((((Gridx[upnode]+inc2x[randomdir])%DENSITY)) == Randomx[nloopx]) || ((((Gridx[upnode]+inc2x[randomdir])%DENSITY)) == ( Randomx[nloopx]-DENSITY))) &&
										(((((Gridy[upnode]+inc2y[randomdir])%DENSITY)) == Randomy[nloopy]) || ((((Gridy[upnode]+inc2y[randomdir])%DENSITY)) == ( Randomy[nloopy]-DENSITY))) &&
										(((((Gridz[upnode]+inc2z[randomdir])%DENSITY)) == Randomz[nloopz]) || ((((Gridz[upnode]+inc2z[randomdir])%DENSITY)) == ( Randomz[nloopz]-DENSITY)))) block = 1;
								}
							}
						}

						if (block == 0)
						{
							Gridx[currentnode] = Gridx[upnode];
							Gridy[currentnode] = Gridy[upnode];
							Gridz[currentnode] = Gridz[upnode];
							Gridx[currentnode]+=inc2x[randomdir];
							Gridy[currentnode]+=inc2y[randomdir];
							Gridz[currentnode]+=inc2z[randomdir];
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
	float check1 = (((SUBDIFFUSE+(SUBDIFFUSE/10)) - (SUBDIFFUSE))/10);
	float check2 = (((DIFFUSE)  - (SUBDIFFUSE+(SUBDIFFUSE/10)))/250);
	float check3 = (((DIFFUSE*5) - (DIFFUSE))/5000);
	float check4 = (((TIMESTEPS) - (DIFFUSE*5))/30000);

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
	outfile.open ("Fullrandomtest.txt"); //*************************************************************************************************************************************************PROGRAM NAME
	if (!outfile.is_open())
	{
		cout << "file not open" << endl;
		return 666;
	}

	outfile << "TimeStep " << "E2EDistance " << "ErrorE2E " << "RadofGy " << "ErrorRadofgy " << "R^2 " << "ErrorR^2 " << "log10(TimeStep) " << "log10(R^2) " << "ErrorLog10(R^2) " << endl;

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
			//outfile << y-SUBDIFFUSE << " " <<  flength.getAverage() << " " <<  rgst.getAverage()  << " "  << rsq.getAverage() << " " << log10((float)(y-SUBDIFFUSE)) << " " << log10(rsq.getAverage()) << endl;
			//REMEMBER - SUBDIFFUSE
			outfile << (y - SUBDIFFUSE) << " " <<  flength.getAverage() << " "  << (sqrt(flength.getSqAverage() - (flength.getAverage()*flength.getAverage()))/((float)NoPOLY)) 
				<< " " <<  rgst.getAverage()  << " " << (sqrt(rgst.getSqAverage() - (rgst.getAverage()*rgst.getAverage()))/((float)NoPOLY)) << " "  
				<< rsq.getAverage() << " " << (sqrt(rsq.getSqAverage() - (rsq.getAverage()*rsq.getAverage()))/((float)NoPOLY))  << " " << log10((float)(y - SUBDIFFUSE)) << " " << log10(rsq.getAverage()) 
				<< "  " << (sqrt(rsq.getSqAverage() - (rsq.getAverage()*rsq.getAverage()))/((float)NoPOLY))/(rsq.getAverage()*2.302585) << endl;
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
