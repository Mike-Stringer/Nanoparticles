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

#define NoPOLY 512

//Timing stuff

#define TIMESTEPS 750000
#define SUBDIFFUSE 80000         //21, do y=100000 41, do y=200000 61, do y=500000 81, do y=750000 121, do y=2000000 101, do y=1600000
#define DIFFUSE 120000           //81, do y=1250000 101, do y=2000000
#define DATAPOINTS (((((SUBDIFFUSE+(SUBDIFFUSE/10)) - (SUBDIFFUSE))/10) + (((DIFFUSE)  - (SUBDIFFUSE+(SUBDIFFUSE/10)))/250) + (((DIFFUSE*5) - (DIFFUSE))/5000) + (((TIMESTEPS) - (DIFFUSE*5))/30000))+10)
//#define DATAPOINTS ((TIMESTEPS/10)+10)

//(((SUBDIFFUSE+(SUBDIFFUSE/10)) - (SUBDIFFUSE))/10)
//(((DIFFUSE)  - (SUBDIFFUSE+(SUBDIFFUSE/10)))/250) 
//(((DIFFUSE*5) - (DIFFUSE))/5000)
//(((TIMESTEPS) - (DIFFUSE*5))/30000)
	
//length of polymer

#define POLYLENGTH 21
#define RESOLUTION 1                  //segment length //carefull drastically reduces number of conformations
#define NANOSIZE 0              //size of nanoparticle
#define DENSITY 2               //NANOSIZE:DENSITY makes the ratio, with 1, 2, 3 .......DENSITY    where NANOSIZE fills 1 if 1,  1,2 if 2 1,2,3 if 3 etc etc.


class Managed
{
public:
  void *operator new(size_t len) {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    return ptr;
  }

  void operator delete(void *ptr) {
    cudaFree(ptr);
  }
};
	

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
	
struct Polymer : public Managed{
              //IF YOU NEED TO SPLIT KERNALS YOU WILL NEED THESE
			  //int Gridx[POLYLENGTH];
			  //int Gridy[POLYLENGTH];
			  //int Gridz[POLYLENGTH];
			  float endtoends[DATAPOINTS];
			  float beadtomids[DATAPOINTS];
			  float radofgys[DATAPOINTS];
			  //IF YOU NEED TO SPLIT KERNALS YOU WILL NEED THESE
			  //float smidx;
			  //float smidy;
			  //float smidz;
             };
	
__global__ void cudarandomwalk(Polymer *polymers, curandState *randStates, unsigned long seed) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < NoPOLY)	
	{
		Polymer& polymer = polymers[idx];
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
        int iran = 0;
		float xSum = 0;
		float ySum = 0;
		float zSum = 0;
		float radx = 0; 
		float rady = 0;
		float radz = 0;
        float vrad = 0;
	    float dx = 0;
		float dy = 0;
		float dz = 0;
        int Gridx[POLYLENGTH];
	    int Gridy[POLYLENGTH];
	    int Gridz[POLYLENGTH];

		//int hhg = 0;
		//int hhj = 0;
		//int hhk = 0;

        for (a=0; a < POLYLENGTH; a++)
				{
					Gridx[a] = NANOSIZE;
					Gridy[a] = NANOSIZE;
				    Gridz[a] = NANOSIZE;
			    }


		for(a=0; a < POLYLENGTH; a++)
		{
			//if (iran == 3); iran = 0; //DEBRENCHING CHECK###################################
			//hhg = (int)(curand_uniform(&randState)*3);
			iran = (int)(curand_uniform(&randState)*3);
			block = 0;

			if(iran==2)
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

				if(iran==1)
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

					if(iran==0)
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
				//iran++; //DEBRANCHING CHECKKKKKKKKKKKKKKKKKKKKKKKKKK
	    }

		for (int y=0; y<=TIMESTEPS; y++)
		{

			for (int z=0; z<(POLYLENGTH); z++)
			{
				block = 0;

				//if (currentnode == POLYLENGTH) currentnode = 0;  //DEBRANCHING CHECK############################
                //if (randomdir == 6) randomdir = 0;

				//hhj = (int)(curand_uniform(&randState)*POLYLENGTH);
				//hhk = (int)(curand_uniform(&randState)*6);

				currentnode = (int)(curand_uniform(&randState)*POLYLENGTH);
				randomdir = (int)(curand_uniform(&randState)*6);

				upnode=(currentnode+1);
				downnode=(currentnode-1);

				if (currentnode == 0)
				{
					Gridx[currentnode] = Gridx[upnode];
					Gridy[currentnode] = Gridy[upnode];
					Gridz[currentnode] = Gridz[upnode];
					if (randomdir == 0) Gridx[currentnode]--;          
					else if (randomdir == 1) Gridx[currentnode]++;        
					else if (randomdir == 2) Gridy[currentnode]--;
					else if (randomdir == 3) Gridy[currentnode]++; 
					else if (randomdir == 4) Gridz[currentnode]--;   
					else Gridz[currentnode]++;    
				}

				if (currentnode == (POLYLENGTH-1))
				{
					Gridx[currentnode] = Gridx[downnode];
					Gridy[currentnode] = Gridy[downnode];
					Gridz[currentnode] = Gridz[downnode];
					if (randomdir == 0) Gridx[currentnode]--;          
					else if (randomdir == 1) Gridx[currentnode]++;        
					else if (randomdir == 2) Gridy[currentnode]--;
					else if (randomdir == 3) Gridy[currentnode]++; 
					else if (randomdir == 4) Gridz[currentnode]--;   
					else Gridz[currentnode]++;    
				}

				if ((0 < currentnode) && (currentnode < (POLYLENGTH-1)))
				{
					if ((Gridx[downnode] == Gridx[upnode]) && (Gridy[downnode] == Gridy[upnode]) && (Gridz[downnode] == Gridz[upnode]) )
					{
					Gridx[currentnode] = Gridx[upnode];
					Gridy[currentnode] = Gridy[upnode];
					Gridz[currentnode] = Gridz[upnode];
					if (randomdir == 0) Gridx[currentnode]--;          
					else if (randomdir == 1) Gridx[currentnode]++;        
					else if (randomdir == 2) Gridy[currentnode]--;
					else if (randomdir == 3) Gridy[currentnode]++; 
					else if (randomdir == 4) Gridz[currentnode]--;   
					else Gridz[currentnode]++;    
					}			
				}
            //currentnode++;
			//randomdir++;            //DEBRANCHING CHECK############################
			}
			if (y == SUBDIFFUSE)
			{
				//float smidx = 0;
				//float smidy = 0;
				//float smidz = 0;
				for (a=0; a < POLYLENGTH; a++)
				{
					 smidx = smidx + (float)Gridx[a];
					 smidy = smidy + (float)Gridy[a];
					 smidz = smidz + (float)Gridz[a];
				}
				//polymer.smidx = smidx;
				//polymer.smidy = smidy;
				//polymer.smidz = smidz;
			}
			//if ((y % 10) == 0 )
			if ((((y % 10) == 0) && (y >= (SUBDIFFUSE+10)) && (y <= (SUBDIFFUSE+(SUBDIFFUSE/10)))) || 
            ((y % 250) == 0 && (y <= (DIFFUSE)) && (y > (SUBDIFFUSE+(SUBDIFFUSE/10)))) || 
            ((y % 5000) == 0 && (y <= (DIFFUSE*5)) && (y > DIFFUSE)) || ((y % 30000) == 0 && (y > (DIFFUSE*5))))
			{
				dx = 0;
				dy = 0;
				dz = 0;

				dx = Gridx[POLYLENGTH - 1] - Gridx[0];
				dy = Gridy[POLYLENGTH - 1] - Gridy[0];
				dz = Gridz[POLYLENGTH - 1] - Gridz[0];
			    
				endtoend = sqrt(dx * dx + dy * dy + dz * dz);

				xSum = 0;
				ySum = 0;
				zSum = 0;
				radx = 0; 
				rady = 0;
				radz = 0;
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
				 radx = (((float(Gridx[a])) - xSum)*((float(Gridx[a])) - xSum));
				 rady = (((float(Gridy[a])) - ySum)*((float(Gridy[a])) - ySum));
				 radz = (((float(Gridz[a])) - zSum)*((float(Gridz[a])) - zSum));
				 vrad = vrad + radx + rady + radz;
				 }

				 radofgy = sqrt(vrad/(float(POLYLENGTH)));

				 //xSum = xSum - (polymer.smidx/float(POLYLENGTH));
				 //ySum = ySum - (polymer.smidy/float(POLYLENGTH));
				 //zSum = zSum - (polymer.smidz/float(POLYLENGTH));

                 xSum = xSum - (smidx/float(POLYLENGTH));
				 ySum = ySum - (smidy/float(POLYLENGTH));
				 zSum = zSum - (smidz/float(POLYLENGTH));

				 beadtomid = ((xSum*xSum)+(ySum*ySum)+(zSum*zSum));

				 polymer.endtoends[datapointindex] = endtoend;
				 polymer.beadtomids[datapointindex] = beadtomid;
				 polymer.radofgys[datapointindex] = radofgy;
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
	int i;
	int y;
	int x;
	int q;
	int check1 = (((SUBDIFFUSE+(SUBDIFFUSE/10)) - (SUBDIFFUSE))/10);
	int check2 = (((DIFFUSE)  - (SUBDIFFUSE+(SUBDIFFUSE/10)))/250);
	int check3 = (((DIFFUSE*5) - (DIFFUSE))/5000);
	int check4 = (((TIMESTEPS) - (DIFFUSE*5))/30000);

	cout << "Should be integers:" << check1 << "," << check2 << "," << check3 << "," << check4 << endl;

	srand(time(NULL));

	ofstream outfile;
    outfile.open ("TEST8.txt"); //*************************************************************************************************************************************************PROGRAM NAME
    if (!outfile.is_open())
    { 
	    cout << "file not open" << endl;
		return 666;
	}

    outfile << "TimeStep " << "E2EDistance " << "RadofGy " << "R^2 " << "log10(TimeStep) " << "log10(R^2) " << endl;

	cout << "_____________" << endl << "STARTING STATS ..." << endl;

		int polycount = NoPOLY;
		Polymer *Allpoly; 
		cudaMallocManaged(&Allpoly, polycount * sizeof(Polymer));
		curandState* randStates;
        cudaMalloc ( &randStates, polycount*sizeof( curandState ) );
	
		    for (i=0; i < polycount; i++) 
			{
				for (q=0; q < DATAPOINTS; q++)
				{
					Allpoly[i].endtoends[q]=0;
			        Allpoly[i].beadtomids[q]=0;
                    Allpoly[i].radofgys[q]=0;
			    }
            }

//-----------------------------------------------------------KERNAL CALL------------------------------------------------------------------------------------//
		    starttime2 = clock();
			cudarandomwalk<<<8, 64>>>(Allpoly, randStates, time(NULL));  //(polycount/(256))+1
			cudaDeviceSynchronize();
			finishtime2 = clock();
			cout<<endl<<"Kernal Run time is "<<((finishtime2 - starttime2)/double(CLOCKS_PER_SEC))<<" seconds"<<endl<<endl;

            int datapointindex = 0;
			for (y=0; y<=TIMESTEPS; y++)
			{
				statistics rsq;
                statistics flength;
				statistics rgst;

				//if ((y % 10) == 0 )
			    if ((((y % 10) == 0) && (y >= (SUBDIFFUSE+10)) && (y <= (SUBDIFFUSE+(SUBDIFFUSE/10)))) || 
                ((y % 250) == 0 && (y <= (DIFFUSE)) && (y > (SUBDIFFUSE+(SUBDIFFUSE/10)))) || 
                ((y % 5000) == 0 && (y <= (DIFFUSE*5)) && (y > DIFFUSE)) || ((y % 30000) == 0 && (y > (DIFFUSE*5))))
				{
						for (x=0; x < polycount; x++) 
				        {
                                rsq.add(Allpoly[x].beadtomids[datapointindex]);
                                flength.add(Allpoly[x].endtoends[datapointindex]);
                                rgst.add(Allpoly[x].radofgys[datapointindex]);
				        }
	            datapointindex++;
				}
				//if ((y % 10) == 0 )
			    if ((((y % 10) == 0) && (y >= (SUBDIFFUSE+10)) && (y <= (SUBDIFFUSE+(SUBDIFFUSE/10)))) || 
                ((y % 250) == 0 && (y <= (DIFFUSE)) && (y > (SUBDIFFUSE+(SUBDIFFUSE/10)))) || 
                ((y % 5000) == 0 && (y <= (DIFFUSE*5)) && (y > DIFFUSE)) || ((y % 30000) == 0 && (y > (DIFFUSE*5))))
                {
                                outfile << (y-SUBDIFFUSE) << " " <<  flength.getAverage() << " " <<  rgst.getAverage()  << " "  << rsq.getAverage() << " " << log10((float)(y-SUBDIFFUSE)) << " " << log10(rsq.getAverage()) << endl;
                }
			}

			cout << "_____________" << endl << "END OF RANDOMWALK" << endl;
			          
			cudaFree(Allpoly);
			cudaFree(randStates);
	
	cout << "_____________" << endl << "PRINTING STATS ..." << endl;

	cout << "_____________" << endl << "FINISHED STATS ..." << endl;
	
	cudaDeviceReset();
	outfile.close();
	long finishTime = clock();
	cout<<endl<<"Program Run time is "<<((finishTime - startTime)/double(CLOCKS_PER_SEC))<<" seconds"<<endl<<endl;
	//system("PAUSE");
    return 0;
}




