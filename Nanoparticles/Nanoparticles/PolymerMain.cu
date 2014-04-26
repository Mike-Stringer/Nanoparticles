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

#define NoPOLY 256
//#define MAXCCPOLY 200         //gpu block number

//Timing stuff

#define TIMESTEPS 5000
#define SUBDIFFUSE 2000         //21, do y=100000 41, do y=200000 61, do y=500000 81, do y=750000 121, do y=2000000 101, do y=1600000
#define DIFFUSE 3000           //81, do y=1250000 101, do y=2000000
#define DATAPOINTS ((TIMESTEPS/100)+100)

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
double sum;
double sumsq;

public:
statistics();
int getNumber() const;
double getAverage() const;
double getSqAverage() const;
void add(double x);
};
	
struct Polymer : public Managed{
              
			  int Gridx[POLYLENGTH];
			  int Gridy[POLYLENGTH];
			  int Gridz[POLYLENGTH];
			  float endtoends[DATAPOINTS];
			  float beadtomids[DATAPOINTS];
			  float radofgys[DATAPOINTS];
			  int tracker;
			  int check;
              int currentnode;
              int upnode;
              int downnode;
              int xsame;
              int ysame;
              int zsame;
              int randomdir;
              double endtoend;
              double beadtomid;
              double radofgy;
              int block;
              int resloop;
              int nloopx;
              int nloopy;
              int nloopz;
              double smidx;
              double smidy;
              double smidz;
			  void intial();
			  void polylength();
			  void com();
			  void comin();
			  void gyration();
			  void random();
             };
	
__global__ void cudarandomwalk(Polymer *polymers, curandState *randStates, unsigned long seed) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < NoPOLY)	
	{
		Polymer& polymer = polymers[idx];
		curandState& randState = randStates[idx];
		curand_init(seed, idx, 0, &randState);

		int datapointindex = 0;
		
		for (int y=0; y<=TIMESTEPS; y++)
		{

			for (int z=0; z<(POLYLENGTH); z++)
			{
				polymer.xsame = 0; //random
				polymer.ysame = 0;
				polymer.zsame = 0;
				polymer.block = 0;

				polymer.currentnode = (int)(curand_uniform(&randState)*POLYLENGTH);
				polymer.randomdir = (int)(curand_uniform(&randState)*6);
				//if (idx==0 && z==0) printf("checking thread on device = %d cn, %d  \n", idx, polymer.currentnode);

				polymer.upnode=(polymer.currentnode+1);
				polymer.downnode=(polymer.currentnode-1);

				if ((0 < polymer.currentnode) && (polymer.currentnode < (POLYLENGTH-1)))
				{
					if (polymer.Gridx[polymer.downnode] == polymer.Gridx[polymer.upnode])   polymer.xsame = 1;
					if (polymer.Gridy[polymer.downnode] == polymer.Gridy[polymer.upnode])   polymer.ysame = 1;
					if (polymer.Gridz[polymer.downnode] == polymer.Gridz[polymer.upnode])   polymer.zsame = 1;
				}

				if (polymer.currentnode == 0)
				{
					polymer.Gridx[polymer.currentnode] = polymer.Gridx[polymer.upnode];
					polymer.Gridy[polymer.currentnode] = polymer.Gridy[polymer.upnode];
					polymer.Gridz[polymer.currentnode] = polymer.Gridz[polymer.upnode];
					if (polymer.randomdir == 0) polymer.Gridx[polymer.currentnode]--;          
					else if (polymer.randomdir == 1) polymer.Gridx[polymer.currentnode]++;        
					else if (polymer.randomdir == 2) polymer.Gridy[polymer.currentnode]--;
					else if (polymer.randomdir == 3) polymer.Gridy[polymer.currentnode]++; 
					else if (polymer.randomdir == 4) polymer.Gridz[polymer.currentnode]--;   
					else polymer.Gridz[polymer.currentnode]++;    
				}

				if (polymer.currentnode == (POLYLENGTH-1))
				{
					polymer.Gridx[polymer.currentnode] = polymer.Gridx[polymer.downnode];
					polymer.Gridy[polymer.currentnode] = polymer.Gridy[polymer.downnode];
					polymer.Gridz[polymer.currentnode] = polymer.Gridz[polymer.downnode];
					if (polymer.randomdir == 0) polymer.Gridx[polymer.currentnode]--;          
					else if (polymer.randomdir == 1) polymer.Gridx[polymer.currentnode]++;        
					else if (polymer.randomdir == 2) polymer.Gridy[polymer.currentnode]--;
					else if (polymer.randomdir == 3) polymer.Gridy[polymer.currentnode]++; 
					else if (polymer.randomdir == 4) polymer.Gridz[polymer.currentnode]--;   
					else polymer.Gridz[polymer.currentnode]++;    
				}

				if ((0 < polymer.currentnode) && (polymer.currentnode < (POLYLENGTH-1)))
				{
					if ((polymer.xsame == 1) && (polymer.ysame == 1) && (polymer.zsame == 1))
					{
					polymer.Gridx[polymer.currentnode] = polymer.Gridx[polymer.upnode];
					polymer.Gridy[polymer.currentnode] = polymer.Gridy[polymer.upnode];
					polymer.Gridz[polymer.currentnode] = polymer.Gridz[polymer.upnode];
					if (polymer.randomdir == 0) polymer.Gridx[polymer.currentnode]--;          
					else if (polymer.randomdir == 1) polymer.Gridx[polymer.currentnode]++;        
					else if (polymer.randomdir == 2) polymer.Gridy[polymer.currentnode]--;
					else if (polymer.randomdir == 3) polymer.Gridy[polymer.currentnode]++; 
					else if (polymer.randomdir == 4) polymer.Gridz[polymer.currentnode]--;   
					else polymer.Gridz[polymer.currentnode]++;    
					}
				}
			}
			if (y == SUBDIFFUSE)
			{
	//			polymer.comin();
				double smidx = 0.0;
				double smidy = 0.0;
				double smidz = 0.0;
				for (int i; i < POLYLENGTH; i++)
				{
					 smidx = smidx + (double)polymer.Gridx[i];
					 smidy = smidy + (double)polymer.Gridy[i];
					 smidz = smidz + (double)polymer.Gridz[i];
				}
				polymer.smidx = smidx;
				polymer.smidy = smidy;
				polymer.smidz = smidz;
			}
			if ((y % 100) == 0 )
			{
//				polymer.polylength();
				double dx = polymer.Gridx[POLYLENGTH - 1] - polymer.Gridx[0];
				double dy = polymer.Gridy[POLYLENGTH - 1] - polymer.Gridy[0];
				double dz = polymer.Gridz[POLYLENGTH - 1] - polymer.Gridz[0];
			    polymer.endtoend = sqrt(dx * dx + dy * dy + dz * dz);

//				polymer.com();
				double xSum = 0.0;
				double ySum = 0.0;
				double zSum = 0.0;
				for (int i; i < POLYLENGTH; i++)
				{
					 xSum = xSum + (double)polymer.Gridx[i];
					 ySum = ySum + (double)polymer.Gridy[i];
					 zSum = zSum + (double)polymer.Gridz[i];
				}
				 xSum = (xSum - polymer.smidx)/double(POLYLENGTH);
				 ySum = (ySum - polymer.smidy)/double(POLYLENGTH);
				 zSum = (zSum - polymer.smidz)/double(POLYLENGTH);

				 polymer.beadtomid = ((xSum*xSum)+(ySum*ySum)+(zSum*zSum));

				 polymer.endtoends[datapointindex] = (float)polymer.endtoend;
				 polymer.beadtomids[datapointindex] = (float)polymer.beadtomid;
				 datapointindex++;
			}
		}
    }
}

void Polymer::intial()
{
       int forcedir = 3;
       int iran = 0;
       int i;
       int blocked = 0;
       int resloup = 0;
       int nloupx = 0;
       int nloupy = 0;
       int nloupz = 0;

       for(i=1; i < POLYLENGTH; i++)
       {
                  iran = rand()%forcedir;
                  blocked = 0;

                  if(iran==2)
                  {
                           for (resloup = 1; resloup <= RESOLUTION; resloup++)
                           {
                                    for (nloupx = 0; nloupx < NANOSIZE; nloupx++)
                                    {
                                               for (nloupy = 0; nloupy < NANOSIZE; nloupy++)
                                               {
                                                        for (nloupz = 0; nloupz < NANOSIZE; nloupz++)
                                                        {
                                                                if (((((Gridx[POLYLENGTH-1]+resloup)%DENSITY) == nloupx)) && ((((Gridy[POLYLENGTH-1])%DENSITY) == nloupy)) && ((((Gridz[POLYLENGTH-1])%DENSITY) == nloupz))) blocked = 1;
                                                        }
                                               }
                                    }
                           }
                           if (blocked == 0) Gridx[POLYLENGTH-1]++;
                  }

                  if(iran==1)
                  {
                           for (resloup = 1; resloup <= RESOLUTION; resloup++)
                           {
                                    for (nloupx = 0; nloupx < NANOSIZE; nloupx++)
                                    {
                                               for (nloupy = 0; nloupy < NANOSIZE; nloupy++)
                                               {
                                                        for (nloupz = 0; nloupz < NANOSIZE; nloupz++)
                                                        {
                                                                if (((((Gridx[POLYLENGTH-1])%DENSITY) == nloupx)) && ((((Gridy[POLYLENGTH-1]+resloup)%DENSITY) == nloupy)) && ((((Gridz[POLYLENGTH-1])%DENSITY) == nloupz))) blocked = 1;
                                                        }
                                               }
                                    }
                           }
                           if (blocked == 0) Gridy[POLYLENGTH-1]++;
                  }

                  if(iran==0)
                  {
                           for (resloup = 1; resloup <= RESOLUTION; resloup++)
                           {
                                    for (nloupx = 0; nloupx < NANOSIZE; nloupx++)
                                    {
                                               for (nloupy = 0; nloupy < NANOSIZE; nloupy++)
                                               {
                                                        for (nloupz = 0; nloupz < NANOSIZE; nloupz++)
                                                        {
                                                                if (((((Gridx[POLYLENGTH-1])%DENSITY) == nloupx)) && ((((Gridy[POLYLENGTH-1])%DENSITY) == nloupy)) && ((((Gridz[POLYLENGTH-1]+resloup)%DENSITY) == nloupz))) blocked = 1;
                                                        }
                                               }
                                    }
                           }
                           if (blocked == 0) Gridz[POLYLENGTH-1]++;
                  }

                  if (blocked == 0) 
					  {
						  Gridx[i]=Gridx[POLYLENGTH-1];
						  Gridy[i]=Gridy[POLYLENGTH-1];
						  Gridz[i]=Gridz[POLYLENGTH-1];
				      }
                  if (blocked == 1) i--;

       }
}

void Polymer::polylength()
{
     double a;
     double b;
     double c;
     a = (Gridx[POLYLENGTH-1] - Gridx[0])*(Gridx[POLYLENGTH-1] - Gridx[0]);
     b = (Gridy[POLYLENGTH-1] - Gridy[0])*(Gridy[POLYLENGTH-1] - Gridy[0]);
     c = (Gridz[POLYLENGTH-1] - Gridz[0])*(Gridz[POLYLENGTH-1] - Gridz[0]);
     endtoend = sqrt(a+b+c);
}

void Polymer::comin()

{
     double x,y,z;
     int i=0;
     x = 0;
     y = 0;
     z = 0;

     for (i;i<POLYLENGTH;i++)
     {
         x =  x +(double(Gridx[i]));
         y =  y +(double(Gridy[i]));
         z =  z +(double(Gridz[i]));
     }

     smidx = x;
     smidy = y;
     smidz = z;
}


void Polymer::com()

{
     double x,y,z;
     int i=0;
     x = 0;
     y = 0;
     z = 0;


     for (i;i<POLYLENGTH;i++)
     {
         x =  x +(double(Gridx[i]));
         y =  y +(double(Gridy[i]));
         z =  z +(double(Gridz[i]));
     }

     x = (x - smidx)/double(POLYLENGTH);
     y = (y - smidy)/double(POLYLENGTH);
     z = (z - smidz)/double(POLYLENGTH);

     beadtomid = ((x*x)+(y*y)+(z*z));
}


void Polymer::gyration()

{
     double x,y,z,radx,rady,radz,vrad;
     int i=0;
     int j=0;
     x = 0;
     y = 0;
     z = 0;
     radx = 0;
     rady = 0;
     radz = 0;
     vrad = 0;

     for (i;i<POLYLENGTH;i++)
     {
         x =  x +(double(Gridx[i]));
         y =  y +(double(Gridy[i]));
         z =  z +(double(Gridz[i]));
     }

     x = (x)/(double(POLYLENGTH));
     y = (y)/(double(POLYLENGTH));
     z = (z)/(double(POLYLENGTH));

     for (j;j<POLYLENGTH;j++)
     {
     radx = (((double(Gridx[j])) - x)*((double(Gridx[j])) - x));
     rady = (((double(Gridy[j])) - y)*((double(Gridy[j])) - y));
     radz = (((double(Gridz[j])) - z)*((double(Gridz[j])) - z));
     vrad = vrad + radx + rady + radz;
     }

     radofgy = sqrt(vrad/(double(POLYLENGTH)));
}

statistics::statistics() {
n=0;
sum=0.0;
sumsq=0.0;
}

int statistics::getNumber() const{
return n;
}

double statistics::getAverage() const {
if(n==0) return 0.;
return sum/n;
}

double statistics::getSqAverage() const {
if(n==0) return -1;
return sumsq/n;
}

void statistics::add(double x) {
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
	//int k;
	int q;
	int g;

	srand(time(NULL));

	ofstream outfile;
    outfile.open ("TEST8.txt");//**************************************************************************************************************
    if (!outfile.is_open())
    { 
	    cout << "file not open" << endl;
		return 666;
	}

    outfile << "TimeStep " << "E2EDistance " << "R^2 " << "log10(TimeStep) " << "log10(R^2) " << endl;

	cout << "_____________" << endl << "STARTING STATS ..." << endl;

	//std::map<int, statistics> rsqmap;
	//std::map<int, statistics> flengthmap;

	//for (y=1; y<TIMESTEPS; y++)
	//{
	//	if ((y % 100) == 0 )
	//	{
	//		statistics rsq;
	//		statistics flength;
	//		rsqmap.insert(rsqmap.begin(), std::pair<int,statistics>(y, rsq));
	//	    flengthmap.insert(flengthmap.begin(), std::pair<int,statistics>(y, flength));
	//	}
	//}

		int polycount = NoPOLY;
		Polymer *Allpoly; 
		cudaMallocManaged(&Allpoly, polycount * sizeof(Polymer));
		curandState* randStates;
        cudaMalloc ( &randStates, polycount*sizeof( curandState ) );
	
		    for (i=0; i < polycount; i++) 
			{
				Allpoly[i].currentnode = 0;
				Allpoly[i].randomdir = 0;
				Allpoly[i].xsame = 0;
				Allpoly[i].ysame = 0;
				Allpoly[i].zsame = 0;
				Allpoly[i].endtoend = 0;
				Allpoly[i].radofgy = 0;
				Allpoly[i].beadtomid = 0;
				Allpoly[i].smidx = 0;
				Allpoly[i].smidy = 0;
				Allpoly[i].smidz = 0;
				Allpoly[i].tracker = 0;
				Allpoly[i].check = 1;
				for (q=0; q < POLYLENGTH; q++)
				{
					Allpoly[i].Gridx[q] = NANOSIZE;
					Allpoly[i].Gridy[q] = NANOSIZE;
					Allpoly[i].Gridz[q] = NANOSIZE;
			    }
				//for (q=0; q < POLYLENGTH; q++)
				//{
				//	Allpoly[i].Stats[q] = (float)q;
			 //   }
				Allpoly[i].intial();
			    Allpoly[i].comin();
            }
			Allpoly[0].polylength();
			cout << "_____________" << endl << "e2edistace=" << Allpoly[0].endtoend << endl << "_____________" << endl;
			for (g=0; g<POLYLENGTH; g++)
			{
				cout << "_____________________________" << endl;
				cout << "Bead no " << (g) <<" at (" << Allpoly[0].Gridx[g] << "," << Allpoly[0].Gridy[g] << "," << Allpoly[0].Gridz[g] << ")"  << endl;
			}

//-----------------------------------------------------------KERNAL CALL------------------------------------------------------------------------------------//


			cudarandomwalk<<<(polycount/(256))+1, polycount>>>(Allpoly, randStates, time(NULL));  //(polycount/(256))+1
			cudaDeviceSynchronize();

            int datapointindex = 0;
			for (y=0; y<=TIMESTEPS; y++)
			{
				statistics rsq;
                statistics flength;
				for (x=0; x < polycount; x++) 
				{
					if ((y % 100) == 0 )
					{
                        rsq.add(Allpoly[x].beadtomids[datapointindex]);
                        flength.add(Allpoly[x].endtoends[datapointindex]);
						datapointindex++;
					}
				}
				if ((y % 100) == 0 ) 
                {
                                outfile << y << " " <<  flength.getAverage()  << " "  << rsq.getAverage() << " " << log10((double)(y)) << " " << log10(rsq.getAverage()) << endl;
                }
			}

			cout << "_____________" << endl << "END OF RANDOMWALK" << endl;
			
			Allpoly[0].polylength();
			cout << "_____________" << endl << "e2edistace=" << Allpoly[0].endtoend << endl << "_____________" << endl;
			
			for (g=0; g<POLYLENGTH; g++)
			{
				cout << "_____________________________" << endl;
				cout << "Bead no " << (g) <<" at (" << Allpoly[0].Gridx[g] << "," << Allpoly[0].Gridy[g] << "," << Allpoly[0].Gridz[g] << ")"  << endl;
			}
           
			cudaFree(Allpoly);
			cudaFree(randStates);
	
	cout << "_____________" << endl << "PRINTING STATS ..." << endl;

	//for (y=1; y<TIMESTEPS; y++) //we can improve later
	//{
	//	if ((y % 100) == 0 )
	//	{
 //           statistics rsq = rsqmap[y];  
 //           statistics flength = flengthmap[y];  
	//		outfile << (y) << " " <<  flength.getAverage()  << " "  << rsq.getAverage() << " " << log10((double)(y)) << " " << log10(rsq.getAverage()) << endl;
	//	}
	//}

	cout << "_____________" << endl << "FINISHED STATS ..." << endl;
	
	cudaDeviceReset();
	outfile.close();
	long finishTime = clock();
	cout<<endl<<"Run time is "<<((finishTime - startTime)/double(CLOCKS_PER_SEC))<<" seconds"<<endl<<endl;
	system("PAUSE");
    return 0;
}




