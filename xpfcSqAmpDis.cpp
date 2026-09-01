#include <iostream>
#include <iomanip>
#include <fstream>
#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <time.h>

#include <mpi.h>
#include <complex>
#include <fftw3-mpi.h>

using namespace std;

#define Pi (2.*acos(0.))
#define dxa (0.125)			//grid spacing regular XPFC - for density reconstruction
complex<double> I(0,1);

//domain parameters
char run[10] = {'\0'};			//label of current run
double dx;			//grid spacing
double spacing;		//normalized lattice spacing
int atomsx,atomsy;	//number of atoms in x and y
ptrdiff_t Nx,Ny;	//number of grid points in x and y
ptrdiff_t Nx2,Nxn2;		//needed for padding real arrays for transform

//time parameters
int totalTime,printFreq;	//total iteration time and print frequency
int printFreqAmp;			//frequency for outping all amplitudes
double dt;					//time step
int restartFlag;			//set to 1 for restart
int restartTime;			//restart time

//MPI parameters
int argc;
char **argv;
int myid,numprocs;
ptrdiff_t alloc_local, local_n0, local_0_start;
ptrdiff_t i,j,yj,ind,indr,indc;

/**************** correlation variables ****************/
//k-zero mode
double HSq0, wSq0;			//height of k-zero mode, width of k-zero mode
double PreCSq0;			//prefactor based on the widths of the k-zero mode

//first and second modes
double kSq1,kSq2;			//equilibrium peak positions for first and second mode
double wSq1,wSq2;			//widths of the first and second mode peaks (sets elasticity, anisotropy, interface widths etc)
double sigMSq1,sigMSq2;	//reference melting temperatures

double DWSq1,DWSq2;			//debye-waller factor to modulate peaks as a function of temperature (sigma)
double PreCSq1,PreCSq2;		//prefactors calculated with the widths of the correlation peaks

/************* initial condition and simulation variables *************/
double iSig;		//initial temperature - related to vibrational amplitude of atom
double omega;		//superaturation

double navg;		//average density
double ns,nl;			//equilibrium solid and liquid density for temperature iSig

double eta,chi;		//coefficients for cubic and quartic term in ideal energy
double Mno,Ma,Mb;	//mobility constant for density, mobility for amplitudes A and B

double facx,facy,normFac;	//fourier factors for scaling lengths in k-space
/*********************** general Arrays ****************************/
double *dens;			//hold the values for the density reconstruction
double *magA;			//holds the magnitue of A amplitudes respectively
double *magB;			//holds the magnitue of B amplitudes respectively
double *sumAmp;

/*********************** average density ****************************/
double *no;		//average density
double *nofNL;	//holds collection of non-linear terms 
double *cubno;	//collection cubic terms (in terms of amplitudes)

complex<double> *kno;		//complex density after transform
complex<double> *knofNL;	//complex non-linear density terms after transform

/*********************** reciprocal lattice vector quantities ****************************/
double k1x[2], k1y[2];	//indicies of the reciprocal lattice vectors --first mode
double magk1[2];				//magnitude of reciprocal lattice vectors --first mode

double k2x[2], k2y[2];	//indicies of the reciprocal lattice vectors --second mode
double magk2[2];				//magnitude of reciprocal lattice vectors --second mode

/*********************** fftw stuff for Amplitude A ****************************/
complex<double> *A[2];			//amplitudes first mode
complex<double> *AfNL[2];		//collection of non-linear terms
complex<double> *cubA[2];		//cubic terms for non-linear
complex<double> *quartA[2];	//quartic terms for non-linear terms

/*********************** fftw stuff for Amplitude B ****************************/
complex<double> *B[2];			//amplitudes first mode
complex<double> *BfNL[2];		//collection of non-linear terms
complex<double> *cubB[2];		//cubic terms for non-linear
complex<double> *quartB[2];	//quartic terms for non-linear terms


//fftw plans
//plans for real fields (out-of-place)
fftw_plan rplanF, rplanB;			//forward and backward plans

//plans for complex fields (in-place)
fftw_plan cplanF, cplanB;			//forward and backward plans

/*********************** correlation arrays ****************************/
double *CSqno;				//correlation for density

double *CSqA[2]; 	//shifted correlation amplitude A
double *CSqB[2];	//shifted correlation amplitude B

//k-sapce operators
double *karr;			//holds k^2 array in k-space real fields
double *karrc;			//holds k^2 array in k-space complex fields
double *karrx,*karry;	//holds gradient arroays in x and y direction

double sigma = 1.;
int temp;

void freeMemory()
{
	int ii;
	/*********************** destroy fftw plans **************************/
	fftw_destroy_plan(rplanF);
	fftw_destroy_plan(rplanB);

	fftw_destroy_plan(cplanF);
	fftw_destroy_plan(cplanB);

	/*********************** free memory **************************/
	fftw_free(dens);	fftw_free(magA);		fftw_free(magB);
	fftw_free(sumAmp);
	fftw_free(no);	fftw_free(nofNL);	fftw_free(cubno);
		
	delete[] kno ;	
	delete[] knofNL ;
	fftw_free(CSqno);

	//amplitude A
	for(ii=0;ii<2;ii++)
	{
		delete[] cubA[ii] ;
		delete[] quartA[ii] ;
		delete[] A[ii] ;
		delete[] AfNL[ii] ;
		fftw_free(CSqA[ii]);
	}
	
	//amplitude B
	for(ii=0;ii<2;ii++)
	{
		delete[] cubB[ii] ;
		delete[] quartB[ii] ;
		delete[] B[ii] ;
		delete[] BfNL[ii] ;
		fftw_free(CSqB[ii]);	
	}

	fftw_free(karr); 	fftw_free(karrc);
	fftw_free(karrx); 	fftw_free(karry);
}
ptrdiff_t index2Dr(ptrdiff_t a, ptrdiff_t b)
{
	return a+b*Nx2 ;
}
ptrdiff_t index2Dkr(ptrdiff_t a, ptrdiff_t b)
{
	return a+b*Nxn2 ;
}
ptrdiff_t index2Dc(ptrdiff_t a, ptrdiff_t b)
{
	return a+b*Nx ;
}
void  restart(int time,double *Array,const char *filepre,char *runtype)
{
	double *tmp;
	tmp = (double*) fftw_malloc(sizeof(double) * Ny*Nx );

	char filename[BUFSIZ] = {'\0'};
    FILE *fp;
    sprintf(filename,"%s%s%d.dat",runtype,filepre,time);
    fp = fopen(filename,"rt");
	
	ptrdiff_t i,j;
	ptrdiff_t jg;	//global index for y direction due to mpi
	
	if( fp == NULL )
	{
		printf("Unable to open file for reading\n");
		printf("Exiting simulation!\n");
		exit(1);
	}
	
	for(j=0;j<Ny;j++)
	{
		for(i=0;i<Nx;i++)
		{
			fscanf(fp,"%lf",&tmp[i+Nx*j]);
		}
	}
    fclose(fp);

	MPI_Barrier(MPI_COMM_WORLD);

	for(j=0;j<Ny;j++)
	{
		for(i=0;i<Nx;i++)
		{
			jg = j - myid*local_n0;	//rescale coordinates to that of local processor
			if (jg >= 0 && jg <= local_n0-1)
				Array[i+Nx2*jg] = tmp[i+Nx*j];
		}
	}
	
	fftw_free(tmp);
	MPI_Barrier(MPI_COMM_WORLD);
}
void output(int time,double *Array,const char *filepre,char *runtype)
{
	char filename[BUFSIZ] = {'\0'};
    FILE *fp;
    sprintf(filename,"%s%s%d.dat",runtype,filepre,time);

	ptrdiff_t i,j;
	int kk;
	
	for (kk=0;kk<numprocs;kk++)
	{
		if (myid == kk)
		{
			if (myid==0)
			{
				fp = fopen(filename,"w");
				if(fp==NULL)
				{
					printf("Unable to open file for writing\n");
					printf("Exiting simulation!\n");
					exit(1);
				}
			}
			else
			{
				fp = fopen(filename,"a");
				if(fp==NULL)
				{
					printf("Unable to reopen file for appending\n");
					printf("Exiting simulation!\n");
					exit(1);
				}
			}

			for(j=0;j<local_n0;j++)
			{
				for(i=0;i<Nx;i++)
				{
					fprintf(fp,"%lf\n",Array[i+Nx2*j]);
				}
			}
			fclose(fp);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}
void  restart(int time,complex<double> *Array,const char *filepre,int ampNum,char *runtype)
{
	double tmpR[Ny*Nx],tmpI[Ny*Nx];
	
	char filename[BUFSIZ] = {'\0'};
    FILE *fp;
    sprintf(filename,"%s%s%d_%d.dat",runtype,filepre,ampNum,time);
    fp = fopen(filename,"r");

    ptrdiff_t i,j;
	ptrdiff_t jg;	//global index for y direction due to mpi
	
	if( fp == NULL )
	{
		printf("Unable to open file for reading\n");
		printf("Exiting simulation!\n");
		exit(1);
	}
	
	for(j=0;j<Ny;j++)
	{
		for(i=0;i<Nx;i++)
		{
			fscanf(fp,"%lf %lf",&tmpR[i+Nx*j],&tmpI[i+Nx*j] );
		}
	}
    fclose(fp);

	MPI_Barrier(MPI_COMM_WORLD);

	for(j=0;j<Ny;j++)
	{
		for(i=0;i<Nx;i++)
		{
			jg = j - myid*local_n0;	//rescale coordinates to that of local processor
			if (jg >= 0 && jg <= local_n0-1)
				Array[i+Nx*jg] = tmpR[i+Nx*j] + I*tmpI[i+Nx*j];
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
}
void output(int time,complex<double> *Array,const char *filepre,int ampNum,char *runtype)
{
	char filename[BUFSIZ] = {'\0'};
    FILE *fp;
    sprintf(filename,"%s%s%d_%d.dat",runtype,filepre,ampNum,time);

	ptrdiff_t i,j;
	int kk;
	
	for (kk=0;kk<numprocs;kk++)
	{
		if (myid == kk)
		{
			if (myid==0)
			{
				fp = fopen(filename,"w");
				if(fp==NULL)
				{
					printf("Unable to open file for writing\n");
					printf("Exiting simulation!\n");
					exit(1);
				}
			}
			else
			{
				fp = fopen(filename,"a");
				if(fp==NULL)
				{
					printf("Unable to reopen file for appending\n");
					printf("Exiting simulation!\n");
					exit(1);
				}
			}

			for(j=0;j<local_n0;j++)
			{
				for(i=0;i<Nx;i++)
				{
					fprintf(fp,"%lf  %lf\n",real( Array[i+Nx*j] ),imag( Array[i+Nx*j] ) );
				}
			}
			fclose(fp);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}
void densReconAmp(double *Array,complex<double> *ArrayA[2],complex<double> *ArrayB[2])
{
	ptrdiff_t i,j;
	ptrdiff_t index,index2;
	ptrdiff_t yj;

	int ii;
	double Asum=0.,Bsum=0.;

	for(j=0;j<local_n0;j++)
	{
		yj = j + local_0_start;

		for(i=0;i<Nx;i++)
		{
			index = i + Nx*j;
			index2 = i + Nx2*j;
						
			Asum = 0.;
			for(ii=0;ii<2;ii++)
				Asum += 2.*( real( ArrayA[ii][index] )*cos( k1x[ii]*i*dx + k1y[ii]*yj*dx ) 
						-    imag( ArrayA[ii][index] )*sin( k1x[ii]*i*dx + k1y[ii]*yj*dx ) );

			Bsum = 0.;
			for(ii=0;ii<2;ii++)
				Bsum += 2.*( real( ArrayB[ii][index] )*cos( k2x[ii]*i*dx + k2y[ii]*yj*dx ) 
						-    imag( ArrayB[ii][index] )*sin( k2x[ii]*i*dx + k2y[ii]*yj*dx ) );

			dens[index2] = 0.0;
			dens[index2] = Array[index2] + Asum + Bsum;
			
			magA[index2] = abs( ArrayA[0][index]*conj( ArrayA[0][index] ) )
						 + abs( ArrayA[1][index]*conj( ArrayA[1][index] ) );

			magB[index2] = abs( ArrayB[0][index]*conj( ArrayB[0][index] ) )
						 + abs( ArrayB[1][index]*conj( ArrayB[1][index] ) );			
		}
	}
}
void normalize(double *Array)
{
	for(j=0;j<local_n0;j++)
	{
		for(i=0;i<Nx;i++)
		{
			Array[index2Dr(i,j)] *= normFac;
		}
	}
}
void normalize(complex<double> *Array)
{
	for(j=0;j<local_n0;j++)
	{
		for(i=0;i<Nx;i++)
		{
			Array[index2Dc(i,j)] *= normFac;
		}
	}
}
double SqCorr(double mk)
{
	double kC0,kC1,kC2;
	double kCSq;

	//calculate correlation overlap and send up peak

	kC0 = HSq0*exp( PreCSq0*mk*mk );
	kC1 = DWSq1*exp( PreCSq1* (mk-kSq1)*(mk-kSq1) );
	kC2 = DWSq2*exp( PreCSq2* (mk-kSq2)*(mk-kSq2) );				

	//creating the correlation overlap
	if (kC0 > kC1)
		kCSq = -kC0;
	else if (kC1 > kC2)
		kCSq = kC1;
	else
		kCSq = kC2;

	return kCSq;
}

/******************** functions for amplitude B ********************/ 
void StepB()
{
	double prefactor;
	int ii;

	/************** Step amplitude B *************/	
	for(j=0;j<local_n0;j++)
	{
		for(i=0;i<Nx;i++)
		{
			ind = index2Dc(i,j);

			for(ii=0;ii<2;ii++)
			{
				prefactor = 1.0/( 1.0 + dt*Mb*( 1. - CSqB[ii][ind] ) );
			
				B[ii][ind] = prefactor*( B[ii][ind] - dt*Mb*BfNL[ii][ind] );
			}
		}
	}
}
void calcCorrB()
{
	int ii;
	
	double kx,ky,k2;
	double rk;
	
	for(j=0;j<local_n0;j++)
	{
		for(i=0;i<Nx;i++)
		{
			ind = index2Dc(i,j);

			kx = karrx[ind];
			ky = karry[ind];

			k2 = karrc[ind];

			//****************************amplitude B**************************//
			for(ii=0;ii<2;ii++)
			{
				rk = sqrt( k2 + 2.*( kx*k2x[ii] + ky*k2y[ii] ) + magk2[ii] );

				CSqB[ii][ind] = SqCorr(rk);
			}

		}
	}
}
void fcalNLB()
{
	int ii;
	double n;

	for(j=0;j<local_n0;j++)
	{
		for(i=0;i<Nx;i++)
		{
			indr = index2Dr(i,j);
			indc = index2Dc(i,j);

			n = no[indr];

			for(ii=0;ii<2;ii++)
			{
				BfNL[ii][indc] = n*( -eta + chi*n )*B[ii][indc] + ( -eta + 2.*chi*n )*cubB[ii][indc]
								+ chi*B[ii][indc]*( 2.*sumAmp[indr] - abs(B[ii][indc]*conj(B[ii][indc])) ) 
								+ chi*quartB[ii][indc] ;
			}
		}
	}
}
void supplArraysB()
{
	for(j=0;j<local_n0;j++)
	{
		for(i=0;i<Nx;i++)
		{
			indc = index2Dc(i,j);

			//from the product of the cubic terms
			cubB[0][indc] = A[0][indc]*A[1][indc] ;

			cubB[1][indc] = A[0][indc]*conj( A[1][indc] ) ;

			//fourth order terms from the expansion
			quartB[0][indc] = A[0][indc]*A[0][indc]*conj( B[1][indc] )
							 + A[1][indc]*A[1][indc]*B[1][indc];
				
			quartB[1][indc] = A[0][indc]*A[0][indc]*conj( B[0][indc] )
							 + conj( A[1][indc] )*conj( A[1][indc] )*B[0][indc];
		}
	}
}
void iterateB()
{
	int ii;
	/****************** amplitude B ******************/
	supplArraysB();		//calculate supplementary arrays need for updating amplitude B
	fcalNLB();			//calculate non-linear terms for amplitude B

	for(ii=0;ii<2;ii++)
	{
		fftw_mpi_execute_dft(cplanF,reinterpret_cast<fftw_complex*>(B[ii]),reinterpret_cast<fftw_complex*>(B[ii]));		//fourier transform (in place) complex amplitude B
		fftw_mpi_execute_dft(cplanF,reinterpret_cast<fftw_complex*>(BfNL[ii]),reinterpret_cast<fftw_complex*>(BfNL[ii]));	//foruier transform non-linear terms (in place)
	}
	
	calcCorrB();
	StepB();			//update amplitude B in k-space
	for(ii=0;ii<2;ii++)
	{
		fftw_mpi_execute_dft(cplanB,reinterpret_cast<fftw_complex*>(B[ii]),reinterpret_cast<fftw_complex*>(B[ii]));		//inverse transform (in place)
		normalize(B[ii]);	//normailze amplitude after inverse transform
	}
}

/******************** functions for amplitude A ********************/ 
void StepA()
{
	double prefactor;
	int ii;

	/************** Step amplitude A *************/	
	for(j=0;j<local_n0;j++)
	{
		for(i=0;i<Nx;i++)
		{
			ind = index2Dc(i,j);

			for(ii=0;ii<2;ii++)
			{
				prefactor = 1.0/( 1.0 + dt*Ma*( 1. - CSqA[ii][ind] ) );
			
				A[ii][ind] = prefactor*( A[ii][ind] - dt*Ma*AfNL[ii][ind] );
			}
		}
	}
}
void calcCorrA()
{
	int ii;
	
	double kx,ky,k2;
	double rk;
	
	for(j=0;j<local_n0;j++)
	{
		for(i=0;i<Nx;i++)
		{
			ind = index2Dc(i,j);

			kx = karrx[ind];
			ky = karry[ind];

			k2 = karrc[ind];

			//****************************amplitude A**************************//
			for(ii=0;ii<2;ii++)
			{
				rk = sqrt( k2 + 2.*( kx*k1x[ii] + ky*k1y[ii] ) + magk1[ii] );

				CSqA[ii][ind] = SqCorr(rk);
			}

		}
	}
}
void fcalNLA()
{
	int ii;
	double n;

	for(j=0;j<local_n0;j++)
	{
		for(i=0;i<Nx;i++)
		{
			indr = index2Dr(i,j);
			indc = index2Dc(i,j);

			n = no[indr];

			for(ii=0;ii<2;ii++)
			{
				AfNL[ii][indc] = n*( -eta + chi*n )*A[ii][indc] + ( -eta + 2.*chi*n )*cubA[ii][indc]
								+ chi*A[ii][indc]*( 2.*sumAmp[indr] - norm(A[ii][indc]) ) + chi*quartA[ii][indc] ;
			}
		}
	}
}
void supplArraysA()
{
	for(j=0;j<local_n0;j++)
	{
		for(i=0;i<Nx;i++)
		{
			indc = index2Dc(i,j);

			//from the product of the cubic terms
			cubA[0][indc] = conj( A[1][indc] )*B[0][indc] 
						   + A[1][indc]*B[1][indc];

			cubA[1][indc] = conj( A[0][indc] )*B[0][indc]
						   + A[0][indc]*conj( B[1][indc] ) ;

			//fourth order terms from the expansion
			quartA[0][indc] = 2.*conj( A[0][indc] )*B[0][indc]*B[1][indc];
			
			quartA[1][indc] = 2.*conj( A[1][indc] )*B[0][indc]*conj( B[1][indc] );
		}
	}
}
void iterateA()
{
	int ii;
	/****************** amplitude A ******************/
	supplArraysA();		//calculate supplementary arrays need for updating amplitude A
	fcalNLA();			//calculate non-linear terms for amplitude A

	for(ii=0;ii<2;ii++)
	{
		fftw_mpi_execute_dft(cplanF,reinterpret_cast<fftw_complex*>(A[ii]),reinterpret_cast<fftw_complex*>(A[ii]));		//fourier transform (in place) complex amplitude A
		fftw_mpi_execute_dft(cplanF,reinterpret_cast<fftw_complex*>(AfNL[ii]),reinterpret_cast<fftw_complex*>(AfNL[ii]));	//foruier transform non-linear terms (in place)
	}
	
	calcCorrA();
	StepA();			//update amplitude A in k-space
	for(ii=0;ii<2;ii++)
	{
		fftw_mpi_execute_dft(cplanB,reinterpret_cast<fftw_complex*>(A[ii]),reinterpret_cast<fftw_complex*>(A[ii]));		//inverse transform (in place)
		normalize(A[ii]);	//normailze amplitude after inverse transform
	}
}

/******************** functions for average density ********************/ 
void Stepno()
{
	double prefactor;
	
	double k2;

	/************** Step average density *************/
	for(j=0;j<local_n0;j++)
	{
		for(i=0;i<Nxn2;i++)
		{
			ind = index2Dkr(i,j);
			
			k2 = karr[ind];

			prefactor = 1.0/( 1.0 + dt*Mno*k2*( 1. - CSqno[ind] ) );
			
			kno[ind] = prefactor*( kno[ind] - dt*Mno*k2*knofNL[ind] );
		}
	}
}
void calcCorrno()
{
	int ii;
	
	double rk,k2;
	double VolAvgKern;	//volume-averaging fitler (only for the average density)
	
	for(j=0;j<local_n0;j++)
	{
		for(i=0;i<Nxn2;i++)
		{
			ind = index2Dkr(i,j);

			k2 = karr[ind];

			VolAvgKern = exp(-.5*k2*sigma*sigma);
			rk = sqrt( k2 );

			CSqno[ind] = SqCorr(rk)*VolAvgKern;;

		}
	}
}
void fcalNLno()
{
	double n;

	for(j=0;j<local_n0;j++)
	{
		for(i=0;i<Nx;i++)
		{
			ind = index2Dr(i,j);

			n = no[ind];

			nofNL[ind] = n*n*( -0.5*eta + chi*n/3. ) + ( -eta + 2.*chi*n )*sumAmp[ind] + 2.*chi*cubno[ind] ;
		}
	}
}
void supplArraysno()
{
	for(j=0;j<local_n0;j++)
	{
		for(i=0;i<Nx;i++)
		{
			indr = index2Dr(i,j);
			indc = index2Dc(i,j);

			//sum of the product of the amplitudes cubic terms				
			cubno[indr] = 2.*real( A[0][indc]*A[1][indc]*conj( B[0][indc] )
						  	  + conj( A[0][indc] )*A[1][indc]*B[1][indc] );

		}
	}
}
void iterateno()
{
	/****************** average density ******************/
	supplArraysno();			//calculate supplementary arrays need for updating average density
	fcalNLno();					//calculate non-linear terms for density

	fftw_mpi_execute_dft_r2c(rplanF,no,reinterpret_cast<fftw_complex*>(kno));		//fourier transform average density
	fftw_mpi_execute_dft_r2c(rplanF,nofNL,reinterpret_cast<fftw_complex*>(knofNL));	//foruier transform average density non-linear terms
		
	calcCorrno();
	Stepno();					//update density in k-space
	fftw_mpi_execute_dft_c2r(rplanB,reinterpret_cast<fftw_complex*>(kno),no);		//inverse transform average density
	normalize(no);	//normalize average density after transform
}

void calcSumAmp()
{
	int ii;
	double sumA=0;
	double sumB=0;

	for(j=0;j<local_n0;j++)
	{
		for(i=0;i<Nx;i++)
		{
			indr = index2Dr(i,j);
			indc = index2Dc(i,j);

			sumA=0;
			for(ii=0;ii<2;ii++)
				sumA += norm( A[ii][indc] );

			sumB=0;
			for(ii=0;ii<2;ii++)
				sumB += norm( B[ii][indc] );

			//total sum of magnitudes of all amplitudes
			sumAmp[indr] = sumA + sumB;
		}
	}
}

void setCorrPeakPrefactors(double Sig)
{
	//set up the equil'm position for first and second peaks
	kSq1 = 2.*Pi;	//first mode
	kSq2 = sqrt(2.)*2.*Pi;			//second mode

	PreCSq0 = -0.5/(wSq0*wSq0);
	PreCSq1 = -0.5/(wSq1*wSq1);
	PreCSq2 = -0.5/(wSq2*wSq2);

	DWSq1 = exp( -Sig / sigMSq1 );
	DWSq2 = exp( -Sig / sigMSq2 );
}
void k2Array()
{
	double kx,ky;
	
	for(j=0;j<local_n0;j++)
	{
		yj = j + local_0_start;

		if ( yj < Ny/2 )
			ky =yj*facy;
		else
			ky = (yj-Ny)*facy;

		for(i=0;i<Nxn2;i++)
		{
			kx = i*facx;

			indr = index2Dkr(i,j);
	
			karr[indr] = ( 3. -  cos(kx) - cos(ky) - cos(kx)*cos(ky) )/(dx*dx);
		}

		for(i=0;i<Nx;i++)
		{
			if( i < Nx/2 ) 
				kx = i*facx;
			else 
				kx = (i-Nx)*facx;

			indc = index2Dc(i,j);

			// karrx[indc] = kx/dx;
			// karry[indc] = ky/dx;
			// karrc[indc] = ( kx*kx + ky*ky );

			// karrx[indc] = sin(kx) * ( 1. + cos(ky) ) / (2.*dx);
			// karry[indc] = sin(ky) * ( 1. + cos(kx) ) / (2.*dx);
			// karrc[indc] = ( 4. - ( 1. + cos(kx) )*(1. + cos(ky) ) )/(dx*dx);

			karrx[indc] = sin(kx) * ( 2. + cos(ky) ) / (3.*dx);
			karry[indc] = sin(ky) * ( 2. + cos(kx) ) / (3.*dx);
			karrc[indc] = (2./3.)*( 9. - ( 2. + cos(kx) )*(2. + cos(ky) ) ) / (dx*dx);
		}
	}				
}


/* ============================================================
   Mod: addPhaseNoise() - Add random phase noise to the 
   amplitudes A and B to introduce dislocations
   ============================================================ */
void addPhaseNoise()
{
	if(dislNoiseAmp <= 0.0) return;
	
	double x, y;
	double xx, yy, r_from_center;
	double xmax = ( (double)(Nx-1) ) *dx;
	double ymax = ( (double)(Ny-1) ) *dx;
	
	// Calculate effective solid area fraction from input navg, ns, nl
	double omega_eff = (navg - nl) / (ns - nl);
	if(omega_eff < 0.0) omega_eff = 0.0;
	if(omega_eff > 1.0) omega_eff = 1.0;
	double Lx = (double)Nx * dx;
	double Ly = (double)Ny * dx;
	double R;
	if(omega_eff > 0.01)
		R = sqrt(omega_eff * Lx * Ly / Pi);
	else
		R = 30.0 * dx;
	

	srand48(time(NULL) + myid * 7919);
	
	int ii;
	for(j=0;j<local_n0;j++)
	{
		y = ( (double)(j+local_0_start) )*dx;
		yy = y - .5*ymax;
		for(i=0;i<Nx;i++)
		{
			x = ((double)i)*dx;
			xx = x - .5*xmax;
			indr = index2Dr(i,j);
			indc = index2Dc(i,j);
			
			r_from_center = sqrt(xx*xx + yy*yy);

			double htan = 0.5 * (1. - tanh( (r_from_center - R) / (8.0*dx) ));
			if(htan < 0.5) continue;  
			

			for(ii=0;ii<2;ii++)
			{
				double phase_noise = dislNoiseAmp * (drand48() - 0.5) * 2.0;
				A[ii][indc] *= exp(I * phase_noise);
			}

			for(ii=0;ii<2;ii++)
			{
				double phase_noise = dislNoiseAmp * (drand48() - 0.5) * 2.0;
				B[ii][indc] *= exp(I * phase_noise);
			}
		}
	}
	
	if(myid==0) printf("Phase noise added with amplitude = %lf\n", dislNoiseAmp);
}
/* ============================================================
   Mod 1: initialize() 
   ============================================================ */
void initialize()
{
	double asq1=.3260106318,asq2=.1972365410;
	int ii;
	
	if(myid==0) printf("iSig = %lf \t ns = %lf \t nl = %lf\n",iSig,ns,nl );
	double qrx,qry;
    double htan;
    double R;
    double xs;
    complex<double> arg;
    // Mod: do not override navg, respect input value
    // Back-calculate effective solid area fraction from input navg, ns, nl
    double omega_eff = (navg - nl) / (ns - nl);
    if(omega_eff < 0.0) omega_eff = 0.0;
    if(omega_eff > 1.0) omega_eff = 1.0;
    
    if(myid==0) printf("navg = %lf (input), effective omega = %lf\n", navg, omega_eff);
    // Mod: calculate nucleus radius from effective area fraction
    double Lx = (double)Nx * dx;
    double Ly = (double)Ny * dx;
    if(omega_eff > 0.01)
        R = sqrt(omega_eff * Lx * Ly / Pi);
    else
        R = 30.0 * dx;  // keep small nucleus when liquid-dominated
    
    if(myid==0) printf("nucleus radius R = %lf\n", R);
    xs = 2.*R*Nx/(Nx*Ny);
    double x,y,xx,yy;
    double xmax = ( (double)(Nx-1) ) *dx;
    double ymax = ( (double)(Ny-1) ) *dx;
    double u;
	//***************begin initial conditions******************	
	//sphere/slab in middle of domain
	for(j=0;j<local_n0;j++)
	{
		y = ( (double)(j+local_0_start) )*dx;
		yy = y - .5*ymax;
		for(i=0;i<Nx;i++)
		{
			x = ((double)i)*dx;
			xx = x - .5*xmax;
			indr = index2Dr(i,j);
			indc = index2Dc(i,j);
			//************************** initially mean density and zero amplitudes (liquid) **************************
			no[indr] = nl;
			//first mode amplitude
			for(ii=0;ii<2;ii++)
				A[ii][indc] = complex<double>(0,0);
			//second mode amplitude
			for(ii=0;ii<2;ii++)
				B[ii][indc] = complex<double>(0,0);
			
			// Mod: Smooth interface using tanh function with increased width (8.0*dx)
			htan = 0.5 * (1. - tanh( (sqrt(xx*xx + yy*yy) - R) / (8.0*dx) ));
			
			// Mod: remove shear deformation u, set to zero
			u = 0.0;
			
			for(ii=0;ii<2;ii++)
			{
				//amplitude A
				arg = -I*k1x[ii]*u ;
				A[ii][indc] = htan*asq1*exp(arg);
				//amplitude B
				arg = -I*k2x[ii]*u ;
				B[ii][indc] = htan*asq2*exp(arg);
			}
			//average density
			no[indr] = ns*htan + (1.-htan)*nl;
		}
	}
	
	// Mod: Add random phase noise to amplitudes A and B to introduce dislocations
	addPhaseNoise();
}
void initialConditions()
{
	int ii;

	//initializa random number generator
	srand48(time(NULL)+myid);

	//initial conditions
	if (restartFlag == 0)
	{
		initialize();

		//output initial conditions
		//density
		if(myid==0) printf("output 0\n");
		output(0,no,"n",run);

		//density reconstruction
		densReconAmp(no,A,B);
		output(0,dens,"rcdens",run);
		output(0,magA,"A",run);
		output(0,magB,"B",run);

		for(ii=0;ii<2;ii++)
			output(0,A[ii],"ampA",ii,run);

		for(ii=0;ii<2;ii++)
			output(0,B[ii],"ampB",ii,run);
		
		if(myid==0) printf("initializing domain complete\n");
	}
	else
	{
		if(myid==0) printf("restart flag has been triggered\n");
		if(myid==0) printf("restarting simulation from time step = %d\n",restartTime);

		//read in average density file
		 restart(restartTime,no,"n",run);

		//read in complex amplitude files
		for(ii=0;ii<2;ii++)
			 restart(restartTime,A[ii],"ampA",ii,run);

		for(ii=0;ii<2;ii++)
			 restart(restartTime,B[ii],"ampB",ii,run);
		
		if(myid==0) printf("domain has been reinitialized with restart file\n");
	}
}
void basisVectors()
{
	/************* reciprocal vectors and magnitudes - first mode ******************/
	k1x[0] = 0.;		k1y[0] = 2.*Pi;
	k1x[1] = 2.*Pi;		k1y[1] = 0.;	

	magk1[0] = k1x[0]*k1x[0] + k1y[0]*k1y[0];
	magk1[1] = k1x[1]*k1x[1] + k1y[1]*k1y[1];

	/************* reciprocal vectors and magnitudes - second mode ******************/
	k2x[0] = 2.*Pi;		k2y[0] = 2.*Pi;
	k2x[1] = -2.*Pi;	k2y[1] = 2.*Pi;

	magk2[0] = k2x[0]*k2x[0] + k2y[0]*k2y[0];
	magk2[1] = k2x[1]*k2x[1] + k2y[1]*k2y[1];
}
void setfftwPlans()
{
	//setting up fftw transforms

	rplanF = fftw_mpi_plan_dft_r2c_2d(Ny, Nx, no, reinterpret_cast<fftw_complex*>(kno), MPI_COMM_WORLD, FFTW_PATIENT);
	rplanB = fftw_mpi_plan_dft_c2r_2d(Ny, Nx, reinterpret_cast<fftw_complex*>(kno), no, MPI_COMM_WORLD, FFTW_PATIENT);

	cplanF = fftw_mpi_plan_dft_2d(Ny, Nx, reinterpret_cast<fftw_complex*>(A[0]), reinterpret_cast<fftw_complex*>(A[0]), MPI_COMM_WORLD, FFTW_FORWARD, FFTW_PATIENT);
	cplanB = fftw_mpi_plan_dft_2d(Ny, Nx, reinterpret_cast<fftw_complex*>(A[0]), reinterpret_cast<fftw_complex*>(A[0]), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_PATIENT);
}
void allocateArrays()
{
	int ii;

	fftw_mpi_init();
	
	//get the local (for current cpu) array sizes
	alloc_local = fftw_mpi_local_size_2d(Ny, Nx/2+1, MPI_COMM_WORLD, &local_n0, &local_0_start);

	//general arrays
	dens = fftw_alloc_real( 2*alloc_local );
	magA = fftw_alloc_real( 2*alloc_local );
	magB = fftw_alloc_real( 2*alloc_local );
	sumAmp = fftw_alloc_real( 2*alloc_local );

	/***************** no ************************/	
	no = fftw_alloc_real( 2*alloc_local );
	nofNL = fftw_alloc_real( 2*alloc_local );

	kno = new complex<double>[ alloc_local ]; 
	knofNL = new complex<double>[ alloc_local ]; 
	cubno = fftw_alloc_real( 2*alloc_local );

	/***************** amplitude A ************************/
	for(ii=0;ii<2;ii++)
	{
		A[ii] = new complex<double>[ 2*alloc_local ]; 
		AfNL[ii] = new complex<double>[ 2*alloc_local ]; 

		cubA[ii] = new complex<double>[ 2*alloc_local ]; 

		quartA[ii] = new complex<double>[ 2*alloc_local ]; 
	}

	/***************** amplitude B ************************/
	for(ii=0;ii<2;ii++)
	{
		B[ii] = new complex<double>[ 2*alloc_local ]; 
		BfNL[ii] = new complex<double>[ 2*alloc_local ]; 

		cubB[ii] = new complex<double>[ 2*alloc_local ]; 

		quartB[ii] = new complex<double>[ 2*alloc_local ]; 
	}

	karr = fftw_alloc_real( alloc_local );
	
	karrx = fftw_alloc_real( 2*alloc_local );
	karry = fftw_alloc_real( 2*alloc_local );
	karrc = fftw_alloc_real( 2*alloc_local );

	// for average density
	CSqno = fftw_alloc_real( alloc_local );

	// for square amplitudes
	for(ii=0;ii<2;ii++)
	{
		CSqA[ii] = fftw_alloc_real( 2*alloc_local );

		CSqB[ii] = fftw_alloc_real( 2*alloc_local );
	}
}
void domainParams()
{
	//set up domain parameters
	if(myid==0) printf("dx =%f\n",dx);
	if(myid==0) printf("Nx = %zu\n",Nx);
	if(myid==0) printf("Ny = %zu\n",Ny);
	
	atomsx = (int)( floor(dxa/spacing*Nx) );
	atomsy = (int)( floor(dxa/spacing*Ny) );

	if(myid==0) printf("dxa=%f  numAtomsx=%d numAtomsy=%d\n",dxa, atomsx,atomsy);
	
	Nxn2 = Nx/2+1;
	Nx2 = 2*Nxn2;

	//set up fourier scaling factors
	facx = 2.*Pi/(Nx);
	facy = 2.*Pi/(Ny);
	normFac = 1.0/(Nx*Ny);
	
	if(myid==0) printf("facx=%lf\n",facx);
	if(myid==0) printf("facy=%lf\n",facy);
	if(myid==0) printf("normFac=%0.10lf\n",normFac);
}
void inputVariables()
{
	char *line = (char *) malloc (BUFSIZ * sizeof(char));
	FILE *in;
	in = fopen("xpfcSqAmpDis.in","rt");
   
	if(myid==0) printf("reading input file\n");
	if (in == NULL)
	{
		printf("Error opening input file\n");
		printf("Either file does not exit or incorrect file name\n");
		printf("Exiting simulation!\n");
		exit(1);	
	}
	else
	{
		line = fgets(line, BUFSIZ, in);
		sscanf (line, "%s", &run);
		line = fgets(line, BUFSIZ, in);
		sscanf (line, "%zu %zu", &Nx,&Ny);
		line = fgets(line, BUFSIZ, in);
		sscanf (line, "%lf %lf %lf",&spacing,&dx,&dt);
		line = fgets(line, BUFSIZ, in);
		sscanf (line, "%d %d", &totalTime,&printFreq);
		line = fgets(line, BUFSIZ, in);
		sscanf (line, "%d", &printFreqAmp);
		line = fgets(line, BUFSIZ, in);
		sscanf (line, "%lf %lf",&HSq0,&wSq0);
		line = fgets(line, BUFSIZ, in);
		sscanf (line, "%lf %lf",&wSq1,&wSq2);
		line = fgets(line, BUFSIZ, in);
		sscanf (line, "%lf %lf",&sigMSq1,&sigMSq2);
		line = fgets(line, BUFSIZ, in);
		sscanf (line, "%lf %lf", &iSig,&omega);
		line = fgets(line, BUFSIZ, in);
		sscanf (line, "%lf %lf %lf",&navg,&ns,&nl);
		line = fgets(line, BUFSIZ, in);
		sscanf (line, "%lf %lf",&eta,&chi);
		line = fgets(line, BUFSIZ, in);
		sscanf (line, "%lf %lf %lf",&Mno,&Ma,&Mb);
		line = fgets(line, BUFSIZ, in);
		sscanf (line, "%d %d",&restartFlag,&restartTime);
	}

	if(myid==0) 
	{
		printf("done reading input file\n");
		printf("run=%s\n", run);
		printf("Nx=%zu	Ny=%zu\n",Nx,Ny);
		printf("spacing=%lf	dx=%lf	dt=%lf\n",spacing,dx,dt);
		printf("totalTime=%d	printFreq=%d\n",totalTime,printFreq);
		printf("printFreqAmp=%d\n",printFreqAmp);
		printf("HSq0=%lf	wSq0=%lf\n",HSq0,wSq0);
		printf("wSq1=%lf	wSq2=%lf\n",wSq1,wSq2);
		printf("sigMSq1=%lf	sigMSq2=%lf\n",sigMSq1,sigMSq2);
		printf("iSig=%lf 	omega=%lf\n",iSig,omega);
		printf("navg=%lf	ns=%lf 	nl=%lf\n",navg,ns,nl);
		printf("eta=%lf	chi=%lf\n",eta,chi);
		printf("Mno=%lf	Ma=%lf	Mb=%lf\n",Mno,Ma,Mb);
		printf("rFlag=%d	rTime=%d\n",restartFlag,restartTime);
	}
}
void start_mpi(int argc, char *argv[])
{
	//starting mpi daemons
	MPI_Init(&argc,&argv); 
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
}
int main(int argc, char *argv[])
{
	int iter;
	int ii;
	double Dx;

	if( argc < 2)
    {
     	if(myid==0) printf("Not enough arguments!\n");
        exit(1);
    }

	//initiate mpi daemons
	start_mpi(argc,argv);

	//read from file the input values
	inputVariables();

	sprintf(run,"%s",argv[1]);
   	Dx = atof(argv[2]);
	dx = 1./Dx;

    if(myid==0) printf("dx = %lf \n",dx );

	//set up domain parameters
	domainParams();

	//allocating relevant arays
	allocateArrays();

	//setting up fft transform plans
	setfftwPlans();

	//set the reciprocal lattice vectors
	basisVectors();

	//set up initial conditions
	initialConditions();

	//calculate arrays for fourier operators
	k2Array();

	//calculate correlation peak related parameters
	setCorrPeakPrefactors(iSig);
	
	/* ============================================================
	   Mod 2: remove old history.dat on fresh run to avoid data accumulation
	   ============================================================ */
	if(restartFlag == 0 && myid == 0)
	{
		remove("history.dat");
		if(myid==0) printf("Old history.dat removed (if existed)\n");
	}
	MPI_Barrier(MPI_COMM_WORLD);
	
	/*********************************************************************************************************/
	//begin interations
	if(myid==0) printf("begin time iteration\n");	
	for(iter=restartTime+1; iter<totalTime+restartTime+1; iter++)
	{
		calcSumAmp();		//calculate the sum of all amplitudes

		iterateno();		//performs function calls for updating average density
		iterateA();			//performs function calls for updating A amplitudes
		iterateB();			//performs function calls for updating B amplitudes		

		//output current data
		if( (iter)%printFreq == 0 )
		{
			densReconAmp(no,A,B);

			// ========== scalar output ==========
			double n_sum = 0.0;
			double magA_sum = 0.0;
			double magB_sum = 0.0;
			double n_min = 1e10, n_max = -1e10;
			
			for(j=0;j<local_n0;j++)
			{
				for(i=0;i<Nx;i++)
				{
					indr = index2Dr(i,j);
					indc = index2Dc(i,j);
					
					n_sum += no[indr];
					magA_sum += magA[indr];
					magB_sum += magB[indr];
					
					if(no[indr] < n_min) n_min = no[indr];
					if(no[indr] > n_max) n_max = no[indr];
				}
			}
			
			// MPI reduce to root process
			double n_sum_global = 0.0, magA_sum_global = 0.0, magB_sum_global = 0.0;
			double n_min_global = 0.0, n_max_global = 0.0;
			
			MPI_Reduce(&n_sum, &n_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&magA_sum, &magA_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&magB_sum, &magB_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&n_min, &n_min_global, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
			MPI_Reduce(&n_max, &n_max_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			
			if(myid == 0)
			{
				double n_avg = n_sum_global / (Nx * Ny);
				double magA_avg = magA_sum_global / (Nx * Ny);
				double magB_avg = magB_sum_global / (Nx * Ny);
				
				FILE *fp_hist = fopen("history.dat", "a");
				if(fp_hist != NULL)
				{
					/* ============================================================
					   Mod 3: write header only on first output (also handles restart correctly)
					   ============================================================ */
					static int first_write = 1;
					if(first_write)
					{
						fprintf(fp_hist, "# iter  time  n_avg  n_min  n_max  magA_avg  magB_avg\n");
						first_write = 0;
					}
					fprintf(fp_hist, "%d  %d  %lf  %lf  %lf  %lf  %lf\n", 
						iter, iter, n_avg, n_min_global, n_max_global, magA_avg, magB_avg);
					fclose(fp_hist);
				}
				
				printf("=== t=%d | <n>=%.6f | n_min=%.3f n_max=%.3f | <|A|²>=%.6f | <|B|²>=%.6f ===\n",
					iter, n_avg, n_min_global, n_max_global, magA_avg, magB_avg);
			}
			// ========== end scalar output ==========

			//output average density
			output(iter,no,"n",run);

			//density reconstruction, amplitude magnitude and output
			output(iter,dens,"rcdens",run);
			output(iter,magA,"A",run);
			output(iter,magB,"B",run);

			for(ii=0;ii<2;ii++)
				output(iter,A[ii],"ampA",ii,run);

			for(ii=0;ii<2;ii++)
				output(iter,B[ii],"ampB",ii,run);

			if(myid==0) printf("time = %d\n",iter);
		}

		// //output amplitudes for restart
		// if( (iter)%printFreqAmp == 0 )
		// {
		// 	if(myid==0) printf("restart output for amplitudes\n");
		// 	if(myid==0) printf("time = %d\n",iter);

		// 	//output complex amplitude
		// 	//complex amplitudes
		// 	for(ii=0;ii<2;ii++)
		// 		output(iter,A[ii],"ampA",ii,run);

		// 	for(ii=0;ii<2;ii++)
		// 		output(iter,B[ii],"ampB",ii,run);
		// }
	}
	
	MPI_Barrier(MPI_COMM_WORLD);	//mpi barrier call -- waits for all cpus to reach this point	
	freeMemory();  //free memory and destroy fftw plans and arrays

	//exit mpi environment
	MPI_Finalize();

	return 0;
}