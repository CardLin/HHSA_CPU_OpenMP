#include<stdio.h>
#include<string.h> 
#include<math.h>
#include<cstdio>
#include<cstdlib>
#include<vector>
#include<iterator>
#include<random>
#include"spline.h"
#include<mgl2/mgl.h> 
#include <omp.h>

void FindExtrema(float* data, int length, bool* UpExtrema, int& UpLength, bool* LowExtrema, int& LowLength) {
	//Initial 
	for (int i = 0; i < length; ++i) {
		UpExtrema[i] = 0;
		LowExtrema[i] = 0;
	}
	UpLength = 0;
	LowLength = 0;

	//Find Extrema
	for (int i = 1; i < length - 1; ++i) {
		if (data[i] > data[i - 1] && data[i] > data[i + 1]) {
			UpExtrema[i] = 1;
			++UpLength;
		}
		else if (data[i] < data[i - 1] && data[i] < data[i + 1]) {
			LowExtrema[i] = 1;
			++LowLength;
		}
	}
}

void Spline(float* data, float* result, int length, bool* Extrema, int Length, int type) { //type 1 for EMD, 0 for NDQ
	//copy data to X-Y
	std::vector<double> X, Y;
	if (type) { //EMD
		X.push_back((double)0);
		Y.push_back((double)0.0);
	}
	for (int i = 1; i < length - 1; ++i) {
		if (Extrema[i]) {
			X.push_back((double)i);
			Y.push_back((double)data[i]);
		}
	}
	if (type) { //EMD
		X.push_back((double)length - 1);
		Y.push_back((double)0.0);
	}

	//initial spline
	tk::spline s;
	s.set_points(X, Y);

	//output spline result
	for (int i = 0; i < length; ++i) {
		result[i] = (float)s((double)i);
	}
}

float STDEV(float* data, int length) {
	float sum, avg, tmp, stdev;

	//Average of data[]
	sum = 0.0;
	for (int i = 0; i < length; ++i) {
		if (data[i] > 1000000000000.0 || data[i] < -1000000000000.0) continue;
		sum += data[i];
	}
	avg = sum / (float)length;

	//Standard Deviation
	sum = 0.0;
	for (int i = 0; i < length; ++i) {
		if (data[i] > 1000000000000.0 || data[i] < -1000000000000.0) continue;
		tmp = data[i] - avg;
		sum += tmp * tmp;
	}
	stdev = sqrt(sum) / (float)length;

	return stdev;
}

//Empirical Mode Decomposition
void EMD(float* data, float** result, float** residual, int length, int mode, float TargetSTDEV) {
	//Initial
	float* Residual = (float*)malloc(sizeof(float) * length);
	float* Current = (float*)malloc(sizeof(float) * length);
	bool* UpExtrema = (bool*)malloc(sizeof(bool) * length);
	bool* LowExtrema = (bool*)malloc(sizeof(bool) * length);
	float* UpEnvelope = (float*)malloc(sizeof(float) * length);
	float* LowEnvelope = (float*)malloc(sizeof(float) * length);
	float* MeanEnvelope = (float*)malloc(sizeof(float) * length);
	int UpLength, LowLength;
	int IterationCount;

	//Initial data
	for (int i = 0; i < length; ++i) {
		Residual[i] = data[i];
	}

	//Start EMD main loop
	for (int imf = 1; imf <= mode; ++imf) {
		//copy data from residual
		for (int i = 0; i < length; ++i) {
			Current[i] = Residual[i];
		}

		//Find IMF max Iteration=250
		IterationCount = 0;
		for (int iteration = 0; iteration < 250; ++iteration) {
			//Calculate Upper/Lower Envelope
			FindExtrema(Current, length, UpExtrema, UpLength, LowExtrema, LowLength);
			if (UpLength < 1 || LowLength < 1) break;
			Spline(Current, UpEnvelope, length, UpExtrema, UpLength, 1);
			Spline(Current, LowEnvelope, length, LowExtrema, LowLength, 1);
			//Calculate Mean Envelope & Subtract
			for (int i = 0; i < length; ++i) {
				MeanEnvelope[i] = (UpEnvelope[i] + LowEnvelope[i]) * 0.5;
				Current[i] = Current[i] - MeanEnvelope[i];
			}
			//printf("%d:%d:%f\n",imf,iteration,STDEV(MeanEnvelope,length));
			//printf("%f\n",TargetSTDEV);
			IterationCount++;
			if (STDEV(MeanEnvelope, length) < STDEV(Current, length) * TargetSTDEV) break;
			//printf("%d:%d\n",imf,iteration);
		}
		//printf("%d:%d:%fdone\n",imf,IterationCount,STDEV(MeanEnvelope,length));

		//Generate result & residual
		for (int i = 0; i < length; ++i) {
			result[imf - 1][i] = Current[i];
			Residual[i] -= Current[i];
			residual[imf - 1][i] = Residual[i];
		}
	}

	//Release memory
	free(Residual);
	free(Current);
	free(UpExtrema);
	free(LowExtrema);
	free(UpEnvelope);
	free(LowEnvelope);
	free(MeanEnvelope);
}

//Ensemble Empirical Mode Decomposition
void EEMD(float* data, float** result, float** residual, int length, int mode, float TargetSTDEV, int ensemble, bool CEEMD, bool* EMD_success) {
	//Enable CEEMD code
	//bool CEEMD=true;

	//omp_set_num_threads(8);

	//Initial
	float data_stdev, EnsembleNumber;
	float* Source = (float*)malloc(sizeof(float) * length);
	float* Current = (float*)malloc(sizeof(float) * length);
	float** MixtureResult = (float**)malloc(mode * sizeof(float*));
	//float** MixtureResidual = (float**)malloc(mode * sizeof(float*));
	for (int j = 0; j < mode; ++j) {
		MixtureResult[j] = (float*)malloc(length * sizeof(float));
		//MixtureResidual[j] = (float*)malloc(length * sizeof(float));
	}
	for (int j = 0; j < mode; ++j) {
		for (int i = 0; i < length; ++i) {
			MixtureResult[j][i] = 0.0;
			//MixtureResidual[j][i] = 0.0;
		}
	}
	float avg = 0.0;
	for (int i = 0; i < length; ++i) {
		avg += data[i];
	}
	avg /= (float)length;

	for (int i = 0; i < length; ++i) {
		data[i] -= avg;
	}

	//Define random generator with Gaussian distribution
	const double mean = 0.0;
	const double stddev = 0.1;
	std::default_random_engine generator;
	std::normal_distribution<double> dist(mean, stddev);

	//Generate input data
	data_stdev = STDEV(data, length);
	if (data_stdev < 0.01) data_stdev = 0.01;
	for (int i = 0; i < length; ++i) {
		Source[i] = data[i] / data_stdev;
	}

	//Start Ensemble loop
	//#pragma omp parallel for reduction(+:MixtureResult[0:mode][0:length]) //Array reduction is available after OpenMP 4.5 but MSVC2022 use OpenMP 2.0, doesn't test yet.
	#pragma omp parallel for
	for (int k = 0; k < ensemble; ++k) {
		//if(CEEMD) printf("CEEMD Ensemble=%d/%d\n",k+1,ensemble);
		//else printf("EEMD Ensemble=%d/%d\n",k+1,ensemble);

		//printf("Ensemble=%d\n", k);

		float* Input = (float*)malloc(sizeof(float) * length);
		float* Noise = (float*)malloc(sizeof(float) * length);
		float** Result = (float**)malloc(mode * sizeof(float*));
		float** Residual = (float**)malloc(mode * sizeof(float*));
		for (int j = 0; j < mode; ++j) {
			Result[j] = (float*)malloc(length * sizeof(float));
			Residual[j] = (float*)malloc(length * sizeof(float));
		}
		for (int j = 0; j < mode; ++j) {
			for (int i = 0; i < length; ++i) {
				MixtureResult[j][i] = 0.0;
				//MixtureResidual[j][i] = 0.0;
			}
		}

		//Add noise
		for (int i = 0; i < length; ++i) {
			Noise[i] = dist(generator);
			Input[i] = Source[i] + Noise[i];
			//printf("%f\n",Noise[i]);
		}

		//Execute EMD
		EMD(Input, Result, Residual, length, mode, TargetSTDEV);

		//#pragma omp critical
		//{
			//Add result to mixture result
			for (int j = 0; j < mode; ++j) {
				for (int i = 0; i < length; ++i) {
					#pragma omp atomic
					MixtureResult[j][i] += Result[j][i];
				}
			}
		//}

		if (CEEMD) {
			for (int i = 0; i < length; ++i) {
				Input[i] = Source[i] - Noise[i];
			}
			EMD(Input, Result, Residual, length, mode, TargetSTDEV);
			//#pragma omp critical
			//{
				for (int j = 0; j < mode; ++j) {
					for (int i = 0; i < length; ++i) {
						#pragma omp atomic
						MixtureResult[j][i] += Result[j][i];
					}
				}
			//}
		}
	}

	//Calculate Result from MixtureResult
	if (CEEMD) EnsembleNumber = ensemble * 2;
	else EnsembleNumber = ensemble;
	for (int j = 0; j < mode; ++j) {
		for (int i = 0; i < length; ++i) {
			result[j][i] = (data_stdev * MixtureResult[j][i]) / EnsembleNumber;
		}
	}

	//Fine Tune Result (fix ensemble noise)
	/*
	bool *UpExtrema=(bool*)malloc(sizeof(bool)*length);
	bool *LowExtrema=(bool*)malloc(sizeof(bool)*length);
	float *UpEnvelope=(float*)malloc(sizeof(float)*length);
	float *LowEnvelope=(float*)malloc(sizeof(float)*length);
	float *MeanEnvelope=(float*)malloc(sizeof(float)*length);
	int UpLength,LowLength;
	int IterationCount;
	for(int j=0;j<mode;++j){
		IterationCount=0;
		for(int iteration=0;iteration<100;++iteration){
			//Calculate Upper/Lower Envelope
			FindExtrema(result[j],length,UpExtrema,UpLength,LowExtrema,LowLength);
			if(UpLength<1 || LowLength<1) break;
			Spline(result[j],UpEnvelope,length,UpExtrema,UpLength,1);
			Spline(result[j],LowEnvelope,length,LowExtrema,LowLength,1);
			//Calculate Mean Envelope & Subtract
			for(int i=0;i<length;++i){
				MeanEnvelope[i]=(UpEnvelope[i]+LowEnvelope[i])*0.5;
				result[j][i]-=MeanEnvelope[i];
			}
			//printf("%d:%d:%f\n",imf,iteration,STDEV(MeanEnvelope,length));
			IterationCount++;
			if(STDEV(MeanEnvelope,length)<STDEV(result[j],length)*TargetSTDEV) break;
		}
		//printf("FineTune:%d:%d:%f\n",j+1,IterationCount,STDEV(MeanEnvelope,length));
	}
	*/

	//Calcuate Residual from Result
	for (int i = 0; i < length; ++i) {
		Current[i] = data[i] + avg;
	}
	for (int j = 0; j < mode; ++j) {
		//Check IMF result
		if (STDEV(result[j], length) > 0.01 * STDEV(data, length)) EMD_success[j] = true;
		else EMD_success[j] = false;

		//Calculate Residual
		for (int i = 0; i < length; ++i) {
			Current[i] = residual[j][i] = Current[i] - result[j][i];
		}
	}

	//Release Memory
	free(Source);
	free(Current);
	//free(Input);
	//free(Noise);
	for (int j = 0; j < mode; ++j) free(MixtureResult[j]);
	free(MixtureResult);
}

//3-points Median filter
float Median(float a, float b, float c) {
	if (a > b) {
		if (b > c) return b;	//abc
		else return c;		//acb
		return a;			//cab
	}
	else { //b>a
		if (a > c) return a;	//bac
		else return c;		//bca
		return b;			//cba
	}
}

//Normalized Direct Quadrature
void NDQ(float* data, float* FM, float* AM, float* IP, float* IF, int length, float dt, bool& NDQ_success) {
	//Initial	
	float* Absolute = (float*)malloc(sizeof(float) * length);
	for (int i = 0; i < length; ++i) {
		if (Absolute[i] < 0) Absolute[i] = -data[i];
		else Absolute[i] = data[i];
	}
	bool* UpExtrema = (bool*)malloc(sizeof(bool) * length);
	bool* LowExtrema = (bool*)malloc(sizeof(bool) * length);
	int UpLength, LowLength;
	float* UpEnvelope = (float*)malloc(sizeof(float) * length);
	float* PhaseAngle = (float*)malloc(sizeof(float) * length);
	float* RealFrequency = (float*)malloc(sizeof(float) * length);

	//Calculate FM
	for (int i = 0; i < length; ++i) FM[i] = data[i];
	for (int k = 0, end = 0; k < 100; ++k) {
		for (int i = 0; i < length; ++i) {
			if (FM[i] < 0.0) Absolute[i] = -FM[i];
			else Absolute[i] = FM[i];
		}
		FindExtrema(Absolute, length, UpExtrema, UpLength, LowExtrema, LowLength);
		if (UpLength < 3) break;
		Spline(Absolute, UpEnvelope, length, UpExtrema, UpLength, 0);
		for (int i = 0; i < length; ++i) {
			if (UpEnvelope[i] < 0.01) UpEnvelope[i] = 0.01;
			FM[i] = FM[i] / UpEnvelope[i];
		}

		//End iteration?
		end = 1;
		for (int i = 0; i < length; ++i) {
			if (FM[i] > 1.0 || FM[i] < -1.0) end = 0;
		}
		if (end) break;
	}

	//Check FM modulation
	NDQ_success = true;
	for (int i = 0; i < length; ++i) {
		if (FM[i] > 1.0 || FM[i] < -1.0) {
			NDQ_success = false;
			return;
		}
	}

	//Calculate AM
	AM[0] = 0.0;
	for (int i = 1; i < length - 1; ++i) {
		AM[i] = data[i] / FM[i];
	}
	AM[length - 1] = 0.0;

	//Direct Quadrature
	for (int i = 0; i < length; ++i) {
		PhaseAngle[i] = atan(FM[i] / sqrt(1 - FM[i] * FM[i]));
	}

	IP[0] = PhaseAngle[0];
	for (int i = 1; i < length; ++i) {
		IP[i] = IP[i - 1] + fabs(PhaseAngle[i - 1] - PhaseAngle[i]);
	}

	//Calculate IF
	for (int i = 0; i < length - 1; ++i) {
		RealFrequency[i] = (IP[i + 1] - IP[i]) / dt;
	}
	RealFrequency[length - 1] = RealFrequency[length - 2];

	//Apply 3-point Median Filter
	IF[0] = RealFrequency[0];
	for (int i = 1; i < length - 1; ++i) {
		IF[i] = Median(RealFrequency[i - 1], RealFrequency[i], RealFrequency[i + 1]);
	}
	IF[length - 1] = RealFrequency[length - 1];


	//Release Memory
	free(Absolute);
	free(UpExtrema);
	free(LowExtrema);
	free(UpEnvelope);
}

//Hilbert Spectrum Analysis
void HSA(float** IF, float** AM, int length, int mode, bool* NDQ_success, float** map, int Time_cell, int Freq_cell, float& max_frequency) {
	//Initial Variable
	for (int j = 0; j < Time_cell; ++j) {
		for (int i = 0; i < Freq_cell; ++i) {
			map[j][i] = 0.0;
		}
	}
	float maxF, dF, maxT, dT;
	int x, y;

	//Find max frequency & time
	maxF = 0.0;
	for (int j = 0; j < mode; ++j) {
		if (NDQ_success[j]) {
			for (int i = 0; i < length; ++i) {
				if (IF[j][i] > maxF) maxF = IF[j][i];
			}
		}
	}
	maxT = (float)length;
	max_frequency = maxF;
	dT = maxT / (float)(Time_cell - 1);
	dF = maxF / (float)(Freq_cell - 1);

	//Calculate HSA map
	for (int i = 0; i < length; ++i) {
		x = (int)((float)i / dT);		//Time
		if (x > Time_cell - 1) printf("x=%d out of range\n", x);
		for (int j = 0; j < mode; ++j) {
			if (NDQ_success[j]) {
				y = (int)(IF[j][i] / dF);		//Frequency
				if (y > Freq_cell - 1) printf("y=%d out of range\n", y);
				map[x][y] += AM[j][i];		//[Time][Frequency]
			}
		}
	}
}

//Holo Hilbert Spectra Analysis
void HHSA(float** imf, int length, int mode, float TargetSTDEV, int ensemble, bool CEEMD, bool* EMD_success, float*** Holo_result, float*** Holo_residual, bool** Holo_EMD_success) {
	bool* UpExtrema = (bool*)malloc(sizeof(bool) * length);
	bool* LowExtrema = (bool*)malloc(sizeof(bool) * length);
	int UpLength, LowLength;
	float* UpEnvelope = (float*)malloc(sizeof(float) * length);

	for (int j = 0; j < mode; ++j) {
		if (EMD_success[j]) { //FM EMD success
			printf("Apply HHSA on mode%d\n", j + 1);
			FindExtrema(imf[j], length, UpExtrema, UpLength, LowExtrema, LowLength);
			if (UpLength < 3) break;
			Spline(imf[j], UpEnvelope, length, UpExtrema, UpLength, 0);
			EEMD(UpEnvelope, Holo_result[j], Holo_residual[j], length, mode, TargetSTDEV, ensemble, CEEMD, Holo_EMD_success[j]);
		}
		else {
			for (int k = 0; k < mode; ++k) {
				Holo_EMD_success[j][k] = false;
			}
		}
	}
}

//Holo Hilbert Spectrum
void HHS(float** IF, float*** Holo_IF, float*** Holo_AM, int length, int mode, bool* NDQ_success, bool** Holo_NDQ_success, float** map, int Freq_cell, int Holo_Freq_cell, float& max_FM_frequency, float& max_AM_frequency) {
	//Initial Variable
	for (int j = 0; j < Freq_cell; ++j) {
		for (int i = 0; i < Holo_Freq_cell; ++i) {
			map[j][i] = 0.0;
		}
	}
	float maxAMF, dAMF, maxFMF, dFMF;
	int x, y;

	//Measure max FM frequency
	maxFMF = 0.0;
	for (int j = 0; j < mode; ++j) {
		if (NDQ_success[j]) {
			for (int i = 0; i < length; ++i) {
				if (IF[j][i] > maxFMF) maxFMF = IF[j][i];
			}
		}
	}
	dFMF = maxFMF / (float)(Freq_cell - 1);
	max_FM_frequency = maxFMF;

	//Measure max AM frequency
	maxAMF = 0.0;
	for (int j = 0; j < mode; ++j) {
		for (int k = 0; k < mode; ++k) {
			if (Holo_NDQ_success[j][k]) {
				for (int i = 0; i < length; ++i) {
					if (Holo_IF[j][k][i] > maxAMF) maxAMF = Holo_IF[j][k][i];
				}
			}
		}
	}
	dAMF = maxAMF / (float)(Holo_Freq_cell - 1);
	max_AM_frequency = maxAMF;

	//Calculate HHS map
	for (int i = 0; i < length; ++i) {
		for (int j = 0; j < mode; ++j) {
			if (NDQ_success[j]) {
				x = (int)(IF[j][i] / dFMF);	//FM
				if (x > Freq_cell - 1) printf("x=%d out of range\n", x);
				for (int k = 0; k < mode; ++k) {
					if (Holo_NDQ_success[j][k]) {
						y = (int)(Holo_IF[j][k][i] / dAMF);	//AM
						if (y > Holo_Freq_cell - 1) printf("y=%d out of range\n", y);
						map[x][y] += Holo_AM[j][k][i];		//[FM][AM]
					}
				}
			}
		}
	}
}

//Holo Hilbert Spectrum
void HHS2(float** IF, float*** Holo_IF, float*** Holo_AM, int length, int mode, bool* NDQ_success, bool** Holo_NDQ_success, float** map, float*** map2, int Time_cell, int Freq_cell, int Holo_Freq_cell, float& max_FM_frequency, float& max_AM_frequency) {
	//Initial Variable
	for (int j = 0; j < Freq_cell; ++j) {
		for (int i = 0; i < Holo_Freq_cell; ++i) {
			map[j][i] = 0.0;
			for (int k = 0; k < Time_cell; ++k) {
				map2[k][j][i] = 0.0;
			}
		}
	}
	float maxT, dT, maxAMF, dAMF, maxFMF, dFMF;
	int t, x, y;

	//Find max time
	maxT = (float)length;
	dT = maxT / (float)(Time_cell - 1);

	//Measure max FM frequency
	maxFMF = 0.0;
	for (int j = 0; j < mode; ++j) {
		if (NDQ_success[j]) {
			for (int i = 0; i < length; ++i) {
				if (IF[j][i] > maxFMF) maxFMF = IF[j][i];
			}
		}
	}
	dFMF = maxFMF / (float)(Freq_cell - 1);
	max_FM_frequency = maxFMF;

	//Measure max AM frequency
	maxAMF = 0.0;
	for (int j = 0; j < mode; ++j) {
		for (int k = 0; k < mode; ++k) {
			if (Holo_NDQ_success[j][k]) {
				for (int i = 0; i < length; ++i) {
					if (Holo_IF[j][k][i] > maxAMF) maxAMF = Holo_IF[j][k][i];
				}
			}
		}
	}
	dAMF = maxAMF / (float)(Holo_Freq_cell - 1);
	max_AM_frequency = maxAMF;

	//Calculate HHS map
	for (int i = 0; i < length; ++i) {
		t = (int)((float)i / dT);		//Time
		for (int j = 0; j < mode; ++j) {
			if (NDQ_success[j]) {
				x = (int)(IF[j][i] / dFMF);	//FM
				if (x > Freq_cell - 1) printf("x=%d out of range\n", x);
				for (int k = 0; k < mode; ++k) {
					if (Holo_NDQ_success[j][k]) {
						y = (int)(Holo_IF[j][k][i] / dAMF);	//AM
						if (y > Holo_Freq_cell - 1) printf("y=%d out of range\n", y);
						map[x][y] += Holo_AM[j][k][i];		//[FM][AM]
						map2[t][x][y] += Holo_AM[j][k][i];	//[Time][FM][AM]
					}
				}
			}
		}
	}
}

int main(int argc, char* argv[]) {
	puts("HHSA version 1.2.3 (CardLin)");

	FILE* fp;
	char InputFileName[255], OutputFileName[255], Text[255];
	float data[10000];
	int data_len = 0;
	int mode, ensemble;
	int Time_cell, Freq_cell, Holo_Freq_cell, ImageSizeX1, ImageSizeY1, ImageSizeX2, ImageSizeY2;
	float TargetSTDEV, dt; //dt=1.0/SampleRate(hz) 
	bool CEEMD = true; //Enable CEEMD?

	//Read default profile
	if ((fp = fopen("HHSA_profile.txt", "r")) == NULL) {
		puts("Read profile Error!!");
		system("pause");
		exit(1);
	}
	fscanf(fp, "%d", &mode);
	fscanf(fp, "%d", &ensemble);
	fscanf(fp, "%f", &TargetSTDEV);
	fscanf(fp, "%f", &dt);
	fscanf(fp, "%d", &Time_cell);
	fscanf(fp, "%d", &Freq_cell);
	fscanf(fp, "%d", &Holo_Freq_cell);
	fscanf(fp, "%d", &ImageSizeX1);
	fscanf(fp, "%d", &ImageSizeY1);
	fscanf(fp, "%d", &ImageSizeX2);
	fscanf(fp, "%d", &ImageSizeY2);
	fclose(fp);

	/*
	HSA(Time,IF,AM)  &&  HHS(IF,HoloIF,HoloAM)
	Time data 		--Processing--> 	IF,AM 				(instant frequency of data, data amplitude)
	IMF amplitude 	--Processing-->		Holo_IF, Holo_AM	(instant frequency of IMF amplitude, amplitude of instant frequency of IMF amplitude)
	The sum of IMF is data.
	*/

	printf("CEEMD=%d\n", CEEMD);
	printf("Mode=%d\n", mode);
	printf("Ensemble=%d\n", ensemble);
	printf("EMD Stop Criteria (STDEV)=%.10f\n", TargetSTDEV);
	printf("NDQ dt (1/SampleRate)=%.10f\n", dt);
	printf("Time_cell=%d\n", Time_cell);
	printf("Freq_cell=%d\n", Freq_cell);
	printf("Holo_Freq_cell=%d\n", Holo_Freq_cell);
	printf("ImageSizeX1=%d\n", ImageSizeX1);
	printf("ImageSizeY1=%d\n", ImageSizeY1);
	printf("ImageSizeX2=%d\n", ImageSizeX2);
	printf("ImageSizeY2=%d\n", ImageSizeY2);
	printf("\n");

	//Read Input File
	if (argc < 2) {
		puts("Please input a raw file!!");
		system("pause");
		exit(1);
	}
	strcpy(InputFileName, argv[1]);
	printf("Read file from %s  ", InputFileName);
	if ((fp = fopen(InputFileName, "r")) == NULL) {
		puts("Open File Error!!");
		system("pause");
		exit(1);
	}
	while (!feof(fp)) {
		fscanf(fp, "%f", &data[data_len++]);
	}
	fclose(fp);
	printf("\nDataLength=%d", data_len);
	printf("\n\n");

	//Initial Variable for HHT
	bool* NDQ_success = (bool*)malloc(mode * sizeof(bool));
	bool* EMD_success = (bool*)malloc(mode * sizeof(bool));
	float** result, ** residual, ** AM, ** FM, ** IP, ** IF; //InstaneousFrequency
	float** UpEnvelope, ** LowEnvelope, ** MeanEnvelope;
	result = (float**)malloc(mode * sizeof(float*));
	residual = (float**)malloc(mode * sizeof(float*));
	FM = (float**)malloc(mode * sizeof(float*));
	AM = (float**)malloc(mode * sizeof(float*));
	IP = (float**)malloc(mode * sizeof(float*));
	IF = (float**)malloc(mode * sizeof(float*));
	UpEnvelope = (float**)malloc(mode * sizeof(float*));
	LowEnvelope = (float**)malloc(mode * sizeof(float*));
	MeanEnvelope = (float**)malloc(mode * sizeof(float*));
	for (int j = 0; j < mode; ++j) {
		EMD_success[j] = true;
		NDQ_success[j] = true;
		result[j] = (float*)malloc(data_len * sizeof(float));
		residual[j] = (float*)malloc(data_len * sizeof(float));
		FM[j] = (float*)malloc(data_len * sizeof(float));
		AM[j] = (float*)malloc(data_len * sizeof(float));
		IP[j] = (float*)malloc(data_len * sizeof(float));
		IF[j] = (float*)malloc(data_len * sizeof(float));
		UpEnvelope[j] = (float*)malloc(data_len * sizeof(float));
		LowEnvelope[j] = (float*)malloc(data_len * sizeof(float));
		MeanEnvelope[j] = (float*)malloc(data_len * sizeof(float));
	}
	bool* UpExtrema = (bool*)malloc(sizeof(bool) * data_len);
	bool* LowExtrema = (bool*)malloc(sizeof(bool) * data_len);
	int UpLength, LowLength;

	//Initial Variable for HHSA
	float*** Holo_result, *** Holo_residual, *** Holo_AM, *** Holo_FM, *** Holo_IP, *** Holo_IF;
	bool** Holo_EMD_success, ** Holo_NDQ_success;
	Holo_result = (float***)malloc(mode * sizeof(float**));
	Holo_residual = (float***)malloc(mode * sizeof(float**));
	Holo_AM = (float***)malloc(mode * sizeof(float**));
	Holo_FM = (float***)malloc(mode * sizeof(float**));
	Holo_IP = (float***)malloc(mode * sizeof(float**));
	Holo_IF = (float***)malloc(mode * sizeof(float**));
	Holo_EMD_success = (bool**)malloc(mode * sizeof(bool*));
	Holo_NDQ_success = (bool**)malloc(mode * sizeof(bool*));
	for (int j = 0; j < mode; ++j) {	//FM
		Holo_result[j] = (float**)malloc(mode * sizeof(float*));
		Holo_residual[j] = (float**)malloc(mode * sizeof(float*));
		Holo_AM[j] = (float**)malloc(mode * sizeof(float*));
		Holo_FM[j] = (float**)malloc(mode * sizeof(float*));
		Holo_IP[j] = (float**)malloc(mode * sizeof(float*));
		Holo_IF[j] = (float**)malloc(mode * sizeof(float*));
		Holo_EMD_success[j] = (bool*)malloc(mode * sizeof(bool));
		Holo_NDQ_success[j] = (bool*)malloc(mode * sizeof(bool));
		for (int k = 0; k < mode; ++k) {	//AM
			Holo_result[j][k] = (float*)malloc(data_len * sizeof(float));
			Holo_residual[j][k] = (float*)malloc(data_len * sizeof(float));
			Holo_AM[j][k] = (float*)malloc(data_len * sizeof(float));
			Holo_FM[j][k] = (float*)malloc(data_len * sizeof(float));
			Holo_IP[j][k] = (float*)malloc(data_len * sizeof(float));
			Holo_IF[j][k] = (float*)malloc(data_len * sizeof(float));
			Holo_EMD_success[j][k] = true;
			Holo_NDQ_success[j][k] = true;
		}
	}

	//Initial Variable for MathGL
	mglData HSA_map(Time_cell, Freq_cell), HHS_map(Freq_cell, Holo_Freq_cell), HHSA_map(Freq_cell, Holo_Freq_cell);
	mglGraph HSA_gr, HHS_gr, HHSA_gr;

	//Initial Variable for HSA graph
	float max_frequency, max_time;
	float HSA_df, HSA_dt, HSA_sum, HSA_avg, HSA_stdev, HSA_max, HSA_min;
	float* HSA_raw = (float*)malloc(sizeof(float) * Time_cell * Freq_cell);

	float** hsa_map; //Time_cell*Freq_cell
	hsa_map = (float**)malloc(Time_cell * sizeof(float*));
	for (int j = 0; j < Time_cell; ++j) hsa_map[j] = (float*)malloc(Freq_cell * sizeof(float));

	//Initial Variable for HHS graph
	float max_FM_frequency, max_AM_frequency;
	float HHS_dAMf, HHS_dFMf, HHS_stdev, HHS_sum, HHS_avg, HHS_max, HHS_min;
	float* HHS_raw = (float*)malloc(sizeof(float) * Freq_cell * Holo_Freq_cell);

	float** hhs_map; //Freq_cell*Holo_Freq_cell
	hhs_map = (float**)malloc(Freq_cell * sizeof(float*));
	for (int j = 0; j < Freq_cell; ++j) hhs_map[j] = (float*)malloc(Holo_Freq_cell * sizeof(float));

	//Initial Variable for HHSA graph
	float HHSA_dt, HHSA_dAMf, HHSA_dFMf, HHSA_stdev, HHSA_sum, HHSA_avg, HHSA_max, HHSA_min;
	float* HHSA_raw = (float*)malloc(sizeof(float) * Time_cell * Freq_cell * Holo_Freq_cell);
	float*** hhsa_map; //Time_cell*Freq_cell*Holo_Freq_cell
	hhsa_map = (float***)malloc(Time_cell * sizeof(float**));
	for (int i = 0; i < Time_cell; ++i) {
		hhsa_map[i] = (float**)malloc(Freq_cell * sizeof(float*));
		for (int j = 0; j < Freq_cell; ++j) {
			hhsa_map[i][j] = (float*)malloc(Holo_Freq_cell * sizeof(float));
		}
	}

	if (hhsa_map[Time_cell - 1][Freq_cell - 1] == NULL) {
		printf("malloc bug!!\n");
		exit(1);
	}

	//Execute HHT
	printf("Apply HHT on input data");
	EEMD(data, result, residual, data_len, mode, TargetSTDEV, ensemble, CEEMD, EMD_success);
	printf("\n\n");

	if (CEEMD) printf("CEEMD report:\n");
	else printf("EEMD report:\n");
	for (int j = 0; j < mode; ++j) printf("%d", (j + 1) % 10);
	puts("");
	for (int j = 0; j < mode; ++j) {
		//printf("IMF Mode%d ",j+1);
		if (EMD_success[j]) printf("O");
		else printf("X");

		if (EMD_success[j]) {
			//Generate Envelope
			FindExtrema(result[j], data_len, UpExtrema, UpLength, LowExtrema, LowLength);
			Spline(result[j], UpEnvelope[j], data_len, UpExtrema, UpLength, 1);
			Spline(result[j], LowEnvelope[j], data_len, LowExtrema, LowLength, 1);
			//Calculate Mean Envelope & Subtract
			for (int i = 0; i < data_len; ++i) {
				MeanEnvelope[j][i] = (UpEnvelope[j][i] + LowEnvelope[j][i]) / 2.0;
			}
		}
	}
	printf("\n\n");

	//Normalized Direct Quadrature
	printf("NDQ report:\n");
	for (int j = 0; j < mode; ++j) printf("%d", (j + 1) % 10);
	puts("");
	for (int j = 0; j < mode; ++j) {
		if (EMD_success[j] == false) {
			NDQ_success[j] = false;
			printf("X");
			continue;
		}
		//printf("Apply NDQ on mode%d... ",j+1);
		NDQ(result[j], FM[j], AM[j], IP[j], IF[j], data_len, dt, NDQ_success[j]);
		if (NDQ_success[j]) printf("O");
		else printf("X");
	}
	printf("\n\n");

	//Initial Output File
	/*
	if(CEEMD) sprintf(OutputFileName,"%s_CEEMD.csv",InputFileName);
	else sprintf(OutputFileName,"%s_EEMD.csv",InputFileName);
	*/
	sprintf(OutputFileName, "%s_HHT.csv", InputFileName);
	if ((fp = fopen(OutputFileName, "w")) == NULL) {
		puts("Open OutputFile Error!!");
		system("pause");
		exit(1);
	}

	//Output Result
	fprintf(fp, "data,");
	for (int imf = 0; imf < mode; ++imf) {
		if (EMD_success[imf]) {
			fprintf(fp, "IMF%d,", imf + 1);
			fprintf(fp, "Residual%d,", imf + 1);
			fprintf(fp, "Up%d,", imf + 1);
			fprintf(fp, "Low%d,", imf + 1);
			fprintf(fp, "Mean%d,", imf + 1);
		}
		if (NDQ_success[imf]) {
			fprintf(fp, "FM%d,", imf + 1);
			fprintf(fp, "AM%d,", imf + 1);
			fprintf(fp, "IP%d,", imf + 1);
			fprintf(fp, "IF%d,", imf + 1);
		}
	}
	fprintf(fp, "\n");
	for (int i = 0; i < data_len; ++i) {
		fprintf(fp, "%f,", data[i]);
		for (int imf = 0; imf < mode; ++imf) {
			if (EMD_success[imf]) {
				fprintf(fp, "%f,", result[imf][i]);
				fprintf(fp, "%f,", residual[imf][i]);
				fprintf(fp, "%f,", UpEnvelope[imf][i]);
				fprintf(fp, "%f,", LowEnvelope[imf][i]);
				fprintf(fp, "%f,", MeanEnvelope[imf][i]);
			}
			if (NDQ_success[imf]) {
				fprintf(fp, "%f,", FM[imf][i]);
				fprintf(fp, "%f,", AM[imf][i]);
				fprintf(fp, "%f,", IP[imf][i]);
				fprintf(fp, "%f,", IF[imf][i]);
			}
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	//Execute Hilbert Spetra Analysis
	HSA(IF, AM, data_len, mode, NDQ_success, hsa_map, Time_cell, Freq_cell, max_frequency);

	printf("Generate HSA result  ");

	//Output HSA parameter
	sprintf(OutputFileName, "%s_HSA.txt", InputFileName);
	if ((fp = fopen(OutputFileName, "w")) == NULL) {
		printf("Open %s Error!!\n", OutputFileName);
		system("pause");
		exit(1);
	}
	max_time = dt * (float)data_len;
	fprintf(fp, "max_time=%f\nmax_frequecny=%f\n", max_time, max_frequency);
	fclose(fp);

	// Output HSA to csv
	FILE* map_fp;
	sprintf(OutputFileName, "%s_HSA.csv", InputFileName);
	if ((map_fp = fopen(OutputFileName, "w")) == NULL) {
		printf("Open %s Error!!\n", OutputFileName);
		system("pause");
		exit(1);
	}
	for (int j = 0; j < Time_cell; ++j) {
		for (int i = 0; i < Freq_cell; ++i) {
			fprintf(map_fp, "%f,", hsa_map[j][i]);	//[Time][Frequency]
		}
		fprintf(map_fp, "\n");
	}
	fclose(map_fp);

	//Output HSA to dat
	HSA_sum = 0.0;
	HSA_max = -1000000000000.0;
	HSA_min = 1000000000000.0;
	for (int j = 0, k = 0; j < Time_cell; ++j) {
		for (int i = 0; i < Freq_cell; ++i) {
			HSA_raw[k++] = hsa_map[j][i];
			if (hsa_map[j][i] > 1000000000000.0 || hsa_map[j][i] < -1000000000000.0) continue;
			HSA_sum += hsa_map[j][i];					//[Time][Frequency]
		}
	}
	HSA_avg = HSA_sum / (float)(Time_cell * Freq_cell);
	HSA_stdev = STDEV(HSA_raw, Time_cell * Freq_cell);
	HSA_dt = max_time / Time_cell;
	HSA_df = max_frequency / Freq_cell;
	//FILE *map_fp;
	sprintf(OutputFileName, "%s_HSA.dat", InputFileName);
	if ((map_fp = fopen(OutputFileName, "w")) == NULL) {
		printf("Open %s Error!!\n", OutputFileName);
		system("pause");
		exit(1);
	}
	for (int j = 0; j < Time_cell; ++j) {
		for (int i = 0; i < Freq_cell; ++i) {
			if (hsa_map[j][i] > 1000000000000.0 || hsa_map[j][i] < -1000000000000.0) continue;
			fprintf(map_fp, "%f %f %f\n", j * HSA_dt, i * HSA_df, hsa_map[j][i]);
			HSA_map[i * Time_cell + j] = hsa_map[j][i] + 0.00000001;		//[Time][Frequency]
			if (hsa_map[j][i] > HSA_max) HSA_max = hsa_map[j][i];
			if (hsa_map[j][i] < HSA_min) HSA_min = hsa_map[j][i];
		}
		//fprintf(map_fp,"\n");
	}
	fclose(map_fp);

	//output HSA to png
	sprintf(OutputFileName, "%s_HSA.png", InputFileName);
	HSA_gr.SetSize(ImageSizeX1, ImageSizeY1);
	HSA_gr.Title("Hilbert Spectra Analysis");
	HSA_gr.Label('x', "Time", 0);
	HSA_gr.Label('y', "Frequency", 0);
	HSA_gr.SetRange('c', HSA_min, log(HSA_max));
	HSA_gr.SetFunc("", "", "", "lg(c)");
	HSA_gr.Dens(HSA_map);
	HSA_gr.Colorbar(">");
	HSA_gr.Puts(mglPoint(1.35, 1.2), "log-scale\n(10^n)");
	HSA_gr.SetRange('x', 0.0, max_time);
	HSA_gr.SetRange('y', 0.0, max_frequency);
	HSA_gr.Axis();
	HSA_gr.WriteFrame(OutputFileName);

	//update HSA parameter
	sprintf(OutputFileName, "%s_HSA.txt", InputFileName);
	if ((fp = fopen(OutputFileName, "a+")) == NULL) {
		printf("Open %s Error!!\n", OutputFileName);
		system("pause");
		exit(1);
	}
	fprintf(fp, "Time_cell=%f\nFrequency_cell=%f\n", Time_cell, Freq_cell);
	fprintf(fp, "Time_per_cell=%f\nFrequency_per_cell=%f\n", HSA_dt, HSA_df);
	fprintf(fp, "HSA_avg=%f\nHSA_stdev=%f\n", HSA_avg, HSA_stdev);
	fprintf(fp, "HSA_max=%f\nHSA_min=%f\n", HSA_max, HSA_min);
	fclose(fp);

	printf("\n\n");

	HHSA(result, data_len, mode, TargetSTDEV, ensemble, CEEMD, EMD_success, Holo_result, Holo_residual, Holo_EMD_success);

	printf("\n");

	if (CEEMD) printf("HHSA CEEMD report:\n");
	else printf("HHSA EEMD report:\n");
	printf(" ");
	for (int k = 0; k < mode; ++k) printf("%d", (k + 1) % 10);
	printf("\n");
	for (int j = 0; j < mode; ++j) {
		printf("%d", (j + 1) % 10);
		for (int k = 0; k < mode; ++k) {
			if (Holo_EMD_success[j][k]) printf("O");
			else printf("X");
		}
		printf("\n");
	}
	printf("\n");

	printf("HHSA NDQ report:\n");
	printf(" ");
	for (int k = 0; k < mode; ++k) printf("%d", (k + 1) % 10);
	printf("\n");
	for (int j = 0; j < mode; ++j) {
		printf("%d", (j + 1) % 10);
		for (int k = 0; k < mode; ++k) {
			if (Holo_EMD_success[j][k] == false) {
				Holo_NDQ_success[j][k] = false;
				printf("X");
				continue;
			}
			//printf("Apply NDQ on FM mode%d, AM mode%d ... ",j+1,k+1);
			NDQ(Holo_result[j][k], Holo_FM[j][k], Holo_AM[j][k], Holo_IP[j][k], Holo_IF[j][k], data_len, dt, Holo_NDQ_success[j][k]);
			if (Holo_NDQ_success[j][k]) printf("O");
			else printf("X");
		}
		printf("\n");
	}
	printf("\n");

	//Output HHSA result
	for (int j = 0; j < mode; ++j) {
		sprintf(OutputFileName, "%s_HHSA_mode%d.csv", InputFileName, j + 1);
		if ((fp = fopen(OutputFileName, "w")) == NULL) {
			printf("Open %s Error!!\n", OutputFileName);
			system("pause");
			exit(1);
		}
		if (EMD_success[j]) {
			fprintf(fp, "Holo%dUp,", j + 1);
		}
		for (int k = 0; k < mode; ++k) {
			if (Holo_EMD_success[j][k]) {
				fprintf(fp, "IMF%d,", k + 1);
				fprintf(fp, "Residual%d,", k + 1);
			}
			if (Holo_NDQ_success[j][k]) {
				fprintf(fp, "FM%d,", k + 1);
				fprintf(fp, "AM%d,", k + 1);
				fprintf(fp, "IP%d,", k + 1);
				fprintf(fp, "IF%d,", k + 1);
			}
		}
		fprintf(fp, "\n");
		for (int i = 0; i < data_len; ++i) {
			if (EMD_success[j]) {
				fprintf(fp, "%f,", result[j][i]);
			}
			for (int k = 0; k < mode; ++k) {
				if (Holo_EMD_success[j][k]) {
					fprintf(fp, "%f,", Holo_result[j][k][i]);
					fprintf(fp, "%f,", Holo_residual[j][k][i]);
				}
				if (Holo_NDQ_success[j][k]) {
					fprintf(fp, "%f,", Holo_FM[j][k][i]);
					fprintf(fp, "%f,", Holo_AM[j][k][i]);
					fprintf(fp, "%f,", Holo_IP[j][k][i]);
					fprintf(fp, "%f,", Holo_IF[j][k][i]);
				}
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	//Execute Holo Hilbert Spectrum
	//HHS(IF,Holo_IF,Holo_AM,data_len,mode,NDQ_success,Holo_NDQ_success,hhs_map,Freq_cell,Holo_Freq_cell,max_FM_frequency,max_AM_frequency);
	HHS2(IF, Holo_IF, Holo_AM, data_len, mode, NDQ_success, Holo_NDQ_success, hhs_map, hhsa_map, Time_cell, Freq_cell, Holo_Freq_cell, max_FM_frequency, max_AM_frequency);

	printf("Generate HHS result  ");

	//Output HHS parameter
	sprintf(OutputFileName, "%s_HHS.txt", InputFileName);
	if ((fp = fopen(OutputFileName, "w")) == NULL) {
		printf("Open %s Error!!\n", OutputFileName);
		system("pause");
		exit(1);
	}
	fprintf(fp, "max_AM_frequecny=%f\nmax_FM_frequency=%f\n", max_AM_frequency, max_FM_frequency);
	fclose(fp);

	// Output HHS to csv
	sprintf(OutputFileName, "%s_HHS.csv", InputFileName);
	if ((map_fp = fopen(OutputFileName, "w")) == NULL) {
		printf("Open %s Error!!\n", OutputFileName);
		system("pause");
		exit(1);
	}
	for (int j = 0; j < Freq_cell; ++j) {
		for (int i = 0; i < Holo_Freq_cell; ++i) {
			fprintf(map_fp, "%f,", hhs_map[j][i]);	//[FM][AM]
		}
		fprintf(map_fp, "\n");
	}
	fclose(map_fp);

	// Output HHS to dat
	HHS_sum = 0.0;
	HHS_max = -1000000000000.0;
	HHS_min = 1000000000000.0;
	for (int j = 0, k = 0; j < Freq_cell; ++j) {
		for (int i = 0; i < Holo_Freq_cell; ++i) {
			HHS_raw[k++] = hhs_map[j][i];
			if (hhs_map[j][i] > 1000000000000.0 || hhs_map[j][i] < -1000000000000.0) continue;
			HHS_sum += hhs_map[j][i];					//[FM][AM]
		}
	}
	HHS_avg = HHS_sum / (float)(Freq_cell * Holo_Freq_cell);
	HHS_stdev = STDEV(HHS_raw, Freq_cell * Holo_Freq_cell);
	HHS_dFMf = max_FM_frequency / Freq_cell;
	HHS_dAMf = max_AM_frequency / Holo_Freq_cell;
	sprintf(OutputFileName, "%s_HHS.dat", InputFileName);
	if ((map_fp = fopen(OutputFileName, "w")) == NULL) {
		printf("Open %s Error!!\n", OutputFileName);
		system("pause");
		exit(1);
	}
	for (int j = 0; j < Freq_cell; ++j) {
		for (int i = 0; i < Holo_Freq_cell; ++i) {
			if (hhs_map[j][i] > 1000000000000.0 || hhs_map[j][i] < -1000000000000.0) continue;
			fprintf(map_fp, "%f %f %f\n", HHS_dFMf * j, HHS_dAMf * i, hhs_map[j][i]);
			HHS_map[i * Freq_cell + j] = hhs_map[j][i] + 0.00000001;	//[FM][AM]
			if (hhs_map[j][i] > HHS_max) HHS_max = hhs_map[j][i];
			if (hhs_map[j][i] < HHS_min) HHS_min = hhs_map[j][i];
		}
		fprintf(map_fp, "\n");
	}
	fclose(map_fp);

	//output HHS to png
	sprintf(OutputFileName, "%s_HHS.png", InputFileName);
	HHS_gr.SetSize(ImageSizeX2, ImageSizeY2);
	HHS_gr.Title("Holo-Hilbert Spetra");
	HHS_gr.Label('x', "FM Frequency", 0);
	HHS_gr.Label('y', "AM Frequency", 0);
	HHS_gr.SetRange('c', HHS_min, log(HHS_max));
	HHS_gr.SetFunc("", "", "", "lg(c)");
	HHS_gr.Dens(HHS_map);
	HHS_gr.Colorbar(">");
	HHS_gr.Puts(mglPoint(1.35, 1.2), "log-scale\n(10^n)");
	HHS_gr.SetRange('x', 0.0, max_FM_frequency);
	HHS_gr.SetRange('y', 0.0, max_AM_frequency);
	HHS_gr.Axis();
	HHS_gr.WriteFrame(OutputFileName);

	//update HHS parameter
	sprintf(OutputFileName, "%s_HHS.txt", InputFileName);
	if ((fp = fopen(OutputFileName, "a+")) == NULL) {
		printf("Open %s Error!!\n", OutputFileName);
		system("pause");
		exit(1);
	}
	fprintf(fp, "FM_cell=%d\nAM_cell=%d\n", Freq_cell, Holo_Freq_cell);
	fprintf(fp, "FM_per_cell=%f\nAM_per_cell=%f\n", HHS_dFMf, HHS_dAMf);
	fprintf(fp, "HHS_avg=%f\nHHS_stdev=%f\n", HHS_avg, HHS_stdev);
	fprintf(fp, "HHS_max=%f\nHHS_min=%f\n", HHS_max, HHS_min);
	fclose(fp);

	printf("\n\n");

	printf("Generate HHSA 3D result  ");

	//Gnenerate HHSA detail
	HHSA_sum = 0.0;
	HHSA_max = -1000000000000.0;
	HHSA_min = 1000000000000.0;
	int HHSA_length = 0;
	float HHSA_threshold = 1.0;
	for (int k = 0, l = 0; k < Time_cell; ++k) {
		for (int j = 0; j < Freq_cell; ++j) {
			for (int i = 0; i < Holo_Freq_cell; ++i) {
				HHSA_raw[l++] = hhsa_map[k][j][i];
				if (hhsa_map[k][j][i] > 1000000000000.0 || hhsa_map[k][j][i] < -1000000000000.0) continue;
				HHSA_sum += hhsa_map[k][j][i];					//[FM][AM]
				if (hhsa_map[k][j][i] > HHSA_max) HHSA_max = hhsa_map[k][j][i];
				if (hhsa_map[k][j][i] < HHSA_min) HHSA_min = hhsa_map[k][j][i];
				if (hhsa_map[k][j][i] > HHSA_threshold) HHSA_length += 1;
			}
		}
	}

	printf("HHSA_3D_dot_length=%d\n", HHSA_length);

	HHSA_avg = HHSA_sum / (float)(Time_cell * Freq_cell * Holo_Freq_cell);
	HHSA_stdev = STDEV(HHSA_raw, Time_cell * Freq_cell * Holo_Freq_cell);
	HHSA_dt = HSA_dt; 	//=max_time/Time_cell;
	HHSA_dFMf = HHS_dFMf; //=max_FM_frequency/Freq_cell;
	HHSA_dAMf = HHS_dAMf; //=max_AM_frequency/Holo_Freq_cell;

	//Output HHSA parameter
	sprintf(OutputFileName, "%s_HHSA.txt", InputFileName);
	if ((fp = fopen(OutputFileName, "w")) == NULL) {
		printf("Open %s Error!!\n", OutputFileName);
		system("pause");
		exit(1);
	}
	fprintf(fp, "max_time=%f\nmax_AM_frequecny=%f\nmax_FM_frequency=%f\n", max_time, max_AM_frequency, max_FM_frequency);
	fprintf(fp, "Time_cell=%d\nFM_cell=%d\nAM_cell=%d\n", Time_cell, Freq_cell, Holo_Freq_cell);
	fprintf(fp, "Time_per_cell=%f\nFM_per_cell=%f\nAM_per_cell=%f\n", HHSA_dt, HHSA_dFMf, HHSA_dAMf);
	fprintf(fp, "HHSA_avg=%f\nHHSA_stdev=%f\n", HHSA_avg, HHSA_stdev);
	fprintf(fp, "HHSA_avg=%f\n", HHSA_avg);
	fprintf(fp, "HHSA_max=%f\nHHSA_min=%f\n", HHSA_max, HHSA_min);
	fclose(fp);

	/*
	//output HHSA to png
	for (int k = 0; k < Time_cell; ++k) {

		
		//Initial HHSA gr map
		//for(int j=0;j<Freq_cell;++j){
		//	for(int i=0;i<Holo_Freq_cell;++i){
		//		HHSA_map[i*Freq_cell+j]=0.0;
		//	}
		//}
		

		//Graph of a time cell
		for (int j = 0; j < Freq_cell; ++j) {
			for (int i = 0; i < Holo_Freq_cell; ++i) {
				if (hhsa_map[k][j][i] > 1000000000000.0 || hhsa_map[k][j][i] < -1000000000000.0) continue;
				HHSA_map[i * Freq_cell + j] = hhsa_map[k][j][i] + 0.00000001;
			}
		}

		printf("\n\nGenerate HHSA time = %f / %f  ", k * HHSA_dt, max_time);

		//output HHSA to png
		sprintf(OutputFileName, "%s_HHSA_TimeCell%04d.png", InputFileName, k);
		HHSA_gr.Clf();
		HHSA_gr.SetSize(ImageSizeX2, ImageSizeY2);
		HHSA_gr.Title("Holo-Hilbert");
		HHSA_gr.Title("Spetra Analysis");
		sprintf(Text, "T=%f", k * HHSA_dt);
		HHSA_gr.Title(Text);
		HHSA_gr.Label('x', "FM Frequency", 0);
		HHSA_gr.Label('y', "AM Frequency", 0);
		HHSA_gr.SetRange('c', HHSA_min, log(HHSA_max));
		HHSA_gr.SetFunc("", "", "", "lg(c)");
		HHSA_gr.Dens(HHSA_map);
		HHSA_gr.Colorbar(">");
		HHSA_gr.SetRange('x', -1.0, 1.0);
		HHSA_gr.SetRange('y', -1.0, 1.0);
		HHSA_gr.Puts(mglPoint(1.35, 1.2), "log-scale\n(10^n)");
		HHSA_gr.SetRange('x', 0.0, max_FM_frequency);
		HHSA_gr.SetRange('y', 0.0, max_AM_frequency);
		HHSA_gr.Axis();
		HHSA_gr.WriteFrame(OutputFileName);
	}
	*/

	//mglData HHSA_3D_map(Time_cell, Freq_cell, Holo_Freq_cell);
	mglData HHSA_3D_x(HHSA_length), HHSA_3D_y(HHSA_length), HHSA_3D_z(HHSA_length), HHSA_3D_a(HHSA_length);
	
	int index = 0;
	for (int k = 0; k < Time_cell; ++k) {
		for (int j = 0; j < Freq_cell; ++j) {
			for (int i = 0; i < Holo_Freq_cell; ++i) {
				if (hhsa_map[k][j][i] > 1000000000000.0 || hhsa_map[k][j][i] < -1000000000000.0) continue;
				//HHSA_3D_map[ k * Freq_cell* Holo_Freq_cell + j* Holo_Freq_cell + i] = hhsa_map[k][j][i] + 0.00000001;
				if (hhsa_map[k][j][i] > HHSA_threshold) {
					HHSA_3D_x[index] = k;
					HHSA_3D_y[index] = j;
					HHSA_3D_z[index] = i;
					HHSA_3D_a[index++] = hhsa_map[k][j][i];
				}
			}
		}
	}

	

	sprintf(OutputFileName,"%s_HHSA_3D.png",InputFileName);

	//printf("Generate HHSA 3D\n");

	//Not finish, you can write your own 3d plot!!
	HHSA_gr.Clf();
	HHSA_gr.SetSize(ImageSizeX2, ImageSizeY2);
	HHSA_gr.Title("Holo-Hilbert");
	HHSA_gr.Title("Spetra Analysis");
	HHSA_gr.SetRange('c',HHSA_min,log(HHSA_max));
	HHSA_gr.SetFunc("","","","lg(c)");
	//HHSA_gr.Alpha(true);
	//HHSA_gr.SetAlphaDef(0.7);
	//HHSA_gr.SetRange('x',0.0,max_time);
	//HHSA_gr.SetRange('y',0.0,max_FM_frequency);
	//HHSA_gr.SetRange('z',0.0,max_AM_frequency);
	HHSA_gr.SetRange('x',0.0, Time_cell);
	HHSA_gr.SetRange('y',0.0, Freq_cell);
	HHSA_gr.SetRange('z',0.0, Holo_Freq_cell);
	//HHSA_gr.Label('x',"Time",0.0);
	//HHSA_gr.Label('y',"FM Frequency",0.0);
	//HHSA_gr.Label('z',"AM Frequency",0.0);
	HHSA_gr.Rotate(45, 0, 45);
	//HHSA_gr.Box();
	//HHSA_gr.SetOrigin(max_time/2.0,max_FM_frequency/2.0,max_AM_frequency/2.0);
	HHSA_gr.Dots(HHSA_3D_x, HHSA_3D_y, HHSA_3D_z, HHSA_3D_a);
	HHSA_gr.Colorbar(">");
	HHSA_gr.Puts(mglPoint(Time_cell, Freq_cell*1.3, 0),"log-scale\n(10^n)");
	HHSA_gr.Puts(mglPoint(Time_cell*0.5, 0, -Holo_Freq_cell*0.25), "Time");
	HHSA_gr.Puts(mglPoint(0, Freq_cell*0.5, Holo_Freq_cell*1.25), "FMfreq");
	HHSA_gr.Puts(mglPoint(-Time_cell * 0.25, 0, Holo_Freq_cell * 0.5), "AMfreq");
	HHSA_gr.Axis();

	HHSA_gr.WriteFrame(OutputFileName);
	

	printf("\n\n");

	system("pause");
}


