#include <windows.h>
#include <cmath>

#define BIAS 1


float* conv1(BYTE* inputImage, float* weights, int& width, int& height, int maskSize, int maskCount, int imageCount){ //maskSize=3 se�ilir maskenin bir kenar�
																													  //maskCount evri�im katman� maske say�s�d�r

	int rMatrixWidth = width - maskSize + 1; //��k�� Katman�n�n geni�li�i
	int rMatrixHeight = height - maskSize + 1; //��k�� Katman�n�n y�ksekli�i
	float tempSum = 0.0;


	float* masks = new float[maskSize * maskSize * maskCount + maskCount]; //maske atamalar� i�in gerekli de�i�ken
	float* resultImages = new float[maskCount * rMatrixWidth * rMatrixHeight]; //��k�� katman� i�in ayr�lan dizi
	BYTE* image = new BYTE[width * height]; //Giri� g�r�nt�s�



	for (int i = 0; i < width * height; i++) {
		image[i] = inputImage[(imageCount * width * height) + i]; //inputImage b�t�n fer2013 g�r�nt�lerini bar�nd�r�r. Bu for d�ng�s� imageCount bilgisi ile  
	}															  //istenen g�r�nt�y� image dizisine atar.

	for (int i = 0; i < maskCount * rMatrixWidth * rMatrixHeight; i++) { //feature space elemanlar� 0.0 a setlenir 
		resultImages[i] = 0.0;
	}

	for (int i = 0; i < maskSize * maskSize; i++) {			//Weight parametresinde a��rl�klar mevcuttur. Fakat a��rl�klar�n dizili� s�ras� pythonda kaydedildi�i gibidir.
		for (int j = 0; j < maskCount; j++) {				//Bu atama i�lemi maske gezdirme i�lemlerini daha anla��l�r yapabilmek i�in yap�lm��t�r. 
			masks[j * maskSize * maskSize + i] = weights[i * maskCount + j]; //Ayr�ca bknz. (Conv1InputWeights.gif, Conv1OutputWeights.gif)
		}
	}

	for (int i = 0; i < maskCount; i++) {				//Bias de�erleri dizinin en sonundad�r. Bu d�ng� ile bias de�erleri atan�r.
		masks[maskCount * maskSize * maskSize + i] = weights[maskCount * maskSize * maskSize + i];
	}

	for (int m = 0; m < maskCount; m++) { // evri�im i�lemi
		for (int i = 0; i < rMatrixHeight; i++) {
			for (int j = 0; j < rMatrixWidth; j++) {
				for (int k = 0; k < maskSize * maskSize; k++) {
					int mCol = k % maskSize;    //maske i�inde gezebilmek i�in mCol ve mRow de�erleri hesaplan�r.
					int mRow = k / maskSize;
					tempSum +=    //Giri� g�r�nt�s� ile maske eleman�n�n �arp�m� yap�l�r.
						(float)image[(width * i + j) + mRow * width + mCol] * masks[m * (maskSize * maskSize) + k]; //K�sacas� maske gezdirme i�lemidir.
				}
				resultImages[(m * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] = tempSum + (float)BIAS * masks[maskCount * (maskSize * maskSize) + m];  //Maske i�leminden sonra bias de�eri ile �arp�l�p toplan�r.
				tempSum = 0.0;
			}
		}
	}

	delete[] masks;
	delete[] image;

	width = width - maskSize + 1; //padding i�lemi yap�lmad��� i�in evri�im i�leminden sonra feature space'in y�kseklik ve geni�li�i de�i�ir.
	height = height - maskSize + 1; //maske boyutuna ba�l� yeni boyut hesaplan�r.

	return resultImages;
}

float* convHidden(float* feature, float* weights, int& fWidth, int& fHeight, int maskSize, int maskCount, int maskDim) {//maskSize=3 se�ilir maskenin bir kenar�
																													    //maskCount evri�im katman� maske say�s�d�r
																														//maskDim maske derinli�idir. Bu da bir �nceki katman�n maske say�s�d�r

	float* masks = new float[maskSize * maskSize * maskCount * maskDim + maskCount]; //maske atamalar� i�in gerekli de�i�ken
	int rMatrixWidth = fWidth - maskSize + 1;        //��k�� Katman�n�n geni�li�i
	int rMatrixHeight = fHeight - maskSize + 1;      //��k�� Katman�n�n y�ksekli�i
	float* resultImages = new float[maskCount * rMatrixWidth * rMatrixHeight]; //��k�� katman� i�in ayr�lan dizi
	float tempSum = 0.0;
	
	for (int i = 0; i < maskCount * rMatrixWidth * rMatrixHeight; i++) {
		resultImages[i] = 0.0;
	}

	//weights resorting
	int count = 0;
	for (int i = 0; i < maskSize * maskSize; i++) { //Weight parametresinde a��rl�klar mevcuttur. Fakat a��rl�klar�n dizili� s�ras� pythonda kaydedildi�i gibidir.
		for (int j = 0; j < maskDim; j++) {         //Bu atama i�lemi maske gezdirme i�lemlerini daha anla��l�r yapabilmek i�in yap�lm��t�r. 
			for (int k = 0; k < maskCount; k++) {   //Ayr�ca bknz. (ConvHiddenInputWeights.gif, ConvHiddenOutputWeights.gif)
				masks[k * maskSize * maskSize * maskDim + (j * maskSize * maskSize) + i] = weights[count];
				count++;
			}
		}
	}
	for (int i = 0; i < maskCount; i++) // Bias de�erleri dizinin en sonundad�r.Bu d�ng� ile bias de�erleri atan�r.
	{
		masks[maskCount * maskDim * maskSize * maskSize + i] = weights[maskCount * maskDim * maskSize * maskSize + i];
	}

	for (int c = 0; c < maskCount; c++) {
		for (int i = 0; i < rMatrixHeight; i++) {
			for (int j = 0; j < rMatrixWidth; j++) {
				for (int d = 0; d < maskDim; d++) {
					for (int k = 0; k < maskSize * maskSize; k++) {
						int mCol = k % maskSize;    //maske i�inde gezebilmek i�in mCol ve mRow de�erleri hesaplan�r.
						int mRow = k / maskSize;
						tempSum +=     //Giri� g�r�nt�s� ile maske eleman�n�n �arp�m� yap�l�r. K�sacas� maske gezdirme i�lemidir.
							(float)feature[d * fWidth * fHeight + (fWidth * i + j) + mRow * fWidth + mCol] * masks[c * (maskDim * maskSize * maskSize) + d * maskSize * maskSize + k];
					}
				}

				resultImages[(c * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] = tempSum + BIAS * masks[maskCount * maskDim * maskSize * maskSize + c]; //Maske i�leminden sonra bias de�eri ile �arp�l�p toplan�r.
				tempSum = 0.0;
			}
		}
	}
	fWidth = fWidth - maskSize + 1; //padding i�lemi yap�lmad��� i�in evri�im i�leminden sonra feature space'in y�kseklik ve geni�li�i de�i�ir.
	fHeight = fHeight - maskSize + 1;  //maske boyutuna ba�l� yeni boyut hesaplan�r.

	delete[] masks;
	return resultImages;
}


float* batchNormalizationConv(float* feature, float* batchWeights, int width, int height, int featureCount) { //featureCount feature derinlik bilgisidir

	float sDeviation = 0.0;//standart sapma i�in
												 //batch weights dizisinde batch normalizasyon parametreleri s�ral� �ekilde b�t�n gammalar, betalar, 
												 //aritmetik ortalamalar, varyanslar olacak �ekilde s�ralanm��t�r. Bu De�erler e�itim a�amas�nda ��renilir.
	for (int m = 0; m < featureCount; m++) {     //her feature i�in ayr� gamma, beta, aritmetik ortalama ve varyans de�erleri vard�r. Bu d�ng� ile featureler birbirinden ayr�l�r.

		sDeviation = sqrt(batchWeights[(featureCount * 3) + m]); //varyans kullan�larak standart sapma hesaplan�r. (featureCount * 3) dizide varyans elemanlar�na eri�ir

		for (int i = 0; i < width * height; i++) 
		{
			feature[(m * width * height) + i] = (feature[(m * width * height) + i] - batchWeights[featureCount * 2 + m]) / sDeviation; //Her bir de�er aritmetik ortalamadan ��kar�l�p standart sapmaya b�l�n�r. (featureCount * 2) aritmetik ortalama de�erlerine eri�ir
			feature[(m * width * height) + i] = feature[(m * width * height) + i] * batchWeights[m] + batchWeights[featureCount + m];  //Sonu� gamma ile �arp�l�r beta ile toplan�r.
		}

		sDeviation = 0.0;
	}

	return feature;
}

float* batchNormalizationDense(float* input, float* batchWeights, int inputSize) { // inputSize giri� n�ron say�s�d�r


	float sDeviation = 0.0; // standart sapma i�in

	for (int i = 0; i < inputSize; i++) {
		sDeviation = sqrt(batchWeights[(inputSize * 3) + i]);					 //varyans kullan�larak standart sapma hesaplan�r. (featureCount * 3) dizide varyans elemanlar�na eri�ir
		input[i] = (input[i] - batchWeights[(inputSize * 2) + i]) / sDeviation;  //Her bir de�er aritmetik ortalamadan ��kar�l�p standart sapmaya b�l�n�r. (featureCount * 2) aritmetik ortalama de�erlerine eri�ir
		input[i] = input[i] * batchWeights[i] + batchWeights[inputSize + i];     //Sonu� gamma ile �arp�l�r beta ile toplan�r.
	}
	return input;
}

float* reLU(float* feature, int width, int height, int featureCount) { //reLu aktivasyon fonksiyonudur. featureCount feature derinlik bilgisidir
	for (int i = 0; i < width * height * featureCount; i++) {
		if (islessequal(feature[i], 0.0)) {
			feature[i] = 0.0;
		}
	}
	return feature;
}

void maxPooling(float* feature, int& width, int& height, int  featureCount, int pool, int stride) {

	float max = 0.0;
	float temp = 0.0;


	for (int m = 0; m < featureCount; m++) {
		for (int row = 0; row < height / stride; row++) {  //feature space'de stride kadar ilerlenece�i i�in y�kseklik ve geni�lik bu de�ere b�l�n�r
			for (int col = 0; col < width / stride; col++) {
				for (int k = 0; k < pool; k++) {           //pool*pool kadarl�k alandaki de�erlerden en y�ksek olan se�ilmelidir.
					for (int n = 0; n < pool; n++) {
						temp = feature[(m * width * height) + row * width * stride + col * stride + k * width + n]; //Pool �er�evesinin denk geldi�i feature de�erleri temp e atan�r.
						if (isgreater(temp, max)) {
							max = temp; //max de�er hesaplan�r.
						}
					}
				}
				feature[(m * (width / stride) * (height / stride)) + (row * (width / stride)) + col] = max; //bellek alan�ndan tasarruf amac�yla yeni dizi a�mak yerine, max de�erler feature dizisine atan�r.
				max = 0.0;
				temp = 0.0;
			}
		}
	}

	width = width / stride;   //yeni y�kseklik ve geni�lik de�erleri hesaplan�r.
	height = height / stride;

}

float* flatten(float* features, int width, int height, int featureCount) { //features -> ��k�� evri�im katman� , featureCount-> bu ��k�� katman�ndaki feature say�s�
																		   //width height ve featureCount 3 boyut bilgisi olarak g�sterilebilir.

	float* flattenFeatures = new float[width * height * featureCount];     //ge�ici dizi olu�turulur
	int count = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int f = 0; f < featureCount; f++) {
				flattenFeatures[count] = features[f * width * height + i * width + j];  //flatten i�lemi burada yap�l�r. Giri� n�ronlar� a��rl�klara uygun gelecek �ekilde s�ralan�r.
				count++;
			}
		}
	}
	for (int i = 0; i < width * height * featureCount; i++)
		features[i] = flattenFeatures[i];

	return features;
}

float* dense(float* inputLayer, float* weights, int inputLayerSize, int outputLayerSize) { 

	float* outputLayer = new float[outputLayerSize]; //��k�� dizisi olu�turulur
	for (int i = 0; i < outputLayerSize; i++) {
		outputLayer[i] = 0.0;
	}

	for (int i = 0; i < outputLayerSize; i++) {
		for (int j = 0; j < inputLayerSize; j++) {
			outputLayer[i] += inputLayer[j] * weights[j * outputLayerSize + i];  //giri� n�ronlar� ve a��rl�klar �arp�l�p toplan�r. ��k�� katman�na yaz�l�r
		}
		outputLayer[i] += BIAS * weights[inputLayerSize * outputLayerSize + i];  //bias de�eri eklenir
	}

	return outputLayer;
}

float* softmax(float* input, int size) {

	float m = -INFINITY;  //-sonsuz de�er olu�turulur
	for (size_t i = 0; i < size; i++) {
		if (input[i] > m) {
			m = input[i];   //dizideki en b�y�k eleman bulunur.
		}
	}

	float sum = 0.0;
	for (size_t i = 0; i < size; i++) {
		sum += expf(input[i] - m);  //B�t�n elemanlardan en b�y�k de�er ��kar�l�r. Bu durumda dizinin elemanlar� negatif de�erler olur
	}								

	float offset = m + logf(sum);   //offset de�eri hesaplan�r.

	for (size_t i = 0; i < size; i++) {
		input[i] = expf(input[i] - offset);  //offset de�eri dizi elemanlar�ndan ��kar�l�p exponansiyeli hesaplan�r.
	}

	return input;
}