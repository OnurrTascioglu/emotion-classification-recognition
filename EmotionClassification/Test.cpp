#include <windows.h>
#include <cmath>

#define BIAS 1


float* conv1(BYTE* inputImage, float* weights, int& width, int& height, int maskSize, int maskCount, int imageCount){ //maskSize=3 seçilir maskenin bir kenarý
																													  //maskCount evriþim katmaný maske sayýsýdýr

	int rMatrixWidth = width - maskSize + 1; //Çýkýþ Katmanýnýn geniþliði
	int rMatrixHeight = height - maskSize + 1; //Çýkýþ Katmanýnýn yüksekliði
	float tempSum = 0.0;


	float* masks = new float[maskSize * maskSize * maskCount + maskCount]; //maske atamalarý için gerekli deðiþken
	float* resultImages = new float[maskCount * rMatrixWidth * rMatrixHeight]; //Çýkýþ katmaný için ayrýlan dizi
	BYTE* image = new BYTE[width * height]; //Giriþ görüntüsü



	for (int i = 0; i < width * height; i++) {
		image[i] = inputImage[(imageCount * width * height) + i]; //inputImage bütün fer2013 görüntülerini barýndýrýr. Bu for döngüsü imageCount bilgisi ile  
	}															  //istenen görüntüyü image dizisine atar.

	for (int i = 0; i < maskCount * rMatrixWidth * rMatrixHeight; i++) { //feature space elemanlarý 0.0 a setlenir 
		resultImages[i] = 0.0;
	}

	for (int i = 0; i < maskSize * maskSize; i++) {			//Weight parametresinde aðýrlýklar mevcuttur. Fakat aðýrlýklarýn diziliþ sýrasý pythonda kaydedildiði gibidir.
		for (int j = 0; j < maskCount; j++) {				//Bu atama iþlemi maske gezdirme iþlemlerini daha anlaþýlýr yapabilmek için yapýlmýþtýr. 
			masks[j * maskSize * maskSize + i] = weights[i * maskCount + j]; //Ayrýca bknz. (Conv1InputWeights.gif, Conv1OutputWeights.gif)
		}
	}

	for (int i = 0; i < maskCount; i++) {				//Bias deðerleri dizinin en sonundadýr. Bu döngü ile bias deðerleri atanýr.
		masks[maskCount * maskSize * maskSize + i] = weights[maskCount * maskSize * maskSize + i];
	}

	for (int m = 0; m < maskCount; m++) { // evriþim iþlemi
		for (int i = 0; i < rMatrixHeight; i++) {
			for (int j = 0; j < rMatrixWidth; j++) {
				for (int k = 0; k < maskSize * maskSize; k++) {
					int mCol = k % maskSize;    //maske içinde gezebilmek için mCol ve mRow deðerleri hesaplanýr.
					int mRow = k / maskSize;
					tempSum +=    //Giriþ görüntüsü ile maske elemanýnýn çarpýmý yapýlýr.
						(float)image[(width * i + j) + mRow * width + mCol] * masks[m * (maskSize * maskSize) + k]; //Kýsacasý maske gezdirme iþlemidir.
				}
				resultImages[(m * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] = tempSum + (float)BIAS * masks[maskCount * (maskSize * maskSize) + m];  //Maske iþleminden sonra bias deðeri ile çarpýlýp toplanýr.
				tempSum = 0.0;
			}
		}
	}

	delete[] masks;
	delete[] image;

	width = width - maskSize + 1; //padding iþlemi yapýlmadýðý için evriþim iþleminden sonra feature space'in yükseklik ve geniþliði deðiþir.
	height = height - maskSize + 1; //maske boyutuna baðlý yeni boyut hesaplanýr.

	return resultImages;
}

float* convHidden(float* feature, float* weights, int& fWidth, int& fHeight, int maskSize, int maskCount, int maskDim) {//maskSize=3 seçilir maskenin bir kenarý
																													    //maskCount evriþim katmaný maske sayýsýdýr
																														//maskDim maske derinliðidir. Bu da bir önceki katmanýn maske sayýsýdýr

	float* masks = new float[maskSize * maskSize * maskCount * maskDim + maskCount]; //maske atamalarý için gerekli deðiþken
	int rMatrixWidth = fWidth - maskSize + 1;        //Çýkýþ Katmanýnýn geniþliði
	int rMatrixHeight = fHeight - maskSize + 1;      //Çýkýþ Katmanýnýn yüksekliði
	float* resultImages = new float[maskCount * rMatrixWidth * rMatrixHeight]; //Çýkýþ katmaný için ayrýlan dizi
	float tempSum = 0.0;
	
	for (int i = 0; i < maskCount * rMatrixWidth * rMatrixHeight; i++) {
		resultImages[i] = 0.0;
	}

	//weights resorting
	int count = 0;
	for (int i = 0; i < maskSize * maskSize; i++) { //Weight parametresinde aðýrlýklar mevcuttur. Fakat aðýrlýklarýn diziliþ sýrasý pythonda kaydedildiði gibidir.
		for (int j = 0; j < maskDim; j++) {         //Bu atama iþlemi maske gezdirme iþlemlerini daha anlaþýlýr yapabilmek için yapýlmýþtýr. 
			for (int k = 0; k < maskCount; k++) {   //Ayrýca bknz. (ConvHiddenInputWeights.gif, ConvHiddenOutputWeights.gif)
				masks[k * maskSize * maskSize * maskDim + (j * maskSize * maskSize) + i] = weights[count];
				count++;
			}
		}
	}
	for (int i = 0; i < maskCount; i++) // Bias deðerleri dizinin en sonundadýr.Bu döngü ile bias deðerleri atanýr.
	{
		masks[maskCount * maskDim * maskSize * maskSize + i] = weights[maskCount * maskDim * maskSize * maskSize + i];
	}

	for (int c = 0; c < maskCount; c++) {
		for (int i = 0; i < rMatrixHeight; i++) {
			for (int j = 0; j < rMatrixWidth; j++) {
				for (int d = 0; d < maskDim; d++) {
					for (int k = 0; k < maskSize * maskSize; k++) {
						int mCol = k % maskSize;    //maske içinde gezebilmek için mCol ve mRow deðerleri hesaplanýr.
						int mRow = k / maskSize;
						tempSum +=     //Giriþ görüntüsü ile maske elemanýnýn çarpýmý yapýlýr. Kýsacasý maske gezdirme iþlemidir.
							(float)feature[d * fWidth * fHeight + (fWidth * i + j) + mRow * fWidth + mCol] * masks[c * (maskDim * maskSize * maskSize) + d * maskSize * maskSize + k];
					}
				}

				resultImages[(c * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] = tempSum + BIAS * masks[maskCount * maskDim * maskSize * maskSize + c]; //Maske iþleminden sonra bias deðeri ile çarpýlýp toplanýr.
				tempSum = 0.0;
			}
		}
	}
	fWidth = fWidth - maskSize + 1; //padding iþlemi yapýlmadýðý için evriþim iþleminden sonra feature space'in yükseklik ve geniþliði deðiþir.
	fHeight = fHeight - maskSize + 1;  //maske boyutuna baðlý yeni boyut hesaplanýr.

	delete[] masks;
	return resultImages;
}


float* batchNormalizationConv(float* feature, float* batchWeights, int width, int height, int featureCount) { //featureCount feature derinlik bilgisidir

	float sDeviation = 0.0;//standart sapma için
												 //batch weights dizisinde batch normalizasyon parametreleri sýralý þekilde bütün gammalar, betalar, 
												 //aritmetik ortalamalar, varyanslar olacak þekilde sýralanmýþtýr. Bu Deðerler eðitim aþamasýnda öðrenilir.
	for (int m = 0; m < featureCount; m++) {     //her feature için ayrý gamma, beta, aritmetik ortalama ve varyans deðerleri vardýr. Bu döngü ile featureler birbirinden ayrýlýr.

		sDeviation = sqrt(batchWeights[(featureCount * 3) + m]); //varyans kullanýlarak standart sapma hesaplanýr. (featureCount * 3) dizide varyans elemanlarýna eriþir

		for (int i = 0; i < width * height; i++) 
		{
			feature[(m * width * height) + i] = (feature[(m * width * height) + i] - batchWeights[featureCount * 2 + m]) / sDeviation; //Her bir deðer aritmetik ortalamadan çýkarýlýp standart sapmaya bölünür. (featureCount * 2) aritmetik ortalama deðerlerine eriþir
			feature[(m * width * height) + i] = feature[(m * width * height) + i] * batchWeights[m] + batchWeights[featureCount + m];  //Sonuç gamma ile çarpýlýr beta ile toplanýr.
		}

		sDeviation = 0.0;
	}

	return feature;
}

float* batchNormalizationDense(float* input, float* batchWeights, int inputSize) { // inputSize giriþ nöron sayýsýdýr


	float sDeviation = 0.0; // standart sapma için

	for (int i = 0; i < inputSize; i++) {
		sDeviation = sqrt(batchWeights[(inputSize * 3) + i]);					 //varyans kullanýlarak standart sapma hesaplanýr. (featureCount * 3) dizide varyans elemanlarýna eriþir
		input[i] = (input[i] - batchWeights[(inputSize * 2) + i]) / sDeviation;  //Her bir deðer aritmetik ortalamadan çýkarýlýp standart sapmaya bölünür. (featureCount * 2) aritmetik ortalama deðerlerine eriþir
		input[i] = input[i] * batchWeights[i] + batchWeights[inputSize + i];     //Sonuç gamma ile çarpýlýr beta ile toplanýr.
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
		for (int row = 0; row < height / stride; row++) {  //feature space'de stride kadar ilerleneceði için yükseklik ve geniþlik bu deðere bölünür
			for (int col = 0; col < width / stride; col++) {
				for (int k = 0; k < pool; k++) {           //pool*pool kadarlýk alandaki deðerlerden en yüksek olan seçilmelidir.
					for (int n = 0; n < pool; n++) {
						temp = feature[(m * width * height) + row * width * stride + col * stride + k * width + n]; //Pool çerçevesinin denk geldiði feature deðerleri temp e atanýr.
						if (isgreater(temp, max)) {
							max = temp; //max deðer hesaplanýr.
						}
					}
				}
				feature[(m * (width / stride) * (height / stride)) + (row * (width / stride)) + col] = max; //bellek alanýndan tasarruf amacýyla yeni dizi açmak yerine, max deðerler feature dizisine atanýr.
				max = 0.0;
				temp = 0.0;
			}
		}
	}

	width = width / stride;   //yeni yükseklik ve geniþlik deðerleri hesaplanýr.
	height = height / stride;

}

float* flatten(float* features, int width, int height, int featureCount) { //features -> çýkýþ evriþim katmaný , featureCount-> bu çýkýþ katmanýndaki feature sayýsý
																		   //width height ve featureCount 3 boyut bilgisi olarak gösterilebilir.

	float* flattenFeatures = new float[width * height * featureCount];     //geçici dizi oluþturulur
	int count = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int f = 0; f < featureCount; f++) {
				flattenFeatures[count] = features[f * width * height + i * width + j];  //flatten iþlemi burada yapýlýr. Giriþ nöronlarý aðýrlýklara uygun gelecek þekilde sýralanýr.
				count++;
			}
		}
	}
	for (int i = 0; i < width * height * featureCount; i++)
		features[i] = flattenFeatures[i];

	return features;
}

float* dense(float* inputLayer, float* weights, int inputLayerSize, int outputLayerSize) { 

	float* outputLayer = new float[outputLayerSize]; //çýkýþ dizisi oluþturulur
	for (int i = 0; i < outputLayerSize; i++) {
		outputLayer[i] = 0.0;
	}

	for (int i = 0; i < outputLayerSize; i++) {
		for (int j = 0; j < inputLayerSize; j++) {
			outputLayer[i] += inputLayer[j] * weights[j * outputLayerSize + i];  //giriþ nöronlarý ve aðýrlýklar çarpýlýp toplanýr. Çýkýþ katmanýna yazýlýr
		}
		outputLayer[i] += BIAS * weights[inputLayerSize * outputLayerSize + i];  //bias deðeri eklenir
	}

	return outputLayer;
}

float* softmax(float* input, int size) {

	float m = -INFINITY;  //-sonsuz deðer oluþturulur
	for (size_t i = 0; i < size; i++) {
		if (input[i] > m) {
			m = input[i];   //dizideki en büyük eleman bulunur.
		}
	}

	float sum = 0.0;
	for (size_t i = 0; i < size; i++) {
		sum += expf(input[i] - m);  //Bütün elemanlardan en büyük deðer çýkarýlýr. Bu durumda dizinin elemanlarý negatif deðerler olur
	}								

	float offset = m + logf(sum);   //offset deðeri hesaplanýr.

	for (size_t i = 0; i < size; i++) {
		input[i] = expf(input[i] - offset);  //offset deðeri dizi elemanlarýndan çýkarýlýp exponansiyeli hesaplanýr.
	}

	return input;
}