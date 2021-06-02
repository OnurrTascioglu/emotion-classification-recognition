#pragma once
#include <windows.h>
#include "image.h"
#include "Test.h"
#include "atlstr.h"
#include "CpuGpu.h"
#include "CpuGpuMem.h"
#include "Test.cuh"
#include "cuda_runtime.h"
#include <iostream>
#include <fstream>
#include <msclr\marshal.h>
#include <vector>
#include <istream>
#include <string>
#include <sstream>
#include <math.h>
#include <time.h>

#define IMAGE_WIDTH 48
#define IMAGE_HEIGHT 48
#define TOTAL_IMAGE 35888
#define MASK_SIZE 3

#define MASK_COUNT_FIRST_LAYER 64
#define MASK_COUNT_HIDDEN_LAYER_1
#define MASK_COUNT_HIDDEN_LAYER_2 
#define MASK_COUNT_HIDDEN_LAYER_3 
#define MASK_COUNT_OUTPUT_LAYER 32 //output from conv layer = dense input layer

#define DENSE_HIDDEN_LAYER_1 128
#define DENSE_OUTPUT_LAYER 7

#define MAX_POOL 2
#define MAX_POOL_STRIDE 2

#define WEIGHT_PATH "D:\\Ders\\bitirme\\agirlikler2\\"
#define FEATURE_RESULT_PATH "D:\\Ders\\bitirme\\features\\"

namespace EmotionClassification {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::IO;
	using namespace msclr::interop;
	using namespace System::Runtime::InteropServices;
	using namespace std;

	/// <summary>
	/// Summary for MyForm
	/// </summary>
	public ref class MyForm : public System::Windows::Forms::Form
	{
		String^ readFile;
		BYTE* bmpColoredImage;

		BYTE* ferImages;
		int lineCount = 0;
		BYTE* emotionLabel;
		BYTE* raw_intensity;


		//statik
		float* convInputLayerWeights;
		float* convHiddenLayerWeights_1;
		float* convOutputLayerWeights;

		float* denseHiddenLayerWeights_1;
		float* denseOutputLayerWeights;
		float* batchNormWeight;
		float* batchNormWeight_1;
		float* batchNormWeight_2;

		int ferTextBoxInput = 0;


	private: System::Windows::Forms::ToolStripMenuItem^ fer2013DSToolStripMenuItem;
	private: System::Windows::Forms::TextBox^ textBox1;
	private: System::Windows::Forms::ToolStripMenuItem^ testToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ runToolStripMenuItem;




	private: System::Windows::Forms::DataVisualization::Charting::Chart^ chart1;
	private: System::Windows::Forms::ToolStripMenuItem^ cudaRunToolStripMenuItem;
	private: System::Windows::Forms::Button^ button2;
	private: System::Windows::Forms::Button^ button1;


	public:

		MyForm(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~MyForm()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::MenuStrip^ menuStrip1;
	protected:
	private: System::Windows::Forms::ToolStripMenuItem^ dosyaToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ openToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ weightsToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ pictureToolStripMenuItem;
	private: System::Windows::Forms::OpenFileDialog^ openFileDialog1;
	private: System::Windows::Forms::RichTextBox^ richTextBox1;
	private: System::Windows::Forms::PictureBox^ pictureBox1;

	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container^ components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			System::Windows::Forms::DataVisualization::Charting::ChartArea^ chartArea1 = (gcnew System::Windows::Forms::DataVisualization::Charting::ChartArea());
			System::Windows::Forms::DataVisualization::Charting::Legend^ legend1 = (gcnew System::Windows::Forms::DataVisualization::Charting::Legend());
			System::Windows::Forms::DataVisualization::Charting::Series^ series1 = (gcnew System::Windows::Forms::DataVisualization::Charting::Series());
			this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
			this->dosyaToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->openToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->weightsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->pictureToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->fer2013DSToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->testToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->runToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->cudaRunToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->richTextBox1 = (gcnew System::Windows::Forms::RichTextBox());
			this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->chart1 = (gcnew System::Windows::Forms::DataVisualization::Charting::Chart());
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->menuStrip1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chart1))->BeginInit();
			this->SuspendLayout();
			// 
			// menuStrip1
			// 
			this->menuStrip1->ImageScalingSize = System::Drawing::Size(20, 20);
			this->menuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {
				this->dosyaToolStripMenuItem,
					this->testToolStripMenuItem
			});
			this->menuStrip1->Location = System::Drawing::Point(0, 0);
			this->menuStrip1->Name = L"menuStrip1";
			this->menuStrip1->Size = System::Drawing::Size(1451, 28);
			this->menuStrip1->TabIndex = 0;
			this->menuStrip1->Text = L"menuStrip1";
			// 
			// dosyaToolStripMenuItem
			// 
			this->dosyaToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) { this->openToolStripMenuItem });
			this->dosyaToolStripMenuItem->Name = L"dosyaToolStripMenuItem";
			this->dosyaToolStripMenuItem->Size = System::Drawing::Size(46, 24);
			this->dosyaToolStripMenuItem->Text = L"File";
			// 
			// openToolStripMenuItem
			// 
			this->openToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {
				this->weightsToolStripMenuItem,
					this->pictureToolStripMenuItem, this->fer2013DSToolStripMenuItem
			});
			this->openToolStripMenuItem->Name = L"openToolStripMenuItem";
			this->openToolStripMenuItem->Size = System::Drawing::Size(128, 26);
			this->openToolStripMenuItem->Text = L"Open";
			// 
			// weightsToolStripMenuItem
			// 
			this->weightsToolStripMenuItem->Name = L"weightsToolStripMenuItem";
			this->weightsToolStripMenuItem->Size = System::Drawing::Size(167, 26);
			this->weightsToolStripMenuItem->Text = L"Weights";
			this->weightsToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::weightsToolStripMenuItem_Click);
			// 
			// pictureToolStripMenuItem
			// 
			this->pictureToolStripMenuItem->Name = L"pictureToolStripMenuItem";
			this->pictureToolStripMenuItem->Size = System::Drawing::Size(167, 26);
			this->pictureToolStripMenuItem->Text = L"Picture";
			this->pictureToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::pictureToolStripMenuItem_Click);
			// 
			// fer2013DSToolStripMenuItem
			// 
			this->fer2013DSToolStripMenuItem->Name = L"fer2013DSToolStripMenuItem";
			this->fer2013DSToolStripMenuItem->Size = System::Drawing::Size(167, 26);
			this->fer2013DSToolStripMenuItem->Text = L"Fer2013 DS";
			this->fer2013DSToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::fer2013DSToolStripMenuItem_Click);
			// 
			// testToolStripMenuItem
			// 
			this->testToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {
				this->runToolStripMenuItem,
					this->cudaRunToolStripMenuItem
			});
			this->testToolStripMenuItem->Name = L"testToolStripMenuItem";
			this->testToolStripMenuItem->Size = System::Drawing::Size(49, 24);
			this->testToolStripMenuItem->Text = L"Test";
			// 
			// runToolStripMenuItem
			// 
			this->runToolStripMenuItem->Name = L"runToolStripMenuItem";
			this->runToolStripMenuItem->Size = System::Drawing::Size(151, 26);
			this->runToolStripMenuItem->Text = L"Run";
			this->runToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::runToolStripMenuItem_Click);
			// 
			// cudaRunToolStripMenuItem
			// 
			this->cudaRunToolStripMenuItem->Name = L"cudaRunToolStripMenuItem";
			this->cudaRunToolStripMenuItem->Size = System::Drawing::Size(151, 26);
			this->cudaRunToolStripMenuItem->Text = L"CudaRun";
			this->cudaRunToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::cudaRunToolStripMenuItem_Click);
			// 
			// openFileDialog1
			// 
			this->openFileDialog1->FileName = L"openFileDialog1";
			// 
			// richTextBox1
			// 
			this->richTextBox1->Location = System::Drawing::Point(943, 385);
			this->richTextBox1->Name = L"richTextBox1";
			this->richTextBox1->Size = System::Drawing::Size(496, 176);
			this->richTextBox1->TabIndex = 1;
			this->richTextBox1->Text = L"";
			// 
			// pictureBox1
			// 
			this->pictureBox1->Location = System::Drawing::Point(13, 32);
			this->pictureBox1->Name = L"pictureBox1";
			this->pictureBox1->Size = System::Drawing::Size(353, 347);
			this->pictureBox1->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pictureBox1->TabIndex = 2;
			this->pictureBox1->TabStop = false;
			// 
			// textBox1
			// 
			this->textBox1->Location = System::Drawing::Point(1339, 12);
			this->textBox1->Name = L"textBox1";
			this->textBox1->RightToLeft = System::Windows::Forms::RightToLeft::No;
			this->textBox1->Size = System::Drawing::Size(100, 22);
			this->textBox1->TabIndex = 3;
			// 
			// button1
			// 
			this->button1->Enabled = false;
			this->button1->Location = System::Drawing::Point(1339, 40);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(100, 28);
			this->button1->TabIndex = 4;
			this->button1->Text = L"ImageCpu";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &MyForm::button1_Click);
			// 
			// chart1
			// 
			chartArea1->Name = L"ChartArea1";
			this->chart1->ChartAreas->Add(chartArea1);
			legend1->Name = L"Legend1";
			this->chart1->Legends->Add(legend1);
			this->chart1->Location = System::Drawing::Point(453, 31);
			this->chart1->Name = L"chart1";
			series1->ChartArea = L"ChartArea1";
			series1->Legend = L"Legend1";
			series1->Name = L"Duygular";
			this->chart1->Series->Add(series1);
			this->chart1->Size = System::Drawing::Size(821, 348);
			this->chart1->TabIndex = 9;
			this->chart1->Text = L"chart1";
			// 
			// button2
			// 
			this->button2->Enabled = false;
			this->button2->Location = System::Drawing::Point(1339, 75);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(100, 27);
			this->button2->TabIndex = 10;
			this->button2->Text = L"ImageGpu";
			this->button2->UseVisualStyleBackColor = true;
			this->button2->Click += gcnew System::EventHandler(this, &MyForm::button2_Click);
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(8, 16);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1451, 592);
			this->Controls->Add(this->button2);
			this->Controls->Add(this->chart1);
			this->Controls->Add(this->button1);
			this->Controls->Add(this->textBox1);
			this->Controls->Add(this->pictureBox1);
			this->Controls->Add(this->richTextBox1);
			this->Controls->Add(this->menuStrip1);
			this->MainMenuStrip = this->menuStrip1;
			this->Name = L"MyForm";
			this->Text = L"MyForm";
			this->FormClosing += gcnew System::Windows::Forms::FormClosingEventHandler(this, &MyForm::MyForm_FormClosing);
			this->Load += gcnew System::EventHandler(this, &MyForm::MyForm_Load);
			this->menuStrip1->ResumeLayout(false);
			this->menuStrip1->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chart1))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion

	private: System::Void weightsToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		//Stream^ mystream;
		//OpenFileDialog^ openFileDialog1 = gcnew OpenFileDialog;

		//openFileDialog1->InitialDirectory = "";
		//openFileDialog1->Filter = "txt files (*.txt)|*.txt|All files (*.*)|*.*";
		//openFileDialog1->FilterIndex = 2;
		//openFileDialog1->RestoreDirectory = true;

		//if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK)
		//{
		//	if ((mystream = openFileDialog1->OpenFile()) != nullptr)
		//	{
		//		// Insert code to read the stream here.
		//		String^ strfilename = openFileDialog1->InitialDirectory + openFileDialog1->FileName;
		//		readFile = File::ReadAllText(strfilename);

		//		richTextBox1->Text = readFile;

		//		mystream->Close();
		//	}
		//}
		int error = 0;
		int success = 0;

		MessageBox::Show("Loading Weights");

		//------ File Path
		IntPtr ip = Marshal::StringToHGlobalAnsi(WEIGHT_PATH);
		const char* inputStr = static_cast<const char*>(ip.ToPointer());
		std::string input(inputStr);
		//------ File Path

		//----- input layer cnn
		string filePath = input + "conv2d.csv";
		convInputLayerWeights = new float[MASK_COUNT_FIRST_LAYER * MASK_SIZE * MASK_SIZE + MASK_COUNT_FIRST_LAYER];
		if (readWeightFromFile(convInputLayerWeights, filePath)) error++;
		else success++;

		int sizeW = IMAGE_WIDTH - MASK_SIZE + 1;
		int sizeH = IMAGE_HEIGHT - MASK_SIZE + 1;
		sizeW = sizeW / MAX_POOL_STRIDE;
		sizeH = sizeH / MAX_POOL_STRIDE;

		//----- 2. layer cnn
		filePath = input + "conv2d_1.csv";
		convOutputLayerWeights = new float[MASK_COUNT_OUTPUT_LAYER * MASK_COUNT_FIRST_LAYER * MASK_SIZE * MASK_SIZE + MASK_COUNT_OUTPUT_LAYER];
		if (readWeightFromFile(convOutputLayerWeights, filePath)) error++;
		else success++; 


		sizeW = sizeW - MASK_SIZE + 1;
		sizeH = sizeH - MASK_SIZE + 1;
		sizeW = sizeW / MAX_POOL_STRIDE;
		sizeH = sizeH / MAX_POOL_STRIDE;


		//----- FullyConnected Layer 1
		int sizeTemp = MASK_COUNT_OUTPUT_LAYER * sizeW * sizeH;
		float* denseResult = new float[DENSE_HIDDEN_LAYER_1];
		filePath = input + "dense.csv";
		denseHiddenLayerWeights_1 = new float[sizeTemp * DENSE_HIDDEN_LAYER_1 + DENSE_HIDDEN_LAYER_1];
		if (readWeightFromFile(denseHiddenLayerWeights_1, filePath)) error++;
		else success++;

		//----- FullyConnected Layer 2
		filePath = input + "dense_1.csv";
		denseOutputLayerWeights = new float[DENSE_OUTPUT_LAYER * DENSE_HIDDEN_LAYER_1 + DENSE_OUTPUT_LAYER];
		if (readWeightFromFile(denseOutputLayerWeights, filePath)) error++;
		else success++;

		//----- 1. batch norm
		filePath = input + "batch_normalization.csv";
		batchNormWeight = new float[MASK_COUNT_FIRST_LAYER * 4];
		if (readWeightFromFile(batchNormWeight, filePath)) error++;
		else success++;

		//----- 2. batch norm
		filePath = input + "batch_normalization_1.csv";
		batchNormWeight_1 = new float[MASK_COUNT_OUTPUT_LAYER * 4];
		if (readWeightFromFile(batchNormWeight_1, filePath)) error++;
		else success++;

		//----- 3. batch norm
		filePath = input + "batch_normalization_2.csv";
		batchNormWeight_2 = new float[DENSE_HIDDEN_LAYER_1 * 4];
		if (readWeightFromFile(batchNormWeight_2, filePath)) error++;
		else success++;


		MessageBox::Show(success + " File Successfully Loaded " + "\n" + error + " File Unsuccessfully Loaded ");
		button1->Enabled = true;
		button2->Enabled = true;
	}
	private: System::Void pictureToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		LPCTSTR input;
		CString str;
		int Width, Height;
		float resizeX = 0.0, resizeY = 0.0;
		long Size;
		int integer = 0;
		float fraction = 0.0, tempFrac = 0.0;
		float total = 0.0;
		int count = 2;

		if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			str = openFileDialog1->FileName;
			input = (LPCTSTR)str;
			float mean = 0.0;

			//BMP Image Reading
			bmpColoredImage = LoadBMP(&Width, &Height, &Size, input);
			raw_intensity = ConvertBMPToIntensity(bmpColoredImage, Width, Height); // BMP Gray picture

			BYTE* buffer = new BYTE[IMAGE_WIDTH * IMAGE_HEIGHT];

			resizeX = round((float)Width / IMAGE_WIDTH);
			resizeY = round((float)Height / IMAGE_HEIGHT);

			if (resizeX < 1 || resizeY < 1) {
				MessageBox::Show("Please choose greater than 48x48 image.");
			}
			else {
				//integer = Width / IMAGE_WIDTH;
				//fraction = ((float)Width / IMAGE_WIDTH) - (float)integer; // fraction for width

				//tempFrac = fraction;

				//while (true) {
				//	if (islessequal(tempFrac, 0.1)) {
				//		break;
				//	}
				//	if (isgreaterequal(tempFrac, 0.9)) {
				//		integer = integer + 1;
				//		break;
				//	}

				//	total = fraction * (float)count;
				//	integer = total;
				//	tempFrac = total - (float)integer;
				//	count++;
				//}


				//for (int row = 0; row < IMAGE_HEIGHT; row++) {
				//	for (int col = 0; col < IMAGE_WIDTH; col++) {
				//		for (int y = 0; y < resizeY; y++) {
				//			for (int x = 0; x < resizeX; x++) {
				//				mean += raw_intensity[Width * row * (int)resizeY + Width * y + col * (int)resizeX + x];
				//				if (col % count == 0) {

				//				}
				//			}
				//		}
				//		mean = mean / (resizeY * resizeX);
				//		buffer[IMAGE_WIDTH * row + col] = round(mean);
				//		mean = 0.0;
				//	}
				//}


				Bitmap^ surface = gcnew Bitmap(Width, Height);
				pictureBox1->Image = surface;

				Color c;

				for (int row = 0; row < Height; row++)
					for (int col = 0; col < Width; col++) {
						c = Color::FromArgb(raw_intensity[row * Width + col], raw_intensity[row * Width + col], raw_intensity[row * Width + col]);
						surface->SetPixel(col, row, c);
					}
			}
		}
	}
	private: System::Void fer2013DSToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {

		Stream^ mystream;
		OpenFileDialog^ openFileDialog1 = gcnew OpenFileDialog;

		openFileDialog1->InitialDirectory = "";
		openFileDialog1->Filter = "txt files (*.txt)|*.txt|All files (*.*)|*.*";
		openFileDialog1->FilterIndex = 2;
		openFileDialog1->RestoreDirectory = true;

		if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK)
		{
			ferImages = new BYTE[IMAGE_WIDTH * IMAGE_HEIGHT * TOTAL_IMAGE];
			emotionLabel = new BYTE[TOTAL_IMAGE];

			if ((mystream = openFileDialog1->OpenFile()) != nullptr)
			{

				// File pointer
				fstream fin;

				String^ fileName = openFileDialog1->FileName;

				// Open an existing file
				IntPtr ip = Marshal::StringToHGlobalAnsi(fileName);
				const char* inputStr = static_cast<const char*>(ip.ToPointer());
				std::string input(inputStr);

				fin.open(inputStr, ios::in);

				string line, word;

				int asd = 0;
				int k = 0, count = 0;
				bool lineBool = 0;
				int imageIndex = 0;
				int ferIndex = 0;


				while (!fin.eof()) {

					std::getline(fin, line);

					k = 0;
					count = 0;

					if (lineBool == 1) {
						for (int i = 0; i < line.length(); i++) {
							if (line[i].Equals(',')) {
								if (count == 0) {
									word = line.substr(k, i - k);
									emotionLabel[lineCount] = stoi(word);
									richTextBox1->Text += emotionLabel[lineCount] + " \n";
								}//take emotions from csv file

								count++;

								if (count == 1) {
									k = i + 1;
								}

								if (count == 2) {
									word = line.substr(k, i - k);
									ferImages[(imageIndex * 48 * 48) + ferIndex] = stoi(word);
									ferIndex++;
								}
							}
							if (count == 1) {
								if (line[i].Equals(' ')) {
									word = line.substr(k, i - k);
									ferImages[(imageIndex * 48 * 48) + ferIndex] = stoi(word);
									ferIndex++;
									k = i;
								}
							} // take image from csv file
						}
						imageIndex++;
						lineCount++;
					}
					else {
						lineBool = 1;
					}
					ferIndex = 0;
				}

				//readFile = File::ReadAllText(strfilename);
				//richTextBox1->Text = readFile;

				mystream->Close();
			}
		}
	}
	private: System::Void button1_Click(System::Object^ sender, System::EventArgs^ e) {
		Int32 myInt = 0;

		if (System::Text::RegularExpressions::Regex::IsMatch(textBox1->Text,
			"^[1-9][0-9]{1,5}$"))
		{
			myInt = System::Convert::ToInt32(textBox1->Text);


			if (myInt > lineCount - 1) {
				MessageBox::Show("Number can't be higher than " + (lineCount - 1));
			}
			else if (myInt < 0) {
				MessageBox::Show("Please enter positive number");
			}
			else {
				ferTextBoxInput = myInt - 1;

				Bitmap^ surface = gcnew Bitmap(IMAGE_WIDTH, IMAGE_HEIGHT);
				pictureBox1->Image = surface;

				int point = IMAGE_HEIGHT * IMAGE_WIDTH * (myInt - 1);

				Color c;
				for (int row = 0; row < IMAGE_HEIGHT; row++)
				{
					for (int column = 0; column < IMAGE_WIDTH; column++)
					{
						c = Color::FromArgb(ferImages[point + row * IMAGE_WIDTH + column], ferImages[point + row * IMAGE_WIDTH + column], ferImages[point + row * IMAGE_WIDTH + column]);
						surface->SetPixel(column, row, c);
					}
				}
				runToolStripMenuItem_Click(sender, e);
			}
		}
		else {
			MessageBox::Show("Not a number");
		}

	}
	void saveFeatureBMP(float* fResult, int sizeW,int sizeH,int featureCount,char convIndex) {
		LPCTSTR input;
		CString str;
		BYTE* buffer = new BYTE[sizeW * sizeH * featureCount];
		long* a = new long;
		BYTE* buffer2;
		int max = 0;
		int min = 0;
		float ratio = 0.0;
		float* tempResult = new float[featureCount* sizeH* sizeW];

		for (int m = 0; m < featureCount; m++) {
			for (int i = 0; i < sizeW * sizeH; i++) {
				if ((int)fResult[(m * sizeW * sizeH) + i] > max) {
					max = fResult[(m * sizeW * sizeH) + i];
				}
				if ((int)fResult[(m * sizeW * sizeH) + i] < min) {
					min = fResult[(m * sizeW * sizeH) + i];
				}
			}
			for (int i = 0; i < sizeW * sizeH; i++) {
				tempResult[(m * sizeW * sizeH) + i] = fResult[(m * sizeW * sizeH) + i] - (min);
			}
			ratio = (float)(max - min) / 240;

			for (int i = 0; i < sizeW * sizeH; i++) {
				buffer[(m * sizeW * sizeH) + i] = (int)(tempResult[(m * sizeW * sizeH) + i] / ratio);
			}
		}
		str = FEATURE_RESULT_PATH + "conv" + convIndex + "feature.bmp";
		input = (LPCTSTR)str;
		buffer2 = ConvertIntensityToBMP(buffer, sizeW, sizeH*featureCount, a);
		SaveBMP(buffer2, sizeW, sizeH * featureCount, *a, input);
	}
	private: System::Void runToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {

		int size = (IMAGE_WIDTH - MASK_SIZE + 1) * (IMAGE_HEIGHT - MASK_SIZE + 1) * MASK_COUNT_FIRST_LAYER;
		int sizeW = IMAGE_WIDTH;
		int sizeH = IMAGE_HEIGHT;

		float* fResult = new float[size];

		clock_t tStart = clock();
		//1. cnn layer
		fResult = conv1(ferImages, convInputLayerWeights, sizeW, sizeH, MASK_SIZE, MASK_COUNT_FIRST_LAYER, ferTextBoxInput);
		batchNormalizationConv(fResult, batchNormWeight, sizeW, sizeH, MASK_COUNT_FIRST_LAYER);
		reLU(fResult, sizeW, sizeH, MASK_COUNT_FIRST_LAYER);
		maxPooling(fResult, sizeW, sizeH, MASK_COUNT_FIRST_LAYER, MAX_POOL, MAX_POOL_STRIDE);

		saveFeatureBMP(fResult, sizeW, sizeH, MASK_COUNT_FIRST_LAYER, 1);

		//2.cnn layer
		float* fHiddenResult = new float[(sizeW - MASK_SIZE + 1) * (sizeH - MASK_SIZE + 1) * MASK_COUNT_OUTPUT_LAYER];
		fHiddenResult = convHidden(fResult, convOutputLayerWeights, sizeW, sizeH, MASK_SIZE, MASK_COUNT_OUTPUT_LAYER, MASK_COUNT_FIRST_LAYER);
		size = sizeW * sizeH * MASK_COUNT_OUTPUT_LAYER;

		saveFeatureBMP(fHiddenResult, sizeW, sizeH, MASK_COUNT_OUTPUT_LAYER, 2);

		batchNormalizationConv(fHiddenResult, batchNormWeight_1, sizeW, sizeH, MASK_COUNT_OUTPUT_LAYER);
		reLU(fHiddenResult, sizeW, sizeH, MASK_COUNT_OUTPUT_LAYER);
		maxPooling(fHiddenResult, sizeW, sizeH, MASK_COUNT_OUTPUT_LAYER, MAX_POOL, MAX_POOL_STRIDE);


		//----- FullyConnected Layer 1
		flatten(fHiddenResult, sizeW, sizeH, MASK_COUNT_OUTPUT_LAYER);
		int sizeTemp = MASK_COUNT_OUTPUT_LAYER * sizeW * sizeH;
		float* denseResult = new float[DENSE_HIDDEN_LAYER_1];
		denseResult = dense(fHiddenResult, denseHiddenLayerWeights_1, sizeTemp, DENSE_HIDDEN_LAYER_1);
		batchNormalizationDense(denseResult, batchNormWeight_2, DENSE_HIDDEN_LAYER_1);
		reLU(denseResult, DENSE_HIDDEN_LAYER_1, 1, 1);



		//----- FullyConnected Layer 2
		denseResult = dense(denseResult, denseOutputLayerWeights, DENSE_HIDDEN_LAYER_1, DENSE_OUTPUT_LAYER);

		
		

		softmax(denseResult, DENSE_OUTPUT_LAYER);
		double cpuClock = (double)(clock() - tStart) / CLOCKS_PER_SEC;
		richTextBox1->Text += "cpu clock time: " + cpuClock + " \n";

		string emot[] = { "Kýzgýn" ,"Nefret" ,"Korku" ,"Mutlu" ,"Üzgün" ,"Þaþkýn" ,"Doðal" };

		chart1->Series["Duygular"]->Points->Clear();
		for (int i = 0; i < 7; i++) {
			String^ str = gcnew String(emot[i].c_str());
			chart1->Series["Duygular"]->Points->AddXY(str, denseResult[i]);
		}

		delete[] fResult;
		delete[] fHiddenResult;

	}
	private: System::Void button2_Click(System::Object^ sender, System::EventArgs^ e) {
		Int32 myInt = 0;

		if (System::Text::RegularExpressions::Regex::IsMatch(textBox1->Text,
			"^[1-9][0-9]{1,5}$"))
		{
			myInt = System::Convert::ToInt32(textBox1->Text);


			if (myInt > lineCount - 1) {
				MessageBox::Show("Number can't be higher than " + (lineCount - 1));
			}
			else if (myInt < 0) {
				MessageBox::Show("Please enter positive number");
			}
			else {
				ferTextBoxInput = myInt - 1;

				Bitmap^ surface = gcnew Bitmap(IMAGE_WIDTH, IMAGE_HEIGHT);
				pictureBox1->Image = surface;

				int point = IMAGE_HEIGHT * IMAGE_WIDTH * (myInt - 1);

				Color c;
				for (int row = 0; row < IMAGE_HEIGHT; row++)
				{
					for (int column = 0; column < IMAGE_WIDTH; column++)
					{
						c = Color::FromArgb(ferImages[point + row * IMAGE_WIDTH + column], ferImages[point + row * IMAGE_WIDTH + column], ferImages[point + row * IMAGE_WIDTH + column]);
						surface->SetPixel(column, row, c);
					}
				}
				cudaRunToolStripMenuItem_Click(sender, e);
			}
		}
		else {
			MessageBox::Show("Not a number");
		}
	}

	void setValuesForGpuConv1(CpuGpuMem* cg) {
		cg->imageHeightSize = IMAGE_HEIGHT;
		cg->imageWidthSize = IMAGE_WIDTH;
		cg->featureWidthSize = IMAGE_WIDTH - MASK_SIZE + 1; // no Padding
		cg->featureHeightSize = IMAGE_HEIGHT - MASK_SIZE + 1; // no Padding
		cg->maskWHSize = MASK_SIZE;
		cg->maskCount = MASK_COUNT_FIRST_LAYER;
		cg->maskDim = 1;
		cg->pool = MAX_POOL;
		cg->stride = MAX_POOL_STRIDE;
		cg->batchWeightSize = cg->maskCount;

		cpuGpuAlloc(cg, imageEnum, sizeof(int));
		cpuGpuAlloc(cg, featureEnum, sizeof(float));
		cpuGpuAlloc(cg, maskEnum, sizeof(float));
		cpuGpuAlloc(cg, batchEnum, sizeof(float));//batch for conv1

		for (int i = 0; i < cg->maskCount * 4; i++)
			cg->cpuBatchPtr[i] = batchNormWeight[i];

		int* cpu_int32 = (int*)cg->cpuImagePtr;
		for (int i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH; i++) {
			cpu_int32[i] = ferImages[(ferTextBoxInput * IMAGE_WIDTH * IMAGE_HEIGHT) + i]; //
		}

		for (int i = 0; i < cg->featureWidthSize * cg->featureHeightSize * cg->maskCount; i++) {
			cg->cpuFeaturePtr[i] = 0.0;
		}

		for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++) {
			for (int j = 0; j < cg->maskCount; j++) {
				cg->cpuMaskPtr[j * MASK_SIZE * MASK_SIZE + i] = convInputLayerWeights[i * cg->maskCount + j];
			}
		}

		for (int i = 0; i < cg->maskCount; i++) {
			cg->cpuMaskPtr[cg->maskCount * MASK_SIZE * MASK_SIZE + i] = convInputLayerWeights[cg->maskCount * MASK_SIZE * MASK_SIZE + i];
		}




	}

	void setValuesForGpuConv2(CpuGpuMem* cg) {
		cg->maskWHSize = MASK_SIZE;
		cg->maskCount = MASK_COUNT_OUTPUT_LAYER;
		cg->maskDim = MASK_COUNT_FIRST_LAYER;
		cg->batchWeightSize = cg->maskCount;

		cpuGpuFree(cg,imageEnum);
		cpuGpuFree(cg,maskEnum);
		cpuGpuFree(cg,batchEnum);

		cpuGpuAlloc(cg, maskEnum, sizeof(float)); //  mask allocation for 2. conv layer
		cpuGpuAlloc(cg, batchEnum, sizeof(float));


		for (int i = 0; i < cg->maskCount * 4; i++)
			cg->cpuBatchPtr[i] = batchNormWeight_1[i];


		//weights resorting
		int count = 0;
		for (int i = 0; i < cg->maskWHSize * cg->maskWHSize; i++) {
			for (int j = 0; j < cg->maskDim; j++) {
				for (int k = 0; k < cg->maskCount; k++) {
					cg->cpuMaskPtr[k * cg->maskWHSize * cg->maskWHSize * cg->maskDim + (j * cg->maskWHSize * cg->maskWHSize) + i] = convOutputLayerWeights[count];
					count++;
				}
			}
		}

		for (int i = 0; i < cg->maskCount; i++)
		{
			cg->cpuMaskPtr[cg->maskCount * cg->maskDim * cg->maskWHSize * cg->maskWHSize + i] =
				convOutputLayerWeights[cg->maskCount * cg->maskDim * cg->maskWHSize * cg->maskWHSize + i];
		}
		cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuMaskPtr, cg->cpuMaskPtr, cg->maskAllocSize);
		cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuBatchPtr, cg->cpuBatchPtr, cg->batchWeightSize);
	}

	void setValuesForGpuDense1(CpuGpuMem* cg) {

		cg->denseInputSize = cg->maskCount * cg->featureWidthSize * cg->featureHeightSize;
		cg->denseOutputSize = DENSE_HIDDEN_LAYER_1;

		cpuGpuAlloc(cg, denseEnum, sizeof(float));
		cpuGpuAlloc(cg, denseWeightEnum, sizeof(float));
		cpuGpuFree(cg,featureEnum);
		cpuGpuFree(cg,maskEnum);
		cpuGpuFree(cg,batchEnum);
		cg->batchWeightSize = cg->denseOutputSize;
		cpuGpuAlloc(cg, batchEnum, sizeof(float));

		cudaMemset(cg->gpuDensePtr, 0, cg->denseOutputAllocSize);

		cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuDenseWeightPtr, denseHiddenLayerWeights_1, cg->denseWeightAllocSize);
		cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuBatchPtr, batchNormWeight_2, cg->batchWeightSize);
	}

	void setValuesForGpuDense2(CpuGpuMem* cg) {

		cg->denseInputSize = DENSE_HIDDEN_LAYER_1;
		cg->denseOutputSize = DENSE_OUTPUT_LAYER;

		cpuGpuFree(cg,batchEnum);
		cpuGpuFree(cg,denseWeightEnum);
		cpuGpuAlloc(cg, denseWeightEnum, sizeof(float));

		cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuDenseWeightPtr, denseOutputLayerWeights, cg->denseWeightAllocSize);
	}


	private: System::Void cudaRunToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {

		clock_t tStart = clock();

		const int instance_count = 1;

		CpuGpuMem cgs[1];

		for (int i = 0; i < instance_count; i++)
		{
			CpuGpuMem* cg = &cgs[i];
			
			//----forConv1
			setValuesForGpuConv1(cg); // func

			cudaError_t result = cudaStreamCreate(&cg->stream);
			assert(result == cudaSuccess);
			

			//cpuGpuPin(cg->cpuFeaturePtr, cg->featureAllocSize ); // pin cpu memory size for first layer feature space

		}

		for (int i = 0; i < instance_count; i++)
		{
			CpuGpuMem* cg = &cgs[i];

			cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuImagePtr, cg->cpuImagePtr, cg->imageAllocSize); // host to device
			cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuFeaturePtr, cg->cpuFeaturePtr, cg->featureAllocSize);
			cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuMaskPtr, cg->cpuMaskPtr, cg->maskAllocSize);
			cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuBatchPtr, cg->cpuBatchPtr, cg->batchWeightSize);

			//---------conv2
			conv1ExecGPU(cg); //conv1

			//---------conv2
			setValuesForGpuConv2(cg); // conv2


			convHidden1ExecGPU(cg);

			//---------Dense1
			setValuesForGpuDense1(cg);
			dense1ExecGPU(cg);

			//richTextBox1->Text = "";
			//for (int i = 0; i < 128; i++) {
			//	richTextBox1->Text += cg->cpuDensePtr[i] + "\n";
			//}

			//---------Dense2
			setValuesForGpuDense2(cg);
			dense2ExecGPU(cg);



			softmax(cg->cpuDensePtr, DENSE_OUTPUT_LAYER);


			double cpuClock = (double)(clock() - tStart) / CLOCKS_PER_SEC;
			richTextBox1->Text += "gpu clock time: " + cpuClock + " \n";


			string emot[] = { "Kýzgýn" ,"Nefret" ,"Korku" ,"Mutlu" ,"Üzgün" ,"Þaþkýn" ,"Doðal" };

			chart1->Series["Duygular"]->Points->Clear();
			for (int i = 0; i < 7; i++) {
				String^ str = gcnew String(emot[i].c_str());
				chart1->Series["Duygular"]->Points->AddXY(str, cg->cpuDensePtr[i]);
			}




			cudaDeviceSynchronize();
		}

		for (int i = 0; i < instance_count; i++)
		{
			CpuGpuMem* cg = &cgs[i];

			cpuGpuFree(cg, denseEnum);
			cpuGpuFree(cg, denseWeightEnum);
			//cpuGpuUnpin(cg->cpuFeaturePtr, cg->featureAllocSize );

			cudaError_t result = cudaStreamDestroy(cg->stream);
			assert(result == cudaSuccess);
		}

		cudaError_t result = cudaDeviceSynchronize();
		assert(result == cudaSuccess);


	}

	private: System::Void MyForm_FormClosing(System::Object^ sender, System::Windows::Forms::FormClosingEventArgs^ e) {
		free(ferImages);
		free(emotionLabel);
		free(bmpColoredImage);
		free(raw_intensity);
		free(convInputLayerWeights);
		free(convOutputLayerWeights);
		free(denseOutputLayerWeights);
		free(denseHiddenLayerWeights_1);
		free(batchNormWeight);
		free(batchNormWeight_1);
		free(batchNormWeight_2);
	}


private: System::Void MyForm_Load(System::Object^ sender, System::EventArgs^ e) {
}
};
}
