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


#define IMAGE_WIDTH 48
#define IMAGE_HEIGHT 48
#define TOTAL_IMAGE 35888
#define EMOTION_COUNT 7
#define MASK_SIZE 3
#define MASK_COUNT_FIRST_LAYER 4
#define MASK_COUNT_HIDDEN_LAYER_1 6
#define DENSE_HIDDEN_LAYER 128

#define WEIGHT_PATH "D:\\Ders\\bitirme\\agirlikler\\"


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
		float* convFirstLayerWeights;
		float* convHiddenLayerWeights_1;
		float* denseFirstLayerWeights;
		float* denseHiddenLayerWeights_1;
		float* batchNormWeight;
		float* batchNormWeight_1;
		float* batchNormWeight_2;


		int ferTextBoxInput = 0;


	private: System::Windows::Forms::ToolStripMenuItem^ fer2013DSToolStripMenuItem;
	private: System::Windows::Forms::TextBox^ textBox1;
	private: System::Windows::Forms::ToolStripMenuItem^ testToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ runToolStripMenuItem;
	private: System::Windows::Forms::PictureBox^ pictureBox2;
	private: System::Windows::Forms::PictureBox^ pictureBox3;
	private: System::Windows::Forms::PictureBox^ pictureBox4;
	private: System::Windows::Forms::PictureBox^ pictureBox5;
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
			System::Windows::Forms::DataVisualization::Charting::ChartArea^ chartArea2 = (gcnew System::Windows::Forms::DataVisualization::Charting::ChartArea());
			System::Windows::Forms::DataVisualization::Charting::Legend^ legend2 = (gcnew System::Windows::Forms::DataVisualization::Charting::Legend());
			System::Windows::Forms::DataVisualization::Charting::Series^ series2 = (gcnew System::Windows::Forms::DataVisualization::Charting::Series());
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
			this->pictureBox2 = (gcnew System::Windows::Forms::PictureBox());
			this->pictureBox3 = (gcnew System::Windows::Forms::PictureBox());
			this->pictureBox4 = (gcnew System::Windows::Forms::PictureBox());
			this->pictureBox5 = (gcnew System::Windows::Forms::PictureBox());
			this->chart1 = (gcnew System::Windows::Forms::DataVisualization::Charting::Chart());
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->menuStrip1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox2))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox3))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox4))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox5))->BeginInit();
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
			this->runToolStripMenuItem->Size = System::Drawing::Size(224, 26);
			this->runToolStripMenuItem->Text = L"Run";
			this->runToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::runToolStripMenuItem_Click);
			// 
			// cudaRunToolStripMenuItem
			// 
			this->cudaRunToolStripMenuItem->Name = L"cudaRunToolStripMenuItem";
			this->cudaRunToolStripMenuItem->Size = System::Drawing::Size(224, 26);
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
			this->button1->Location = System::Drawing::Point(1339, 40);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(100, 28);
			this->button1->TabIndex = 4;
			this->button1->Text = L"ImageCpu";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &MyForm::button1_Click);
			// 
			// pictureBox2
			// 
			this->pictureBox2->Location = System::Drawing::Point(12, 385);
			this->pictureBox2->Name = L"pictureBox2";
			this->pictureBox2->Size = System::Drawing::Size(207, 192);
			this->pictureBox2->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pictureBox2->TabIndex = 5;
			this->pictureBox2->TabStop = false;
			// 
			// pictureBox3
			// 
			this->pictureBox3->Location = System::Drawing::Point(240, 385);
			this->pictureBox3->Name = L"pictureBox3";
			this->pictureBox3->Size = System::Drawing::Size(207, 192);
			this->pictureBox3->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pictureBox3->TabIndex = 6;
			this->pictureBox3->TabStop = false;
			// 
			// pictureBox4
			// 
			this->pictureBox4->Location = System::Drawing::Point(470, 385);
			this->pictureBox4->Name = L"pictureBox4";
			this->pictureBox4->Size = System::Drawing::Size(207, 192);
			this->pictureBox4->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pictureBox4->TabIndex = 7;
			this->pictureBox4->TabStop = false;
			// 
			// pictureBox5
			// 
			this->pictureBox5->Location = System::Drawing::Point(696, 385);
			this->pictureBox5->Name = L"pictureBox5";
			this->pictureBox5->Size = System::Drawing::Size(207, 192);
			this->pictureBox5->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pictureBox5->TabIndex = 8;
			this->pictureBox5->TabStop = false;
			// 
			// chart1
			// 
			chartArea2->Name = L"ChartArea1";
			this->chart1->ChartAreas->Add(chartArea2);
			legend2->Name = L"Legend1";
			this->chart1->Legends->Add(legend2);
			this->chart1->Location = System::Drawing::Point(453, 31);
			this->chart1->Name = L"chart1";
			series2->ChartArea = L"ChartArea1";
			series2->Legend = L"Legend1";
			series2->Name = L"Duygular";
			this->chart1->Series->Add(series2);
			this->chart1->Size = System::Drawing::Size(821, 348);
			this->chart1->TabIndex = 9;
			this->chart1->Text = L"chart1";
			// 
			// button2
			// 
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
			this->Controls->Add(this->pictureBox5);
			this->Controls->Add(this->pictureBox4);
			this->Controls->Add(this->pictureBox3);
			this->Controls->Add(this->pictureBox2);
			this->Controls->Add(this->button1);
			this->Controls->Add(this->textBox1);
			this->Controls->Add(this->pictureBox1);
			this->Controls->Add(this->richTextBox1);
			this->Controls->Add(this->menuStrip1);
			this->MainMenuStrip = this->menuStrip1;
			this->Name = L"MyForm";
			this->Text = L"MyForm";
			this->FormClosing += gcnew System::Windows::Forms::FormClosingEventHandler(this, &MyForm::MyForm_FormClosing);
			this->menuStrip1->ResumeLayout(false);
			this->menuStrip1->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox2))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox3))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox4))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox5))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chart1))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion

	private: System::Void weightsToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		Stream^ mystream;
		OpenFileDialog^ openFileDialog1 = gcnew OpenFileDialog;

		openFileDialog1->InitialDirectory = "";
		openFileDialog1->Filter = "txt files (*.txt)|*.txt|All files (*.*)|*.*";
		openFileDialog1->FilterIndex = 2;
		openFileDialog1->RestoreDirectory = true;

		if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK)
		{
			if ((mystream = openFileDialog1->OpenFile()) != nullptr)
			{
				// Insert code to read the stream here.
				String^ strfilename = openFileDialog1->InitialDirectory + openFileDialog1->FileName;
				readFile = File::ReadAllText(strfilename);

				richTextBox1->Text = readFile;

				mystream->Close();
			}
		}
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
			"^[1-9]\d*$"))
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
	private: System::Void runToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {

		//------ File Path
		IntPtr ip = Marshal::StringToHGlobalAnsi(WEIGHT_PATH);
		const char* inputStr = static_cast<const char*>(ip.ToPointer());
		std::string input(inputStr);
		//------ File Path




		//----- input layer cnn
		string filePath = input + "conv2d.csv";
		convFirstLayerWeights = new float[MASK_COUNT_FIRST_LAYER * MASK_SIZE * MASK_SIZE + MASK_COUNT_FIRST_LAYER];
		readWeightFromFile(convFirstLayerWeights, filePath);


		//----- 2. layer cnn
		filePath = input + "conv2d_1.csv";
		convHiddenLayerWeights_1 = new float[MASK_COUNT_HIDDEN_LAYER_1 * MASK_COUNT_FIRST_LAYER * MASK_SIZE * MASK_SIZE + MASK_COUNT_HIDDEN_LAYER_1];
		readWeightFromFile(convHiddenLayerWeights_1, filePath);

		//----- 1. batch norm
		filePath = input + "batch_normalization.csv";
		batchNormWeight = new float[MASK_COUNT_FIRST_LAYER * 4];
		readWeightFromFile(batchNormWeight, filePath);

		//----- 2. batch norm
		filePath = input + "batch_normalization_1.csv";
		batchNormWeight_1 = new float[MASK_COUNT_HIDDEN_LAYER_1 * 4];
		readWeightFromFile(batchNormWeight_1, filePath);

		//----- 3. batch norm
		filePath = input + "batch_normalization_2.csv";
		batchNormWeight_2 = new float[DENSE_HIDDEN_LAYER * 4];
		readWeightFromFile(batchNormWeight_2, filePath);


		int size = (IMAGE_WIDTH - MASK_SIZE + 1) * (IMAGE_HEIGHT - MASK_SIZE + 1) * MASK_COUNT_FIRST_LAYER;
		int sizeW = IMAGE_WIDTH;
		int sizeH = IMAGE_HEIGHT;


		float* fResult = new float[size];

		//1. cnn layer
		fResult = conv1(ferImages, convFirstLayerWeights, sizeW, sizeH, MASK_SIZE, MASK_COUNT_FIRST_LAYER, ferTextBoxInput);
		batchNormalizationConv(fResult, batchNormWeight, sizeW, sizeH, MASK_COUNT_FIRST_LAYER);
		reLU(fResult, sizeW, sizeH, MASK_COUNT_FIRST_LAYER);
		richTextBox1->Text = " ";
		for (int i = 0; i < 100; i++) {
			richTextBox1->Text += fResult[i] + "\n";
		}
		maxPooling(fResult, sizeW, sizeH, MASK_COUNT_FIRST_LAYER, 2, 2);
		size = sizeW * sizeH * MASK_COUNT_FIRST_LAYER; // geçici



		//2.cnn layer
		float* fHiddenResult = new float[(sizeW - MASK_SIZE + 1) * (sizeH - MASK_SIZE + 1) * MASK_COUNT_HIDDEN_LAYER_1];
		fHiddenResult = convHidden(fResult, convHiddenLayerWeights_1, sizeW, sizeH, MASK_SIZE, MASK_COUNT_HIDDEN_LAYER_1, MASK_COUNT_FIRST_LAYER);
		size = sizeW * sizeH * MASK_COUNT_HIDDEN_LAYER_1;




		//int max = 0;
		//int min = 0;
		//float ratio = 0.0;
		BYTE* result = new BYTE[size];

		int sizeTW = sizeW;
		int sizeHW = sizeH;
		//for (int m = 0; m < 6; m++) {
		//	for (int i = 0; i < sizeW * sizeH; i++) {
		//		if ((int)fHiddenResult[(m * sizeW * sizeH) + i] > max) {
		//			max = fHiddenResult[(m * sizeW * sizeH) + i];
		//		}
		//		if ((int)fHiddenResult[(m * sizeW * sizeH) + i] < min) {
		//			min = fHiddenResult[(m * sizeW * sizeH) + i];
		//		}
		//	}
		//	for (int i = 0; i < sizeW * sizeH; i++) {
		//		fHiddenResult[(m * sizeW * sizeH) + i] = fHiddenResult[(m * sizeW * sizeH) + i] - (min);
		//	}
		//	ratio = (float)(max - min) / 255;

		//	for (int i = 0; i < sizeW * sizeH; i++) {
		//		result[(m * sizeW * sizeH) + i] = fHiddenResult[(m * sizeW * sizeH) + i] / ratio;
		//	}
		//}



		batchNormalizationConv(fHiddenResult, batchNormWeight_1, sizeW, sizeH, MASK_COUNT_HIDDEN_LAYER_1);
		reLU(fHiddenResult, sizeW, sizeH, MASK_COUNT_HIDDEN_LAYER_1);
		maxPooling(fHiddenResult, sizeW, sizeH, MASK_COUNT_HIDDEN_LAYER_1, 2, 2);

		//----- FullyConnected Layer 1
		flatten(fHiddenResult, sizeW, sizeH, MASK_COUNT_HIDDEN_LAYER_1);
		int sizeTemp = MASK_COUNT_HIDDEN_LAYER_1 * sizeW * sizeH;
		float* denseResult = new float[DENSE_HIDDEN_LAYER];
		filePath = input + "dense.csv";
		denseFirstLayerWeights = new float[sizeTemp * DENSE_HIDDEN_LAYER + DENSE_HIDDEN_LAYER];
		readWeightFromFile(denseFirstLayerWeights, filePath);
		denseResult = dense(fHiddenResult, denseFirstLayerWeights, sizeTemp, DENSE_HIDDEN_LAYER);

		batchNormalizationDense(denseResult, batchNormWeight_2, DENSE_HIDDEN_LAYER);
		reLU(denseResult, DENSE_HIDDEN_LAYER, 1, 1);

		//----- FullyConnected Layer 2
		filePath = input + "dense_1.csv";
		denseHiddenLayerWeights_1 = new float[7 * DENSE_HIDDEN_LAYER + 7];
		readWeightFromFile(denseHiddenLayerWeights_1, filePath);
		denseResult = dense(denseResult, denseHiddenLayerWeights_1, DENSE_HIDDEN_LAYER, 7);

		softmax(denseResult, 7);

		string emot[] = { "Kýzgýn" ,"Nefret" ,"Korku" ,"Mutlu" ,"Üzgün" ,"Þaþkýn" ,"Doðal" };

		chart1->Series["Duygular"]->Points->Clear();
		for (int i = 0; i < 7; i++) {
			String^ str = gcnew String(emot[i].c_str());
			chart1->Series["Duygular"]->Points->AddXY(str, denseResult[i]);
		}
		

		


		/*for (int i = 0; i < MASK_COUNT_FIRST_LAYER; i++) {
			richTextBox1->Text += "[ ";
			for (int row = 0; row < sizeH; row++) {
				for (int col = 0; col < sizeW; col++) {
					richTextBox1->Text += fHiddenResult[i * sizeW * sizeH + row * sizeW + col] + " ";
				}
				richTextBox1->Text += "\n";
			}
			richTextBox1->Text += " ] \n";
		}*/










		//BYTE* result = new BYTE[size];
		//int tempo = 0;

		//for (int i = 0; i < size; i++) {
		//	fHiddenResult[i] = fHiddenResult[i] * 128;
		//	tempo = (int)fHiddenResult[i];
		//	if (tempo < 0)
		//		result[i] = 0;

		//	else if (tempo > 255)
		//		result[i] = 255;

		//	else
		//		result[i] = tempo;
		//}


		//Bitmap^ surface = gcnew Bitmap(sizeTW, sizeHW);
		//pictureBox2->Image = surface;

		//Bitmap^ surface2 = gcnew Bitmap(sizeTW, sizeHW);
		//pictureBox3->Image = surface2;

		//Bitmap^ surface3 = gcnew Bitmap(sizeTW, sizeHW);
		//pictureBox4->Image = surface3;

		//Bitmap^ surface4 = gcnew Bitmap(sizeTW, sizeHW);
		//pictureBox5->Image = surface4;


		//Color c;


		//for (int row = 0; row < sizeHW; row++)
		//{

		//	for (int column = 0; column < sizeTW; column++)
		//	{
		//		c = Color::FromArgb(result[row * sizeTW + column], result[row * sizeTW + column], result[row * sizeTW + column]);
		//		surface->SetPixel(column, row, c);

		//		c = Color::FromArgb(result[(sizeHW * sizeTW) + row * sizeTW + column], result[(sizeHW * sizeTW) + row * sizeTW + column], result[(sizeHW * sizeTW) + row * sizeTW + column]);
		//		surface2->SetPixel(column, row, c);

		//		c = Color::FromArgb(result[(sizeHW * sizeTW * 2) + row * sizeTW + column], result[(sizeHW * sizeTW * 2) + row * sizeTW + column], result[(sizeHW * sizeTW * 2) + row * sizeTW + column]);
		//		surface3->SetPixel(column, row, c);

		//		c = Color::FromArgb(result[(sizeHW * sizeTW * 3) + row * sizeTW + column], result[(sizeHW * sizeTW * 3) + row * sizeTW + column], result[(sizeHW * sizeTW * 3) + row * sizeTW + column]);
		//		surface4->SetPixel(column, row, c);
		//	}
		//}


		delete[] result;
		delete[] fResult;
		delete[] fHiddenResult;

	}

	private: System::Void button2_Click(System::Object^ sender, System::EventArgs^ e) {
		Int32 myInt = 0;

		if (System::Text::RegularExpressions::Regex::IsMatch(textBox1->Text,
			"^[1-9]\d*$"))
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

		cpuGpuAlloc(cg, 'i', sizeof(int));
		cpuGpuAlloc(cg, 'f', sizeof(float));
		cpuGpuAlloc(cg, 'm', sizeof(float));
		cpuGpuAlloc(cg, 'b', sizeof(float));//batch for conv1

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
				cg->cpuMaskPtr[j * MASK_SIZE * MASK_SIZE + i] = convFirstLayerWeights[i * cg->maskCount + j];
			}
		}

		for (int i = 0; i < cg->maskCount; i++) {
			cg->cpuMaskPtr[cg->maskCount * MASK_SIZE * MASK_SIZE + i] = convFirstLayerWeights[cg->maskCount * MASK_SIZE * MASK_SIZE + i];
		}

	}

	private: System::Void cudaRunToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		//------ File Path
		IntPtr ip = Marshal::StringToHGlobalAnsi(WEIGHT_PATH);
		const char* inputStr = static_cast<const char*>(ip.ToPointer());
		std::string input(inputStr);
		//------ File Path

		//----- read weights from file
		string filePath = input + "conv2d.csv";
		convFirstLayerWeights = new float[MASK_COUNT_FIRST_LAYER * MASK_SIZE * MASK_SIZE + MASK_COUNT_FIRST_LAYER];
		readWeightFromFile(convFirstLayerWeights, filePath);

		//----- 2. layer cnn
		filePath = input + "conv2d_1.csv";
		convHiddenLayerWeights_1 = new float[MASK_COUNT_HIDDEN_LAYER_1 * MASK_COUNT_FIRST_LAYER * MASK_SIZE * MASK_SIZE + MASK_COUNT_HIDDEN_LAYER_1];
		readWeightFromFile(convHiddenLayerWeights_1, filePath);

		//----- 1. batch norm
		filePath = input + "batch_normalization.csv";
		batchNormWeight = new float[MASK_COUNT_FIRST_LAYER * 4];
		readWeightFromFile(batchNormWeight, filePath);


		const int instance_count = 1;

		CpuGpuMem cgs[1];

		for (int i = 0; i < instance_count; i++)
		{
			CpuGpuMem* cg = &cgs[i];
			
			setValuesForGpuConv1(cg); // func

			cudaError_t result = cudaStreamCreate(&cg->stream);
			assert(result == cudaSuccess);


			cpuGpuPin(cg->cpuFeaturePtr, cg->featureAllocSize ); // pin cpu memory size for first layer feature space
		}

		for (int i = 0; i < instance_count; i++)
		{
			CpuGpuMem* cg = &cgs[i];
			
			cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuImagePtr, cg->cpuImagePtr, cg->imageAllocSize ); // host to device
			cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuFeaturePtr, cg->cpuFeaturePtr, cg->featureAllocSize );
			cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuMaskPtr, cg->cpuMaskPtr, cg->maskAllocSize);
			cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuBatchPtr, cg->cpuBatchPtr, cg->batchWeightSize);


			conv1ExecGPU(cg, MASK_COUNT_FIRST_LAYER); //conv1

			//cudamemset();

			cudaDeviceSynchronize();
			cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuImagePtr, cg->gpuImagePtr, cg->imageAllocSize); // device to host
			cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuFeaturePtr, cg->gpuFeaturePtr, cg->featureAllocSize);
			cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuMaskPtr, cg->gpuMaskPtr, cg->maskAllocSize);

			richTextBox1->Text = " ";
 			for (int i = 0; i < 100; i++) {
				richTextBox1->Text += cg->cpuFeaturePtr[i] + "\n";
			}
		}

		for (int i = 0; i < instance_count; i++)
		{
			CpuGpuMem* cg = &cgs[i];

			cpuGpuUnpin(cg->cpuFeaturePtr, cg->featureAllocSize );
			cpuGpuFree(cg, 'i');
			cpuGpuFree(cg, 'f');
			cpuGpuFree(cg, 'm');
			cpuGpuFree(cg, 'b');
			cudaError_t result = cudaStreamDestroy(cg->stream);
			assert(result == cudaSuccess);
		}

		cudaError_t result = cudaDeviceSynchronize();
		assert(result == cudaSuccess);

		//cpu_gpu_print_results(&cg);

	}

	private: System::Void MyForm_FormClosing(System::Object^ sender, System::Windows::Forms::FormClosingEventArgs^ e) {
		delete[] ferImages;
		delete[] emotionLabel;
		delete[] bmpColoredImage;
		delete[] raw_intensity;
		delete[] convFirstLayerWeights;
		delete[] convHiddenLayerWeights_1;
		delete[] denseFirstLayerWeights;
		delete[] denseHiddenLayerWeights_1;
		delete[] batchNormWeight;
		delete[] batchNormWeight_1;
		delete[] batchNormWeight_2;
	}


};
}
