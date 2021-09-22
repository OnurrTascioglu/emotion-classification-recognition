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
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <omp.h>


#define IMAGE_WIDTH 48
#define IMAGE_HEIGHT 48
#define TOTAL_IMAGE 7200
#define MASK_SIZE 3

#define MAX_POOL 2
#define MAX_POOL_STRIDE 2
#define FACE_DETECTION_SCALE 150
#define DENSE_OUTPUT_LAYER 7

#define FEATURE_RESULT_PATH "D:\\Ders\\bitirme\\features\\"
#define HAAR_CASCADE_PATH "C:\\Program Files\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml"


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
	using namespace cv;

	/// <summary>
	/// Summary for MyForm
	/// </summary>
	public ref class MyForm : public System::Windows::Forms::Form
	{
		System::String^ readFile;
		BYTE* bmpColoredImage;

		BYTE* ferImages;
		int lineCount = 0;
		BYTE* emotionLabel;
		BYTE* raw_intensity;
		bool gpuRuntimeBool = 1;
		bool uploadWeights = 0;
		bool openCamera = 0;
		bool openVideo = 0;
		bool freeFerImage = 0;
		int modelId = 0;
		int temp = 0;
		int rank1 = 0;
		int rank2 = 0;
		int accTrue = 0;
		int acc2True = 0;
		//statik
		float* convInputLayerWeights;
		float* convHiddenLayerWeights_1;
		float* convHiddenLayerWeights_2;
		float* convOutputLayerWeights;

		float* denseHiddenLayerWeights_1;
		float* denseHiddenLayerWeights_2;
		float* denseOutputLayerWeights;

		float* batchNormWeight;
		float* batchNormWeight_1;
		float* batchNormWeight_2;
		float* batchNormWeight_3;
		float* batchNormWeight_4;
		float* batchNormWeight_5;

		int ferTextBoxInput = 0;
		int pictureBox3Click = 0;
		int pictureBox4Click = 0;
		int pictureBox5Click = 0;
		int pictureBox6Click = 0;

		int MASK_COUNT_FIRST_LAYER = 0;
		int MASK_COUNT_HIDDEN_LAYER_1 = 0;
		int MASK_COUNT_HIDDEN_LAYER_2 = 0;
		int MASK_COUNT_OUTPUT_LAYER = 0; //output from conv layer = dense input layer
		int DENSE_HIDDEN_LAYER_1 = 0;
		int DENSE_HIDDEN_LAYER_2 = 0;


		System::String^ WEIGHT_PATH;


	private: System::Windows::Forms::ToolStripMenuItem^ fer2013DSToolStripMenuItem;
	private: System::Windows::Forms::TextBox^ textBox1;
	private: System::Windows::Forms::ToolStripMenuItem^ testToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ runToolStripMenuItem;




	private: System::Windows::Forms::DataVisualization::Charting::Chart^ chart1;
	private: System::Windows::Forms::ToolStripMenuItem^ cudaRunToolStripMenuItem;
	private: System::Windows::Forms::Button^ button2;
	private: System::Windows::Forms::Button^ button3;
	private: System::Windows::Forms::PictureBox^ pictureBox2;
	private: System::Windows::Forms::PictureBox^ pictureBox3;
	private: System::Windows::Forms::PictureBox^ pictureBox4;
	private: System::Windows::Forms::Label^ label1;
	private: System::Windows::Forms::ToolStripMenuItem^ displayToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ conv1FeaturesToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ conv2FeaturesToolStripMenuItem;
	private: System::Windows::Forms::Label^ label2;
	private: System::Windows::Forms::Label^ label3;
	private: System::Windows::Forms::Label^ label4;
	private: System::Windows::Forms::ToolStripMenuItem^ cudaRunModel2ToolStripMenuItem;
	private: System::Windows::Forms::Button^ button4;
	private: System::Windows::Forms::Label^ label5;
	private: System::Windows::Forms::Button^ button5;
	private: System::Windows::Forms::PictureBox^ pictureBox5;
	private: System::Windows::Forms::PictureBox^ pictureBox6;
	private: System::Windows::Forms::Label^ label6;
	private: System::Windows::Forms::Label^ label7;
	private: System::Windows::Forms::ToolStripMenuItem^ conv3FeaturesToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ conv4FeaturesToolStripMenuItem;
	private: System::Windows::Forms::Button^ button6;





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
			this->cudaRunModel2ToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->displayToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->conv1FeaturesToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->conv2FeaturesToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->conv3FeaturesToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->conv4FeaturesToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->richTextBox1 = (gcnew System::Windows::Forms::RichTextBox());
			this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->chart1 = (gcnew System::Windows::Forms::DataVisualization::Charting::Chart());
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->button3 = (gcnew System::Windows::Forms::Button());
			this->pictureBox2 = (gcnew System::Windows::Forms::PictureBox());
			this->pictureBox3 = (gcnew System::Windows::Forms::PictureBox());
			this->pictureBox4 = (gcnew System::Windows::Forms::PictureBox());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->button4 = (gcnew System::Windows::Forms::Button());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->button5 = (gcnew System::Windows::Forms::Button());
			this->pictureBox5 = (gcnew System::Windows::Forms::PictureBox());
			this->pictureBox6 = (gcnew System::Windows::Forms::PictureBox());
			this->label6 = (gcnew System::Windows::Forms::Label());
			this->label7 = (gcnew System::Windows::Forms::Label());
			this->button6 = (gcnew System::Windows::Forms::Button());
			this->menuStrip1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chart1))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox2))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox3))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox4))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox5))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox6))->BeginInit();
			this->SuspendLayout();
			// 
			// menuStrip1
			// 
			this->menuStrip1->ImageScalingSize = System::Drawing::Size(20, 20);
			this->menuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {
				this->dosyaToolStripMenuItem,
					this->testToolStripMenuItem, this->displayToolStripMenuItem
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
			this->testToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {
				this->runToolStripMenuItem,
					this->cudaRunToolStripMenuItem, this->cudaRunModel2ToolStripMenuItem
			});
			this->testToolStripMenuItem->Name = L"testToolStripMenuItem";
			this->testToolStripMenuItem->Size = System::Drawing::Size(49, 24);
			this->testToolStripMenuItem->Text = L"Test";
			this->testToolStripMenuItem->Visible = false;
			// 
			// runToolStripMenuItem
			// 
			this->runToolStripMenuItem->Name = L"runToolStripMenuItem";
			this->runToolStripMenuItem->Size = System::Drawing::Size(202, 26);
			this->runToolStripMenuItem->Text = L"Run";
			this->runToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::runToolStripMenuItem_Click);
			// 
			// cudaRunToolStripMenuItem
			// 
			this->cudaRunToolStripMenuItem->Name = L"cudaRunToolStripMenuItem";
			this->cudaRunToolStripMenuItem->Size = System::Drawing::Size(202, 26);
			this->cudaRunToolStripMenuItem->Text = L"CudaRunModel1";
			this->cudaRunToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::cudaRunToolStripMenuItem_Click);
			// 
			// cudaRunModel2ToolStripMenuItem
			// 
			this->cudaRunModel2ToolStripMenuItem->Name = L"cudaRunModel2ToolStripMenuItem";
			this->cudaRunModel2ToolStripMenuItem->Size = System::Drawing::Size(202, 26);
			this->cudaRunModel2ToolStripMenuItem->Text = L"CudaRunModel2";
			this->cudaRunModel2ToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::cudaRunModel2ToolStripMenuItem_Click);
			// 
			// displayToolStripMenuItem
			// 
			this->displayToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(4) {
				this->conv1FeaturesToolStripMenuItem,
					this->conv2FeaturesToolStripMenuItem, this->conv3FeaturesToolStripMenuItem, this->conv4FeaturesToolStripMenuItem
			});
			this->displayToolStripMenuItem->Name = L"displayToolStripMenuItem";
			this->displayToolStripMenuItem->Size = System::Drawing::Size(72, 24);
			this->displayToolStripMenuItem->Text = L"Display";
			// 
			// conv1FeaturesToolStripMenuItem
			// 
			this->conv1FeaturesToolStripMenuItem->Name = L"conv1FeaturesToolStripMenuItem";
			this->conv1FeaturesToolStripMenuItem->Size = System::Drawing::Size(192, 26);
			this->conv1FeaturesToolStripMenuItem->Text = L"Conv1 Features";
			this->conv1FeaturesToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::conv1FeaturesToolStripMenuItem_Click);
			// 
			// conv2FeaturesToolStripMenuItem
			// 
			this->conv2FeaturesToolStripMenuItem->Name = L"conv2FeaturesToolStripMenuItem";
			this->conv2FeaturesToolStripMenuItem->Size = System::Drawing::Size(192, 26);
			this->conv2FeaturesToolStripMenuItem->Text = L"Conv2 Features";
			this->conv2FeaturesToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::conv2FeaturesToolStripMenuItem_Click);
			// 
			// conv3FeaturesToolStripMenuItem
			// 
			this->conv3FeaturesToolStripMenuItem->Name = L"conv3FeaturesToolStripMenuItem";
			this->conv3FeaturesToolStripMenuItem->Size = System::Drawing::Size(192, 26);
			this->conv3FeaturesToolStripMenuItem->Text = L"Conv3 Features";
			this->conv3FeaturesToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::conv3FeaturesToolStripMenuItem_Click);
			// 
			// conv4FeaturesToolStripMenuItem
			// 
			this->conv4FeaturesToolStripMenuItem->Name = L"conv4FeaturesToolStripMenuItem";
			this->conv4FeaturesToolStripMenuItem->Size = System::Drawing::Size(192, 26);
			this->conv4FeaturesToolStripMenuItem->Text = L"Conv4 Features";
			this->conv4FeaturesToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::conv4FeaturesToolStripMenuItem_Click);
			// 
			// openFileDialog1
			// 
			this->openFileDialog1->FileName = L"openFileDialog1";
			// 
			// richTextBox1
			// 
			this->richTextBox1->Location = System::Drawing::Point(12, 407);
			this->richTextBox1->Name = L"richTextBox1";
			this->richTextBox1->Size = System::Drawing::Size(496, 154);
			this->richTextBox1->TabIndex = 1;
			this->richTextBox1->Text = L"";
			this->richTextBox1->TextChanged += gcnew System::EventHandler(this, &MyForm::richTextBox1_TextChanged);
			// 
			// pictureBox1
			// 
			this->pictureBox1->Location = System::Drawing::Point(13, 32);
			this->pictureBox1->Name = L"pictureBox1";
			this->pictureBox1->Size = System::Drawing::Size(495, 347);
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
			chartArea2->Name = L"ChartArea1";
			this->chart1->ChartAreas->Add(chartArea2);
			legend2->Name = L"Legend1";
			this->chart1->Legends->Add(legend2);
			this->chart1->Location = System::Drawing::Point(899, 298);
			this->chart1->Name = L"chart1";
			series2->ChartArea = L"ChartArea1";
			series2->Legend = L"Legend1";
			series2->Name = L"Duygular";
			this->chart1->Series->Add(series2);
			this->chart1->Size = System::Drawing::Size(540, 263);
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
			// button3
			// 
			this->button3->Location = System::Drawing::Point(1339, 162);
			this->button3->Name = L"button3";
			this->button3->Size = System::Drawing::Size(100, 27);
			this->button3->TabIndex = 11;
			this->button3->Text = L"WebCam";
			this->button3->UseVisualStyleBackColor = true;
			this->button3->Click += gcnew System::EventHandler(this, &MyForm::button3_Click);
			// 
			// pictureBox2
			// 
			this->pictureBox2->Location = System::Drawing::Point(514, 31);
			this->pictureBox2->Name = L"pictureBox2";
			this->pictureBox2->Size = System::Drawing::Size(202, 194);
			this->pictureBox2->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pictureBox2->TabIndex = 12;
			this->pictureBox2->TabStop = false;
			// 
			// pictureBox3
			// 
			this->pictureBox3->Location = System::Drawing::Point(723, 32);
			this->pictureBox3->Name = L"pictureBox3";
			this->pictureBox3->Size = System::Drawing::Size(170, 166);
			this->pictureBox3->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pictureBox3->TabIndex = 13;
			this->pictureBox3->TabStop = false;
			this->pictureBox3->Visible = false;
			this->pictureBox3->Click += gcnew System::EventHandler(this, &MyForm::pictureBox3_Click);
			// 
			// pictureBox4
			// 
			this->pictureBox4->Location = System::Drawing::Point(899, 31);
			this->pictureBox4->Name = L"pictureBox4";
			this->pictureBox4->Size = System::Drawing::Size(134, 138);
			this->pictureBox4->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pictureBox4->TabIndex = 14;
			this->pictureBox4->TabStop = false;
			this->pictureBox4->Visible = false;
			this->pictureBox4->Click += gcnew System::EventHandler(this, &MyForm::pictureBox4_Click);
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(722, 201);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(48, 17);
			this->label1->TabIndex = 15;
			this->label1->Text = L"Conv1";
			this->label1->Visible = false;
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(896, 172);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(48, 17);
			this->label2->TabIndex = 16;
			this->label2->Text = L"Conv2";
			this->label2->Visible = false;
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(12, 382);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(0, 17);
			this->label3->TabIndex = 17;
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Location = System::Drawing::Point(569, 228);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(0, 17);
			this->label4->TabIndex = 18;
			// 
			// button4
			// 
			this->button4->Location = System::Drawing::Point(1339, 201);
			this->button4->Name = L"button4";
			this->button4->Size = System::Drawing::Size(100, 24);
			this->button4->TabIndex = 19;
			this->button4->Text = L"Video";
			this->button4->UseVisualStyleBackColor = true;
			this->button4->Click += gcnew System::EventHandler(this, &MyForm::button4_Click);
			// 
			// label5
			// 
			this->label5->AutoSize = true;
			this->label5->Location = System::Drawing::Point(454, 382);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(0, 17);
			this->label5->TabIndex = 20;
			// 
			// button5
			// 
			this->button5->Location = System::Drawing::Point(514, 538);
			this->button5->Name = L"button5";
			this->button5->Size = System::Drawing::Size(75, 23);
			this->button5->TabIndex = 21;
			this->button5->Text = L"Clear";
			this->button5->UseVisualStyleBackColor = true;
			this->button5->Click += gcnew System::EventHandler(this, &MyForm::button5_Click);
			// 
			// pictureBox5
			// 
			this->pictureBox5->Location = System::Drawing::Point(1040, 32);
			this->pictureBox5->Name = L"pictureBox5";
			this->pictureBox5->Size = System::Drawing::Size(112, 113);
			this->pictureBox5->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pictureBox5->TabIndex = 22;
			this->pictureBox5->TabStop = false;
			this->pictureBox5->Visible = false;
			this->pictureBox5->Click += gcnew System::EventHandler(this, &MyForm::pictureBox5_Click);
			// 
			// pictureBox6
			// 
			this->pictureBox6->Location = System::Drawing::Point(1158, 32);
			this->pictureBox6->Name = L"pictureBox6";
			this->pictureBox6->Size = System::Drawing::Size(87, 87);
			this->pictureBox6->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pictureBox6->TabIndex = 23;
			this->pictureBox6->TabStop = false;
			this->pictureBox6->Visible = false;
			this->pictureBox6->Click += gcnew System::EventHandler(this, &MyForm::pictureBox6_Click);
			// 
			// label6
			// 
			this->label6->AutoSize = true;
			this->label6->Location = System::Drawing::Point(1040, 151);
			this->label6->Name = L"label6";
			this->label6->Size = System::Drawing::Size(48, 17);
			this->label6->TabIndex = 24;
			this->label6->Text = L"Conv3";
			this->label6->Visible = false;
			// 
			// label7
			// 
			this->label7->AutoSize = true;
			this->label7->Location = System::Drawing::Point(1158, 122);
			this->label7->Name = L"label7";
			this->label7->Size = System::Drawing::Size(48, 17);
			this->label7->TabIndex = 25;
			this->label7->Text = L"Conv4";
			this->label7->Visible = false;
			// 
			// button6
			// 
			this->button6->Location = System::Drawing::Point(869, 538);
			this->button6->Name = L"button6";
			this->button6->Size = System::Drawing::Size(75, 23);
			this->button6->TabIndex = 26;
			this->button6->Text = L"Test Val";
			this->button6->UseVisualStyleBackColor = true;
			this->button6->Click += gcnew System::EventHandler(this, &MyForm::button6_Click);
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(8, 16);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1451, 592);
			this->Controls->Add(this->button6);
			this->Controls->Add(this->label7);
			this->Controls->Add(this->label6);
			this->Controls->Add(this->pictureBox6);
			this->Controls->Add(this->pictureBox5);
			this->Controls->Add(this->button5);
			this->Controls->Add(this->label5);
			this->Controls->Add(this->button4);
			this->Controls->Add(this->label4);
			this->Controls->Add(this->label3);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->label1);
			this->Controls->Add(this->pictureBox4);
			this->Controls->Add(this->pictureBox3);
			this->Controls->Add(this->pictureBox2);
			this->Controls->Add(this->button3);
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
			this->menuStrip1->ResumeLayout(false);
			this->menuStrip1->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chart1))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox2))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox3))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox4))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox5))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox6))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();

		}


#pragma endregion


	private: System::Void weightsToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {

		FolderBrowserDialog^ folderFileDialog1 = gcnew FolderBrowserDialog();
		folderFileDialog1->Description = "Please choose file path for your weights.";

		if (folderFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK)
		{
			WEIGHT_PATH = folderFileDialog1->SelectedPath + "\\";
		}


		uploadWeights = 1;
		int error = 0;
		int success = 0;

		//------ File Path
		IntPtr ip = Marshal::StringToHGlobalAnsi(WEIGHT_PATH);
		const char* inputStr = static_cast<const char*>(ip.ToPointer());
		std::string input(inputStr);
		//------ File Path

		// File pointer
		string filePath = input + "model.txt";
		fstream fin;
		fin.open(filePath, ios::in);
		if (!fin.good()) error++;

		string line;
		string model;

		std::getline(fin, line); //ilk satýrda model bilgisi yazar
		model = line;
		if (model == "model1") {  //model1 için mimari yapýnýn deðiþkenlere atanmasý ve ekrana yazýlmasý
			button1->Visible = true;
			std::getline(fin, line);
			MASK_COUNT_FIRST_LAYER = stoi(line);
			std::getline(fin, line);
			MASK_COUNT_OUTPUT_LAYER = stoi(line);
			std::getline(fin, line);
			DENSE_HIDDEN_LAYER_1 = stoi(line);
			richTextBox1->Text += "Model 1" + "\n";
			richTextBox1->Text += "Conv1 :" + MASK_COUNT_FIRST_LAYER + "\n";
			richTextBox1->Text += "Conv2 :" + MASK_COUNT_OUTPUT_LAYER + "\n";
			richTextBox1->Text += "FullyC1 :" + DENSE_HIDDEN_LAYER_1 + "\n";
		}
		else if (model == "model2") { //model2 için mimari yapýnýn deðiþkenlere atanmasý ve ekrana yazýlmasý
			button1->Visible = false;
			std::getline(fin, line);
			MASK_COUNT_FIRST_LAYER = stoi(line);
			std::getline(fin, line);
			MASK_COUNT_HIDDEN_LAYER_1 = stoi(line);
			std::getline(fin, line);
			MASK_COUNT_HIDDEN_LAYER_2 = stoi(line);
			std::getline(fin, line);
			MASK_COUNT_OUTPUT_LAYER = stoi(line);
			std::getline(fin, line);
			DENSE_HIDDEN_LAYER_1 = stoi(line);
			std::getline(fin, line);
			DENSE_HIDDEN_LAYER_2 = stoi(line);
			richTextBox1->Text += "Model 2" + "\n";
			richTextBox1->Text += "Conv1 :" + MASK_COUNT_FIRST_LAYER + "\n";
			richTextBox1->Text += "Conv2 :" + MASK_COUNT_HIDDEN_LAYER_1 + "\n";
			richTextBox1->Text += "Conv3 :" + MASK_COUNT_HIDDEN_LAYER_2 + "\n";
			richTextBox1->Text += "Conv4 :" + MASK_COUNT_OUTPUT_LAYER + "\n";
			richTextBox1->Text += "FullyC1 :" + DENSE_HIDDEN_LAYER_1 + "\n";
			richTextBox1->Text += "FullyC2 :" + DENSE_HIDDEN_LAYER_2 + "\n";
		}
		else {
			MessageBox::Show("model.txt file's content is not correct.", "ERROR",
				MessageBoxButtons::OK, MessageBoxIcon::Error);

		}
		fin.close();

		if (model == "model1") { // model1 e uygun þekilde aðýrlýklarýn çekilmesi
			modelId = 1;		 // 2 evriþim 1 tam baðlantýlý gizli, 1 tam baðlantýlý çýkýþ katmaný ve 3 adet batch normalizasyon aðýrlýklarý mevcut	

			//----- input layer cnn
			filePath = input + "conv2d.csv";
			delete[] convInputLayerWeights;
			convInputLayerWeights = new float[MASK_COUNT_FIRST_LAYER * MASK_SIZE * MASK_SIZE + MASK_COUNT_FIRST_LAYER]; //1. evriþim katmanýnda maskelerin derinliði yoktur.
			//Bu yüzden maskelerin boyutlarý [conv1*3*3 + conv1(bias)] þeklindedir.

			if (readWeightFromFile(convInputLayerWeights, filePath)) error++; // readWeightFromFile fonksiyonu csv dosyasýndan aðýrlýklarý okur.
			else success++;

			int sizeW = IMAGE_WIDTH - MASK_SIZE + 1;  //Katmanlar için gerekli olan aðýrlýk dizilerinin boyutlarýný hesaplayabilmek için, 
			int sizeH = IMAGE_HEIGHT - MASK_SIZE + 1; //maxpool ve padding iþlemine baðlý olarak hesaplamalar yapýlýr.
			sizeW = sizeW / MAX_POOL_STRIDE;
			sizeH = sizeH / MAX_POOL_STRIDE;

			//----- 2. layer cnn
			filePath = input + "conv2d_1.csv";
			delete[] convOutputLayerWeights;
			convOutputLayerWeights = new float[MASK_COUNT_OUTPUT_LAYER * MASK_COUNT_FIRST_LAYER * MASK_SIZE * MASK_SIZE + MASK_COUNT_OUTPUT_LAYER];	//2. evriþim katmanýnda maskelerin derinliði
			//vardýr. Çünkü kendisinden önceki katman 3 boyutlu bir feature spacedir. Maskeler derinliði bir önceki katmanýn maske sayýsý kadardýr.
			//Bu durumda aðýrlýk dizisinin boyutu [conv2*conv1*3*3 + conv2(bias)] þeklindedir.

			if (readWeightFromFile(convOutputLayerWeights, filePath)) error++;
			else success++;


			sizeW = sizeW - MASK_SIZE + 1; //Katmanlar için gerekli olan aðýrlýk dizilerinin boyutlarýný hesaplayabilmek için, 
			sizeH = sizeH - MASK_SIZE + 1; //maxpool ve padding iþlemine baðlý olarak hesaplamalar yapýlýr.
			sizeW = sizeW / MAX_POOL_STRIDE;
			sizeH = sizeH / MAX_POOL_STRIDE;


			//----- FullyConnected Layer 1

			int sizeTemp = MASK_COUNT_OUTPUT_LAYER * sizeW * sizeH; //sizeTemp Son evriþim katmanýndaki nöron(piksel) sayýsýdýr.
			filePath = input + "dense.csv";
			delete[] denseHiddenLayerWeights_1;
			denseHiddenLayerWeights_1 = new float[sizeTemp * DENSE_HIDDEN_LAYER_1 + DENSE_HIDDEN_LAYER_1]; //Son evriþim katmanýndaki nöron(piksel) sayýsý dense katmanýnýn giriþidir.
			//Bu yüzden tam baðlantýlý katmanýndaki aðýrlýklarýn sayýsý [sizeTemp*dense1 + dense1(bias)] olur.

			if (readWeightFromFile(denseHiddenLayerWeights_1, filePath)) error++;
			else success++;

			//----- FullyConnected Layer 2
			filePath = input + "dense_1.csv";
			delete[] denseOutputLayerWeights;
			denseOutputLayerWeights = new float[DENSE_OUTPUT_LAYER * DENSE_HIDDEN_LAYER_1 + DENSE_OUTPUT_LAYER]; //Gizli tam baðlantýlý katmanda giriþ*çýkýþ + çýkýþ(bias) kadar aðýrlýk mevcuttur.
			if (readWeightFromFile(denseOutputLayerWeights, filePath)) error++;
			else success++;

			//----- 1. batch norm
			filePath = input + "batch_normalization.csv";
			delete[] batchNormWeight;
			batchNormWeight = new float[MASK_COUNT_FIRST_LAYER * 4]; // Evriþim katmaný çýkýþýnda batch normalizasyonunun, her feature (maske sayýsý kadar) için gamma, beta, 
			//aritmetik ortalama ve varyans deðerleri vardýr. Bu yüzden dizi boyutu 4 ile çarpýlýr.
			if (readWeightFromFile(batchNormWeight, filePath)) error++;
			else success++;

			//----- 2. batch norm
			filePath = input + "batch_normalization_1.csv";
			delete[] batchNormWeight_1;
			batchNormWeight_1 = new float[MASK_COUNT_OUTPUT_LAYER * 4];
			if (readWeightFromFile(batchNormWeight_1, filePath)) error++;
			else success++;

			//----- 3. batch norm
			filePath = input + "batch_normalization_2.csv"; // Dense katmanýnda her bir nöron için 4 adet batch normalizasyonu parametresi vardýr.
			delete[] batchNormWeight_2;
			batchNormWeight_2 = new float[DENSE_HIDDEN_LAYER_1 * 4];
			if (readWeightFromFile(batchNormWeight_2, filePath)) error++;
			else success++;

			MessageBox::Show(success + " File Successfully Loaded. " + "\n" + error + " Errors occured while loading files.", "File Load Information",
				MessageBoxButtons::OK, MessageBoxIcon::Information);
			button1->Enabled = true;
			button2->Enabled = true;
		}

		else if (model == "model2") { // model2 e uygun þekilde aðýrlýklarýn çekilmesi
			modelId = 2;			  // 4 evriþim 2 tam baðlantýlý gizli, 1 tam baðlantýlý çýkýþ katmaný ve 6 adet batch normalizasyon aðýrlýklarý mevcut
									  // model1'deki iþlemler model2'de tekrarlanýr.
			//----- input layer cnn
			filePath = input + "conv2d.csv";  
			delete[] convInputLayerWeights;
			convInputLayerWeights = new float[MASK_COUNT_FIRST_LAYER * MASK_SIZE * MASK_SIZE + MASK_COUNT_FIRST_LAYER];
			if (readWeightFromFile(convInputLayerWeights, filePath)) error++;
			else success++;

			int sizeW = IMAGE_WIDTH - MASK_SIZE + 1;
			int sizeH = IMAGE_HEIGHT - MASK_SIZE + 1;
			sizeW = sizeW / MAX_POOL_STRIDE;
			sizeH = sizeH / MAX_POOL_STRIDE;

			//----- 2. layer cnn
			filePath = input + "conv2d_1.csv";
			delete[] convHiddenLayerWeights_1;
			convHiddenLayerWeights_1 = new float[MASK_COUNT_HIDDEN_LAYER_1 * MASK_COUNT_FIRST_LAYER * MASK_SIZE * MASK_SIZE + MASK_COUNT_HIDDEN_LAYER_1];
			if (readWeightFromFile(convHiddenLayerWeights_1, filePath)) error++;
			else success++;

			sizeW = sizeW - MASK_SIZE + 1;
			sizeH = sizeH - MASK_SIZE + 1;

			//----- 3. layer cnn
			filePath = input + "conv2d_2.csv";
			delete[] convHiddenLayerWeights_2;
			convHiddenLayerWeights_2 = new float[MASK_COUNT_HIDDEN_LAYER_2 * MASK_COUNT_HIDDEN_LAYER_1 * MASK_SIZE * MASK_SIZE + MASK_COUNT_HIDDEN_LAYER_2];
			if (readWeightFromFile(convHiddenLayerWeights_2, filePath)) error++;
			else success++;

			sizeW = sizeW - MASK_SIZE + 1;
			sizeH = sizeH - MASK_SIZE + 1;
			sizeW = sizeW / MAX_POOL_STRIDE;
			sizeH = sizeH / MAX_POOL_STRIDE;

			//----- 4. layer cnn
			filePath = input + "conv2d_3.csv";
			delete[] convOutputLayerWeights;
			convOutputLayerWeights = new float[MASK_COUNT_OUTPUT_LAYER * MASK_COUNT_HIDDEN_LAYER_2 * MASK_SIZE * MASK_SIZE + MASK_COUNT_OUTPUT_LAYER];
			if (readWeightFromFile(convOutputLayerWeights, filePath)) error++;
			else success++;

			sizeW = sizeW - MASK_SIZE + 1;
			sizeH = sizeH - MASK_SIZE + 1;

			//--------------------------------------------------------------
			//----- FullyConnected Layer 1

			int sizeTemp = MASK_COUNT_OUTPUT_LAYER * sizeW * sizeH;
			filePath = input + "dense.csv";
			delete[] denseHiddenLayerWeights_1;
			denseHiddenLayerWeights_1 = new float[sizeTemp * DENSE_HIDDEN_LAYER_1 + DENSE_HIDDEN_LAYER_1];
			if (readWeightFromFile(denseHiddenLayerWeights_1, filePath)) error++;
			else success++;

			//----- FullyConnected Layer 2
			filePath = input + "dense_1.csv";
			delete[] denseHiddenLayerWeights_2;
			denseHiddenLayerWeights_2 = new float[DENSE_HIDDEN_LAYER_1 * DENSE_HIDDEN_LAYER_2 + DENSE_HIDDEN_LAYER_2];
			if (readWeightFromFile(denseHiddenLayerWeights_2, filePath)) error++;
			else success++;

			//----- FullyConnected Layer 3
			filePath = input + "dense_2.csv";
			delete[] denseOutputLayerWeights;
			denseOutputLayerWeights = new float[DENSE_OUTPUT_LAYER * DENSE_HIDDEN_LAYER_2 + DENSE_OUTPUT_LAYER];
			if (readWeightFromFile(denseOutputLayerWeights, filePath)) error++;
			else success++;

			//----- 1. batch norm
			filePath = input + "batch_normalization.csv";
			delete[] batchNormWeight;
			batchNormWeight = new float[MASK_COUNT_FIRST_LAYER * 4];
			if (readWeightFromFile(batchNormWeight, filePath)) error++;
			else success++;

			//----- 2. batch norm
			filePath = input + "batch_normalization_1.csv";
			delete[] batchNormWeight_1;
			batchNormWeight_1 = new float[MASK_COUNT_HIDDEN_LAYER_1 * 4];
			if (readWeightFromFile(batchNormWeight_1, filePath)) error++;
			else success++;

			//----- 3. batch norm
			filePath = input + "batch_normalization_2.csv";
			delete[] batchNormWeight_2;
			batchNormWeight_2 = new float[MASK_COUNT_HIDDEN_LAYER_2 * 4];
			if (readWeightFromFile(batchNormWeight_2, filePath)) error++;
			else success++;

			//----- 4. batch norm
			filePath = input + "batch_normalization_3.csv";
			delete[] batchNormWeight_3;
			batchNormWeight_3 = new float[MASK_COUNT_OUTPUT_LAYER * 4];
			if (readWeightFromFile(batchNormWeight_3, filePath)) error++;
			else success++;

			//----- 5. batch norm
			filePath = input + "batch_normalization_4.csv";
			delete[] batchNormWeight_4;
			batchNormWeight_4 = new float[DENSE_HIDDEN_LAYER_1 * 4];
			if (readWeightFromFile(batchNormWeight_4, filePath)) error++;
			else success++;

			//----- 6. batch norm
			filePath = input + "batch_normalization_5.csv";
			delete[] batchNormWeight_5;
			batchNormWeight_5 = new float[DENSE_HIDDEN_LAYER_2 * 4];
			if (readWeightFromFile(batchNormWeight_5, filePath)) error++;
			else success++;


			MessageBox::Show(success + " File Successfully Loaded. " + "\n" + error + " Errors occured while loading files.", "File Load Information",
				MessageBoxButtons::OK, MessageBoxIcon::Information);
			button1->Enabled = true;
			button2->Enabled = true;
		}

	}
	private: System::Void pictureToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		//LPCTSTR input;
		//CString str;
		//int Width, Height;
		//float resizeX = 0.0, resizeY = 0.0;
		//long Size;
		//int integer = 0;
		//float fraction = 0.0, tempFrac = 0.0;
		//float total = 0.0;
		//int count = 2;

		//if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
		//	str = openFileDialog1->FileName;
		//	input = (LPCTSTR)str;
		//	float mean = 0.0;

		//	//BMP Image Reading
		//	bmpColoredImage = LoadBMP(&Width, &Height, &Size, input);
		//	raw_intensity = ConvertBMPToIntensity(bmpColoredImage, Width, Height); // BMP Gray picture

		//	BYTE* buffer = new BYTE[IMAGE_WIDTH * IMAGE_HEIGHT];

		//	resizeX = round((float)Width / IMAGE_WIDTH);
		//	resizeY = round((float)Height / IMAGE_HEIGHT);

		//	if (resizeX < 1 || resizeY < 1) {
		//		MessageBox::Show("Please choose greater than 48x48 image.");
		//	}
		//	else {

		//		//vector<Rect> face;
		//		//CascadeClassifier cascade;
		//		//cascade.load(HAAR_CASCADE_PATH);

		//		//Mat img(Height, Width, CV_8S, raw_intensity);

		//		//equalizeHist(img, img);

		//		//cascade.detectMultiScale(img, face, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, cv::Size(FACE_DETECTION_SCALE, FACE_DETECTION_SCALE));

		//		//System::Drawing::Bitmap^ b;
		//		//System::IntPtr ptr(img.ptr());
		//		//b = gcnew System::Drawing::Bitmap(img.cols, img.rows, img.step, System::Drawing::Imaging::PixelFormat::Format24bppRgb, ptr);
		//		//pictureBox1->Image = b;

		//		//printFaces(face, img);

		//		//if (modelId == 1) {
		//		//	cudaRunToolStripMenuItem_Click(sender, e);
		//		//}
		//		//if (modelId == 2) {
		//		//	cudaRunModel2ToolStripMenuItem_Click(sender, e);
		//		//}

		//	}
		//}
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
			delete[] ferImages;
			ferImages = new BYTE[IMAGE_WIDTH * IMAGE_HEIGHT * TOTAL_IMAGE]; //Toplam boyut 48*48*GörselSayýsý þeklindedir.
			emotionLabel = new BYTE[TOTAL_IMAGE];

			if ((mystream = openFileDialog1->OpenFile()) != nullptr)
			{

				// File pointer
				fstream fin;

				System::String^ fileName = openFileDialog1->FileName;

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
				lineCount = 0;

				while (!fin.eof()) {

					std::getline(fin, line);

					k = 0;
					count = 0;

					if (lineBool == 1) { //dosyada ilk satýrda kolon isimleri mevcut bu yüzden ilk satýra girmemesi gerekir.
						for (int i = 0; i < line.length(); i++) { //satýr sonuna kadar harf harf bakýlýr
							if (line[i].Equals(',')) { 
								if (count == 0) { 
									word = line.substr(k, i - k); //ilk virgülden önceki harf kaydedilir.
									emotionLabel[lineCount] = stoi(word); //Emotion bilgileri alýnýr.
									k = i + 1;
									//richTextBox1->Text += emotionLabel[lineCount] + " \n";
								}

								count++;

								if (count == 2) { //son piksel deðerininden sonra  ' ' karakteri olmadýðý için, virgülden önceki sonuncu piksel deðeri alýnýr.
									word = line.substr(k, i - k);
									ferImages[(imageIndex * 48 * 48) + ferIndex] = stoi(word);
									ferIndex++;
								}
							}
							if (count == 1) {
								if (line[i].Equals(' ')) {
									word = line.substr(k, i - k);
									ferImages[(imageIndex * 48 * 48) + ferIndex] = stoi(word); //piksel deðerleri okunur
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

		if (System::Text::RegularExpressions::Regex::IsMatch(textBox1->Text, //textbox'a girilecek deðerin koþullarý
			"^[0-9]{1,6}$"))
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
				for (int row = 0; row < IMAGE_HEIGHT; row++) // Test görüntüsünün ekrana basýlmasý 
				{
					for (int column = 0; column < IMAGE_WIDTH; column++)
					{
						c = Color::FromArgb(ferImages[point + row * IMAGE_WIDTH + column], ferImages[point + row * IMAGE_WIDTH + column], ferImages[point + row * IMAGE_WIDTH + column]);
						surface->SetPixel(column, row, c);
					}
				}
				runToolStripMenuItem_Click(sender, e); // Test görüntüsünün cpu ile koþulmasý
			}
		}
		else {
			MessageBox::Show("Not a number");
		}

	}
		   void saveFeatureBMP(float* fResult, int sizeW, int sizeH, int featureCount, char convIndex) {
			   LPCTSTR input;
			   CString str;
			   BYTE* buffer = new BYTE[sizeW * sizeH * featureCount];
			   long* a = new long;
			   BYTE* buffer2;
			   int max = 0;
			   int min = 0;
			   float ratio = 0.0;
			   float* tempResult = new float[featureCount * sizeH * sizeW];

			   for (int m = 0; m < featureCount; m++) { 
				   for (int i = 0; i < sizeW * sizeH; i++) { // feature deðerleri 0-255 arasýna
					   if ((int)fResult[(m * sizeW * sizeH) + i] > max) {
						   max = fResult[(m * sizeW * sizeH) + i];
					   }
					   if ((int)fResult[(m * sizeW * sizeH) + i] < min) {
						   min = fResult[(m * sizeW * sizeH) + i];
					   }
				   }
				   for (int i = 0; i < sizeW * sizeH; i++) { // bütün deðerlerden en küçük deðer çýkarýlýr. Bu durumda en küçük deðer 0 olur.
					   tempResult[(m * sizeW * sizeH) + i] = fResult[(m * sizeW * sizeH) + i] - (min);
				   }
				   ratio = (float)(max - min) / 254; // Bütün pikseller bu orana bölünür.

				   for (int i = 0; i < sizeW * sizeH; i++) {
					   buffer[(m * sizeW * sizeH) + i] = (int)(tempResult[(m * sizeW * sizeH) + i] / ratio);
				   }
			   }
			   str = FEATURE_RESULT_PATH + "conv" + convIndex + "feature.bmp";
			   input = (LPCTSTR)str;
			   buffer2 = ConvertIntensityToBMP(buffer, sizeW, sizeH * featureCount, a); // intensity, BMP Formatýna çevrilir
			   SaveBMP(buffer2, sizeW, sizeH * featureCount, *a, input); // BMP resmi kaydeder
		   }

	private: System::Void runToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {

		int size = (IMAGE_WIDTH - MASK_SIZE + 1) * (IMAGE_HEIGHT - MASK_SIZE + 1) * MASK_COUNT_FIRST_LAYER; //Giriþ evriþim katmanýn boyutu padding iþlemi yok
		int sizeW = IMAGE_WIDTH; // sizeW ve sizeH evriþim, maxpooling fonksiyonlarýnda güncellenir.
		int sizeH = IMAGE_HEIGHT;

		float* fResult = new float[size];

		clock_t tStart = clock(); // iþlem süresinin ölçümü için
		//1. cnn layer
		fResult = conv1(ferImages, convInputLayerWeights, sizeW, sizeH, MASK_SIZE, MASK_COUNT_FIRST_LAYER, ferTextBoxInput); //1. evriþim katmaný fonksiyonu padding yok!
		batchNormalizationConv(fResult, batchNormWeight, sizeW, sizeH, MASK_COUNT_FIRST_LAYER); //BatchNormalizasyonu katmaný fonksiyonu
		reLU(fResult, sizeW, sizeH, MASK_COUNT_FIRST_LAYER); //Relu iþlemi
		maxPooling(fResult, sizeW, sizeH, MASK_COUNT_FIRST_LAYER, MAX_POOL, MAX_POOL_STRIDE); // Maxpooling fonksiyonu

		saveFeatureBMP(fResult, sizeW, sizeH, MASK_COUNT_FIRST_LAYER, 1); // 1. evriþim katmaný görsellerini kaydeden fonksiyon

		//2.cnn layer
		float* fHiddenResult = new float[(sizeW - MASK_SIZE + 1) * (sizeH - MASK_SIZE + 1) * MASK_COUNT_OUTPUT_LAYER];
		fHiddenResult = convHidden(fResult, convOutputLayerWeights, sizeW, sizeH, MASK_SIZE, MASK_COUNT_OUTPUT_LAYER, MASK_COUNT_FIRST_LAYER);
		size = sizeW * sizeH * MASK_COUNT_OUTPUT_LAYER; // sizeW ve sizeH her evriþim ve max pool katmanýnda güncellenir!

		saveFeatureBMP(fHiddenResult, sizeW, sizeH, MASK_COUNT_OUTPUT_LAYER, 2);

		batchNormalizationConv(fHiddenResult, batchNormWeight_1, sizeW, sizeH, MASK_COUNT_OUTPUT_LAYER);
		reLU(fHiddenResult, sizeW, sizeH, MASK_COUNT_OUTPUT_LAYER);
		maxPooling(fHiddenResult, sizeW, sizeH, MASK_COUNT_OUTPUT_LAYER, MAX_POOL, MAX_POOL_STRIDE);

		//----- FullyConnected Layer 1
		flatten(fHiddenResult, sizeW, sizeH, MASK_COUNT_OUTPUT_LAYER); //son evriþim katmanýnýn  flatten iþlemi ile dense katmanýna uygun hale getirilir
		int sizeTemp = MASK_COUNT_OUTPUT_LAYER * sizeW * sizeH;
		float* denseResult = new float[DENSE_HIDDEN_LAYER_1];
		denseResult = dense(fHiddenResult, denseHiddenLayerWeights_1, sizeTemp, DENSE_HIDDEN_LAYER_1); // dense iþlemini yapan fonksiyon
		batchNormalizationDense(denseResult, batchNormWeight_2, DENSE_HIDDEN_LAYER_1); //tam baðlantýlý katman için batch iþlemi yapan fonksiyon
		reLU(denseResult, DENSE_HIDDEN_LAYER_1, 1, 1);

		//----- FullyConnected Layer 2
		denseResult = dense(denseResult, denseOutputLayerWeights, DENSE_HIDDEN_LAYER_1, DENSE_OUTPUT_LAYER); // son dense katmaný

		softmax(denseResult, DENSE_OUTPUT_LAYER); // sofmax iþlemi yapan fonksiyon
		double cpuClock = (double)(clock() - tStart) / CLOCKS_PER_SEC;
		richTextBox1->Text += "cpu clock time: " + cpuClock + " \n";

		string emot[] = { "Kýzgýn" ,"Nefret" ,"Korku" ,"Mutlu" ,"Üzgün" ,"Þaþkýn" ,"Doðal" };

		chart1->Series["Duygular"]->Points->Clear(); //Sonuçlar grafik üzerinde gösterilir.
		for (int i = 0; i < 7; i++) {
			System::String^ str = gcnew System::String(emot[i].c_str());
			chart1->Series["Duygular"]->Points->AddXY(str, denseResult[i]);
		}

		delete[] fResult;
		delete[] fHiddenResult;

	}
	private: System::Void button2_Click(System::Object^ sender, System::EventArgs^ e) {
		Int32 myInt = 0;

		if (System::Text::RegularExpressions::Regex::IsMatch(textBox1->Text,
			"^[0-9]{1,6}$"))
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
				if (modelId == 1) {
					cudaRunToolStripMenuItem_Click(sender, e);
				}
				if (modelId == 2) {
					cudaRunModel2ToolStripMenuItem_Click(sender, e);
				}
			}
		}
		else {
			MessageBox::Show("Not a number");
		}
	}

		   void setValuesForGpuConv1(CpuGpuMem* cg) {
			   cg->imageHeightSize = IMAGE_HEIGHT; //görüntü yüksekliði
			   cg->imageWidthSize = IMAGE_WIDTH; //görüntü geniþliði
			   cg->featureWidthSize = IMAGE_WIDTH - MASK_SIZE + 1; // Feature space geniþliði Padding yok
			   cg->featureHeightSize = IMAGE_HEIGHT - MASK_SIZE + 1; // Feature space geniþliði Padding yok
			   cg->maskWHSize = MASK_SIZE; //maskenin yükseklik (geniþlik) bilgisi örn.3x3
			   cg->maskCount = MASK_COUNT_FIRST_LAYER; //ilk katmanýn maske sayýsý 
			   cg->maskDim = 1;  //giriþ görüntüsünün derinliði tek katmanlý olduðu için maskelerin derinliði 1 olur
			   cg->pool = MAX_POOL;
			   cg->stride = MAX_POOL_STRIDE;
			   cg->batchWeightSize = cg->maskCount; //maske sayýsý kadar batch deðerlerimiz var. Her maske 4 batch (gamma,beta,a.o,varyans) deðerine sahiptir


			   //conv1 için gerekli bellek tahsisleri
			   cpuGpuAlloc(cg, imageEnum, sizeof(int));  // görüntü için bellek bölgesi tahsisi
			   cpuGpuAlloc(cg, featureEnum, sizeof(float)); // feature space için bellek bölgesi tahsisi
			   cpuGpuAlloc(cg, maskEnum, sizeof(float)); // maskeler için bellek bölgesi tahsisi
			   cpuGpuAlloc(cg, batchEnum, sizeof(float)); // batch aðýrlýklarý için bellek bölgesi tahsisi

			   for (int i = 0; i < cg->maskCount * 4; i++)
				   cg->cpuBatchPtr[i] = batchNormWeight[i]; //dosyadan okunan aðýrlýklar cg->cpuBatchPtr dizisine atanýr.

			   int* cpu_int32 = (int*)cg->cpuImagePtr; //cg->cpuImagePtr void pointer olduðundan int data tipine setlenmeli. cpu_int32 ile cg->cpuImagePtr ayný bellek bölgesini iþaret eder
			   for (int i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH; i++) {
				   cpu_int32[i] = ferImages[(ferTextBoxInput * IMAGE_WIDTH * IMAGE_HEIGHT) + i]; //Veri setinden istenen görüntü cg->cpuImagePtr ye atanýr.
			   }

			   for (int i = 0; i < cg->featureWidthSize * cg->featureHeightSize * cg->maskCount; i++) {
				   cg->cpuFeaturePtr[i] = 0.0; //Feature uzayý GPU ya aktarýlmadan önce 0'a setlenir. 
			   }

			   //aðýrlýklarýn sýralanmasý (CPU'daki (test.cpp) iþlemin aynýsý)
			   for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++) {  //convInputLayerWeights parametresinde aðýrlýklar mevcuttur. Fakat aðýrlýklarýn diziliþ sýrasý pythonda kaydedildiði gibidir.
				   for (int j = 0; j < cg->maskCount; j++) {	  //Bu atama iþlemi maske gezdirme iþlemlerini daha anlaþýlýr yapabilmek için yapýlmýþtýr. CPU kýsmýnda yapýlan iþlemin aynýsýdýr.
					   cg->cpuMaskPtr[j * MASK_SIZE * MASK_SIZE + i] = convInputLayerWeights[i * cg->maskCount + j]; //maskelerin, cg->cpuMaskPtr struct yapýsýna atanmasý
				   }
			   }

			   for (int i = 0; i < cg->maskCount; i++) {
				   cg->cpuMaskPtr[cg->maskCount * MASK_SIZE * MASK_SIZE + i] = convInputLayerWeights[cg->maskCount * MASK_SIZE * MASK_SIZE + i]; // bias deðerlerinin atanmasý.
			   }
		   }
		   void setValuesForGpuConv2(CpuGpuMem* cg) {
			   cg->maskWHSize = MASK_SIZE;  //maskenin yükseklik (geniþlik) bilgisi örn.3x3
			   cg->maskCount = MASK_COUNT_OUTPUT_LAYER;  //yeni evriþim katmanýnýn boyutu(maske sayýsý)
			   cg->maskDim = MASK_COUNT_FIRST_LAYER;  //bir önceki katmanýn boyutu Bu katmandaki maskelerin derinliðine eþittir
			   cg->batchWeightSize = cg->maskCount;  //batch aðýrlýklarý boyutu

			   cpuGpuFree(cg, imageEnum); // Görüntü(image) yalnýzca giriþ evriþim katmanýnda kullanýlýr. Bu katmanda bellek bölgesi serbest býrakýlýr.
			   cpuGpuFree(cg, maskEnum);  //maskelerin boyutlarý bu katmanda farklý. Bu yüzden yeniden tahsis edilmeli
			   cpuGpuFree(cg, batchEnum); //batch aðýrlýklarý bu katmanda farklý. Bu yüzden yeniden tahsis edilmeli

			   cpuGpuAlloc(cg, maskEnum, sizeof(float)); //  2. katman için maskelerin bellek bölgesi tahsisi
			   cpuGpuAlloc(cg, batchEnum, sizeof(float)); //  2. katman için batch aðýrlýklarýnýn bellek bölgesi tahsisi


			   //aðýrlýklarýn sýralanmasý (CPU'daki (test.cpp) iþlemin aynýsý)
			   int count = 0;
			   for (int i = 0; i < cg->maskWHSize * cg->maskWHSize; i++) {
				   for (int j = 0; j < cg->maskDim; j++) {
					   for (int k = 0; k < cg->maskCount; k++) {
						   cg->cpuMaskPtr[k * cg->maskWHSize * cg->maskWHSize * cg->maskDim + (j * cg->maskWHSize * cg->maskWHSize) + i] = convOutputLayerWeights[count];
						   count++;
					   }
				   }
			   }

			   // maskelerdeki bias deðerlerinin atanmasý
			   for (int i = 0; i < cg->maskCount; i++)
			   {
				   cg->cpuMaskPtr[cg->maskCount * cg->maskDim * cg->maskWHSize * cg->maskWHSize + i] =
					   convOutputLayerWeights[cg->maskCount * cg->maskDim * cg->maskWHSize * cg->maskWHSize + i];
			   }
			   cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuMaskPtr, cg->cpuMaskPtr, cg->maskAllocSize); // Maskelerin RAM bellekten GPU belleðine atanmasý 
			   cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuBatchPtr, batchNormWeight_1, cg->batchWeightSize); // Batch aðýrlýklarýnýn RAM bellekten GPU belleðine atanmasý 
		   }
		   void setValuesForGpuDense1(CpuGpuMem* cg) {

			   cg->denseInputSize = cg->maskCount * cg->featureWidthSize * cg->featureHeightSize;  // dense giriþ katmaný boyu (son evriþim katmanýnýn çýkýþý)
			   cg->denseOutputSize = DENSE_HIDDEN_LAYER_1; // dense çýkýþ katmaný boyu

			   cpuGpuAlloc(cg, denseEnum, sizeof(float)); //dense çýkýþ katmaný bellek tahsisi
			   cpuGpuAlloc(cg, denseWeightEnum, sizeof(float)); //dense aðýrlýklarý bellek tahsisi
			   cpuGpuFree(cg, featureEnum); //dense katmanýnda feature ve mask dizilerine ihtiyaç kalmadý
			   cpuGpuFree(cg, maskEnum);    //bu yüzden bu bellek bölgeleri serbest býrakýlmalý
			   cpuGpuFree(cg, batchEnum);   //batch aðýrlýklarý yeniden boyutlandýrýlmalý
			   cg->batchWeightSize = cg->denseOutputSize;
			   cpuGpuAlloc(cg, batchEnum, sizeof(float));  // batch bellek bölgesi tahsisi

			   cudaMemset(cg->gpuDensePtr, 0, cg->denseOutputAllocSize); //Gpu bellekteki gpuDensePtr içeriði 0'a setlenir

			   cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuDenseWeightPtr, denseHiddenLayerWeights_1, cg->denseWeightAllocSize); //dense katmaný Ram bellekten GPU belleðine atanýr
			   cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuBatchPtr, batchNormWeight_2, cg->batchWeightSize); //batch katmaný Ram bellekten GPU belleðine atanýr
		   }
		   void setValuesForGpuDense2(CpuGpuMem* cg) {

			   cg->denseInputSize = DENSE_HIDDEN_LAYER_1; //2. dense katmanýnýn giriþi 
			   cg->denseOutputSize = DENSE_OUTPUT_LAYER; //2. dense katmanýnýn çýkýþý

			   cpuGpuFree(cg, batchEnum); 
			   cpuGpuFree(cg, denseWeightEnum);
			   cpuGpuAlloc(cg, denseWeightEnum, sizeof(float));

			   cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuDenseWeightPtr, denseOutputLayerWeights, cg->denseWeightAllocSize);
		   }
		   void showFeatureOnPictureBox(float* fResult, int sizeW, int sizeH, int featureCount, int pictureBoxIndex, int indexM) {
			   LPCTSTR input;
			   CString str;
			   BYTE* buffer = new BYTE[sizeW * sizeH];
			   long* a = new long;
			   BYTE* buffer2;
			   int max = 0;
			   int min = 0;
			   float ratio = 0.0;
			   float* tempResult = new float[sizeH * sizeW];
			   if (indexM > featureCount) { //indexM feature uzayýndaki gösterilecek feature ýn indisidir.
				   indexM = 0;
			   }

			   for (int i = 0; i < sizeW * sizeH; i++) {
				   if ((int)fResult[(indexM * sizeW * sizeH) + i] > max) {
					   max = fResult[(indexM * sizeW * sizeH) + i];  // featureda ki max ve min deðerleri bulunur
				   }
				   if ((int)fResult[(indexM * sizeW * sizeH) + i] < min) {
					   min = fResult[(indexM * sizeW * sizeH) + i];
				   }
			   }
			   for (int i = 0; i < sizeW * sizeH; i++) {
				   tempResult[i] = fResult[(indexM * sizeW * sizeH) + i] - (min); //bütün deðerlerden min deðer çýkarýlýr. Bu durumda en küçük deðer 0, en yüksek deðer max-min deðeri olur
			   }
			   ratio = (float)(max - min) / 254; //max ile 254 (0-255 piksel aralýðý, hata payý için 254 deðeri kullanýlýr) arasýndaki oran bulunur

			   for (int i = 0; i < sizeW * sizeH; i++) {
				   buffer[i] = (int)(tempResult[i] / ratio); //bütün deðerler bu orana bölünür. Bu sayede 0-255 arasý piksel deðerleri elde edilmiþ olur.
			   }

			   Bitmap^ surface = gcnew Bitmap(sizeW, sizeH);
			   if (pictureBoxIndex == 0) {  //istenen picture box'a surface atanýr
				   pictureBox3->Image = surface;
			   }
			   if (pictureBoxIndex == 1) {
				   pictureBox4->Image = surface;
			   }
			   if (pictureBoxIndex == 2) {
				   pictureBox5->Image = surface;
			   }
			   if (pictureBoxIndex == 3) {
				   pictureBox6->Image = surface;
			   }

			   Color c;
			   for (int row = 0; row < sizeH; row++)
			   {
				   for (int column = 0; column < sizeW; column++)
				   {
					   c = Color::FromArgb(buffer[row * sizeW + column], buffer[row * sizeW + column], buffer[row * sizeW + column]);
					   surface->SetPixel(column, row, c);
				   }
			   }

		   }
		   void printGraph(CpuGpuMem* cg, double gpuClock) {

			   //Formdaki grafik kýsmýnýn düzenlenmesi

			   label3->Text = "Gpu clock time: " + gpuClock + " \n";

			   string emot[] = { "Kýzgýn" ,"Nefret" ,"Korku" ,"Mutlu" ,"Üzgün" ,"Þaþkýn" ,"Doðal" }; //duygular

			   chart1->Series["Duygular"]->Points->Clear();
			   for (int i = 0; i < DENSE_OUTPUT_LAYER; i++) {
				   System::String^ str = gcnew System::String(emot[i].c_str()); // string den String^ ne dönüþüm
				   chart1->Series["Duygular"]->Points->AddXY(str, cg->cpuDensePtr[i]); //duygu ve deðerini chartta gösterir
			   }

			   float max = 0.0, max2 = 0.0;
			   int maxIndex = 0, max2Index = 0;
			   for (int i = 0; i < DENSE_OUTPUT_LAYER; i++) {   //rank 2 ye göre sýralamak için 2 kez max deðer bulunur.
				   if (cg->cpuDensePtr[i] > max) {
					   max = cg->cpuDensePtr[i];
					   maxIndex = i;
				   }
			   }
			   for (int i = 0; i < DENSE_OUTPUT_LAYER; i++) {
				   if (cg->cpuDensePtr[i] > max2 && i != maxIndex) {
					   max2 = cg->cpuDensePtr[i];
					   max2Index = i;
				   }
			   }

				
			   string label[7] = {"Kýzgýn ","Nefret ","Korku ","Mutlu ","Üzgün ","Þaþkýn ","Doðal "};

			   System::String^ Str = gcnew System::String(label[maxIndex].c_str());
			   System::String^ Str2 = gcnew System::String(label[max2Index].c_str());

			   label4->Text = "";
			   label4->Text += Str;
			   label4->Text += Str2;

		   }

	private: System::Void cudaRunToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {

		clock_t tStart = clock();

		CpuGpuMem new_cg;   
		CpuGpuMem* cg = &new_cg; //CpuGpuMem türünden bir yapý örneði oluþturulur.

		//----forConv1
		setValuesForGpuConv1(cg); //1. evriþim katmaný öncesi gerekli bellek bölgelerinin tahsisi 

		

		// CPU dan GPU data transferleri
		cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuImagePtr, cg->cpuImagePtr, cg->imageAllocSize); //görüntü datasý transferi
		cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuFeaturePtr, cg->cpuFeaturePtr, cg->featureAllocSize); //0 datasýyla dolu Feature space transferi
		cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuMaskPtr, cg->cpuMaskPtr, cg->maskAllocSize); //maskelerin (aðýrlýklar) transferi
		cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuBatchPtr, cg->cpuBatchPtr, cg->batchWeightSize); //batch aðýrlýklarý transferi


		conv1ExecGPU(cg); //1. evriþim katmanýný koþan GPU köprü fonksiyonu

		showFeatureOnPictureBox(cg->cpuFeaturePtr, cg->featureWidthSize, cg->featureHeightSize, MASK_COUNT_FIRST_LAYER, 0, pictureBox3Click); //picbox üzerinde 1. katman Feature space in gözlemlenmesini saðlayan fonksiyon

		//---------conv2
		setValuesForGpuConv2(cg); //2. evriþim katmaný öncesi gerekli bellek bölgelerinin tahsisi 

		convHidden1ExecGPU(cg);  //1. gizli katmaný koþan GPU köprü fonksiyonu

		showFeatureOnPictureBox(cg->cpuFeaturePtr, cg->featureWidthSize, cg->featureHeightSize, MASK_COUNT_OUTPUT_LAYER, 1, pictureBox4Click); //picbox üzerinde 2. katman Feature space in gözlemlenmesini saðlayan fonksiyon

		//---------Dense1
		setValuesForGpuDense1(cg);  //1. dense katmaný öncesi gerekli bellek bölgelerinin tahsisi 
		dense1ExecGPU(cg);	  //1. dense katmanýný koþan GPU köprü fonksiyonu

		//---------Dense2
		setValuesForGpuDense2(cg);  //2. dense katmaný öncesi gerekli bellek bölgelerinin tahsisi 
		dense2ExecGPU(cg);    //2. dense katmanýný koþan GPU köprü fonksiyonu

		softmax(cg->cpuDensePtr, DENSE_OUTPUT_LAYER);  //softmax iþlemi CPU'da yapýlýr.

		double gpuClock = (double)(clock() - tStart) / CLOCKS_PER_SEC; //süre ölçümü için gerekli

		printGraph(cg, gpuClock);  // form arayüzündeki grafiðin çizimi

		cudaDeviceSynchronize(); //Cuda cihaz senkronizasyonu için


		cpuGpuFree(cg, denseEnum);  // ayrýlan bellek bölgelerinin serbest býrakýlmasý (image, feature, batch gibi dizilar köprü fonksiyonlarda serbest býrakýlmýþtýr.)
		cpuGpuFree(cg, denseWeightEnum);



		cudaError_t result = cudaDeviceSynchronize();
		assert(result == cudaSuccess);
	}

		   //---------------------------------------------------------------------------------
		   //Model2---------------------------------------------------------------------------

		   void setValuesForGpuModel2Conv1(CpuGpuMem* cg) {
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

			   int* cpu_int32 = (int*)cg->cpuImagePtr;
			   for (int i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH; i++) {
				   cpu_int32[i] = ferImages[(ferTextBoxInput * IMAGE_WIDTH * IMAGE_HEIGHT) + i]; //
			   }

			   //for (int i = 0; i < cg->featureWidthSize * cg->featureHeightSize * cg->maskCount; i++) {
				  // cg->cpuFeaturePtr[i] = 0.0;
			   //}

			   for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++) {
				   for (int j = 0; j < cg->maskCount; j++) {
					   cg->cpuMaskPtr[j * MASK_SIZE * MASK_SIZE + i] = convInputLayerWeights[i * cg->maskCount + j];
				   }
			   }

			   for (int i = 0; i < cg->maskCount; i++) {
				   cg->cpuMaskPtr[cg->maskCount * MASK_SIZE * MASK_SIZE + i] = convInputLayerWeights[cg->maskCount * MASK_SIZE * MASK_SIZE + i];
			   }

		   }
		   void setValuesForGpuModel2Conv2(CpuGpuMem* cg) {

			   cg->maskWHSize = MASK_SIZE;
			   cg->maskCount = MASK_COUNT_HIDDEN_LAYER_1;
			   cg->maskDim = MASK_COUNT_FIRST_LAYER;
			   cg->batchWeightSize = cg->maskCount;

			   cpuGpuFree(cg, imageEnum);
			   cpuGpuFree(cg, maskEnum);
			   cpuGpuFree(cg, batchEnum);

			   cpuGpuAlloc(cg, maskEnum, sizeof(float)); //  mask allocation for 2. conv layer
			   cpuGpuAlloc(cg, batchEnum, sizeof(float));

			   //weights resorting
			   int count = 0;
			   for (int i = 0; i < cg->maskWHSize * cg->maskWHSize; i++) {
				   for (int j = 0; j < cg->maskDim; j++) {
					   for (int k = 0; k < cg->maskCount; k++) {
						   cg->cpuMaskPtr[k * cg->maskWHSize * cg->maskWHSize * cg->maskDim + (j * cg->maskWHSize * cg->maskWHSize) + i] = convHiddenLayerWeights_1[count];
						   count++;
					   }
				   }
			   }

			   for (int i = 0; i < cg->maskCount; i++)
			   {
				   cg->cpuMaskPtr[cg->maskCount * cg->maskDim * cg->maskWHSize * cg->maskWHSize + i] =
					   convHiddenLayerWeights_1[cg->maskCount * cg->maskDim * cg->maskWHSize * cg->maskWHSize + i];
			   }
			   cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuMaskPtr, cg->cpuMaskPtr, cg->maskAllocSize);
			   cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuBatchPtr, batchNormWeight_1, cg->batchWeightSize);
		   }
		   void setValuesForGpuModel2Conv3(CpuGpuMem* cg) {
			   cg->maskWHSize = MASK_SIZE;
			   cg->maskCount = MASK_COUNT_HIDDEN_LAYER_2;
			   cg->maskDim = MASK_COUNT_HIDDEN_LAYER_1;
			   cg->batchWeightSize = cg->maskCount;

			   cpuGpuFree(cg, maskEnum);
			   cpuGpuFree(cg, batchEnum);

			   cpuGpuAlloc(cg, maskEnum, sizeof(float)); //  mask allocation for 2. conv layer
			   cpuGpuAlloc(cg, batchEnum, sizeof(float));

			   //weights resorting
			   int count = 0;
			   for (int i = 0; i < cg->maskWHSize * cg->maskWHSize; i++) {
				   for (int j = 0; j < cg->maskDim; j++) {
					   for (int k = 0; k < cg->maskCount; k++) {
						   cg->cpuMaskPtr[k * cg->maskWHSize * cg->maskWHSize * cg->maskDim + (j * cg->maskWHSize * cg->maskWHSize) + i] = convHiddenLayerWeights_2[count];
						   count++;
					   }
				   }
			   }

			   for (int i = 0; i < cg->maskCount; i++)
			   {
				   cg->cpuMaskPtr[cg->maskCount * cg->maskDim * cg->maskWHSize * cg->maskWHSize + i] =
					   convHiddenLayerWeights_2[cg->maskCount * cg->maskDim * cg->maskWHSize * cg->maskWHSize + i];
			   }
			   cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuMaskPtr, cg->cpuMaskPtr, cg->maskAllocSize);
			   cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuBatchPtr, batchNormWeight_2, cg->batchWeightSize);
		   }
		   void setValuesForGpuModel2Conv4(CpuGpuMem* cg) {
			   cg->maskWHSize = MASK_SIZE;
			   cg->maskCount = MASK_COUNT_OUTPUT_LAYER;
			   cg->maskDim = MASK_COUNT_HIDDEN_LAYER_2;
			   cg->batchWeightSize = cg->maskCount;

			   cpuGpuFree(cg, maskEnum);
			   cpuGpuFree(cg, batchEnum);

			   cpuGpuAlloc(cg, maskEnum, sizeof(float)); //  mask allocation for 2. conv layer
			   cpuGpuAlloc(cg, batchEnum, sizeof(float));

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
			   cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuBatchPtr, batchNormWeight_3, cg->batchWeightSize);
		   }
		   void setValuesForGpuModel2Dense1(CpuGpuMem* cg) {

			   cg->denseInputSize = cg->maskCount * cg->featureWidthSize * cg->featureHeightSize;
			   cg->denseOutputSize = DENSE_HIDDEN_LAYER_1;

			   cpuGpuAlloc(cg, denseEnum, sizeof(float));
			   cpuGpuAlloc(cg, denseWeightEnum, sizeof(float));
			   cpuGpuFree(cg, maskEnum);
			   cpuGpuFree(cg, batchEnum);
			   cpuGpuFree(cg, featureEnum);

			   cg->batchWeightSize = cg->denseOutputSize;
			   cpuGpuAlloc(cg, batchEnum, sizeof(float));

			   cudaMemset(cg->gpuDensePtr, 0, cg->denseOutputAllocSize);

			   cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuDenseWeightPtr, denseHiddenLayerWeights_1, cg->denseWeightAllocSize);
			   cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuBatchPtr, batchNormWeight_4, cg->batchWeightSize);
		   }
		   void setValuesForGpuModel2Dense2(CpuGpuMem* cg) {

			   cg->denseInputSize = DENSE_HIDDEN_LAYER_1;
			   cg->denseOutputSize = DENSE_HIDDEN_LAYER_2;


			   cpuGpuFree(cg, denseWeightEnum);
			   cpuGpuAlloc(cg, denseWeightEnum, sizeof(float));

			   cpuGpuFree(cg, batchEnum);
			   cg->batchWeightSize = cg->denseOutputSize;
			   cpuGpuAlloc(cg, batchEnum, sizeof(float));


			   cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuDenseWeightPtr, denseHiddenLayerWeights_2, cg->denseWeightAllocSize);
			   cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuBatchPtr, batchNormWeight_5, cg->batchWeightSize);
		   }
		   void setValuesForGpuModel2Dense3(CpuGpuMem* cg) {

			   cg->denseInputSize = DENSE_HIDDEN_LAYER_2;
			   cg->denseOutputSize = DENSE_OUTPUT_LAYER;

			   cpuGpuFree(cg, batchEnum);
			   cpuGpuFree(cg, denseWeightEnum);
			   cpuGpuAlloc(cg, denseWeightEnum, sizeof(float));

			   cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuDenseWeightPtr, denseOutputLayerWeights, cg->denseWeightAllocSize);
		   }

	private: System::Void cudaRunModel2ToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {


		clock_t tStart = clock();
		CpuGpuMem new_cg;
		CpuGpuMem* cg = &new_cg;

		//----forConv1
		setValuesForGpuModel2Conv1(cg); // func

		cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuImagePtr, cg->cpuImagePtr, cg->imageAllocSize); // host to device
		cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuFeaturePtr, cg->cpuFeaturePtr, cg->featureAllocSize);
		cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuMaskPtr, cg->cpuMaskPtr, cg->maskAllocSize);
		cpuGpuMemCopy(cudaMemcpyHostToDevice, cg, cg->gpuBatchPtr, batchNormWeight, cg->batchWeightSize);

		//---------conv2
		model2Conv1ExecGPU(cg); //conv1

		showFeatureOnPictureBox(cg->cpuFeaturePtr, cg->featureWidthSize, cg->featureHeightSize, MASK_COUNT_FIRST_LAYER, 0, pictureBox3Click);


		//---------conv2
		setValuesForGpuModel2Conv2(cg); // conv2

		model2Conv2ExecGpu(cg);

		showFeatureOnPictureBox(cg->cpuFeaturePtr, cg->featureWidthSize, cg->featureHeightSize, MASK_COUNT_HIDDEN_LAYER_1, 1, pictureBox4Click);

		//---------conv3
		setValuesForGpuModel2Conv3(cg);
		model2Conv3ExecGpu(cg);

		showFeatureOnPictureBox(cg->cpuFeaturePtr, cg->featureWidthSize, cg->featureHeightSize, MASK_COUNT_HIDDEN_LAYER_2, 2, pictureBox5Click);

		//---------conv4
		setValuesForGpuModel2Conv4(cg);
		model2Conv4ExecGpu(cg);

		showFeatureOnPictureBox(cg->cpuFeaturePtr, cg->featureWidthSize, cg->featureHeightSize, MASK_COUNT_HIDDEN_LAYER_2, 3, pictureBox6Click);

		//---------dense1
		setValuesForGpuModel2Dense1(cg);
		model2Dense1ExecGPU(cg);

		//---------dense2
		setValuesForGpuModel2Dense2(cg);
		model2Dense2ExecGPU(cg);



		//---------dense3
		setValuesForGpuModel2Dense3(cg);
		model2Dense3ExecGPU(cg);

		softmax(cg->cpuDensePtr, DENSE_OUTPUT_LAYER);

		float max = 0.0;
		for (int i = 0; i < DENSE_OUTPUT_LAYER; i++) {
			if (max < cg->cpuDensePtr[i]) {
				max = cg->cpuDensePtr[i];
				rank1 = i;
			}
		}
		float max2 = 0.0;
		for (int i = 0; i < DENSE_OUTPUT_LAYER; i++) {
			if (cg->cpuDensePtr[i] > max2 && i != rank1) {
				max2 = cg->cpuDensePtr[i];
				rank2 = i;
			}
		}

		double gpuClock = (double)(clock() - tStart) / CLOCKS_PER_SEC;

		printGraph(cg, gpuClock);

		cpuGpuFree(cg, denseEnum);
		cpuGpuFree(cg, denseWeightEnum);
		
		cudaError_t result = cudaDeviceSynchronize();
		assert(result == cudaSuccess);

	}

		   vector<Rect> detectAndDraw(Mat& img, CascadeClassifier& cascade) {
			   vector<Rect> faces; // yüz datalarý için
			   Mat gray; 

			   cvtColor(img, gray, COLOR_BGR2GRAY); // görüntünün griye çevrilmesi
			   equalizeHist(gray, gray); // histogram eþitleme

			   cascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, cv::Size(FACE_DETECTION_SCALE, FACE_DETECTION_SCALE)); // yüzlerin algýlanmasý

			   System::Drawing::Bitmap^ b;
			   System::IntPtr ptr(img.ptr());
			   b = gcnew System::Drawing::Bitmap(img.cols, img.rows, img.step, System::Drawing::Imaging::PixelFormat::Format24bppRgb, ptr);
			   pictureBox1->Image = b; //kameranýn pictureBox1 de gösterilmesi
			   return faces;
		   }
		   void printFaces(vector<Rect> faces, Mat frame) {
			   int x, y, width, height; 

			   x = faces[0].x; //Yüzün konumlarý iþaretlenir
			   y = faces[0].y;
			   width = faces[0].width;
			   height = faces[0].height;

			   Rect myROI(x, y, width, height); // yüz fotoðraftan ayýklanýr
			   Mat face = frame(myROI);
			   Mat face2;
			   Color c;

			   resize(face, face2, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), 1, 1, INTER_AREA); // yüz 48x48 boyutuna çevilir.

			   System::Drawing::Bitmap^ b;
			   System::IntPtr ptr(face2.ptr());
			   b = gcnew System::Drawing::Bitmap(face2.cols, face2.rows, face2.step, System::Drawing::Imaging::PixelFormat::Format24bppRgb, ptr);


			   Bitmap^ surface = gcnew Bitmap(IMAGE_WIDTH, IMAGE_HEIGHT);
			   pictureBox2->Image = surface; //pictureBox2 ye yüz basýlýr.
			   for (int row = 0; row < IMAGE_HEIGHT; row++)
			   {
				   for (int column = 0; column < IMAGE_WIDTH; column++)
				   {
					   c = b->GetPixel(column, row);
					   int index = (0.3 * c.R + 0.59 * c.G + 0.11 * c.B);
					   c = Color::FromArgb(index, index, index);
					   ferImages[row * IMAGE_WIDTH + column] = index; //fer images datasýna yüz fotoðrafý atanýr.
					   surface->SetPixel(column, row, c);
				   }
			   }
		   }

	private: System::Void button3_Click(System::Object^ sender, System::EventArgs^ e) { //kamera fonksiyonu
		openCamera = !openCamera;
		button4->Enabled = !openCamera;

		if (uploadWeights == 1 && openCamera == 1) {
			delete[] ferImages;
			ferImages = new BYTE[IMAGE_WIDTH * IMAGE_HEIGHT]; 
			VideoCapture capture;
			Mat frame, image;
			vector<Rect> faces;

			CascadeClassifier cascade;
			double scale = 1;

			cascade.load(HAAR_CASCADE_PATH);

			label1->Visible = true;
			label2->Visible = true;
			label3->Visible = true;
			label4->Visible = true;
			label5->Visible = true;
			label6->Visible = true;
			label7->Visible = true;

			capture.open(0);
			if (capture.isOpened())
			{
				// Capture frames from video and detect faces
				richTextBox1->Text += "Emotion Classification Started....\n";

				while (openCamera) // kamera açýlýr
				{
					clock_t tStart = clock();

					capture >> frame;
					if (frame.empty())
						break;
					Mat frame1 = frame.clone();
					faces = detectAndDraw(frame1, cascade);
					char c = (char)waitKey(25);
					if (openCamera == 0) {
						Bitmap^ surface;
						pictureBox1->Image = surface;
						pictureBox2->Image = surface;
						pictureBox3->Image = surface;
						pictureBox4->Image = surface;
						pictureBox5->Image = surface;
						pictureBox6->Image = surface;
						label1->Visible = false;
						label2->Visible = false;
						label3->Visible = false;
						label4->Visible = false;
						label5->Visible = false;
						label6->Visible = false;
						label7->Visible = false;
						break;
					}
					if (faces.size() == 1) {
						printFaces(faces, frame1);
						ferTextBoxInput = 0;
						if (modelId == 1) {
							cudaRunToolStripMenuItem_Click(sender, e);
						}
						if (modelId == 2) {
							cudaRunModel2ToolStripMenuItem_Click(sender, e);
						}
					}
					double cpuClock = (double)(clock() - tStart) / CLOCKS_PER_SEC;
					int fps = 1 / cpuClock;
					label5->Text = "Fps: " + fps;
				}

			}
			else
				cout << "Could not Open Camera";
		}
		else {
			if (openCamera == 0 && uploadWeights == 1) {
				richTextBox1->Text += "Emotion Classification Stopped...\n";
				label5->Visible = false;
			}
			else if (uploadWeights == 0) {
				MessageBox::Show("Please Upload Weights");
			}
		}
	}
	private: System::Void button4_Click(System::Object^ sender, System::EventArgs^ e) {
		openVideo = !openVideo;
		button3->Enabled = !openVideo;
		if (uploadWeights == 1 && openVideo == 1) {

			Stream^ mystream;
			OpenFileDialog^ openFileDialog1 = gcnew OpenFileDialog;
			System::String^ strfilename;

			openFileDialog1->InitialDirectory = "";
			openFileDialog1->Filter = "Video files |*.avi; *.m4v; *.mkv; *.mov; *.mp4; *.mp4v;*.mpeg; *.mpeg1; *.mpeg2; *.mpeg4;*.rec; *.webm; *.dat; ";
			openFileDialog1->FilterIndex = 5;
			openFileDialog1->RestoreDirectory = true;

			if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK)
			{
				strfilename = openFileDialog1->InitialDirectory + openFileDialog1->FileName;
			}
			if (strfilename == nullptr) {
				MessageBox::Show(" Errors occured while loading video.", "ERROR",
					MessageBoxButtons::OK, MessageBoxIcon::Error);
				openVideo = 0;
				return;
			}
			IntPtr ip = Marshal::StringToHGlobalAnsi(strfilename);
			const char* inputStr = static_cast<const char*>(ip.ToPointer());
			std::string input(inputStr);


			delete[] ferImages;
			ferImages = new BYTE[IMAGE_WIDTH * IMAGE_HEIGHT];
			VideoCapture capture(input);
			Mat frame, image;
			vector<Rect> faces;

			CascadeClassifier cascade;
			double scale = 1;
			int fpsWait = 0;

			cascade.load(HAAR_CASCADE_PATH);
			int videoFps = capture.get(CAP_PROP_FPS);

			if (capture.isOpened())
			{
				// Capture frames from video and detect faces
				richTextBox1->Text += "Emotion Classification Started....\n";


				label1->Visible = true;
				label2->Visible = true;
				label3->Visible = true;
				label4->Visible = true;
				label5->Visible = true;
				label6->Visible = true;
				label7->Visible = true;
				while (openVideo)
				{
					clock_t tStart = clock();

					capture.read(frame);
					if (frame.empty())
						break;
					Mat frame1 = frame.clone();
					faces = detectAndDraw(frame1, cascade);
					if(modelId == 1){
						fpsWait = 6;
					}
					if (modelId == 2) {
						fpsWait = 25; // for fps wait
					}
					char c = (char)waitKey(1000 / (videoFps + fpsWait));
					if (openVideo == 0) {
						Bitmap^ surface;
						pictureBox1->Image = surface;
						pictureBox2->Image = surface;
						pictureBox3->Image = surface;
						pictureBox4->Image = surface;
						pictureBox5->Image = surface;
						pictureBox6->Image = surface;
						label1->Visible = false;
						label2->Visible = false;
						label3->Visible = false;
						label4->Visible = false;
						label5->Visible = false;
						label6->Visible = false;
						label7->Visible = false;
						break;
					}
					if (faces.size() == 1) {
						printFaces(faces, frame1);
						ferTextBoxInput = 0;
						if (modelId == 1) {
							cudaRunToolStripMenuItem_Click(sender, e);
						}
						if (modelId == 2) {
							cudaRunModel2ToolStripMenuItem_Click(sender, e);
						}
					}
					double cpuClock = (double)(clock() - tStart) / CLOCKS_PER_SEC;
					int  fps = 1 / cpuClock;
					label5->Text = "Fps: " + fps;
				}

			}
			else
				cout << "Could not Open Camera";
		}
		else {
			if (openCamera == 0 && uploadWeights == 1) {
				richTextBox1->Text += "Emotion Classification Stopped...\n";
				label5->Visible = false;
			}
			else if (uploadWeights == 0) {
				MessageBox::Show("Please Upload Weights");
			}
		}
	}
	private: System::Void pictureBox3_Click(System::Object^ sender, System::EventArgs^ e) {
		pictureBox3Click++;
	}
	private: System::Void pictureBox4_Click(System::Object^ sender, System::EventArgs^ e) {
		pictureBox4Click++;
	}
	private: System::Void conv1FeaturesToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		pictureBox3->Visible = !(pictureBox3->Visible);
		label1->Visible = !(label1->Visible);
	}
	private: System::Void conv2FeaturesToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		pictureBox4->Visible = !(pictureBox4->Visible);
		label2->Visible = !(label2->Visible);
	}
	private: System::Void richTextBox1_TextChanged(System::Object^ sender, System::EventArgs^ e) {
		// set the current caret position to the end
		richTextBox1->SelectionStart = richTextBox1->Text->Length;
		// scroll it automatically
		richTextBox1->ScrollToCaret();
	}
	private: System::Void MyForm_FormClosing(System::Object^ sender, System::Windows::Forms::FormClosingEventArgs^ e) {
		if (openCamera) {
			openCamera = 0;
		}
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

private: System::Void button5_Click(System::Object^ sender, System::EventArgs^ e) {
	richTextBox1->Text = "";
}
private: System::Void pictureBox5_Click(System::Object^ sender, System::EventArgs^ e) {
	pictureBox5Click++;
}
private: System::Void pictureBox6_Click(System::Object^ sender, System::EventArgs^ e) {
	pictureBox5Click++;
}
private: System::Void conv3FeaturesToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
	pictureBox5->Visible = !(pictureBox5->Visible);
	label6->Visible = !(label6->Visible);
}
private: System::Void conv4FeaturesToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
	pictureBox6->Visible = !(pictureBox6->Visible);
	label7->Visible = !(label7->Visible);
}
private: System::Void button6_Click(System::Object^ sender, System::EventArgs^ e) {
	//accuracy
	ferTextBoxInput = 0;
	for (int i = 0; i < lineCount; i++) {
		cudaRunModel2ToolStripMenuItem_Click(sender,e);
		ferTextBoxInput++;
		if (rank1 == emotionLabel[i]) {
			accTrue++;
		}
		else if (rank2 == emotionLabel[i]) {
			acc2True++;
		}
	}

	float result = (int)(((float)accTrue / (float)lineCount) * 1000);
	result = result / 10;
	richTextBox1->Text +=  "Rank1 Acc : %" + result + "\n";

	result = (int)(((float)(accTrue+acc2True) / (float)lineCount) * 1000);
	result = result / 10;
	richTextBox1->Text += "Rank2 Acc : %" + result + "\n";
}
};
}
