#pragma once
#include <windows.h>
#include "image.h"
#include "Test.h"
#include "atlstr.h"
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
#define MASK_COUNT 4
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
		float* conv2d;
		float* conv2d_1;
		int ferTextBoxInput = 0;


	private: System::Windows::Forms::ToolStripMenuItem^ fer2013DSToolStripMenuItem;
	private: System::Windows::Forms::TextBox^ textBox1;
	private: System::Windows::Forms::ToolStripMenuItem^ testToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ runToolStripMenuItem;
	private: System::Windows::Forms::PictureBox^ pictureBox2;
	private: System::Windows::Forms::PictureBox^ pictureBox3;
	private: System::Windows::Forms::PictureBox^ pictureBox4;
	private: System::Windows::Forms::PictureBox^ pictureBox5;
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
			this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
			this->dosyaToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->openToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->weightsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->pictureToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->fer2013DSToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->testToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->runToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->richTextBox1 = (gcnew System::Windows::Forms::RichTextBox());
			this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->pictureBox2 = (gcnew System::Windows::Forms::PictureBox());
			this->pictureBox3 = (gcnew System::Windows::Forms::PictureBox());
			this->pictureBox4 = (gcnew System::Windows::Forms::PictureBox());
			this->pictureBox5 = (gcnew System::Windows::Forms::PictureBox());
			this->menuStrip1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox2))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox3))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox4))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox5))->BeginInit();
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
			this->testToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) { this->runToolStripMenuItem });
			this->testToolStripMenuItem->Name = L"testToolStripMenuItem";
			this->testToolStripMenuItem->Size = System::Drawing::Size(49, 24);
			this->testToolStripMenuItem->Text = L"Test";
			// 
			// runToolStripMenuItem
			// 
			this->runToolStripMenuItem->Name = L"runToolStripMenuItem";
			this->runToolStripMenuItem->Size = System::Drawing::Size(117, 26);
			this->runToolStripMenuItem->Text = L"Run";
			this->runToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::runToolStripMenuItem_Click);
			// 
			// openFileDialog1
			// 
			this->openFileDialog1->FileName = L"openFileDialog1";
			// 
			// richTextBox1
			// 
			this->richTextBox1->Location = System::Drawing::Point(1159, 281);
			this->richTextBox1->Name = L"richTextBox1";
			this->richTextBox1->Size = System::Drawing::Size(244, 232);
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
			this->button1->Text = L"ShowImage";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &MyForm::button1_Click);
			// 
			// pictureBox2
			// 
			this->pictureBox2->Location = System::Drawing::Point(424, 32);
			this->pictureBox2->Name = L"pictureBox2";
			this->pictureBox2->Size = System::Drawing::Size(207, 192);
			this->pictureBox2->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pictureBox2->TabIndex = 5;
			this->pictureBox2->TabStop = false;
			// 
			// pictureBox3
			// 
			this->pictureBox3->Location = System::Drawing::Point(637, 32);
			this->pictureBox3->Name = L"pictureBox3";
			this->pictureBox3->Size = System::Drawing::Size(207, 192);
			this->pictureBox3->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pictureBox3->TabIndex = 6;
			this->pictureBox3->TabStop = false;
			// 
			// pictureBox4
			// 
			this->pictureBox4->Location = System::Drawing::Point(424, 230);
			this->pictureBox4->Name = L"pictureBox4";
			this->pictureBox4->Size = System::Drawing::Size(207, 192);
			this->pictureBox4->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pictureBox4->TabIndex = 7;
			this->pictureBox4->TabStop = false;
			// 
			// pictureBox5
			// 
			this->pictureBox5->Location = System::Drawing::Point(637, 230);
			this->pictureBox5->Name = L"pictureBox5";
			this->pictureBox5->Size = System::Drawing::Size(207, 192);
			this->pictureBox5->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pictureBox5->TabIndex = 8;
			this->pictureBox5->TabStop = false;
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(8, 16);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1451, 573);
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


				for (int row = 0; row < IMAGE_HEIGHT; row++) {
					for (int col = 0; col < IMAGE_WIDTH; col++) {
						for (int y = 0; y < resizeY; y++) {
							for (int x = 0; x < resizeX; x++) {
								mean += raw_intensity[Width * row * (int)resizeY + Width * y + col * (int)resizeX + x];
								if (col % count == 0) {

								}
							}
						}
						mean = mean / (resizeY * resizeX);
						buffer[IMAGE_WIDTH * row + col] = round(mean);
						mean = 0.0;
					}
				}
				Bitmap^ surface = gcnew Bitmap(IMAGE_WIDTH, IMAGE_HEIGHT);
				pictureBox1->Image = surface;

				Color c;

				for (int row = 0; row < IMAGE_HEIGHT; row++)
					for (int col = 0; col < IMAGE_WIDTH; col++) {
						c = Color::FromArgb(buffer[row * IMAGE_WIDTH + col], buffer[row * IMAGE_WIDTH + col], buffer[row * IMAGE_WIDTH + col]);
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
				runToolStripMenuItem_Click(sender,e);
			}
		}
		else {
			MessageBox::Show("Not a number");
		}

	}
	private: System::Void runToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {

		conv2d = new float[100];
		int index = 0;

		// File pointer
		fstream fin;

		IntPtr ip = Marshal::StringToHGlobalAnsi(WEIGHT_PATH);
		const char* inputStr = static_cast<const char*>(ip.ToPointer());
		std::string input(inputStr);

		string filePath = input + "conv2d.csv";

		fin.open(filePath , ios::in);
		string line;

		while (!fin.eof()) {
			std::getline(fin, line);
			if (line != "") {
				conv2d[index] = stof(line);
				index++;
			}
		}

		int size = (IMAGE_WIDTH - MASK_SIZE + 1) * (IMAGE_HEIGHT - MASK_SIZE + 1) * MASK_COUNT;
		int sizeW = (IMAGE_WIDTH - MASK_SIZE + 1);
		int sizeH = (IMAGE_HEIGHT - MASK_SIZE + 1);
		
		float* fResult = new float[size];

		fResult = conv1(ferImages, conv2d, IMAGE_WIDTH, IMAGE_HEIGHT, MASK_SIZE, MASK_COUNT, ferTextBoxInput, sizeW, sizeH);
		batchNormalization(fResult, sizeW, sizeH, MASK_COUNT);
		reLU(fResult, sizeW, sizeH, MASK_COUNT);
		maxPooling(fResult, sizeW, sizeH, MASK_COUNT, 2, 2);



		BYTE* result = new BYTE[size];
		int tempo = 0;


		for (int i = 0; i < size; i++) {
			fResult[i] = fResult[i] * 128 +30;
			tempo = (int)fResult[i];
			if (tempo < 0)
				result[i] = 0;

			else if (tempo > 255)
				result[i] = 255;

			else
				result[i] = tempo;
		}


		Bitmap^ surface = gcnew Bitmap((IMAGE_WIDTH - MASK_SIZE + 1), (IMAGE_HEIGHT - MASK_SIZE + 1));
		pictureBox2->Image = surface;

		Bitmap^ surface2 = gcnew Bitmap((IMAGE_WIDTH - MASK_SIZE + 1), (IMAGE_HEIGHT - MASK_SIZE + 1));
		pictureBox3->Image = surface2;

		Bitmap^ surface3 = gcnew Bitmap((IMAGE_WIDTH - MASK_SIZE + 1), (IMAGE_HEIGHT - MASK_SIZE + 1));
		pictureBox4->Image = surface3;

		Bitmap^ surface4 = gcnew Bitmap((IMAGE_WIDTH - MASK_SIZE + 1), (IMAGE_HEIGHT - MASK_SIZE + 1));
		pictureBox5->Image = surface4;


		Color c;

		int rowx = (IMAGE_HEIGHT - MASK_SIZE + 1);
		int rowy = (IMAGE_WIDTH - MASK_SIZE + 1);

		for (int row = 0; row < rowx; row++)
		{

			for (int column = 0; column < rowy; column++)
			{
				c = Color::FromArgb(result[row * rowy + column], result[row * rowy + column], result[row * rowy + column]);
				surface->SetPixel(column, row, c);

				c = Color::FromArgb(result[(rowx * rowy) + row * rowy + column], result[(rowx * rowy) + row * rowy + column], result[(rowx * rowy) + row * rowy + column]);
				surface2->SetPixel(column, row, c);

				c = Color::FromArgb(result[(rowx * rowy * 2) + row * rowy + column], result[(rowx * rowy * 2) + row * rowy + column], result[(rowx * rowy * 2) + row * rowy + column]);
				surface3->SetPixel(column, row, c);

				c = Color::FromArgb(result[(rowx * rowy * 3) + row * rowy + column], result[(rowx * rowy * 3) + row * rowy + column], result[(rowx * rowy * 3) + row * rowy + column]);
				surface4->SetPixel(column, row, c);
			}
		}


		delete[] result;
		delete[] fResult;

	}
	private: System::Void MyForm_FormClosing(System::Object^ sender, System::Windows::Forms::FormClosingEventArgs^ e) {
		delete[] ferImages;
		delete[] emotionLabel;
		delete[] bmpColoredImage;
		delete[] raw_intensity;
		delete[] conv2d;
		delete[] conv2d_1;
	}
	};
}
