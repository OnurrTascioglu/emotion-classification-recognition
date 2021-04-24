#pragma once
#include <windows.h>
#include "image.h"
#include "atlstr.h"
#include <iostream>
#include <fstream>
#include <msclr\marshal.h>
#include <vector>
#include <istream>
#include <string>
#include <sstream>
#define IMAGE_WIDTH 48
#define IMAGE_HEIGHT 48
#define TOTAL_IMAGE 35888
#define EMOTION_COUNT 7


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
		BYTE* buffer;
	private: System::Windows::Forms::ToolStripMenuItem^ fer2013DSToolStripMenuItem;

		   BYTE* raw_intensity;
	private: System::Windows::Forms::TextBox^ textBox1;
	private: System::Windows::Forms::Button^ button1;
		   BYTE* ferImages;
		   int lineCount = 0;
		   BYTE* emotionLabel;
		
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
		System::ComponentModel::Container ^components;

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
			this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->richTextBox1 = (gcnew System::Windows::Forms::RichTextBox());
			this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->menuStrip1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			this->SuspendLayout();
			// 
			// menuStrip1
			// 
			this->menuStrip1->ImageScalingSize = System::Drawing::Size(20, 20);
			this->menuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) { this->dosyaToolStripMenuItem });
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
			// openFileDialog1
			// 
			this->openFileDialog1->FileName = L"openFileDialog1";
			// 
			// richTextBox1
			// 
			this->richTextBox1->Location = System::Drawing::Point(410, 31);
			this->richTextBox1->Name = L"richTextBox1";
			this->richTextBox1->Size = System::Drawing::Size(901, 556);
			this->richTextBox1->TabIndex = 1;
			this->richTextBox1->Text = L"";
			// 
			// pictureBox1
			// 
			this->pictureBox1->Location = System::Drawing::Point(13, 32);
			this->pictureBox1->Name = L"pictureBox1";
			this->pictureBox1->Size = System::Drawing::Size(391, 373);
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
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(8, 16);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1451, 573);
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
		long Size;

		if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			str = openFileDialog1->FileName;
			input = (LPCTSTR)str;

			//BMP Image Reading
			buffer = LoadBMP(&Width, &Height, &Size, input);
			raw_intensity = ConvertBMPToIntensity(buffer, Width, Height); // BMP Gray picture

			pictureBox1->Width = Width;
			pictureBox1->Height = Height;

			//Display Gray Image into pictureBox2
			Bitmap^ surface = gcnew Bitmap(pictureBox1->Width, pictureBox1->Height);
			pictureBox1->Image = surface;
			Color c;

			for (int row = 0; row < Height; row++)
				for (int col = 0; col < Width; col++) {
					c = Color::FromArgb(*(raw_intensity + row * Width + col), *(raw_intensity + row * Width + col), *(raw_intensity + row * Width + col));
					surface->SetPixel(col, row, c);
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


			vector<string> row;
			string line, word;

			int asd=0;
			int k = 0, count = 0;
			bool lineBool = 0;
			int imageIndex = 0;
			int ferIndex = 0;


			while (!fin.eof()) {

				row.clear();

				std::getline(fin, line);

				k = 0;
				count = 0;

				if (lineBool == 1) {
					for (int i = 0; i < line.length(); i++) {
						if (line[i].Equals(',')) {
							if (count == 0) {
								word = line.substr(k, i - k);
								emotionLabel[lineCount] = stoi(word);
								richTextBox1->Text += emotionLabel[lineCount] + " ";
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
private: System::Void MyForm_FormClosing(System::Object^ sender, System::Windows::Forms::FormClosingEventArgs^ e) {
	delete[] ferImages;
	delete[] emotionLabel;

}
private: System::Void button1_Click(System::Object^ sender, System::EventArgs^ e) {
	Int32 myInt = 0;


	if (System::Text::RegularExpressions::Regex::IsMatch(textBox1->Text,
		"^[1-9]\d*$"))
	{
		myInt = System::Convert::ToInt32(textBox1->Text);
		if (myInt > lineCount-1) {
			MessageBox::Show("Number can't be higher than " + (lineCount - 1));
		}
		else if (myInt < 0) {
			MessageBox::Show("Please enter positive number");
		}
		else {
			Bitmap^ surface = gcnew Bitmap(IMAGE_WIDTH,IMAGE_HEIGHT);
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
		}
	}
	else{
		MessageBox::Show("Not a number");
	}

}
};
}
