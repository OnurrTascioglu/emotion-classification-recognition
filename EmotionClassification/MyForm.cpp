#include "MyForm.h"

using namespace System;
using namespace System::Windows::Forms;

// sea


[STAThread]
void Main(array<String^>^ args)
{
    Application::EnableVisualStyles();
    Application::SetCompatibleTextRenderingDefault(false);
    EmotionClassification::MyForm form;
    Application::Run(% form);
}