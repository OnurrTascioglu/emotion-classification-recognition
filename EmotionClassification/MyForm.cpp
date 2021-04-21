#include "MyForm.h"

using namespace System;
using namespace System::Windows::Forms;

// sea waysss


[STAThread]
void Main(array<String^>^ args)
{
    Application::EnableVisualStyles();
    Application::SetCompatibleTextRenderingDefault(false);
    EmotionClassification::MyForm form;
    Application::Run(% form);
}