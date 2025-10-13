PYCHARM ENVIRONMENT SETUP WITH CUDA + CUDNN + OLLAMA (WINDOWS)
=====================================================================

This guide explains how to recreate your Python environment with NVIDIA CUDA/cuDNN,
SAPI5, Edge TTS, and Ollama on Windows, using PyCharm. It also covers how to download
and manage GGUF models for Ollama.

---------------------------------------------------------------------
1. CREATE A NEW VIRTUAL ENVIRONMENT IN PYCHARM
---------------------------------------------------------------------

1. In PyCharm, go to:
   File → Settings → Project → Python Interpreter
2. Click the gear icon (⚙️) → Add Interpreter → Virtualenv.
3. Choose "New environment" and click OK.
4. PyCharm will create a .venv folder in your project.

---------------------------------------------------------------------
2. INSTALL PACKAGES FROM requirements.txt
---------------------------------------------------------------------

If you already have a requirements.txt file from a previous working environment:

1. Open the Terminal at the bottom of PyCharm.
2. Run:
       pip install --upgrade pip
       pip install -r requirements.txt

ALTERNATIVE (GUI):
1. File → Settings → Project → Python Interpreter
2. Click "Install from Requirements" (top right)
3. Select requirements.txt and install.

---------------------------------------------------------------------
3. INSTALL OLLAMA (FOR GGUF MODELS)
---------------------------------------------------------------------

Ollama lets you run local AI models (GGUF format) easily.

1. Download the Windows installer:
   https://ollama.com/download

2. Run the installer and restart your PC (if prompted).

3. Test it by opening a terminal (PowerShell or CMD):
       ollama run llama3

4. (Optional) To start Ollama silently in the background:
       Start-Process -FilePath "ollama" -ArgumentList "serve" `
         -WindowStyle Hidden `
         -RedirectStandardOutput "$env:LOCALAPPDATA\Ollama\serve.out.log" `
         -RedirectStandardError  "$env:LOCALAPPDATA\Ollama\serve.err.log"

5. You can now use Ollama from your Python Tkinter or Gradio apps.

---------------------------------------------------------------------
4. DOWNLOADING AND MANAGING GGUF MODELS
---------------------------------------------------------------------

There are two main ways to get models:

A) LET OLLAMA HANDLE IT AUTOMATICALLY
--------------------------------------
1. Open PowerShell or CMD.
2. Run:
       ollama pull llama3
   or
       ollama run mistral

3. Ollama will download and store the model under:
       C:\Users\<YourUsername>\.ollama\models

You can then use the model directly by name in your application.

B) MANUAL MODEL MANAGEMENT (C:\models)
---------------------------------------
1. Create a directory for your models:
       C:\models

2. Download a GGUF model file manually from sources like:
   - https://ollama.com/library
   - https://huggingface.co

3. Place the downloaded file into:
       C:\models\<modelname>.gguf

4. Create a file named "Modelfile" in the same folder with the content:
       FROM ./<modelname>.gguf

5. Build the model into Ollama's registry:
       cd C:\models
       ollama create mymodel -f Modelfile

6. Test it:
       ollama run mymodel

Once created, you can use the model name ("mymodel") in Python or the terminal.
This method avoids redownloading large models and makes them easy to manage.

---------------------------------------------------------------------
5. EXAMPLE PYTHON CALL TO OLLAMA
---------------------------------------------------------------------

Example of using an Ollama model from Python:

    import requests

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mymodel", "prompt": "Hello from Ollama!"}
    )

    for line in response.iter_lines():
        if line:
            print(line.decode())

---------------------------------------------------------------------
6. COPY CUDA/cuDNN DLLS TO THE INTERPRETER FOLDER
---------------------------------------------------------------------

Windows loads DLLs from the interpreter folder first. Copy the NVIDIA DLLs here once
per environment to avoid PATH issues.

Run in PowerShell:
    Copy-Item "$env:VIRTUAL_ENV\Lib\site-packages\nvidia\*\bin\*.dll" "$env:VIRTUAL_ENV\Scripts" -Force

This makes the environment portable and stable.

---------------------------------------------------------------------
7. CREATE requirements.txt
---------------------------------------------------------------------

From your working project:

1. Open Terminal in PyCharm.
2. Run:
       pip freeze > requirements.txt

You can also export from GUI:
    File → Settings → Project → Python Interpreter → ⚙️ → Export to requirements.txt

---------------------------------------------------------------------
8. RUN YOUR PROJECT
---------------------------------------------------------------------

Simply press RUN (▶) in PyCharm.
Because the CUDA/cuDNN DLLs are already in .venv\Scripts,
they’ll always be found without modifying PATH.

---------------------------------------------------------------------
OPTIONAL: AUTOMATE SETUP WITH A SCRIPT
---------------------------------------------------------------------

Create setup_env.ps1 in your project root:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    Copy-Item "$env:VIRTUAL_ENV\Lib\site-packages\nvidia\*\bin\*.dll" "$env:VIRTUAL_ENV\Scripts" -Force

Run this script once to fully prepare a new environment.

---------------------------------------------------------------------
SUMMARY
---------------------------------------------------------------------

- requirements.txt lets you quickly recreate environments.
- Copying CUDA DLLs prevents PATH problems.
- Ollama allows running GGUF models locally.
- Models can be stored in C:\models or pulled automatically.
- Works cleanly inside PyCharm GUI or Terminal.
- Easily portable to another machine.
- Keep your requirements.txt in version control (Git).

