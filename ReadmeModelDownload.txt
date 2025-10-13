Tkinter Speech Assistant — Model Management Guide (2025)
===========================================================

This guide explains how to download, import, and switch models for your Tkinter-based AI speech assistant.
Your assistant now runs models through Ollama, making model management simpler, faster, and more flexible.

------------------------------------------------------------
1. How the System Works
------------------------------------------------------------

We use  Ollama model tags, for example:
"qwen_model_path": "deepseek-r1-chat"

This gives you:
- Clean configuration without hard-coded file paths
- Multiple personalities (e.g. “math” mode vs “chat” mode)
- Easier model swapping and versioning
- Central management of models

------------------------------------------------------------
2. JSON Configuration File
------------------------------------------------------------

Your main configuration is stored in:
config.json

Example config.json:

{
  "sample_rate": 16000,
  "force_out_samplerate": 48000,
  "voice": "en-GB-RyanNeural",
  "rate": "-10%",
  "out_wav": "out/last_reply.wav",

  "whisper_device": "cuda",
  "whisper_model": "base.en",

  "qwen_model_path": "deepseek-r1-chat",
  "qwen_temperature": 0.6,
  "qwen_max_tokens": 1024,
  "qwen_system_prompt": "You are Zen. Speak clearly, use LaTeX for equations, and never introduce yourself.",

  "latex_text_pt": 12,
  "latex_math_pt": 8
}

Key fields:
- qwen_model_path: Name of the Ollama model tag to load
- qwen_temperature: Model creativity (0.2–0.8 typical)
- qwen_max_tokens: Max output length
- qwen_system_prompt: Optional runtime system prompt override
- whisper_model: Speech-to-text model used for ASR
- voice: Text-to-speech voice for spoken output

------------------------------------------------------------
3. Recommended Models
------------------------------------------------------------

qwen2.5:7b-instruct      — General purpose
qwen2.5-math             — Math / LaTeX output (custom tag)
deepseek-r1-chat         — Step-by-step reasoning (custom tag)
llama-3.1:8b-instruct    — General fallback

------------------------------------------------------------
4. Installing Models
------------------------------------------------------------

Option A — Pull from Ollama Library:
  ollama pull qwen2.5:7b-instruct
  ollama pull llama-3.1:8b-instruct

Option B — Import your own .gguf file:
1. Place file in C:\models
2. Create a Modelfile:

FROM C:\models\your-model-name.gguf
TEMPLATE """
{{- if .System }}System:
{{ .System }}

{{ end -}}
{{- range .Messages -}}
{{- if eq .Role "user" -}}User:
{{ else if eq .Role "assistant" -}}Assistant:
{{ else -}}{{ .Role }}:
{{ end -}}
{{ .Content }}

{{- end -}}
Assistant:
"""

3. Register it with a tag:
  ollama create my-custom-model -f .\Modelfile

4. Check it:
  ollama list
  ollama show my-custom-model --modelfile

------------------------------------------------------------
5. Switching Between Models
------------------------------------------------------------

To switch models:

1. Edit config.json and change:
  "qwen_model_path": "qwen2.5:7b-instruct"
to:
  "qwen_model_path": "deepseek-r1-chat"

2. Restart your assistant.

No code changes required.

------------------------------------------------------------
6. Creating Multiple Model “Flavors”
------------------------------------------------------------

You can create multiple Modelfiles using the same base model but with different system prompts.

Modelfile.math:
FROM deepseek-r1-local:latest
SYSTEM "You are a brilliant mathematician. Use LaTeX for all equations."
TEMPLATE """ ... """
PARAMETER temperature 0.4

Modelfile.chat:
FROM deepseek-r1-local:latest
SYSTEM "You are friendly and concise."
TEMPLATE """ ... """
PARAMETER temperature 0.7

ollama create deepseek-r1-math -f .\Modelfile.math
ollama create deepseek-r1-chat -f .\Modelfile.chat

Switch between them in config.json without changing code.

------------------------------------------------------------
7. Optional Post-Processing
------------------------------------------------------------

Some models (e.g. DeepSeek) output math in ASCII like:
int_0^inf sqrt(x)/(x^2+4) dx

You can convert this automatically to LaTeX in Python:

import re
def to_latex(text):
    text = re.sub(r'sqrt\((.*?)\)', r'\\sqrt{\1}', text)
    text = re.sub(r'int_0\^inf', r'\\int_0^{\\infty}', text)
    text = re.sub(r'\*\*(\d+)', r'^{\1}', text)
    return text

------------------------------------------------------------
8. Managing Models
------------------------------------------------------------

ollama list                  List installed models
ollama show <tag>            Show details
ollama rm <tag>              Remove a model tag
ollama prune                 Clean up unused blobs
ollama create <tag> -f ...   Register a new model
ollama pull <model>          Download from registry

------------------------------------------------------------
9. Recommended Setup Summary
------------------------------------------------------------

- Use tags instead of file paths.
- Keep multiple Modelfiles for different personalities (math/chat).
- Use the same Python code path for all models.
- Use the JSON config to switch models instantly.
- Post-process DeepSeek’s output if you want LaTeX.

------------------------------------------------------------
10. Example Workflow
------------------------------------------------------------

1. Download or import a model
2. Create Modelfiles for different modes
3. ollama create tags like:
   deepseek-r1-chat
   deepseek-r1-math
   qwen2.5-math
4. Set "qwen_model_path" in config.json
5. Restart the assistant

This structure gives you total flexibility:
- You can swap between models without changing code.
- You can keep math and conversational variants of the same model.
- You can layer system prompts without editing your app.
