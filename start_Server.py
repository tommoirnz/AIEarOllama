import subprocess
import os

# Log files — same as in your PowerShell version
out_log = os.path.join(os.getenv("LOCALAPPDATA"), "Ollama", "serve.out.log")
err_log = os.path.join(os.getenv("LOCALAPPDATA"), "Ollama", "serve.err.log")

# Start Ollama detached in the background
subprocess.Popen(
    [
        "ollama", "serve"
    ],
    stdout=open(out_log, "a"),
    stderr=open(err_log, "a"),
    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
)

print("[✅] Ollama started in background. You can close this Python script now.")
print(f"Logs: {out_log}")
