import subprocess
import time
import requests
import sys

OLLAMA_URL = "http://127.0.0.1:11434"

# Models to warm
TEXT_MODEL = "qwen2.5:7b-instruct"
VISION_MODEL = "qwen2.5vl:7b"


#Kills and restarts the Ollama server cleanly.

#Checks version (ensures server is responding).

#Lists installed and loaded models.

#Automatically pulls any missing models (text or vision).

#Sends a test “ping” to warm both models.

#Confirms loaded models at the end

# ---------------- Core helpers ----------------

def restart_ollama():
    print("[*] Killing any running Ollama server...")
    subprocess.run(
        ["taskkill", "/IM", "ollama.exe", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    print("[*] Starting Ollama...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)  # give it a moment to start

def check_version():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/version", timeout=5)
        r.raise_for_status()
        print(f"[✓] Ollama version: {r.json().get('version')}")
        return True
    except Exception as e:
        print(f"[✗] Ollama not responding: {e}")
        return False

def list_models():
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    r.raise_for_status()
    models = [m["name"] for m in r.json().get("models", [])]
    print(f"[✓] Installed models: {models}")
    return models

def list_running():
    r = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
    r.raise_for_status()
    models = [m["name"] for m in r.json().get("models", [])]
    if models:
        print(f"[✓] Loaded models: {models}")
    else:
        print("[*] No models currently loaded")
    return models

def warm_model(model):
    print(f"[*] Warming model: {model} ...")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=30)
    r.raise_for_status()
    reply = r.json().get("message", {}).get("content", "").strip()
    print(f"[✓] Warm-up reply from {model!r}: {reply}")

def ensure_model(name):
    models = list_models()
    if name not in models:
        print(f"[*] Pulling missing model: {name} ...")
        subprocess.run(["ollama", "pull", name], check=True)
    warm_model(name)

# ---------------- Main ----------------

if __name__ == "__main__":
    restart_ollama()

    if not check_version():
        sys.exit(1)

    list_models()
    list_running()

    # Warm models (auto pull if needed)
    ensure_model(TEXT_MODEL)
    ensure_model(VISION_MODEL)

    list_running()
    print("[✓] Ollama is warmed up and ready.")
