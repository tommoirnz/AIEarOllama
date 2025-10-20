import os, requests, re

class QwenLLM:
    def __init__(self, model_path=None, **kwargs):
        base = kwargs.get("base_url") or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.base = base.rstrip("/")
        self.model = model_path or kwargs.get("model") or os.environ.get("OLLAMA_MODEL") or "qwen2.5:7b-instruct"

        self.temperature = float(kwargs.get("temperature", 0.6))
        self.max_tokens  = int(kwargs.get("max_tokens", 120))
        self.system_prompt = kwargs.get("system_prompt",
            "You are a friendly and detailed AI assistant. Speak naturally, explain clearly, and remember context.")
        self.history = []
        self.session = requests.Session()
        self.timeout = kwargs.get("timeout", 120)

        # Optional external control
        self.force_template = bool(kwargs.get("force_template", False))

        self.chat_url = None
        self.chat_mode = None
        self._template_probe = None  # "prompt_only" | "role_aware" | "unknown"

    def _detect_endpoint(self):
        self.session.get(f"{self.base}/api/tags", timeout=3).raise_for_status()
        for url, mode in [(f"{self.base}/api/chat", "ollama"),
                          (f"{self.base}/v1/chat/completions", "openai")]:
            try:
                r = self.session.post(url, json={
                    "model": self.model,
                    "messages": [{"role": "system", "content": "ping"}, {"role": "user", "content": "ok"}],
                    "stream": False,
                    "options": {"num_predict": 1} if mode=="ollama" else None,
                    "temperature": 0.0 if mode=="openai" else None,
                    "max_tokens": 1 if mode=="openai" else None,
                }, timeout=6)
                if r.status_code == 200:
                    self.chat_url, self.chat_mode = url, mode
                    return
            except Exception:
                pass
        raise RuntimeError("No working chat endpoint found. Is `ollama serve` running?")

    def _probe_template(self):
        """Ask Ollama for the model’s Modelfile and decide if it’s prompt-only."""
        if self._template_probe is not None:
            return self._template_probe
        try:
            # Ollama show API: POST /api/show { "name": "<model>" }
            r = self.session.post(f"{self.base}/api/show", json={"name": self.model}, timeout=4)
            if r.status_code == 200:
                text = r.text.lower()
                if re.search(r'^\s*template\s+{{\s*\.prompt\s*}}', r.text, re.MULTILINE):
                    self._template_probe = "prompt_only"
                elif "template" in text and (".messages" in text or ".system" in text):
                    self._template_probe = "role_aware"
                else:
                    self._template_probe = "unknown"
            else:
                self._template_probe = "unknown"
        except Exception:
            self._template_probe = "unknown"
        return self._template_probe

    def _ensure_ready(self):
        if self.chat_url is None:
            self._detect_endpoint()

    def _should_override_template(self):
        if self.force_template:
            return True
        # Heuristic: DeepSeek R1 variants, or confirmed prompt-only template
        name_l = self.model.lower()
        if "deepseek" in name_l and "r1" in name_l:
            return True
        return self._probe_template() == "prompt_only"

    def generate(self, user_text: str) -> str:
        self._ensure_ready()

        messages = [{"role": "system", "content": self.system_prompt}]
        messages += self.history
        messages.append({"role": "user", "content": user_text})

        if self.chat_mode == "ollama":
            options = {"temperature": self.temperature, "num_predict": self.max_tokens}
            payload = {
                "model": self.model,
                "messages": messages,
                "system": self.system_prompt,   # harmless if unused; helpful if respected
                "stream": False,
                "options": options,
            }
            if self._should_override_template():
                payload["template"] = (
                    "{{- if .System }}System:\\n{{ .System }}\\n\\n{{ end -}}"
                    "{{- range .Messages -}}"
                    "{{- if eq .Role \"user\" -}}User:{{ else if eq .Role \"assistant\" -}}Assistant:{{ else -}}{{ .Role | title }}:{{ end }}\\n"
                    "{{ .Content }}\\n\\n"
                    "{{- end -}}"
                    "Assistant:\\n"
                )
                # Helps rein in DeepSeek R1’s <think> and turn resets:
                payload["options"]["stop"] = ["</think>", "User:", "\nUser:", "\nAssistant:"]
        else:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }

        r = self.session.post(self.chat_url, json=payload, timeout=self.timeout)
        if r.status_code in (400, 404):
            self.chat_url = None
            self._ensure_ready()
            r = self.session.post(self.chat_url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        reply = (
            data["choices"][0]["message"]["content"].strip()
            if self.chat_mode == "openai"
            else data["message"]["content"].strip()
        )
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        if len(self.history) > 20:
            self.history = self.history[-20:]
        return reply

    def clear_history(self):
        self.history.clear()


    def set_search_handler(self, search_handler):
        """Allow the LLM to trigger web searches"""
        self.search_handler = search_handler


    def generate_with_search(self, prompt: str) -> str:
        """Generate with web search capability"""
        if hasattr(self, 'search_handler') and self.search_handler:
            # Enhanced system prompt for better search decisions
            search_enhanced_system = self.system_prompt + """
    
            WEB SEARCH CAPABILITY:
            You can search the web for current information when needed using: [SEARCH: your query]
    
            Use web searches for:
            - Current events, news, and recent developments (last 1-2 years)
            - Specific facts, statistics, data, or technical specifications
            - Recent research papers or scientific discoveries
            - Current prices, product information, or market data
            - Information that may have changed since your training data
            - Political Information
            - Bus or Train timetables
            - Flights or flight times
            - Weather information
            - TV or Television programming timetables
    
            Do NOT search for:
            - General knowledge that you already know well
            - Historical facts that are well-established
            - Basic mathematical formulas or scientific principles
            - Information that is unlikely to have changed
    
            Search examples:
            Good: [SEARCH: latest iPhone 15 specifications and prices]
            Good: [SEARCH: who is the current prime minister of a country]
            Good: [SEARCH: current climate change policy updates 2024]
            Good: [SEARCH: recent breakthroughs in quantum computing 2024]
            Avoid: [SEARCH: what is photosynthesis]
            Avoid: [SEARCH: basic algebra formulas]
    
            After receiving search results, analyze and incorporate them naturally into your response.
            """

            # Store original system prompt temporarily
            original_system = self.system_prompt
            self.system_prompt = search_enhanced_system

            response = self.generate(prompt)

            # Restore original system prompt
            self.system_prompt = original_system
            return response
        else:
            return self.generate(prompt)

