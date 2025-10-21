import os, requests, re


class QwenLLM:
    def __init__(self, model_path=None, **kwargs):
        base = kwargs.get("base_url") or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.base = base.rstrip("/")
        self.model = model_path or kwargs.get("model") or os.environ.get("OLLAMA_MODEL") or "qwen2.5:7b-instruct"

        self.temperature = float(kwargs.get("temperature", 0.6))
        self.max_tokens = int(kwargs.get("max_tokens", 120))
        self.original_system_prompt = kwargs.get("system_prompt", "You are a helpful AI called Zen")
        self.system_prompt = self.original_system_prompt
        self.history = []
        self.session = requests.Session()
        self.timeout = kwargs.get("timeout", 120)

        # Optional external control
        self.force_template = bool(kwargs.get("force_template", False))

        self.chat_url = None
        self.chat_mode = None
        self._template_probe = None  # "prompt_only" | "role_aware" | "unknown"

        # Reference to main app for proper search coordination
        self.main_app = None

    def set_main_app(self, app):
        """Set reference to main App for proper search coordination"""
        self.main_app = app

    def _detect_endpoint(self):
        self.session.get(f"{self.base}/api/tags", timeout=3).raise_for_status()
        for url, mode in [(f"{self.base}/api/chat", "ollama"),
                          (f"{self.base}/v1/chat/completions", "openai")]:
            try:
                r = self.session.post(url, json={
                    "model": self.model,
                    "messages": [{"role": "system", "content": "ping"}, {"role": "user", "content": "ok"}],
                    "stream": False,
                    "options": {"num_predict": 1} if mode == "ollama" else None,
                    "temperature": 0.0 if mode == "openai" else None,
                    "max_tokens": 1 if mode == "openai" else None,
                }, timeout=6)
                if r.status_code == 200:
                    self.chat_url, self.chat_mode = url, mode
                    return
            except Exception:
                pass
        raise RuntimeError("No working chat endpoint found. Is `ollama serve` running?")

    def _probe_template(self):
        """Ask Ollama for the model's Modelfile and decide if it's prompt-only."""
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

    def generate(self, user_text: str, from_search_method: bool = False) -> str:
        self._ensure_ready()

        messages = [{"role": "system", "content": self.system_prompt}]
        messages += self.history
        messages.append({"role": "user", "content": user_text})

        if self.chat_mode == "ollama":
            options = {"temperature": self.temperature, "num_predict": self.max_tokens}
            payload = {
                "model": self.model,
                "messages": messages,
                "system": self.system_prompt,
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

        # Process response for search commands
        processed_reply = self._process_ai_response(reply, from_search_method)

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": processed_reply})
        if len(self.history) > 20:
            self.history = self.history[-20:]
        return processed_reply

    def _process_ai_response(self, response: str, from_search_method: bool = False) -> str:
        """
        Process AI response and execute any search commands through main App
        """
        import re

        # Look for search commands in the response
        search_pattern = r'\[SEARCH:\s*(.*?)\]'
        searches = re.findall(search_pattern, response, re.IGNORECASE)

        if searches and self.main_app:
            self.main_app.logln(f"[AI] Detected {len(searches)} search request(s)")

            all_search_results = ""
            for search_query in searches:
                clean_query = search_query.strip()
                self.main_app.logln(f"[AI] Executing search: {clean_query}")

                # ANNOUNCE SEARCH VOICE FEEDBACK
                if not from_search_method:
                    self._announce_search_voice(clean_query)

                # Execute the search using main app's search system
                search_results = self.main_app.handle_ai_search_request(clean_query)
                all_search_results += f"\n\n--- SEARCH RESULTS: {clean_query} ---\n{search_results}"

                # Update the response to include search results
                response = response.replace(f"[SEARCH: {search_query}]", f"\n[I searched for: {clean_query}]")

            # Append all search results to the final response
            response += f"\n\n--- INCORPORATED SEARCH RESULTS ---{all_search_results}"

        return response

    def _announce_search_voice(self, query: str):
        """Provide voice feedback that a search is being performed"""
        if not self.main_app:
            return

        try:
            # Create a brief search announcement
            announcement = f"Searching the internet for {query}"

            # Generate speech file directly without cleaning
            search_announce_path = "out/search_announce.wav"

            # Use the same TTS system as main app
            if self.main_app.synthesize_to_wav(announcement, search_announce_path, role="text"):
                # Play the announcement with proper interrupt handling
                with self.main_app._play_lock:
                    self.main_app._play_token += 1
                    my_token = self.main_app._play_token
                    self.main_app.interrupt_flag = False
                    self.main_app.speaking_flag = True

                self.main_app.set_light("speaking")

                # Apply echo if enabled
                play_path = search_announce_path
                if bool(self.main_app.echo_enabled_var.get()):
                    try:
                        play_path, _ = self.main_app.echo_engine.process_file(search_announce_path,
                                                                              "out/search_announce_echo.wav")
                    except Exception:
                        pass

                # Play the announcement and wait for completion
                self.main_app.play_wav_with_interrupt(play_path, token=my_token)

        except Exception as e:
            self.main_app.logln(f"[search][announce] Error: {e}")

    def clear_history(self):
        self.history.clear()

    def set_search_handler(self, search_handler):
        """Allow the LLM to trigger web searches"""
        self.search_handler = search_handler

    def generate_with_search(self, prompt: str) -> str:
        """Generate with web search capability - FIXED VERSION"""
        if self.main_app:
            # Use a more direct and clear system prompt
            search_enhanced_system = self.original_system_prompt + """

    IMPORTANT: You have access to REAL-TIME WEB SEARCH using: [SEARCH: your search query]

    Use web search for current information like:
    - News, headlines, breaking news
    - Weather forecasts and current conditions  
    - Recent events and developments
    - Live sports scores, stock prices
    - Current political information

    When you need current information, use [SEARCH: your query] and I will provide you with real search results.

    Examples:
    "What's the latest news?" → [SEARCH: latest news headlines today]
    "What's the weather in London?" → [SEARCH: current weather London]
    "What are the latest sports scores?" → [SEARCH: latest sports scores]

    After receiving search results, use them to answer the question.
    """

            # Store current system prompt temporarily
            current_system = self.system_prompt
            self.system_prompt = search_enhanced_system

            try:
                # Let the AI decide if it wants to search
                initial_response = self.generate(prompt, from_search_method=True)

                # Check if the AI included a search command
                import re
                search_pattern = r'\[SEARCH:\s*(.*?)\]'
                searches = re.findall(search_pattern, initial_response, re.IGNORECASE)

                if searches:
                    # Use main app's search system for consistent results
                    all_search_results = ""
                    for search_query in searches:
                        clean_query = search_query.strip()
                        self.main_app.logln(f"[AI] Performing coordinated search: {clean_query}")

                        # ANNOUNCE SEARCH VOICE FEEDBACK
                        self._announce_search_voice(clean_query)

                        # Get REAL search results from the main app's search handler
                        real_results = self.main_app.handle_ai_search_request(clean_query)
                        all_search_results += f"\n\n--- SEARCH RESULTS: {clean_query} ---\n{real_results}"

                    # Now generate a final response that incorporates the REAL search results
                    if any(word in prompt.lower() for word in
                           ['weather', 'temperature', 'forecast', '°c', '°f', 'rain']):
                        follow_up_prompt = f"""
    Original question: {prompt}

    ACTUAL WEATHER DATA FROM SEARCH:
    {all_search_results}

    IMPORTANT FOR WEATHER RESPONSE:
    - Use EXACT temperatures from search results (e.g., 11°C, 23°C, 20°C high, 11°C low)
    - Use specific conditions mentioned (e.g., "showers", "cloudy", "sunny intervals")
    - Include wind speeds if available (e.g., "15-40 km/h")
    - Include timeframes (e.g., "Thursday", "next 10 days")
    - DO NOT use [current temperature] or other placeholders
    - If specific dates are mentioned, use them

    Provide a detailed weather forecast using ONLY the actual data from the search results above.
    """
                    else:
                        follow_up_prompt = f"""
    Original question: {prompt}

    ACTUAL SEARCH RESULTS:
    {all_search_results}

    CRITICAL: You MUST use the specific numbers, temperatures, names, and facts from the search results above.
    DO NOT use generic placeholders like [current temperature], [date], or [location].
    If the search results contain specific data like temperatures (e.g., 11°C, 23°C), use them exactly.
    If the search results don't contain certain information, acknowledge what IS available.

    Please provide a final answer that incorporates these real search results using the actual data provided.
    """

                    final_response = self.generate(follow_up_prompt, from_search_method=True)

                    # Restore original system prompt
                    self.system_prompt = current_system
                    return final_response
                else:
                    # No search needed, return original response
                    self.system_prompt = current_system
                    return initial_response
            except Exception as e:
                # Restore original system prompt on error
                self.system_prompt = current_system
                raise e
        else:
            # Fallback to regular generation
            return self.generate(prompt)
