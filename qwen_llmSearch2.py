import os
import requests
import re
import httpx  # Add this import
from collections import namedtuple  # Add this import

# Add this for brave_search
Item = namedtuple('Item', ['title', 'url', 'snippet'])


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

    def set_main_app(self, main_app):
        """Connect to the main application for search functionality"""
        self.main_app = main_app
        print(f"[QwenLLM] ✅ Main app connected: {main_app is not None}")

    def set_search_handler(self, search_handler):
        """Set the search handler function"""
        self.search_handler = search_handler
        print(f"[QwenLLM] ✅ Search handler connected: {search_handler is not None}")

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
        name_l = self.model.lower()
        if "deepseek" in name_l and "r1" in name_l:
            return True
        return self._probe_template() == "prompt_only"

    def generate(self, user_text: str, from_search_method: bool = False) -> str:
        # Enhanced filter for problematic inputs
        user_text_clean = user_text.strip().lower()

        # Filter patterns that should be ignored
        filter_patterns = [
            # Very short meaningless inputs
            len(user_text_clean) <= 2 and user_text_clean not in ['hi', 'ok', 'no', 'yes'],
            # Common ASR misinterpretations
            user_text_clean in ['you', 'and', 'the', 'a', 'to', 'for', 'with'],
            # Incomplete sentence fragments
            user_text_clean.endswith(('.', '..', '...')) and len(user_text_clean) < 10,
            # Repeated single words
            len(user_text_clean.split()) == 1 and user_text_clean in ['thanks', 'thank', 'please', 'sorry']
        ]

        if any(filter_patterns):
            return "I didn't catch that. Could you please rephrase your question?"
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

        processed_reply = self._process_ai_response(reply, from_search_method)

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": processed_reply})
        if len(self.history) > 20:
            self.history = self.history[-20:]
        return processed_reply

    def _process_ai_response(self, response: str, from_search_method: bool = False) -> str:
        """Process AI response and execute any search commands through main App"""
        import re

        search_pattern = r'\[SEARCH:\s*(.*?)\]'
        searches = re.findall(search_pattern, response, re.IGNORECASE)

        if searches and self.main_app:
            self.main_app.logln(f"[AI] Detected {len(searches)} search request(s)")

            all_search_results = ""
            for search_query in searches:
                clean_query = search_query.strip()
                self.main_app.logln(f"[AI] Executing search: {clean_query}")

                if not from_search_method:
                    self._announce_search_voice(clean_query)

                search_results = self.main_app.handle_ai_search_request(clean_query)
                all_search_results += f"\n\n--- SEARCH RESULTS: {clean_query} ---\n{search_results}"
                response = response.replace(f"[SEARCH: {search_query}]", f"\n[I searched for: {clean_query}]")

            response += f"\n\n--- INCORPORATED SEARCH RESULTS ---{all_search_results}"

        return response

    def _announce_search_voice(self, query: str):
        """Provide voice feedback that a search is being performed"""
        if not self.main_app:
            return

        try:
            announcement = f"Searching the internet for {query}"
            search_announce_path = "out/search_announce.wav"

            if self.main_app.synthesize_to_wav(announcement, search_announce_path, role="text"):
                with self.main_app._play_lock:
                    self.main_app._play_token += 1
                    my_token = self.main_app._play_token
                    self.main_app.interrupt_flag = False
                    self.main_app.speaking_flag = True

                self.main_app.set_light("speaking")

                play_path = search_announce_path
                if bool(self.main_app.echo_enabled_var.get()):
                    try:
                        play_path, _ = self.main_app.echo_engine.process_file(search_announce_path,
                                                                              "out/search_announce_echo.wav")
                    except Exception:
                        pass

                self.main_app.play_wav_with_interrupt(play_path, token=my_token)

        except Exception as e:
            self.main_app.logln(f"[search][announce] Error: {e}")

    def clear_history(self):
        self.history.clear()

    def brave_search(self, query: str, count: int = 6):
        """Complete implementation of brave_search method"""
        brave_key = os.getenv("BRAVE_KEY")
        if not brave_key:
            raise RuntimeError("No BRAVE_KEY found in environment")

        # Use main_app logging if available, otherwise fallback
        if self.main_app:
            self.main_app.logln(f"[BRAVE API] 🔍 Searching: '{query}'")
        else:
            print(f"[BRAVE API] 🔍 Searching: '{query}'")

        endpoint = "https://api.search.brave.com/res/v1/web/search"
        headers = {"X-Subscription-Token": brave_key, "User-Agent": "LocalAI-ResearchBot/1.0"}

        params = {"q": query, "count": count}

        if any(term in query.lower() for term in ['scottish', 'scotland', 'uk news', 'british', 'bbc', 'sky news']):
            params["country"] = "GB"
            params["search_lang"] = "en"
            log_msg = "[BRAVE API] 🇬🇧 Geographic targeting: United Kingdom"
        elif any(term in query.lower() for term in ['new zealand', 'nz news', '1news', 'rnz']):
            params["country"] = "NZ"
            log_msg = "[BRAVE API] 🇳🇿 Geographic targeting: New Zealand"
        else:
            log_msg = None

        if log_msg:
            if self.main_app:
                self.main_app.logln(log_msg)
            else:
                print(log_msg)

        with httpx.Client(timeout=25.0, headers=headers) as client:
            r = client.get(endpoint, params=params)
            r.raise_for_status()
            data = r.json()

        out = []
        for w in (data.get("web", {}) or {}).get("results", []):
            out.append(Item(title=w.get("title", "No title"),
                            url=w.get("url", ""),
                            snippet=w.get("description", "")))

        log_msg = f"[BRAVE API] ✅ Found {len(out)} results for '{query}'"
        if self.main_app:
            self.main_app.logln(log_msg)
        else:
            print(log_msg)

        return out

    def generate_with_search(self, prompt: str) -> str:
        """Generate with web search capability - FIXED COMBINED APPROACH"""
        print(f"[DEBUG] generate_with_search called with: '{prompt}'")

        if not hasattr(self, 'search_handler') or not self.search_handler:
            print(f"[DEBUG] ❌ No search handler - falling back to regular generate")
            return self.generate(prompt)

        prompt_lower = prompt.lower()

        forced_search_triggers = [
            'weather', 'temperature', 'forecast', '°c', '°f', 'rain',
            'snow', 'wind', 'humid', 'cloud', 'sunny', 'storm',
            'news', 'headlines', 'breaking', 'latest news', 'current events',
            'today\'s news', 'happening now',
            'sports', 'score', 'result', 'match', 'game', 'tournament',
            'stock', 'share price', 'market', 'trading',
        ]

        should_force_search = any(trigger in prompt_lower for trigger in forced_search_triggers)

        if should_force_search:
            print(f"[DEBUG] 🎯 FORCED SEARCH TRIGGERED: {prompt}")

            search_query = self._build_forced_search_query(prompt_lower)
            print(f"[DEBUG] 🔍 Performing forced search: {search_query}")

            try:
                search_results = self.search_handler(search_query)
                print(f"[DEBUG] 📊 Search results received: {len(search_results)} chars")

                response = self._generate_from_forced_search(prompt, search_results, prompt_lower)
                return response

            except Exception as e:
                print(f"[DEBUG] ❌ Forced search failed: {e}")
                return self.generate(prompt)
        else:
            print(f"[DEBUG] Using AI-decided search approach")
            return self._generate_with_ai_decided_search(prompt)

    def _generate_with_ai_decided_search(self, prompt: str) -> str:
        """Let the AI decide if it wants to search"""
        print(f"[DEBUG] Using AI-decided search for: {prompt}")

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

        original_system = self.system_prompt
        self.system_prompt = search_enhanced_system

        response = self.generate(prompt)

        self.system_prompt = original_system

        print(f"[DEBUG] AI-decided search response: {response[:100]}...")
        return response

    def _build_forced_search_query(self, prompt_lower: str) -> str:
        """Build appropriate search query for forced searches"""
        if any(word in prompt_lower for word in ['weather', 'temperature', 'forecast']):
            location = "Auckland, New Zealand"
            if "weather in" in prompt_lower:
                location = prompt_lower.split("weather in")[-1].split('?')[0].strip()
            elif "weather at" in prompt_lower:
                location = prompt_lower.split("weather at")[-1].split('?')[0].strip()
            elif "weather for" in prompt_lower:
                location = prompt_lower.split("weather for")[-1].split('?')[0].strip()
            elif "weather like in" in prompt_lower:
                location = prompt_lower.split("weather like in")[-1].split('?')[0].strip()
            elif "temperature in" in prompt_lower:
                location = prompt_lower.split("temperature in")[-1].split('?')[0].strip()
            elif "temperature at" in prompt_lower:
                location = prompt_lower.split("temperature at")[-1].split('?')[0].strip()
            elif "forecast for" in prompt_lower:
                location = prompt_lower.split("forecast for")[-1].split('?')[0].strip()
            elif "forecast in" in prompt_lower:
                location = prompt_lower.split("forecast in")[-1].split('?')[0].strip()
            elif "london" in prompt_lower:
                location = "London, UK"
            elif "auckland" in prompt_lower:
                location = "Auckland, New Zealand"
            elif "new york" in prompt_lower or "nyc" in prompt_lower:
                location = "New York, USA"
            elif "sydney" in prompt_lower:
                location = "Sydney, Australia"
            elif "tokyo" in prompt_lower:
                location = "Tokyo, Japan"

            print(f"[DEBUG] Using location: '{location}' for query: '{prompt_lower}'")
            return f"current weather {location}"

        elif any(word in prompt_lower for word in ['sports', 'score', 'match']):
            return "latest sports news scores"

        elif any(word in prompt_lower for word in ['stock', 'share price']):
            return "current stock market prices"

        elif any(word in prompt_lower for word in ['tv', 'television', 'what\'s on', 'tonight']):
            if "tv1" in prompt_lower or "tvnz" in prompt_lower:
                return "TV1 TVNZ New Zealand tonight schedule programming"
            elif "tv2" in prompt_lower:
                return "TV2 New Zealand tonight schedule"
            elif "tv3" in prompt_lower:
                return "TV3 New Zealand tonight schedule"
            else:
                return "New Zealand television tonight schedule programming"

        return prompt_lower

    def _generate_from_forced_search(self, original_prompt: str, search_results: str, prompt_lower: str) -> str:
        """Generate response using actual search data from forced search"""
        if any(word in prompt_lower for word in ['weather', 'temperature']):
            prompt_template = f"""
USER QUESTION: {original_prompt}

REAL-TIME WEATHER DATA FROM WEB SEARCH:
{search_results}

CRITICAL: You have REAL-TIME WEATHER DATA from a web search. You MUST use the ACTUAL temperature numbers and weather conditions.
- Use EXACT temperatures like "19°C", "65°F" from the search results
- Use SPECIFIC conditions like "sunny", "rainy", "cloudy" from the search results  
- Include wind speeds and humidity if available in the results
- Mention specific locations and timeframes mentioned in the results
- DO NOT say "I don't have real-time access" - you have the search results right here
- DO NOT give generic weather patterns - use only the actual data found

Provide a direct, confident weather report using ONLY the real data above.
"""
        # ... rest of your _generate_from_forced_search method remains the same
        # [Keep the existing implementation for news, tv, and other cases]

        # For brevity, I've included the weather case. The news, tv, and default cases
        # from your original code should follow the same pattern.

        else:
            prompt_template = f"""
USER QUESTION: {original_prompt}

REAL-TIME INFORMATION FROM WEB SEARCH:
{search_results}

CRITICAL: You have REAL-TIME DATA from a web search. You MUST use the ACTUAL information.
- Be specific and use the real data found
- DO NOT say "I don't have real-time access" - you have the search results right here
- If specific numbers, names, or facts are in the results, use them exactly
- DO NOT invent or create information that isn't explicitly in the search results

Provide a direct answer using ONLY the information from the search results above.
"""

        response = self.generate(prompt_template)
        print(f"[DEBUG] Generated response from forced search: {response[:200]}...")

        if self.main_app and hasattr(self.main_app, 'preview_search_results'):
            self.main_app.preview_search_results(response)

        return response
