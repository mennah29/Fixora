"""Grounded RAG service extracted from the project's notebook pipeline."""

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from app.config import Settings

NOT_FOUND = "NOT_FOUND_IN_MANUAL"
WEB_FALLBACK_LABEL = "WEB FALLBACK — NOT FOUND IN MANUAL"
SAFETY_WORDS = ("DANGER", "WARNING", "CAUTION")
QUERY_CODE_PATTERN = re.compile(
    r"\b(?:ERR(?:OR)?[\s\-]*CODE|ERR(?:OR)?|CODE|FAULT|ALARM|E|F)[\s\-]*0*(\d{1,7})\b",
    re.IGNORECASE,
)
FAULT_CODE_PATTERN = re.compile(
    r"\b(E\d{2,5}|F\d{2,5}|ERR-\d{2,5}|Fault\s*\d{1,5}|Code\s*\d{1,5}|Alarm\s*\d{1,5})\b",
    re.IGNORECASE,
)

MANUAL_PROMPT = """You are Fixora, an experienced senior maintenance engineer on a live hands-free phone call with an on-site technician.
Your job is to guide the technician safely and clearly based ONLY on the provided manual context.

=== STRICT GROUNDING RULES ===
1. Base all technical advice strictly on the provided manual context. Never invent procedures.
2. If the manual context lacks the answer, reply EXACTLY with:
   {{"status": "NOT_FOUND"}}
3. Flag high-voltage, radiation, chemical, oxygen, or mechanical hazards as high-priority safety risks.

=== VOICE SPEAKING INSTRUCTIONS (`speech_text`) ===
- Speak like a supportive human coworker talking on the phone.
- DO NOT read out robotic lists like "Step 1...", "Step 2...", or bullet points.
- Instead, conversationalize the instructions: explain what needs to be checked, what should be avoided, and the rationale in simple, flowing sentences.
- Always begin with an immediate, clear warning if there is any hazard.
- Strictly avoid emojis, symbols, markdown formatting, bullet characters, and raw file names in `speech_text`.

=== OUTPUT FORMAT ===
Return ONLY raw, valid JSON with no markdown code blocks:

{{
  "has_high_priority_safety": true,
  "safety_header": "⚠️ HIGH PRIORITY SAFETY INSTRUCTIONS DETECTED",
  "safety_body": "DANGER/WARNING details for the UI panel.",
  "fault_meaning": "Simple explanation of the fault code or symptom.",
  "checklist": [
    "Step 1: Formal action item for the screen",
    "Step 2: Formal action item for the screen"
  ],
  "source_citation": {{
    "manual": "{manual_name}",
    "page": "{page_number}",
    "section": "{section_name}"
  }},
  "speech_text": "Spoken explanation written in natural conversational English, guiding the technician on what to do and what not to do without reading out list numbers."
}}

=== CONTEXT DATA ===
Selected Equipment: {device_name}
Manual File: {manual_name}
Page: {page_number}
Section: {section_name}
Context from Manual:
{context}

=== TECHNICIAN QUERY ===
Technician Spoke: "{question}"

JSON Output:"""

WEB_PROMPT = """You are Fixora. The manuals did not answer the technician's question.
Start with exactly: WEB FALLBACK — NOT FOUND IN MANUAL. Use only the supplied web results.
Prioritize official manufacturer sources, include source URLs, and state clearly if the result
cannot be verified. Keep it concise.

Web search results:
{search_results}

Technician Query:
{query}

Answer:"""


class ServiceNotReadyError(RuntimeError):
    pass


def extract_error_codes(query: str) -> list[str]:
    return sorted({str(int(value)) for value in QUERY_CODE_PATTERN.findall(query)}, key=lambda value: (len(value), value))


def _row_is_error_entry(row: str, code: str) -> bool:
    cells = [cell.strip() for cell in row.split("|") if cell.strip()]
    if len(cells) < 2 or not re.fullmatch(rf"0*{re.escape(code)}", cells[0]):
        return False
    return True


@dataclass
class RagService:
    settings: Settings
    collection: Any = None
    embedding_model: Any = None
    startup_error: str | None = None
    _qwen_model: Any = None
    _qwen_tokenizer: Any = None

    def _load_embedding_model(self):
        """Load the prebuilt embedding model without an online startup dependency."""
        if self.settings.embedding_local_files_only:
            # `SentenceTransformer` delegates part of model loading to
            # Hugging Face. These flags prevent metadata requests for a model
            # that is already baked into the deployment image/cache.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            self.settings.embedding_model,
            device="cpu",
            local_files_only=self.settings.embedding_local_files_only,
        )

    def _load_qwen_model(self) -> None:
        """Load Qwen from the configured local cache, using GPU 4-bit when available."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_name = str(self.settings.qwen_model_path or self.settings.qwen_model)
        self._qwen_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
        )
        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available() and self.settings.qwen_load_in_4bit:
            load_kwargs.update({
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                ),
                "device_map": {"": 0},
                "attn_implementation": "eager",
            })
        else:
            load_kwargs.update({"torch_dtype": torch.float32, "device_map": "cpu"})
        self._qwen_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    def load(self) -> None:
        """Connect to the persisted index and load the embedding model once."""
        import chromadb

        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
        self.collection = client.get_or_create_collection(
            name=self.settings.collection_name, metadata={"hnsw:space": "cosine"}
        )
        if self.collection.count() == 0 and self.settings.auto_index:
            self._index_chunks()
        if self.collection.count() == 0:
            self.startup_error = (
                f"No indexed manuals found. Mount {self.settings.data_dir} with a populated "
                f"'{self.settings.chroma_dir}' directory, or run scripts/build_index.py."
            )
            return
        # The Docker image fetches this model during its build and the index is
        # built with the same local cache. Do not make startup depend on an
        # external Hugging Face metadata request.
        self.embedding_model = self._load_embedding_model()

        if self.settings.llm_provider.lower() == "qwen_local":
            self._load_qwen_model()

        self.startup_error = None

    @property
    def ready(self) -> bool:
        return self.collection is not None and self.embedding_model is not None and not self.startup_error

    @property
    def document_count(self) -> int:
        return self.collection.count() if self.collection is not None else 0

    def _index_chunks(self) -> None:
        if not self.settings.chunks_path.is_file():
            raise ServiceNotReadyError(f"Chunks file not found: {self.settings.chunks_path}")
        chunks = json.loads(self.settings.chunks_path.read_text(encoding="utf-8"))
        model = self._load_embedding_model()
        for start in range(0, len(chunks), 128):
            batch = chunks[start : start + 128]
            documents = [item["content"] for item in batch]
            metadata = []
            for item in batch:
                source = item.get("metadata", {})
                codes = source.get("error_codes", [])
                metadata.append({
                    "device": source.get("device", "unknown"), "manual": source.get("manual", "unknown"),
                    "page": int(source.get("page", 0)), "error_codes": ",".join(codes) if isinstance(codes, list) else str(codes),
                    "has_safety_warning": bool(source.get("has_safety_warning", False)), "type": item.get("type", "text"),
                })
            self.collection.upsert(
                ids=[item["chunk_id"] for item in batch], documents=documents, metadatas=metadata,
                embeddings=model.encode(documents, convert_to_numpy=True).tolist(),
            )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not self.embedding_model:
            raise ServiceNotReadyError(self.startup_error or "Embedding model is not loaded.")
        return self.embedding_model.encode(texts, convert_to_numpy=True).tolist()

    def semantic_search(self, query: str, device_name: str | None, k: int) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=self._embed([query]), n_results=k,
            where={"device": device_name} if device_name else None,
        )
        limit = 1 - self.settings.min_semantic_score
        values = []
        for chunk_id, text, metadata, distance in zip(
            result.get("ids", [[]])[0], result.get("documents", [[]])[0],
            result.get("metadatas", [[]])[0], result.get("distances", [[]])[0],
        ):
            if distance <= limit:
                values.append(self._as_result(chunk_id, text, metadata, "semantic", 1 - distance))
        return values

    def exact_search(self, query: str, device_name: str | None, k: int) -> list[dict[str, Any]]:
        matches: dict[str, dict[str, Any]] = {}
        q_words = {w.lower() for w in re.findall(r"\b\w{3,}\b", query) if w.lower() not in {"what", "does", "mean", "how", "solve", "error", "code", "the", "for", "please", "about"}}
        for code in extract_error_codes(query):
            candidates = self.collection.get(
                where={"device": device_name} if device_name else None,
                where_document={"$contains": code}, include=["documents", "metadatas"],
            )
            validation = re.compile(
                rf"\bE\s?0*{re.escape(code)}\b|\bERR(?:OR)?[\s\-]*0*{re.escape(code)}\b|"
                rf"\bF\s?0*{re.escape(code)}\b|\b(?:FAULT|CODE|ALARM)[\s\-]*0*{re.escape(code)}\b",
                re.IGNORECASE,
            )
            for chunk_id, text, metadata in zip(candidates.get("ids", []), candidates.get("documents", []), candidates.get("metadatas", [])):
                lines = text.splitlines()
                evidence = next((line.strip() for line in lines if validation.search(line) or _row_is_error_entry(line, code)), None)
                if evidence:
                    score = 2.0 if _row_is_error_entry(evidence, code) else 1.0
                    evidence_upper = evidence.upper()
                    if any(w in evidence_upper for w in ("ERR", "FAULT", "ALARM", "RANGE", "FAIL", "CHECK", "REPLACE", "TEST", "LIMIT", "OVERHEAT", "WARN", "BATTERY")):
                        score += 3.0
                    doc_lower = (text + " " + str(metadata.get("manual", "")) + " " + str(metadata.get("device", ""))).lower()
                    overlap = sum(1 for w in q_words if w in doc_lower)
                    score += overlap * 1.0
                    item = self._as_result(chunk_id, text, metadata, "exact", score)
                    item.update({"matched_text": evidence, "matched_code": code})
                    matches[chunk_id] = item
        return sorted(matches.values(), key=lambda x: -x["score"])[:k]

    @staticmethod
    def _as_result(chunk_id: str, text: str, metadata: dict[str, Any], retrieval_type: str, score: float) -> dict[str, Any]:
        return {"chunk_id": chunk_id, "text": text, "manual_name": metadata.get("manual", "unknown"),
                "page_number": metadata.get("page", "unknown"), "section_name": metadata.get("type", "unknown"),
                "device": metadata.get("device", "unknown"), "retrieval_type": retrieval_type, "score": score}

    def retrieve(self, query: str, device_name: str | None, k: int) -> list[dict[str, Any]]:
        codes = extract_error_codes(query)
        exact = self.exact_search(query, device_name, k)
        if not exact and device_name:
            # Fallback across all devices if strict device filter had no matches
            exact = self.exact_search(query, None, k)

        cleaned = QUERY_CODE_PATTERN.sub(" ", query).strip()
        meaningful = re.sub(r"[^\w\s]", "", re.sub(r"\b(?:what|does|mean|is|error|code|fault|alarm|the|a|an|please|tell|me|about|in|for|on|okay|about|how)\b", "", cleaned, flags=re.IGNORECASE)).strip()
        if codes and not exact and not meaningful:
            return []
        semantic = self.semantic_search(cleaned or query, device_name, k) if (not codes or exact or meaningful) else []
        if not semantic and device_name and (not codes or exact or meaningful):
            semantic = self.semantic_search(cleaned or query, None, k)

        merged = {item["chunk_id"]: item for item in semantic}
        merged.update({item["chunk_id"]: item for item in exact})
        return sorted(merged.values(), key=lambda item: (item["retrieval_type"] != "exact", -item["score"]))[:k]

    @staticmethod
    def build_context(results: list[dict[str, Any]]) -> str:
        if not results:
            return NOT_FOUND
        return "\n---\n\n".join(
            f"[Source {index}]\nDevice: {item['device']}\nManual: {item['manual_name']}\n"
            f"Page: {item['page_number']}\nSection: {item['section_name']}\n\nText:\n{item['text']}"
            for index, item in enumerate(results, 1)
        )

    @staticmethod
    def _format_json_to_markdown(data: dict[str, Any]) -> str:
        parts = []
        if data.get("has_high_priority_safety") and data.get("safety_body"):
            header = data.get("safety_header") or "⚠️ HIGH PRIORITY SAFETY INSTRUCTIONS DETECTED"
            parts.append(f"### {header}\n**{data['safety_body']}**\n")
        if data.get("fault_meaning"):
            parts.append(f"**Fault Overview:** {data['fault_meaning']}\n")
        if data.get("checklist"):
            parts.append("### 🔧 Step-by-Step Checklist")
            for step in data["checklist"]:
                parts.append(f"- {step}")
        if data.get("source_citation"):
            c = data["source_citation"]
            parts.append(f"\n[Source 1: {c.get('manual', 'Manual')} · Page {c.get('page', '?')}]")
        return "\n".join(parts) if parts else str(data.get("speech_text", ""))

    @staticmethod
    def _extractive_json(results: list[dict[str, Any]], query: str, device_name: str, manual_name: str, page_number: str, section_name: str) -> dict[str, Any]:
        if not results:
            return {"status": "NOT_FOUND"}
        
        top = results[0]
        text = top["text"].strip()
        codes = extract_error_codes(query)
        
        safety_lines = []
        for line in text.splitlines():
            upper = line.upper()
            if any(w in upper for w in SAFETY_WORDS):
                safety_lines.append(line.strip())
        
        checklist = []
        fault_meaning = ""
        if codes:
            for code in codes:
                for line in text.splitlines():
                    if "|" in line and line.strip().startswith(code):
                        cells = [c.strip() for c in line.split("|") if c.strip()]
                        if len(cells) >= 2:
                            err_title = cells[1].replace("_", " ").title()
                            fault_meaning = f"Error {code} points to a {err_title} condition on the unit."
                            action = cells[2] if len(cells) > 2 else ""
                            if action and action.upper() != "N/A":
                                sub_steps = [s.strip() for s in re.split(r"\d+\.\s*", action) if s.strip()]
                                for idx, stp in enumerate(sub_steps, 1):
                                    checklist.append(f"Step {idx}: {stp}")
                            else:
                                checklist.append(f"Step 1: Inspect the {err_title} wiring and board connectors.")
                                checklist.append(f"Step 2: Check signal continuity and calibrate transducer according to manual page {page_number}.")
        
        if not checklist:
            sentences = [s.strip() for s in text.split("\n") if s.strip() and not s.strip().startswith("SPR8-") and not s.strip().startswith("Page ")]
            fault_meaning = sentences[0] if sentences else f"Maintenance procedure for {query} on {device_name}."
            for idx, sent in enumerate(sentences[1:5], 1):
                checklist.append(f"Step {idx}: {sent}")
        
        has_safety = bool(safety_lines)
        safety_body = safety_lines[0] if safety_lines else ""
        
        spoken_parts = []
        if has_safety:
            clean_safety = re.sub(r"^[⚠️\s*#-]+", "", safety_body).strip()
            clean_safety = re.sub(r"^(?:DANGER|WARNING|CAUTION)[\s:!]+", "", clean_safety, flags=re.IGNORECASE).strip()
            spoken_parts.append(f"Important safety note: {clean_safety}.")
        
        if fault_meaning:
            spoken_parts.append(fault_meaning)
        
        if checklist:
            clean_steps = [re.sub(r"^Step \d+:\s*", "", s).strip().rstrip(".") for s in checklist if s.strip()]
            if clean_steps:
                if len(clean_steps) == 1:
                    spoken_parts.append(f"To handle this, you should {clean_steps[0][0].lower() + clean_steps[0][1:]}.")
                elif len(clean_steps) == 2:
                    spoken_parts.append(f"Here is how to handle it: first, {clean_steps[0][0].lower() + clean_steps[0][1:]}. Next, {clean_steps[1][0].lower() + clean_steps[1][1:]}.")
                else:
                    action_flow = ". Then, ".join(clean_steps)
                    spoken_parts.append(f"Here is how to handle it: {action_flow}.")

        return {
            "has_high_priority_safety": has_safety,
            "safety_header": "⚠️ HIGH PRIORITY SAFETY INSTRUCTIONS DETECTED" if has_safety else "",
            "safety_body": safety_body,
            "fault_meaning": fault_meaning,
            "checklist": checklist,
            "source_citation": {
                "manual": manual_name,
                "page": page_number,
                "section": section_name
            },
            "speech_text": " ".join(spoken_parts).strip()
        }

    @classmethod
    def _extractive_answer(cls, results: list[dict[str, Any]], query: str = "") -> str:
        if not results:
            return NOT_FOUND
        top = results[0]
        data = cls._extractive_json(
            results, query,
            str(top.get("device", "Device")),
            str(top.get("manual_name", "Manual")),
            str(top.get("page_number", "?")),
            str(top.get("section_name", "General"))
        )
        return cls._format_json_to_markdown(data)

    def _call_llm(self, prompt: str) -> str:
        provider = self.settings.llm_provider.lower()
        if provider == "extractive":
            return NOT_FOUND
        if provider == "stub":
            return NOT_FOUND if "[Source" not in prompt else "[stub backend — no LLM configured] " + NOT_FOUND
        if provider == "ollama":
            import json
            import urllib.request
            payload = json.dumps({
                "model": self.settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 400}
            }).encode("utf-8")
            req = urllib.request.Request(f"{self.settings.ollama_base_url}/api/generate", data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (data.get("response") or "").strip()
        if provider == "gemini":
            if not self.settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini.")
            import json
            import urllib.request
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.settings.gemini_model}:generateContent?key={self.settings.gemini_api_key}"
            payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    return "".join(p.get("text", "") for p in parts).strip()
                return NOT_FOUND
        if provider == "groq":
            if not self.settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is required when LLM_PROVIDER=groq.")
            import requests
            url = "https://api.groq.com/openai/v1/chat/completions"
            models_to_try = [self.settings.groq_model, "qwen/qwen3.6-27b", "allam-2-7b", "groq/compound-mini"]
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.groq_api_key}",
                "User-Agent": "Fixora/1.0"
            }
            last_err = None
            for m in models_to_try:
                try:
                    payload = {
                        "model": m,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 1000
                    }
                    resp = requests.post(url, json=payload, headers=headers, timeout=20)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data["choices"][0]["message"]["content"].strip()
                except Exception as ex:
                    last_err = ex
            if last_err:
                raise last_err
            return NOT_FOUND
        if provider == "openai":
            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
            from openai import OpenAI
            response = OpenAI(api_key=self.settings.openai_api_key).chat.completions.create(
                model=self.settings.openai_model, messages=[{"role": "user", "content": prompt}], max_tokens=800,
            )
            return (response.choices[0].message.content or "").strip()
        if provider == "anthropic":
            if not self.settings.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic.")
            import anthropic
            response = anthropic.Anthropic(api_key=self.settings.anthropic_api_key).messages.create(
                model=self.settings.anthropic_model, max_tokens=800, messages=[{"role": "user", "content": prompt}],
            )
            return "".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()
        if provider == "qwen_local":
            if self._qwen_model is None:
                self._load_qwen_model()
            messages = [{"role": "user", "content": prompt}]
            text = self._qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._qwen_tokenizer([text], return_tensors="pt").to(self._qwen_model.device)
            import torch
            with torch.inference_mode():
                output_ids = self._qwen_model.generate(
                    **inputs,
                    max_new_tokens=self.settings.qwen_max_new_tokens,
                    do_sample=False,
                    eos_token_id=self._qwen_tokenizer.eos_token_id,
                    pad_token_id=self._qwen_tokenizer.pad_token_id or self._qwen_tokenizer.eos_token_id,
                )
            generated = output_ids[0][inputs.input_ids.shape[-1]:]
            return self._qwen_tokenizer.decode(generated, skip_special_tokens=True).strip()
        raise RuntimeError(f"Unsupported LLM_PROVIDER: {provider}")

    @staticmethod
    def validate(answer: str, context: str, results: list[dict[str, Any]]) -> dict[str, Any]:
        if context == NOT_FOUND:
            return {"status": "NOT_FOUND_IN_MANUAL", "issues": []}
        if answer == NOT_FOUND:
            return {"status": "NOT_FOUND_IN_MANUAL", "issues": []}
        issues: list[str] = []
        answer_upper, context_upper = answer.upper(), context.upper()
        answer_codes = {item.upper() for item in FAULT_CODE_PATTERN.findall(answer)}
        context_codes = {item.upper() for item in FAULT_CODE_PATTERN.findall(context)}
        if unexpected := answer_codes - context_codes:
            issues.append(f"Answer has error code(s) absent from retrieved context: {sorted(unexpected)}")
        for word in SAFETY_WORDS:
            if word in context_upper and word not in answer_upper:
                issues.append(f"Context contains {word}, which is absent from the answer.")
        has_citation = "[SOURCE" in answer_upper or any(str(item["manual_name"]).upper() in answer_upper for item in results)
        if not has_citation:
            issues.append("Answer does not cite a retrieved source.")
        return {"status": "FLAGGED" if issues else "FOUND_IN_MANUAL", "issues": issues}

    def _web_fallback(self, query: str, device_name: str | None, manufacturer_domain: str | None) -> tuple[str, list[dict[str, str]]]:
        if not self.settings.tavily_api_key:
            return f"{WEB_FALLBACK_LABEL}\nNo matching error code or troubleshooting procedure was found in the indexed manuals for: \"{query}\".\nPlease verify the error code format or specify the target device model.", []
        payload: dict[str, Any] = {"api_key": self.settings.tavily_api_key, "query": f"{device_name or ''} {query} official manual support".strip(), "max_results": 5, "search_depth": "advanced"}
        if manufacturer_domain:
            payload["include_domains"] = [manufacturer_domain]
        response = requests.post("https://api.tavily.com/search", json=payload, timeout=20)
        response.raise_for_status()
        hits = response.json().get("results", [])
        sources = [{"title": item.get("title", ""), "url": item.get("url", "")} for item in hits]
        if not sources:
            return f"{WEB_FALLBACK_LABEL}\nNo relevant official web results found.", []
        search_results = "\n\n".join(f"Title: {item['title']}\nURL: {item['url']}\nContent: {item.get('content', '')}" for item in hits)
        answer = self._call_llm(WEB_PROMPT.format(search_results=search_results, query=query))
        return (answer if answer.startswith(WEB_FALLBACK_LABEL) else f"{WEB_FALLBACK_LABEL}\n{answer}"), sources

    async def answer(self, query: str, device_name: str | None, manufacturer_domain: str | None, top_k: int | None) -> dict[str, Any]:
        if not self.ready:
            raise ServiceNotReadyError(self.startup_error or "Service is not ready.")
        results = await asyncio.to_thread(self.retrieve, query, device_name, top_k or self.settings.top_k)
        context = self.build_context(results)
        
        top_res = results[0] if results else {}
        manual_name = str(top_res.get("manual_name", "Unknown Manual"))
        page_number = str(top_res.get("page_number", "?"))
        section_name = str(top_res.get("section_name", "General"))
        device_label = str(top_res.get("device", device_name or "Medical Equipment"))

        parsed_json: dict[str, Any] = {}
        if context == NOT_FOUND:
            answer = NOT_FOUND
        elif self.settings.llm_provider.lower() == "extractive":
            parsed_json = self._extractive_json(results, query, device_label, manual_name, page_number, section_name)
            answer = self._format_json_to_markdown(parsed_json)
        else:
            prompt_text = MANUAL_PROMPT.format(
                device_name=device_label,
                manual_name=manual_name,
                page_number=page_number,
                section_name=section_name,
                context=context,
                question=query,
            )
            raw_llm_out = await asyncio.to_thread(self._call_llm, prompt_text)
            # Clean thinking tags from reasoning models
            cleaned_text = re.sub(r"<think>[\s\S]*?</think>", "", raw_llm_out, flags=re.IGNORECASE).strip()
            if cleaned_text.startswith("```"):
                cleaned_text = re.sub(r"^```(?:json)?\n|\n```$", "", cleaned_text, flags=re.MULTILINE).strip()
            
            # Robust JSON extraction
            json_match = re.search(r"(\{[\s\S]*\})", cleaned_text)
            candidate = json_match.group(1) if json_match else cleaned_text
            parsed_json = {}
            try:
                parsed_json = json.loads(candidate)
            except Exception:
                try:
                    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                    parsed_json = json.loads(fixed)
                except Exception:
                    # Fallback key-value regex extraction
                    m_speech = re.search(r'"speech_text"\s*:\s*"([^"]+)"', candidate)
                    m_fault = re.search(r'"fault_meaning"\s*:\s*"([^"]+)"', candidate)
                    m_safety = re.search(r'"safety_body"\s*:\s*"([^"]+)"', candidate)
                    if m_speech or m_fault or m_safety:
                        parsed_json = {
                            "fault_meaning": m_fault.group(1) if m_fault else "",
                            "speech_text": m_speech.group(1) if m_speech else "",
                            "safety_body": m_safety.group(1) if m_safety else "",
                            "has_high_priority_safety": bool(m_safety),
                        }

            if parsed_json.get("status") == "NOT_FOUND":
                answer = NOT_FOUND
            elif parsed_json:
                # Check safety override if context or query contains critical safety words
                if not parsed_json.get("has_high_priority_safety"):
                    if any(w in query.upper() or w in context.upper() for w in ["HIGH VOLTAGE", "HIGH-VOLTAGE", "RADIATION", "LETHAL"]):
                        parsed_json["has_high_priority_safety"] = True
                        if not parsed_json.get("safety_body"):
                            parsed_json["safety_body"] = "High voltage / hazard detected. Follow lockout/tagout (LOTO) protocols and power isolation before inspection."
                        if not parsed_json.get("safety_header"):
                            parsed_json["safety_header"] = "⚠️ HIGH PRIORITY SAFETY INSTRUCTIONS DETECTED"
                answer = self._format_json_to_markdown(parsed_json)
            else:
                # Never return raw unparsed JSON string as markdown
                if cleaned_text.startswith("{") and "}" in cleaned_text:
                    answer = re.sub(r'[{}\"\']', '', cleaned_text).strip()
                else:
                    answer = cleaned_text

        validation = self.validate(answer, context, results)
        used_web_fallback = context == NOT_FOUND or answer == NOT_FOUND
        if used_web_fallback:
            answer, sources = await asyncio.to_thread(self._web_fallback, query, device_name, manufacturer_domain)
            status, validation = "NOT_FOUND_IN_MANUAL", {"status": "NOT_FOUND_IN_MANUAL", "issues": []}
        else:
            sources = [{"manual": item["manual_name"], "page": item["page_number"], "device": item["device"], "retrieval_type": item["retrieval_type"]} for item in results]
            status = validation["status"]

        return {
            "answer": answer,
            "status": status,
            "sources": sources,
            "used_web_fallback": used_web_fallback,
            "validation": validation,
            "has_high_priority_safety": parsed_json.get("has_high_priority_safety", False),
            "safety_header": parsed_json.get("safety_header", ""),
            "safety_body": parsed_json.get("safety_body", ""),
            "fault_meaning": parsed_json.get("fault_meaning", ""),
            "checklist": parsed_json.get("checklist", []),
            "speech_text": parsed_json.get("speech_text") or answer,
            "source_citation": parsed_json.get("source_citation") or {"manual": manual_name, "page": page_number, "section": section_name},
        }
