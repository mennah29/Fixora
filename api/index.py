from http.server import BaseHTTPRequestHandler
import json
import os
import re
from pathlib import Path
import urllib.request
import urllib.error

# Load device fault chunks in memory
CHUNKS_DATA = []
possible_paths = [
    Path.cwd() / "data" / "all_device_fault_chunks.json",
    Path(__file__).resolve().parent.parent / "data" / "all_device_fault_chunks.json",
    Path(__file__).resolve().parent / "data" / "all_device_fault_chunks.json",
    Path("/var/task/data/all_device_fault_chunks.json"),
]

for p in possible_paths:
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                CHUNKS_DATA = json.load(f)
            break
        except Exception:
            pass

def extract_error_codes(query: str) -> list:
    pattern = re.compile(r"\b(?:ERR(?:OR)?[\s\-]*CODE|ERR(?:OR)?|CODE|FAULT|ALARM|E|F)[\s\-]*0*(\d{1,7})\b", re.IGNORECASE)
    matches = pattern.findall(query)
    direct = re.findall(r"\b([EF]\d{1,5})\b", query, re.IGNORECASE)
    return list(dict.fromkeys(matches + direct))

def retrieve_chunks(query: str, device_name: str = None, top_k: int = 5) -> list:
    if not CHUNKS_DATA:
        return []
    
    q_lower = query.lower()
    q_words = set(re.findall(r"\w+", q_lower))
    codes = extract_error_codes(query)
    
    scored = []
    for chunk in CHUNKS_DATA:
        dev = str(chunk.get("device", "")).lower()
        if device_name and device_name.lower() not in ("all devices", "any", "none", ""):
            if device_name.lower() not in dev and dev not in device_name.lower():
                continue
        
        text = str(chunk.get("text", ""))
        text_lower = text.lower()
        manual = str(chunk.get("manual", "")).lower()
        score = 0.0
        
        for c in codes:
            if re.search(r"\b" + re.escape(c) + r"\b", text, re.IGNORECASE):
                score += 15.0
                if any(w in text.upper() for w in ("ERR", "FAULT", "ALARM", "RANGE", "FAIL", "CHECK", "REPLACE", "BATTERY")):
                    score += 10.0
        
        overlap = sum(1 for w in q_words if w in text_lower or w in manual)
        score += overlap * 1.2
        
        if score > 0:
            scored.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "text": text,
                "manual_name": chunk.get("manual", "Service Manual"),
                "page_number": chunk.get("page", "1"),
                "section_name": chunk.get("type", "General"),
                "device": chunk.get("device", "Medical Equipment"),
                "score": score
            })
    
    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]

def call_groq_llm(query: str, context_chunks: list, device_name: str = None) -> dict:
    api_key = os.getenv("GROQ_API_KEY", "").strip() or os.getenv("GROQ_KEY", "").strip()
    
    if not context_chunks:
        return {
            "status": "NOT_FOUND_IN_MANUAL",
            "answer": "This fault or code was not found in the indexed equipment service manuals.",
            "speech_text": "I checked the service manuals, but I could not find a procedure for this specific request.",
            "has_high_priority_safety": False,
            "checklist": [],
            "source_citation": {"manual": "Manual", "page": "N/A"}
        }
    
    context_text = "\n\n---\n\n".join([
        f"[Source {i+1}] Device: {c['device']} | Manual: {c['manual_name']} (Page {c['page_number']})\n{c['text']}"
        for i, c in enumerate(context_chunks)
    ])
    
    system_prompt = f"""You are Fixora, an elite industrial biomedical AI assistant guiding a technician on-site.
Provide clear, grounded troubleshooting steps based ONLY on the provided manual context.

STRICT RULES:
1. Base all advice strictly on the manual context.
2. Flag high-voltage, radiation, chemical, or pressure hazards as high priority safety.
3. Return ONLY valid raw JSON matching this schema:
{{
  "has_high_priority_safety": boolean,
  "safety_header": "⚠️ HIGH PRIORITY SAFETY INSTRUCTIONS DETECTED" (or null),
  "safety_body": "Critical safety warning details" (or null),
  "fault_meaning": "Plain English explanation of what this error or symptom means.",
  "checklist": [
    "Step 1: Description of check or action",
    "Step 2: Description of check or action"
  ],
  "source_citation": {{
    "manual": "{context_chunks[0]['manual_name']}",
    "page": "{context_chunks[0]['page_number']}"
  }},
  "speech_text": "Conversational, natural spoken explanation for voice mode."
}}"""

    user_prompt = f"Target Device: {device_name or 'General'}\nUser Query: {query}\n\nManual Context:\n{context_text}"
    
    if api_key:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Fixora/1.0"
        }
        payload = {
            "model": os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"}
        }
        try:
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                if resp.status == 200:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    content = res_json["choices"][0]["message"]["content"].strip()
                    content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
                    parsed = json.loads(content)
                    
                    parts = []
                    if parsed.get("has_high_priority_safety") and parsed.get("safety_body"):
                        parts.append(f"### {parsed.get('safety_header', '⚠️ HIGH PRIORITY SAFETY INSTRUCTIONS DETECTED')}\n**{parsed['safety_body']}**\n")
                    if parsed.get("fault_meaning"):
                        parts.append(f"**Fault Overview:** {parsed['fault_meaning']}\n")
                    if parsed.get("checklist"):
                        parts.append("### 🔧 Step-by-Step Checklist")
                        for s in parsed["checklist"]:
                            parts.append(f"- {s}")
                    if parsed.get("source_citation"):
                        c = parsed["source_citation"]
                        parts.append(f"\n[{c.get('manual', 'Manual')} - p.{c.get('page', '?')}]")
                    parsed["status"] = "FOUND_IN_MANUAL"
                    parsed["answer"] = "\n".join(parts)
                    return parsed
        except Exception as e:
            print(f"Groq API call error: {e}")
    
    top = context_chunks[0]
    is_hazard = any(w in (query + " " + top["text"]).upper() for w in ("HIGH VOLTAGE", "LOTO", "SHOCK", "LETHAL", "RADIATION", "HAZARD", "DANGER"))
    checklist = [line.strip() for line in top["text"].splitlines() if len(line.strip()) > 10][:5]
    if not checklist:
        checklist = ["Inspect connectors and check diagnostic test points per manual schematics."]
        
    return {
        "status": "FOUND_IN_MANUAL",
        "has_high_priority_safety": is_hazard,
        "safety_header": "⚠️ HIGH PRIORITY SAFETY INSTRUCTIONS DETECTED" if is_hazard else None,
        "safety_body": "Lethal voltage / hazardous condition detected. Follow Lockout/Tagout (LOTO) protocols before opening panels." if is_hazard else None,
        "fault_meaning": f"Procedure extracted from {top['manual_name']} (Page {top['page_number']}).",
        "checklist": checklist,
        "source_citation": {"manual": top["manual_name"], "page": top["page_number"]},
        "speech_text": f"I found the procedure in the service manual on page {top['page_number']}.",
        "answer": f"**Manual Reference ({top['manual_name']} p.{top['page_number']}):**\n\n{top['text']}"
    }

class handler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        self._set_headers(200)
        res = {"status": "ok", "message": "Fixora API is active", "chunks_loaded": len(CHUNKS_DATA)}
        self.wfile.write(json.dumps(res).encode("utf-8"))

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}
            query = data.get("query", "")
            device_name = data.get("device_name", None)
            top_k = int(data.get("top_k", 5))

            chunks = retrieve_chunks(query, device_name, top_k)
            result = call_groq_llm(query, chunks, device_name)
            
            self._set_headers(200)
            self.wfile.write(json.dumps(result).encode("utf-8"))
        except Exception as e:
            self._set_headers(200)
            err_res = {
                "status": "ERROR",
                "answer": f"Error processing query: {str(e)}",
                "speech_text": "An error occurred while processing your request.",
                "checklist": [],
                "source_citation": {"manual": "System", "page": "N/A"}
            }
            self.wfile.write(json.dumps(err_res).encode("utf-8"))

