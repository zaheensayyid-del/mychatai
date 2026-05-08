"""
MyChatAI — AI chat + image generation + internet image search.
"""

import json, os, threading, time, re, random
import urllib.parse
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
from memory import Memory

HERE     = os.path.dirname(os.path.abspath(__file__))
PORT     = int(os.environ.get("PORT", 8080))
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OLLAMA   = "http://localhost:11434"

# API key: env variable takes priority, then config.json
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_KEY:
    cfg_path = os.path.join(HERE, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            _cfg = json.load(f)
        GROQ_KEY = _cfg.get("groq_api_key", "")

GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
OLLAMA_MODEL = "llama3.2"

# per-session memory cache
_sessions: dict = {}
_sessions_lock = threading.Lock()

def get_memory(session_id: str) -> Memory:
    with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = Memory(session_id)
        return _sessions[session_id]


# ══ IMAGE DETECTION ════════════════════════════════════════════════════════

# Patterns that mean "generate/draw an AI image"
_GEN_PATTERNS = [
    r'(?:generate|create|make|draw|paint|design|render)\s+(?:an?\s+)?(?:image|picture|photo|illustration|artwork|drawing|painting|sketch)\s+(?:of\s+)?(.+)',
    r'(?:an?\s+)?(?:image|picture|illustration|drawing|artwork)\s+of\s+(.+)',
    r'(?:draw|illustrate|paint|render)\s+(?:me\s+)?(.+)',
    r'(?:generate|create|make)\s+(?:me\s+)?(?:an?\s+)?(?:ai\s+)?(?:image|pic|photo)\s+(?:of\s+|showing\s+)?(.+)',
]

# Patterns that mean "find/search a real photo from the internet"
_SEARCH_PATTERNS = [
    r'(?:find|search|show|get|look up)\s+(?:an?\s+)?(?:image|picture|photo)\s+(?:of|for)\s+(.+)',
    r'(?:find|search|show|get)\s+(?:me\s+)?(?:an?\s+)?(?:image|picture|photo)\s+(?:of|for)\s+(.+)',
    r'(?:internet\s+)?(?:image|picture|photo)\s+(?:search\s+)?(?:of|for)\s+(.+)',
    r'show\s+me\s+(?:an?\s+)?(?:image|picture|photo)\s+of\s+(.+)',
    r'(?:real|actual)\s+(?:image|photo|picture)\s+of\s+(.+)',
]

def detect_image_request(text: str):
    """
    Returns ('generate', subject) | ('search', subject) | (None, None)
    """
    t = text.lower().strip().rstrip('?.!')

    # search/find first (more specific)
    for pat in _SEARCH_PATTERNS:
        m = re.search(pat, t)
        if m:
            return 'search', m.group(1).strip()

    # generate
    for pat in _GEN_PATTERNS:
        m = re.search(pat, t)
        if m:
            return 'generate', m.group(1).strip()

    # loose keywords
    gen_kw  = ['generate image', 'create image', 'make image', 'draw me', 'generate a pic', 'ai image of']
    srch_kw = ['find image', 'search image', 'internet photo', 'real photo of', 'show image of']
    for kw in gen_kw:
        if kw in t:
            subject = t.split(kw)[-1].strip().lstrip('of a an the').strip()
            return 'generate', subject or t
    for kw in srch_kw:
        if kw in t:
            subject = t.split(kw)[-1].strip().lstrip('of a an the').strip()
            return 'search', subject or t

    return None, None


# ══ IMAGE GENERATION — Pollinations.ai (free, no key) ═════════════════════

def generate_image(prompt: str) -> str:
    seed = random.randint(1, 99999)
    clean = re.sub(r'[^\w\s,.-]', '', prompt)[:300]
    encoded = urllib.parse.quote(clean)
    return (f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width=800&height=500&nologo=true&seed={seed}&enhance=true")


# ══ INTERNET IMAGE SEARCH — DuckDuckGo + Wikimedia fallback ══════════════

def _ddg_search(query: str, headers: dict) -> list:
    """DuckDuckGo image search."""
    try:
        r = requests.get("https://duckduckgo.com/",
                         params={"q": query, "iax": "images", "ia": "images"},
                         headers=headers, timeout=6)
        token = None
        for pat in [r'vqd=(["\'])([^"\']+)\1', r'"vqd":"([^"]+)"',
                    r'vqd=([\d-]+)', r"vqd%3D([^&%]+)"]:
            m = re.search(pat, r.text)
            if m:
                token = m.group(2) if len(m.groups()) > 1 else m.group(1)
                break
        if not token:
            return []
        r2 = requests.get("https://duckduckgo.com/i.js",
                          params={"l":"us-en","o":"json","q":query,
                                  "vqd":token,"f":",,,,,","p":"1"},
                          headers=headers, timeout=6)
        results = r2.json().get("results", [])
        return [x["image"] for x in results[:4] if x.get("image")]
    except Exception:
        return []


def _wikimedia_search(query: str) -> list:
    """Wikimedia Commons image search — always free, no key needed."""
    try:
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={"action":"query","generator":"search","gsrnamespace":"6",
                    "gsrsearch":query,"gsrlimit":"8","prop":"imageinfo",
                    "iiprop":"url","iiurlwidth":"800","format":"json"},
            timeout=8)
        pages = r.json().get("query", {}).get("pages", {})
        urls = []
        for page in pages.values():
            ii  = page.get("imageinfo", [{}])[0]
            url = ii.get("thumburl") or ii.get("url", "")
            if url and any(url.lower().endswith(e)
                           for e in (".jpg",".jpeg",".png",".webp")):
                urls.append(url)
            if len(urls) >= 4:
                break
        return urls
    except Exception:
        return []


def search_image(query: str) -> list:
    """Return up to 4 image URLs — tries DuckDuckGo first, then Wikimedia."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    urls = _ddg_search(query, headers)
    if not urls:
        urls = _wikimedia_search(query)
    return urls


# ══ GROQ STREAMING ═════════════════════════════════════════════════════════

def stream_groq(all_msgs, system, handler):
    full = []
    ok   = False
    try:
        resp = requests.post(
            GROQ_URL,
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "system", "content": system}] + all_msgs,
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 512,
            },
            headers={"Authorization": f"Bearer {GROQ_KEY}",
                     "Content-Type": "application/json"},
            stream=True, timeout=30,
        )
        if resp.ok:
            ok = True
            for raw in resp.iter_lines():
                if not raw: continue
                line = raw.decode()
                if not line.startswith("data: "): continue
                chunk = line[6:]
                if chunk == "[DONE]": break
                try:
                    token = json.loads(chunk)["choices"][0]["delta"].get("content", "")
                    if token:
                        full.append(token)
                        handler.wfile.write(
                            f"data: {json.dumps({'token': token})}\n\n".encode())
                        handler.wfile.flush()
                except Exception:
                    continue
    except Exception:
        pass
    return ok, "".join(full)


def ollama_available():
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=2)
        return any(OLLAMA_MODEL in m["name"] for m in r.json().get("models", []))
    except Exception:
        return False


def stream_ollama(all_msgs, system, handler):
    try:
        resp = requests.post(
            f"{OLLAMA}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "system", "content": system}] + all_msgs,
                "stream": False,
                "options": {"temperature": 0.75, "num_predict": 512},
            },
            timeout=45,
        )
        if resp.ok:
            reply = resp.json()["message"]["content"]
            for word in re.split(r'(\s+)', reply):
                if word:
                    handler.wfile.write(
                        f"data: {json.dumps({'token': word})}\n\n".encode())
                    handler.wfile.flush()
                    time.sleep(0.01)
            return True
    except Exception:
        pass
    return False


# ══ HTTP HANDLER ════════════════════════════════════════════════════════════

def _sse_start(handler):
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()

def _sse_send(handler, data: dict):
    handler.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
    handler.wfile.flush()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200); self._cors(); self.end_headers()

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/status":
            self._json(200, {"ready": True, "ollama": ollama_available()})

        elif self.path.startswith("/profile"):
            from urllib.parse import urlparse, parse_qs
            qs  = parse_qs(urlparse(self.path).query)
            sid = qs.get("session_id", ["default"])[0]
            self._json(200, get_memory(sid).get_summary())

        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        n    = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)

        if self.path == "/chat":
            try:
                data = json.loads(body)
            except Exception:
                self._json(400, {"error": "bad json"}); return

            user_msg = data.get("message", "").strip()
            history  = data.get("history", [])
            if not user_msg:
                self._json(400, {"error": "empty"}); return

            session_id = data.get("session_id", "default")
            mem = get_memory(session_id)
            mem.process_message(user_msg)
            system = mem.build_system_prompt()

            msgs = [{"role": t["role"], "content": t["content"]}
                    for t in history[-6:]
                    if t.get("role") in ("user", "assistant") and t.get("content")]
            all_msgs = msgs + [{"role": "user", "content": user_msg}]

            # ── Detect image request ───────────────────────────────────
            img_type, subject = detect_image_request(user_msg)

            _sse_start(self)

            if img_type == 'generate':
                # AI image generation via Pollinations
                img_url = generate_image(subject)
                _sse_send(self, {"token": f"Here's your image of **{subject}**:\n\n"})
                _sse_send(self, {"image": img_url, "caption": subject})

            elif img_type == 'search':
                # Real photos from DuckDuckGo
                _sse_send(self, {"token": f"Searching the internet for **{subject}**...\n\n"})
                urls = search_image(subject)
                if urls:
                    _sse_send(self, {"token": f"Found {len(urls)} result{'s' if len(urls)>1 else ''}:\n\n"})
                    for url in urls:
                        _sse_send(self, {"image": url, "caption": subject})
                else:
                    # fallback: generate one
                    _sse_send(self, {"token": "Couldn't find internet results — generating one instead:\n\n"})
                    img_url = generate_image(subject)
                    _sse_send(self, {"image": img_url, "caption": subject})

            else:
                # ── Normal chat — stream Groq ──────────────────────────
                ok, _ = stream_groq(all_msgs, system, self)
                if not ok and ollama_available():
                    stream_ollama(all_msgs, system, self)

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            mem.save()

        elif self.path == "/profile/reset":
            data = json.loads(body or b'{}')
            session_id = data.get("session_id", "default")
            mem = get_memory(session_id)
            path = mem._path()
            if os.path.exists(path): os.remove(path)
            with _sessions_lock:
                _sessions.pop(session_id, None)
            self._json(200, {"ok": True})

        else:
            self._json(404, {"error": "not found"})


if __name__ == "__main__":
    print(f"\n  MyChatAI")
    print(f"  http://localhost:{PORT}  —  Ctrl+C to stop\n")
    try:
        HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
