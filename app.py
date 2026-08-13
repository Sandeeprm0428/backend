# ============================================================
#  app.py  –  Advanced Flask Chatbot Backend  (2000+ lines)
#  Run  :  python app.py
#  API  :  http://localhost:5000
#
#  Features:
#   ✅ Session management with TTL expiry
#   ✅ User authentication (register / login / JWT tokens)
#   ✅ Rate limiting per user/IP
#   ✅ Smart NLP keyword intent engine
#   ✅ Multi-language greeting detection
#   ✅ Sentiment analysis (rule-based)
#   ✅ Conversation context tracking
#   ✅ Typing indicators endpoint (SSE)
#   ✅ Message reactions (like / dislike)
#   ✅ Conversation tagging & search
#   ✅ File/image message support (metadata only)
#   ✅ Admin dashboard endpoints
#   ✅ Analytics & usage stats
#   ✅ Feedback / rating system
#   ✅ FAQ knowledge base (JSON-driven)
#   ✅ Webhook support
#   ✅ Export chat history (JSON / CSV)
#   ✅ Health check with system info
#   ✅ OpenAI / Gemini / Claude plug-in stubs
# ============================================================

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from datetime import datetime, timedelta
from functools import wraps
import uuid, os, json, time, re, csv, io, hashlib, hmac, random, logging, threading

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
class Config:
    SECRET_KEY        = os.getenv("SECRET_KEY", "dev-secret-key-change-in-prod")
    JWT_EXPIRY_HOURS  = int(os.getenv("JWT_EXPIRY_HOURS", 24))
    SESSION_TTL_MINS  = int(os.getenv("SESSION_TTL_MINS", 60))
    RATE_LIMIT_PER_MIN= int(os.getenv("RATE_LIMIT_PER_MIN", 30))
    MAX_MSG_LENGTH    = int(os.getenv("MAX_MSG_LENGTH", 2000))
    MAX_HISTORY_LEN   = int(os.getenv("MAX_HISTORY_LEN", 200))
    BOT_NAME          = os.getenv("BOT_NAME", "ChatBot")
    VERSION           = "2.0.0"
    START_TIME        = datetime.now()

cfg = Config()

# ── In-memory stores ──────────────────────────────────────────────────────────
sessions     = {}   # session_id → SessionData
users        = {}   # user_id    → UserData
tokens       = {}   # token      → user_id
rate_limits  = {}   # ip/user_id → [timestamps]
reactions    = {}   # msg_id     → {likes, dislikes, users}
feedback_log = []   # list of feedback entries
webhooks     = {}   # webhook_id → WebhookConfig
analytics    = {
    "total_messages": 0,
    "total_sessions": 0,
    "total_users":    0,
    "intents_hit":    {},
    "hourly_traffic": {},
    "errors":         0,
}

# ── Locks for thread safety ───────────────────────────────────────────────────
session_lock  = threading.Lock()
user_lock     = threading.Lock()
analytics_lock= threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
#  DATA MODELS (dict-based, no ORM needed)
# ─────────────────────────────────────────────────────────────────────────────

def make_session(session_id, user_id=None, tags=None):
    return {
        "id":           session_id,
        "user_id":      user_id,
        "messages":     [],
        "created_at":   datetime.now().isoformat(),
        "updated_at":   datetime.now().isoformat(),
        "tags":         tags or [],
        "archived":     False,
        "title":        "New conversation",
        "sentiment_avg": 0.0,
        "message_count": 0,
        "context":      {},   # carries conversation context between turns
    }

def make_message(role, content, msg_type="text", meta=None):
    return {
        "id":        str(uuid.uuid4()),
        "role":      role,          # "user" | "assistant" | "system"
        "content":   content,
        "type":      msg_type,      # "text" | "image" | "file" | "card"
        "timestamp": datetime.now().isoformat(),
        "meta":      meta or {},
        "sentiment": None,
        "edited":    False,
        "deleted":   False,
    }

def make_user(user_id, username, email, password_hash):
    return {
        "id":           user_id,
        "username":     username,
        "email":        email,
        "password_hash":password_hash,
        "created_at":   datetime.now().isoformat(),
        "last_login":   None,
        "role":         "user",       # "user" | "admin"
        "preferences":  {
            "language":  "en",
            "theme":     "dark",
            "bot_name":  cfg.BOT_NAME,
        },
        "session_count": 0,
        "message_count": 0,
        "active":        True,
    }

# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def hash_password(pwd: str) -> str:
    return hashlib.sha256((pwd + cfg.SECRET_KEY).encode()).hexdigest()

def make_token(user_id: str) -> str:
    raw   = f"{user_id}:{time.time()}:{uuid.uuid4()}"
    token = hashlib.sha256(raw.encode()).hexdigest()
    tokens[token] = {
        "user_id":   user_id,
        "created_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(hours=cfg.JWT_EXPIRY_HOURS)).isoformat(),
    }
    return token

def validate_token(token: str):
    if not token or token not in tokens:
        return None
    entry = tokens[token]
    if datetime.now() > datetime.fromisoformat(entry["expires_at"]):
        del tokens[token]
        return None
    return entry["user_id"]

def get_client_id(req) -> str:
    """Return user_id if auth header present, else fall back to IP."""
    auth = req.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        uid = validate_token(auth[7:])
        if uid:
            return uid
    return req.remote_addr or "unknown"

def now_iso() -> str:
    return datetime.now().isoformat()

def record_analytic(intent: str):
    with analytics_lock:
        analytics["total_messages"] += 1
        analytics["intents_hit"][intent] = analytics["intents_hit"].get(intent, 0) + 1
        hour = datetime.now().strftime("%Y-%m-%d %H:00")
        analytics["hourly_traffic"][hour] = analytics["hourly_traffic"].get(hour, 0) + 1

def prune_sessions():
    """Remove sessions older than SESSION_TTL_MINS with no messages."""
    cutoff = datetime.now() - timedelta(minutes=cfg.SESSION_TTL_MINS)
    to_del = []
    with session_lock:
        for sid, sess in sessions.items():
            if not sess["messages"]:
                updated = datetime.fromisoformat(sess["updated_at"])
                if updated < cutoff:
                    to_del.append(sid)
        for sid in to_del:
            del sessions[sid]
    return len(to_del)

# ─────────────────────────────────────────────────────────────────────────────
#  RATE LIMITER
# ─────────────────────────────────────────────────────────────────────────────

def is_rate_limited(client_id: str) -> bool:
    now_ts = time.time()
    window = 60  # 1 minute
    if client_id not in rate_limits:
        rate_limits[client_id] = []
    # keep only timestamps inside the window
    rate_limits[client_id] = [t for t in rate_limits[client_id] if now_ts - t < window]
    if len(rate_limits[client_id]) >= cfg.RATE_LIMIT_PER_MIN:
        return True
    rate_limits[client_id].append(now_ts)
    return False

def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        client_id = get_client_id(request)
        if is_rate_limited(client_id):
            return jsonify({
                "error":   "Rate limit exceeded",
                "message": f"Max {cfg.RATE_LIMIT_PER_MIN} requests/minute. Please slow down.",
                "retry_after": 60,
            }), 429
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────────────────────────────────────
#  AUTH DECORATOR
# ─────────────────────────────────────────────────────────────────────────────

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        user_id = validate_token(auth[7:])
        if not user_id:
            return jsonify({"error": "Token expired or invalid"}), 401
        request.current_user_id = user_id
        return f(*args, **kwargs)
    return decorated

def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401
        user_id = validate_token(auth[7:])
        if not user_id or users.get(user_id, {}).get("role") != "admin":
            return jsonify({"error": "Admin access required"}), 403
        request.current_user_id = user_id
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────────────────────────────────────
#  FAQ KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────

FAQ_KB = {
    "leave policy": (
        "📋 **Leave Policy**\n"
        "• Annual Leave: 18 days/year\n"
        "• Sick Leave: 12 days/year\n"
        "• Casual Leave: 6 days/year\n"
        "• Maternity Leave: 26 weeks\n"
        "• Paternity Leave: 5 days\n"
        "Apply via the HR portal under 'Leave Balance'."
    ),
    "salary": (
        "💰 **Salary & Payroll**\n"
        "Salaries are credited on the last working day of each month.\n"
        "Payslips are available under 'Letter / Payslip' in the HR portal.\n"
        "Contact payroll@utthunga.com for queries."
    ),
    "attendance": (
        "📊 **Attendance**\n"
        "Office hours: 9:00 AM – 6:00 PM IST\n"
        "Check-in via the HR app or biometric system.\n"
        "WFH must be pre-approved by your manager."
    ),
    "holiday": (
        "🏖️ **Holidays 2025**\n"
        "Public holidays are listed in the 'Holiday Calendar' section.\n"
        "There are 10 mandatory + 4 optional holidays this year."
    ),
    "performance": (
        "📈 **Performance Review**\n"
        "Reviews are conducted bi-annually: April and October.\n"
        "Self-assessment forms are available in the HR portal."
    ),
    "training": (
        "🎓 **Training & Development**\n"
        "Online courses are available via the LMS.\n"
        "Reimbursement up to ₹15,000/year for external certifications."
    ),
    "it support": (
        "💻 **IT Support**\n"
        "Raise a ticket at: helpdesk@utthunga.com\n"
        "For urgent issues: IT Helpline ext. 100\n"
        "Working hours: Mon–Fri, 9 AM – 7 PM"
    ),
    "reimbursement": (
        "🧾 **Expense Reimbursement**\n"
        "Submit claims via the HR portal within 30 days.\n"
        "Attach original bills and manager approval email.\n"
        "Processing time: 5–7 working days."
    ),
    "insurance": (
        "🏥 **Medical Insurance**\n"
        "Coverage: ₹3 lakhs/year per employee\n"
        "Family cover available at a subsidized rate.\n"
        "Contact HR for adding dependents."
    ),
    "transfer": (
        "🔄 **Transfer / Relocation**\n"
        "Requests must be submitted 60 days in advance.\n"
        "Relocation allowance is provided as per policy.\n"
        "Contact your HR Business Partner."
    ),
    "resignation": (
        "🚪 **Resignation Process**\n"
        "Notice period: 60 days (as per offer letter)\n"
        "Submit resignation via HR portal or email to hr@utthunga.com\n"
        "Clearance form must be completed before last working day."
    ),
    "referral": (
        "👥 **Employee Referral Program**\n"
        "Referral bonus: ₹10,000–₹25,000 (role-based)\n"
        "Paid after referred employee completes 90 days.\n"
        "Submit referrals via the HR portal."
    ),
    "appraisal": (
        "⭐ **Appraisal Process**\n"
        "Bi-annual reviews in April and October.\n"
        "Rating scale: 1 (Needs Improvement) – 5 (Exceptional)\n"
        "Increment letters issued within 30 days of review."
    ),
    "onboarding": (
        "🎉 **Onboarding**\n"
        "Day 1: Orientation at 9:30 AM in the main conference room.\n"
        "Laptop and access cards will be ready.\n"
        "Buddy program assigned for first 30 days."
    ),
}

def search_faq(msg: str):
    msg_lower = msg.lower()
    for key, answer in FAQ_KB.items():
        if key in msg_lower:
            return answer
    return None

# ─────────────────────────────────────────────────────────────────────────────
#  SENTIMENT ANALYSIS (rule-based)
# ─────────────────────────────────────────────────────────────────────────────

POSITIVE_WORDS = {
    "great","good","awesome","excellent","fantastic","love","perfect","amazing",
    "wonderful","happy","thanks","thank","appreciate","helpful","nice","brilliant",
    "super","brilliant","outstanding","impressive","well done","best","beautiful",
    "glad","pleased","enjoy","joy","excited","magnificent","superb","splendid",
}
NEGATIVE_WORDS = {
    "bad","terrible","awful","hate","horrible","worst","useless","broken","fail",
    "angry","frustrated","annoyed","disappointed","upset","sad","wrong","stupid",
    "idiot","rubbish","garbage","pathetic","disgrace","nonsense","poor","boring",
}

def analyze_sentiment(text: str) -> dict:
    words = set(re.findall(r'\b\w+\b', text.lower()))
    pos   = len(words & POSITIVE_WORDS)
    neg   = len(words & NEGATIVE_WORDS)
    if pos > neg:
        label, score = "positive", round(0.5 + min(pos * 0.1, 0.49), 2)
    elif neg > pos:
        label, score = "negative", round(0.5 - min(neg * 0.1, 0.49), 2)
    else:
        label, score = "neutral", 0.5
    return {"label": label, "score": score, "pos": pos, "neg": neg}

# ─────────────────────────────────────────────────────────────────────────────
#  MULTI-LANGUAGE GREETINGS
# ─────────────────────────────────────────────────────────────────────────────

GREETINGS = {
    "en": ["hi","hello","hey","howdy","greetings","sup","what's up","yo"],
    "es": ["hola","buenos días","buenas tardes","buenas noches"],
    "fr": ["bonjour","salut","bonsoir","coucou"],
    "de": ["hallo","guten morgen","guten tag","guten abend","hi"],
    "hi": ["नमस्ते","नमस्कार","हेलो","हाय"],
    "kn": ["ನಮಸ್ಕಾರ","ಹೆಲ್ಲೋ","ಹಾಯ್"],
    "ta": ["வணக்கம்","ஹலோ"],
    "te": ["నమస్కారం","హలో"],
}

GREETING_REPLIES = {
    "en": "Hello! 👋 How can I help you today?",
    "es": "¡Hola! 👋 ¿Cómo puedo ayudarte hoy?",
    "fr": "Bonjour! 👋 Comment puis-je vous aider?",
    "de": "Hallo! 👋 Wie kann ich Ihnen helfen?",
    "hi": "नमस्ते! 👋 आज मैं आपकी कैसे मदद कर सकता हूँ?",
    "kn": "ನಮಸ್ಕಾರ! 👋 ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
    "ta": "வணக்கம்! 👋 நான் உங்களுக்கு எப்படி உதவ முடியும்?",
    "te": "నమస్కారం! 👋 నేను మీకు ఎలా సహాయం చేయగలను?",
}

def detect_language_and_greet(msg: str):
    msg_lower = msg.lower().strip()
    for lang, words in GREETINGS.items():
        for w in words:
            if msg_lower.startswith(w) or msg_lower == w:
                return GREETING_REPLIES.get(lang, GREETING_REPLIES["en"])
    return None

# ─────────────────────────────────────────────────────────────────────────────
#  INTENT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
    "Why did the developer go broke? Because he used up all his cache! 💸",
    "What's a computer's favorite snack? Microchips! 🍟",
    "How do you comfort a JavaScript bug? You console it. 😄",
    "Why did the function break up with the variable? It had too many arguments. 💔",
    "What's a Python developer's favourite song? 'Baby got Booleans'! 🎵",
    "Why do Java developers wear glasses? Because they don't C#! 👓",
    "What is a computer's first sign of old age? Loss of memory! 💾",
    "Why was the math book sad? It had too many problems. 📚",
    "What do you call a bear with no teeth? A gummy bear! 🐻",
]

MOTIVATIONAL = [
    "💪 Keep pushing! Every expert was once a beginner.",
    "🌟 You're doing great — progress over perfection!",
    "🚀 Dream big, work hard, stay focused.",
    "✨ Every line of code you write is a step forward!",
    "🎯 Success is the sum of small efforts repeated daily.",
    "💡 The best time to start was yesterday; the next best time is now.",
    "🔥 Stay hungry, stay foolish, stay coding!",
]

TECH_FACTS = [
    "🌐 The first website went live on August 6, 1991.",
    "💾 The first hard drive (1956) weighed over a ton and stored 5MB.",
    "🐧 Linux powers over 90% of the world's servers.",
    "🐍 Python was named after Monty Python, not the snake.",
    "⚛️ React was created by Jordan Walke at Facebook in 2011.",
    "☁️ Amazon AWS launched in 2006 with just one service: S3.",
    "🤖 The word 'robot' comes from the Czech word 'robota' meaning forced labour.",
    "📱 The first iPhone was announced by Steve Jobs on January 9, 2007.",
]

MATH_KEYWORDS = re.compile(
    r"(\d+)\s*([+\-*/^%])\s*(\d+)|what is (\d+)\s*([+\-*/^%])\s*(\d+)"
)

def do_math(a, op, b):
    a, b = float(a), float(b)
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/": return (a / b) if b != 0 else None
    if op == "^": return a ** b
    if op == "%": return a % b
    return None

def get_bot_response(user_message: str, history: list, context: dict, preferences: dict) -> tuple:
    """
    Returns (reply_text, intent_label).
    context: mutable dict shared across turns.
    preferences: user prefs (language, etc.)
    """
    msg     = user_message.strip()
    msg_lo  = msg.lower()
    intent  = "unknown"

    # ── 1. FAQ knowledge base ───────────────────────────────────────────────
    faq_hit = search_faq(msg_lo)
    if faq_hit:
        return faq_hit, "faq"

    # ── 2. Multi-language greeting ──────────────────────────────────────────
    greet = detect_language_and_greet(msg_lo)
    if greet:
        # remember the user's name if they told us in a prior turn
        name = context.get("user_name", "")
        if name:
            greet = greet.replace("!", f", {name}!")
        return greet, "greeting"

    # ── 3. Maths calculator ─────────────────────────────────────────────────
    m = MATH_KEYWORDS.search(msg_lo)
    if m:
        a  = m.group(1) or m.group(4)
        op = m.group(2) or m.group(5)
        b  = m.group(3) or m.group(6)
        result = do_math(a, op, b)
        if result is not None:
            result_str = int(result) if result == int(result) else round(result, 6)
            return f"🧮 {a} {op} {b} = {result_str} ", "math"
        return "⚠️ Division by zero is undefined!", "math_error"

    # ── 4. Remember user's name ─────────────────────────────────────────────
    name_match = re.search(r"my name is ([A-Za-z]+)", msg_lo)
    if name_match:
        name = name_match.group(1).capitalize()
        context["user_name"] = name
        return f"Nice to meet you, {name}! 😊 I'll remember your name.", "name_set"

    # ── 5. What's my name? ───────────────────────────────────────────────────
    if "my name" in msg_lo and "?" in msg:
        name = context.get("user_name")
        if name:
            return f"Your name is **{name}**! 😊", "name_recall"
        return "I don't know your name yet! Tell me with 'My name is ...'", "name_unknown"

    # ── 6. How are you / feelings ───────────────────────────────────────────
    if any(p in msg_lo for p in ["how are you","how r u","you ok","are you ok","you good"]):
        return "I'm running perfectly, thank you! 😄 How can I help you today?", "bot_status"

    # ── 7. Bot identity ─────────────────────────────────────────────────────
    if any(p in msg_lo for p in ["who are you","your name","what are you","what's your name"]):
        bot_name = preferences.get("bot_name", cfg.BOT_NAME)
        return (
            f"I'm **{bot_name}**, your AI-powered assistant built with Python & React! 🤖\n"
            "I can answer HR questions, tell jokes, do maths, and much more.\n"
            "Type `help` to see everything I can do."
        ), "identity"

    # ── 8. Help menu ─────────────────────────────────────────────────────────
    if msg_lo in ["help","?","commands","what can you do"]:
        return (
            "🆘 **Here's what I can do:**\n\n"
            "📋 **HR Topics** — leave, salary, attendance, holidays, performance, training\n"
            "🧮 **Calculator** — e.g. `15 * 7` or `what is 100 / 4`\n"
            "😂 **Jokes** — type `joke`\n"
            "💪 **Motivation** — type `motivate me`\n"
            "🌐 **Tech facts** — type `tech fact`\n"
            "⏰ **Time & Date** — type `time` or `date`\n"
            "🔄 **Flip a coin** — type `flip coin`\n"
            "🎲 **Roll a dice** — type `roll dice`\n"
            "🌡️ **Convert temp** — e.g. `100 celsius to fahrenheit`\n"
            "📏 **Unit convert** — e.g. `5 km to miles`\n"
            "📝 **Count words** — type `count words: your text here`\n"
            "🔤 **Reverse text** — type `reverse: your text`\n"
            "🔐 **Remember name** — type `my name is Sandeep`\n"
            "🌈 **Quote of the day** — type `quote`\n"
            "👋 **Goodbye** — type `bye`"
        ), "help"

    # ── 9. Goodbye ───────────────────────────────────────────────────────────
    if any(p in msg_lo for p in ["bye","goodbye","see you","see ya","exit","quit","later","cya"]):
        name = context.get("user_name", "")
        farewell = f"Goodbye, {name}! 👋" if name else "Goodbye! 👋"
        return farewell + " Have a great day! Feel free to come back anytime. 😊", "goodbye"

    # ── 10. Joke ─────────────────────────────────────────────────────────────
    if any(p in msg_lo for p in ["joke","funny","make me laugh","tell me a joke","lol"]):
        context["last_joke"] = True
        return random.choice(JOKES), "joke"

    if context.get("last_joke") and any(p in msg_lo for p in ["another","more","again","one more"]):
        return random.choice(JOKES), "joke_repeat"

    # ── 11. Motivation ───────────────────────────────────────────────────────
    if any(p in msg_lo for p in ["motivat","inspire","encourage","pick me up","cheer me"]):
        return random.choice(MOTIVATIONAL), "motivation"

    # ── 12. Tech fact ────────────────────────────────────────────────────────
    if any(p in msg_lo for p in ["tech fact","fun fact","did you know","interesting fact"]):
        return random.choice(TECH_FACTS), "tech_fact"

    # ── 13. Time ─────────────────────────────────────────────────────────────
    if re.search(r"\btime\b", msg_lo) and "?" not in msg_lo.replace("time?",""):
        now = datetime.now().strftime("%I:%M:%S %p")
        date = datetime.now().strftime("%A, %d %B %Y")
        return f"⏰ Current server time: **{now}**\n📅 {date}", "time"

    # ── 14. Date ─────────────────────────────────────────────────────────────
    if any(p in msg_lo for p in ["today","what's the date","current date","what day"]):
        today = datetime.now().strftime("%A, %d %B %Y")
        return f"📅 Today is **{today}**", "date"

    # ── 15. Flip coin ─────────────────────────────────────────────────────────
    if any(p in msg_lo for p in ["flip coin","flip a coin","heads or tails","coin toss"]):
        result = random.choice(["Heads 🪙", "Tails 🪙"])
        return f"Flipping the coin... **{result}**!", "coin_flip"

    # ── 16. Roll dice ─────────────────────────────────────────────────────────
    if any(p in msg_lo for p in ["roll dice","roll a dice","roll the dice","dice"]):
        n = random.randint(1, 6)
        faces = ["","⚀","⚁","⚂","⚃","⚄","⚅"]
        return f"Rolling the dice... You got **{n}** {faces[n]}!", "dice_roll"

    # ── 17. Temperature conversion ───────────────────────────────────────────
    temp_c2f = re.search(r"(\d+\.?\d*)\s*c(?:elsius)?\s*to\s*f(?:ahrenheit)?", msg_lo)
    temp_f2c = re.search(r"(\d+\.?\d*)\s*f(?:ahrenheit)?\s*to\s*c(?:elsius)?", msg_lo)
    if temp_c2f:
        c = float(temp_c2f.group(1))
        f = round(c * 9/5 + 32, 2)
        return f"🌡️ {c}°C = **{f}°F**", "temp_convert"
    if temp_f2c:
        f = float(temp_f2c.group(1))
        c = round((f - 32) * 5/9, 2)
        return f"🌡️ {f}°F = **{c}°C**", "temp_convert"

    # ── 18. Distance conversion ──────────────────────────────────────────────
    km2mi = re.search(r"(\d+\.?\d*)\s*km\s*to\s*mi(?:les?)?", msg_lo)
    mi2km = re.search(r"(\d+\.?\d*)\s*mi(?:les?)?\s*to\s*km", msg_lo)
    if km2mi:
        km = float(km2mi.group(1))
        mi = round(km * 0.621371, 3)
        return f"📏 {km} km = **{mi} miles**", "unit_convert"
    if mi2km:
        mi = float(mi2km.group(1))
        km = round(mi / 0.621371, 3)
        return f"📏 {mi} miles = **{km} km**", "unit_convert"

    # ── 19. Word count ───────────────────────────────────────────────────────
    wc = re.search(r"count words?:\s*(.+)", msg_lo)
    if wc:
        text    = wc.group(1).strip()
        words   = len(text.split())
        chars   = len(text)
        sentences = len(re.split(r'[.!?]+', text))
        return (
            f"📝 **Word count:** {words}\n"
            f"📊 **Characters:** {chars}\n"
            f"🔖 **Sentences (approx):** {sentences}"
        ), "word_count"

    # ── 20. Reverse text ─────────────────────────────────────────────────────
    rv = re.search(r"reverse:\s*(.+)", msg_lo)
    if rv:
        original  = rv.group(1).strip()
        reversed_ = original[::-1]
        return f"🔄 **Reversed:** {reversed_}", "reverse_text"

    # ── 21. Quote of the day ─────────────────────────────────────────────────
    QUOTES = [
        ("The best way to predict the future is to create it.", "Abraham Lincoln"),
        ("Innovation distinguishes between a leader and a follower.", "Steve Jobs"),
        ("Code is like humor. When you have to explain it, it's bad.", "Cory House"),
        ("First, solve the problem. Then, write the code.", "John Johnson"),
        ("Experience is the name everyone gives to their mistakes.", "Oscar Wilde"),
        ("The only way to do great work is to love what you do.", "Steve Jobs"),
        ("It always seems impossible until it's done.", "Nelson Mandela"),
        ("In the middle of every difficulty lies opportunity.", "Albert Einstein"),
        ("Talk is cheap. Show me the code.", "Linus Torvalds"),
        ("Simplicity is the soul of efficiency.", "Austin Freeman"),
    ]
    if any(p in msg_lo for p in ["quote","quotation","wise words","inspire quote","daily quote"]):
        q, author = random.choice(QUOTES)
        return f"💬 *\"{q}\"*\n— **{author}**", "quote"

    # ── 22. Python / Tech topics ─────────────────────────────────────────────
    if "python" in msg_lo:
        return (
            "🐍 **Python** is a high-level, versatile programming language.\n"
            "Great for: backend (Flask/Django), data science (Pandas/NumPy), AI/ML, automation.\n"
            "Current stable version: Python 3.12\n"
            "Learn more: https://python.org"
        ), "tech_python"

    if "react" in msg_lo:
        return (
            "⚛️ **React** is a JavaScript library for building UIs, maintained by Meta.\n"
            "Key concepts: Components, JSX, Hooks, State, Props, Virtual DOM.\n"
            "Current stable version: React 18\n"
            "Learn more: https://react.dev"
        ), "tech_react"

    if "flask" in msg_lo:
        return (
            "🌶️ **Flask** is a lightweight Python web framework.\n"
            "Best for: REST APIs, microservices, prototyping.\n"
            "Install: `pip install flask`\n"
            "Docs: https://flask.palletsprojects.com"
        ), "tech_flask"

    if "git" in msg_lo:
        return (
            "🔀 **Git** is the world's most popular version control system.\n\n"
            "**Common commands:**\n"
            "• `git init` — initialize repo\n"
            "• `git clone <url>` — clone repo\n"
            "• `git add .` — stage changes\n"
            "• `git commit -m 'message'` — commit\n"
            "• `git push` — push to remote\n"
            "• `git pull` — pull latest changes\n"
            "• `git status` — check status"
        ), "tech_git"

    if "sql" in msg_lo or "database" in msg_lo or "db" in msg_lo:
        return (
            "🗄️ **Databases & SQL**\n\n"
            "**Popular databases:**\n"
            "• PostgreSQL — powerful relational DB\n"
            "• MySQL — widely used open-source\n"
            "• SQLite — lightweight, file-based\n"
            "• MongoDB — NoSQL document store\n"
            "• Redis — in-memory key-value store\n\n"
            "**Common SQL commands:**\n"
            "`SELECT`, `INSERT`, `UPDATE`, `DELETE`, `JOIN`, `GROUP BY`"
        ), "tech_db"

    if "docker" in msg_lo:
        return (
            "🐳 **Docker** packages apps into containers for consistent deployment.\n\n"
            "**Key commands:**\n"
            "• `docker build -t app .`\n"
            "• `docker run -p 5000:5000 app`\n"
            "• `docker ps` — list running containers\n"
            "• `docker stop <id>` — stop container\n"
            "• `docker-compose up` — multi-service apps"
        ), "tech_docker"

    if "api" in msg_lo:
        return (
            "🔌 **API (Application Programming Interface)**\n\n"
            "An API allows different software to communicate.\n\n"
            "**REST API methods:**\n"
            "• `GET` — Retrieve data\n"
            "• `POST` — Create new data\n"
            "• `PUT/PATCH` — Update data\n"
            "• `DELETE` — Remove data\n\n"
            "**Status codes:**\n"
            "200 OK | 201 Created | 400 Bad Request | 401 Unauthorized | 404 Not Found | 500 Server Error"
        ), "tech_api"

    # ── 23. Positive sentiment encouragement ─────────────────────────────────
    sentiment = analyze_sentiment(msg)
    if sentiment["label"] == "negative":
        return (
            "I'm sorry to hear that. 😔 Things can be tough sometimes.\n"
            "If you need HR support, type `help` to see available options.\n"
            "Remember: every challenge is an opportunity to grow! 💪"
        ), "empathy_negative"

    if sentiment["label"] == "positive" and sentiment["score"] > 0.7:
        return (
            f"That's great to hear! 😊 {random.choice(MOTIVATIONAL)}\n"
            "Is there anything I can help you with today?"
        ), "empathy_positive"

    # ── 24. Thank you ─────────────────────────────────────────────────────────
    if any(p in msg_lo for p in ["thank","thanks","thx","ty","thank you","appreciate"]):
        return "You're welcome! 😊 Feel free to ask me anything anytime!", "thanks"

    # ── 25. Sorry / apology ───────────────────────────────────────────────────
    if any(p in msg_lo for p in ["sorry","apolog","excuse me","my bad","i'm sorry"]):
        return "No worries at all! 😊 How can I help you today?", "apology"

    # ── 26. Swear word / abusive (gentle redirect) ───────────────────────────
    SWEAR_PATTERNS = ["damn","hell","wtf","shut up","idiot","stupid"]
    if any(p in msg_lo for p in SWEAR_PATTERNS):
        return "Let's keep our conversation friendly! 😊 How can I help you?", "profanity_redirect"

    # ── 27. Unknown — context-aware fallback ──────────────────────────────────
    fallbacks = [
        f"I'm not sure about that. Try typing `help` to see what I can do! 😊",
        f"Hmm, I didn't quite catch that. Could you rephrase? Type `help` for a full list of commands.",
        f"I'm still learning! 🤖 Try asking about HR topics, jokes, or type `help`.",
        f"That's an interesting question! I'm not sure I have an answer yet. Type `help` for options.",
    ]
    return random.choice(fallbacks), "unknown"


# ─────────────────────────────────────────────────────────────────────────────
#  WEBHOOK DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

def fire_webhook(event: str, payload: dict):
    """Fire registered webhooks asynchronously (best-effort)."""
    import urllib.request
    for wh_id, wh in list(webhooks.items()):
        if event in wh.get("events", []):
            try:
                body = json.dumps({"event": event, "payload": payload, "ts": now_iso()}).encode()
                req  = urllib.request.Request(
                    wh["url"],
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=3)
            except Exception as e:
                log.warning(f"Webhook {wh_id} failed: {e}")

def async_webhook(event: str, payload: dict):
    t = threading.Thread(target=fire_webhook, args=(event, payload), daemon=True)
    t.start()

# ─────────────────────────────────────────────────────────────────────────────
#  ROUTE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ok(data: dict, status: int = 200):
    return jsonify({"success": True, **data}), status

def err(msg: str, status: int = 400):
    with analytics_lock:
        analytics["errors"] += 1
    return jsonify({"success": False, "error": msg}), status

def get_session(session_id: str):
    return sessions.get(session_id)

def get_or_create_session(session_id: str, user_id=None):
    with session_lock:
        if session_id not in sessions:
            sessions[session_id] = make_session(session_id, user_id)
            with analytics_lock:
                analytics["total_sessions"] += 1
        return sessions[session_id]

# ─────────────────────────────────────────────────────────────────────────────
#  ██████╗  ██████╗ ██╗   ██╗████████╗███████╗███████╗
#  ██╔══██╗██╔═══██╗██║   ██║╚══██╔══╝██╔════╝██╔════╝
#  ██████╔╝██║   ██║██║   ██║   ██║   █████╗  ███████╗
#  ██╔══██╗██║   ██║██║   ██║   ██║   ██╔══╝  ╚════██║
#  ██║  ██║╚██████╔╝╚██████╔╝   ██║   ███████╗███████║
#  ╚═╝  ╚═╝ ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚══════╝
# ─────────────────────────────────────────────────────────────────────────────

# ── Health & Status ───────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def root():
    return ok({
        "message":  "ChatBot API is running ✅",
        "version":  cfg.VERSION,
        "bot_name": cfg.BOT_NAME,
    })

@app.route("/api/health", methods=["GET"])
def health():
    uptime = str(datetime.now() - cfg.START_TIME).split(".")[0]
    return ok({
        "status":         "healthy",
        "version":        cfg.VERSION,
        "uptime":         uptime,
        "sessions_active": len(sessions),
        "users_registered": len(users),
        "total_messages": analytics["total_messages"],
        "timestamp":      now_iso(),
    })

@app.route("/api/info", methods=["GET"])
def info():
    return ok({
        "bot_name":        cfg.BOT_NAME,
        "version":         cfg.VERSION,
        "max_msg_length":  cfg.MAX_MSG_LENGTH,
        "rate_limit_rpm":  cfg.RATE_LIMIT_PER_MIN,
        "session_ttl_min": cfg.SESSION_TTL_MINS,
        "features": [
            "session_management","user_auth","rate_limiting","sentiment_analysis",
            "multi_language","faq_kb","math_calculator","unit_converter",
            "reactions","tags","feedback","webhooks","export","analytics",
        ],
    })

# ── Auth ──────────────────────────────────────────────────────────────────────

@app.route("/api/auth/register", methods=["POST"])
def register():
    body = request.get_json(silent=True) or {}
    username = (body.get("username") or "").strip()
    email    = (body.get("email")    or "").strip().lower()
    password = (body.get("password") or "").strip()

    if not username or not email or not password:
        return err("username, email, and password are required")
    if len(password) < 6:
        return err("Password must be at least 6 characters")
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return err("Invalid email address")

    # check for duplicate
    with user_lock:
        for u in users.values():
            if u["email"] == email:
                return err("Email already registered", 409)
        user_id = str(uuid.uuid4())
        users[user_id] = make_user(user_id, username, email, hash_password(password))
        with analytics_lock:
            analytics["total_users"] += 1

    token = make_token(user_id)
    log.info(f"New user registered: {email}")
    return ok({"message": "Registered successfully", "token": token, "user_id": user_id}, 201)

@app.route("/api/auth/login", methods=["POST"])
def login():
    body     = request.get_json(silent=True) or {}
    email    = (body.get("email")    or "").strip().lower()
    password = (body.get("password") or "").strip()

    if not email or not password:
        return err("email and password are required")

    with user_lock:
        user = next((u for u in users.values() if u["email"] == email), None)

    if not user or user["password_hash"] != hash_password(password):
        return err("Invalid email or password", 401)
    if not user["active"]:
        return err("Account is deactivated. Contact HR.", 403)

    token = make_token(user["id"])
    with user_lock:
        users[user["id"]]["last_login"] = now_iso()

    log.info(f"User login: {email}")
    return ok({
        "message":  "Login successful",
        "token":    token,
        "user_id":  user["id"],
        "username": user["username"],
        "role":     user["role"],
    })

@app.route("/api/auth/logout", methods=["POST"])
@require_auth
def logout():
    auth  = request.headers.get("Authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else ""
    if token in tokens:
        del tokens[token]
    return ok({"message": "Logged out successfully"})

@app.route("/api/auth/me", methods=["GET"])
@require_auth
def me():
    user = users.get(request.current_user_id)
    if not user:
        return err("User not found", 404)
    safe = {k: v for k, v in user.items() if k != "password_hash"}
    return ok({"user": safe})

@app.route("/api/auth/preferences", methods=["PATCH"])
@require_auth
def update_preferences():
    body  = request.get_json(silent=True) or {}
    user  = users.get(request.current_user_id)
    if not user:
        return err("User not found", 404)
    allowed = {"language", "theme", "bot_name"}
    for k, v in body.items():
        if k in allowed:
            users[request.current_user_id]["preferences"][k] = v
    return ok({"preferences": users[request.current_user_id]["preferences"]})

# ── Session management ────────────────────────────────────────────────────────

@app.route("/api/session", methods=["POST"])
@rate_limit
def create_session_route():
    body    = request.get_json(silent=True) or {}
    user_id = None
    auth    = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        user_id = validate_token(auth[7:])

    session_id = str(uuid.uuid4())
    sess = get_or_create_session(session_id, user_id)
    if user_id and user_id in users:
        with user_lock:
            users[user_id]["session_count"] += 1

    # optional initial tags
    tags = body.get("tags", [])
    if isinstance(tags, list):
        sess["tags"] = tags

    return ok({"session_id": session_id, "created_at": sess["created_at"]}, 201)

@app.route("/api/sessions", methods=["GET"])
def list_sessions_route():
    user_id = None
    auth    = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        user_id = validate_token(auth[7:])

    result = []
    for sid, s in sessions.items():
        if user_id and s.get("user_id") and s["user_id"] != user_id:
            continue
        result.append({
            "session_id":    sid,
            "title":         s.get("title","New conversation"),
            "message_count": s["message_count"],
            "tags":          s["tags"],
            "archived":      s["archived"],
            "created_at":    s["created_at"],
            "updated_at":    s["updated_at"],
        })
    result.sort(key=lambda x: x["updated_at"], reverse=True)
    return ok({"sessions": result, "count": len(result)})

@app.route("/api/session/<session_id>", methods=["GET"])
def get_session_route(session_id):
    sess = get_session(session_id)
    if not sess:
        return err("Session not found", 404)
    return ok({"session": {
        "id":            session_id,
        "title":         sess.get("title",""),
        "tags":          sess["tags"],
        "archived":      sess["archived"],
        "created_at":    sess["created_at"],
        "updated_at":    sess["updated_at"],
        "message_count": sess["message_count"],
        "sentiment_avg": sess["sentiment_avg"],
    }})

@app.route("/api/session/<session_id>", methods=["PATCH"])
def update_session(session_id):
    sess = get_session(session_id)
    if not sess:
        return err("Session not found", 404)
    body = request.get_json(silent=True) or {}
    if "title" in body:
        sess["title"] = str(body["title"])[:100]
    if "tags" in body and isinstance(body["tags"], list):
        sess["tags"] = body["tags"]
    if "archived" in body:
        sess["archived"] = bool(body["archived"])
    sess["updated_at"] = now_iso()
    return ok({"session_id": session_id, "title": sess["title"], "tags": sess["tags"]})

@app.route("/api/session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    if session_id not in sessions:
        return err("Session not found", 404)
    with session_lock:
        del sessions[session_id]
    return ok({"message": "Session deleted", "session_id": session_id})

# ── Tags ──────────────────────────────────────────────────────────────────────

@app.route("/api/session/<session_id>/tags", methods=["POST"])
def add_tag(session_id):
    sess = get_session(session_id)
    if not sess:
        return err("Session not found", 404)
    body = request.get_json(silent=True) or {}
    tag  = (body.get("tag") or "").strip()
    if not tag:
        return err("tag is required")
    if tag not in sess["tags"]:
        sess["tags"].append(tag)
    return ok({"tags": sess["tags"]})

@app.route("/api/session/<session_id>/tags/<tag>", methods=["DELETE"])
def remove_tag(session_id, tag):
    sess = get_session(session_id)
    if not sess:
        return err("Session not found", 404)
    sess["tags"] = [t for t in sess["tags"] if t != tag]
    return ok({"tags": sess["tags"]})

# ── Chat (core) ───────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
@rate_limit
def chat():
    body         = request.get_json(silent=True) or {}
    user_message = (body.get("message") or "").strip()
    session_id   = body.get("session_id", "default")
    msg_type     = body.get("type", "text")   # text | image | file
    meta         = body.get("meta", {})       # extra metadata

    if not user_message:
        return err("message cannot be empty")
    if len(user_message) > cfg.MAX_MSG_LENGTH:
        return err(f"Message exceeds maximum length of {cfg.MAX_MSG_LENGTH} characters")

    # get or create session
    auth    = request.headers.get("Authorization", "")
    user_id = validate_token(auth[7:]) if auth.startswith("Bearer ") else None
    sess    = get_or_create_session(session_id, user_id)

    # enforce history limit
    if len(sess["messages"]) >= cfg.MAX_HISTORY_LEN:
        sess["messages"] = sess["messages"][-(cfg.MAX_HISTORY_LEN - 2):]

    # sentiment
    sentiment = analyze_sentiment(user_message)

    # user preferences
    prefs = {}
    if user_id and user_id in users:
        prefs = users[user_id].get("preferences", {})
        with user_lock:
            users[user_id]["message_count"] += 1

    # save user message
    user_msg = make_message("user", user_message, msg_type, meta)
    user_msg["sentiment"] = sentiment
    sess["messages"].append(user_msg)

    # auto-title after first message
    if sess["message_count"] == 0:
        sess["title"] = user_message[:60] + ("…" if len(user_message) > 60 else "")

    sess["message_count"] += 1
    sess["updated_at"] = now_iso()

    # update rolling sentiment average
    old_avg = sess["sentiment_avg"]
    count   = sess["message_count"]
    sess["sentiment_avg"] = round(
        (old_avg * (count - 1) + sentiment["score"]) / count, 3
    )

    # small processing pause (remove in production with real AI)
    time.sleep(0.2)

    # generate bot reply
    bot_text, intent = get_bot_response(
        user_message,
        sess["messages"],
        sess["context"],
        prefs,
    )

    bot_msg = make_message("assistant", bot_text)
    sess["messages"].append(bot_msg)
    sess["message_count"] += 1

    # analytics
    record_analytic(intent)

    # fire webhook
    async_webhook("message", {
        "session_id": session_id,
        "user_message": user_message,
        "bot_reply": bot_text,
        "intent": intent,
    })

    log.info(f"[{session_id[:8]}] intent={intent} sentiment={sentiment['label']}")

    return ok({
        "response":      bot_text,
        "message_id":    bot_msg["id"],
        "session_id":    session_id,
        "intent":        intent,
        "sentiment":     sentiment,
        "message_count": sess["message_count"],
        "timestamp":     bot_msg["timestamp"],
    })

# ── History ───────────────────────────────────────────────────────────────────

@app.route("/api/history/<session_id>", methods=["GET"])
def get_history(session_id):
    sess = get_session(session_id)
    if not sess:
        return err("Session not found", 404)

    page  = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 50))
    msgs  = [m for m in sess["messages"] if not m.get("deleted")]

    start = max(0, (page - 1) * limit)
    end   = start + limit
    page_msgs = msgs[start:end]

    return ok({
        "session_id":  session_id,
        "history":     page_msgs,
        "total":       len(msgs),
        "page":        page,
        "limit":       limit,
        "pages":       max(1, -(-len(msgs) // limit)),
    })

@app.route("/api/history/<session_id>", methods=["DELETE"])
def clear_history(session_id):
    sess = get_session(session_id)
    if not sess:
        return err("Session not found", 404)
    sess["messages"]      = []
    sess["message_count"] = 0
    sess["updated_at"]    = now_iso()
    sess["context"]       = {}
    sess["title"]         = "New conversation"
    return ok({"message": "History cleared", "session_id": session_id})

# ── Message operations ────────────────────────────────────────────────────────

@app.route("/api/history/<session_id>/message/<msg_id>", methods=["PATCH"])
def edit_message(session_id, msg_id):
    sess = get_session(session_id)
    if not sess:
        return err("Session not found", 404)
    body     = request.get_json(silent=True) or {}
    new_text = (body.get("content") or "").strip()
    if not new_text:
        return err("content is required")

    for m in sess["messages"]:
        if m["id"] == msg_id:
            if m["role"] != "user":
                return err("Only user messages can be edited")
            m["content"]    = new_text
            m["edited"]     = True
            m["edited_at"]  = now_iso()
            return ok({"message": m})
    return err("Message not found", 404)

@app.route("/api/history/<session_id>/message/<msg_id>", methods=["DELETE"])
def delete_message(session_id, msg_id):
    sess = get_session(session_id)
    if not sess:
        return err("Session not found", 404)
    for m in sess["messages"]:
        if m["id"] == msg_id:
            m["deleted"]    = True
            m["content"]    = "[Message deleted]"
            m["deleted_at"] = now_iso()
            return ok({"message": "Message deleted"})
    return err("Message not found", 404)

@app.route("/api/history/<session_id>/search", methods=["GET"])
def search_history(session_id):
    sess = get_session(session_id)
    if not sess:
        return err("Session not found", 404)
    q    = (request.args.get("q") or "").lower().strip()
    if not q:
        return err("q (query) parameter required")
    hits = [
        m for m in sess["messages"]
        if not m.get("deleted") and q in m["content"].lower()
    ]
    return ok({"query": q, "results": hits, "count": len(hits)})

# ── Reactions ─────────────────────────────────────────────────────────────────

@app.route("/api/reaction/<msg_id>", methods=["POST"])
def react(msg_id):
    body      = request.get_json(silent=True) or {}
    reaction  = body.get("reaction")     # "like" | "dislike"
    client_id = get_client_id(request)

    if reaction not in ("like", "dislike"):
        return err("reaction must be 'like' or 'dislike'")

    if msg_id not in reactions:
        reactions[msg_id] = {"likes": 0, "dislikes": 0, "users": {}}

    prev = reactions[msg_id]["users"].get(client_id)
    if prev == reaction:
        # toggle off
        reactions[msg_id][prev + "s"] -= 1
        del reactions[msg_id]["users"][client_id]
    else:
        if prev:
            reactions[msg_id][prev + "s"] -= 1
        reactions[msg_id][reaction + "s"] += 1
        reactions[msg_id]["users"][client_id] = reaction

    return ok({
        "msg_id":   msg_id,
        "likes":    reactions[msg_id]["likes"],
        "dislikes": reactions[msg_id]["dislikes"],
    })

@app.route("/api/reaction/<msg_id>", methods=["GET"])
def get_reactions(msg_id):
    r = reactions.get(msg_id, {"likes": 0, "dislikes": 0})
    return ok({"msg_id": msg_id, "likes": r["likes"], "dislikes": r["dislikes"]})

# ── Typing indicator (SSE) ────────────────────────────────────────────────────

@app.route("/api/typing/<session_id>", methods=["GET"])
def typing_stream(session_id):
    """Server-Sent Events endpoint — sends typing state every 2s."""
    def generate():
        for _ in range(5):   # stream for ~10 seconds
            data = json.dumps({"typing": True, "session_id": session_id, "ts": now_iso()})
            yield f"data: {data}\n\n"
            time.sleep(2)
        yield f"data: {json.dumps({'typing': False})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

# ── Feedback / rating ─────────────────────────────────────────────────────────

@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    body       = request.get_json(silent=True) or {}
    session_id = body.get("session_id", "")
    rating     = body.get("rating")        # 1–5
    comment    = (body.get("comment") or "").strip()

    if rating is None or not (1 <= int(rating) <= 5):
        return err("rating must be 1–5")

    entry = {
        "id":         str(uuid.uuid4()),
        "session_id": session_id,
        "rating":     int(rating),
        "comment":    comment,
        "timestamp":  now_iso(),
        "client_id":  get_client_id(request),
    }
    feedback_log.append(entry)

    async_webhook("feedback", entry)
    log.info(f"Feedback received: rating={rating} session={session_id}")
    return ok({"message": "Feedback submitted. Thank you! 🙏", "id": entry["id"]}, 201)

@app.route("/api/feedback", methods=["GET"])
@require_admin
def get_feedback():
    page  = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 20))
    start = (page - 1) * limit
    page_data = feedback_log[start:start + limit]
    avg = round(sum(f["rating"] for f in feedback_log) / len(feedback_log), 2) if feedback_log else 0
    return ok({
        "feedback": page_data,
        "total":    len(feedback_log),
        "average_rating": avg,
    })

# ── Export ────────────────────────────────────────────────────────────────────

@app.route("/api/export/<session_id>", methods=["GET"])
def export_history(session_id):
    sess   = get_session(session_id)
    if not sess:
        return err("Session not found", 404)

    fmt = request.args.get("format", "json").lower()
    msgs = [m for m in sess["messages"] if not m.get("deleted")]

    if fmt == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["id","role","content","timestamp","type"])
        writer.writeheader()
        for m in msgs:
            writer.writerow({
                "id":        m["id"],
                "role":      m["role"],
                "content":   m["content"],
                "timestamp": m["timestamp"],
                "type":      m.get("type","text"),
            })
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=chat_{session_id[:8]}.csv"},
        )

    # default JSON
    return Response(
        json.dumps({"session_id": session_id, "history": msgs}, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": f"attachment; filename=chat_{session_id[:8]}.json"},
    )

# ── Sentiment summary ─────────────────────────────────────────────────────────

@app.route("/api/sentiment/<session_id>", methods=["GET"])
def sentiment_summary(session_id):
    sess = get_session(session_id)
    if not sess:
        return err("Session not found", 404)

    user_msgs = [m for m in sess["messages"] if m["role"] == "user" and m.get("sentiment")]
    if not user_msgs:
        return ok({"session_id": session_id, "sentiment": "no data"})

    sentiments = [m["sentiment"]["label"] for m in user_msgs]
    counts = {"positive": sentiments.count("positive"),
              "neutral":  sentiments.count("neutral"),
              "negative": sentiments.count("negative")}
    dominant = max(counts, key=counts.get)
    avg_score = round(sum(m["sentiment"]["score"] for m in user_msgs) / len(user_msgs), 3)

    return ok({
        "session_id":  session_id,
        "dominant":    dominant,
        "avg_score":   avg_score,
        "counts":      counts,
        "total_msgs":  len(user_msgs),
    })

# ── FAQ ───────────────────────────────────────────────────────────────────────

@app.route("/api/faq", methods=["GET"])
def list_faq():
    q = (request.args.get("q") or "").lower().strip()
    if q:
        filtered = {k: v for k, v in FAQ_KB.items() if q in k}
    else:
        filtered = FAQ_KB
    return ok({"faqs": list(filtered.keys()), "count": len(filtered)})

@app.route("/api/faq/<topic>", methods=["GET"])
def get_faq(topic):
    topic_lo = topic.lower().replace("-", " ")
    answer   = FAQ_KB.get(topic_lo)
    if not answer:
        return err(f"No FAQ found for topic: {topic}", 404)
    return ok({"topic": topic_lo, "answer": answer})

@app.route("/api/faq", methods=["POST"])
@require_admin
def add_faq():
    body   = request.get_json(silent=True) or {}
    topic  = (body.get("topic") or "").strip().lower()
    answer = (body.get("answer") or "").strip()
    if not topic or not answer:
        return err("topic and answer are required")
    FAQ_KB[topic] = answer
    return ok({"message": "FAQ added", "topic": topic}, 201)

@app.route("/api/faq/<topic>", methods=["DELETE"])
@require_admin
def delete_faq(topic):
    topic_lo = topic.lower().replace("-", " ")
    if topic_lo not in FAQ_KB:
        return err("FAQ not found", 404)
    del FAQ_KB[topic_lo]
    return ok({"message": "FAQ deleted", "topic": topic_lo})

# ── Webhooks ──────────────────────────────────────────────────────────────────

@app.route("/api/webhooks", methods=["GET"])
@require_admin
def list_webhooks():
    return ok({"webhooks": list(webhooks.values()), "count": len(webhooks)})

@app.route("/api/webhooks", methods=["POST"])
@require_admin
def create_webhook():
    body   = request.get_json(silent=True) or {}
    url    = (body.get("url") or "").strip()
    events = body.get("events", ["message"])

    if not url:
        return err("url is required")
    if not url.startswith("http"):
        return err("url must start with http:// or https://")

    wh_id = str(uuid.uuid4())
    webhooks[wh_id] = {
        "id":         wh_id,
        "url":        url,
        "events":     events,
        "created_at": now_iso(),
    }
    return ok({"webhook_id": wh_id, "url": url, "events": events}, 201)

@app.route("/api/webhooks/<wh_id>", methods=["DELETE"])
@require_admin
def delete_webhook(wh_id):
    if wh_id not in webhooks:
        return err("Webhook not found", 404)
    del webhooks[wh_id]
    return ok({"message": "Webhook deleted", "id": wh_id})

# ── Analytics ─────────────────────────────────────────────────────────────────

@app.route("/api/analytics", methods=["GET"])
@require_admin
def get_analytics():
    top_intents = sorted(
        analytics["intents_hit"].items(),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    return ok({
        "total_messages":     analytics["total_messages"],
        "total_sessions":     analytics["total_sessions"],
        "total_users":        analytics["total_users"],
        "total_feedback":     len(feedback_log),
        "avg_feedback_rating": (
            round(sum(f["rating"] for f in feedback_log) / len(feedback_log), 2)
            if feedback_log else 0
        ),
        "top_intents":        top_intents,
        "active_sessions":    len(sessions),
        "active_tokens":      len(tokens),
        "errors_count":       analytics["errors"],
        "uptime":             str(datetime.now() - cfg.START_TIME).split(".")[0],
        "hourly_traffic":     dict(list(sorted(
            analytics["hourly_traffic"].items(), reverse=True
        ))[:24]),
    })

# ── Admin — User management ───────────────────────────────────────────────────

@app.route("/api/admin/users", methods=["GET"])
@require_admin
def admin_list_users():
    page  = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 20))
    start = (page - 1) * limit
    user_list = [
        {k: v for k, v in u.items() if k != "password_hash"}
        for u in list(users.values())[start:start + limit]
    ]
    return ok({"users": user_list, "total": len(users), "page": page})

@app.route("/api/admin/users/<user_id>", methods=["PATCH"])
@require_admin
def admin_update_user(user_id):
    if user_id not in users:
        return err("User not found", 404)
    body = request.get_json(silent=True) or {}
    if "role" in body and body["role"] in ("user", "admin"):
        users[user_id]["role"] = body["role"]
    if "active" in body:
        users[user_id]["active"] = bool(body["active"])
    return ok({"user_id": user_id, "role": users[user_id]["role"], "active": users[user_id]["active"]})

@app.route("/api/admin/prune", methods=["POST"])
@require_admin
def admin_prune():
    deleted = prune_sessions()
    return ok({"message": f"Pruned {deleted} empty/expired sessions"})

@app.route("/api/admin/broadcast", methods=["POST"])
@require_admin
def admin_broadcast():
    """Inject a system message into all active sessions."""
    body = request.get_json(silent=True) or {}
    text = (body.get("message") or "").strip()
    if not text:
        return err("message is required")
    count = 0
    for sess in sessions.values():
        sess["messages"].append(make_message("system", text))
        count += 1
    return ok({"message": f"Broadcast sent to {count} sessions"})

# ── Misc ──────────────────────────────────────────────────────────────────────

@app.route("/api/ping", methods=["GET"])
def ping():
    return ok({"pong": True, "ts": now_iso()})

@app.route("/api/session/<session_id>/archive", methods=["POST"])
def archive_session(session_id):
    sess = get_session(session_id)
    if not sess:
        return err("Session not found", 404)
    sess["archived"] = True
    sess["updated_at"] = now_iso()
    return ok({"message": "Session archived", "session_id": session_id})

@app.route("/api/session/<session_id>/unarchive", methods=["POST"])
def unarchive_session(session_id):
    sess = get_session(session_id)
    if not sess:
        return err("Session not found", 404)
    sess["archived"] = False
    sess["updated_at"] = now_iso()
    return ok({"message": "Session unarchived", "session_id": session_id})

@app.route("/api/suggest", methods=["GET"])
def suggestions():
    """Return quick-reply suggestions based on a context hint."""
    ctx = (request.args.get("context") or "").lower()
    defaults = [
        "Tell me a joke", "What's today's date?", "Help",
        "Leave policy", "Salary info", "Motivate me",
    ]
    if "leave" in ctx:
        return ok({"suggestions": ["Leave policy", "Holiday calendar", "Leave balance", "Apply leave"]})
    if "salary" in ctx:
        return ok({"suggestions": ["Salary info", "Payslip download", "Reimbursement", "Insurance"]})
    return ok({"suggestions": defaults})

# ── AI Provider stubs ─────────────────────────────────────────────────────────

@app.route("/api/ai/openai", methods=["POST"])
def stub_openai():
    """
    Stub — replace body with real openai.chat.completions.create()
    pip install openai
    """
    body = request.get_json(silent=True) or {}
    return ok({
        "stub": True,
        "message": (
            "To enable OpenAI, install the SDK and add your API key:\n"
            "  pip install openai\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "Then replace get_bot_response() with openai.chat.completions.create()."
        ),
        "model": body.get("model", "gpt-4o-mini"),
    })

@app.route("/api/ai/claude", methods=["POST"])
def stub_claude():
    """
    Stub — replace with anthropic.Anthropic().messages.create()
    pip install anthropic
    """
    return ok({
        "stub": True,
        "message": (
            "To enable Claude:\n"
            "  pip install anthropic\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "Then call anthropic.Anthropic().messages.create() inside get_bot_response()."
        ),
    })

@app.route("/api/ai/gemini", methods=["POST"])
def stub_gemini():
    """
    Stub — replace with google.generativeai
    pip install google-generativeai
    """
    return ok({
        "stub": True,
        "message": (
            "To enable Gemini:\n"
            "  pip install google-generativeai\n"
            "  export GOOGLE_API_KEY=...\n"
            "Then call genai.GenerativeModel('gemini-1.5-flash').generate_content()."
        ),
    })

# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return err("Endpoint not found", 404)

@app.errorhandler(405)
def method_not_allowed(e):
    return err("Method not allowed", 405)

@app.errorhandler(500)
def internal_error(e):
    log.error(f"Internal error: {e}")
    return err("Internal server error", 500)

# ── Request / response logging ────────────────────────────────────────────────

@app.before_request
def before():
    request._start = time.time()

@app.after_request
def after(response):
    if hasattr(request, "_start"):
        ms = round((time.time() - request._start) * 1000, 1)
        log.info(f"{request.method} {request.path} → {response.status_code} ({ms}ms)")
    response.headers["X-Bot-Version"] = cfg.VERSION
    response.headers["X-Powered-By"]  = "Flask + ChatBot"
    return response

# ── Startup seed: create default admin ───────────────────────────────────────

def seed_admin():
    admin_email = os.getenv("ADMIN_EMAIL", "admin@utthunga.com")
    admin_pass  = os.getenv("ADMIN_PASSWORD", "Admin@123")
    if not any(u["email"] == admin_email for u in users.values()):
        uid = str(uuid.uuid4())
        users[uid] = make_user(uid, "Admin", admin_email, hash_password(admin_pass))
        users[uid]["role"] = "admin"
        log.info(f"Default admin created: {admin_email} / {admin_pass}")

# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    seed_admin()
    print("\n" + "="*60)
    print(f"  🚀  {cfg.BOT_NAME} API v{cfg.VERSION}")
    print(f"  📡  http://localhost:5000")
    print(f"  🔑  Admin: admin@utthunga.com / Admin@123")
    print("="*60)
    print("\n  Key endpoints:")
    print("  POST /api/auth/register  — create account")
    print("  POST /api/auth/login     — get token")
    print("  POST /api/session        — start chat session")
    print("  POST /api/chat           — send message")
    print("  GET  /api/health         — status check")
    print("  GET  /api/analytics      — usage stats (admin)")
    print("  GET  /api/export/<sid>   — download chat")
    print("  GET  /api/faq            — HR knowledge base")
    print("="*60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)