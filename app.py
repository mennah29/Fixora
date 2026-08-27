import streamlit as st
import json
import urllib.request
import os
import re
import uuid
import streamlit.components.v1 as components

# =============================================================================
# 1. Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Fixora — Industrial AI Assistant",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 2. Session State Initialization (Multi-Chat History & Registration)
# =============================================================================
if "registered" not in st.session_state:
    st.session_state.registered = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "device_selected" not in st.session_state:
    st.session_state.device_selected = "Siemens Servo 900 Ventilator"
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "chat"
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Multi-Chat History Store: { chat_id: { "title": str, "device": str, "messages": list } }
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {
        "default": {
            "title": "Welcome Session",
            "device": "Siemens Servo 900 Ventilator",
            "messages": []
        }
    }
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = "default"

# Helper for current active chat
def get_current_messages():
    cid = st.session_state.current_chat_id
    if cid not in st.session_state.chat_history:
        st.session_state.chat_history[cid] = {
            "title": "New Chat",
            "device": st.session_state.device_selected,
            "messages": []
        }
    return st.session_state.chat_history[cid]["messages"]

# =============================================================================
# 3. Design Tokens (Fixora Brand + Modern ChatGPT Architecture)
# =============================================================================
BRAND = "#0b496b"           # Fixora Deep Navy
ACTIVE = "#188a72"          # Fixora Emerald Teal
AI_MINT = "#d3f0cb"         # Fixora Mint Accent
BORDER_ICE = "#cae7e8"      # Fixora Ice Blue Border
CRITICAL = "#D92D20"        # Safety Hazard Red
WARNING_AMBER = "#E6A817"

is_dark = st.session_state.theme == "dark"

T = {
    "bg":           "#071923"   if is_dark else "#ffffff",
    "panel":        "#0d2430"   if is_dark else "#f9fbfc",
    "sidebar_bg":   "#091c27"   if is_dark else "#f3f7f8",
    "text":         "#edf5f7"   if is_dark else "#0d1b22",
    "text2":        "#8fa8b3"   if is_dark else "#5c737d",
    "border":       "#1b3a47"   if is_dark else "#e1edf0",
    "hover":        "#122f3c"   if is_dark else "#eaf2f4",
    "input_bg":     "#0d2430"   if is_dark else "#ffffff",
    "user_bubble":  "#133342"   if is_dark else "#edf6f8",
    "user_text":    "#edf5f7"   if is_dark else "#0b496b",
    "pill_bg":      "#102936"   if is_dark else "#ffffff",
    "pill_border":  "#1b3a47"   if is_dark else "#cae7e8",
}

# =============================================================================
# 4. SVG Icon Library
# =============================================================================
def svg(name, size=20, color=None):
    c = color or T["text2"]
    icons = {
        "logo": f'''<svg width="{size}" height="{size}" viewBox="0 0 32 32" fill="none">
            <polygon points="16,1 28.5,8.5 28.5,23.5 16,31 3.5,23.5 3.5,8.5" stroke="{BRAND}" stroke-width="2.2" fill="none"/>
            <circle cx="16" cy="16" r="6" stroke="{ACTIVE}" stroke-width="2" fill="none"/>
            <circle cx="16" cy="16" r="2.5" fill="{ACTIVE}"/>
            <line x1="16" y1="10" x2="16" y2="4" stroke="{BRAND}" stroke-width="1.8"/>
            <line x1="16" y1="22" x2="16" y2="28" stroke="{BRAND}" stroke-width="1.8"/>
            <line x1="10.8" y1="13" x2="6" y2="10" stroke="{BRAND}" stroke-width="1.8"/>
            <line x1="21.2" y1="19" x2="26" y2="22" stroke="{BRAND}" stroke-width="1.8"/>
        </svg>''',
        "new_chat": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
        </svg>''',
        "search": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>''',
        "library": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z"/><path d="M6 6h10"/><path d="M6 10h10"/>
        </svg>''',
        "msg": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>''',
        "trash": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
        </svg>''',
        "gear": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
        </svg>''',
        "warn": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{CRITICAL}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>''',
        "doc": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>
        </svg>''',
        "shield": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        </svg>''',
        "logout": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>
        </svg>''',
    }
    return icons.get(name, "")

# =============================================================================
# 5. Global CSS
# =============================================================================
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    /* Global Base */
    .stApp {{
        background-color: {T['bg']};
        color: {T['text']};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    /* Clean Transparent Header (Keep Native Sidebar Toggle) */
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}
    .stDeployButton, #MainMenu, footer {{
        display: none !important;
    }}
    
    /* Style Streamlit Native Sidebar Controls */
    button[kind="header"] {{
        color: {"#edf5f7" if is_dark else BRAND} !important;
    }}
    [data-testid="stSidebarCollapsedControl"] button {{
        color: {"#edf5f7" if is_dark else BRAND} !important;
    }}
    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 860px !important;
        margin: 0 auto !important;
    }}

    /* Sidebar - ChatGPT Style */
    div[data-testid="stSidebar"] {{
        background-color: {T['sidebar_bg']};
        border-right: 1px solid {T['border']};
        padding: 8px 12px;
    }}
    div[data-testid="stSidebar"] .stMarkdown p, 
    div[data-testid="stSidebar"] .stMarkdown span {{
        color: {T['text']};
    }}

    .gpt-sidebar-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 6px 14px 6px;
    }}
    .gpt-brand-title {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.18rem;
        font-weight: 800;
        letter-spacing: -0.3px;
        color: {BRAND};
    }}

    /* Chat History Sidebar Section */
    .gpt-history-section {{
        margin-top: 14px;
        margin-bottom: 8px;
    }}
    .gpt-history-label {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: {T['text2']};
        padding: 4px 8px;
    }}
    .gpt-history-item {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 10px;
        border-radius: 8px;
        font-size: 0.84rem;
        color: {T['text']};
        cursor: pointer;
        text-decoration: none;
        transition: all 0.15s ease;
        margin-bottom: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .gpt-history-item:hover {{
        background: {T['hover']};
    }}
    .gpt-history-item.active {{
        background: {T['hover']};
        color: {BRAND};
        font-weight: 600;
        border-left: 3px solid {ACTIVE};
    }}

    .gpt-user-profile {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 6px;
        border-top: 1px solid {T['border']};
        margin-top: 16px;
    }}
    .gpt-avatar {{
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: {BRAND};
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
    }}
    .gpt-user-info {{
        display: flex;
        flex-direction: column;
    }}
    .gpt-user-name {{
        font-size: 0.88rem;
        font-weight: 600;
        color: {T['text']};
    }}
    .gpt-user-role {{
        font-size: 0.72rem;
        color: {ACTIVE};
        font-weight: 600;
    }}

    /* Registration Portal */
    .fx-reg-card {{
        background: {T['panel']};
        border: 1px solid {T['border']};
        border-radius: 14px;
        padding: 40px 36px;
        width: 100%;
        max-width: 480px;
        margin: 40px auto;
        text-align: center;
        box-shadow: 0 8px 30px rgba(11, 73, 107, 0.08);
    }}
    .fx-reg-title {{
        font-size: 1.5rem;
        font-weight: 800;
        color: {T['text']};
        margin-top: 16px;
        letter-spacing: -0.4px;
    }}
    .fx-reg-subtitle {{
        font-size: 0.92rem;
        color: {T['text2']};
        margin-top: 6px;
        line-height: 1.5;
    }}
    .fx-reg-footer {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        margin-top: 20px;
        font-size: 0.78rem;
        color: {ACTIVE};
        font-weight: 500;
    }}

    /* Hero Empty State */
    .gpt-hero {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding-top: 12vh;
        text-align: center;
        margin-bottom: 24px;
    }}
    .gpt-hero-title {{
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.6px;
        color: {T['text']};
        margin-top: 14px;
        margin-bottom: 6px;
    }}
    .gpt-hero-sub {{
        font-size: 0.95rem;
        color: {T['text2']};
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .gpt-device-pill {{
        background: {T['user_bubble']};
        border: 1px solid {BORDER_ICE};
        color: {BRAND};
        padding: 2px 10px;
        border-radius: 12px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
    }}

    /* Suggestion Cards */
    .gpt-prompt-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        width: 100%;
        max-width: 680px;
        margin: 0 auto 24px auto;
    }}
    .gpt-prompt-card {{
        background: {T['pill_bg']};
        border: 1px solid {T['pill_border']};
        border-radius: 12px;
        padding: 12px 16px;
        cursor: pointer;
        text-align: left;
        transition: all 0.15s ease;
    }}
    .gpt-prompt-card:hover {{
        background: {T['hover']};
        border-color: {ACTIVE};
        transform: translateY(-1px);
    }}
    .gpt-prompt-title {{
        font-size: 0.85rem;
        font-weight: 600;
        color: {T['text']};
    }}
    .gpt-prompt-desc {{
        font-size: 0.76rem;
        color: {T['text2']};
        margin-top: 2px;
    }}

    /* Messages Thread */
    .gpt-msg-user {{
        display: flex;
        justify-content: flex-end;
        margin: 18px 0;
    }}
    .gpt-msg-user-bubble {{
        background: {T['user_bubble']};
        color: {T['user_text']};
        padding: 12px 20px;
        border-radius: 20px 20px 4px 20px;
        max-width: 78%;
        font-size: 0.94rem;
        line-height: 1.6;
        border: 1px solid {BORDER_ICE}60;
    }}

    .gpt-msg-ai {{
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        margin: 20px 0;
    }}
    .gpt-msg-ai-header {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.9rem;
        font-weight: 700;
        color: {BRAND};
        margin-bottom: 8px;
    }}
    .gpt-msg-ai-body {{
        color: {T['text']};
        font-size: 0.94rem;
        line-height: 1.7;
        width: 100%;
    }}
    .gpt-msg-ai-body p {{
        margin-bottom: 10px;
    }}

    /* Clean Procedure Card */
    .gpt-proc-card {{
        background: {T['panel']};
        border: 1px solid {T['border']};
        border-radius: 10px;
        margin: 14px 0;
        overflow: hidden;
    }}
    .gpt-proc-head {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 16px;
        background: {T['hover']};
        border-bottom: 1px solid {T['border']};
        font-size: 0.8rem;
        font-weight: 700;
        color: {BRAND};
        letter-spacing: 0.4px;
    }}
    .gpt-proc-step {{
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 9px 16px;
        border-bottom: 1px solid {T['border']}30;
    }}
    .gpt-proc-step:last-child {{ border-bottom: none; }}
    .gpt-step-badge {{
        min-width: 24px;
        height: 24px;
        border-radius: 6px;
        background: {BRAND}15;
        color: {BRAND};
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.76rem;
        font-weight: 700;
    }}
    .gpt-step-text {{
        font-size: 0.88rem;
        color: {T['text']};
        line-height: 1.5;
        padding-top: 1px;
    }}
    .gpt-proc-foot {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 16px;
        background: {T['hover']};
        border-top: 1px solid {T['border']};
        font-size: 0.75rem;
        color: {T['text2']};
        font-family: 'IBM Plex Mono', monospace;
    }}

    /* Safety Banner */
    .gpt-safety-critical {{
        display: flex;
        align-items: flex-start;
        gap: 10px;
        background: rgba(217, 45, 32, 0.08);
        border: 1px solid rgba(217, 45, 32, 0.3);
        border-left: 4px solid {CRITICAL};
        padding: 12px 16px;
        border-radius: 8px;
        color: {CRITICAL};
        font-size: 0.88rem;
        font-weight: 600;
        margin: 12px 0;
        line-height: 1.5;
    }}

    /* Source Citation Chip */
    .gpt-source-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: {T['hover']};
        border: 1px solid {T['border']};
        color: {T['text2']};
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.76rem;
        font-family: 'IBM Plex Mono', monospace;
        margin-top: 8px;
        margin-right: 6px;
    }}

    /* Streamlit Button Overrides (Dark & Light Modes) */
    .stButton > button {{
        border-radius: 20px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        padding: 7px 18px !important;
        font-size: 0.88rem !important;
        transition: all 0.15s ease !important;
    }}
    
    /* Main / Primary Buttons: White in Dark Mode with Dark Navy Text */
    div[data-testid="stAppViewContainer"] .stButton > button {{
        background-color: {"#ffffff" if is_dark else BRAND} !important;
        color: {"#071923" if is_dark else "#ffffff"} !important;
        border: 1px solid {"#ffffff" if is_dark else BRAND} !important;
    }}
    div[data-testid="stAppViewContainer"] .stButton > button:hover {{
        background-color: {"#e2edf1" if is_dark else ACTIVE} !important;
        color: {"#071923" if is_dark else "#ffffff"} !important;
        border-color: {"#e2edf1" if is_dark else ACTIVE} !important;
        transform: translateY(-1px) !important;
    }}
    div[data-testid="stAppViewContainer"] .stButton > button p,
    div[data-testid="stAppViewContainer"] .stButton > button span,
    div[data-testid="stAppViewContainer"] .stButton > button div {{
        color: {"#071923" if is_dark else "#ffffff"} !important;
        font-weight: 600 !important;
    }}

    /* Secondary Buttons in Sidebar */
    div[data-testid="stSidebar"] .stButton > button[kind="secondary"] {{
        background-color: {"#102936" if is_dark else "#ffffff"} !important;
        color: {"#edf5f7" if is_dark else "#0b496b"} !important;
        border: 1px solid {"#1b3a47" if is_dark else BORDER_ICE} !important;
    }}
    div[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {{
        background-color: {"#183e52" if is_dark else "#edf6f8"} !important;
        color: {"#ffffff" if is_dark else BRAND} !important;
        border-color: {ACTIVE} !important;
    }}
    div[data-testid="stSidebar"] .stButton > button[kind="secondary"] p,
    div[data-testid="stSidebar"] .stButton > button[kind="secondary"] span {{
        color: {"#edf5f7" if is_dark else "#0b496b"} !important;
    }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 6. Backend RAG Query Function
# =============================================================================
def query_rag_engine(user_query, device_name=None):
    try:
        req_data = {
            "query": user_query,
            "device_name": device_name if device_name and device_name != "All Devices" else None,
            "top_k": 5,
        }
        req = urllib.request.Request(
            "http://127.0.0.1:8000/v1/query",
            data=json.dumps(req_data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {
            "answer": f"Searching manuals for {user_query}. Manual excerpt not directly available via offline fallback.",
            "status": "NOT_FOUND_IN_MANUAL",
            "sources": []
        }

# =============================================================================
# 7. Clean Rendering Helpers (No Code Box Duplication)
# =============================================================================
def render_clean_ai_response(msg):
    raw_content = msg.get("content", "")
    checklist = msg.get("checklist", [])
    source_citation = msg.get("source_citation")
    sources = msg.get("sources", [])
    has_safety = msg.get("has_high_priority_safety", False)
    safety_body = msg.get("safety_body", "")

    # Clean raw content to avoid repeating checklist or sources already shown in cards
    cleaned_overview = raw_content
    cleaned_overview = re.sub(r'###\s*🔧?\s*Step-by-Step Checklist[\s\S]*?(?=\n\[Source|\Z)', '', cleaned_overview, flags=re.IGNORECASE).strip()
    cleaned_overview = re.sub(r'\[Source\s*\d+:?[^\]]*\]', '', cleaned_overview, flags=re.IGNORECASE).strip()
    if has_safety:
        cleaned_overview = re.sub(r'###\s*⚠️?\s*HIGH PRIORITY SAFETY[\s\S]*?\*\*', '', cleaned_overview, flags=re.IGNORECASE).strip()

    # Convert markdown bolds to HTML
    overview_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', cleaned_overview)
    overview_html = overview_html.replace('\n', '<br>')

    # 1. Safety Banner
    safety_html = ""
    if has_safety and safety_body:
        safety_html = f'<div class="gpt-safety-critical">{svg("warn", 18)}<div><strong>CRITICAL SAFETY INSTRUCTION</strong><br>{safety_body}</div></div>'

    # 2. Procedure Card
    proc_html = ""
    if checklist and len(checklist) > 0:
        steps = ""
        for i, step in enumerate(checklist, 1):
            clean = re.sub(r'^Step\s*\d+\s*[:\.]?\s*', '', step, flags=re.IGNORECASE).strip()
            steps += f'<div class="gpt-proc-step"><div class="gpt-step-badge">{i:02d}</div><div class="gpt-step-text">{clean}</div></div>'
        
        src_label = ""
        cit = source_citation or (sources[0] if sources else None)
        if cit and isinstance(cit, dict):
            m = cit.get("manual", cit.get("device", "Manual"))
            p = cit.get("page", cit.get("page_number", ""))
            src_label = f'{svg("doc", 13)} {m}{f" · p.{p}" if p else ""}'
        
        proc_html = f'<div class="gpt-proc-card"><div class="gpt-proc-head">{svg("gear", 15, BRAND)} RECOMMENDED MAINTENANCE PROCEDURE</div>{steps}<div class="gpt-proc-foot"><span>{src_label}</span><span>{len(checklist)} STEPS</span></div></div>'

    # 3. Source Chip (only if procedure card didn't already display it)
    sources_html = ""
    if not proc_html and (source_citation or sources):
        cit = source_citation or sources[0]
        if isinstance(cit, dict):
            m = cit.get("manual", cit.get("device", "Manual"))
            p = cit.get("page", cit.get("page_number", ""))
            sources_html = f'<span class="gpt-source-chip">{svg("doc", 12)} {m}{f" · p.{p}" if p else ""}</span>'

    # Unindented compact HTML to prevent Markdown parser from treating it as <pre><code>
    return f'<div class="gpt-msg-ai"><div class="gpt-msg-ai-header">{svg("logo", 20)} <span>Fixora</span></div><div class="gpt-msg-ai-body">{safety_html}{overview_html}{proc_html}{sources_html}</div></div>'

def render_user_response(content):
    clean = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
    return f'<div class="gpt-msg-user"><div class="gpt-msg-user-bubble">{clean}</div></div>'


# =============================================================================
# 8. SCREEN 1: Registration Screen (Technician Workspace Entry)
# =============================================================================
if not st.session_state.registered:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.8, 1])
    with c2:
        st.markdown(f'''
        <div class="fx-reg-card">
            <div style="margin-bottom:12px;">{svg("logo", 54)}</div>
            <div class="fx-reg-title">Technician Workspace</div>
            <div class="fx-reg-sub">Connect to your equipment-specific AI support environment.</div>
        </div>
        ''', unsafe_allow_html=True)

        with st.form("reg_form"):
            name_input = st.text_input("Technician Name", placeholder="e.g. Menna Ashraf", value=st.session_state.username)
            device_input = st.selectbox("Assigned Machine / Model", [
                "Siemens Servo 900 Ventilator",
                "Siemens Mobilett Plus HP",
                "Siemens Powermobil Power Module",
                "CT Scanner Hardware Subsystems",
                "Philips Agilent Patient Monitor",
                "Compressor X200",
                "All Devices",
            ], index=0)
            
            st.caption("AI responses will be grounded strictly in the selected technical manual.")
            submit = st.form_submit_button("Launch Workspace →", use_container_width=True)

            if submit:
                if name_input.strip():
                    st.session_state.username = name_input.strip()
                    st.session_state.device_selected = device_input
                    st.session_state.registered = True
                    st.rerun()
                else:
                    st.error("Please enter your name to proceed.")

        st.markdown(f'<div class="fx-reg-footer">{svg("shield", 14, ACTIVE)} Secure Technical Field Environment</div>', unsafe_allow_html=True)


# =============================================================================
# 9. SCREEN 2: Main Workspace with ChatGPT Sidebar & Chat History
# =============================================================================
else:
    # -------------------------------------------------------------------------
    # Left Sidebar (ChatGPT Multi-Session History + Navigation)
    # -------------------------------------------------------------------------
    with st.sidebar:
        # Header: Brand & Search
        st.markdown(f'''
        <div class="gpt-sidebar-header">
            <div class="gpt-brand-title">
                {svg("logo", 26)} <span>Fixora</span>
            </div>
            <div>{svg("search", 18)}</div>
        </div>
        ''', unsafe_allow_html=True)

        # "+ New chat" Button
        if st.button("✚  New chat", use_container_width=True, type="secondary"):
            new_id = f"chat_{uuid.uuid4().hex[:6]}"
            st.session_state.chat_history[new_id] = {
                "title": "New Diagnosis",
                "device": st.session_state.device_selected,
                "messages": []
            }
            st.session_state.current_chat_id = new_id
            st.session_state.view_mode = "chat"
            st.rerun()

        # Chat History List
        st.markdown('<div class="gpt-history-section"><div class="gpt-history-label">Recent Sessions</div></div>', unsafe_allow_html=True)

        for chat_id, chat_data in list(st.session_state.chat_history.items()):
            title = chat_data.get("title", "Session")
            is_active = (chat_id == st.session_state.current_chat_id)
            
            col_sess, col_del = st.columns([5, 1])
            with col_sess:
                btn_type = "primary" if is_active else "secondary"
                if st.button(f"💬 {title[:20]}", key=f"sess_{chat_id}", use_container_width=True, type=btn_type):
                    st.session_state.current_chat_id = chat_id
                    st.session_state.device_selected = chat_data.get("device", st.session_state.device_selected)
                    st.session_state.view_mode = "chat"
                    st.rerun()
            with col_del:
                if len(st.session_state.chat_history) > 1:
                    if st.button("×", key=f"del_{chat_id}", help="Delete chat"):
                        del st.session_state.chat_history[chat_id]
                        st.session_state.current_chat_id = list(st.session_state.chat_history.keys())[0]
                        st.rerun()

        st.markdown("<hr style='border:none; border-top:1px solid rgba(128,128,128,0.15); margin: 16px 0;'>", unsafe_allow_html=True)

        # Target Equipment Selector
        st.caption("TARGET EQUIPMENT MANUAL")
        selected_device = st.selectbox(
            "Equipment Manual",
            [
                "Siemens Servo 900 Ventilator",
                "Siemens Mobilett Plus HP",
                "Siemens Powermobil Power Module",
                "CT Scanner Hardware Subsystems",
                "Philips Agilent Patient Monitor",
                "Compressor X200",
                "All Devices",
            ],
            index=0,
            label_visibility="collapsed"
        )
        if selected_device != st.session_state.device_selected:
            st.session_state.device_selected = selected_device
            if st.session_state.current_chat_id in st.session_state.chat_history:
                st.session_state.chat_history[st.session_state.current_chat_id]["device"] = selected_device
            st.rerun()

        # Voice Mode Toggle
        if st.button("🎙  Enter Voice Call Mode", use_container_width=True):
            st.session_state.view_mode = "voice_call" if st.session_state.view_mode == "chat" else "chat"
            st.rerun()

        # Theme Switcher
        theme_icon = "☀️ Light" if is_dark else "🌙 Dark"
        if st.button(f"Switch Theme ({theme_icon})", use_container_width=True, type="secondary"):
            st.session_state.theme = "light" if is_dark else "dark"
            st.rerun()

        # Logout / Return to Registration
        if st.button("👤 Switch Technician", use_container_width=True, type="secondary"):
            st.session_state.registered = False
            st.rerun()

        # Download Architecture & Pipeline PDF
        pdf_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fixora_Architecture_and_Pipeline.pdf")
        if os.path.exists(pdf_file_path):
            with open(pdf_file_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="📥 Download Architecture PDF",
                data=pdf_bytes,
                file_name="Fixora_Architecture_and_Pipeline.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="secondary",
            )

        # User Profile Footer
        initials = "".join([p[0] for p in st.session_state.username.split()[:2]]).upper() if st.session_state.username else "MA"
        st.markdown(f'''
        <div class="gpt-user-profile">
            <div class="gpt-avatar">{initials}</div>
            <div class="gpt-user-info">
                <span class="gpt-user-name">{st.session_state.username}</span>
                <span class="gpt-user-role">Field Technician · Fixora AI</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Main View (Voice Mode OR Chat Thread)
    # -------------------------------------------------------------------------
    if st.session_state.view_mode == "voice_call":
        col_back, _ = st.columns([1.5, 4])
        with col_back:
            if st.button("← Back to Chat", use_container_width=True):
                st.session_state.view_mode = "chat"
                st.rerun()

        device_escaped = st.session_state.device_selected.replace('"', '\\"')
        voice_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_component.html")
        if os.path.exists(voice_file):
            with open(voice_file, "r", encoding="utf-8") as f:
                voice_html = f.read()
            voice_html = (
                voice_html
                .replace("__THEME_BG__", T["bg"])
                .replace("__THEME_PANEL__", T["panel"])
                .replace("__THEME_TEXT__", T["text"])
                .replace("__THEME_BORDER__", T["border"])
                .replace("__DEVICE_NAME__", device_escaped)
            )
        else:
            voice_html = "<div style='text-align:center; padding:40px;'>Voice component loading...</div>"

        components.html(voice_html, height=560, scrolling=False)

    else:
        current_messages = get_current_messages()

        # 1. EMPTY HERO STATE (When current session has no messages)
        if not current_messages:
            st.markdown(f'''
            <div class="gpt-hero">
                {svg("logo", 48)}
                <div class="gpt-hero-title">What's on your mind today?</div>
                <div class="gpt-hero-sub">
                    Grounded technical assistant for <span class="gpt-device-pill">{st.session_state.device_selected}</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)

            # Interactive Suggestion Prompt Buttons (2x2 Grid)
            col_p1, col_p2 = st.columns(2)
            prompt_selected = None
            with col_p1:
                if st.button("⚡ Error Code 37 Troubleshooting\n\nInspect flow meter range error & connectors", key="p_err37", use_container_width=True, type="secondary"):
                    prompt_selected = "Error Code 37 troubleshooting"
                if st.button("⚠️ High Voltage Power Isolation\n\nLockout/tagout (LOTO) safety protocols", key="p_hv", use_container_width=True, type="secondary"):
                    prompt_selected = "High voltage power isolation and LOTO precautions"
            with col_p2:
                if st.button("🔋 Alarm 29 Battery Replacement\n\nLithium battery service on PC1772 board", key="p_bat29", use_container_width=True, type="secondary"):
                    prompt_selected = "Alarm 29 Lithium Battery Replacement"
                if st.button("❄️ Cooling System Specifications\n\nWater flow rates & chiller options", key="p_cool", use_container_width=True, type="secondary"):
                    prompt_selected = "Cooling system specifications and chiller options"

            if prompt_selected:
                st.session_state.chat_history[st.session_state.current_chat_id]["title"] = prompt_selected[:24]
                current_messages.append({"role": "user", "content": prompt_selected})
                with st.spinner("Fixora is searching technical manuals..."):
                    rag_res = query_rag_engine(prompt_selected, st.session_state.device_selected)
                    current_messages.append({
                        "role": "assistant",
                        "content": rag_res.get("answer", "No relevant manual procedure found."),
                        "sources": rag_res.get("sources", []),
                        "checklist": rag_res.get("checklist", []),
                        "source_citation": rag_res.get("source_citation"),
                        "has_high_priority_safety": rag_res.get("has_high_priority_safety", False),
                        "safety_body": rag_res.get("safety_body", ""),
                    })
                st.rerun()

        # 2. ACTIVE CHAT THREAD
        else:
            for msg in current_messages:
                if msg["role"] == "user":
                    st.markdown(render_user_response(msg["content"]), unsafe_allow_html=True)
                else:
                    st.markdown(render_clean_ai_response(msg), unsafe_allow_html=True)

        # 3. FLOATING CHAT INPUT
        user_text = st.chat_input("Ask anything about faults, codes, procedures...")

        if user_text:
            # Auto-title chat session on first message
            if not current_messages:
                st.session_state.chat_history[st.session_state.current_chat_id]["title"] = user_text[:24] + ("..." if len(user_text) > 24 else "")

            current_messages.append({"role": "user", "content": user_text})
            with st.spinner("Fixora is searching technical manuals..."):
                rag_res = query_rag_engine(user_text, st.session_state.device_selected)
                current_messages.append({
                    "role": "assistant",
                    "content": rag_res.get("answer", "No relevant manual procedure found."),
                    "sources": rag_res.get("sources", []),
                    "checklist": rag_res.get("checklist", []),
                    "source_citation": rag_res.get("source_citation"),
                    "has_high_priority_safety": rag_res.get("has_high_priority_safety", False),
                    "safety_body": rag_res.get("safety_body", ""),
                })
            st.rerun()
