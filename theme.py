"""
theme.py - CampusGPT look and feel: minimalist glass widgets, animated gradient background,
live colour themes, font pairs and an optional background photo.

HOW TO CUSTOMISE (all in this file):
  PRESETS  -> add or edit colour themes (background colours, accents, text colour)
  FONTS    -> add or edit font pairs (any Google Font name works)
  DEFAULT_PRESET / DEFAULT_FONT / DEFAULT_GLASS -> what a new visitor sees first

The user can also change theme, font, glass strength and background photo live from the
"Appearance" panel in the sidebar (appearance_picker()).
"""

import base64
import re

import streamlit as st

# ----------------------------------------------------------------------------------------------
# Colour themes. bg = 4 background gradient colours, a1/a2/a3 = accents, txt/sub = text colours.
# ----------------------------------------------------------------------------------------------
PRESETS = {
    "Aurora":   {"bg": ["#EEF2FF", "#FCE7F3", "#E0F2FE", "#ECFDF5"], "a1": "#6366F1", "a2": "#EC4899", "a3": "#06B6D4",
                 "txt": "#1F2937", "sub": "#4B5563", "dark": False},
    "Forest":   {"bg": ["#ECFDF5", "#D1FAE5", "#CFFAFE", "#F0FDF4"], "a1": "#059669", "a2": "#10B981", "a3": "#84CC16",
                 "txt": "#12372A", "sub": "#3F6B5A", "dark": False},
    "Sunset":   {"bg": ["#FFF7ED", "#FFE4E6", "#FEF3C7", "#FCE7F3"], "a1": "#F97316", "a2": "#E11D48", "a3": "#F59E0B",
                 "txt": "#3B1D14", "sub": "#7A4B3A", "dark": False},
    "Ocean":    {"bg": ["#ECFEFF", "#E0F2FE", "#EDE9FE", "#F0F9FF"], "a1": "#0284C7", "a2": "#6366F1", "a3": "#14B8A6",
                 "txt": "#0C2A3F", "sub": "#3C5F78", "dark": False},
    "Midnight": {"bg": ["#0F172A", "#1E1B4B", "#172554", "#1F1235"], "a1": "#818CF8", "a2": "#F472B6", "a3": "#22D3EE",
                 "txt": "#E5E7EB", "sub": "#9CA3AF", "dark": True},
}

# Font pairs: heading font, body font, and the Google Fonts family string.
FONTS = {
    "Poppins":        {"head": "Poppins",          "body": "Poppins",   "gf": "Poppins:wght@400;500;600;700;800"},
    "Inter":          {"head": "Inter",            "body": "Inter",     "gf": "Inter:wght@400;500;600;700;800"},
    "Space Grotesk":  {"head": "Space Grotesk",    "body": "Inter",     "gf": "Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600"},
    "Playfair + Inter": {"head": "Playfair Display", "body": "Inter",   "gf": "Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600"},
    "Nunito (rounded)": {"head": "Nunito",         "body": "Nunito",    "gf": "Nunito:wght@400;600;700;800"},
}

DEFAULT_PRESET = "Aurora"
DEFAULT_FONT = "Poppins"
DEFAULT_GLASS = 62  # card opacity in percent: lower = more see-through

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=__GF__&display=swap');

:root {
  --a1: __A1__; --a2: __A2__; --a3: __A3__;
  --txt: __TXT__; --sub: __SUB__;
  --glass: __GLASS__; --glass-line: __LINE__; --glass-shadow: __SHADOW__;
  --head: '__HEAD__', sans-serif; --body: '__BODY__', sans-serif;
}

/* ---------- background ---------- */
.stApp {
  background: __BG__;
  background-size: __BGSIZE__;
  animation: __ANIM__;
  font-family: var(--body);
}
@keyframes cg-flow { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
.stApp::before, .stApp::after { content:""; position:fixed; z-index:0; pointer-events:none; border-radius:50%;
  filter: blur(80px); opacity:.5; }
.stApp::before { width:420px; height:420px; top:-90px; right:-70px;
  background: radial-gradient(circle, var(--a2), transparent 70%); animation: cg-f1 17s ease-in-out infinite; }
.stApp::after { width:480px; height:480px; bottom:-130px; left:18%;
  background: radial-gradient(circle, var(--a3), transparent 70%); animation: cg-f2 21s ease-in-out infinite; }
@keyframes cg-f1 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(-120px,90px)} }
@keyframes cg-f2 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(140px,-80px)} }

[data-testid="stHeader"] { background: transparent; }
.block-container { position:relative; z-index:1; padding-top:1.6rem; max-width: 1180px; }

/* ---------- text ---------- */
html, body, .stApp, p, li, label, span, div, input, textarea, button { font-family: var(--body); }
h1, h2, h3, h4, .cg-hero h1 { font-family: var(--head) !important; letter-spacing:-.01em; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li, .stApp label,
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] { color: var(--txt); }
[data-testid="stCaptionContainer"], .stApp small { color: var(--sub) !important; }

/* ---------- sidebar (glass) ---------- */
[data-testid="stSidebar"] {
  background: var(--glass); backdrop-filter: blur(18px) saturate(140%); -webkit-backdrop-filter: blur(18px) saturate(140%);
  border-right: 1px solid var(--glass-line);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  background: linear-gradient(90deg, var(--a1), var(--a2)); -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color: transparent; font-weight: 800;
}

/* ---------- tabs (pill style) ---------- */
[role="tablist"] { gap:6px; background: var(--glass); padding:6px; border-radius:16px; border:1px solid var(--glass-line);
  backdrop-filter: blur(14px); box-shadow: 0 6px 22px var(--glass-shadow); }
[role="tab"] { border-radius:12px; padding:8px 16px; transition: all .25s ease; color: var(--txt); }
[role="tab"]:hover { background: rgba(127,127,255,.16); transform: translateY(-1px); }
[role="tab"][aria-selected="true"] { background: linear-gradient(90deg, var(--a1), var(--a2)) !important;
  border-bottom:0 !important; box-shadow: 0 5px 14px var(--glass-shadow); }
[role="tab"][aria-selected="true"] * { color:#fff !important; font-weight:700; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display:none !important; }

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border:0; border-radius:12px; color:#fff; font-weight:600; padding:.5rem 1.1rem;
  background: linear-gradient(90deg, var(--a1), var(--a2)); box-shadow: 0 5px 16px var(--glass-shadow);
  transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px) scale(1.02); filter: brightness(1.07); color:#fff; }
.stButton > button:active { transform: scale(.98); }

/* ---------- glass widgets ---------- */
[data-testid="stMetric"], [data-testid="stExpander"], [data-testid="stChatMessage"],
[data-testid="stFileUploader"] section, [data-testid="stAlert"], [data-testid="stForm"] {
  background: var(--glass); backdrop-filter: blur(14px) saturate(140%); -webkit-backdrop-filter: blur(14px) saturate(140%);
  border-radius:18px; border:1px solid var(--glass-line); box-shadow: 0 8px 26px var(--glass-shadow);
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] details, [data-testid="stExpander"] summary * {
  background: transparent !important; color: var(--txt) !important; }
[data-testid="stFileUploader"] small, [data-testid="stFileUploaderDropzoneInstructions"] * { color: var(--sub) !important; }
[data-testid="stFileUploader"] button, [data-testid="stFileUploader"] button * { color: #1F2937 !important; }
[data-testid="stMetric"] { padding:14px 18px; transition: transform .2s ease; }
[data-testid="stMetric"]:hover, [data-testid="stExpander"]:hover { transform: translateY(-3px); }
[data-testid="stFileUploader"] section { border:2px dashed var(--a1); }
[data-testid="stChatMessage"] { padding:12px 16px; }

/* ---------- uploader file list + slider values ---------- */
[data-testid="stFileUploaderFile"], [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
  background: rgba(10,15,30,.68) !important; border:1px solid rgba(255,255,255,.18); border-radius:12px;
  padding:6px 10px; margin-top:8px; backdrop-filter: blur(8px); }
[data-testid="stFileUploaderFile"] *, [data-testid="stFileUploaderFileName"] { color:#F1F5F9 !important; opacity:1 !important; }
[data-testid="stFileUploaderDeleteBtn"] button, [data-testid="stFileUploaderDeleteBtn"] svg { color:#F1F5F9 !important; opacity:1 !important; }
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"],
[data-testid="stSlider"] [data-testid="stThumbValue"] { color:#FFFFFF !important; font-size:.85rem !important; font-weight:600; }

/* ---------- file pills, help icons ---------- */
.cg-file { background: rgba(10,15,30,.68); border:1px solid rgba(255,255,255,.18); color:#F1F5F9;
  padding:10px 14px; border-radius:12px; backdrop-filter: blur(8px); font-size:.92rem; word-break:break-all; }
[data-testid="stTooltipIcon"] svg, [data-testid="stTooltipHoverTarget"] svg { color:#E2E8F0 !important; fill:#E2E8F0 !important; opacity:1 !important; }
[data-testid="stTooltipIcon"] { opacity:1 !important; }

/* ---------- hero + chips + stat cards ---------- */
.cg-hero { position:relative; overflow:hidden; border-radius:24px; padding:34px 36px; margin-bottom:18px; color:#fff;
  background: linear-gradient(120deg, var(--a1) 0%, var(--a2) 60%, var(--a3) 100%); background-size:200% 200%;
  filter: saturate(.85) brightness(.9);
  animation: cg-flow 12s ease infinite; box-shadow: 0 16px 40px var(--glass-shadow); }
.cg-hero::after { content:""; position:absolute; inset:0; pointer-events:none;
  background: radial-gradient(circle at 85% 10%, rgba(255,255,255,.35), transparent 45%); }
.cg-hero h1 { margin:0; font-size:2.3rem; color:#fff !important; padding:0; }
.cg-hero p { margin:8px 0 0; opacity:1; text-shadow:0 1px 6px rgba(0,0,0,.45); font-weight:500; font-size:1.02rem; color:#fff !important; }
.cg-chips { margin-top:16px; display:flex; flex-wrap:wrap; gap:8px; position:relative; z-index:1; }
.cg-chip { background: rgba(10,15,30,.38); border:1px solid rgba(255,255,255,.6); text-shadow:0 1px 3px rgba(0,0,0,.4); backdrop-filter: blur(8px);
  padding:6px 14px; border-radius:999px; font-size:.86rem; color:#fff; font-weight:500; transition: all .2s ease; }
.cg-chip:hover { background: rgba(10,15,30,.55); transform: translateY(-2px); }

.cg-card { border-radius:20px; padding:18px 20px; color:#fff; box-shadow: 0 12px 28px var(--glass-shadow);
  transition: transform .25s ease; position:relative; overflow:hidden; }
.cg-card::after { content:""; position:absolute; right:-30px; top:-30px; width:110px; height:110px; border-radius:50%;
  background: rgba(255,255,255,.18); }
.cg-card:hover { transform: translateY(-5px) scale(1.02); }
.cg-card .v { font-size:2.3rem; font-weight:800; line-height:1.05; font-family: var(--head); color:#fff; }
.cg-card .l { font-size:.9rem; opacity:.95; margin-top:4px; color:#fff; }
.cg-g1 { background: linear-gradient(135deg, var(--a1), var(--a2)); }
.cg-g2 { background: linear-gradient(135deg, var(--a3), var(--a1)); }
.cg-g3 { background: linear-gradient(135deg, var(--a2), #F59E0B); }
.cg-g4 { background: linear-gradient(135deg, var(--a1), var(--a3)); }
</style>
"""


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _safe_url(url):
    url = (url or "").strip()
    if url.startswith(("http://", "https://")) and not re.search(r"[\s\"'()<>\\]", url):
        return url
    return ""


def settings():
    """Current appearance settings (from the sidebar panel, with defaults)."""
    ss = st.session_state
    return {
        "preset": ss.get("ui_preset", DEFAULT_PRESET),
        "font": ss.get("ui_font", DEFAULT_FONT),
        "glass": ss.get("ui_glass", DEFAULT_GLASS),
        "dim": ss.get("ui_dim", 55),
        "bg_url": ss.get("ui_bg", ""),
        "bg_file": ss.get("ui_bg_file"),
    }


def apply_theme():
    """Call once, right after st.set_page_config(). Reads the live Appearance settings."""
    s = settings()
    p = PRESETS.get(s["preset"], PRESETS[DEFAULT_PRESET])
    f = FONTS.get(s["font"], FONTS[DEFAULT_FONT])
    alpha = max(0.15, min(0.95, s["glass"] / 100))

    if p["dark"]:
        glass = f"rgba(30,41,59,{alpha:.2f})"
        line, shadow = "rgba(255,255,255,.14)", "rgba(0,0,0,.35)"
    else:
        glass = f"rgba(255,255,255,{alpha:.2f})"
        line = "rgba(255,255,255,.85)"
        r, g, b = _hex_to_rgb(p["a1"])
        shadow = f"rgba({r},{g},{b},.20)"

    bg_colors = ", ".join(p["bg"] + [p["bg"][0]])
    url = _safe_url(s["bg_url"])
    up = s.get("bg_file")
    if up is not None:  # an uploaded image wins over a link, and always works (no hot-link blocking)
        try:
            data = base64.b64encode(up.getvalue()).decode()
            mime = up.type or "image/jpeg"
            url = f"data:{mime};base64,{data}"
        except Exception:  # noqa: BLE001
            url = ""
    if url:
        dim = max(0, min(90, s["dim"])) / 100
        overlay = f"rgba(15,23,42,{dim:.2f})" if p["dark"] else f"rgba(255,255,255,{dim:.2f})"
        bg = f"linear-gradient({overlay}, {overlay}), url({url}) center / cover fixed"
        bgsize, anim = "cover", "none"
    else:
        bg = f"linear-gradient(125deg, {bg_colors})"
        bgsize, anim = "400% 400%", "cg-flow 24s ease infinite"

    css = _CSS
    for key, val in {
        "__GF__": f["gf"], "__A1__": p["a1"], "__A2__": p["a2"], "__A3__": p["a3"],
        "__TXT__": p["txt"], "__SUB__": p["sub"], "__GLASS__": glass, "__LINE__": line, "__SHADOW__": shadow,
        "__HEAD__": f["head"], "__BODY__": f["body"], "__BG__": bg, "__BGSIZE__": bgsize, "__ANIM__": anim,
    }.items():
        css = css.replace(key, val)
    st.markdown(css, unsafe_allow_html=True)


def appearance_picker():
    """Sidebar panel: live theme, font, glass strength and optional background photo."""
    with st.expander("🎨 Appearance", expanded=False):
        st.selectbox("Colour theme", list(PRESETS), key="ui_preset",
                     index=list(PRESETS).index(DEFAULT_PRESET))
        st.selectbox("Font", list(FONTS), key="ui_font", index=list(FONTS).index(DEFAULT_FONT))
        st.slider("Glass strength", 15, 95, DEFAULT_GLASS, key="ui_glass",
                  help="Lower = more see-through widgets")
        st.slider("Photo dimming", 0, 90, 55, key="ui_dim",
                  help="Higher = darker photo, easier to read text")
        st.file_uploader("Background photo (upload)", type=["jpg", "jpeg", "png", "webp"], key="ui_bg_file",
                         help="Best option: pick an image from your computer. Remove it to go back to the gradient.")
        st.text_input("or background photo link", key="ui_bg",
                      placeholder="https://.../photo.jpg",
                      help="Must be a DIRECT image link (ends in .jpg/.png) and press Enter. Some sites block this; uploading always works.")
        link = st.session_state.get("ui_bg", "").strip()
        if link and not _safe_url(link):
            st.warning("That is not a usable image link. It must start with https:// and have no spaces. Upload the image instead.")


def hero(title, subtitle="", chips=None):
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    chip_html = ""
    if chips:
        chip_html = '<div class="cg-chips">' + "".join(f'<span class="cg-chip">{c}</span>' for c in chips) + "</div>"
    st.markdown(f'<div class="cg-hero"><h1>{title}</h1>{sub}{chip_html}</div>', unsafe_allow_html=True)


def stat_card(value, label, style=1):
    """Coloured number card. style 1 to 4 picks the gradient."""
    st.markdown(
        f'<div class="cg-card cg-g{style}"><div class="v">{value}</div><div class="l">{label}</div></div>',
        unsafe_allow_html=True,
    )