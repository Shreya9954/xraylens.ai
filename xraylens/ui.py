"""Visual system for the Streamlit app: theme CSS and small HTML components.

Theme: a radiology lightbox. Film-black ground, bone-white type, a faint
graticule, amber for attention. Instrument Serif for display, Newsreader for
reading and JetBrains Mono for every number the model produces.
"""
from __future__ import annotations

import html

import streamlit as st

# Chart palette: validated with the dataviz CVD checker on #10161a (dark).
NORMAL, PNEUMONIA, OTHER = "#38a596", "#e05c35", "#8b7ff0"
INK, BONE, MUTED, AMBER, RULE = "#0b0f12", "#ece6d6", "#93a1a6", "#ffb547", "rgba(236,230,214,.16)"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Newsreader:ital,opsz,wght@0,6..72,300..700;1,6..72,300..700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --ink: #0b0f12;
  --film: #10161a;
  --film-2: #141c21;
  --bone: #ece6d6;
  --bone-dim: #c9c2b2;
  --muted: #93a1a6;
  --amber: #ffb547;
  --normal: #38a596;
  --pneumonia: #e05c35;
  --other: #8b7ff0;
  --rule: rgba(236,230,214,.16);
  --rule-strong: rgba(236,230,214,.38);
  --display: 'Instrument Serif', 'Newsreader', Georgia, serif;
  --serif: 'Newsreader', Georgia, serif;
  --mono: 'JetBrains Mono', ui-monospace, monospace;
}

html, body, .stApp { background: var(--ink); color: var(--bone); }

/* Layered ground: lightbox glow, warm under-light, then a 48px graticule. */
.stApp {
  background:
    radial-gradient(ellipse 80% 55% at 78% -8%, rgba(127,209,199,.11), transparent 60%),
    radial-gradient(ellipse 60% 50% at -5% 105%, rgba(255,181,71,.07), transparent 60%),
    repeating-linear-gradient(0deg, rgba(236,230,214,.028) 0 1px, transparent 1px 48px),
    repeating-linear-gradient(90deg, rgba(236,230,214,.028) 0 1px, transparent 1px 48px),
    var(--ink) !important;
  background-attachment: fixed !important;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 1240px; padding-top: 2.2rem; padding-left: clamp(16px, 4vw, 56px); padding-right: clamp(16px, 4vw, 56px);
}

.stApp, .stMarkdown, p, li, label, .stRadio, .stSelectbox, input, textarea, button {
  font-family: var(--serif) !important;
}
.stMarkdown p, .stMarkdown li { font-size: 1.08rem; line-height: 1.6; color: var(--bone-dim); }
h1, h2, h3, h4 { font-family: var(--display) !important; font-weight: 400 !important; color: var(--bone) !important; letter-spacing: -.01em; }
code, .mono, [data-testid="stMetricValue"], [data-testid="stDataFrame"] * { font-family: var(--mono) !important; }
code { color: var(--amber) !important; background: rgba(255,181,71,.08) !important; }
a { color: var(--amber) !important; text-decoration-thickness: 1px; text-underline-offset: 3px; }

/* Sidebar: a dark film strip edged with sprocket holes. */
[data-testid="stSidebar"] {
  background:
    radial-gradient(circle at 10px 14px, var(--ink) 3.5px, transparent 4px) 0 0 / 100% 28px repeat-y,
    linear-gradient(180deg, #0e1417, #0b0f12) !important;
  border-right: 1px solid var(--rule);
}
[data-testid="stSidebarNav"] a span { font-family: var(--mono) !important; font-size: .82rem; letter-spacing: .06em; text-transform: uppercase; }

/* Rotated running head on the right edge. Purely decorative. */
.xl-runhead {
  position: fixed; right: -8px; top: 50%; transform: rotate(90deg) translateX(50%); transform-origin: right top;
  font: 600 10px/1 var(--mono); letter-spacing: .32em; color: rgba(236,230,214,.28); z-index: 0; pointer-events: none; white-space: nowrap;
}
@media (max-width: 900px) { .xl-runhead { display: none; } }

/* Masthead */
.xl-mast { position: relative; margin: 0 0 2.4rem; padding-bottom: 1.4rem; border-bottom: 1px solid var(--rule-strong); }
.xl-kicker { font: 600 .72rem/1 var(--mono); letter-spacing: .26em; text-transform: uppercase; color: var(--amber); }
.xl-title { font-family: var(--display); font-size: clamp(3.4rem, 9vw, 7.6rem); line-height: .86; margin: .5rem 0 0; color: var(--bone); letter-spacing: -.03em; }
.xl-title em { color: var(--amber); font-style: italic; }
.xl-dek { font-family: var(--serif); font-size: clamp(1.1rem, 1.6vw, 1.35rem); line-height: 1.45; color: var(--bone-dim); max-width: 40ch; margin: 1.9rem 0 0 clamp(0px, 14vw, 200px); font-style: italic; }
.xl-stamp {
  position: absolute; right: 0; top: .2rem; transform: rotate(-4deg);
  border: 1.5px solid var(--pneumonia); color: var(--pneumonia); padding: .45rem .7rem;
  font: 600 .7rem/1.25 var(--mono); letter-spacing: .14em; text-transform: uppercase; text-align: center;
}
@media (max-width: 700px) { .xl-stamp { position: static; display: inline-block; margin-top: 1rem; transform: rotate(-2deg); } .xl-dek { margin-left: 0; } }

/* Section heads: an oversized numeral hanging into the left gutter. */
.xl-sec { display: grid; grid-template-columns: minmax(64px, 120px) 1fr; align-items: end; gap: 0 18px; margin: 3.2rem 0 1.1rem; }
.xl-sec .num { font-family: var(--display); font-style: italic; font-size: clamp(3.6rem, 7vw, 6rem); line-height: .78; color: transparent; -webkit-text-stroke: 1px var(--amber); }
.xl-sec h2 { margin: 0 !important; padding: 0 !important; font-size: clamp(1.9rem, 3.2vw, 2.6rem) !important; line-height: 1.05; border-bottom: 1px solid var(--rule); padding-bottom: .5rem !important; }
.xl-sec .sub { grid-column: 2; color: var(--muted); font-size: 1.02rem; margin-top: .5rem; font-family: var(--serif); max-width: 62ch; }

/* Score readout */
.xl-readout { border-top: 3px solid var(--bone); padding-top: .9rem; }
.xl-readout .lab { font: 600 .7rem/1 var(--mono); letter-spacing: .22em; color: var(--muted); text-transform: uppercase; }
.xl-readout .call { font-family: var(--display); font-size: clamp(2.8rem, 6vw, 4.6rem); line-height: .95; margin: .35rem 0 .15rem; }
.xl-readout .call.normal { color: var(--normal); }
.xl-readout .call.pneumonia { color: var(--pneumonia); }
.xl-track { position: relative; height: 46px; margin: 1.3rem 0 .4rem; }
.xl-track .bar { position: absolute; left: 0; right: 0; top: 18px; height: 10px;
  background: linear-gradient(90deg, var(--normal), #3c4a50 50%, var(--pneumonia)); border-radius: 5px; opacity: .85; }
.xl-track .thr { position: absolute; top: 6px; width: 0; height: 34px; border-left: 1.5px dashed var(--bone); }
.xl-track .thr span { position: absolute; top: -14px; left: 4px; font: .64rem var(--mono); color: var(--bone); white-space: nowrap; }
.xl-track .pin { position: absolute; top: 10px; width: 16px; height: 26px; margin-left: -8px; background: var(--bone); border: 2px solid var(--ink); border-radius: 3px; box-shadow: 0 0 0 1px var(--bone); }
.xl-axis { display: flex; justify-content: space-between; font: .68rem var(--mono); color: var(--muted); }
.xl-kv { display: grid; grid-template-columns: auto 1fr; gap: .3rem 1.2rem; margin-top: 1rem; font: .84rem/1.5 var(--mono); }
.xl-kv dt { color: var(--muted); } .xl-kv dd { margin: 0; color: var(--bone); text-align: right; }

/* Notes and warnings */
.xl-note { border-left: 3px solid var(--amber); padding: .7rem 0 .7rem 1rem; margin: 1rem 0; background: linear-gradient(90deg, rgba(255,181,71,.07), transparent 70%); }
.xl-note.warn { border-color: var(--pneumonia); background: linear-gradient(90deg, rgba(224,92,53,.09), transparent 70%); }
.xl-note.cool { border-color: var(--normal); background: linear-gradient(90deg, rgba(56,165,150,.08), transparent 70%); }
.xl-note b { color: var(--bone); font-weight: 600; }
.xl-note p { margin: 0; color: var(--bone-dim); font-size: 1rem; line-height: 1.55; }
.xl-tag { display: inline-block; font: 600 .66rem/1 var(--mono); letter-spacing: .14em; text-transform: uppercase; padding: .32rem .5rem; border: 1px solid currentColor; margin: 0 .35rem .35rem 0; }

/* Big comparison numbers */
.xl-hero { display: grid; grid-template-columns: 1.15fr .85fr; gap: 0; border-top: 1px solid var(--rule-strong); border-bottom: 1px solid var(--rule-strong); }
.xl-hero > div { padding: 1.4rem 1.4rem 1.2rem 0; }
.xl-hero > div + div { padding-left: 1.6rem; border-left: 1px solid var(--rule-strong); }
.xl-hero .n { font-family: var(--display); font-size: clamp(3.4rem, 8vw, 6.4rem); line-height: .9; }
.xl-hero .n small { font-size: .38em; color: var(--muted); }
.xl-hero .cap { font: .74rem/1.45 var(--mono); color: var(--muted); letter-spacing: .04em; margin-top: .5rem; max-width: 42ch; }
.xl-strike { text-decoration: line-through; text-decoration-color: var(--pneumonia); text-decoration-thickness: 3px; color: var(--bone-dim); }
@media (max-width: 700px) { .xl-hero { grid-template-columns: 1fr; } .xl-hero > div + div { border-left: 0; border-top: 1px solid var(--rule-strong); padding-left: 0; } }

/* Stat rows (inline numbers with CIs) */
.xl-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); border-top: 1px solid var(--rule); }
.xl-stats > div { padding: .8rem .9rem .8rem 0; border-bottom: 1px solid var(--rule); }
.xl-stats .k { font: 600 .66rem/1 var(--mono); letter-spacing: .18em; text-transform: uppercase; color: var(--muted); }
.xl-stats .v { font: 600 1.7rem/1.1 var(--mono); color: var(--bone); margin-top: .35rem; }
.xl-stats .ci { font: .7rem var(--mono); color: var(--muted); }

/* Figures */
.xl-fig { font: .72rem/1.4 var(--mono); color: var(--muted); letter-spacing: .03em; margin-top: .35rem; }
.xl-fig b { color: var(--amber); font-weight: 600; }
[data-testid="stImage"] img { border: 1px solid var(--rule); }
[data-testid="stImageCaption"], [data-testid="caption"] { font-family: var(--mono) !important; font-size: .72rem !important; color: var(--muted) !important; }

/* Widgets */
.stButton > button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
  border-radius: 0 !important; font-family: var(--mono) !important; font-size: .74rem !important;
  letter-spacing: .12em; text-transform: uppercase; border: 1px solid var(--rule-strong) !important; background: transparent !important; color: var(--bone) !important;
}
.stButton > button:hover { border-color: var(--amber) !important; color: var(--amber) !important; }
.stButton > button[data-testid="stBaseButton-primary"], .stButton > button[kind="primary"] { background: var(--amber) !important; color: var(--ink) !important; border-color: var(--amber) !important; }
[data-testid="stFileUploaderDropzone"] { background: repeating-linear-gradient(-45deg, rgba(236,230,214,.03) 0 8px, transparent 8px 16px), var(--film) !important; border: 1px dashed var(--rule-strong) !important; border-radius: 0 !important; }
[data-testid="stExpander"] details { border-radius: 0 !important; border: 1px solid var(--rule) !important; background: rgba(16,22,26,.6); }
[data-testid="stExpander"] summary p { font-family: var(--mono) !important; font-size: .8rem !important; letter-spacing: .08em; text-transform: uppercase; }
[data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid var(--rule); }
[data-baseweb="tab"] p { font-family: var(--mono) !important; font-size: .78rem !important; letter-spacing: .1em; text-transform: uppercase; }
[data-testid="stSlider"] label p, [data-testid="stWidgetLabel"] p { font-family: var(--mono) !important; font-size: .72rem !important; letter-spacing: .14em; text-transform: uppercase; color: var(--muted) !important; }
hr { border-color: var(--rule) !important; }

/* Contact sheet captions */
.xl-sheet-cap { font: .7rem/1.35 var(--mono); color: var(--muted); margin: .25rem 0 .4rem; min-height: 2.8em; }
.xl-sheet-cap b { color: var(--bone); font-weight: 600; display: block; letter-spacing: .04em; }

.xl-foot { margin-top: 4rem; padding-top: 1rem; border-top: 1px solid var(--rule); font: .72rem/1.6 var(--mono); color: var(--muted); }
"""


def inject_theme() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    st.markdown('<div class="xl-runhead">XRAYLENS · RESEARCH PROTOTYPE · NOT A DIAGNOSTIC DEVICE</div>',
                unsafe_allow_html=True)


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def esc(text) -> str:
    return html.escape(str(text))


def section(num: str, title: str, sub: str = "") -> None:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    html_block(f'<div class="xl-sec"><div class="num">{num}</div><h2>{title}</h2>{sub_html}</div>')


def note(body: str, kind: str = "") -> None:
    html_block(f'<div class="xl-note {kind}"><p>{body}</p></div>')


def pct(x: float, digits: int = 1) -> str:
    return "—" if x != x else f"{x * 100:.{digits}f}%"


def ci(lo_hi, digits: int = 1) -> str:
    lo, hi = lo_hi
    return "" if lo != lo else f"95% CI {lo * 100:.{digits}f}–{hi * 100:.{digits}f}%"


def stats_row(items: list[tuple[str, str, str]]) -> None:
    cells = "".join(f'<div><div class="k">{k}</div><div class="v">{v}</div><div class="ci">{c}</div></div>'
                    for k, v, c in items)
    html_block(f'<div class="xl-stats">{cells}</div>')


def score_readout(label: str, score: float, threshold: float, extra: dict[str, str]) -> None:
    pos = min(max(score, 0.0), 1.0) * 100
    kv = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in extra.items())
    html_block(f"""
<div class="xl-readout">
  <div class="lab">Model output · class with score ≥ threshold</div>
  <div class="call {label.lower()}">{label.title()}</div>
  <div class="xl-track" role="img" aria-label="Pneumonia score {score:.4f} against threshold {threshold}">
    <div class="bar"></div>
    <div class="thr" style="left:{threshold * 100}%"><span>threshold {threshold}</span></div>
    <div class="pin" style="left:{pos}%"></div>
  </div>
  <div class="xl-axis"><span>0 · NORMAL</span><span>PNEUMONIA · 1</span></div>
  <dl class="xl-kv">{kv}</dl>
</div>""")


def altair_theme():
    """Register a matching Altair theme (transparent, mono axes, recessive grid)."""
    import altair as alt

    @alt.theme.register("xraylens", enable=True)
    def _theme():
        return alt.theme.ThemeConfig({
            "config": {
                "background": "transparent",
                "font": "JetBrains Mono",
                "view": {"stroke": None},
                "axis": {"labelColor": MUTED, "titleColor": MUTED, "gridColor": "rgba(236,230,214,.08)",
                         "domainColor": "rgba(236,230,214,.3)", "tickColor": "rgba(236,230,214,.3)",
                         "labelFont": "JetBrains Mono", "titleFont": "JetBrains Mono",
                         "titleFontWeight": 400, "labelFontSize": 10, "titleFontSize": 10},
                "legend": {"labelColor": BONE, "titleColor": MUTED, "labelFont": "JetBrains Mono",
                           "titleFont": "JetBrains Mono", "titleFontWeight": 400, "orient": "top"},
                "header": {"labelColor": BONE, "titleColor": MUTED, "labelFont": "JetBrains Mono",
                           "titleFont": "JetBrains Mono", "labelFontSize": 11},
                "title": {"color": BONE, "font": "Newsreader", "fontWeight": 400, "fontSize": 15, "anchor": "start"},
                "text": {"color": BONE, "font": "JetBrains Mono"},
            }
        })
