"""
Bloomberg Terminal-style design system for the NBA Prop Alpha Engine.

Provides:
    GLOBAL_CSS       - Inject via st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    PLOTLY_TEMPLATE  - Pass as layout template to any Plotly figure
    PAGE_CONFIG      - Pass as **kwargs to st.set_page_config()
    Helper functions  - metric_card, status_dot, badge, card_container,
                        table_html, kill_switch_bar, nav_item
"""

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
PAGE_CONFIG: dict = {
    "page_title": "NBA Prop Alpha Engine",
    "page_icon": ":basketball:",
    "layout": "wide",
    "initial_sidebar_state": "collapsed",
}

# ---------------------------------------------------------------------------
# PLOTLY TEMPLATE
# ---------------------------------------------------------------------------
PLOTLY_TEMPLATE: dict = {
    "layout": {
        "paper_bgcolor": "transparent",
        "plot_bgcolor": "transparent",
        "font": {
            "family": "JetBrains Mono, monospace",
            "size": 11,
            "color": "#8B8B96",
        },
        "xaxis": {
            "gridcolor": "#2A2A2E",
            "zerolinecolor": "#2A2A2E",
            "tickfont": {"family": "JetBrains Mono, monospace", "size": 11, "color": "#8B8B96"},
        },
        "yaxis": {
            "gridcolor": "#2A2A2E",
            "zerolinecolor": "#2A2A2E",
            "tickfont": {"family": "JetBrains Mono, monospace", "size": 11, "color": "#8B8B96"},
        },
        "colorway": ["#4C9AFF", "#00D26A", "#FF4757", "#FFBE0B"],
        "title": {"text": ""},
        "margin": {"l": 40, "r": 20, "t": 20, "b": 40},
    }
}

# ---------------------------------------------------------------------------
# GLOBAL CSS
# ---------------------------------------------------------------------------
GLOBAL_CSS = """
<style>
/* ── Google Fonts ──────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

/* ── CSS Custom Properties ─────────────────────────────────────────────── */
:root {
    --bg: #0A0A0B;
    --surface: #141416;
    --surface-elevated: #1C1C1F;
    --border: #2A2A2E;
    --text-primary: #E8E8EC;
    --text-secondary: #8B8B96;
    --green: #00D26A;
    --red: #FF4757;
    --amber: #FFBE0B;
    --blue: #4C9AFF;
}

/* ── Hide Streamlit Chrome ─────────────────────────────────────────────── */
#MainMenu {visibility: hidden !important;}
footer {visibility: hidden !important;}
header {visibility: hidden !important;}
[data-testid="stToolbar"] {display: none !important;}
[data-testid="stDecoration"] {display: none !important;}
[data-testid="stStatusWidget"] {display: none !important;}
.viewerBadge_container__r5tak {display: none !important;}
[data-testid="manage-app-button"] {display: none !important;}

/* ── Page Background ───────────────────────────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main,
.block-container {
    background-color: var(--bg) !important;
}

html, body {
    background-color: var(--bg) !important;
}

/* ── Default Typography ────────────────────────────────────────────────── */
html, body, [class*="css"],
.stApp, .stMarkdown, p, span, label, li, td, th, div {
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
    font-size: 11px;
}

h1 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 24px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
    margin: 0 0 8px 0 !important;
    padding: 0 !important;
}

h2 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.01em !important;
    margin: 0 0 4px 0 !important;
    padding: 0 !important;
}

h3 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    margin: 0 0 4px 0 !important;
    padding: 0 !important;
}

/* ── Monospaced Numbers / Metrics ──────────────────────────────────────── */
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
.metric-value, .mono,
code, pre,
td.mono, span.mono {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}

/* ── st.metric Override ────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 16px 20px !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 24px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}

/* ── st.dataframe / st.table Override ──────────────────────────────────── */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

[data-testid="stDataFrame"] table,
[data-testid="stTable"] table {
    background-color: transparent !important;
}

[data-testid="stDataFrame"] th,
[data-testid="stTable"] th {
    background-color: var(--surface-elevated) !important;
    color: var(--text-secondary) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 8px 12px !important;
    height: 40px !important;
}

[data-testid="stDataFrame"] td,
[data-testid="stTable"] td {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: var(--text-primary) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 8px 12px !important;
    height: 40px !important;
}

[data-testid="stDataFrame"] tr:nth-child(odd) td,
[data-testid="stTable"] tr:nth-child(odd) td {
    background-color: var(--surface) !important;
}

[data-testid="stDataFrame"] tr:nth-child(even) td,
[data-testid="stTable"] tr:nth-child(even) td {
    background-color: var(--bg) !important;
}

/* ── st.button Override ────────────────────────────────────────────────── */
.stButton > button {
    background-color: var(--surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}

.stButton > button:hover {
    background-color: var(--surface-elevated) !important;
    border-color: var(--blue) !important;
    color: var(--text-primary) !important;
}

.stButton > button:active {
    background-color: var(--border) !important;
}

/* ── st.selectbox Override ─────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 11px !important;
}

[data-testid="stSelectbox"] label {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

/* ── st.multiselect Override ───────────────────────────────────────────── */
[data-testid="stMultiSelect"] > div > div {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 11px !important;
}

[data-testid="stMultiSelect"] label {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background-color: var(--surface-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 11px !important;
}

/* ── st.slider Override ────────────────────────────────────────────────── */
[data-testid="stSlider"] label {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] div {
    background-color: var(--border) !important;
}

[data-testid="stSlider"] [role="slider"] {
    background-color: var(--blue) !important;
    border-color: var(--blue) !important;
}

[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: var(--text-secondary) !important;
}

/* ── st.number_input Override ──────────────────────────────────────────── */
[data-testid="stNumberInput"] label {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

[data-testid="stNumberInput"] input {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}

/* ── st.tabs Override ──────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-secondary) !important;
    background-color: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 20px !important;
    transition: all 0.15s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--blue) !important;
    background-color: transparent !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--blue) !important;
}

.stTabs [data-baseweb="tab-panel"] {
    background-color: transparent !important;
    padding-top: 16px !important;
}

/* ── Sidebar Override ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background-color: var(--surface) !important;
    padding: 24px 16px !important;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    justify-content: flex-start !important;
    text-align: left !important;
    background-color: transparent !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 12px !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background-color: var(--surface-elevated) !important;
    border: none !important;
}

/* ── Kill Streamlit Padding / Margin Defaults ──────────────────────────── */
.block-container {
    padding-top: 24px !important;
    padding-bottom: 24px !important;
    max-width: 100% !important;
}

[data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

.element-container {
    margin: 0 !important;
}

/* ── Card Styling ──────────────────────────────────────────────────────── */
.gqe-card {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px;
}

/* ── Status Dot Animation ──────────────────────────────────────────────── */
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0, 210, 106, 0.4); }
    50% { box-shadow: 0 0 0 4px rgba(0, 210, 106, 0); }
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.status-dot.active {
    background-color: var(--green);
    animation: pulse-green 2s ease-in-out infinite;
}

.status-dot.warning {
    background-color: var(--amber);
}

.status-dot.dead {
    background-color: var(--red);
}

.status-dot.unknown {
    background-color: var(--text-secondary);
}

/* ── Row Hover Effect ──────────────────────────────────────────────────── */
.gqe-table tr {
    transition: all 0.1s ease;
    border-left: 3px solid transparent;
}

.gqe-table tr:hover {
    border-left: 3px solid var(--blue);
    background-color: var(--surface-elevated) !important;
}

/* ── Custom Table Styling ──────────────────────────────────────────────── */
.gqe-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 11px;
}

.gqe-table th {
    background-color: var(--surface-elevated);
    color: var(--text-secondary);
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    text-align: left;
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    height: 40px;
}

.gqe-table td {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-primary);
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    height: 40px;
    vertical-align: middle;
}

.gqe-table tr:nth-child(odd) td {
    background-color: var(--surface);
}

.gqe-table tr:nth-child(even) td {
    background-color: var(--bg);
}

/* ── Badge / Pill ──────────────────────────────────────────────────────── */
.gqe-badge {
    display: inline-block;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 2px 8px;
    border-radius: 4px;
    line-height: 1.6;
}

/* ── Metric Card ───────────────────────────────────────────────────────── */
.gqe-metric {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
}

.gqe-metric-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 0 0 8px 0;
    line-height: 1;
}

.gqe-metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1;
    margin: 0;
}

.gqe-metric-delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    margin: 8px 0 0 0;
    line-height: 1;
}

/* ── Kill Switch Bar ───────────────────────────────────────────────────── */
.gqe-kill-bar {
    display: flex;
    gap: 24px;
    align-items: center;
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 20px;
    flex-wrap: wrap;
}

.gqe-kill-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Navigation Item ───────────────────────────────────────────────────── */
.gqe-nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 6px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s ease;
    text-decoration: none;
}

.gqe-nav-item:hover {
    background-color: var(--surface-elevated);
    color: var(--text-primary);
}

.gqe-nav-item.active {
    background-color: var(--surface-elevated);
    color: var(--text-primary);
    border-left: 3px solid var(--blue);
}

.gqe-nav-icon {
    font-size: 16px;
    width: 20px;
    text-align: center;
}

/* ── Dropdown menus (selectbox / multiselect popover) ──────────────────── */
[data-baseweb="popover"] {
    background-color: var(--surface-elevated) !important;
    border: 1px solid var(--border) !important;
}

[data-baseweb="popover"] li {
    background-color: var(--surface-elevated) !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 11px !important;
}

[data-baseweb="popover"] li:hover {
    background-color: var(--border) !important;
}

/* ── Text input fields (search boxes etc.) ─────────────────────────────── */
[data-testid="stTextInput"] input {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 11px !important;
}

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: var(--bg);
}

::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-secondary);
}
</style>
"""

# ---------------------------------------------------------------------------
# COLOR CONSTANTS (for use in Python code / Plotly traces)
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#0A0A0B",
    "surface": "#141416",
    "surface_elevated": "#1C1C1F",
    "border": "#2A2A2E",
    "text_primary": "#E8E8EC",
    "text_secondary": "#8B8B96",
    "green": "#00D26A",
    "red": "#FF4757",
    "amber": "#FFBE0B",
    "blue": "#4C9AFF",
}

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

_DELTA_COLORS = {
    "green": "#00D26A",
    "red": "#FF4757",
    "amber": "#FFBE0B",
    "blue": "#4C9AFF",
}


def metric_card(label: str, value: str, delta: str | None = None, delta_color: str = "green") -> str:
    """Return HTML for a hero metric card.

    Args:
        label: Micro-label displayed above the number.
        value: The big number string.
        delta: Optional delta string (e.g. "+3.2%").
        delta_color: One of "green", "red", "amber", "blue".
    """
    color = _DELTA_COLORS.get(delta_color, _DELTA_COLORS["green"])
    delta_html = ""
    if delta is not None:
        delta_html = (
            f'<div class="gqe-metric-delta" style="color:{color};">{delta}</div>'
        )
    return (
        f'<div class="gqe-metric">'
        f'  <div class="gqe-metric-label">{label}</div>'
        f'  <div class="gqe-metric-value">{value}</div>'
        f'  {delta_html}'
        f'</div>'
    )


def status_dot(status: str) -> str:
    """Return HTML for an 8px status circle.

    Args:
        status: One of "active" (green pulse), "warning" (amber),
                "dead" (red), "unknown" (gray).
    """
    valid = {"active", "warning", "dead", "unknown"}
    css_class = status if status in valid else "unknown"
    return f'<span class="status-dot {css_class}"></span>'


def badge(text: str, color: str = "blue") -> str:
    """Return HTML for a small badge/pill.

    Args:
        text: Badge label.
        color: One of "green", "red", "amber", "blue".
    """
    fg = _DELTA_COLORS.get(color, _DELTA_COLORS["blue"])
    # 15% opacity background derived from the color
    return (
        f'<span class="gqe-badge" '
        f'style="color:{fg}; background-color:{fg}26;">'
        f'{text}</span>'
    )


def card_container(content_html: str) -> str:
    """Wrap *content_html* in a styled card div."""
    return f'<div class="gqe-card">{content_html}</div>'


def table_html(
    headers: list[str],
    rows: list[list],
    column_formats: dict[int, callable] | None = None,
) -> str:
    """Build a full HTML table with the design system styling.

    Args:
        headers: List of column header strings.
        rows: List of row-lists (each row is a list of cell values).
        column_formats: Optional dict mapping column index to a callable
                        that receives the cell value and returns a string.
    """
    if column_formats is None:
        column_formats = {}

    header_cells = "".join(f"<th>{h}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = []
        for idx, cell in enumerate(row):
            fmt = column_formats.get(idx)
            display = fmt(cell) if fmt else str(cell)
            cells.append(f"<td>{display}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        f'<table class="gqe-table">'
        f'<thead><tr>{header_cells}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        f'</table>'
    )


def kill_switch_bar(switches: list[dict]) -> str:
    """Return HTML for the horizontal kill switch status bar.

    Args:
        switches: List of dicts with keys ``name`` (str) and ``status``
                  (one of "active", "warning", "dead", "unknown").
    """
    items = []
    for sw in switches:
        dot = status_dot(sw.get("status", "unknown"))
        name = sw.get("name", "")
        items.append(
            f'<div class="gqe-kill-item">{dot}<span>{name}</span></div>'
        )
    return f'<div class="gqe-kill-bar">{"".join(items)}</div>'


def nav_item(icon: str, label: str, active: bool = False) -> str:
    """Return HTML for a navigation item.

    Args:
        icon: Emoji or icon character.
        label: Navigation label text.
        active: Whether this item is currently selected.
    """
    active_class = " active" if active else ""
    return (
        f'<div class="gqe-nav-item{active_class}">'
        f'  <span class="gqe-nav-icon">{icon}</span>'
        f'  <span>{label}</span>'
        f'</div>'
    )
