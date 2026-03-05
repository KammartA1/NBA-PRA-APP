from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

OUTPUT = "/home/user/NBA_Quant_Engine_Overview.pdf"

BG       = colors.HexColor("#0A1628")
ACCENT   = colors.HexColor("#00FFB2")
BLUE     = colors.HexColor("#00AAFF")
DIM      = colors.HexColor("#4A607A")
WHITE    = colors.HexColor("#EEF4FF")
ORANGE   = colors.HexColor("#FFB800")
RED      = colors.HexColor("#FF3358")
CARD_BG  = colors.HexColor("#0D1F35")
BORDER   = colors.HexColor("#1E2D3D")

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=letter,
    leftMargin=0.65*inch,
    rightMargin=0.65*inch,
    topMargin=0.65*inch,
    bottomMargin=0.65*inch,
    title="NBA Prop Quant Engine v2.1 — Overview",
    author="NBA Quant Engine",
)

# ── Canvas background ────────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(BG)
    canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
    # Header bar
    canvas.setFillColor(ACCENT)
    canvas.rect(0, letter[1] - 0.38*inch, letter[0], 0.38*inch, fill=1, stroke=0)
    # Header text
    canvas.setFillColor(BG)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(0.65*inch, letter[1] - 0.24*inch, "NBA PROP QUANT ENGINE v2.1")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(letter[0] - 0.65*inch, letter[1] - 0.24*inch, "CONFIDENTIAL — TRADING OVERVIEW")
    # Footer
    canvas.setFillColor(DIM)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(0.65*inch, 0.32*inch, "NBA Quant Engine v2.1 — Audit-Hardened + Enhanced")
    canvas.drawRightString(letter[0] - 0.65*inch, 0.32*inch, f"Page {doc.page}")
    # Footer line
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(0.65*inch, 0.44*inch, letter[0] - 0.65*inch, 0.44*inch)
    canvas.restoreState()

# ── Styles ───────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def sty(name, **kwargs):
    base = dict(
        fontName="Helvetica", fontSize=9, textColor=WHITE,
        leading=14, spaceBefore=0, spaceAfter=0,
        backColor=None, leftIndent=0,
    )
    base.update(kwargs)
    return ParagraphStyle(name, **base)

H1   = sty("H1",  fontName="Helvetica-Bold", fontSize=16, textColor=ACCENT, leading=20, spaceAfter=4)
H2   = sty("H2",  fontName="Helvetica-Bold", fontSize=11, textColor=ACCENT, leading=15, spaceBefore=14, spaceAfter=3)
H3   = sty("H3",  fontName="Helvetica-Bold", fontSize=9,  textColor=BLUE,   leading=13, spaceBefore=8,  spaceAfter=2)
BODY = sty("BODY",fontSize=8.5, textColor=WHITE, leading=13, spaceAfter=3)
DIM_ = sty("DIM", fontSize=8,   textColor=DIM,  leading=12)
CODE = sty("CODE",fontName="Courier", fontSize=7.5, textColor=ACCENT, leading=11,
           backColor=CARD_BG, leftIndent=8, spaceAfter=2)
RULE = sty("RULE",fontName="Helvetica-Bold", fontSize=7.5, textColor=ORANGE, leading=11, leftIndent=4)
SUB  = sty("SUB", fontSize=7.5, textColor=DIM, leading=11, spaceAfter=1)

def h1(t):  return Paragraph(t, H1)
def h2(t):  return Paragraph(t, H2)
def h3(t):  return Paragraph(t, H3)
def body(t): return Paragraph(t, BODY)
def dim(t): return Paragraph(t, DIM_)
def code(t): return Paragraph(t, CODE)
def rule(t): return Paragraph(t, RULE)
def sp(h=6): return Spacer(1, h)
def hr():   return HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=6, spaceBefore=4)

def section_label(t):
    return Paragraph(
        f"<font color='#{ACCENT.hexval()[2:]}' size='7'>"
        f"◆ {t.upper()}</font>",
        sty("SL", fontName="Helvetica-Bold", fontSize=7, textColor=ACCENT,
            leading=10, spaceBefore=12, spaceAfter=4)
    )

def bullet(items, color=WHITE):
    rows = []
    hex_c = f"#{color.hexval()[2:]}"
    for item in items:
        rows.append(Paragraph(f"<font color='{hex_c}'>•</font>  {item}", BODY))
        rows.append(sp(2))
    return rows

def edge_table(rows_data, col_widths, header=None):
    data = []
    if header:
        data.append(header)
    data.extend(rows_data)
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("BACKGROUND",   (0, 0), (-1, 0 if not header else 0), CARD_BG),
        ("TEXTCOLOR",    (0, 0), (-1, -1), WHITE),
        ("FONTNAME",     (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 0), (-1, -1), 7.5),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [BG, CARD_BG]),
        ("GRID",         (0, 0), (-1, -1), 0.3, BORDER),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]
    if header:
        style += [
            ("FONTNAME",  (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (-1, 0), ACCENT),
            ("BACKGROUND",(0, 0), (-1, 0), CARD_BG),
        ]
    t.setStyle(TableStyle(style))
    return t

# ── Content ──────────────────────────────────────────────────────
story = [sp(14)]

# Title block
story += [
    h1("NBA PROP QUANT ENGINE v2.1"),
    Paragraph("<font color='#4A607A' size='8'>Audit-Hardened · 9 Core Fixes · Gaussian Copula Parlays · Real Half-Game Data</font>",
              sty("sub2", fontSize=8, textColor=DIM, leading=12)),
    sp(6), hr(),
]

# ─── SECTION 1: Pipeline ─────────────────────────────────────────
story.append(section_label("1. The Full Pipeline"))
story.append(h2("How Every Projection Is Built"))
story.append(body(
    "The engine runs every prop through a deterministic 9-stage pipeline. "
    "Each stage adds signal. Each stage has a failure mode that's handled. "
    "Nothing reaches a stake recommendation without passing all gates."
))
story.append(sp(6))

pipeline_rows = [
    ["Stage", "What Happens", "Why It Matters"],
    ["1  ODDS FETCH",     "The Odds API → all books in parallel", "Best available price, not just one book"],
    ["2  NO-VIG STRIP",   "Remove bookmaker margin from implied prob", "Compare model vs true market, not vig-inflated"],
    ["3  GAME LOG FETCH", "NBA API bulk cache (30-min TTL)", "Freshest data; intra-day lineup changes captured"],
    ["4  HALF SPLITS",    "BoxScoreTraditionalV2 per period range", "Real H1/H2/Q1 distributions, not naive 52%/48%"],
    ["5  BOOTSTRAP",      "8,000 resamples with exponential decay (λ per stat)", "Recent games weighted more; low-variance P(OVER)"],
    ["6  BAYESIAN SHRINK","Prior from Guard/Wing/Big positional table", "Small samples pulled toward position baseline"],
    ["7  CONTEXT MULTS",  "Rest × Home/Away × Opp × Blowout × Injury", "Situational factors applied multiplicatively"],
    ["8  SKEW GATE",      "CV ≤ 0.35; tighter for skewed markets+wrong direction", "Blocks/3PM Unders need higher EV to qualify"],
    ["9  KELLY SIZING",   "Fractional Kelly (default ¼) capped at 5% BR", "Optimal compounding; ruin probability near zero"],
]
story.append(edge_table(
    pipeline_rows[1:],
    col_widths=[1.0*inch, 2.4*inch, 2.8*inch],
    header=pipeline_rows[0],
))
story.append(sp(6))

# ─── SECTION 2: Edge Sources ─────────────────────────────────────
story.append(section_label("2. Where the Edge Comes From"))
story.append(h2("8 Structural Advantages"))

edges = [
    ("Exponential Recency Decay",
     ACCENT,
     "Books price props from 15-game rolling averages. Your bootstrap weights the "
     "last 5 games at λ=0.88 per game. When a player is hot or cold, your model "
     "moves faster than the market reprices."),
    ("No-Vig True Probability",
     ACCENT,
     "A bet at +100 (50.0% implied with vig) might be 52.4% true implied after "
     "stripping the book's 4.8% margin. All edge calculations use the true number. "
     "This prevents overstating your advantage."),
    ("Real Half-Game Distributions",
     ACCENT,
     "H1/H2/Q1 markets: when ≥3 games of BoxScoreTraditionalV2 period data exist, "
     "half_factor resets to 1.0 and the full pipeline runs on actual splits — not "
     "52% of a full-game number. Books use the naive scaling. You don't."),
    ("Bayesian Shrinkage (Position-Specific)",
     BLUE,
     "5-game samples get ~40% prior weight toward Guard/Wing/Big baselines. "
     "20-game samples get ~15%. Prevents overreacting to short-term noise. "
     "Half-game priors are scaled by the half_factor so units match."),
    ("Blowout Probability Adjustment",
     BLUE,
     "Team spread and total → blowout probability from Scoreboard. "
     "When blowout_prob > 20%, proj_minutes trimmed up to 12%. "
     "Books don't pull props for garbage-time risk. You already discount it."),
    ("PSD Parlay Correlation Matrix",
     BLUE,
     "Joint probability uses a full N×N Spearman correlation matrix, "
     "eigenvalue-clipped to positive semidefinite, with 3,000 Gaussian copula "
     "simulations per combo. The old heuristic (corr_adj × 0.3) was wrong. "
     "Stakes are now mathematically correct."),
    ("Skewness-Adjusted Gate",
     ORANGE,
     "Right-skewed markets (Blocks, 3PM, Steals) require EV ≥ 8% to qualify "
     "when CV > 0.30 and the bet direction is exposed to the fat tail. "
     "The gate is direction-aware: positive skew + Under is tightened, "
     "negative skew + Over is tightened."),
    ("Closing Line Value Tracking",
     ORANGE,
     "CLV = your model probability vs closing implied price (no-vig). "
     "Consistent positive CLV is the gold-standard proof that a model has real "
     "edge — not variance. Track this in HISTORY. If CLV is consistently "
     "negative, tighten your thresholds."),
]

for title, col, desc in edges:
    hex_c = f"#{col.hexval()[2:]}"
    story.append(KeepTogether([
        Paragraph(f"<font color='{hex_c}'>▸</font>  <b>{title}</b>", BODY),
        Paragraph(f"<font color='#4A607A'>    {desc}</font>",
                  sty("edgedesc", fontSize=8, textColor=DIM, leading=12, leftIndent=12, spaceAfter=6)),
    ]))

story.append(hr())

# ─── SECTION 3: Compounding Strategy ─────────────────────────────
story.append(section_label("3. Compounding Strategy"))
story.append(h2("How to Profit Compoundly"))
story.append(body(
    "Kelly criterion compounding is mathematically proven to maximize long-run "
    "bankroll growth rate. At ¼ Kelly the ruin probability is near zero while "
    "growth is still roughly 60–70% of optimal Kelly. Every bet is sized as a "
    "<b>fraction of current bankroll</b>, not a fixed dollar amount. "
    "Wins grow the base. Losses shrink the next bet."
))
story.append(sp(8))

# Phase table
phase_rows = [
    ["Phase", "Bankroll", "Action", "Threshold"],
    ["1  Calibrate\n(Wks 1–2)",   "$1,000",       "Scanner only. Log every result in HISTORY.\nBlock Chaotic Regime: ON", "p_cal ≥ 0.57 · EV% ≥ 3% · Gate PASS"],
    ["2  Add Unders\n(Wks 3–4)",  "$1,100–1,200", "Enable 'Show Under opportunities'.\nTarget: Blocks/3PM/FTM Unders with right skew", "p_under ≥ 0.57 · CV < 0.28 · skew > 0.3"],
    ["3  Parlays\n(Month 2)",      "$1,200–1,400", "Only Parlay Optimizer output.\n2-leg only initially. PSD joint prob > 40%", "EV% > 8% after copula sim · n_legs ≤ 3"],
    ["4  Scale\n(Month 3+)",       "$1,400+",       "Increase Kelly to 0.30 if Brier ≤ 0.22.\nFocus on H1/Alt/Q1 markets", "Consistent CLV > 0 · Brier ≤ 0.22"],
]
story.append(edge_table(
    phase_rows[1:],
    col_widths=[1.0*inch, 0.95*inch, 2.4*inch, 2.0*inch],
    header=phase_rows[0],
))
story.append(sp(10))

# Best markets
story.append(h3("Best Markets for Compounding"))
market_rows = [
    ["Market",          "Why It Compounds Well",                               "Edge Source"],
    ["H1 Points / PRA", "Real boxscore splits replace naive 52% scaling",      "Half-game data pipeline"],
    ["Alt Lines",        "Conservative book pricing vs bootstrap distribution", "Exponential decay"],
    ["Blocks / Steals Under","Right-skewed → Unders underpriced when CV controlled","Skewness gate"],
    ["FTM / FTA",        "Very low variance for consistent FT shooters",        "CV gate (low vol)"],
    ["Q1 Points",        "Smallest sample booked — highest pricing error",      "Bayesian shrinkage"],
]
story.append(edge_table(
    market_rows[1:],
    col_widths=[1.2*inch, 2.8*inch, 2.3*inch],
    header=market_rows[0],
))
story.append(sp(8))

# Avoid
story.append(h3("Avoid for Compounding"))
story += bullet([
    "<b>Full-game Points on star players</b> — most liquid, tightest lines, books are most efficient here",
    "<b>Triple Double / First Basket</b> — binary outcome, no signal from bootstrap",
    "<b>B2B games on high-rest-risk players</b> — blowout risk compounds DNP risk, model variance spikes",
    "<b>Any leg where Gate = FAIL</b> — never override the volatility gate",
], color=DIM)
story.append(sp(6))
story.append(hr())

# ─── SECTION 4: Kelly Math ───────────────────────────────────────
story.append(section_label("4. Kelly Sizing — The Math"))
story.append(h2("Example Calculation"))

kelly_rows = [
    ["Variable",             "Example Value",  "Notes"],
    ["p_cal (your model)",   "0.60",           "60% probability of Over hitting"],
    ["p_implied (no-vig)",   "0.52",           "True market implied (vig stripped)"],
    ["decimal odds",          "1.92",           "Equivalent to -120 American"],
    ["EV raw",               "+15.4%",         "0.60×0.92 − 0.40×1.0"],
    ["Full Kelly fraction",   "16.7%",          "EV / (decimal − 1) = 0.154 / 0.92"],
    ["¼ Kelly fraction",      "4.2%",           "Conservative; near-zero ruin risk"],
    ["Stake on $1,000 BR",    "$42",            "min(BR × 0.25 × Kelly_f, BR × 5%)"],
    ["After 100 such bets",   "~$1,450–1,600",  "Expected range at 3–5% avg EV"],
]
story.append(edge_table(
    kelly_rows[1:],
    col_widths=[1.8*inch, 1.4*inch, 3.1*inch],
    header=kelly_rows[0],
))
story.append(sp(10))

# ─── SECTION 5: Daily Workflow ───────────────────────────────────
story.append(section_label("5. Daily Operating Workflow"))
story.append(h2("Game-Day Checklist"))

steps = [
    ("Pre-game (2–4 hrs before tip)", ACCENT, [
        "Sidebar → confirm bankroll is accurate (update after yesterday's results)",
        "LIVE SCANNER → Load All Game Logs → Run Scan",
        "Filter: p_cal ≥ 0.57 · EV% ≥ 3% · Gate PASS · Regime not Chaotic",
        "Enable 'Show Under opportunities' → check Under panel",
        "Check DNP / injury news for any flagged players",
    ]),
    ("Parlay building", BLUE, [
        "RESULTS tab → Parlay Optimizer → set payout multiplier",
        "Only use combos where EV% > 0 after copula simulation",
        "Max 3 legs · Max 5% of bankroll total parlay exposure",
        "Prefer cross-team legs (lower correlation → better joint prob)",
    ]),
    ("Post-game logging", ORANGE, [
        "HISTORY tab → log each bet as HIT or MISS",
        "CALIBRATION tab → check Brier score trend",
        "If CLV tracking shows consistent negative CLV → raise p_cal threshold to 0.60",
        "Weekly: check if any markets have systematic over/under-performance",
    ]),
]

for phase, col, items in steps:
    hex_c = f"#{col.hexval()[2:]}"
    story.append(Paragraph(f"<font color='{hex_c}'><b>{phase}</b></font>", BODY))
    story.append(sp(2))
    story += bullet(items, color=WHITE)
    story.append(sp(4))

story.append(hr())

# ─── Section 6: Rules ────────────────────────────────────────────
story.append(section_label("6. Hard Rules"))
story.append(h2("Never Break These"))

rules = [
    "Never bet more than 5% of bankroll on a single leg (engine hard-caps this)",
    "Daily Loss Stop at 15% of bankroll — built into sidebar; if hit, stop for the day",
    "Never manually combine parlay legs — only use optimizer output with PSD joint prob",
    "Never override a GATE FAIL — the volatility gate exists to protect you from high-CV traps",
    "Always log results in HISTORY — the calibrator cannot improve without ground truth",
    "CLV is the truth test — if you're not beating closing lines, the model isn't working; tighten thresholds",
]

for i, r in enumerate(rules, 1):
    hex_r = f"#{RED.hexval()[2:]}"
    story.append(Paragraph(
        f"<font color='{hex_r}'><b>{i}.</b></font>  {r}",
        sty(f"rule{i}", fontSize=8.5, textColor=WHITE, leading=13, spaceAfter=5, leftIndent=4)
    ))

story.append(sp(10))
story.append(hr())

# Footer note
story.append(Paragraph(
    "NBA Prop Quant Engine v2.1 · All 9 audit fixes applied · "
    "Bayesian shrinkage · PSD copula parlays · Real half-game splits · Under toggle",
    sty("footer_note", fontSize=7, textColor=DIM, leading=11, alignment=TA_CENTER)
))

# ── Build ─────────────────────────────────────────────────────────
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"PDF written to: {OUTPUT}")
