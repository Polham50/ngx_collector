import os
import textwrap
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import gspread
import json
import plotly.express as px

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "data" / "metadata"
QUEUE_FILE   = METADATA_DIR / "annotation_queue.csv"

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NGX-FND Annotator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── State ──────────────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "annotator" not in st.session_state:
    st.session_state.annotator = None
if "sector_filter" not in st.session_state:
    st.session_state.sector_filter = "All"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

IS_DARK = st.session_state.theme == "dark"

# ── CSS Design System ──────────────────────────────────────────────────────────
css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');

:root {{
    --bg: {'#09090b' if IS_DARK else '#ffffff'};
    --bg-subtle: {'#0c0c0f' if IS_DARK else '#f9fafb'};
    --card: {'#18181b' if IS_DARK else '#ffffff'};
    --border: {'#27272a' if IS_DARK else '#e4e4e7'};
    --text: {'#fafafa' if IS_DARK else '#09090b'};
    --text-muted: {'#a1a1aa' if IS_DARK else '#71717a'};
    --accent: #2563eb;
    --accent-green: #16a34a;
    --accent-red: #dc2626;
    --radius: 12px;
    --shadow: {'none' if IS_DARK else '0 4px 6px -1px rgba(0,0,0,0.1),0 2px 4px -1px rgba(0,0,0,0.06)'};
}}

/* Hide Chrome */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
div[data-testid="stSidebarCollapsedControl"] {{
    display: none !important;
}}

/* Global */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
.main, .block-container, section[data-testid="stMain"] {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}}

.block-container {{
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1200px !important;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: var(--bg-subtle) !important;
    border-right: 1px solid var(--border) !important;
}}
[data-testid="stSidebar"] * {{
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}}

/* Custom Card */
.content-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.5rem;
}}

/* Brand */
.brand {{
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 10px;
}}
.brand-subtitle {{
    font-size: 0.9rem;
    color: var(--text-muted);
    font-weight: 400;
}}

/* Passage Text */
.passage-text {{
    font-size: 1.1rem;
    line-height: 1.7;
    color: var(--text);
    background: var(--bg-subtle);
    padding: 1.5rem;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    margin: 1.5rem 0;
}}

/* Metadata Badges */
.metadata-badge {{
    display: inline-block;
    padding: 4px 10px;
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.8rem;
    color: var(--text-muted);
    font-weight: 500;
    margin-right: 8px;
    margin-bottom: 8px;
}}

/* Labels */
.form-label {{
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 0.5rem;
    display: block;
    color: var(--text);
}}

/* Progress */
.progress-text {{
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}}

/* Stat box */
.stat-box {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
}}
.stat-label {{
    color: var(--text-muted);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.stat-value {{
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text);
}}
.hint-box {{
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 1rem;
}}

</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1)
def get_gspread_client():
    if "gcp_service_account" in st.secrets:
        credentials = dict(st.secrets["gcp_service_account"])
        return gspread.service_account_from_dict(credentials)
    return None

@st.cache_data(ttl=1)
def load_queue() -> pd.DataFrame:
    gc = get_gspread_client()
    if gc and "spreadsheet_url" in st.secrets:
        try:
            sh = gc.open_by_url(st.secrets["spreadsheet_url"])
            ws = sh.sheet1
            data = ws.get_all_records()
            if data:
                df = pd.DataFrame(data)
                df = df.astype(str)
                df = df.replace(["None", "nan", "NaN"], "")
                label_map = {"p": "positive", "n": "negative", "u": "neutral"}
                if "sentiment_label" in df.columns:
                    df["sentiment_label"] = df["sentiment_label"].replace(label_map)
                guid_map = {"p": "positive", "n": "negative", "u": "neutral", "c": "conditional"}
                if "guidance_type" in df.columns:
                    df["guidance_type"] = df["guidance_type"].replace(guid_map)
                return df
        except Exception as e:
            st.error(f"Error loading from Google Sheets: {e}")
            st.stop()

    if not QUEUE_FILE.exists():
        st.error(f"Annotation queue not found: {QUEUE_FILE}")
        st.stop()
    df = pd.read_csv(QUEUE_FILE, dtype=str)
    df = df.fillna("")
    label_map = {"p": "positive", "n": "negative", "u": "neutral"}
    if "sentiment_label" in df.columns:
        df["sentiment_label"] = df["sentiment_label"].replace(label_map)
    guid_map = {"p": "positive", "n": "negative", "u": "neutral", "c": "conditional"}
    if "guidance_type" in df.columns:
        df["guidance_type"] = df["guidance_type"].replace(guid_map)
    return df

def save_annotation(idx: int, updates: dict):
    gc = get_gspread_client()
    if gc and "spreadsheet_url" in st.secrets:
        try:
            sh = gc.open_by_url(st.secrets["spreadsheet_url"])
            ws = sh.sheet1
            headers = ws.row_values(1)
            
            cells_to_update = []
            sheet_row = int(idx) + 2
            
            for col_name, val in updates.items():
                if col_name in headers:
                    col_idx = headers.index(col_name) + 1
                    cells_to_update.append(
                        gspread.Cell(row=sheet_row, col=col_idx, value=str(val))
                    )
            
            if cells_to_update:
                ws.update_cells(cells_to_update)
            
            st.cache_data.clear()
            return
        except Exception as e:
            st.error(f"Error saving to Google Sheets: {e}")
            st.stop()

    # Fallback to local CSV update
    df = pd.read_csv(QUEUE_FILE, dtype=str)
    for k, v in updates.items():
        df.at[idx, k] = str(v)
    df.to_csv(QUEUE_FILE, index=False)
    st.cache_data.clear()

# ── App Header ─────────────────────────────────────────────────────────────────
head_left, head_right = st.columns([8, 1])
with head_left:
    st.markdown("""
    <div class="brand">
        📝 NGX-FND Annotator <span class="brand-subtitle">Financial Narrative Dataset</span>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    theme_label = "☀️ Light" if IS_DARK else "🌙 Dark"
    st.button(theme_label, on_click=toggle_theme, use_container_width=True)

st.markdown("<hr style='border-color: var(--border); margin: 1rem 0 2rem 0;'>", unsafe_allow_html=True)

# ── Login Screen ───────────────────────────────────────────────────────────────
if not st.session_state.annotator:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### Welcome to the Annotation Portal")
    st.markdown("Please enter your name or ID to begin your session. Your progress will be tracked automatically.")

    col1, _ = st.columns([1, 2])
    with col1:
        name_input = st.text_input("Annotator Name", placeholder="e.g. Timothy")
        if st.button("Start Annotating", type="primary", use_container_width=True):
            if name_input.strip():
                st.session_state.annotator = name_input.strip()
                st.rerun()
            else:
                st.warning("Please enter a valid name.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ── Load Data ─────────────────────────────────────────────────────────────────
df = load_queue()
annotator = st.session_state.annotator

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"**👤 Annotator:** {annotator}")
    st.markdown("---")

    # Sector filter
    all_sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
    sector_filter = st.selectbox(
        "🏭 Filter by Sector",
        options=all_sectors,
        index=all_sectors.index(st.session_state.sector_filter)
        if st.session_state.sector_filter in all_sectors else 0,
        key="sector_filter_select"
    )
    st.session_state.sector_filter = sector_filter

    st.markdown("---")

    # Progress stats
    total_passages = len(df)
    done_passages  = (df["annotation_status"] == "done").sum()
    skipped        = (df["annotation_status"] == "skipped").sum()
    # Count only good/acceptable quality
    annotatable    = df[df["quality"].isin(["good", "acceptable"])]
    ann_done       = (annotatable["annotation_status"] == "done").sum()

    st.markdown('<div class="stat-label">Overall Progress</div>', unsafe_allow_html=True)
    st.progress(int(done_passages) / max(int(total_passages), 1))
    st.markdown(f'<div class="stat-value">{done_passages}</div><div class="stat-label">annotated of {total_passages} total</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**[chart] By Sector**")
    for sec in sorted(df["sector"].dropna().unique()):
        sec_df   = df[df["sector"] == sec]
        sec_done = (sec_df["annotation_status"] == "done").sum()
        sec_ann  = sec_df[sec_df["quality"].isin(["good", "acceptable"])]
        sec_tot  = len(sec_ann)
        pct      = int(sec_done / max(sec_tot, 1) * 100)
        st.markdown(
            f'<div class="stat-box"><div class="stat-label">{sec}</div>'
            f'<div>{sec_done}/{sec_tot} ({pct}%)</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        '<div class="hint-box">⌨️ <b>Tips:</b><br>'
        '• Use <b>Tab</b> to move between fields<br>'
        '• Press <b>Enter</b> to submit<br>'
        '• Skip boilerplate passages<br>'
        '• Focus on outlook & CEO sections</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.annotator = None
        st.rerun()

# ── Main App Layout ────────────────────────────────────────────────────────────
tab_annotate, tab_dashboard = st.tabs(["📝 Annotate", "📊 Analytics Dashboard"])

with tab_annotate:
    # Only show good/acceptable quality passages; skip poor automatically
    quality_filter  = ["good", "acceptable"]
    pending_all = df[
        df["quality"].isin(quality_filter) &
        (
            (df["annotation_status"] == "pending") |
            (df["annotation_status"] == "ai_annotated") |
            (df["annotator"].isna()) |
            (df["annotator"] == "")
        )
    ].copy()

    # Apply sector filter
    if sector_filter != "All":
        pending_all = pending_all[pending_all["sector"] == sector_filter]
    
    # Sort by quality priority then section priority
    priority_order = {"good": 0, "acceptable": 1}
    section_order  = {"outlook": 0, "chairman_statement": 1, "ceo_review": 2, "operating_review": 3}
    pending_all["_q_order"] = pending_all["quality"].map(priority_order).fillna(9)
    pending_all["_s_order"] = pending_all["section"].map(section_order).fillna(9)
    pending_all = pending_all.sort_values(["_q_order", "_s_order"])
    
    # Progress for selected view
    total_view = len(df[df["quality"].isin(quality_filter)] if sector_filter == "All"
                     else df[(df["quality"].isin(quality_filter)) & (df["sector"] == sector_filter)])
    done_view  = len(df[
        (df["quality"].isin(quality_filter)) &
        (df["annotation_status"] == "done") &
        (df["sector"] == sector_filter if sector_filter != "All" else True)
    ])
    
    st.markdown(
        f'<div class="progress-text">Progress ({sector_filter}): {done_view} / {total_view} annotatable passages done</div>',
        unsafe_allow_html=True
    )
    st.progress(done_view / max(total_view, 1))
    
    if pending_all.empty:
        st.success(
            "🎉 All annotatable passages in this view are done! "
            + ("Try switching sector in the sidebar." if sector_filter != "All" else "")
        )
        # We don't stop the app completely, just this tab
    else:
        # Current passage
        idx = pending_all.index[0]
        row = df.loc[idx]
        is_ai_annotated = row.get("annotation_status") == "ai_annotated"
        ai_prefix = "🤖 (AI Pre-filled) " if is_ai_annotated else ""

        # ── Annotation UI ──────────────────────────────────────────────────────────────
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        
        # Metadata
        st.markdown(f"""
        <div>
            <span class="metadata-badge">🏢 {row.get('ticker', '')} ({row.get('company', '')})</span>
            <span class="metadata-badge">📅 {row.get('year', '')}</span>
            <span class="metadata-badge">[doc] {row.get('doc_type', '').replace('_',' ').title()}</span>
            <span class="metadata-badge">📑 {row.get('section', '').replace('_', ' ').title()}</span>
            <span class="metadata-badge">🏭 {row.get('sector', '')}</span>
            <span class="metadata-badge">📏 {row.get('word_count', '')} words</span>
            <span class="metadata-badge">⭐ Quality: {row.get('quality', '')}</span>
            <span class="metadata-badge">📝 Status: {row.get('annotation_status', 'pending')}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Passage Text
        text = str(row.get("text", ""))
        st.markdown(f'<div class="passage-text">{text}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Annotation Form ────────────────────────────────────────────────────────────
        with st.form(key="annotation_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
        
            with col1:
                st.markdown(f'<span class="form-label">1. Overall Sentiment {ai_prefix}</span>', unsafe_allow_html=True)
                sentiment_options = ["positive", "negative", "neutral"]
                default_sentiment = row.get("sentiment_label", "neutral") if is_ai_annotated else "neutral"
                if default_sentiment not in sentiment_options: default_sentiment = "neutral"
                sentiment = st.radio(
                    "What is the overall tone of this passage?",
                    options=sentiment_options,
                    index=sentiment_options.index(default_sentiment),
                    format_func=lambda x: x.capitalize(),
                    horizontal=True,
                    label_visibility="collapsed"
                )
        
                st.markdown(f'<span class="form-label" style="margin-top: 1rem;">2. Intensity {ai_prefix}</span>', unsafe_allow_html=True)
                intensity_options = ["mild", "moderate", "strong"]
                default_intensity = row.get("sentiment_intensity", "moderate") if is_ai_annotated else "moderate"
                if default_intensity not in intensity_options: default_intensity = "moderate"
                intensity = st.radio(
                    "How strong is the sentiment?",
                    options=intensity_options,
                    index=intensity_options.index(default_intensity),
                    format_func=lambda x: x.capitalize(),
                    horizontal=True,
                    label_visibility="collapsed"
                )
        
            with col2:
                st.markdown(f'<span class="form-label">3. Forward Guidance {ai_prefix}</span>', unsafe_allow_html=True)
                guidance_options = ["no", "yes"]
                default_guidance = "yes" if str(row.get("has_guidance", "")).lower() == "true" else "no"
                has_guidance_str = st.radio(
                    "Does this contain forward-looking statements?",
                    options=guidance_options,
                    index=guidance_options.index(default_guidance) if is_ai_annotated else 0,
                    format_func=lambda x: x.capitalize(),
                    horizontal=True,
                    label_visibility="collapsed"
                )
                has_guidance = has_guidance_str == "yes"
        
                guidance_type = None
                guidance_span = None
        
                if has_guidance:
                    st.markdown(f'<span class="form-label" style="margin-top: 1rem;">4. Guidance Type {ai_prefix}</span>', unsafe_allow_html=True)
                    gtype_options = ["positive", "negative", "neutral", "conditional"]
                    default_gtype = row.get("guidance_type", "neutral") if is_ai_annotated else "neutral"
                    if default_gtype not in gtype_options: default_gtype = "neutral"
                    guidance_type = st.selectbox(
                        "What type of guidance is this?",
                        options=gtype_options,
                        index=gtype_options.index(default_gtype),
                        format_func=lambda x: x.capitalize(),
                        label_visibility="collapsed"
                    )
        
                    st.markdown(f'<span class="form-label" style="margin-top: 1rem;">5. Key Guidance Sentence {ai_prefix}</span>', unsafe_allow_html=True)
                    default_gspan = row.get("guidance_span", "") if is_ai_annotated else ""
                    guidance_span = st.text_area("Paste the exact sentence", value=default_gspan, height=80, label_visibility="collapsed")

            st.markdown('<span class="form-label" style="margin-top: 1rem;">6. Annotation Notes (Optional)</span>', unsafe_allow_html=True)
            notes = st.text_input("Any extra context?", label_visibility="collapsed")
        
            st.markdown("<hr style='border-color: var(--border);'>", unsafe_allow_html=True)
        
            col_submit, col_skip, _ = st.columns([2, 1, 5])
            with col_submit:
                submitted = st.form_submit_button("[OK] Save & Next Passage", type="primary", use_container_width=True)
            with col_skip:
                skipped = st.form_submit_button("⏭ Skip", use_container_width=True)
        
            if submitted:
                updates = {
                    "sentiment_label": sentiment,
                    "sentiment_intensity": intensity,
                    "has_guidance": str(has_guidance),
                    "guidance_type": guidance_type if guidance_type else "",
                    "guidance_span": guidance_span if guidance_span else "",
                    "annotation_notes": notes,
                    "annotator": annotator,
                    "annotation_status": "done",
                    "annotated_at": datetime.now().isoformat()
                }
                save_annotation(idx, updates)
                st.rerun()
        
            if skipped:
                updates = {
                    "annotation_status": "skipped",
                    "annotator": annotator
                }
                save_annotation(idx, updates)
                st.rerun()

# ── Analytics Dashboard ────────────────────────────────────────────────────────
with tab_dashboard:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### Market Sentiment Analytics")
    st.markdown("Insights derived from all annotated passages in the dataset.")
    
    # Filter only 'done' passages
    done_df = df[df["annotation_status"] == "done"].copy()
    
    if done_df.empty:
        st.info("No annotated passages available yet. Complete some annotations to see the analytics!")
    else:
        # Key Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Annotations", len(done_df))
        positive_pct = len(done_df[done_df["sentiment_label"] == "positive"]) / len(done_df) * 100
        m2.metric("Overall Positivity", f"{positive_pct:.1f}%")
        guidance_pct = len(done_df[done_df["has_guidance"].str.lower() == "true"]) / len(done_df) * 100
        m3.metric("Passages with Guidance", f"{guidance_pct:.1f}%")
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        
        with c1:
            # Overall Sentiment Pie Chart
            sentiment_counts = done_df["sentiment_label"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            color_map = {"positive": "#16a34a", "neutral": "#71717a", "negative": "#dc2626"}
            fig1 = px.pie(sentiment_counts, values="Count", names="Sentiment", 
                          title="Overall Sentiment Distribution",
                          color="Sentiment", color_discrete_map=color_map,
                          hole=0.4)
            fig1.update_layout(margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            # Sentiment by Sector
            sector_sentiment = done_df.groupby(["sector", "sentiment_label"]).size().reset_index(name="Count")
            fig2 = px.bar(sector_sentiment, x="sector", y="Count", color="sentiment_label",
                          title="Sentiment by Sector",
                          color_discrete_map=color_map, barmode="stack")
            fig2.update_layout(xaxis_title="", margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig2, use_container_width=True)
            
        st.markdown("---")
        
        # Forward Guidance Chart
        guidance_df = done_df[done_df["has_guidance"].str.lower() == "true"].copy()
        if not guidance_df.empty:
            g_sector = guidance_df.groupby(["sector", "guidance_type"]).size().reset_index(name="Count")
            g_color_map = {"positive": "#16a34a", "neutral": "#71717a", "negative": "#dc2626", "conditional": "#2563eb"}
            fig3 = px.bar(g_sector, x="sector", y="Count", color="guidance_type",
                          title="Forward Guidance Breakdown by Sector",
                          color_discrete_map=g_color_map, barmode="group")
            fig3.update_layout(xaxis_title="")
            st.plotly_chart(fig3, use_container_width=True)
            
            # Recent Guidance Dataframe
            st.markdown("**Recent Forward-Looking Statements**")
            recent_guidance = guidance_df[["company", "sector", "guidance_type", "guidance_span"]].dropna()
            # Rename for display
            recent_guidance.columns = ["Company", "Sector", "Type", "Statement"]
            st.dataframe(recent_guidance.head(10), use_container_width=True, hide_index=True)
            
    st.markdown('</div>', unsafe_allow_html=True)
