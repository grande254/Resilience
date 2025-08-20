import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
# Set page config
st.set_page_config(
    page_title="ResiliSense",
    page_icon="🌍",
    layout="wide"
)

# Add a clean title header
st.title("🌍 ResiliSense")
st.caption("Interactive Policy Dashboard for Vulnerability & Nutrition Insights")
# st.set_page_config(page_title="Community Vulnerability & Nutrition Intelligence", layout="wide")

# ==============================
# Helpers
# ==============================
@st.cache_data
def load_csv_any(path: Path):
    """Auto-detect delimiter (comma/tab) and load CSV; return DataFrame."""
    try:
        return pd.read_csv(path, engine="python", sep=None)
    except Exception:
        return pd.read_csv(path, sep="\t")

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[()/%-]", "", regex=True)
        .str.replace("__+", "_", regex=True)
    )
    return df

def ensure_region(df: pd.DataFrame) -> pd.DataFrame:
    if "Region" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Region"})
    df["Region"] = (
        df["Region"].astype(str).str.strip().str.replace(r"\s+"," ", regex=True).str.title()
    )
    # numeric coercion
    for c in df.columns:
        if c != "Region":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _norm(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def find_col(df: pd.DataFrame, *candidates, must_include_all=None):
    """Fuzzy-find a column by exact, normalized, or keyword match."""
    cols = list(df.columns)
    norm_map = {_norm(c): c for c in cols}
    # 1) exact
    for cand in candidates:
        if cand in df.columns:
            return cand
    # 2) normalized equality
    for cand in candidates:
        n = _norm(cand)
        if n in norm_map:
            return norm_map[n]
    # 3) keyword all-must-match
    if must_include_all:
        must = [_norm(w) for w in must_include_all]
        for c in cols:
            t = _norm(c)
            if all(w in t for w in must):
                return c
    return None

def kpi_card(title, value, fmt=None):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        st.metric(title, "—")
    else:
        st.metric(title, fmt.format(value) if fmt else value)

def make_radar(row, excluded_cols):
    nutrients = [c for c in row.index if c not in excluded_cols and pd.api.types.is_number(row[c])]
    if not nutrients:
        return go.Figure()
    values = [row[c] for c in nutrients]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=nutrients + [nutrients[0]], fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=False)
    return fig

def generate_policy_pdf(region_name, nut_row, vuln_row, colmap):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    x, y = 2*cm, height - 2*cm
    c.setFont("Helvetica-Bold", 14); c.drawString(x, y, f"Policy Brief: {region_name}"); y -= 1.2*cm
    c.setFont("Helvetica", 11)
    if vuln_row is not None:
        comp = float(vuln_row.get(colmap["comp"], np.nan)) if colmap["comp"] else np.nan
        drought = float(vuln_row.get(colmap["drought"], np.nan)) if colmap["drought"] else np.nan
        health = float(vuln_row.get(colmap["health"], np.nan)) if colmap["health"] else np.nan
        food = float(vuln_row.get(colmap["food"], np.nan)) if colmap["food"] else np.nan
        climate = float(vuln_row.get(colmap["climate"], np.nan)) if colmap["climate"] else np.nan
        c.drawString(x, y, f"Vulnerability — Composite: {comp:.2f}, Drought: {drought:.2f}, Health: {health:.2f}, Food: {food:.2f}, Climate: {climate:.2f}")
        y -= 0.8*cm
    if nut_row is not None:
        excl = {"Region"}
        s = nut_row.drop(labels=[c for c in nut_row.index if c in excl]).apply(pd.to_numeric, errors="coerce")
        gaps = s.sort_values().head(5).dropna()
        c.drawString(x, y, "Top Nutrient Gaps (lowest adequacy %):"); y -= 0.7*cm
        for k, v in gaps.items():
            c.drawString(x+0.5*cm, y, f"- {k}: {v:.1f}%"); y -= 0.6*cm
    y -= 0.3*cm
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Suggested Actions:"); y -= 0.8*cm
    c.setFont("Helvetica", 11)
    recs = []
    if vuln_row is not None and colmap["drought"] and float(vuln_row.get(colmap["drought"], 0)) > 0.5:
        recs.append("Scale drought-resilient crops and water harvesting.")
    if nut_row is not None and float(nut_row.get("Proteins", 100)) < 60:
        recs.append("Introduce targeted protein supplementation/fortification programs.")
    if nut_row is not None and float(nut_row.get("Iron", 100)) < 60:
        recs.append("Expand iron-rich food programs and anemia screening.")
    if vuln_row is not None and colmap["health"] and float(vuln_row.get(colmap["health"], 0)) > 0.5:
        recs.append("Strengthen primary health systems and nutrition supply chains.")
    if not recs:
        recs.append("Maintain monitoring; indicators are within acceptable range.")
    # Dedup while preserving order
    seen=set(); dedup=[]
    for r in recs:
        if r not in seen:
            seen.add(r); dedup.append(r)
    for r in dedup[:6]:
        c.drawString(x+0.5*cm, y, f"- {r}"); y -= 0.6*cm
    c.showPage(); c.save(); buffer.seek(0); return buffer

# === POLICY HELPERS ===
def risk_bucket(comp_index: float, avg_nutrient: float):
    """
    Simple rule: High risk if comp_index>=0.5 AND avg_nutrient<60
                 Medium if comp_index>=0.4 OR avg_nutrient<70
                 else Low
    """
    if comp_index is None or avg_nutrient is None:
        return "Unknown", "gray"
    if comp_index >= 0.5 and avg_nutrient < 60:
        return "High", "red"
    if comp_index >= 0.4 or avg_nutrient < 70:
        return "Medium", "orange"
    return "Low", "green"

def plain_english_insights(region, comp_idx, drought_idx, avg_nut, worst_nutrients):
    insights = []
    if comp_idx is not None:
        if comp_idx >= 0.6: insights.append(f"{region} shows high overall vulnerability (≥0.60).")
        elif comp_idx >= 0.4: insights.append(f"{region} shows moderate overall vulnerability.")
        else: insights.append(f"{region} shows low overall vulnerability.")
    if avg_nut is not None:
        if avg_nut < 50: insights.append("Average diet adequacy is poor (<50%).")
        elif avg_nut < 70: insights.append("Average diet adequacy is below target (<70%).")
        else: insights.append("Average diet adequacy is generally adequate (≥70%).")
    if drought_idx is not None and drought_idx >= 0.5:
        insights.append("Drought risk is elevated (≥0.50).")
    if worst_nutrients:
        gaps = ", ".join([f"{k} ({v:.0f}%)" for k, v in worst_nutrients[:3]])
        insights.append(f"Largest nutrient gaps: {gaps}.")
    return insights[:3]  # top 3

def policy_actions(comp_idx, drought_idx, avg_nut, worst_nutrients):
    actions = []
    if drought_idx is not None and drought_idx >= 0.5:
        actions.append("Scale drought-resilient crops and water harvesting.")
    if avg_nut is not None and avg_nut < 60:
        actions.append("Expand targeted food assistance and fortification.")
    if worst_nutrients:
        for k, _v in worst_nutrients[:2]:
            k_lower = k.lower()
            if k_lower.startswith("iron"):
                actions.append("Add iron-rich foods and anemia screening in clinics.")
            if k_lower.startswith(("protein","proteins")):
                actions.append("Add protein supplementation for vulnerable households.")
            if k_lower.startswith(("vitamin_a","vitamina")):
                actions.append("Vitamin A supplementation and diversify diets.")
    if comp_idx is not None and comp_idx >= 0.5:
        actions.append("Prioritize this region for resilience funding and early-warning monitoring.")
    if not actions:
        actions.append("Maintain monitoring; indicators are within acceptable range.")
    seen, deduped = set(), []
    for a in actions:
        if a not in seen:
            seen.add(a); deduped.append(a)
    return deduped[:5]

# ==============================
# Sidebar
# ==============================
st.title("Community Vulnerability & Nutrition Intelligence")
st.caption("Local files only • No uploads")

with st.sidebar:
    st.header("Data folder")
    data_dir = Path(st.text_input("Path", value=str(Path('.').resolve()))).expanduser().resolve()
    st.write(f"Using: {data_dir}")
    st.markdown("---")
    st.header("Hotspot thresholds")
    nut_thresh = st.slider("Min Avg Nutrient Adequacy (%)", 0, 100, 60)
    vuln_thresh = st.slider("Min Composite Vulnerability (0–1)", 0.0, 1.0, 0.4, step=0.01)
    st.markdown("---")
    st.header("Scenario presets")
    policy_mode = st.toggle("Policy Mode (simplified view)", value=True)
    preset = st.radio(
        "Choose a preset",
        ["None", "Mild drought (+10%)", "Severe drought (+20%)"],
        horizontal=True
    )
    drought_delta = 0
    if preset == "Mild drought (+10%)":
        drought_delta = 10
    elif preset == "Severe drought (+20%)":
        drought_delta = 20
    kcal_delta = st.slider("Δ Kcal Adequacy (%)", -20, 20, 0)

# Try common file names in priority order
nut_paths = [(data_dir / "nutrient_clean.csv"),
             (data_dir / "Nutrient adequacy data.csv")]
vuln_paths = [(data_dir / "vulnerability_clean.csv"),
              (data_dir / "composite-vulnerability.csv")]

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return paths[0]  # default to first for error message

nut_path = first_existing(nut_paths)
vuln_path = first_existing(vuln_paths)

# ==============================
# Load data
# ==============================
nut_df = None; vuln_df = None
nut_err = None; vuln_err = None
try:
    nut_df = load_csv_any(nut_path); nut_df = clean_columns(ensure_region(nut_df))
except Exception as e:
    nut_err = str(e)
try:
    vuln_df = load_csv_any(vuln_path); vuln_df = clean_columns(ensure_region(vuln_df))
except Exception as e:
    vuln_err = str(e)

c1, c2 = st.columns(2)
with c1:
    if isinstance(nut_df, pd.DataFrame):
        st.success(f"Loaded {nut_path.name}  •  {len(nut_df)} rows")
    else:
        st.error(f"Failed to load {nut_path} — {nut_err or 'file missing?'}")
with c2:
    if isinstance(vuln_df, pd.DataFrame):
        st.success(f"Loaded {vuln_path.name}  •  {len(vuln_df)} rows")
    else:
        st.error(f"Failed to load {vuln_path} — {vuln_err or 'file missing?'}")

if not isinstance(nut_df, pd.DataFrame) or not isinstance(vuln_df, pd.DataFrame):
    st.stop()

# Resolve vulnerability columns (fuzzy)
comp_col    = find_col(vuln_df, "Composite_Vulnerability_Index", must_include_all=["composite","vulnerability","index"])
drought_col = find_col(vuln_df, "Drought_index", must_include_all=["drought"])
health_col  = find_col(vuln_df, "Health_System_Vulnerability_Index", must_include_all=["health","system","vulnerability","index"])
mnar_col    = find_col(vuln_df, "Mean_Nutrient_Adequacy_Ratio_Index", must_include_all=["mean","nutrient","adequacy","index"])
food_col    = find_col(vuln_df, "Per_Capita_Food_Consumption_Index", must_include_all=["food","consumption","index"])
climate_col = find_col(vuln_df, "Vulnerability_to_Climate_Change_Index", must_include_all=["climate","change","vulnerability","index"])
colmap = {"comp": comp_col, "drought": drought_col, "health": health_col, "mnar": mnar_col, "food": food_col, "climate": climate_col}

# ======================
# GLOBAL REGION SELECTOR
# ======================
regions = sorted(list(set(nut_df["Region"].tolist()) | set(vuln_df["Region"].tolist())))
selected_region = st.selectbox("🌍 Select Region", regions if regions else ["—"], key="main_region_selector")

# ==============================
# Tabs
# ==============================
tabs = st.tabs(["Overview", "Nutrition Profile", "Vulnerability Heat", "Correlation Explorer", "Scenario & Brief", "Compare Regions"])

# ------------------------------
# Overview
# ------------------------------
with tabs[0]:
    st.subheader("Regional snapshot KPIs")

    # Compute region metrics once
    nut_row = nut_df[nut_df["Region"]==selected_region].iloc[0] if selected_region in nut_df["Region"].values else None
    vuln_row = vuln_df[vuln_df["Region"]==selected_region].iloc[0] if selected_region in vuln_df["Region"].values else None
    avg_n = float(nut_row[[c for c in nut_df.columns if c!="Region"]].mean()) if nut_row is not None else None
    comp_idx = float(vuln_row[comp_col]) if (vuln_row is not None and comp_col) else None
    drought_idx = float(vuln_row[drought_col]) if (vuln_row is not None and drought_col) else None

    # Find worst nutrient gaps (lowest adequacy)
    worst = []
    if nut_row is not None:
        ns = nut_row.drop(labels=["Region"]).apply(pd.to_numeric, errors="coerce").sort_values()
        worst = [(k, v) for k, v in ns.items() if pd.notnull(v)][:5]

    # Policy header
    if policy_mode:
        bucket, color = risk_bucket(comp_idx, avg_n)
        st.markdown(f"### Overall Risk: :{color}[**{bucket}**]")
        for tip in plain_english_insights(selected_region, comp_idx, drought_idx, avg_n, worst):
            st.markdown(f"- {tip}")
        st.markdown("---")

    a,b,c,d,e = st.columns(5)
    with a: kpi_card("Composite Vulnerability", comp_idx, "{:.2f}")
    with b: kpi_card("Avg Nutrient Adequacy", avg_n, "{:.1f}%")
    with c: kpi_card("Drought Index", drought_idx, "{:.2f}")
    with d:
        val = float(vuln_row[food_col]) if (vuln_row is not None and food_col) else None
        kpi_card("Food Consumption", val, "{:.2f}")
    with e:
        val = float(vuln_row[health_col]) if (vuln_row is not None and health_col) else None
        kpi_card("Health System Vuln.", val, "{:.2f}")

    st.markdown("### Hotspot Detector (Global)")
    tmp = nut_df.copy()
    tmp["avg_nutrient"] = tmp[[c for c in tmp.columns if c!="Region"]].mean(axis=1)
    if comp_col and comp_col in vuln_df.columns:
        m = tmp[["Region","avg_nutrient"]].merge(vuln_df[["Region", comp_col]], on="Region", how="inner")
        m["Hotspot"] = (m["avg_nutrient"] < nut_thresh) & (m[comp_col] >= vuln_thresh)

        fig = px.scatter(
            m, x="avg_nutrient", y=comp_col, color="Hotspot", hover_name="Region",
            title="Hotspot Quadrant — Focus top-left (low nutrition, high vulnerability)",
            labels={"avg_nutrient":"Avg Nutrient Adequacy (%)", comp_col:"Composite Vulnerability (0–1)"}
        )
        fig.update_traces(marker=dict(size=14, line=dict(width=1, color="white")))
        fig.add_vline(x=nut_thresh, line_dash="dash", annotation_text=f"Nutrition target: {nut_thresh}%")
        fig.add_hline(y=vuln_thresh, line_dash="dash", annotation_text=f"Vulnerability alert: {vuln_thresh:.2f}")
        fig.add_shape(type="rect", x0=0, x1=nut_thresh, y0=vuln_thresh, y1=1.0,
                      fillcolor="red", opacity=0.08, line_width=0, layer="below")
        fig.update_layout(margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Composite vulnerability column not found in vulnerability CSV.")

    if policy_mode:
        st.markdown("### Recommended Actions")
        for act in policy_actions(comp_idx, drought_idx, avg_n, worst):
            st.markdown(f"- {act}")

        with st.expander("Why this matters / Glossary"):
            st.markdown("""
- **Composite Vulnerability**: Overall exposure and limited capacity to cope (0 = low, 1 = high).
- **Avg Nutrient Adequacy**: How close diets are to recommended intakes (100% = meets needs).
- **Hotspot**: Region below the nutrition target **and** above the vulnerability alert line.
- **Drought Index**: Higher means more drought pressure.
            """)

# ------------------------------
# Nutrition Profile
# ------------------------------
with tabs[1]:
    st.subheader(f"{selected_region}: Nutrition fingerprint (radar)")
    if selected_region in nut_df["Region"].values:
        row = nut_df[nut_df["Region"]==selected_region].iloc[0]
        exclude = {"Region"}  # keep only nutrients
        st.plotly_chart(make_radar(row, exclude), use_container_width=True)
        st.markdown("#### Raw Nutrient Values"); st.dataframe(nut_df[nut_df["Region"]==selected_region], use_container_width=True)
    else:
        st.info("Region not found in nutrient dataset.")

# ------------------------------
# Vulnerability Heat
# ------------------------------
with tabs[2]:
    st.subheader("Vulnerability heatmap")
    idx_candidates = [c for c in [comp_col, drought_col, health_col, mnar_col, food_col, climate_col] if c]
    if idx_candidates:
        hm = vuln_df.melt(id_vars=["Region"], value_vars=idx_candidates, var_name="Index", value_name="Value")
        fig = px.density_heatmap(hm, x="Index", y="Region", z="Value", nbinsx=len(idx_candidates), nbinsy=len(vuln_df), histfunc="avg",
                                 color_continuous_scale="RdYlGn_r", title="Vulnerability Heatmap (0–1)")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Raw Vulnerability Values"); st.dataframe(vuln_df, use_container_width=True)
    else:
        st.info("No recognizable vulnerability index columns found.")

# ------------------------------
# Correlation Explorer
# ------------------------------
# with tabs[3]:
#     st.subheader("Correlation explorer")
#     join = nut_df.merge(vuln_df, on="Region", how="inner")
#     nut_cols = [c for c in nut_df.columns if c != "Region"]
#     idx_cols = [c for c in vuln_df.columns if c != "Region"]
#     if len(join) and nut_cols and idx_cols:
#         cA, cB = st.columns(2)
#         with cA:
#             x_col = st.selectbox("Nutrient (X)", nut_cols, index=nut_cols.index("Proteins") if "Proteins" in nut_cols else 0)
#         with cB:
#             y_default = drought_col if drought_col in idx_cols else idx_cols[0]
#             y_col = st.selectbox("Vulnerability Index (Y)", idx_cols, index=idx_cols.index(y_default))
#         fig = px.scatter(join, x=x_col, y=y_col, color="Region", trendline="ols",
#                          title=f"Correlation: {x_col} vs {y_col}")
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("Join produced no rows or missing columns.")
with tabs[3]:
    st.subheader("Correlation Explorer (Ranked Bar Chart)")

    # Join datasets
    join = nut_df.merge(vuln_df, on="Region", how="inner")
    nut_cols = [c for c in nut_df.columns if c != "Region"]
    vuln_cols = [c for c in vuln_df.columns if c != "Region"]

    if len(join) and nut_cols and vuln_cols:
        # Choose analysis mode
        mode = st.radio(
            "What do you want to explain?",
            ["Vulnerability explained by Nutrients", "Avg Nutrient Adequacy explained by Vulnerability indices"],
            horizontal=True
        )

        if mode == "Vulnerability explained by Nutrients":
            # Target = a vulnerability index; Features = nutrient columns
            default_target = comp_col if comp_col in vuln_cols else vuln_cols[0]
            target = st.selectbox("Target vulnerability index", options=vuln_cols, index=vuln_cols.index(default_target))
            X = join[nut_cols].apply(pd.to_numeric, errors="coerce")
            y = pd.to_numeric(join[target], errors="coerce")

            corr = X.corrwith(y)
            title = f"Correlation of Nutrients with {target}"

        else:
            # Target = avg nutrient; Features = vulnerability indices
            join["avg_nutrient"] = join[nut_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            target = "avg_nutrient"
            X = join[vuln_cols].apply(pd.to_numeric, errors="coerce")
            y = join[target]

            corr = X.corrwith(y)
            title = "Correlation of Vulnerability Indices with Avg Nutrient Adequacy"

        # Build ranked dataframe
        dfc = corr.dropna().to_frame("Correlation").reset_index()
        dfc = dfc.rename(columns={"index": "Indicator"})
        dfc["sign"] = np.where(dfc["Correlation"] >= 0, "Positive", "Negative")
        dfc = dfc.iloc[dfc["Correlation"].abs().sort_values(ascending=False).index]

        # Let user choose top-N bars
        top_n = st.slider("Show top N indicators", 3, max(3, len(dfc)), min(10, len(dfc)))
        dfc = dfc.head(top_n)

        # Bar chart (–1 to 1)
        fig = px.bar(
            dfc,
            x="Indicator",
            y="Correlation",
            color="sign",
            title=title,
            labels={"Correlation": "Pearson correlation (r)"},
        )
        fig.update_layout(yaxis=dict(range=[-1, 1]), showlegend=False, xaxis_tickangle=-30, margin=dict(l=10, r=10, t=60, b=10))
        fig.add_hline(y=0, line_dash="dash")

        st.plotly_chart(fig, use_container_width=True)

        st.caption("Bars closer to ±1 indicate stronger relationships. Positive = move together; Negative = move opposite.")

        # Optional: show the ranked table underneath
        with st.expander("View ranked correlation table"):
            st.dataframe(dfc, use_container_width=True)

    else:
        st.info("Provide both CSVs to explore correlations.")

# ------------------------------
# Scenario & Brief
# ------------------------------
with tabs[4]:
    st.subheader("Scenario simulation")
    sim_v = vuln_df.copy(); sim_n = nut_df.copy()
    # apply deltas if columns exist
    if drought_col and drought_col in sim_v.columns:
        sim_v[drought_col] = (sim_v[drought_col] * (1 + drought_delta/100.0)).clip(0,1)
    if "Kcal" in sim_n.columns:
        sim_n["Kcal"] = (sim_n["Kcal"] * (1 + kcal_delta/100.0)).clip(0,100)

    tmp = sim_n.copy(); tmp["avg_nutrient"] = tmp[[c for c in tmp.columns if c!="Region"]].mean(axis=1)
    st.markdown("#### Simulated hotspots (quadrant)")
    title_suffix = "" if (drought_delta == 0 and kcal_delta == 0) else f" — Preset: drought {drought_delta:+d}% / kcal {kcal_delta:+d}%"
    if comp_col and comp_col in sim_v.columns:
        m = tmp[["Region","avg_nutrient"]].merge(sim_v[["Region", comp_col]], on="Region", how="inner")
        m["Hotspot (Sim)"] = (m["avg_nutrient"] < nut_thresh) & (m[comp_col] >= vuln_thresh)
        fig2 = px.scatter(m, x="avg_nutrient", y=comp_col, color="Hotspot (Sim)", hover_name="Region",
                          title="Simulated Hotspot Quadrant" + title_suffix,
                          labels={"avg_nutrient":"Avg Nutrient Adequacy (%)", comp_col:"Composite Vulnerability (0–1)"})
        fig2.add_vline(x=nut_thresh, line_dash="dash", annotation_text=f"Nutrition: {nut_thresh}%")
        fig2.add_hline(y=vuln_thresh, line_dash="dash", annotation_text=f"Vulnerability: {vuln_thresh:.2f}")
        fig2.add_shape(type="rect", x0=0, x1=nut_thresh, y0=vuln_thresh, y1=1.0, fillcolor="red", opacity=0.08, line_width=0, layer="below")
        fig2.update_traces(marker=dict(size=12, line=dict(width=1, color="white"))); fig2.update_layout(margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Composite vulnerability column not found; simulation chart unavailable.")

    st.markdown("---")
    st.subheader("Download policy brief (PDF)")
    if selected_region and (selected_region in nut_df["Region"].values or selected_region in vuln_df["Region"].values):
        nut_row = nut_df[nut_df["Region"]==selected_region].iloc[0] if selected_region in nut_df["Region"].values else None
        vuln_row = vuln_df[vuln_df["Region"]==selected_region].iloc[0] if selected_region in vuln_df["Region"].values else None
        pdf = generate_policy_pdf(selected_region, nut_row, vuln_row, colmap)
        st.download_button("Download Policy Brief", data=pdf, file_name=f"{selected_region}_policy_brief.pdf", mime="application/pdf")
    else:
        st.info("Select a valid region to export a brief.")

# ------------------------------
# Compare Regions
# ------------------------------
with tabs[5]:
    st.subheader("Compare two regions (side-by-side)")
    r1, r2 = st.columns(2)
    with r1:
        left = st.selectbox("Left region", regions, index=0, key="left_region")
    with r2:
        right = st.selectbox("Right region", regions, index=1 if len(regions)>1 else 0, key="right_region")

    def region_summary(r):
        nr = nut_df[nut_df["Region"]==r].iloc[0] if r in nut_df["Region"].values else None
        vr = vuln_df[vuln_df["Region"]==r].iloc[0] if r in vuln_df["Region"].values else None
        avg = float(nr[[c for c in nut_df.columns if c!="Region"]].mean()) if nr is not None else None
        comp = float(vr[comp_col]) if (vr is not None and comp_col) else None
        drought = float(vr[drought_col]) if (vr is not None and drought_col) else None
        worst = []
        if nr is not None:
            ns = nr.drop(labels=["Region"]).apply(pd.to_numeric, errors="coerce").sort_values()
            worst = [(k, v) for k, v in ns.items() if pd.notnull(v)][:3]
        bucket, color = risk_bucket(comp, avg)
        return avg, comp, drought, worst, bucket, color

    avgL, compL, drL, worstL, bL, cL = region_summary(left)
    avgR, compR, drR, worstR, bR, cR = region_summary(right)

    cA, cB = st.columns(2)
    with cA:
        st.markdown(f"### {left} — :{cL}[{bL} risk]")
        st.metric("Composite Vulnerability", f"{compL:.2f}" if compL is not None else "—")
        st.metric("Avg Nutrient Adequacy", f"{avgL:.1f}%" if avgL is not None else "—")
        if worstL: st.write("Top gaps:", ", ".join([f"{k} ({v:.0f}%)" for k,v in worstL]))
    with cB:
        st.markdown(f"### {right} — :{cR}[{bR} risk]")
        st.metric("Composite Vulnerability", f"{compR:.2f}" if compR is not None else "—")
        st.metric("Avg Nutrient Adequacy", f"{avgR:.1f}%" if avgR is not None else "—")
        if worstR: st.write("Top gaps:", ", ".join([f"{k} ({v:.0f}%)" for k,v in worstR]))
