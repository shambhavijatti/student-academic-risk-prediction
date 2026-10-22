"""Student Academic Risk Predictor - Streamlit UI.
Run:  streamlit run app.py
"""
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from risk_core import FEATURES, prepare, train_all, predict, explain

CSV_NAME = "Students Performance Dataset.csv"
RED, GREEN, BLUE = "#e5484d", "#2f9e6e", "#3b6fd4"

st.set_page_config(page_title="Academic Risk Predictor", page_icon="🎓", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
div[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: 14px 16px; border-radius: 12px;}
.badge {display:inline-block; padding: 6px 16px; border-radius: 999px; font-weight: 600; color: white;}
</style>""", unsafe_allow_html=True)


# ---------- data + model loading ----------
@st.cache_data(show_spinner=False)
def load_csv(source) -> pd.DataFrame:
    return prepare(pd.read_csv(source))


@st.cache_resource(show_spinner="Training models…")
def get_artifacts(_df: pd.DataFrame, key: int):
    return train_all(_df)


st.sidebar.title("🎓 Risk Predictor")
upload = st.sidebar.file_uploader("Dataset (CSV)", type="csv", help="Defaults to the CSV next to app.py")
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_NAME)
if upload is not None:
    df = load_csv(upload)
elif os.path.exists(here):
    df = load_csv(here)
else:
    st.title("🎓 Student Academic Risk Predictor")
    st.info(f"Upload **{CSV_NAME}** in the sidebar (or place it next to app.py) to begin.")
    st.stop()

art = get_artifacts(df, len(df))
model_names = list(art["models"])
choice = st.sidebar.selectbox("Model", model_names, index=model_names.index(art["best"]),
                              help="Best model (highest F1) is preselected.")
model = art["models"][choice]
st.sidebar.caption(f"{len(df):,} students · {art['n_train']:,} train / {art['n_test']:,} test")


def risk_band(p: float):
    if p < 0.30: return "Low risk", GREEN
    if p < 0.60: return "Moderate risk", "#e0a020"
    return "High risk", RED


def gauge(p: float) -> go.Figure:
    _, color = risk_band(p)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "valueformat": ".1f"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 30], "color": "rgba(47,158,110,.15)"},
                         {"range": [30, 60], "color": "rgba(224,160,32,.15)"},
                         {"range": [60, 100], "color": "rgba(229,72,77,.15)"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=0))
    return fig


def show_result(values: dict):
    p = predict(model, values)
    label, color = risk_band(p)
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.plotly_chart(gauge(p), use_container_width=True)
        verdict = "AT RISK" if p >= 0.5 else "NOT AT RISK"
        st.markdown(f'<span class="badge" style="background:{color}">{verdict} · {label}</span>',
                    unsafe_allow_html=True)
        st.caption(f"Probability At Risk **{p*100:.1f}%** · Not At Risk **{(1-p)*100:.1f}%**")
    with c2:
        contrib = explain(model, values, art["means"])
        st.markdown("**Why this prediction?**")
        st.caption("Change in risk probability vs. an average student on that feature. "
                   "Red pushes toward At Risk, green pushes away.")
        d = contrib.reset_index(); d.columns = ["Feature", "Impact"]; d["Impact"] *= 100
        fig = px.bar(d, x="Impact", y="Feature", orientation="h",
                     color=d["Impact"] > 0, color_discrete_map={True: RED, False: GREEN})
        fig.update_layout(height=300, showlegend=False, margin=dict(l=0, r=10, t=10, b=0),
                          xaxis_title="Impact on risk (percentage points)", yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    drivers = contrib[contrib > 0.005].sort_values(ascending=False).head(3)
    if len(drivers):
        st.markdown("**Biggest risk drivers:** " + ", ".join(f"`{f}`" for f in drivers.index)
                    + " — improving these would reduce this student's risk the most.")
    else:
        st.success("No feature is meaningfully increasing this student's risk.")


# ---------- pages ----------
st.title("Student Academic Risk Predictor")
tab_pred, tab_lookup, tab_perf, tab_data = st.tabs(
    ["🔮 Predict", "🔎 Student lookup", "📊 Model performance", "🧭 Data insights"])

with tab_pred:
    st.subheader("Enter a student's academic profile")
    st.caption("Sliders start at the dataset average. Adjust them to run what-if scenarios.")
    cols = st.columns(2)
    values = {}
    for i, f in enumerate(FEATURES):
        lo, hi, mean = float(df[f].min()), float(df[f].max()), float(df[f].mean())
        with cols[i % 2]:
            values[f] = st.slider(f, lo, hi, round(mean, 1), step=0.5 if hi > 40 else 0.1)
    st.divider()
    show_result(values)

with tab_lookup:
    st.subheader("Look up an existing student")
    sid = st.selectbox("Student ID", df["Student_ID"].astype(str).tolist(), index=None,
                       placeholder="Type or pick an ID…")
    if sid:
        row = df[df["Student_ID"].astype(str) == sid].iloc[0]
        name = f"{row.get('First_Name', '')} {row.get('Last_Name', '')}".strip()
        meta = [str(row[c]) for c in ("Department", "Gender") if c in row.index]
        st.markdown(f"**{name or sid}**  ·  " + "  ·  ".join(meta) + f"  ·  Actual grade: **{row['Grade']}**")
        m = st.columns(4)
        for i, f in enumerate(FEATURES):
            m[i % 4].metric(f, f"{row[f]:.1f}", f"{row[f] - df[f].mean():+.1f} vs avg")
        st.divider()
        show_result({f: float(row[f]) for f in FEATURES})

with tab_perf:
    st.subheader("Model comparison")
    res = art["results"]
    st.dataframe(res.style.format({c: "{:.3f}" for c in res.columns if c != "Model"})
                 .highlight_max(subset=[c for c in res.columns if c != "Model"], color="#cfe8da"),
                 use_container_width=True, hide_index=True)
    st.caption(f"Best model by F1: **{art['best']}**. Showing details for **{choice}**.")
    a, b = st.columns(2)
    with a:
        cm = art["cm"][choice]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Not At Risk", "At Risk"], y=["Not At Risk", "At Risk"],
                        labels=dict(x="Predicted", y="Actual"), title=f"Confusion matrix · {choice}")
        fig.update_layout(coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = go.Figure()
        for n, (fpr, tpr) in art["roc"].items():
            fig.add_scatter(x=fpr, y=tpr, name=n, line=dict(width=4 if n == choice else 1.5))
        fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), name="Chance")
        fig.update_layout(title="ROC curves", xaxis_title="False positive rate",
                          yaxis_title="True positive rate", height=380)
        st.plotly_chart(fig, use_container_width=True)

with tab_data:
    st.subheader("Explore the dataset")
    k = st.columns(4)
    k[0].metric("Students", f"{len(df):,}")
    k[1].metric("At Risk", f"{df['Academic_Risk_Num'].mean()*100:.1f}%")
    k[2].metric("Avg attendance", f"{df['Attendance (%)'].mean():.1f}%")
    k[3].metric("Avg study hrs/wk", f"{df['Study_Hours_per_Week'].mean():.1f}")
    a, b = st.columns(2)
    with a:
        order = sorted(df["Grade"].dropna().unique())
        fig = px.histogram(df, x="Grade", color="Academic_Risk", category_orders={"Grade": order},
                           color_discrete_map={"At Risk": RED, "Not At Risk": BLUE}, title="Grades")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        f = st.selectbox("Feature", FEATURES, key="dist_feat")
        fig = px.box(df, x="Academic_Risk", y=f, color="Academic_Risk",
                     color_discrete_map={"At Risk": RED, "Not At Risk": BLUE}, title=f"{f} by risk")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    c, d = st.columns(2)
    with c:
        fig = px.imshow(df[FEATURES].corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, title="Feature correlations")
        st.plotly_chart(fig, use_container_width=True)
    with d:
        if "Department" in df.columns:
            dep = df.groupby("Department")["Academic_Risk_Num"].mean().mul(100).reset_index()
            fig = px.bar(dep, x="Department", y="Academic_Risk_Num", title="At-risk rate by department (%)")
            fig.update_layout(yaxis_title="% At Risk")
            st.plotly_chart(fig, use_container_width=True)
