# streamlit_app.py
import os
import io
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ----------------------------
# CONFIG / TITLE
# ----------------------------
st.set_page_config(page_title="Trading Performance Tracker", layout="wide")
st.markdown(
    """
    <h1 style="margin-bottom:0">TRADING PERFORMANCE TRACKER</h1>
    <p style="color:#5da05d; font-weight:600; margin-top:4px">Live from Google Sheets (Data tab only)</p>
    <hr/>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# DATA LOADERS
# ----------------------------
def read_from_google_sheets(service_account_json: dict, spreadsheet_id: str, worksheet_name: str = "Data") -> pd.DataFrame:
    """Read a worksheet to DataFrame using service account (no public sharing needed)."""
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(service_account_json, scopes=scopes)
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(spreadsheet_id).worksheet(worksheet_name)
    data = ws.get_all_records()
    return pd.DataFrame(data)

def read_from_csv_export(csv_url: str) -> pd.DataFrame:
    """Read a published CSV export link of a Google Sheet tab."""
    return pd.read_csv(csv_url)

@st.cache_data(show_spinner=False)
def load_data(source_mode, spreadsheet_id=None, csv_url=None, worksheet="Data", service_json=None):
    if source_mode == "Service Account":
        df = read_from_google_sheets(service_json, spreadsheet_id, worksheet)
    elif source_mode == "CSV Export URL":
        df = read_from_csv_export(csv_url)
    else:
        uploaded = st.file_uploader("Upload a local Excel/CSV to test", type=["xlsx","csv"])
        if not uploaded:
            st.stop()
        if uploaded.name.lower().endswith(".xlsx"):
            df = pd.read_excel(uploaded, sheet_name=worksheet if worksheet else 0)
        else:
            df = pd.read_csv(uploaded)
    return df

# ----------------------------
# SIDEBAR: DATA SOURCE + FILTERS
# ----------------------------
st.sidebar.header("Data Source")

# DEFAULT TO CSV EXPORT URL (your live Sheet)
source_mode = st.sidebar.selectbox(
    "How should we load the data?",
    ["Service Account", "CSV Export URL", "Local File (debug)"],
    index=1  # <-- default: CSV Export URL
)

service_json = None
spreadsheet_id = None
csv_url = None

if source_mode == "Service Account":
    st.sidebar.markdown(
        """
        1) Put your Google Cloud service account JSON into **.streamlit/secrets.toml** as `gcp_service_account`  
        2) Share the Sheet with that service account email (Viewer)  
        """
    )
    try:
        service_json = st.secrets["gcp_service_account"]
    except Exception:
        st.error("Add your service account to .streamlit/secrets.toml as `gcp_service_account`.")
        st.stop()
    spreadsheet_id = st.sidebar.text_input("Spreadsheet ID", help="From https://docs.google.com/spreadsheets/d/<THIS>/edit")
    worksheet = st.sidebar.text_input("Worksheet name", value="Data")

elif source_mode == "CSV Export URL":
    # Pre-filled with YOUR working CSV export link
    csv_url = st.sidebar.text_input(
        "CSV export link",
        value="https://docs.google.com/spreadsheets/d/10jjhBoNQW_mljKiYlOrrt0z3MxMEe7Nq/export?format=csv&gid=56749682",
        help="Format: https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>"
    )
    worksheet = None

else:
    worksheet = st.sidebar.text_input("Worksheet name (if Excel)", value="Data")

st.sidebar.markdown("---")
st.sidebar.header("Filters")

# ----------------------------
# LOAD + CLEAN DATA
# ----------------------------
with st.spinner("Loading data..."):
    df = load_data(source_mode, spreadsheet_id, csv_url, worksheet, service_json)

# Normalize headers (handle Unnamed columns)
df = df.rename(columns=lambda c: c.strip())
drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
df = df.drop(columns=drop_cols, errors="ignore")

# Expected columns (from your file)
expected_cols = ["Date", "Product", "L/S", "Process", "Grade", "Category"]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}. Make sure the sheet has these exact headers.")
    st.stop()

# Parse date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

# Clean strings
for c in ["Product","L/S","Process","Grade","Category"]:
    df[c] = df[c].astype(str).str.strip()

# Drop fully empty rows
df = df.dropna(how="all")

# Sidebar filters (date range, process, grade, status)
min_date = pd.to_datetime(df["Date"].min())
max_date = pd.to_datetime(df["Date"].max())
date_range = st.sidebar.date_input(
    "Date range",
    (min_date.date() if pd.notnull(min_date) else None,
     max_date.date() if pd.notnull(max_date) else None)
)
if isinstance(date_range, tuple) and len(date_range) == 2 and all(date_range):
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(df["Date"] >= start) & (df["Date"] <= end)]

process_sel = st.sidebar.multiselect("Process (T1-T4)", sorted([x for x in df["Process"].dropna().unique() if x]), default=None)
if process_sel:
    df = df[df["Process"].isin(process_sel)]

grade_sel = st.sidebar.multiselect("Grade (A-D/C)", sorted([x for x in df["Grade"].dropna().unique() if x]), default=None)
if grade_sel:
    df = df[df["Grade"].isin(grade_sel)]

status_sel = st.sidebar.multiselect("Status (Processed/Unprocessed)", sorted([x for x in df["Category"].dropna().unique() if x]), default=None)
if status_sel:
    df = df[df["Category"].isin(status_sel)]

st.sidebar.success(f"Active rows: {len(df):,}")

# ----------------------------
# KPI CARDS
# ----------------------------
total = len(df)
by_status = df["Category"].value_counts().rename_axis("Category").reset_index(name="Count")
proc = int(by_status.loc[by_status["Category"].eq("Processed"), "Count"].sum())
unproc = int(by_status.loc[by_status["Category"].eq("Unprocessed"), "Count"].sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trades", f"{total:,}")
col2.metric("Processed", f"{proc:,}", delta=f"{(proc/total*100 if total else 0):.1f}%")
col3.metric("Unprocessed", f"{unproc:,}", delta=f"{(unproc/total*100 if total else 0):.1f}%")
unique_products = df["Product"].nunique()
col4.metric("Unique Products", f"{unique_products:,}")

st.markdown("---")

# ----------------------------
# 1) Process Wise Numbers (counts + %)
# ----------------------------
proc_counts = (df.groupby("Process", dropna=False)
                 .size()
                 .rename("Total")
                 .reset_index())
proc_counts["Process"] = proc_counts["Process"].fillna("Unknown")
proc_counts = proc_counts.sort_values("Process")
proc_counts["Percent"] = np.where(total > 0, proc_counts["Total"] / total * 100, 0.0)

left, right = st.columns((2,1))
with left:
    st.subheader("Process Wise Numbers")
    fig_bar = px.bar(
        proc_counts,
        x="Process",
        y="Total",
        text=proc_counts["Percent"].map(lambda x: f"{x:.0f}%"),
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

with right:
    fig_donut = go.Figure(
        data=[go.Pie(labels=proc_counts["Process"], values=proc_counts["Total"], hole=0.65)]
    )
    fig_donut.update_layout(title="Process Share", height=380, margin=dict(l=10, r=10, t=40, b=20), showlegend=True)
    st.plotly_chart(fig_donut, use_container_width=True)

# ----------------------------
# 2) Processed vs Unprocessed donut
# ----------------------------
st.subheader("Processed vs Unprocessed")
status_counts = df["Category"].fillna("Unknown").value_counts().reset_index()
status_counts.columns = ["Category", "Count"]
fig_status = go.Figure(data=[go.Pie(labels=status_counts["Category"], values=status_counts["Count"], hole=0.65)])
fig_status.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_status, use_container_width=True)

# ----------------------------
# 3) Process × Grade matrix / stacked bars
# ----------------------------
st.subheader("Process Wise Grades")
pivot_pg = pd.crosstab(df["Process"].fillna("Unknown"), df["Grade"].fillna("Unknown"))
pivot_pg = pivot_pg[sorted(pivot_pg.columns)]
st.dataframe(pivot_pg, use_container_width=True)

fig_pg = px.bar(
    pivot_pg.reset_index().melt(id_vars="Process", var_name="Grade", value_name="Count"),
    x="Process", y="Count", color="Grade", barmode="stack"
)
fig_pg.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig_pg, use_container_width=True)

# ----------------------------
# 4) Product × Status (top N)
# ----------------------------
st.subheader("Product Wise: Processed vs Unprocessed")
top_n = st.slider("Show Top N Products", min_value=5, max_value=50, value=20, step=5)
prod_status = pd.crosstab(df["Product"].fillna("Unknown"), df["Category"].fillna("Unknown"))
prod_status["Total"] = prod_status.sum(axis=1)
prod_top = prod_status.sort_values("Total", ascending=False).head(top_n).drop(columns=["Total"])

fig_prod = px.bar(
    prod_top.reset_index().melt(id_vars="Product", var_name="Status", value_name="Count"),
    y="Product", x="Count", color="Status", orientation="h", barmode="stack"
)
fig_prod.update_layout(height=600, margin=dict(l=10, r=10, t=20, b=20))
st.plotly_chart(fig_prod, use_container_width=True)

# ----------------------------
# 5) Grade Wise Analysis (totals + share)
# ----------------------------
st.subheader("Grade Wise Analysis")
grade_counts = df["Grade"].fillna("Unknown").value_counts().reset_index()
grade_counts.columns = ["Grade", "Total"]
grade_counts["Products %"] = np.where(total > 0, grade_counts["Total"] / total * 100, 0.0)

c1, c2 = st.columns(2)
with c1:
    st.dataframe(grade_counts, use_container_width=True)
with c2:
    fig_grade = px.bar(grade_counts, x="Grade", y="Total", text=grade_counts["Total"])
    fig_grade.update_traces(textposition="outside")
    fig_grade.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_grade, use_container_width=True)

# ----------------------------
# 6) Top Products (overall totals)
# ----------------------------
st.subheader("Top Products")
top_products = df["Product"].value_counts().head(10).reset_index()
top_products.columns = ["Product", "Total Products"]
fig_top = px.bar(top_products, y="Product", x="Total Products", orientation="h", text="Total Products")
fig_top.update_traces(textposition="outside")
fig_top.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig_top, use_container_width=True)

st.markdown("---")
st.caption("Tip: Use the sidebar filters to slice by Process, Grade, Status, and Date range.")
