#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import io

# ─── Page setup ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TikTok Viral-Sound Analyzer", layout="wide")
st.title("TikTok Viral-Sound Analyzer")

# ─── 1) Upload & load ────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload TikTok export (.xlsx/.csv)", type=["xlsx", "csv"])
if not uploaded:
    st.info("Please upload a TikTok export to proceed.")
    st.stop()

@st.cache_data
def load_raw(f):
    return (
        pd.read_csv(f, header=None)
        if f.name.lower().endswith(".csv")
        else pd.read_excel(f, header=None)
    )

raw = load_raw(uploaded)

# ─── 2) Normalize & clean ────────────────────────────────────────────────────
def clean_df(df_raw):
    # locate header row (US or Brazil)
    for i, row in df_raw.iterrows():
        txt = row.astype(str).str.lower()
        if txt.str.contains("song name|clip id", regex=True, na=False).any():
            hdr = i
            break
    else:
        raise ValueError("No header row with 'Song Name' or 'Clip id'")

    df = df_raw.iloc[hdr + 1 :].copy().reset_index(drop=True)
    df.columns = df_raw.iloc[hdr].astype(str).str.strip()

    # drop optional "unit" row
    if df.iloc[0].astype(str).str.contains("unit", case=False, na=False).any():
        df = df.iloc[1:].reset_index(drop=True)

    # unify numeric columns
    num_map = {
        "VV This Week": "views_this_week",
        "VV Last Week": "views_last_week",
        "Shares This Week": "shares_this_week",
        "favorite_cnt_this_week": "favorites_this_week",
        "Delta VV": "delta_views",
        "Creations This Week": "creations_this_week",
        "Creations Last Week": "creations_last_week",
        "Delta Creations": "delta_creations",
    }
    df = df.rename(columns=num_map)
    for c in num_map.values():
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)

    # growth rate
    df["views_growth_rate"] = (
        (df.views_this_week - df.views_last_week)
        / df.views_last_week.replace(0, np.nan)
    ).fillna(0)

    # unify metadata columns
    meta_map = {
        **dict.fromkeys(["clip id", "Clip id", "Clip ID"], "Clip ID"),
        **dict.fromkeys(
            [
                "clip name",
                "Clip name",
                "Song Name",
                "Meta Song Name",
                "meta_song_name",
            ],
            "Song Name",
        ),
        **dict.fromkeys(["Artist", "Meta Artist", "meta_artist"], "Artist"),
        "Label": "Label",
        **dict.fromkeys(["meta_song_isrc", "ISRC"], "ISRC"),
    }
    df = df.rename(columns=meta_map)

    # ensure existence
    for col in ["Clip ID", "Song Name", "Artist", "Label", "ISRC"]:
        if col not in df:
            df[col] = ""

    # drop accidental duplicates
    return df.loc[:, ~df.columns.duplicated()]

df = clean_df(raw)

# ─── 3) Ranking function ─────────────────────────────────────────────────────
@st.cache_data
def rank_df(df, weights):
    w = np.array(weights, float)
    w = w / w.sum() if w.sum() else np.ones_like(w) / len(w)
    df2 = df.copy()
    df2["share_rate"] = np.where(
        df2.views_this_week > 0, df2.shares_this_week / df2.views_this_week * 100, 0
    )
    df2["fav_rate"] = np.where(
        df2.views_this_week > 0, df2.favorites_this_week / df2.views_this_week * 100, 0
    )
    dims = [
        "views_this_week",
        "views_growth_rate",
        "share_rate",
        "fav_rate",
        "creations_this_week",
    ]
    ranks = {c: df2[c].rank(pct=True) for c in dims}
    df2["engagement_score"] = sum(ranks[c] * w[i] for i, c in enumerate(dims))
    return df2.sort_values("engagement_score", ascending=False).reset_index(drop=True)

# ─── 4) Controls & ranking ───────────────────────────────────────────────────
st.sidebar.header("Weight Sliders")
w = [st.sidebar.slider(label, 0.0, 10.0, default) for label, default in 
     [("Total Views",5.0),("WoW Growth",2.0),("Share Rate",3.0),("Fav Rate",1.0),("Creations",1.0)]]

with st.spinner("Ranking…"):
    ranked = rank_df(df, w)

# ─── 5) Filter & add Spotify link ─────────────────────────────────────────────
mode = st.radio("Filter label:", ("All", "UGC", "DistroKid"), horizontal=True)
if mode != "All":
    ranked = ranked[ranked.Label == mode]

ranked.insert(0, "🔗", ranked.ISRC.apply(
    lambda isrc: f"https://open.spotify.com/search/isrc:{isrc}"
))
ranked = ranked.loc[:, ~ranked.columns.duplicated()]

# ─── 6) Display top-30 ────────────────────────────────────────────────────────
disp = ranked.head(30)[
    ["🔗", "Clip ID", "Song Name", "Artist", "Label", "ISRC"]
    + [c for c in ranked.columns if c not in {"🔗","Clip ID","Song Name","Artist","Label","ISRC"}]
]

st.write(f"## Top {len(disp)} ({mode})")
st.dataframe(
    disp,
    use_container_width=True,
    height=600,
    column_config={"🔗": st.column_config.LinkColumn("Open in Spotify")},
)

# ─── 7) Download ──────────────────────────────────────────────────────────────
buf = io.BytesIO()
disp_to_save = ranked if mode == "All" else ranked[ranked.Label == mode]
disp_to_save.to_excel(buf, index=False)
buf.seek(0)
st.download_button("Download Excel", buf, file_name="ranked_sounds.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
