#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(
    page_title="TikTok Viral-Sound Analyzer",
    layout="wide"
)

st.title("TikTok Viral-Sound Analyzer")

# 1) File uploader
uploaded = st.file_uploader(
    "Upload your TikTok export (.xlsx or .csv)",
    type=["xlsx", "csv"]
)
if not uploaded:
    st.info("Please upload an export to get started.")
    st.stop()

# 2) Load raw
@st.cache_data
def load_raw(f):
    if f.name.lower().endswith(".csv"):
        return pd.read_csv(f, header=None)
    else:
        return pd.read_excel(f, header=None)
raw = load_raw(uploaded)

# 3) Clean & unify US/Brazil formats
def find_header_row(df):
    for i, row in df.iterrows():
        txt = row.astype(str).str.lower()
        if txt.str.contains("song name", na=False).any() \
        or txt.str.contains("clip id", na=False).any():
            return i
    raise ValueError("No header row with 'Song Name' or 'Clip id'")

def clean_df(df_raw):
    hdr = find_header_row(df_raw)
    df_raw.columns = df_raw.iloc[hdr].astype(str)
    df = df_raw.iloc[hdr+1:].reset_index(drop=True)

    # drop optional “unit” row
    if (df.shape[0] > 0
        and df.iloc[0].astype(str)
               .str.contains("unit", case=False, na=False)
               .any()):
        df = df.iloc[1:].reset_index(drop=True)

    df.columns = df.columns.str.strip()

    # numeric columns
    num_map = {
        'VV This Week':           'views_this_week',
        'VV Last Week':           'views_last_week',
        'Shares This Week':       'shares_this_week',
        'favorite_cnt_this_week': 'favorites_this_week',
        'Delta VV':               'delta_views',
        'Creations This Week':    'creations_this_week',
        'Creations Last Week':    'creations_last_week',
        'Delta Creations':        'delta_creations'
    }
    df = df.rename(columns=num_map)
    for c in ['views_this_week','views_last_week',
              'shares_this_week','favorites_this_week',
              'delta_views','delta_creations','creations_this_week']:
        df[c] = pd.to_numeric(df.get(c, 0), errors='coerce').fillna(0)

    df['views_growth_rate'] = (
        (df['views_this_week'] - df['views_last_week'])
        / df['views_last_week'].replace(0, np.nan)
    ).fillna(0)

    # metadata columns
    meta_map = {
        'Clip id':            'Clip ID',
        'clip id':            'Clip ID',
        'Clip ID':            'Clip ID',
        'Clip name':          'Song Name',
        'Song Name':          'Song Name',
        'Meta Song Name':     'Song Name',
        'meta_song_name':     'Song Name',
        'Meta Artist':        'Artist',
        'Artist':             'Artist',
        'meta_artist':        'Artist',
        'Label':              'Label',
        'meta_song_isrc':     'ISRC',
        'ISRC':               'ISRC',
    }
    df = df.rename(columns=meta_map)

    # ensure existence
    for txt in ['Clip ID','Song Name','Artist','Label','ISRC']:
        if txt not in df.columns:
            df[txt] = ""
    return df

df = clean_df(raw)

@st.cache_data
def rank_df(df, weights):
    w = np.array(weights, dtype=float)
    w = w / w.sum() if w.sum()>0 else np.ones_like(w)/len(w)
    df2 = df.copy()
    df2['share_rate'] = np.where(df2.views_this_week>0,
                                 df2.shares_this_week/df2.views_this_week*100, 0)
    df2['fav_rate']   = np.where(df2.views_this_week>0,
                                 df2.favorites_this_week/df2.views_this_week*100, 0)
    dims = ['views_this_week','views_growth_rate','share_rate','fav_rate','creations_this_week']
    ranks = {c: df2[c].rank(pct=True) for c in dims}
    df2['engagement_score'] = sum(ranks[c]*w[i] for i,c in enumerate(dims))
    return df2.sort_values('engagement_score', ascending=False).reset_index(drop=True)

# 4) Sidebar sliders
st.sidebar.header("Weight Sliders")
w0 = st.sidebar.slider("Total Views", 0.0, 10.0, 5.0)
w1 = st.sidebar.slider("WoW Growth",  0.0, 10.0, 2.0)
w2 = st.sidebar.slider("Share Rate",  0.0, 10.0, 3.0)
w3 = st.sidebar.slider("Fav Rate",    0.0, 10.0, 1.0)
w4 = st.sidebar.slider("Creations",   0.0, 10.0, 1.0)

# 5) Rank
with st.spinner("Ranking..."):
    ranked = rank_df(df, [w0,w1,w2,w3,w4])

# 6) Filter
filter_mode = st.radio("Filter label:", ["All","UGC","DistroKid"], horizontal=True)
if filter_mode != "All":
    ranked = ranked[ranked['Label']==filter_mode]

# 7) Insert ASCII “Spotify” link column
ranked.insert(
    0,
    "Spotify",
    ranked["ISRC"].astype(str).apply(
        lambda isrc: f"https://open.spotify.com/search/isrc:{isrc}"
    )
)

st.write(f"## Top {min(30, len(ranked))} ({filter_mode})")
st.dataframe(
    ranked.head(30)[[
        "Spotify","Clip ID","Song Name","Artist","Label","ISRC",
        "views_this_week","views_growth_rate","share_rate",
        "fav_rate","creations_this_week","engagement_score"
    ]],
    use_container_width=True,
    height=600,
    column_config={
        "Spotify": st.column_config.LinkColumn(
            "Open in Spotify",
            help="Click to search this ISRC on Spotify"
        )
    }
)

# 8) Export button
buf = io.BytesIO()
to_export = ranked if filter_mode=="All" else ranked[ranked['Label']==filter_mode]
to_export.to_excel(buf, index=False)
buf.seek(0)
st.download_button(
    "Download Excel",
    buf,
    file_name="ranked_sounds.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
