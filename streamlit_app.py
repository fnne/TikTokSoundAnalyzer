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
    type=["xlsx","csv"]
)
if not uploaded:
    st.info("Please upload an export to get started.")
    st.stop()

# 2) Load raw
@st.cache_data
def load_raw(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file, header=None)
    else:
        return pd.read_excel(file, header=None)
raw = load_raw(uploaded)

# 3) Clean & compute (reuse your functions)
def find_header_row(df):
    for i,row in df.iterrows():
        if row.astype(str).str.contains('song name', case=False, na=False).any():
            return i
    raise ValueError("No header with 'Song Name'")
def clean_df(df_raw):
    hdr = find_header_row(df_raw)
    df_raw.columns = df_raw.iloc[hdr]
    df = df_raw.iloc[hdr+1:].reset_index(drop=True)
    # … same cleaning you have …
    df.columns = df.columns.str.strip()
    rename_map = {
      'VV This Week':'views_this_week','VV Last Week':'views_last_week',
      'Shares This Week':'shares_this_week','favorite_cnt_this_week':'favorites_this_week',
      'Delta VV':'delta_views','Creations This Week':'creations_this_week',
      'Creations Last Week':'creations_last_week','Delta Creations':'delta_creations'
    }
    df = df.rename(columns=rename_map)
    for c in ['views_this_week','views_last_week','shares_this_week','favorites_this_week','delta_views','delta_creations']:
        df[c] = pd.to_numeric(df.get(c,0), errors='coerce').fillna(0)
    df['views_growth_rate'] = (
      (df.views_this_week - df.views_last_week)
      / df.views_last_week.replace(0,np.nan)
    ).fillna(0)
    return df

@st.cache_data
def rank_df(df, weights):
    w = np.array(weights, dtype=float)
    w = w/w.sum() if w.sum()>0 else np.ones_like(w)/len(w)
    df2 = df.copy()
    df2['share_rate'] = np.where(df2.views_this_week>0,
                                 df2.shares_this_week/df2.views_this_week*100, 0)
    df2['fav_rate']   = np.where(df2.views_this_week>0,
                                 df2.favorites_this_week/df2.views_this_week*100, 0)
    cols = ['views_this_week','views_growth_rate','share_rate','fav_rate','creations_this_week']
    ranks = {c: df2[c].rank(pct=True) for c in cols}
    df2['engagement_score'] = sum(ranks[c]*w[i] for i,c in enumerate(cols))
    return df2.sort_values('engagement_score', ascending=False).reset_index(drop=True)

df = clean_df(raw)

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

# 6) Show top-30, with filter buttons
filter_mode = st.radio("Filter label:", ["All","UGC","DistroKid"], horizontal=True)
if filter_mode != "All":
    ranked = ranked[ranked['Label']==filter_mode]

# ── 1) Add a ‘Spotify’ column upfront with a clickable link ──
def make_spotify_link(isrc: str) -> str:
    url = f"https://open.spotify.com/search/isrc:{isrc}"
    # Return an actual HTML link
    return f'<a href="{url}" target="_blank">🔗</a>'

# 6b) Build a “Spotify” column of _plain_ URLs (no HTML).
# after you’ve built `ranked` and done your filtering…

# build a plain-URL column in front:
ranked.insert(
    0,
    "Spotify",
    ranked["meta_song_isrc"].apply(
        lambda isrc: f"https://open.spotify.com/search/isrc:{isrc}"
    ),
)

# now show the top-30 with a real URL column
st.write(f"## Top {min(30, len(ranked))} ({filter_mode})")
st.dataframe(
    ranked
    .head(30)[[
        "Spotify",
        "Song Name",
        "Artist",
        "Label",
        "meta_song_isrc",
        "views_this_week",
        "views_growth_rate",
        "share_rate",
        "fav_rate",
        "creations_this_week",
        "engagement_score",
    ]],
    use_container_width=True,
    height=600,
    column_config={
        "Spotify": st.column_config.LinkColumn(
            "Search on Spotify",
            help="Click to open in Spotify"
        )
    }
)

# 7) Export button
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
