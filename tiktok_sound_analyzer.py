import os

# ─────────────── License ───────────────
# now loaded from the env var "PYSG_LICENSE"
PySimpleGUI_License = os.getenv("PYSG_LICENSE", "")


# ────────── Then import everything as before ──────────
import webbrowser
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import PySimpleGUI as sg

# -------------- Core logic --------------

def find_header_row(df):
    for i, row in df.iterrows():
        if row.astype(str).str.contains('song name', case=False, na=False).any() \
        or row.astype(str).str.contains('Clip id',   case=False, na=False).any():
            return i
    raise ValueError("Can't find a header row containing 'Song Name' or 'Clip id'")

def clean_df(df_raw):
    # 1) find the true header row
    hdr = find_header_row(df_raw)

    # 2) set the DataFrame columns from that row
    df_raw.columns = df_raw.iloc[hdr].astype(str)
    df = df_raw.iloc[hdr+1:].reset_index(drop=True)

    # 3) drop an optional "unit" row
    if (
        df.shape[0] > 0
        and df.iloc[0].astype(str)
               .str.contains('unit', case=False, na=False)
               .any()
    ):
        df = df.iloc[1:].reset_index(drop=True)

    # 4) trim whitespace
    df.columns = df.columns.str.strip()

    # 5) standardize your numeric columns
    rename_map = {
        'VV This Week':           'views_this_week',
        'VV Last Week':           'views_last_week',
        'Shares This Week':       'shares_this_week',
        'favorite_cnt_this_week': 'favorites_this_week',
        'Delta VV':               'delta_views',
        'Creations This Week':    'creations_this_week',
        'Creations Last Week':    'creations_last_week',
        'Delta Creations':        'delta_creations'
    }
    df = df.rename(columns=rename_map)
    for col in [
        'views_this_week','views_last_week',
        'shares_this_week','favorites_this_week',
        'delta_views','delta_creations','creations_this_week'
    ]:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

    # 6) compute growth
    df['views_growth_rate'] = (
        (df['views_this_week'] - df['views_last_week'])
        / df['views_last_week'].replace(0, np.nan)
    ).fillna(0)

    # 7) map *all* your metadata into a fixed set of column names
    metadata_map = {
        'Clip id':            'Clip ID',
        'clip id':            'Clip ID',
        'Clip name':          'Song Name',
        'Song Name':          'Song Name',
        'Meta Song Name':     'Song Name',
        'meta_song_name':     'Song Name',
        'Meta Artist':        'Artist',
        'Artist':             'Artist',
        'meta_artist':        'Artist',
        'Meta Song ID':       'Song ID',    # if you need it
        'Song ID':            'Song ID',
        'meta_song_isrc':     'ISRC',
        'ISRC':               'ISRC',
        'Label':              'Label',
    }
    df = df.rename(columns=metadata_map)

    # 8) ensure *all* of those exist
    for must_have in ['Clip ID','Song Name','Artist','Label','ISRC']:
        if must_have not in df.columns:
            df[must_have] = ''

        # ——— make sure our downstream columns always exist ———
    # string columns default to empty string, numeric to 0
    for col in ['Song Name','Artist','Label','meta_song_isrc']:
        if col not in df.columns:
            df[col] = ''
    if 'creations_this_week' not in df.columns:
        df['creations_this_week'] = 0

    return df

def rank_df(df, raw_weights, window):
    w = np.array(raw_weights, dtype=float)
    w = w / w.sum() if w.sum()>0 else np.ones_like(w)/len(w)
    # fake progress
    for pct in range(0, 90, 10):
        window.write_event_value('-PROG-', pct)
        time.sleep(0.02)
    df2 = df.copy()
    df2['share_rate'] = np.where(df2.views_this_week>0,
                                 df2.shares_this_week/df2.views_this_week*100, 0)
    df2['fav_rate']   = np.where(df2.views_this_week>0,
                                 df2.favorites_this_week/df2.views_this_week*100, 0)
    ranks_cols = [
        'views_this_week','views_growth_rate',
        'share_rate','fav_rate','creations_this_week'
    ]
    ranks = {c: df2[c].rank(pct=True) for c in ranks_cols}
    df2['engagement_score'] = sum(ranks[c]*w[i] for i,c in enumerate(ranks_cols))
    df2 = df2.sort_values('engagement_score', ascending=False).reset_index(drop=True)
    window.write_event_value('-PROG-', 100)
    return df2

# ------------ GUI layout ------------

if hasattr(sg, 'theme'):
    sg.theme('DarkGrey12')
elif hasattr(sg, 'ChangeLookAndFeel'):
    sg.ChangeLookAndFeel('DarkGrey12')

layout = [
    [sg.Text('Export File:'), sg.Input(key='FILE', readonly=True, expand_x=True),
     sg.Button('Browse', key='-BROWSE-')],
    # five horizontal sliders + live readouts
    [sg.Column([
        [sg.Text('Total views',     size=(14,1)),
         sg.Slider(range=(0,10), default_value=5, orientation='h', size=(40,15),
                   key='w0', enable_events=True), sg.Text('5.00', key='v0')],
        [sg.Text('WoW growth',      size=(14,1)),
         sg.Slider(range=(0,10), default_value=2, orientation='h', size=(40,15),
                   key='w1', enable_events=True), sg.Text('2.00', key='v1')],
        [sg.Text('Share rate (%)',  size=(14,1)),
         sg.Slider(range=(0,10), default_value=3, orientation='h', size=(40,15),
                   key='w2', enable_events=True), sg.Text('3.00', key='v2')],
        [sg.Text('Fav rate (%)',    size=(14,1)),
         sg.Slider(range=(0,10), default_value=1, orientation='h', size=(40,15),
                   key='w3', enable_events=True), sg.Text('1.00', key='v3')],
        [sg.Text('Creations count', size=(14,1)),
         sg.Slider(range=(0,10), default_value=1, orientation='h', size=(40,15),
                   key='w4', enable_events=True), sg.Text('1.00', key='v4')],
    ], expand_x=True)],
    [sg.Button('Process', key='-PROCESS-'),
     sg.Button('Show UGC', key='-FILTER-UGC-',      disabled=True),
     sg.Button('Show DistroKid', key='-FILTER-DK-', disabled=True),
     sg.Button('Save ranked Excel', key='-SAVE-',   disabled=True)],
    [sg.Text('', key='STATUS', size=(60,1))],
    [sg.ProgressBar(100, orientation='h', size=(40,10), key='-PROG-', visible=False)],
    [sg.Table(
        values=[],
        headings=['ID','Title','Artist','Label','ISRC','Views','Growth','Share %','Fav %','Creations','Score'],
        key='-TABLE-', visible=False,
        auto_size_columns=True, num_rows=30,
        enable_events=True,
        expand_x=True, expand_y=True,
    )]
]

sg.set_options(icon=None)

# Make the window resizable and larger by default
window = sg.Window(
    'TikTok Viral-Sound Analyzer',
    layout,
    finalize=True,
    resizable=True,
    size=(1200, 800),
)

df_full = None
current_df = None
preview_df = None

def update_labels(vals):
    for i in range(5):
        window[f'v{i}'].update(f"{vals[f'w{i}']:.2f}")

# -------------- Event loop --------------

while True:
    event, vals = window.read(timeout=100)
    if event in (sg.WIN_CLOSED, None):
        break

    # slider live‐labels
    if event in [f'w{i}' for i in range(5)]:
        update_labels(vals)

    # Browse button offloads to thread
    if event == '-BROWSE-':
        def get_file():
            fn = sg.popup_get_file(
                'Select TikTok export',
                file_types=[('Excel','*.xlsx'),('CSV','*.csv')],
                no_window=True
            )
            window.write_event_value('-FILE-', fn)
        threading.Thread(target=get_file, daemon=True).start()

    if event == '-FILE-' and vals['-FILE-']:
        window['FILE'].update(vals['-FILE-'])

    # Process button
    if event == '-PROCESS-':
        path = vals['FILE']
        if not path:
            window['STATUS'].update('⚠️ Select a file first')
            continue
        window['STATUS'].update('🔄 Loading & ranking…')
        window['-PROG-'].update(0, visible=True)
        def worker():
            try:
                df_raw = pd.read_csv(path, header=None) if path.lower().endswith('.csv') else pd.read_excel(path, header=None)
                clean = clean_df(df_raw)
                ranked = rank_df(clean, [vals[f'w{i}'] for i in range(5)], window)
                window.write_event_value('-DONE-', ranked)
            except Exception as e:
                window.write_event_value('-ERROR-', str(e))
        threading.Thread(target=worker, daemon=True).start()

    # Progress updates
    if event == '-PROG-':
        window['-PROG-'].update(vals['-PROG-'])

    # Done ranking → build & show preview
    if event == '-DONE-':
        df_full = vals['-DONE-']
        current_df = df_full
        top10   = df_full.head(30).copy()

        # 1) Clip ID (truncate last 6 digits)
        top10['ID'] = top10['Clip ID'].astype(str).str[:-6]

        # 2) Metadata
        top10['Title']  = top10['Song Name']
        top10['Artist'] = top10['Artist']
        top10['Label']  = top10['Label']
        top10['ISRC']   = top10['meta_song_isrc']

        # 3) Computed metrics
        top10['Views']     = top10['views_this_week']
        top10['Growth']    = top10['views_growth_rate']
        top10['Share %']   = top10['share_rate']
        top10['Fav %']     = top10['fav_rate']
        top10['Creations'] = top10['creations_this_week']
        top10['Score']     = top10['engagement_score']

        # 4) Slice in heading order
        preview_df = top10[[
            'ID','Title','Artist','Label','ISRC',
            'Views','Growth','Share %','Fav %','Creations','Score'
        ]]

        window['-TABLE-'].update(values=preview_df.values.tolist(), visible=True)
        window['-SAVE-'].update(disabled=False)
        window['-FILTER-UGC-'].update(disabled=False)
        window['-FILTER-DK-'].update(disabled=False)
        current_df = df_full   # ensure Save targets the full set until a filter is clicked
        window['STATUS'].update(f'✅ Ranked {len(df_full)} rows')
        window['-PROG-'].update(visible=False)

    # Error
    if event == '-ERROR-':
        window['STATUS'].update('❌ ' + vals['-ERROR-'])
        window['-PROG-'].update(visible=False)

    # Save to Excel
    if event == '-SAVE-' and current_df is not None:
        src  = Path(vals['FILE'])
        stem = src.stem + '_ranked'
        out  = src.with_name(stem + '.xlsx')
        i = 1
        while out.exists():
            out = src.with_name(f"{stem}_{i}.xlsx")
            i += 1

        # --- build a new DataFrame with a Spotify Link column ---
        df_out = current_df.copy()
        def make_link(isrc):
            return f'=HYPERLINK("https://open.spotify.com/search/isrc:{isrc}", "{isrc}")'
        df_out['Spotify Link'] = df_out['meta_song_isrc'].apply(make_link)

        # write out with the link formula intact
        df_out.to_excel(out, index=False)

        window['STATUS'].update('📥 Saved → ' + out.name)
    
    # … inside your `while True:` event loop …

    # -------------- Click a row → open Spotify by ISRC --------------
    if event == '-TABLE-' and preview_df is not None:
        print("TABLE event!", vals['-TABLE-'])   # <-- debug: see the selected row indices
        sel = vals['-TABLE-']                     # list of selected row indices
        if sel:
            row_idx = sel[0]
            isrc    = preview_df.iloc[row_idx]['ISRC']
            url     = f"https://open.spotify.com/search/isrc:{isrc}"
            print("Opening:", url)                # <-- debug: check URL
            webbrowser.open(url)
    
        # -------------- Filter to UGC / DistroKid --------------
    # -------------- Filter to UGC only --------------
    if event == '-FILTER-UGC-' and df_full is not None:
        filtered = df_full[df_full['Label']=='UGC'].reset_index(drop=True)
        current_df = filtered
        preview30 = filtered.head(30).copy()
        # (reapply the same column‐prep you did in -DONE-)
        preview30['ID']       = preview30['Clip ID'].astype(str).str[:-6]
        preview30['Title']    = preview30['Song Name']
        preview30['Artist']   = preview30['Artist']
        preview30['Label']    = preview30['Label']
        preview30['ISRC']     = preview30['meta_song_isrc']
        preview30['Views']    = preview30['views_this_week']
        preview30['Growth']   = preview30['views_growth_rate']
        preview30['Share %']  = preview30['share_rate']
        preview30['Fav %']    = preview30['fav_rate']
        preview30['Creations']= preview30['creations_this_week']
        preview30['Score']    = preview30['engagement_score']
        preview_df = preview30[[
            'ID','Title','Artist','Label','ISRC',
            'Views','Growth','Share %','Fav %','Creations','Score'
        ]]
        # ← NEW: add the URL text into its own column
        preview_df['Spotify Link'] = preview_df['ISRC'].apply(
            lambda isrc: f"https://open.spotify.com/search/isrc:{isrc}"
        )
        window['-TABLE-'].update(values=preview_df.values.tolist(), visible=True)
        window['STATUS'].update(f'🔎 Showing {len(filtered)} UGC entries')

    # -------------- Filter to DistroKid only --------------
    if event == '-FILTER-DK-' and df_full is not None:
        filtered = df_full[df_full['Label']=='DistroKid'].reset_index(drop=True)
        current_df = filtered
        preview30 = filtered.head(30).copy()
        # same column‐prep as above
        preview30['ID']       = preview30['Clip ID'].astype(str).str[:-6]
        preview30['Title']    = preview30['Song Name']
        preview30['Artist']   = preview30['Artist']
        preview30['Label']    = preview30['Label']
        preview30['ISRC']     = preview30['meta_song_isrc']
        preview30['Views']    = preview30['views_this_week']
        preview30['Growth']   = preview30['views_growth_rate']
        preview30['Share %']  = preview30['share_rate']
        preview30['Fav %']    = preview30['fav_rate']
        preview30['Creations']= preview30['creations_this_week']
        preview30['Score']    = preview30['engagement_score']
        preview_df = preview30[[
            'ID','Title','Artist','Label','ISRC',
            'Views','Growth','Share %','Fav %','Creations','Score'
        ]]
        window['-TABLE-'].update(values=preview_df.values.tolist(), visible=True)
        window['STATUS'].update(f'🔎 Showing {len(filtered)} DistroKid entries')

window.close()
