#!/usr/bin/env python3
"""
TikTok Viral-Sound Analyzer – PySimpleGUI Edition
"""

# ────────── PySimpleGUI License Key ──────────
# Replace the string below with the Distribution Key you received from PySimpleGUI.com
PySimpleGUI_License = 'eWyTJ4M9aAWgNqlQbynVNolHViHblOwOZwSZIA6jIjkORxp7cN3lRBySafW5J51TdnG0lEvcb4iZIZsIIwkLx7plYJ2FVJuIcV2ZVqJ8R8CtIF6oMITYcb0NNCjwUx1KN0DPcq4PM0ymwwi6T5Gtl6jaZSWM5KzhZIUGR4lZcJGFx6vsexWe1wl3bhn5RlWyZLXZJvz0aMW89nuNIHjNoSi8NmSm4Iw0Ivicw5iQTHmMFZteZqUpZEpAcunHNo04IXjRohizRTmOl6ugbYi7IUsfIokK5QhSboWSVdMRY2XCNq0iIXjUowiaTgmflMlScimO1Oh3bGmr4iiZLLCuJ1DSbB2z11wsYtWl5O5GICj1oFi2Vekd9BJSRiCsBuD8cfmgVFhbdsGklm2pZkSGBKHDbUW5JaIXIOicwdiCQJ3oVKzVdyG39xt9ZhXjJyJFRDCFI06iIZjvU657MeTKcV4AIFiHwMirRWGnFV0xZMUAlzzyc23iVHlKZMCpIQ6QIgjkIBwqMSjKUAtlMbDMUktMMtD6YJiVLIClJYEbYhXDRvlnRMXwhXwtazX4J9lucMyBIm6fICjPIhw6MujWYHtLMcDNUptxMCDnYjiBL2C1JxF3bfWcFIpIb6EaFCkxZPHzJQlZco3iMTicOpi6JLmaavWP5ZuIL6mh5mpgZeX1JhtQYZWF5TupQoGvdHt6Y6WVlJsTLgmXNFvYbUSYIhsqIkkjlbQGQiWaRzkccUmRVAzucHyJIC6oIHjAE5wtO8SN483MNqSE4H5CMYy14d1kMaSRJy9i21721d12960a0988989c9ba1bb21ce993c468b181fb1b3e2c4e7ad1954c8e3efbc84486e918f60ad5a0f576fd5f1e2f9dbf0e0b6373e19b9e0f1afc5ca2fcd5572236df3b45734d6d0493a59ea66a1093e3ece37f495ae1e3298c533bbebe1331ae9fae45b9508e6491e8d235734336b7aad7ec8e010e8f8ea4fd34f641aabbf8d7c65de3a6e73e08abe4d8a447779db201b887e03705a57f8411adb1361de621c0362863d7c1209fb20cc205dc3a635ff1b596baa1f65707b14f837f1468300771b00c2a6ddf69709bd1801bac3495943bdebad9dedd696393a395460d6bef60d33b3a58de650dd768324a637da9c5960b055bcb163a6f7c9a13a4d6365687457c1647f03d948bd3e1955aab91f756ca76ae2f61de2a2cacb2d1e94666a1d79734b7b87c86c5adc72a646ccaa1d2b2df0bb590ecb820541e65d8e06df39bac142a7162abe1afe0caa5715a70c21d06fcfdfe73cdbaad08b0e7c496e2f50b352358442580676d3d4b8b212c2b64efe8801bf3e1c1401667c7dcce7f18032ce1a9f6aa1db75989c43171607bb46995b29482eac30fac49d2381e6d1a3c0ff9f034609d750796a728a8075a46484d4346ebedba9818bf9e61c3d55dca03fafa89029c6011b4acfda87be891cef9e3ceb005c1d11ebf5957be0a89a2a7ef9bfc2fb8cfaa1cf708a0b5b62b973a7fb8334cf609708e39e05bea045bbcc41bf9c8ab2'

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
        if row.astype(str).str.contains('song name', case=False, na=False).any():
            return i
    raise ValueError("Can't find a header row containing 'Song Name'")

def clean_df(df_raw):
    hdr = find_header_row(df_raw)
    df_raw.columns = df_raw.iloc[hdr].astype(str)
    df = df_raw.iloc[hdr+1:].reset_index(drop=True)
    # drop unit‐row
    if df.shape[0] and df.iloc[0].astype(str).str.contains('unit', case=False, na=False).any():
        df = df.iloc[1:].reset_index(drop=True)
    df.columns = df.columns.str.strip()
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
    # numeric coercion
    for col in ['views_this_week','views_last_week',
                'shares_this_week','favorites_this_week',
                'delta_views','delta_creations']:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)
    df['views_growth_rate'] = (
        (df['views_this_week'] - df['views_last_week'])
        / df['views_last_week'].replace(0, np.nan)
    ).fillna(0)
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
    
    # Click a row → open Spotify by ISRC
    if event == '-TABLE-' and preview_df is not None:
        sel = vals['-TABLE-']           # list of selected row indices
        if sel:
            isrc = preview_df.iloc[sel[0]]['ISRC']
            webbrowser.open(f"https://open.spotify.com/search/isrc:{isrc}")
    
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
