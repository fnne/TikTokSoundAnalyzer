import streamlit as st
import pandas as pd
import numpy as np
import io

# -------------- Core logic --------------

def find_header_row(df):
    for i,row in df.iterrows():
        txt = row.astype(str).str.lower()
        if "song name" in txt.values or "clip id" in txt.values:
            return i
    raise ValueError("Can't find a header row containing 'Song Name' or 'Clip id'")

def clean_df(df_raw):
    hdr = find_header_row(df_raw)
    df_raw.columns = df_raw.iloc[hdr].astype(str)
    df = df_raw.iloc[hdr+1:].reset_index(drop=True)

    # drop optional unit row
    if df.shape[0] and df.iloc[0].astype(str).str.contains('unit', case=False, na=False).any():
        df = df.iloc[1:].reset_index(drop=True)

    df.columns = df.columns.str.strip()

    # numeric renames
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
        df[c] = pd.to_numeric(df.get(c,0), errors='coerce').fillna(0)

    df['views_growth_rate'] = (
        (df['views_this_week'] - df['views_last_week'])
        / df['views_last_week'].replace(0, np.nan)
    ).fillna(0)

    # metadata renames
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

    # ← DROP ANY DUPLICATE COLUMN NAMES ←
    df = df.loc[:, ~df.columns.duplicated()]

    # ensure all exist
    for col in ['Clip ID','Song Name','Artist','Label','ISRC']:
        if col not in df.columns:
            df[col] = ''
    for col in ['views_this_week','views_last_week',
                'shares_this_week','favorites_this_week',
                'delta_views','delta_creations','creations_this_week',
                'views_growth_rate']:
        if col not in df.columns:
            df[col] = 0

    return df

def rank_df(df, weights, window):
    w = np.array(weights, dtype=float)
    w = w/w.sum() if w.sum()>0 else np.ones_like(w)/len(w)
    for pct in range(0,90,10):
        window.write_event_value('-PROG-', pct)
        time.sleep(0.02)
    df2 = df.copy()
    df2['share_rate'] = np.where(df2.views_this_week>0,
                                 df2.shares_this_week/df2.views_this_week*100, 0)
    df2['fav_rate']   = np.where(df2.views_this_week>0,
                                 df2.favorites_this_week/df2.views_this_week*100, 0)
    dims = ['views_this_week','views_growth_rate','share_rate','fav_rate','creations_this_week']
    ranks = {c: df2[c].rank(pct=True) for c in dims}
    df2['engagement_score'] = sum(ranks[c]*w[i] for i,c in enumerate(dims))
    df2 = df2.sort_values('engagement_score', ascending=False).reset_index(drop=True)
    window.write_event_value('-PROG-', 100)
    return df2

def update_labels(vals):
    for i in range(5):
        window[f'v{i}'].update(f"{vals[f'w{i}']:.2f}")

while True:
    event, vals = window.read(timeout=100)
    if event in (sg.WIN_CLOSED, None): break
    if event in [f'w{i}' for i in range(5)]: update_labels(vals)

    if event=='-BROWSE-':
        threading.Thread(target=lambda: window.write_event_value('-FILE-', sg.popup_get_file(
            'Select TikTok export', file_types=[('Excel','*.xlsx'),('CSV','*.csv')], no_window=True
        )), daemon=True).start()

    if event=='-FILE-' and vals['-FILE-']: window['FILE'].update(vals['-FILE-'])

    if event=='-PROCESS-':
        path=vals['FILE']
        if not path:
            window['STATUS'].update('⚠️ Select a file first'); continue
        window['STATUS'].update('🔄 Loading & ranking…')
        window['-PROG-'].update(0,visible=True)
        threading.Thread(target=lambda: window.write_event_value('-DONE-', rank_df(
            clean_df(pd.read_csv(path,header=None) if path.lower().endswith('.csv') else pd.read_excel(path,header=None)),
            [vals[f'w{i}'] for i in range(5)], window
        )),daemon=True).start()

    if event=='-PROG-':
        window['-PROG-'].update(vals['-PROG-'])

    if event=='-DONE-':
        df_full = vals['-DONE-']; current_df = df_full
        top = df_full.head(30).copy()
        top['ID']       = top['Clip ID'].astype(str).str[:-6]
        top['Views']    = top['views_this_week']
        top['Growth']   = top['views_growth_rate']
        top['Share %']  = top['share_rate']
        top['Fav %']    = top['fav_rate']
        top['Creations']= top['creations_this_week']
        top['Score']    = top['engagement_score']
        preview_df = top[['ID','Song Name','Artist','Label','ISRC','Views','Growth','Share %','Fav %','Creations','Score']]
        window['-TABLE-'].update(values=preview_df.values.tolist(), visible=True)
        window['-FILTER-UGC-'].update(disabled=False)
        window['-FILTER-DK-'].update(disabled=False)
        window['-SAVE-'].update(disabled=False)
        window['STATUS'].update(f'✅ Ranked {len(df_full)} rows')
        window['-PROG-'].update(visible=False)

    if event=='-SAVE-' and current_df is not None:
        out = Path(vals['FILE']).with_name(Path(vals['FILE']).stem + '_ranked.xlsx')
        i = 1
        while out.exists():
            out = out.with_name(f"{out.stem}_{i}.xlsx"); i+=1
        df_out = current_df.copy()
        df_out['Spotify Link'] = df_out['ISRC'].astype(str).apply(
            lambda isrc: f'=HYPERLINK("https://open.spotify.com/search/isrc:{isrc}", "{isrc}")'
        )
        df_out.to_excel(out, index=False)
        window['STATUS'].update('📥 Saved → ' + out.name)

    if event=='-TABLE-' and preview_df is not None:
        sel = vals['-TABLE-']
        if sel:
            isrc = preview_df.iloc[sel[0]]['ISRC']
            webbrowser.open(f"https://open.spotify.com/search/isrc:{isrc}")

    if event=='-FILTER-UGC-' and df_full is not None:
        filtered = df_full[df_full['Label']=='UGC']
        current_df = filtered
        top = filtered.head(30).copy()
        # (same prep as above) …
        # then window['-TABLE-'].update(...)

    if event=='-FILTER-DK-' and df_full is not None:
        filtered = df_full[df_full['Label']=='DistroKid']
        current_df = filtered
        top = filtered.head(30).copy()
        # (same prep as above) …
        # then window['-TABLE-'].update(...)

window.close()
