#!/usr/bin/env python3
import os
import threading, time, webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import PySimpleGUI as sg

# ─── License from ENV ─────────────────────────────────────────────────────────
PySimpleGUI_License = os.getenv("PYSG_LICENSE", "")

# ─── Core data prep ────────────────────────────────────────────────────────────
def find_header(df):
    for i, row in df.iterrows():
        txt = row.astype(str).str.lower()
        if txt.str.contains("song name|clip id", regex=True, na=False).any():
            return i
    raise ValueError("Header row not found")

def clean_df(df_raw):
    hdr = find_header(df_raw)
    df_raw.columns = df_raw.iloc[hdr].astype(str).str.strip()
    df = df_raw.iloc[hdr + 1 :].reset_index(drop=True)

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

    # unify metadata
    meta_map = {
        **dict.fromkeys(["clip id", "Clip id", "Clip ID"], "Clip ID"),
        **dict.fromkeys(
            ["clip name", "Clip name", "Song Name", "Meta Song Name", "meta_song_name"],
            "Song Name",
        ),
        **dict.fromkeys(["Artist", "Meta Artist", "meta_artist"], "Artist"),
        "Label": "Label",
        **dict.fromkeys(["meta_song_isrc", "ISRC"], "ISRC"),
    }
    df = df.rename(columns=meta_map)

    # ensure columns
    for col in ["Clip ID", "Song Name", "Artist", "Label", "ISRC"]:
        df.setdefault(col, "")
    return df.loc[:, ~df.columns.duplicated()]

def rank_df(df, weights, window):
    w = np.array(weights, float)
    w = w / w.sum() if w.sum() else np.ones_like(w) / len(w)

    # fake progress bar
    for pct in range(0, 100, 10):
        window.write_event_value("-PROG-", pct)
        time.sleep(0.02)

    df2 = df.copy()
    df2["share_rate"] = np.where(
        df2.views_this_week > 0, df2.shares_this_week / df2.views_this_week * 100, 0
    )
    df2["fav_rate"] = np.where(
        df2.views_this_week > 0, df2.favorites_this_week / df2.views_this_week * 100, 0
    )

    dims = ["views_this_week", "views_growth_rate", "share_rate", "fav_rate", "creations_this_week"]
    ranks = {c: df2[c].rank(pct=True) for c in dims}
    df2["engagement_score"] = sum(ranks[c] * w[i] for i, c in enumerate(dims))

    window.write_event_value("-PROG-", 100)
    return df2.sort_values("engagement_score", ascending=False).reset_index(drop=True)

# ─── GUI layout ────────────────────────────────────────────────────────────────
sg.theme("DarkGrey12")
layout = [
    [sg.Text("Export File:"), sg.Input(key="FILE", readonly=True, expand_x=True),
     sg.Button("Browse", key="-BROWSE-")],
    [sg.Column([
        [sg.Text(lbl, size=(14,1)),
         sg.Slider((0,10), val, orientation="h", size=(40,15), key=f"w{i}", enable_events=True),
         sg.Text(f"{val:.2f}", key=f"v{i}")]
        for i,(lbl,val) in enumerate([
            ("Total views",5),("WoW growth",2),
            ("Share rate (%)",3),("Fav rate (%)",1),
            ("Creations",1),
        ])
    ], expand_x=True)],
    [sg.Button("Process", key="-PROCESS-"),
     sg.Button("UGC only", key="-UGC-", disabled=True),
     sg.Button("DistroKid only", key="-DK-", disabled=True),
     sg.Button("Save Excel", key="-SAVE-", disabled=True)],
    [sg.Text("", key="STATUS", size=(60,1))],
    [sg.ProgressBar(100, orientation="h", size=(40,10), key="-PROG-", visible=False)],
    [sg.Table(
        values=[], headings=["ID","Title","Artist","Label","ISRC","Views","Growth",
                             "Share %","Fav %","Creations","Score"],
        key="-TABLE-", visible=False, num_rows=30,
        expand_x=True, expand_y=True, enable_events=True
    )]
]
window = sg.Window("TikTok Viral-Sound Analyzer", layout, finalize=True,
                   resizable=True, size=(1200,800))

def update_sliders(vals):
    for i in range(5):
        window[f"v{i}"].update(f"{vals[f'w{i}']:.2f}")

df_full = preview_df = None

# ─── Event loop ───────────────────────────────────────────────────────────────
while True:
    event, vals = window.read(timeout=100)
    if event in (sg.WIN_CLOSED, None):
        break
    if event in [f"w{i}" for i in range(5)]:
        update_sliders(vals)

    if event == "-BROWSE-":
        threading.Thread(target=lambda: window.write_event_value(
            "-FILE-", sg.popup_get_file("Select export", no_window=True, file_types=[("Excel","*.xlsx"),("CSV","*.csv")])
        ), daemon=True).start()

    if event == "-FILE-" and vals["FILE"]:
        window["FILE"].update(vals["FILE"])

    if event == "-PROCESS-":
        path = vals["FILE"]
        if not path:
            window["STATUS"].update("⚠️ Select a file first")
            continue
        window["STATUS"].update("🔄 Processing…")
        window["-PROG-"].update(0, visible=True)
        def worker():
            try:
                df_raw = pd.read_csv(path, header=None) if path.lower().endswith(".csv") else pd.read_excel(path, header=None)
                clean = clean_df(df_raw)
                ranked = rank_df(clean, [vals[f"w{i}"] for i in range(5)], window)
                window.write_event_value("-DONE-", ranked)
            except Exception as e:
                window.write_event_value("-ERROR-", str(e))
        threading.Thread(target=worker, daemon=True).start()

    if event == "-PROG-":
        window["-PROG-"].update(vals["-PROG-"])

    if event == "-DONE-":
        df_full = vals["-DONE-"]
        preview = df_full.head(30).copy()
        preview["ID"]    = preview["Clip ID"].astype(str).str[:-6]
        preview["Title"] = preview["Song Name"]
        preview["Score"] = preview["engagement_score"]
        preview_df = preview[["ID","Title","Artist","Label","ISRC",
                              "views_this_week","views_growth_rate",
                              "share_rate","fav_rate","creations_this_week","Score"]]
        window["-TABLE-"].update(values=preview_df.values, visible=True)
        for btn in ("-UGC-","-DK-","-SAVE-"):
            window[btn].update(disabled=False)
        window["STATUS"].update(f"✅ Ranked {len(df_full)} rows")
        window["-PROG-"].update(visible=False)

    if event == "-ERROR-":
        window["STATUS"].update("❌ " + vals["-ERROR-"])
        window["-PROG-"].update(visible=False)

    if event in ("-UGC-","-DK-") and df_full is not None:
        label = "UGC" if event=="-UGC-" else "DistroKid"
        filtered = df_full[df_full.Label == label].reset_index(drop=True)
        preview30 = filtered.head(30).copy()
        preview30["ID"] = preview30["Clip ID"].astype(str).str[:-6]
        preview30["Title"] = preview30["Song Name"]
        preview30["Score"] = preview30["engagement_score"]
        preview_df = preview30[["ID","Title","Artist","Label","ISRC",
                                "views_this_week","views_growth_rate",
                                "share_rate","fav_rate","creations_this_week","Score"]]
        window["-TABLE-"].update(values=preview_df.values)
        window["STATUS"].update(f"🔎 Showing {len(filtered)} {label}")

    if event == "-SAVE-" and df_full is not None:
        out = Path(vals["FILE"]).with_suffix("_ranked.xlsx")
        i = 1
        while out.exists():
            out = out.with_name(f"{out.stem}_{i}{out.suffix}")
            i += 1
        df_out = df_full.copy()
        df_out["Spotify Link"] = df_out["ISRC"].apply(
            lambda isrc: f'=HYPERLINK("https://open.spotify.com/search/isrc:{isrc}", "{isrc}")'
        )
        df_out.to_excel(out, index=False)
        window["STATUS"].update(f"📥 Saved → {out.name}")

window.close()
