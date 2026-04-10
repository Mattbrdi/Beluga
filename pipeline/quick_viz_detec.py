# detections_dashboard.py
# -*- coding: utf-8 -*-
#
# Dash viewer for two 4-ch wavs + detection CSVs, with overplotted rectangles.
# - Visual window: 3 seconds (sliding)
# - Controls: PLAY/PAUSE, PREVIOUS, NEXT, RESTART
# - Spectrograms in PSD [dB re 1 µPa²/Hz] using fixed sensitivity = -165.1 dB re 1 V/µPa
# - Rectangles:
#     * Whistle: 500–12000 Hz
#     * ECHO   : 30000–120000 Hz
#   Color code (dashboard.py style-ish):
#     * common (both tetras, same Timestamp + same type): red outline + black translucent fill
#     * single tetra only: gray outline + black lighter fill
# - Under each spectro: table (reverse chronological) with Timestamp, Type, Prob
#
# NOTE: "common" is matched on EXACT Timestamp + Type. If your timestamps differ by a few ms,
#       we can switch to a tolerant match (merge_asof) easily.

import os
import threading
import time as pytime
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import librosa
import plotly.graph_objs as go

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State


# =========================
# CONFIG (tes chemins)
# =========================
WAV_8296 = r"\\lisse-tache.uqo.ca\Commun\Pour Irene\4-canaux Cacouna\8296.240804084225.wav"
WAV_8295 = r"\\lisse-tache.uqo.ca\Commun\Pour Irene\4-canaux Cacouna\8295.240804084225.wav"

CSV_8296 = r"\\lisse-tache.uqo.ca\Commun\Pour Irene\4-canaux Cacouna\Output\detections_8296.csv"
CSV_8295 = r"\\lisse-tache.uqo.ca\Commun\Pour Irene\4-canaux Cacouna\Output\detections_8295.csv"

# Fenêtre affichée (s) et pas d'avance (s)
WINDOW_DURATION_S = 3.0
CHUNK_DURATION_S = 1.0

# Durée fixe d'une détection (s)
DETECTION_DURATION_S = 1.0

# Spectro
N_FFT = 4096
HOP_LENGTH = 1024

# Sensibilité (fixe) -> PSD en µPa²/Hz
SENS_DB = -165.1  # dB re 1 V / µPa
SENS_LIN = 10 ** (SENS_DB / 20)  # V / µPa

# Fréquences d'affichage (log)
FMIN = 250
FMAX = 192000

# Bandes de rectangles
BAND_WHISTLE = (500, 12000)
BAND_ECHO = (30000, 120000)

# Seuil optionnel (si tu veux enlever les lignes "faibles")
CALL_DET_THRESHOLD = None  # ex: 0.5, ou None pour garder tout


# =========================
# Helpers
# =========================
def parse_audio_start_dt_from_filename(path: str) -> datetime:
    """
    Extrait YYMMDDHHMMSS depuis '8296.240804084225.wav' -> 2024-08-04 08:42:25
    """
    base = os.path.basename(path)
    stamp = base.split(".")[1]  # "240804084225"
    yy = int(stamp[0:2])
    mm = int(stamp[2:4])
    dd = int(stamp[4:6])
    HH = int(stamp[6:8])
    MI = int(stamp[8:10])
    SS = int(stamp[10:12])
    return datetime(2000 + yy, mm, dd, HH, MI, SS)


def safe_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes", "y")
    return bool(x)


def load_detection_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    for col in ["ECHO", "Whistle"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_bool)
        else:
            df[col] = False

    for col in ["ECHO_prob", "Whistle_prob"]:
        if col not in df.columns:
            df[col] = np.nan

    if CALL_DET_THRESHOLD is not None and "Call_Detection" in df.columns:
        df = df[df["Call_Detection"] > float(CALL_DET_THRESHOLD)].copy()

    keep = ["Timestamp", "ECHO", "ECHO_prob", "Whistle", "Whistle_prob"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].sort_values("Timestamp").reset_index(drop=True)
    return df


def expand_events(df: pd.DataFrame, tetra_tag: str) -> pd.DataFrame:
    """
    Convertit df (bool ECHO/Whistle) -> table d'événements:
    une ligne par (Timestamp, type), avec prob.
    """
    rows = []
    for _, r in df.iterrows():
        ts = r["Timestamp"]
        if safe_bool(r.get("ECHO", False)):
            rows.append({"tetra": tetra_tag, "Timestamp": ts, "type": "ECHO", "prob": r.get("ECHO_prob", np.nan)})
        if safe_bool(r.get("Whistle", False)):
            rows.append({"tetra": tetra_tag, "Timestamp": ts, "type": "Whistle", "prob": r.get("Whistle_prob", np.nan)})

    if not rows:
        return pd.DataFrame(columns=["tetra", "Timestamp", "type", "prob"])
    return pd.DataFrame(rows).sort_values("Timestamp").reset_index(drop=True)


def build_common_key_set(events_a: pd.DataFrame, events_b: pd.DataFrame) -> set:
    """
    Détections communes = même Timestamp exact + même type.
    """
    if events_a.empty or events_b.empty:
        return set()
    a_keys = set(zip(events_a["Timestamp"], events_a["type"]))
    b_keys = set(zip(events_b["Timestamp"], events_b["type"]))
    return a_keys.intersection(b_keys)


def fmt_prob(x):
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.3f}"
    except Exception:
        return ""


# =========================
# Load data (audio + detections)
# =========================
audio_start_dt = parse_audio_start_dt_from_filename(WAV_8296)

# Audio (4 canaux). On prend un canal simple pour faire le spectro (canal 2 si dispo, sinon mono).
y, sr_8296 = librosa.load(WAV_8296, sr=None, mono=False)
sig_8296 = y[1] if (isinstance(y, np.ndarray) and y.ndim > 1 and y.shape[0] > 1) else y

y, sr_8295 = librosa.load(WAV_8295, sr=None, mono=False)
sig_8295 = y[1] if (isinstance(y, np.ndarray) and y.ndim > 1 and y.shape[0] > 1) else y

# Harmoniser sample rate (on suppose identique; sinon, resample le 2e)
if sr_8295 != sr_8296:
    sig_8295 = librosa.resample(sig_8295, orig_sr=sr_8295, target_sr=sr_8296)
    sr_8295 = sr_8296

# Harmoniser longueur (min)
min_len = min(len(sig_8296), len(sig_8295))
sig_8296 = sig_8296[:min_len]
sig_8295 = sig_8295[:min_len]

# Détections
df_8296 = load_detection_csv(CSV_8296)
df_8295 = load_detection_csv(CSV_8295)

events_8296 = expand_events(df_8296, "8296")
events_8295 = expand_events(df_8295, "8295")
common_keys = build_common_key_set(events_8296, events_8295)

total_seconds = min_len / sr_8296


# =========================
# Audio streamer
# =========================
class AudioStreamer:
    def __init__(self, data1, data2, sample_rate, chunk_duration=1.0, window_duration=3.0):
        self.data1 = data1
        self.data2 = data2
        self.sample_rate = sample_rate
        self.chunk_samples = max(1, int(chunk_duration * sample_rate))
        self.window_samples = max(1, int(window_duration * sample_rate))
        self.total_samples = len(data1)

        # démarre avec une fenêtre pleine
        self.index = min(self.total_samples, self.window_samples)

        self.lock = threading.Lock()
        self.running = True

        self.paused = True
        self.started = False
        self.finished = False

        self.thread = threading.Thread(target=self.update_index, daemon=True)

    def start(self):
        if not self.started:
            self.started = True
            self.thread.start()

    def reset(self):
        with self.lock:
            self.index = min(self.total_samples, self.window_samples)
            self.finished = False

    def update_index(self):
        while self.running:
            pytime.sleep(CHUNK_DURATION_S)
            if self.paused:
                continue
            with self.lock:
                if self.index + self.chunk_samples < self.total_samples:
                    self.index += self.chunk_samples
                else:
                    # fin -> auto pause
                    self.index = self.total_samples
                    self.paused = True
                    self.finished = True

    def step_chunks(self, delta_chunks: int):
        with self.lock:
            self.finished = False
            new_index = self.index + delta_chunks * self.chunk_samples

            # wrap
            if new_index < self.window_samples:
                new_index = self.total_samples - self.chunk_samples
            elif new_index >= self.total_samples:
                new_index = self.window_samples

            self.index = max(self.window_samples, min(self.total_samples, new_index))

    def get_current_window(self):
        with self.lock:
            end = self.index
            start = max(0, end - self.window_samples)
            return self.data1[start:end], self.data2[start:end], start / self.sample_rate, end / self.sample_rate


streamer = AudioStreamer(sig_8296, sig_8295, sr_8296, CHUNK_DURATION_S, WINDOW_DURATION_S)


# =========================
# Dash app
# =========================
app = dash.Dash(__name__)

app.layout = html.Div(
    [
        dcc.Store(id="paused-store", data=True),

        html.Div(
            [
                html.Button("PLAY", id="pause-btn", n_clicks=0, style={"marginRight": "10px"}),
                html.Button("PREVIOUS", id="prev-btn", n_clicks=0, style={"marginRight": "10px"}),
                html.Button("NEXT", id="next-btn", n_clicks=0, style={"marginRight": "10px"}),
                html.Button("RESTART", id="restart-btn", n_clicks=0, style={"marginRight": "10px"}),
            ],
            style={"marginBottom": "10px"},
        ),

        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id="spectro-8296", style={"height": "320px"}),
                        dash_table.DataTable(
                            id="table-8296",
                            columns=[
                                {"name": "Timestamp", "id": "Timestamp"},
                                {"name": "Type", "id": "Type"},
                                {"name": "Prob", "id": "Prob"},
                            ],
                            data=[],
                            style_table={"height": "220px", "overflowY": "auto"},
                            style_cell={"textAlign": "center", "padding": "4px"},
                            style_header={"fontWeight": "bold"},
                        ),
                    ],
                    style={"width": "49%"},
                ),
                html.Div(
                    [
                        dcc.Graph(id="spectro-8295", style={"height": "320px"}),
                        dash_table.DataTable(
                            id="table-8295",
                            columns=[
                                {"name": "Timestamp", "id": "Timestamp"},
                                {"name": "Type", "id": "Type"},
                                {"name": "Prob", "id": "Prob"},
                            ],
                            data=[],
                            style_table={"height": "220px", "overflowY": "auto"},
                            style_cell={"textAlign": "center", "padding": "4px"},
                            style_header={"fontWeight": "bold"},
                        ),
                    ],
                    style={"width": "49%"},
                ),
            ],
            style={"display": "flex", "justifyContent": "space-between"},
        ),

        dcc.Interval(id="startup", interval=250, n_intervals=0, max_intervals=1),
        dcc.Interval(id="interval", interval=int(CHUNK_DURATION_S * 1000), n_intervals=0, disabled=True),
    ],
    style={"padding": "10px"},
)


@app.callback(
    Output("paused-store", "data"),
    Output("interval", "disabled"),
    Output("pause-btn", "children"),
    Input("startup", "n_intervals"),
    Input("pause-btn", "n_clicks"),
    Input("restart-btn", "n_clicks"),
    Input("interval", "n_intervals"),
    State("paused-store", "data"),
    prevent_initial_call=False,
)
def start_and_toggle(startup_n, pause_clicks, restart_clicks, n_intervals, paused):
    ctx = dash.callback_context
    trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "startup"

    if trig == "startup":
        streamer.start()
        streamer.paused = True
        streamer.finished = False
        return True, True, "PLAY"

    if trig == "restart-btn":
        streamer.reset()
        streamer.paused = True
        streamer.finished = False
        return True, True, "PLAY"

    # auto PAUSE quand arrivé à la fin
    if trig == "interval":
        if getattr(streamer, "finished", False):
            streamer.paused = True
            return True, True, "PLAY"
        raise dash.exceptions.PreventUpdate

    # toggle
    paused = not bool(paused)
    streamer.paused = paused
    if not paused:
        streamer.finished = False
    return paused, paused, ("PLAY" if paused else "PAUSE")


@app.callback(
    Output("spectro-8296", "figure"),
    Output("spectro-8295", "figure"),
    Output("table-8296", "data"),
    Output("table-8295", "data"),
    Input("startup", "n_intervals"),
    Input("interval", "n_intervals"),
    Input("prev-btn", "n_clicks"),
    Input("next-btn", "n_clicks"),
    Input("restart-btn", "n_clicks"),
    State("paused-store", "data"),
)
def update_visuals(startup_n, n, prev_clicks, next_clicks, restart_clicks, paused):
    ctx = dash.callback_context
    trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "startup"

    if not streamer.started:
        streamer.start()

    if trig == "prev-btn":
        streamer.step_chunks(-1)
    elif trig == "next-btn":
        streamer.step_chunks(+1)
    elif trig == "restart-btn":
        # reset handled elsewhere; ok
        pass

    data_8296, data_8295, t0_s, t1_s = streamer.get_current_window()

    # ================
    # STFT -> PSD dB re 1 µPa²/Hz
    # ================
    S1 = np.abs(librosa.stft(data_8296, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S2 = np.abs(librosa.stft(data_8295, n_fft=N_FFT, hop_length=HOP_LENGTH))

    S1_uPa = S1 / SENS_LIN
    S2_uPa = S2 / SENS_LIN

    PSD1 = (S1_uPa ** 2) / (sr_8296 / N_FFT)
    PSD2 = (S2_uPa ** 2) / (sr_8296 / N_FFT)

    PSD1_dB = 10 * np.log10(PSD1 + 1e-24)
    PSD2_dB = 10 * np.log10(PSD2 + 1e-24)

    # Axes temps
    times = librosa.frames_to_time(np.arange(S1.shape[1]), sr=sr_8296, hop_length=HOP_LENGTH)
    times = times + t0_s
    times_dt = [audio_start_dt + timedelta(seconds=t) for t in times]

    # Axes fréquence
    freqs = librosa.fft_frequencies(sr=sr_8296, n_fft=N_FFT)
    freq_mask = (freqs >= FMIN) & (freqs <= FMAX)
    freqs_f = freqs[freq_mask]

    z1 = PSD1_dB[freq_mask, :]
    z2 = PSD2_dB[freq_mask, :]

    # Robust color range
    z1_min, z1_max = 40, 130
    z2_min, z2_max = 40, 130

    fig1 = go.Figure(
        data=go.Heatmap(
            z=z1,
            x=times_dt,
            y=freqs_f,
            colorscale="cividis",
            zmin=z1_min,
            zmax=z1_max,
            colorbar=dict(title=dict(text="PSD [dB re 1 µPa²/Hz]", side="right")),
        )
    )
    fig2 = go.Figure(
        data=go.Heatmap(
            z=z2,
            x=times_dt,
            y=freqs_f,
            colorscale="cividis",
            zmin=z2_min,
            zmax=z2_max,
            colorbar=dict(title=dict(text="PSD [dB re 1 µPa²/Hz]", side="right")),
        )
    )

    # Axes cosmetics (3s window -> ticks serrés)
    fig1.update_xaxes(
        tickformat="%H:%M:%S",          # plus lisible pour les ticks majeurs
        dtick=1_000,                # 1 seconde
        ticklen=8,
        tickwidth=2,
        ticks="outside",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.35)",
        gridwidth=1,
        minor=dict(
            dtick=100,            # 0.1 secondes
            ticklen=4,
            tickwidth=1,
            ticks="outside",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            gridwidth=0.5,
        )
    )

    fig2.update_xaxes(
        tickformat="%H:%M:%S",
        dtick=1_000,
        ticklen=8,
        tickwidth=2,
        ticks="outside",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.35)",
        gridwidth=1,
        minor=dict(
            dtick=100,            # 0.1 secondes
            ticklen=4,
            tickwidth=1,
            ticks="outside",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            gridwidth=0.5,
        )
    )


    fig1.update_yaxes(
        type="log",
        range=[np.log10(250), np.log10(192000)],
        title="Fréquence [Hz]",

        # --- Ticks majeurs (décades) ---
        tickmode="array",
        tickvals=[1e2, 1e3, 1e4, 1e5],
        ticktext=["100", "1000", "10000", "100000"],

        ticklen=8,
        tickwidth=2,
        ticks="outside",

        showgrid=True,
        gridcolor="rgba(0,0,0,0.35)",
        gridwidth=1,

        # --- Ticks mineurs ---
        minor=dict(
            tickmode="array",
            tickvals=[
                2e2, 3e2, 4e2, 5e2, 6e2, 7e2, 8e2, 9e2,
                2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3,
                2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4,
                2e5
            ],
            ticklen=4,
            tickwidth=1,
            ticks="outside",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            gridwidth=0.5,
        ),
    )

    fig2.update_yaxes(
        type="log",
        range=[np.log10(250), np.log10(192000)],
        title="Fréquence [Hz]",
        tickmode="array",
        tickvals=[1e2, 1e3, 1e4, 1e5],
        ticktext=["100", "1000", "10000", "100000"],
        ticklen=8,
        tickwidth=2,
        ticks="outside",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.35)",
        gridwidth=1,
        minor=dict(
            tickmode="array",
            tickvals=[
                2e2, 3e2, 4e2, 5e2, 6e2, 7e2, 8e2, 9e2,
                2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3,
                2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4,
                2e5
            ],
            ticklen=4,
            tickwidth=1,
            ticks="outside",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            gridwidth=0.5,
        ),
    )

    # Fenêtre actuelle (datetime)
    win_start_dt = audio_start_dt + timedelta(seconds=float(t0_s))
    win_end_dt = audio_start_dt + timedelta(seconds=float(t1_s))

    def add_rectangles_and_table(fig, events_df: pd.DataFrame):
        in_win = events_df[events_df["Timestamp"] <= win_end_dt].copy()

        for _, ev in in_win.iterrows():
            ts = ev["Timestamp"]
            typ = ev["type"]
            key = (ts, typ)
            common = key in common_keys

            # Couleurs style dashboard.py: contour rouge + fill noir très léger (ou gris)
            if common:
                line_col = "red"
                fill_col = "rgba(0, 0, 0, 0.05)"
            else:
                line_col = "gray"
                fill_col = "rgba(0, 0, 0, 0.03)"

            x0 = ts
            x1 = ts + timedelta(seconds=DETECTION_DURATION_S)

            if typ == "ECHO":
                y0, y1 = BAND_ECHO
            else:
                y0, y1 = BAND_WHISTLE

            # clamp au viewport
            if x1 < win_start_dt or x0 > win_end_dt:
                continue
            x0c = max(x0, win_start_dt)
            x1c = min(x1, win_end_dt)
            if x1c <= x0c:
                continue

            fig.add_shape(
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=x0c,
                    x1=x1c,
                    y0=y0,
                    y1=y1,
                    line=dict(color=line_col, width=2),
                    fillcolor=fill_col,
                    layer="above",
                )
            )

        # Table (ordre inverse)
        table_rows = []
        in_win_sorted = in_win.sort_values("Timestamp", ascending=False)
        for _, ev in in_win_sorted.iterrows():
            table_rows.append(
                {
                    "Timestamp": ev["Timestamp"].strftime("%H:%M:%S"),
                    "Type": ev["type"],
                    "Prob": fmt_prob(ev["prob"]),
                }
            )
        return table_rows

    table_8296 = add_rectangles_and_table(fig1, events_8296)
    table_8295 = add_rectangles_and_table(fig2, events_8295)

    return fig1, fig2, table_8296, table_8295


if __name__ == "__main__":
    print("Dash is running on http://127.0.0.1:8050/")
    app.run(debug=False)
