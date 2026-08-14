import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_leaflet as dl
import plotly.graph_objs as go
import numpy as np
import librosa
import threading
import time
import folium
import os
from datetime import timedelta, datetime
from pyproj import Transformer
from math import radians, sin, cos, sqrt, atan2
import csv
import pandas as pd

def _scalar(x):
    """Retourne un float à partir d'un scalaire numpy/array de shape (1,) ou (1,1)."""
    arr = np.asarray(x)
    return float(arr.reshape(-1)[0])


def enu_to_lla(e, n, u, lat_ref, lon_ref, alt_ref):
    """Fonction pour convertir ENU centre sur un point de reference a LLA."""
    transformer = Transformer.from_crs("epsg:4326", "epsg:4978")
    X_ref, Y_ref, Z_ref = transformer.transform(lat_ref, lon_ref, alt_ref)

    lat_ref = np.radians(lat_ref)
    lon_ref = np.radians(lon_ref)

    R = np.array(
        [
            [-np.sin(lon_ref), np.cos(lon_ref), 0],
            [
                -np.sin(lat_ref) * np.cos(lon_ref),
                -np.sin(lat_ref) * np.sin(lon_ref),
                np.cos(lat_ref),
            ],
            [
                np.cos(lat_ref) * np.cos(lon_ref),
                np.cos(lat_ref) * np.sin(lon_ref),
                np.sin(lat_ref),
            ],
        ]
    )

    enu = np.array([e, n, u])
    d = R.T @ enu

    X = d[0] + X_ref
    Y = d[1] + Y_ref
    Z = d[2] + Z_ref

    transformer = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)
    lon, lat, alt = transformer.transform(X, Y, Z)

    return lat, lon, alt


def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    R = 6371.0

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def set_dashboard(audio_files, positions, errors, timestamps, durations, call_types,
                  groundtruths, environment,
                  event_times=None, event_durations=None, event_call_types=None, event_status=None,
                  detections_dfs=None, detection_threshold=0.5):

    """
    Create a dashboard from the data of the main pipeline. So far, it can only be used with 2 tetrahedrons.

    Parameters:
    - audio_files: List of audio_files.
    - positions, errors, timestamps, durations, call_types: Outputs of the positions_from_audio function.
    - groundtruths: List of groundtruths with format (latitude, longitude) for each groundtruth.
    - environment: Environment containing tetrahedra information.

    Returns:
    - app: Dashboard to launch.
    """

    event_times = event_times or []
    event_durations = event_durations or []
    event_call_types = event_call_types or []
    event_status = event_status or []

    # Load audio files
    audio_path1, audio_path2 = audio_files
    year, month, day, hour, minute, second = (
        2000 + int(audio_path1[-16:-14]),
        int(audio_path1[-14:-12]),
        int(audio_path1[-12:-10]),
        int(audio_path1[-10:-8]),
        int(audio_path1[-8:-6]),
        int(audio_path1[-6:-4]),
    )
    audio_start_time = f"{audio_path1[-10:-8]}:{audio_path1[-8:-6]}:{audio_path1[-6:-4]}"

    event_timestamps = [
        (t - datetime(year, month, day, hour, minute, second)).total_seconds()
        for t in event_times
    ]

    y, sr1 = librosa.load(audio_path1, sr=None, mono=False)
    y1 = y[1] if y.ndim > 1 and y.shape[0] > 1 else y
    y, sr2 = librosa.load(audio_path2, sr=None, mono=False)
    y2 = y[1] if y.ndim > 1 and y.shape[0] > 1 else y

    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]

    # Real-time streaming of audios
    class AudioStreamer:
        def __init__(self, data1, data2, sample_rate, chunk_duration=1.0, window_duration=3.0):
            self.data1 = data1
            self.data2 = data2
            self.sample_rate = sample_rate
            self.chunk_samples = int(chunk_duration * sample_rate)
            self.window_samples = int(window_duration * sample_rate)
            self.total_samples = len(data1)

            # IMPORTANT: démarre avec une fenêtre pleine (évite STFT sur tableau vide)
            self.index = self.window_samples

            self.lock = threading.Lock()
            self.running = True

            # NEW: pause flag
            self.paused = True
            self.started = False
            self.finished = False

            self.thread = threading.Thread(target=self.update_index, daemon=True)
            #self.thread.start()

            self.rms_times = []
            self.rms1_values = []
            self.rms2_values = []

        def reset(self):
            """Retour au début du signal (première fenêtre)."""
            with self.lock:
                self.index = self.window_samples
                self.finished = False

        def start(self):
            """Démarre le thread une seule fois."""
            if not self.started:
                self.started = True
                self.thread.start()

        def update_index(self):
            while self.running:
                time.sleep(1.0)
                if self.paused:
                    continue
                with self.lock:
                    # avance normalement si on n'est pas à la fin
                    if self.index + self.chunk_samples < self.total_samples:
                        self.index += self.chunk_samples
                    else:
                        # NEW: auto-PAUSE à la fin (pas de loop)
                        self.index = self.total_samples  # clamp
                        self.paused = True
                        self.finished = True


        def step_chunks(self, delta_chunks: int):
            with self.lock:
                self.finished = False
                new_index = self.index + delta_chunks * self.chunk_samples

                if new_index < self.window_samples:
                    new_index = self.total_samples - self.chunk_samples
                elif new_index >= self.total_samples:
                    new_index = self.window_samples

                self.index = new_index


        def get_current_chunk(self):
            with self.lock:
                start = max(0, self.index - self.window_samples)
                end = self.index
                return self.data1[start:end], self.data2[start:end], start / self.sample_rate

        def stop(self):
            self.running = False
            self.thread.join()

    streamer = AudioStreamer(y1, y2, sr1)

    # Input preparation
    tetrahedrons_coords = [v.origin_lla[:2] for v in environment.tetrahedras.values()]
    T1_lat, T1_lon = tetrahedrons_coords[0]

    lla_positions = [enu_to_lla(*position, T1_lat, T1_lon, 0) for position in positions]

    # Inputs
    highlight_coords = [(position[0], position[1]) for position in lla_positions]
    errors = [(error[0], error[1]) for error in errors]
    highlight_timestamps = [
        (timestamp - datetime(year, month, day, hour, minute, second)).total_seconds()
        for timestamp in timestamps
    ]

    tetra_events = [set(), set()]
    common_events = set()

    INTEREST_LABELS = ["Whistle", "HFPC"]  # 👈 uniquement ceux-ci

    def _iter_events_from_df(df, thr):
        if df is None or len(df) == 0:
            return
        df = df.copy()

        # Filtre optionnel sur Call_Detection si la colonne existe
        if thr is not None and "Call_Detection" in df.columns:
            df = df[df["Call_Detection"] > float(thr)]

        # On utilise Timestamp -> on "floor" à la seconde (HH:MM:SS)
        if "Timestamp" not in df.columns:
            return
        ts = pd.to_datetime(df["Timestamp"]).dt.floor("s")

        for i, row in df.iterrows():
            ts_sec = ts.loc[i]
            for lab in INTEREST_LABELS:
                if lab in df.columns and bool(row[lab]):
                    yield (ts_sec, lab)

    # Construire sets par tétra + intersection
    if detections_dfs and len(detections_dfs) >= 2:
        for k in [0, 1]:
            for ev in _iter_events_from_df(detections_dfs[k], detection_threshold):
                tetra_events[k].add(ev)

        common_events = tetra_events[0].intersection(tetra_events[1])

    intervals_nodes = [int(x) for x in highlight_timestamps]
    highlight_intervals = [(intervals_nodes[i], intervals_nodes[i + 1]) for i in range(len(intervals_nodes) - 1)]
    distances_to_groundtruths = [min(haversine(position, ref) * 1000 for ref in groundtruths) for position in highlight_coords]

    # Structure of the app
    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            # NEW: store pause + controls
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
                    dcc.Graph(id="spectrogram1", style={"width": "48%", "height": "300px", "display": "inline-block"}),
                    dcc.Graph(id="spectrogram2", style={"width": "48%", "height": "300px", "display": "inline-block"}),
                ],
                style={"display": "flex", "justifyContent": "space-between"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dl.Map(
                                id="leaflet-map",
                                center=[T1_lat, T1_lon],
                                zoom=13.5,
                                zoomSnap=0.1,
                                zoomDelta=0.1,
                                style={"width": "100%", "height": "100%"},
                                children=[
                                    dl.LayersControl(
                                        [
                                            # Vue géographique (par défaut)
                                            dl.BaseLayer(
                                                dl.TileLayer(
                                                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                                                    attribution="© OpenStreetMap contributors",
                                                ),
                                                name="Carte",
                                                checked=True,
                                            ),

                                            # Vue satellite
                                            dl.BaseLayer(
                                                dl.TileLayer(
                                                    url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                                                    attribution="© OpenTopoMap contributors",
                                                ),
                                                name="Topo",
                                                checked=False,
                                            ),

                                            # Satellite ESRI (recommandé)
                                            dl.BaseLayer(
                                                dl.TileLayer(
                                                    url="https://server.arcgisonline.com/ArcGIS/rest/services/"
                                                        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
                                                    attribution="Tiles © Esri",
                                                ),
                                                name="Satellite",
                                                checked=False,
                                            ),
                                        ],
                                        position="topright",
                                    ),
                                    dl.ScaleControl(position="bottomleft", metric=True, imperial=False),
                                    dl.LayerGroup(id="map-markers"),
                                ],
                            )
                        ],
                        style={"width": "30%", "marginLeft": "5%"},
                    ),
                    html.Div(
                        [
                            dash_table.DataTable(
                                id="highlight-table",
                                columns=[
                                    {"name": "Heure", "id": "start"},
                                    {"name": "Latitude", "id": "lat"},
                                    {"name": "Longitude", "id": "lon"},
                                    {"name": "Distance (m)", "id": "dis"},
                                    {"name": "Erreur (m)", "id": "err"},
                                    {"name": "Cri", "id": "call"},
                                ],
                                data=[],
                                style_table={"height": "100%", "overflowY": "auto"},
                                style_cell={"textAlign": "center"},
                                style_header={"fontWeight": "bold"},
                            )
                        ],
                        style={"width": "60%", "marginLeft": "20px", "marginRight": "5%", "overflowY": "auto"},
                    ),
                ],
                style={"display": "flex", "height": "250px", "marginTop": "20px", "justifyContent": "space-between"},
            ),
            dcc.Interval(id="startup", interval=250, n_intervals=0, max_intervals=1),
            dcc.Interval(id="interval", interval=1000, n_intervals=0, disabled=True),
        ]
    )

    # NEW: Pause toggle (disable interval + stop streamer advancing)
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

        # Au chargement
        if trig == "startup":
            streamer.start()
            streamer.paused = True
            streamer.finished = False
            return True, True, "PLAY"

        # RESTART
        if trig == "restart-btn":
            streamer.reset()
            streamer.paused = True
            streamer.finished = False
            return True, True, "PLAY"

        # NEW: tick interval -> si fin atteinte, auto-PAUSE + désactiver l'interval
        if trig == "interval":
            if getattr(streamer, "finished", False):
                streamer.paused = True
                return True, True, "PLAY"
            raise dash.exceptions.PreventUpdate

        # PLAY / PAUSE
        paused = not bool(paused)
        streamer.paused = paused
        if not paused:
            streamer.finished = False  # NEW: si on relance, on n'est plus "fin"
        return paused, paused, ("PLAY" if paused else "PAUSE")

    def style_for_status(st):
        if st == "reject_tdoa":
            return "gray", "rgba(0,0,0,0.05)"
        if st == "reject_fusion":
            return "purple", "rgba(0,0,0,0.05)"
        if st == "reject_setup":
            return "orange", "rgba(0,0,0,0.05)"
        return "red", "rgba(0,0,0,0.05)" 

    # Update at each second OR on PREV/NEXT
    @app.callback(
        [
            Output("spectrogram1", "figure"),
            Output("spectrogram2", "figure"),
            Output("map-markers", "children"),
            Output("highlight-table", "data"),
        ],
        [
            Input("startup", "n_intervals"),     # 👈 pour forcer le 1er rendu
            Input("interval", "n_intervals"),
            Input("prev-btn", "n_clicks"),
            Input("next-btn", "n_clicks"),
            Input("restart-btn", "n_clicks"),
        ],
        State("paused-store", "data"),
    )
    def update_visuals(startup_n, n, prev_clicks, next_clicks, restart_clicks, paused):
        ctx = dash.callback_context
        trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "startup"

        # Safety: si jamais
        if not streamer.started:
            streamer.start()

        # Step uniquement si PREV/NEXT
        if trig == "prev-btn":
            streamer.step_chunks(-1)
        elif trig == "next-btn":
            streamer.step_chunks(+1)
        elif trig == "restart-btn":
            pass

        data1, data2, start_time = streamer.get_current_chunk()

        # Playhead audio en secondes (avance en PLAY, recule en PREVIOUS/NEXT)
        with streamer.lock:
            playhead_sec = streamer.index / streamer.sample_rate

        # Paramètres du spectrogramme
        n_fft = 4096
        hop_length = 1024
        sens_db = environment.sensitivity  # dB re 1 V / µPa
        sens_lin = 10 ** (sens_db / 20)  # V / µPa

        # STFT et conversion en dB
        S1 = np.abs(librosa.stft(data1, n_fft=n_fft, hop_length=hop_length))
        S1_uPa = S1 / sens_lin  # µPa
        PSD1 = (S1_uPa**2) / (sr1 / n_fft)  # µPa² / Hz
        PSD1_dB = 10 * np.log10(PSD1 + 1e-24)

        S2 = np.abs(librosa.stft(data2, n_fft=n_fft, hop_length=hop_length))
        S2_uPa = S2 / sens_lin  # µPa
        PSD2 = (S2_uPa**2) / (sr2 / n_fft)  # µPa² / Hz
        PSD2_dB = 10 * np.log10(PSD2 + 1e-24)

        # Axes temps et fréquence
        times = librosa.frames_to_time(np.arange(S1.shape[1]), sr=sr1, hop_length=hop_length)
        times += start_time
        # Temps absolu pour l’axe X
        audio_start_dt = datetime(year, month, day, hour, minute, second)
        times_dt = [audio_start_dt + timedelta(seconds=t) for t in times]
        freqs = librosa.fft_frequencies(sr=sr1, n_fft=n_fft)

        # Filtrage des fréquences
        freq_mask = (freqs >= 250) & (freqs <= 192000)
        PSD1_filtered = PSD1_dB[freq_mask, :]
        PSD2_filtered = PSD2_dB[freq_mask, :]
        freqs_filtered = freqs[freq_mask]

        # Création des figures
        DISPLAY_MODE = "spl"  # "spl" ou "contrast"

        if DISPLAY_MODE == "spl":
            z1 = PSD1_filtered
            z2 = PSD2_filtered
            zmin, zmax = 40, 130
            cbar_title = "PSD [dB re 1 µPa²/Hz]"
        else:
            baseline1 = np.percentile(PSD1_filtered, 20, axis=1, keepdims=True)
            z1 = PSD1_filtered - baseline1
            z1 = np.clip(z1, -5, 25)
            baseline2 = np.percentile(PSD2_filtered, 20, axis=1, keepdims=True)
            z2 = PSD2_filtered - baseline2
            z2 = np.clip(z2, -5, 25)
            zmin, zmax = -5, 25
            cbar_title = "ΔSPL [dB]"

        fig1 = go.Figure(
            data=go.Heatmap(
                z=z1,
                x=times_dt,
                y=freqs_filtered,
                colorscale="cividis",
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(title=dict(text=cbar_title, side="right")),
            )
        )

        fig2 = go.Figure(
            data=go.Heatmap(
                z=z2,
                x=times_dt,
                y=freqs_filtered,
                colorscale="cividis",
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(title=dict(text=cbar_title, side="right")),
            )
        )

        fig1.update_layout(
            title=f"T1 - {audio_path1[-21:-17]}",
            xaxis_title="Heure locale",
            yaxis_title="Fréquence [Hz]"
        )

        fig2.update_layout(
            title=f"T2 - {audio_path2[-21:-17]}",
            xaxis_title="Heure locale",
            yaxis_title="Fréquence [Hz]"
        )

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


        # Surlignages temporels
        current_start = times[0]
        current_end = times[-1]
        current_start_dt = audio_start_dt + timedelta(seconds=current_start)
        current_end_dt   = audio_start_dt + timedelta(seconds=current_end)

        for i, start_timestamp in enumerate(event_timestamps):
            if start_timestamp + event_durations[i] < current_start or start_timestamp > current_end:
                continue

            start_dt = pd.to_datetime(event_times[i]).floor("s").to_pydatetime()
            end_dt   = start_dt + timedelta(seconds=0.96)  # on veut que ça couvre la seconde entière (0.96 pour éviter les chevauchements visuels)
            x0 = max(start_dt, current_start_dt) + timedelta(milliseconds=20)  # petit offset pour éviter les chevauchements visuels
            x1 = min(end_dt, current_end_dt) + timedelta(milliseconds=20)
            if x1 <= x0:
                continue

            st = event_status[i]
            line_col = "green" if st == "ok" else "red"
            fill_col = "rgba(0,0,0,0.05)"

            ct = event_call_types[i]
            if ct == "Whistle":
                y0, y1 = 500, 10000
            else:
                y0, y1 = 20000, 120000

            shape = dict(
                type="rect",
                xref="x", yref="y",
                x0=x0, x1=x1,
                y0=y0, y1=y1,
                line=dict(color=line_col, width=2),
                fillcolor=fill_col,
                layer="above",
            )
            fig1.add_shape(shape)
            fig2.add_shape(shape)

        def _add_solo_rect(fig, timestamp_sec_dt, lab):
            start_dt = timestamp_sec_dt
            end_dt   = start_dt + timedelta(seconds=0.96)  # même logique que pour les events normaux

            # clamp au viewport
            if end_dt < current_start_dt or start_dt > current_end_dt:
                return
            x0 = max(start_dt, current_start_dt) + timedelta(milliseconds=20)  # petit offset pour éviter les chevauchements visuels
            x1 = min(end_dt, current_end_dt) + timedelta(milliseconds=20)
            if x1 <= x0:
                return

            if lab == "Whistle":
                y0, y1 = 500, 10000
            else:  # HFPC
                y0, y1 = 20000, 120000

            fig.add_shape(dict(
                type="rect",
                xref="x", yref="y",
                x0=x0, x1=x1,
                y0=y0, y1=y1,
                line=dict(color="gray", width=2),
                fillcolor="rgba(0,0,0,0.05)",
                layer="above",
            ))

        if detections_dfs and len(detections_dfs) >= 2:
            solo1 = tetra_events[0] - common_events
            solo2 = tetra_events[1] - common_events

            for (ts_sec, lab) in solo1:
                _add_solo_rect(fig1, ts_sec, lab)

            for (ts_sec, lab) in solo2:
                _add_solo_rect(fig2, ts_sec, lab)


        # Détection de l'index actif (basé sur le playhead audio)
        active_index = None
        if highlight_timestamps:
            idx = int(np.searchsorted(np.array(highlight_timestamps), playhead_sec, side="right") - 1)
            if idx >= 0:
                active_index = min(idx, len(highlight_coords) - 1)

        # Carte 
        markers = []
        # Groundtruths : +
        for lat, lon in groundtruths:
            markers.append(
                dl.Marker(
                    position=[lat, lon],
                    zIndexOffset=300,
                    icon=dict(
                        iconUrl="https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-orange.png",
                        iconSize=[25, 41],
                        iconAnchor=[12, 41],
                    ),
                )
            )

        # Tetrahedres : punaises violettes + label T1/T2 (tooltip permanent)
        for idx, (lat, lon) in enumerate(tetrahedrons_coords):
            label = f"T{idx+1}"

            markers.append(
                dl.Marker(
                    position=[lat, lon],
                    zIndexOffset=200,
                    icon=dict(
                        iconUrl="https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-violet.png",
                        iconSize=[25, 41],
                        iconAnchor=[12, 41],
                    ),
                )
            )

            markers.append(
                dl.DivMarker(
                    position=[lat, lon],
                    zIndexOffset=210,
                    interactive=False,
                    iconOptions=dict(
                        html=f"""
                        <div style="
                            position:absolute;
                            left:50%;
                            top:-6px;                    
                            transform:translateX(-50%);
                            width:0; height:0;
                            border-left:6px solid transparent;
                            border-right:6px solid transparent;
                            border-bottom:6px solid rgba(255,255,255,0.95);
                        "></div>
                        <div style="
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            font-weight:700;
                            font-size:13px;
                            color:#6a0dad;
                            background:rgba(255,255,255,0.95);
                            border:1px solid rgba(0,0,0,0.3);
                            border-radius:8px;
                            padding:1px 4px;
                            box-shadow:0 1px 4px rgba(0,0,0,0.2);
                            white-space:nowrap;
                        ">
                            {label}
                        </div>
                        """,
                        className="",
                        iconSize=[30, 20],
                        iconAnchor=[15, -8],
                    ),
                )
            )

        # Historique des vocalises (punaises vertes avec fade-out sur 60 s)
        FADE_SECONDS = 60.0
        t_now = float(playhead_sec)

        # Dummy invisible marker pour forcer Leaflet à rafraîchir chaque seconde de playhead
        # (sinon l'opacité des Markers peut rester figée visuellement jusqu'à un changement structurel)
        markers.append(
            dl.CircleMarker(
                id=f"_tick_{int(t_now)%2}",
                center=[T1_lat, T1_lon],
                radius=1,
                opacity=0.0,
                fillOpacity=0.0,
                color="rgba(0,0,0,0)",
            )
        )

        # Afficher toutes les vocalises déjà "arrivées" (t_i <= t_now) et pas trop vieilles (age <= FADE_SECONDS)
        if highlight_timestamps:
            for i, t_i in enumerate(highlight_timestamps):
                t_i = float(t_i)
                age = t_now - t_i
                if age < 0:
                    continue
                if age > FADE_SECONDS:
                    continue

                opacity = max(0.0, 1.0 - age / FADE_SECONDS)

                lat, lon = highlight_coords[i]
                markers.append(
                    dl.Marker(
                        id=f"green-{i}-{int(t_now)}",
                        position=[lat, lon],
                        icon=dict(
                            iconUrl="https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png",
                            iconSize=[25, 41],
                            iconAnchor=[12, 41],
                        ),
                        opacity=float(opacity),
                        zIndexOffset=2000,
                        riseOnHover=True,
                    )
                )

        # Marqueur actif (vert vif) au-dessus
        #if active_index is not None and active_index < len(highlight_coords):
        #    lat, lon = highlight_coords[active_index]
        #    markers.append(
        #        dl.Marker(
        #            id=f"active-{active_index}-{int(t_now)}",
        #            position=[lat, lon],
        #            icon=dict(
        #                iconUrl="https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png",
        #                iconSize=[25, 41],
        #                iconAnchor=[12, 41],
        #            ),
        #            opacity=1.0,
        #            zIndexOffset=2000,
        #            riseOnHover=True,
        #        )
        #    )


        # Tableau de données
        if active_index is not None and active_index >= 0:
            table_data = []
            for i in range(active_index + 1):
                start_sec = float(highlight_timestamps[i])
                table_data.append(
                    {
                        "start": (audio_start_dt + timedelta(seconds=start_sec)).strftime("%H:%M:%S"),
                        "lat": highlight_coords[i][0],
                        "lon": highlight_coords[i][1],
                        "dis": distances_to_groundtruths[i],
                        "err": round(np.sqrt(_scalar(errors[i][0]) ** 2 + _scalar(errors[i][1]) ** 2), 3),
                        "call": call_types[i],
                    }
                )

            # NEW: ordre inverse (plus récent en haut)
            table_data = table_data[::-1]
        else:
            table_data = []

        return fig1, fig2, markers, table_data

    # --- Écriture CSV après lecture complète (robuste aux longueurs différentes) ---
    output_dir = os.path.join(".", "test_data", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "resultats.csv")

    # sécurité: on borne au minimum des longueurs disponibles
    n_rows = min(
        len(highlight_coords),
        len(errors),
        len(call_types),
        len(distances_to_groundtruths),
        len(highlight_timestamps) if "highlight_timestamps" in locals() else 0,
    )

    # base temporelle absolue
    audio_start_dt = datetime(year, month, day, hour, minute, second)

    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Heure", "Latitude", "Longitude", "Distance (m)", "Erreur (m)", "Cri"])

        for i in range(n_rows):
            # Si un intervalle existe pour i, on l’utilise; sinon on retombe sur le timestamp direct
            if "highlight_intervals" in locals() and i < len(highlight_intervals):
                start_sec = int(highlight_intervals[i][0])
            else:
                start_sec = int(highlight_timestamps[i])

            heure_str = (audio_start_dt + timedelta(seconds=start_sec)).strftime("%H:%M:%S")

            lat, lon = highlight_coords[i]
            distance = float(distances_to_groundtruths[i])
            err_xy = errors[i]
            erreur = round(float(np.sqrt(err_xy[0] ** 2 + err_xy[1] ** 2)), 3)
            cri = call_types[i]

            writer.writerow([heure_str, lat, lon, distance, erreur, cri])

    print(f"✅ Résultats exportés dans {output_file}")
    return app
