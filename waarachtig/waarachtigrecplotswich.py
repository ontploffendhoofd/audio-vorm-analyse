"""
Realtime Authenticiteitsanalyse — met opnamefunctie
=====================================================
Installatievereisten:
    pip install sounddevice numpy matplotlib scipy

Gebruik:
    python realtime_analyse.py

Routing:
    ROUTE = "microfoon"   directe microfooninput
    ROUTE = "blackhole"   BlackHole 2ch (YouTube, Teams, etc.)
    ROUTE = "simulatie"   synthetisch testsignaal

Opname:
    Druk R om opname te starten/stoppen
    Opgeslagen in map: recordings/
    Per sessie:
        YYYY-MM-DD_HH-MM-SS.wav       audio
        YYYY-MM-DD_HH-MM-SS_data.csv  indices (t, C, H, P, W)
        YYYY-MM-DD_HH-MM-SS_plot.png  grafiek van de sessie

Transcriptie:
    Voeg later een kolom 'notitie' toe in het CSV
    of gebruik whisper: pip install openai-whisper

Druk Ctrl+C om te stoppen.
"""

import zlib
import threading
import queue
import os
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.io.wavfile as wf

try:
    import sounddevice as sd
    SOUNDDEVICE_OK = True
except ImportError:
    SOUNDDEVICE_OK = False
    print("sounddevice niet gevonden. Simulatiemodus actief.")


# ══════════════════════════════════════════════════════════════════════════════
# INSTELLINGEN
# ══════════════════════════════════════════════════════════════════════════════

ROUTE       = "microfoon"   # "microfoon" | "blackhole" | "simulatie"
SMOOTH_K    = 15
RECORD_DIR  = "recordings"

SR          = 16000
BLOCK_SIZE  = 512
WINDOW_SEC  = 0.15
WINDOW_SAMP = int(SR * WINDOW_SEC)
HISTORY     = 200
N_BINS      = 32

# ══════════════════════════════════════════════════════════════════════════════


# ── Opnamestatus ──────────────────────────────────────────────────────────────

class Recorder:
    def __init__(self):
        self.recording   = False
        self.audio_buf   = []
        self.data_rows   = []   # [t_sec, C, H, P, W]
        self.session_id  = None

    def start(self):
        if self.recording:
            return
        self.recording  = True
        self.audio_buf  = []
        self.data_rows  = []
        self.session_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(RECORD_DIR, exist_ok=True)
        print(f"\n● Opname gestart  [{self.session_id}]")

    def add_audio(self, chunk):
        if self.recording:
            self.audio_buf.append(chunk.copy())

    def add_data(self, t_sec, C, H, P, W):
        if self.recording:
            self.data_rows.append([round(t_sec, 4), round(C, 5),
                                   round(H, 5), round(P, 5), round(W, 5)])

    def stop(self, history_W, history_C, history_H, history_P):
        if not self.recording:
            return
        self.recording = False
        sid = self.session_id
        print(f"\n■ Opname gestopt  [{sid}]  — opslaan...")

        # Audio
        wav_path = os.path.join(RECORD_DIR, f"{sid}.wav")
        if self.audio_buf:
            audio_arr = np.concatenate(self.audio_buf).astype(np.float32)
            # scipy wil int16 of float32
            wf.write(wav_path, SR, audio_arr)
            print(f"  Audio:  {wav_path}  ({round(len(audio_arr)/SR, 2)} sec)")

        # CSV
        csv_path = os.path.join(RECORD_DIR, f"{sid}_data.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t_sec", "C", "H", "P", "W", "notitie"])
            for row in self.data_rows:
                writer.writerow(row + [""])   # lege notitie-kolom
        print(f"  Data:   {csv_path}  ({len(self.data_rows)} frames)")

        # Plot
        plot_path = os.path.join(RECORD_DIR, f"{sid}_plot.png")
        self._save_plot(plot_path)
        print(f"  Plot:   {plot_path}")
        print(f"\n  Voeg later notities toe in de CSV kolom 'notitie'.")
        print(f"  Of transcribeer audio met: whisper {wav_path}\n")

    def _save_plot(self, path):
        if not self.data_rows:
            return
        arr   = np.array(self.data_rows)
        t     = arr[:, 0]
        C, H, P, W = arr[:,1], arr[:,2], arr[:,3], arr[:,4]

        def sm(v):
            k = min(SMOOTH_K, len(v))
            return np.convolve(v, np.ones(k)/k, mode='same')

        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Sessie {self.session_id}", fontsize=11)

        for ax, vals, label, color in [
            (axes[0], C, "Compressibility", "#e74c3c"),
            (axes[1], H, "Entropy",         "#3498db"),
            (axes[2], P, "Phase",           "#9b59b6"),
            (axes[3], W, "Authenticiteit W","#f1c40f"),
        ]:
            ax.plot(t, vals,   color=color, linewidth=0.5, alpha=0.3)
            ax.plot(t, sm(vals), color=color, linewidth=1.8)
            ax.set_ylabel(label, fontsize=8)
            ax.grid(True, alpha=0.2)
            ax.axhline(np.mean(vals), color=color,
                       linewidth=0.6, linestyle="--", alpha=0.5)

        axes[-1].set_xlabel("tijd (seconden)", fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


recorder = Recorder()
record_time = [0.0]   # looptijd opname in seconden


# ── Routing ───────────────────────────────────────────────────────────────────

def resolve_device():
    if ROUTE == "blackhole":
        if not SOUNDDEVICE_OK:
            return None
        for d in sd.query_devices():
            if "BlackHole" in d["name"] and d["max_input_channels"] > 0:
                print(f"BlackHole gevonden: {d['name']}")
                return d["name"]
        print("BlackHole niet gevonden — standaard microfoon.")
        return None
    return None


# ── Indexfuncties ─────────────────────────────────────────────────────────────

def quantize(sig, n=N_BINS):
    m = np.max(np.abs(sig))
    if m < 1e-12:
        return np.zeros(len(sig), dtype=np.int16)
    x = sig / m
    bins = np.linspace(-1, 1, n + 1)
    q = np.digitize(x, bins) - 1
    return np.clip(q, 0, n - 1).astype(np.int16)

def compute_C(sig):
    q = quantize(sig)
    raw = q.tobytes()
    if not raw: return 0.0
    return float(np.clip(1 - len(zlib.compress(raw)) / len(raw), 0, 1))

def compute_H(sig, n=N_BINS):
    q = quantize(sig, n)
    counts = np.bincount(q, minlength=n).astype(float)
    total = counts.sum()
    if total == 0: return 0.0
    p = counts[counts > 0] / total
    return float(np.clip(-np.sum(p * np.log(p)) / np.log(n), 0, 1))

def compute_P(sig, scale=64):
    if len(sig) < 2 * scale: return 0.0
    c = np.corrcoef(sig[:scale], sig[scale:2*scale])[0, 1]
    return float(np.clip(abs(c) if not np.isnan(c) else 0.0, 0, 1))

def compute_indices(frame):
    C = compute_C(frame)
    H = compute_H(frame)
    P = compute_P(frame)
    W = (H * P) / (C + 1e-6)
    return C, H, P, W

def smooth(values, k=SMOOTH_K):
    if k <= 1: return values.copy()
    return np.convolve(values, np.ones(k)/k, mode='same')


# ── Audiobuffer ───────────────────────────────────────────────────────────────

audio_buffer = np.zeros(WINDOW_SAMP, dtype=np.float32)
data_queue   = queue.Queue()
lock         = threading.Lock()

def audio_callback(indata, frames, time, status):
    mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
    chunk = mono.copy()
    data_queue.put(chunk)
    recorder.add_audio(chunk)

def buffer_updater():
    global audio_buffer
    while True:
        try:
            chunk = data_queue.get(timeout=1.0)
            with lock:
                audio_buffer = np.roll(audio_buffer, -len(chunk))
                audio_buffer[-len(chunk):] = chunk
        except queue.Empty:
            continue


# ── Simulatie ─────────────────────────────────────────────────────────────────

_sim_t = [0.0]

def simulate_frame():
    t = np.linspace(_sim_t[0], _sim_t[0] + WINDOW_SEC, WINDOW_SAMP)
    _sim_t[0] += WINDOW_SEC / 10
    phase = np.sin(2 * np.pi * 0.3 * _sim_t[0])
    carrier = 180 + 40 * phase
    sig = (
        0.5 * np.sin(2 * np.pi * carrier * t) +
        0.2 * np.sin(2 * np.pi * (carrier * 2.1) * t) +
        0.1 * np.random.randn(len(t))
    )
    return sig.astype(np.float32)


# ── Grafiek ───────────────────────────────────────────────────────────────────

history_C = np.zeros(HISTORY)
history_H = np.zeros(HISTORY)
history_P = np.zeros(HISTORY)
history_W = np.zeros(HISTORY)
x_axis    = np.arange(HISTORY)
frame_time = [0.0]

route_label = {
    "microfoon": "Microfoon",
    "blackhole": "BlackHole",
    "simulatie": "Simulatie",
}.get(ROUTE, ROUTE)

fig, axes = plt.subplots(4, 1, figsize=(12, 9))
fig.patch.set_facecolor("#1a1a2e")

COLORS = {"C": "#e74c3c", "H": "#3498db", "P": "#9b59b6", "W": "#f1c40f"}
LABELS = {
    "C": "Compressibility",
    "H": "Entropy",
    "P": "Phase",
    "W": "Authenticiteit W",
}

lines_raw = {}
lines_sm  = {}
fill_ref  = [None]
rec_line  = [None]   # rode verticale lijn als opname-indicator

for ax, (key, color) in zip(axes, COLORS.items()):
    ax.set_facecolor("#0f0f1a")
    ax.set_ylim(-0.05, 1.55 if key == "W" else 1.05)
    ax.set_xlim(0, HISTORY)
    ax.set_ylabel(LABELS[key], fontsize=8, color="white")
    ax.tick_params(colors="gray", labelsize=7)
    ax.grid(True, alpha=0.15, color="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    lr, = ax.plot(x_axis, np.zeros(HISTORY), color=color,
                  linewidth=0.5, alpha=0.2)
    ls, = ax.plot(x_axis, np.zeros(HISTORY), color=color,
                  linewidth=2.0, alpha=0.95)
    lines_raw[key] = lr
    lines_sm[key]  = ls

axes[-1].set_xlabel("tijd →  |  R = start/stop opname", fontsize=8, color="gray")

# Opname-indicator tekst rechtsboven
rec_text = fig.text(0.98, 0.97, "", color="#e74c3c",
                    fontsize=11, ha="right", va="top", fontweight="bold")


def on_key(event):
    if event.key == "r":
        if recorder.recording:
            recorder.stop(history_W, history_C, history_H, history_P)
        else:
            recorder.start()
            record_time[0] = frame_time[0]

fig.canvas.mpl_connect("key_press_event", on_key)


def update(frame_num):
    global history_C, history_H, history_P, history_W, fill_ref

    use_sim = (ROUTE == "simulatie") or (not SOUNDDEVICE_OK)

    if use_sim:
        current_frame = simulate_frame()
        recorder.add_audio(current_frame)
    else:
        with lock:
            current_frame = audio_buffer.copy()

    C, H, P, W = compute_indices(current_frame)
    frame_time[0] += WINDOW_SEC / 10

    # Voeg data toe aan opname
    recorder.add_data(frame_time[0], C, H, P, W)

    history_C = np.roll(history_C, -1); history_C[-1] = C
    history_H = np.roll(history_H, -1); history_H[-1] = H
    history_P = np.roll(history_P, -1); history_P[-1] = P
    history_W = np.roll(history_W, -1); history_W[-1] = W

    for key, raw, his in [
        ("C", history_C, history_C),
        ("H", history_H, history_H),
        ("P", history_P, history_P),
        ("W", history_W, history_W),
    ]:
        lines_raw[key].set_ydata(raw)
        lines_sm[key].set_ydata(smooth(his))

    if fill_ref[0] is not None:
        fill_ref[0].remove()
    fill_ref[0] = axes[3].fill_between(
        x_axis, smooth(history_W), alpha=0.15, color=COLORS["W"]
    )

    # Opname-indicator
    if recorder.recording:
        elapsed = round(frame_time[0] - record_time[0], 1)
        rec_text.set_text(f"● REC  {elapsed}s")
    else:
        rec_text.set_text("")

    fig.suptitle(
        f"[{route_label}]  C={C:.3f}  H={H:.3f}  P={P:.3f}  W={W:.3f}"
        f"  |  smooth k={SMOOTH_K}  |  R = opname",
        fontsize=9, color="white"
    )

    return list(lines_raw.values()) + list(lines_sm.values())


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    use_sim = (ROUTE == "simulatie") or (not SOUNDDEVICE_OK)

    if not use_sim:
        device = resolve_device()
        print(f"Routing: {route_label}")
        print("Druk R om opname te starten/stoppen.")
        print("Druk Ctrl+C om te stoppen.\n")
        t = threading.Thread(target=buffer_updater, daemon=True)
        t.start()
        stream = sd.InputStream(
            samplerate=SR, channels=1, blocksize=BLOCK_SIZE,
            callback=audio_callback, device=device,
        )
        stream.start()
    else:
        print(f"Modus: {route_label}")
        print("Druk R om opname te starten/stoppen.\n")
        if not SOUNDDEVICE_OK:
            print("pip install sounddevice  voor echte meting\n")

    ani = animation.FuncAnimation(
        fig, update, interval=80, blit=False, cache_frame_data=False,
    )

    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        if recorder.recording:
            recorder.stop(history_W, history_C, history_H, history_P)
        if not use_sim and SOUNDDEVICE_OK:
            stream.stop()
            stream.close()
        print("Gestopt.")