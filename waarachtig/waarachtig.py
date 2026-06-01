"""
Realtime Authenticiteitsanalyse
================================
Installatievereisten:
    pip install sounddevice numpy matplotlib scipy

Gebruik:
    python realtime_analyse.py

Druk Ctrl+C om te stoppen.
"""

import zlib
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    import sounddevice as sd
    SOUNDDEVICE_OK = True
except ImportError:
    SOUNDDEVICE_OK = False
    print("sounddevice niet gevonden. Simulatiemodus actief.")


# ── Instellingen ──────────────────────────────────────────────────────────────

SR          = 16000       # sample rate (16kHz genoeg voor spraak)
BLOCK_SIZE  = 512         # samples per callback (~32ms bij 16kHz)
WINDOW_SEC  = 0.15        # analysevenster in seconden
WINDOW_SAMP = int(SR * WINDOW_SEC)
HISTORY     = 200         # aantal frames in de grafiek

N_BINS      = 32          # quantisatiebins


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
    if not raw:
        return 0.0
    c = zlib.compress(raw)
    return float(np.clip(1 - len(c) / len(raw), 0, 1))


def compute_H(sig, n=N_BINS):
    q = quantize(sig, n)
    counts = np.bincount(q, minlength=n).astype(float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    h = -np.sum(p * np.log(p))
    return float(np.clip(h / np.log(n), 0, 1))


def compute_P(sig, scale=64):
    if len(sig) < 2 * scale:
        return 0.0
    a = sig[:scale]
    b = sig[scale:2 * scale]
    c = np.corrcoef(a, b)[0, 1]
    return float(np.clip(abs(c) if not np.isnan(c) else 0.0, 0, 1))


def compute_indices(frame):
    C = compute_C(frame)
    H = compute_H(frame)
    P = compute_P(frame)
    W = (H * P) / (C + 1e-6)   # I weggelaten voor snelheid
    return C, H, P, W


# ── Audiobuffer ───────────────────────────────────────────────────────────────

audio_buffer = np.zeros(WINDOW_SAMP, dtype=np.float32)
data_queue   = queue.Queue()
lock         = threading.Lock()


def audio_callback(indata, frames, time, status):
    """Wordt aangeroepen door sounddevice voor elk blok audio."""
    mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
    data_queue.put(mono.copy())


def buffer_updater():
    """Achtergrondthread: houdt rollend venster bij."""
    global audio_buffer
    while True:
        try:
            chunk = data_queue.get(timeout=1.0)
            with lock:
                audio_buffer = np.roll(audio_buffer, -len(chunk))
                audio_buffer[-len(chunk):] = chunk
        except queue.Empty:
            continue


# ── Simulatie (als sounddevice niet beschikbaar is) ───────────────────────────

_sim_t = [0.0]

def simulate_frame():
    """Genereert een synthetisch spraakachtig signaal voor demo."""
    t = np.linspace(_sim_t[0], _sim_t[0] + WINDOW_SEC, WINDOW_SAMP)
    _sim_t[0] += WINDOW_SEC / 10

    # wisselend: soms rijker, soms vlak
    phase = np.sin(2 * np.pi * 0.3 * _sim_t[0])
    carrier = 180 + 40 * phase
    sig = (
        0.5 * np.sin(2 * np.pi * carrier * t) +
        0.2 * np.sin(2 * np.pi * (carrier * 2.1) * t) +
        0.1 * np.random.randn(len(t))
    )
    return sig.astype(np.float32)


# ── Grafiekdata ───────────────────────────────────────────────────────────────

history_C = np.zeros(HISTORY)
history_H = np.zeros(HISTORY)
history_P = np.zeros(HISTORY)
history_W = np.zeros(HISTORY)
x_axis    = np.arange(HISTORY)


# ── Plot opzetten ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 1, figsize=(11, 8))
fig.suptitle("Realtime Authenticiteitsanalyse", fontsize=12)
fig.patch.set_facecolor("#1a1a2e")

COLORS = {
    "C": "#e74c3c",
    "H": "#3498db",
    "P": "#9b59b6",
    "W": "#f1c40f",
}

LABELS = {
    "C": "Compressibility  (omhulsel ↑)",
    "H": "Entropy  (variatie ↑)",
    "P": "Phase  (coherentie ↑)",
    "W": "Authenticiteit W",
}

lines = {}
for ax, (key, color) in zip(axes, COLORS.items()):
    ax.set_facecolor("#0f0f1a")
    ax.set_ylim(-0.05, 1.55 if key == "W" else 1.05)
    ax.set_xlim(0, HISTORY)
    ax.set_ylabel(LABELS[key], fontsize=8, color="white")
    ax.tick_params(colors="gray", labelsize=7)
    ax.grid(True, alpha=0.15, color="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    line, = ax.plot(x_axis, np.zeros(HISTORY), color=color, linewidth=1.5)
    lines[key] = line
    # gemiddeldenlijn (wordt bijgewerkt)
    lines[f"{key}_mean"] = ax.axhline(0, color=color, linewidth=0.7,
                                       linestyle="--", alpha=0.4)

axes[-1].set_xlabel("tijd →", fontsize=8, color="gray")

# W heeft een opvulling
fill_ref = [None]


def update(frame_num):
    global history_C, history_H, history_P, history_W, fill_ref

    # Haal frame op
    if SOUNDDEVICE_OK:
        with lock:
            current_frame = audio_buffer.copy()
    else:
        current_frame = simulate_frame()

    # Bereken indices
    C, H, P, W = compute_indices(current_frame)

    # Rol geschiedenis
    history_C = np.roll(history_C, -1); history_C[-1] = C
    history_H = np.roll(history_H, -1); history_H[-1] = H
    history_P = np.roll(history_P, -1); history_P[-1] = P
    history_W = np.roll(history_W, -1); history_W[-1] = W

    # Update lijnen
    lines["C"].set_ydata(history_C)
    lines["H"].set_ydata(history_H)
    lines["P"].set_ydata(history_P)
    lines["W"].set_ydata(history_W)

    # Update gemiddelden
    for key, hist in [("C", history_C), ("H", history_H),
                      ("P", history_P), ("W", history_W)]:
        m = np.mean(hist[hist > 0]) if np.any(hist > 0) else 0
        lines[f"{key}_mean"].set_ydata([m, m])

    # W opvulling vernieuwen
    if fill_ref[0] is not None:
        fill_ref[0].remove()
    fill_ref[0] = axes[3].fill_between(
        x_axis, history_W,
        alpha=0.15, color=COLORS["W"]
    )

    # Live waarden in titel
    fig.suptitle(
        f"Realtime Analyse  │  C={C:.3f}  H={H:.3f}  P={P:.3f}  W={W:.3f}",
        fontsize=11, color="white"
    )

    return list(lines.values())


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if SOUNDDEVICE_OK:
        print("Microfoon actief. Spreek in de microfoon.")
        print("Druk Ctrl+C om te stoppen.\n")

        # Start audiobuffer-thread
        t = threading.Thread(target=buffer_updater, daemon=True)
        t.start()

        # Start microfoon
        stream = sd.InputStream(
            samplerate=SR,
            channels=1,
            blocksize=BLOCK_SIZE,
            callback=audio_callback,
        )
        stream.start()
    else:
        print("Simulatiemodus — geen microfoon vereist.")
        print("Installeer sounddevice voor echte meting:\n")
        print("    pip install sounddevice\n")

    ani = animation.FuncAnimation(
        fig,
        update,
        interval=80,        # ~12 updates per seconde
        blit=False,
        cache_frame_data=False,
    )

    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        if SOUNDDEVICE_OK:
            stream.stop()
            stream.close()
        print("Gestopt.")