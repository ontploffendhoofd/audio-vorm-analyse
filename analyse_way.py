from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import zlib
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


# =========================================================
# 1. Data containers
# =========================================================

@dataclass
class InputFrame:
    t_index: int
    audio: np.ndarray
    sample_rate: int
    text_tokens: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamState:
    stream_id: str
    signal: np.ndarray
    formants: Optional[np.ndarray] = None
    lpc_coeffs: Optional[np.ndarray] = None
    residual: Optional[np.ndarray] = None
    energy: Optional[float] = None
    phase_features: Dict[str, float] = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class RelationState:
    stream_a: str
    stream_b: str
    phase_diff: Dict[str, float] = field(default_factory=dict)
    counterpoint_scores: Dict[str, float] = field(default_factory=dict)
    fugue_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class SystemState:
    t_index: int
    streams: Dict[str, StreamState]
    relations: List[RelationState]
    latent_state: Dict[str, float]
    prediction_error: Dict[str, float] = field(default_factory=dict)
    indices: Dict[str, float] = field(default_factory=dict)


# =========================================================
# 2. Input / framing
# =========================================================

class InputBuffer:
    def __init__(self, frame_size: int, hop_size: int, sample_rate: int) -> None:
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate

    def make_frames(self, audio: np.ndarray) -> List[InputFrame]:
        frames: List[InputFrame] = []
        if audio.ndim != 1:
            raise ValueError("Alleen mono audio verwacht.")
        t_index = 0
        for start in range(0, max(1, len(audio) - self.frame_size + 1), self.hop_size):
            end = start + self.frame_size
            chunk = audio[start:end]
            if len(chunk) < self.frame_size:
                pad = np.zeros(self.frame_size - len(chunk))
                chunk = np.concatenate([chunk, pad])
            frames.append(InputFrame(t_index=t_index, audio=chunk, sample_rate=self.sample_rate))
            t_index += 1
        return frames


# =========================================================
# 3. Stream decomposition
# =========================================================

class StreamDecomposer:
    def __init__(self, n_streams: int = 4) -> None:
        self.n_streams = n_streams

    def decompose(self, frame: InputFrame) -> Dict[str, StreamState]:
        audio = frame.audio
        streams: Dict[str, StreamState] = {}

        spectrum = np.fft.rfft(audio)
        n_bins = len(spectrum)
        edges = np.linspace(0, n_bins, self.n_streams + 1, dtype=int)

        for i in range(self.n_streams):
            masked = np.zeros_like(spectrum)
            masked[edges[i]:edges[i + 1]] = spectrum[edges[i]:edges[i + 1]]
            recon = np.fft.irfft(masked, n=len(audio))
            stream_id = f"stream_{i}"
            streams[stream_id] = StreamState(stream_id=stream_id, signal=recon)

        return streams


# =========================================================
# 4. Per-stream analyzers
# =========================================================

class FormantAnalyzer:
    def analyze(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)

        if len(spectrum) < 3:
            return np.array([])
        peak_idx = np.argsort(spectrum)[-3:]
        peak_freqs = np.sort(freqs[peak_idx])
        return peak_freqs


class LPCAnalyzer:
    def __init__(self, order: int = 8) -> None:
        self.order = order

    def analyze(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(signal, dtype=float)
        if len(x) <= self.order + 1:
            coeffs = np.zeros(self.order)
            residual = x.copy()
            return coeffs, residual

        X = []
        y = []
        for t in range(self.order, len(x)):
            X.append(x[t - self.order:t][::-1])
            y.append(x[t])
        X_mat = np.vstack(X)
        y_vec = np.array(y)

        coeffs, *_ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
        pred = X_mat @ coeffs
        residual_core = y_vec - pred

        residual = np.zeros_like(x)
        residual[self.order:] = residual_core
        return coeffs, residual


class EnergyAnalyzer:
    def analyze(self, signal: np.ndarray) -> float:
        return float(np.mean(signal ** 2))


# =========================================================
# 5. Between-stream analyzers
# =========================================================

class PhaseAnalyzer:
    def __init__(self, scales: List[int] | None = None) -> None:
        self.scales = scales if scales is not None else [16, 32, 64]

    def analyze_pair(self, a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for scale in self.scales:
            aa = a[:scale] if len(a) >= scale else a
            bb = b[:scale] if len(b) >= scale else b
            if len(aa) == 0 or len(bb) == 0:
                out[f"phase_scale_{scale}"] = 0.0
                continue

            corr = np.correlate(aa, bb, mode="full")
            lag = int(np.argmax(corr) - (len(bb) - 1))
            out[f"phase_scale_{scale}"] = float(lag)
        return out


class CounterpointAnalyzer:
    def analyze_pair(self, a: StreamState, b: StreamState) -> Dict[str, float]:
        same_energy = 1.0 - abs((a.energy or 0.0) - (b.energy or 0.0))
        corr = float(np.corrcoef(a.signal, b.signal)[0, 1]) if len(a.signal) > 1 else 0.0
        if np.isnan(corr):
            corr = 0.0

        return {
            "convergence": max(0.0, corr),
            "divergence": max(0.0, -corr),
            "independence": 1.0 - min(1.0, abs(corr)),
            "balance": max(0.0, same_energy),
        }


class FugueAnalyzer:
    def analyze_pair(self, a: StreamState, b: StreamState) -> Dict[str, float]:
        sig_a = a.signal
        sig_b = b.signal

        rev_b = sig_b[::-1]
        inv_b = -sig_b

        def sim(x: np.ndarray, y: np.ndarray) -> float:
            if len(x) != len(y):
                n = min(len(x), len(y))
                x = x[:n]
                y = y[:n]
            if len(x) < 2:
                return 0.0
            c = np.corrcoef(x, y)[0, 1]
            return 0.0 if np.isnan(c) else float(c)

        return {
            "retrograde": sim(sig_a, rev_b),
            "inversion": sim(sig_a, inv_b),
            "parallel": sim(np.diff(sig_a), np.diff(sig_b)) if len(sig_a) > 2 else 0.0,
        }


# =========================================================
# 6. State integration
# =========================================================

def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


class StateIntegrator:
    def __init__(self, learning_rate: float = 0.05) -> None:
        self.learning_rate = learning_rate

    def update_weights(self, streams: Dict[str, StreamState]) -> None:
        for stream in streams.values():
            residual_score = float(np.mean(np.abs(stream.residual))) if stream.residual is not None else 0.0
            energy_score = stream.energy or 0.0
            delta = 0.1 * energy_score - 0.1 * residual_score
            stream.weight = max(0.0, stream.weight + self.learning_rate * delta)

    def build_latent_state(
        self,
        streams: Dict[str, StreamState],
        relations: List[RelationState],
    ) -> Dict[str, float]:
        if not streams:
            return {"resonance_index": 0.0, "predictability_index": 0.0, "phase_index": 0.0, "structure_index": 0.0}

        resonance_values = []
        predict_values = []
        for s in streams.values():
            if s.formants is not None and len(s.formants) > 0:
                resonance_values.append(float(np.mean(s.formants)))
            if s.residual is not None:
                predict_values.append(float(np.mean(np.abs(s.residual))))

        phase_values = []
        structure_values = []
        for rel in relations:
            phase_values.extend(rel.phase_diff.values())
            structure_values.extend(rel.counterpoint_scores.values())
            structure_values.extend(rel.fugue_scores.values())

        latent_state = {
            "resonance_index": float(np.mean(resonance_values)) if resonance_values else 0.0,
            "predictability_index": float(np.mean(predict_values)) if predict_values else 0.0,
            "phase_index": float(np.mean(phase_values)) if phase_values else 0.0,
            "structure_index": float(np.mean(structure_values)) if structure_values else 0.0,
        }
        return latent_state


# =========================================================
# 7. Predictor
# =========================================================

class Predictor:
    def predict_stream(self, stream: StreamState) -> np.ndarray:
        if stream.lpc_coeffs is None or len(stream.lpc_coeffs) == 0:
            return stream.signal.copy()

        x = stream.signal
        p = len(stream.lpc_coeffs)
        pred = np.zeros_like(x)

        for t in range(p, len(x)):
            pred[t] = np.dot(stream.lpc_coeffs, x[t - p:t][::-1])

        return pred

    def prediction_error(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        n = min(len(actual), len(predicted))
        if n == 0:
            return 0.0
        return float(np.mean(np.abs(actual[:n] - predicted[:n])))


# =========================================================
# 8. Index functions
# =========================================================

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    max_abs = np.max(np.abs(x)) if len(x) else 0.0
    if max_abs < 1e-12:
        return np.zeros_like(x)
    return x / max_abs


def quantize_signal(signal: np.ndarray, n_bins: int = 32) -> np.ndarray:
    x = normalize_signal(signal)
    bins = np.linspace(-1.0, 1.0, n_bins + 1)
    q = np.digitize(x, bins) - 1
    q = np.clip(q, 0, n_bins - 1)
    return q.astype(np.int16)


def compute_compressibility_index(signal: np.ndarray, n_bins: int = 32) -> float:
    q = quantize_signal(signal, n_bins=n_bins)
    raw_bytes = q.tobytes()
    if len(raw_bytes) == 0:
        return 0.0
    compressed = zlib.compress(raw_bytes)
    c = 1.0 - (len(compressed) / len(raw_bytes))
    return float(np.clip(c, 0.0, 1.0))


def compute_entropy_index(signal: np.ndarray, n_bins: int = 32) -> float:
    q = quantize_signal(signal, n_bins=n_bins)
    if len(q) == 0:
        return 0.0

    counts = np.bincount(q, minlength=n_bins).astype(float)
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0

    h = -np.sum(probs * np.log(probs))
    h_max = np.log(n_bins)
    return float(np.clip(h / h_max, 0.0, 1.0))


def _phase_lag_for_scale(a: np.ndarray, b: np.ndarray, scale: int) -> float:
    aa = a[:scale] if len(a) >= scale else a
    bb = b[:scale] if len(b) >= scale else b
    if len(aa) < 2 or len(bb) < 2:
        return 0.0

    corr = np.correlate(aa, bb, mode="full")
    lag = int(np.argmax(corr) - (len(bb) - 1))
    return float(np.clip(lag / max(scale, 1), -1.0, 1.0))


def compute_multiscale_phase_index(
    streams: dict[str, StreamState],
    scales: list[int] | None = None,
) -> float:
    if scales is None:
        scales = [16, 32, 64]

    stream_list = list(streams.values())
    if len(stream_list) < 2:
        return 0.0

    coherences: list[float] = []

    for i in range(len(stream_list)):
        for j in range(i + 1, len(stream_list)):
            a = stream_list[i].signal
            b = stream_list[j].signal

            pair_scores = []
            for scale in scales:
                lag = _phase_lag_for_scale(a, b, scale)
                pair_scores.append(1.0 - abs(lag))

            coherences.append(safe_mean(pair_scores))

    return float(np.clip(safe_mean(coherences), 0.0, 1.0))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    aa = a[:n]
    bb = b[:n]
    c = np.corrcoef(aa, bb)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def compute_structural_independence_index(
    streams: dict[str, StreamState],
    relations: list[RelationState] | None = None,
) -> float:
    stream_list = list(streams.values())
    if len(stream_list) < 2:
        return 0.0

    pair_independence: list[float] = []
    rel_lookup: dict[tuple[str, str], RelationState] = {}

    if relations is not None:
        for rel in relations:
            key = tuple(sorted((rel.stream_a, rel.stream_b)))
            rel_lookup[key] = rel

    for i in range(len(stream_list)):
        for j in range(i + 1, len(stream_list)):
            s1 = stream_list[i]
            s2 = stream_list[j]

            corr = abs(_safe_corr(s1.signal, s2.signal))
            similarity = corr

            key = tuple(sorted((s1.stream_id, s2.stream_id)))
            rel = rel_lookup.get(key)

            if rel is not None:
                cp = safe_mean(list(rel.counterpoint_scores.values()))
                fg = safe_mean(list(rel.fugue_scores.values()))
                similarity = np.clip(0.5 * corr + 0.25 * abs(cp) + 0.25 * abs(fg), 0.0, 1.0)

            independence = 1.0 - similarity
            pair_independence.append(independence)

    return float(np.clip(safe_mean(pair_independence), 0.0, 1.0))


def compute_state_indices(state: SystemState) -> dict[str, float]:
    compress_vals = []
    entropy_vals = []

    for stream in state.streams.values():
        compress_vals.append(compute_compressibility_index(stream.signal))
        entropy_vals.append(compute_entropy_index(stream.signal))

    C = safe_mean(compress_vals)
    H = safe_mean(entropy_vals)
    P = compute_multiscale_phase_index(state.streams)
    I = compute_structural_independence_index(state.streams, state.relations)
    W = (H * I) / (C + 1e-6)

    return {
        "compressibility": C,
        "entropy": H,
        "phase": P,
        "independence": I,
        "authenticity_index": W,
    }


# =========================================================
# 9. Full pipeline
# =========================================================

class MultitimbralAnalysisSystem:
    def __init__(
        self,
        frame_size: int = 1024,
        hop_size: int = 512,
        sample_rate: int = 16000,
        n_streams: int = 4,
        lpc_order: int = 8,
    ) -> None:
        self.buffer = InputBuffer(frame_size=frame_size, hop_size=hop_size, sample_rate=sample_rate)
        self.decomposer = StreamDecomposer(n_streams=n_streams)
        self.formant_analyzer = FormantAnalyzer()
        self.lpc_analyzer = LPCAnalyzer(order=lpc_order)
        self.energy_analyzer = EnergyAnalyzer()
        self.phase_analyzer = PhaseAnalyzer()
        self.counterpoint_analyzer = CounterpointAnalyzer()
        self.fugue_analyzer = FugueAnalyzer()
        self.integrator = StateIntegrator()
        self.predictor = Predictor()

    def process_frame(self, frame: InputFrame) -> SystemState:
        streams = self.decomposer.decompose(frame)

        for stream in streams.values():
            stream.formants = self.formant_analyzer.analyze(stream.signal, frame.sample_rate)
            stream.lpc_coeffs, stream.residual = self.lpc_analyzer.analyze(stream.signal)
            stream.energy = self.energy_analyzer.analyze(stream.signal)

        stream_list = list(streams.values())
        relations: List[RelationState] = []

        for i in range(len(stream_list)):
            for j in range(i + 1, len(stream_list)):
                a = stream_list[i]
                b = stream_list[j]

                rel = RelationState(stream_a=a.stream_id, stream_b=b.stream_id)
                rel.phase_diff = self.phase_analyzer.analyze_pair(a.signal, b.signal)
                rel.counterpoint_scores = self.counterpoint_analyzer.analyze_pair(a, b)
                rel.fugue_scores = self.fugue_analyzer.analyze_pair(a, b)
                relations.append(rel)

        self.integrator.update_weights(streams)
        latent_state = self.integrator.build_latent_state(streams, relations)

        prediction_error: Dict[str, float] = {}
        for stream in streams.values():
            pred = self.predictor.predict_stream(stream)
            prediction_error[stream.stream_id] = self.predictor.prediction_error(stream.signal, pred)

        state = SystemState(
            t_index=frame.t_index,
            streams=streams,
            relations=relations,
            latent_state=latent_state,
            prediction_error=prediction_error,
        )

        state.indices = compute_state_indices(state)
        return state

    def process_audio(self, audio: np.ndarray) -> List[SystemState]:
        frames = self.buffer.make_frames(audio)
        states: List[SystemState] = []
        for frame in frames:
            states.append(self.process_frame(frame))
        return states


# =========================================================
# 10. Plotting
# =========================================================

def plot_four_indices(states: list[SystemState], output_path: str | None = None) -> None:
    if not states:
        raise ValueError("Geen states om te plotten.")

    t = [s.t_index for s in states]
    C = [s.indices.get("compressibility", 0.0) for s in states]
    H = [s.indices.get("entropy", 0.0) for s in states]
    P = [s.indices.get("phase", 0.0) for s in states]
    I = [s.indices.get("independence", 0.0) for s in states]
    W = [s.indices.get("authenticity_index", 0.0) for s in states]

    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(t, C, color="darkred")
    axes[0].set_ylabel("Compressie")

    axes[1].plot(t, H, color="navy")
    axes[1].set_ylabel("Entropie")

    axes[2].plot(t, P, color="purple")
    axes[2].set_ylabel("Fase")

    axes[3].plot(t, I, color="darkgreen")
    axes[3].set_ylabel("Onafhank.")

    axes[4].plot(t, W, color="black")
    axes[4].set_ylabel("Waaracht.")
    axes[4].set_xlabel("frame")

    fig.suptitle("Vier kernindices + waarachtigheidsindex")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


# =========================================================
# 11. WAV analysis entry point
# =========================================================

def load_audio_mono(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    return audio, sr


def analyze_wav_file(
    wav_path: str,
    output_plot_path: str = "analysis_indices.png",
    frame_size: int = 1024,
    hop_size: int = 512,
    n_streams: int = 4,
    lpc_order: int = 8,
) -> List[SystemState]:
    audio, sr = load_audio_mono(wav_path)

    system = MultitimbralAnalysisSystem(
        frame_size=frame_size,
        hop_size=hop_size,
        sample_rate=sr,
        n_streams=n_streams,
        lpc_order=lpc_order,
    )

    states = system.process_audio(audio)
    plot_four_indices(states, output_plot_path)

    return states


# =========================================================
# 12. Example usage
# =========================================================

if __name__ == "__main__":
    wav_path = "jouw_fragment.wav"  # vervang dit
    states = analyze_wav_file(
        wav_path=wav_path,
        output_plot_path="analysis_indices.png",
        frame_size=1024,
        hop_size=512,
        n_streams=4,
        lpc_order=8,
    )

    print(f"Aantal frames: {len(states)}")
    if states:
        last = states[-1]
        print("Laatste indices:")
        for k, v in last.indices.items():
            print(f"  {k}: {v:.4f}")