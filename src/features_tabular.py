"""Branch-2 behavioural digest computed from the real clips (the PDF's z-vector, subset that is computable).

The PDF's 250-dim digest needs a live 15 s calibration window plus openSMILE, Praat,
py-feat and pyVHR. Real-Life Trials has no calibration window, and only some of
those extractors are available here, so this module computes the honest subset:

  text   (PDF z[192:217]) disfluency, hedging, negation, pronoun, Reality-Monitoring word classes,
                          MATTR-25, speaking rate - from the human transcripts
  intent (PDF z[217:250]) 33 forensic dialogue-act similarities via frozen Qwen3-Embedding
  voice  (PDF z[90:192])  energy/pause/VAD statistics, autocorrelation F0 (semitones), F0 jitter,
                          spectral centroid/flatness/ZCR - computed with numpy from 16 kHz audio
  emotion                 emotion2vec+ 9-class probabilities
  face   (PDF z[0:90])    head-motion kinematics from MediaPipe face boxes and upper/lower-face
                          motion energy from the face crops (proxies; no AU detector available)

Within-subject z-scores (PDF Stage 0) are approximated by per-identity standardisation
over that person's clips. This uses no labels; it uses unlabeled clips of the same
person, which is what a live calibration window would provide.
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

SR = 16000
# Version of the digest definition. Bump whenever a formula, name or the feature list changes; caches and
# exported XGBoost models record it and refuse to mix versions.
FEATURE_VERSION = "v3-fixed-lag-motion"
# Head/pixel motion is measured between frames this far apart in time (not between adjacent cached frames):
# 128 frames over a 7 s webcam answer are 55 ms apart, over an 81 s courtroom clip 640 ms apart, so adjacent-
# frame differences changed with answer length. A fixed lag keeps the features comparable across durations.
MOTION_LAG_S = 0.7

DISFLUENCIES = {"um", "uh", "er", "ah", "erm", "hmm", "mm", "uhm", "umm"}
FIRST_SING = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll"}
FIRST_PLUR = {"we", "us", "our", "ours", "ourselves", "we're", "we've"}
THIRD = {"he", "she", "they", "him", "her", "them", "his", "hers", "their", "theirs", "he's", "she's", "they're"}
NEGATIONS = {"no", "not", "never", "nothing", "nobody", "none", "neither", "nor", "nowhere"}
HEDGES = {"maybe", "perhaps", "think", "guess", "probably", "possibly", "might", "suppose", "believe", "seems",
          "apparently", "somewhat", "sort", "kind"}
CERTAINTY = {"absolutely", "definitely", "certainly", "sure", "always", "honestly", "truly", "swear", "never",
             "clearly", "positive", "totally"}
SENSORY = {"see", "saw", "seen", "look", "looked", "watch", "watched", "hear", "heard", "sound", "noise", "loud",
           "smell", "smelled", "felt", "feel", "touch", "touched", "taste", "color", "red", "blue", "dark", "bright",
           "cold", "hot", "warm", "quiet", "voice", "yelled", "screamed"}
SPATIAL = {"in", "on", "at", "near", "behind", "front", "inside", "outside", "under", "above", "left", "right",
           "next", "beside", "between", "across", "room", "house", "car", "door", "street", "kitchen", "bedroom"}
TEMPORAL = {"then", "after", "before", "when", "while", "later", "first", "next", "finally", "morning", "night",
            "evening", "afternoon", "minutes", "hours", "o'clock", "yesterday", "today", "during", "until", "since"}
COGNITIVE = {"think", "thought", "know", "knew", "believe", "remember", "recall", "realize", "realized", "understand",
             "figured", "decided", "wonder", "assume", "because", "reason", "maybe", "guess", "suppose"}

INTENTS = [
    "a direct, flat denial of having done the act",
    "evasive deflection that avoids answering the question",
    "over-justification with excessive unrequested reasons",
    "counter-accusation that blames the questioner or someone else",
    "feigned memory lapse, claiming not to remember",
    "an emphatic oath or swearing that one is telling the truth",
    "temporal hedging that is vague about when things happened",
    "minimization that downplays the seriousness of events",
    "rich detail elaboration with concrete specifics",
    "narrative reconstruction telling events in chronological order",
    "appeasement trying to please or agree with the questioner",
    "a straightforward admission or confession",
    "expressing strong emotion such as grief, fear or anger",
    "describing what was seen, heard or physically felt",
    "describing locations and spatial arrangement of people and objects",
    "repeating the question back or stalling for time",
    "attacking the credibility of a witness or evidence",
    "appealing to one's own good character or reputation",
    "claiming lack of knowledge about the topic",
    "providing an alibi about being somewhere else",
    "answering a different question than the one asked",
    "contradicting an earlier statement",
    "short yes or no answer with no elaboration",
    "using formal or legalistic language to sound credible",
    "expressing sympathy or concern for the victim",
    "describing one's own thoughts and reasoning at the time",
    "quoting what other people said in conversation",
    "describing a routine daily activity",
    "uncertain speech with many hedges and qualifiers",
    "describing a relationship with another person",
    "talking about money, payments or financial matters",
    "describing a violent or physical confrontation",
    "describing contact with the police or the investigation",
]
assert len(INTENTS) == 33


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def mattr(words: List[str], w: int = 25) -> float:
    if len(words) < w:
        return len(set(words)) / max(1, len(words))
    return float(np.mean([len(set(words[i:i + w])) / w for i in range(len(words) - w + 1)]))


def text_features(text: str, duration_s: float) -> Dict[str, float]:
    w = _tokens(text)
    n = max(1, len(w))
    per100 = lambda s: 100.0 * sum(t in s for t in w) / n  # noqa: E731
    reps = sum(1 for a, b in zip(w, w[1:]) if a == b)
    neg = sum(t in NEGATIONS or t.endswith("n't") for t in w)
    rm_ext = per100(SENSORY) + per100(SPATIAL) + per100(TEMPORAL)
    rm_int = per100(COGNITIVE)
    return {
        "txt_words_per_s": len(w) / max(duration_s, 1e-3),
        "txt_disfluency_per100": per100(DISFLUENCIES),
        "txt_repetition_per100": 100.0 * reps / n,
        "txt_false_start_per100": 100.0 * text.count("-") / n,
        "txt_first_sing_per100": per100(FIRST_SING),
        "txt_first_plur_per100": per100(FIRST_PLUR),
        "txt_third_per100": per100(THIRD),
        "txt_negation_per100": 100.0 * neg / n,
        "txt_hedge_per100": per100(HEDGES),
        "txt_certainty_per100": per100(CERTAINTY),
        "txt_mattr25": mattr(w),
        "txt_mean_word_len": float(np.mean([len(t) for t in w])) if w else 0.0,
        "txt_rm_sensory_per100": per100(SENSORY),
        "txt_rm_spatial_per100": per100(SPATIAL),
        "txt_rm_temporal_per100": per100(TEMPORAL),
        "txt_rm_cognitive_per100": rm_int,
        "txt_rm_net_external_minus_internal": rm_ext - rm_int,
        "txt_question_marks_per100": 100.0 * text.count("?") / n,
    }


def _frames(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def f0_track(x: np.ndarray, sr: int = SR, fmin=60.0, fmax=400.0, win=640, hop=160, thr=0.45) -> np.ndarray:
    """Normalised-autocorrelation F0 per 40 ms frame (NaN when unvoiced)."""
    fr = _frames(x, win, hop) * np.hanning(win)[None]
    fr = fr - fr.mean(1, keepdims=True)
    spec = np.fft.rfft(fr, n=2 * win, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :win]
    ac = ac / np.maximum(ac[:, :1], 1e-12)
    lo, hi = int(sr / fmax), int(sr / fmin)
    seg = ac[:, lo:hi]
    lag = seg.argmax(1) + lo
    peak = seg.max(1)
    energy = (fr ** 2).mean(1)
    voiced = (peak > thr) & (energy > np.percentile(energy, 30))
    f0 = np.where(voiced, sr / lag, np.nan)
    return f0


def voice_features(x: np.ndarray, sr: int = SR) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < sr // 2 or np.abs(x).max() < 1e-4:
        return {}
    x = x / (np.abs(x).max() + 1e-9)
    fr = _frames(x, 400, 160)  # 25 ms / 10 ms
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    thr = np.percentile(db, 20) + 0.4 * (np.percentile(db, 95) - np.percentile(db, 20))
    speech = db > thr
    # pauses: runs of non-speech >= 250 ms between speech
    pauses, run = [], 0
    for s in speech:
        if not s:
            run += 1
        else:
            if run >= 25:
                pauses.append(run / 100.0)
            run = 0
    dur = len(x) / sr
    first_speech = np.argmax(speech) / 100.0 if speech.any() else dur
    f0 = f0_track(x, sr)
    v = f0[np.isfinite(f0)]
    st = 12 * np.log2(v / 100.0) if len(v) else np.array([np.nan])
    jit = np.nanmean(np.abs(np.diff(f0)) / f0[:-1]) if len(v) > 3 else np.nan
    win = _frames(x, 512, 256) * np.hanning(512)[None]
    mag = np.abs(np.fft.rfft(win, axis=1)) + 1e-9
    freqs = np.fft.rfftfreq(512, 1 / sr)
    centroid = (mag * freqs).sum(1) / mag.sum(1)
    flatness = np.exp(np.log(mag).mean(1)) / mag.mean(1)
    zcr = (np.abs(np.diff(np.sign(fr), axis=1)) > 0).mean(1)
    return {
        "voc_speech_ratio": float(speech.mean()),
        "voc_pauses_per_min": 60.0 * len(pauses) / max(dur, 1e-3),
        "voc_mean_pause_s": float(np.mean(pauses)) if pauses else 0.0,
        "voc_max_pause_s": float(np.max(pauses)) if pauses else 0.0,
        "voc_onset_latency_s": float(first_speech),
        "voc_rms_db_mean": float(db[speech].mean()) if speech.any() else float(db.mean()),
        "voc_rms_db_std": float(db[speech].std()) if speech.any() else 0.0,
        "voc_f0_st_mean": float(np.nanmean(st)),
        "voc_f0_st_std": float(np.nanstd(st)),
        "voc_f0_st_range": float(np.nanpercentile(st, 95) - np.nanpercentile(st, 5)) if len(v) > 5 else np.nan,
        "voc_f0_jitter": float(jit),
        "voc_voiced_ratio": float(np.isfinite(f0).mean()),
        "voc_centroid_mean": float(centroid.mean()),
        "voc_centroid_std": float(centroid.std()),
        "voc_flatness_mean": float(flatness.mean()),
        "voc_zcr_mean": float(zcr.mean()),
    }


def lag_pairs(ts: np.ndarray, lag: float = MOTION_LAG_S, tol: float = 0.5):
    """Index pairs (i, j) with ts[j] - ts[i] as close as possible to `lag` (within +-tol*lag).

    Falls back to adjacent frames when the sampling is too sparse for the lag (never happens for
    128 frames over <= 90 s). Returns (i, j, dt) with dt > 0.
    """
    ts = np.asarray(ts, dtype=np.float64)
    i = np.arange(len(ts))
    nxt = np.clip(np.searchsorted(ts, ts + lag), 0, len(ts) - 1)
    prv = np.clip(nxt - 1, 0, len(ts) - 1)
    j = np.where(np.abs(ts[prv] - ts - lag) < np.abs(ts[nxt] - ts - lag), prv, nxt)
    ok = (j > i) & (np.abs(ts[j] - ts[i] - lag) <= tol * lag)
    if ok.sum() < 3:
        i, j = np.arange(len(ts) - 1), np.arange(1, len(ts))
    else:
        i, j = i[ok], j[ok]
    return i, j, np.maximum(ts[j] - ts[i], 1e-3)


def face_features(frames: np.ndarray, meta: Dict) -> Dict[str, float]:
    """frames: (N, H, W, 3) uint8 face crops; meta: preprocess meta (boxes, timestamps).

    All motion terms are rates over a fixed MOTION_LAG_S time lag, so they do not depend on how long
    the answer was (see MOTION_LAG_S). Maxima are replaced by 95th percentiles, which do not grow with
    the number of frame pairs.
    """
    import cv2

    b = np.asarray(meta["boxes_cxcys"], dtype=np.float64)
    ts = np.asarray(meta["timestamps"], dtype=np.float64)
    order = np.argsort(ts, kind="stable")
    b, ts = b[order], ts[order]
    frames = np.asarray(frames)[order]
    i, j, dt = lag_pairs(ts)
    side = np.maximum(b[:, 2], 1.0)
    vel = np.hypot(b[j, 0] - b[i, 0], b[j, 1] - b[i, 1]) / side[i] / dt
    scale = (np.log(side[j]) - np.log(side[i])) / dt
    g = np.stack([cv2.resize(cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2GRAY), (96, 96)) for f in frames])
    g = g.astype(np.float32) / 255.0
    d = np.abs(g[j] - g[i]) / dt[:, None, None]          # intensity change per second at a fixed lag
    upper = d[:, 10:45, 15:81].mean((1, 2))   # brows / eyes region of the centred face crop
    lower = d[:, 58:88, 25:71].mean((1, 2))   # mouth / chin region
    whole = d.mean((1, 2))
    return {
        "face_detect_rate": float(meta["face_detect_rate"]),
        "face_head_speed_mean": float(vel.mean()),
        "face_head_speed_std": float(vel.std()),
        "face_head_speed_p95": float(np.percentile(vel, 95)),
        "face_scale_change_std": float(scale.std()),
        "face_motion_mean": float(whole.mean()),
        "face_motion_std": float(whole.std()),
        "face_upper_motion_mean": float(upper.mean()),
        "face_upper_motion_p95": float(np.percentile(upper, 95)),
        "face_lower_motion_mean": float(lower.mean()),
        "face_lower_motion_p95": float(np.percentile(lower, 95)),
        "face_upper_lower_ratio": float(upper.mean() / (lower.mean() + 1e-6)),
    }


def within_identity_z(X: np.ndarray, groups: np.ndarray, min_size: int = 3) -> np.ndarray:
    """Per-identity standardisation (label-free). NaN for identities with < min_size clips."""
    Z = np.full_like(X, np.nan, dtype=np.float64)
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < min_size:
            continue
        mu = np.nanmean(X[m], 0)
        sd = np.nanstd(X[m], 0)
        Z[m] = np.clip((X[m] - mu) / np.maximum(sd, 1e-6), -10, 10)  # PDF clip(+-10)
    return Z
