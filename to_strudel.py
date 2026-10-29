#!/usr/bin/env python3
"""notes.json -> Strudel code.

Pure standard library, so it runs in milliseconds and can be re-run with other
settings (tempo, bars, alignment) without transcribing the song again.

notes.json is written by pipeline.py:
{
  "source": "song.mp3",
  "grid": {"bpm": 123.0, "beats_per_bar": 4, "first_downbeat": 0.51,
           "beats": [0.03, 0.52, ...]},          # tracked beat times in seconds
  "tracks": [{"name": "bass", "kind": "melodic" | "drums",
              "notes": [{"pitch": 39, "start": 1.23, "end": 1.48,
                         "instrument": "electric_bass"}, ...]}, ...]
}
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------- pitch names

NAMES_SHARP = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]
NAMES_FLAT = ["c", "db", "d", "eb", "e", "f", "gb", "g", "ab", "a", "bb", "b"]
KEY_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
KEY_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles.
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
# Major keys written with flats: F Bb Eb Ab Db Gb.
FLAT_MAJOR_TONICS = {5, 10, 3, 8, 1, 6}

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]


def _correlation(a: list[float], b: list[float]) -> float:
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else 0.0


def detect_key(histogram: list[float]) -> tuple[int, str, float]:
    """(tonic pitch class, 'major'|'minor', correlation) for a pitch-class histogram."""
    best = (0, "major", -2.0)
    for tonic in range(12):
        rotated = [histogram[(tonic + i) % 12] for i in range(12)]
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            r = _correlation(rotated, profile)
            if r > best[2]:
                best = (tonic, mode, r)
    return best


def uses_flats(tonic: int, mode: str) -> bool:
    major_tonic = tonic if mode == "major" else (tonic + 3) % 12
    return major_tonic in FLAT_MAJOR_TONICS


def key_label(tonic: int, mode: str) -> str:
    names = KEY_NAMES_FLAT if uses_flats(tonic, mode) else KEY_NAMES_SHARP
    return f"{names[tonic]} {mode}"


def note_name(midi: int, flats: bool) -> str:
    """Strudel/tonal convention: MIDI 60 = c4."""
    names = NAMES_FLAT if flats else NAMES_SHARP
    return f"{names[midi % 12]}{midi // 12 - 1}"


def snap_to_scale(midi: int, tonic: int, mode: str) -> int:
    scale = {(tonic + s) % 12 for s in (MAJOR_SCALE if mode == "major" else MINOR_SCALE)}
    if midi % 12 in scale:
        return midi
    # Out-of-key notes are a semitone from a scale note; prefer the one below.
    return midi - 1 if (midi - 1) % 12 in scale else midi + 1


# ------------------------------------------------------------------ beat grid


class Grid:
    """Maps seconds in the audio to beat positions, with beat 0 on the first downbeat.

    Follows the tracked beats (so small tempo wobbles in a live recording don't
    drift the bar lines), or a constant tempo when --bpm is given.
    """

    def __init__(self, grid: dict, *, bpm: float | None, first_downbeat: float | None,
                 beats_per_bar: int | None, beat_scale: float, duration: float):
        self.beats_per_bar = beats_per_bar or grid.get("beats_per_bar") or 4
        downbeat = first_downbeat if first_downbeat is not None else grid.get("first_downbeat")
        beats = list(grid.get("beats") or [])

        if bpm is not None or len(beats) < 4:
            # Constant grid anchored on the downbeat.
            self.bpm = bpm or grid.get("bpm") or 120.0
            period = 60.0 / self.bpm
            anchor = downbeat if downbeat is not None else (beats[0] if beats else 0.0)
            start = anchor - period * math.ceil(anchor / period + 1)
            count = int((duration - start) / period) + 3
            beats = [start + i * period for i in range(count)]
            beat_scale = 1.0  # the given bpm already says what a beat is
        else:
            self.bpm = (grid.get("bpm") or 60.0 / _median_gap(beats)) * beat_scale

        if beat_scale == 2.0:
            beats = [t for a, b in zip(beats, beats[1:]) for t in (a, (a + b) / 2)] + [beats[-1]]
        elif beat_scale == 0.5:
            anchor_index = _nearest(beats, downbeat if downbeat is not None else beats[0])
            beats = beats[anchor_index % 2::2]
        elif beat_scale != 1.0:
            raise SystemExit("--beat-scale must be 0.5, 1 or 2")

        self.beats = beats
        self.zero = self._raw(downbeat if downbeat is not None else beats[0])

    def _raw(self, t: float) -> float:
        b = self.beats
        i = bisect.bisect_right(b, t) - 1
        if i < 0:
            return (t - b[0]) / (b[1] - b[0])
        if i >= len(b) - 1:
            i = len(b) - 2
        return i + (t - b[i]) / (b[i + 1] - b[i])

    def beat_at(self, t: float) -> float:
        return self._raw(t) - self.zero

    def time_at(self, beat: float) -> float:
        raw = beat + self.zero
        b = self.beats
        i = min(max(int(math.floor(raw)), 0), len(b) - 2)
        return b[i] + (raw - i) * (b[i + 1] - b[i])


def _median_gap(times: list[float]) -> float:
    gaps = sorted(b - a for a, b in zip(times, times[1:]))
    return gaps[len(gaps) // 2]


def _nearest(times: list[float], t: float) -> int:
    return min(range(len(times)), key=lambda i: abs(times[i] - t))


def estimate_offset(positions: list[float]) -> float:
    """Circular mean of how far onsets sit from the nearest grid step, in steps.

    Transcribed onsets tend to be a few ms late (or early) relative to the
    tracked beats. Measured over all notes, that systematic offset is removed
    before rounding, so notes don't all fall on the wrong side of a step.
    """
    if not positions:
        return 0.0
    s = sum(math.sin(2 * math.pi * p) for p in positions)
    c = sum(math.cos(2 * math.pi * p) for p in positions)
    return math.atan2(s, c) / (2 * math.pi)


# ------------------------------------------------------------- mini-notation
#
# A "lane" is one stream of events, one bar per cycle. A bar is a tuple of
# events (start step, token, end step); drum events have end None and are
# filled up to the next hit, since a one-shot sample has no length of its own.

Event = tuple[int, str, "int | None"]
Bar = tuple[Event, ...]


def _rest(steps: int) -> str:
    return "~" if steps == 1 else f"~@{steps}"


def _bracket(tokens: list[str]) -> str:
    if len(tokens) == 1 and "!" not in tokens[0]:
        return tokens[0]
    return "[" + " ".join(tokens) + "]"


def _pieces(events: Bar, steps: int) -> list[str]:
    """Events inside `steps` steps as tokens with @-weights, reduced by their gcd."""
    pieces: list[tuple[str, int]] = []
    pos = 0
    for i, (s, token, end) in enumerate(events):
        if s > pos:
            pieces.append(("~", s - pos))
        nxt = events[i + 1][0] if i + 1 < len(events) else steps
        end = nxt if end is None else max(s + 1, min(end, nxt))
        pieces.append((token, end - s))
        pos = end
    if pos < steps:
        pieces.append(("~", steps - pos))
    g = 0
    for _, w in pieces:
        g = math.gcd(g, w)
    out = []
    i = 0
    while i < len(pieces):
        token, w = pieces[i][0], pieces[i][1] // g
        n = 1
        while i + n < len(pieces) and pieces[i + n] == pieces[i]:
            n += 1
        i += n
        if token == "~":
            out.append(_rest(w))
            continue
        single = token if w == 1 else f"{token}@{w}"
        plain = " ".join([single] * n)
        # Repeats: "x!4" (only for weight 1: "x@2!3" would drop the weight in
        # Strudel), or "[x!3]@6" for weighted ones; whichever is shortest.
        packed = f"{token}!{n}" if w == 1 else f"[{token}!{n}]@{w * n}"
        if n > 1 and len(packed) < len(plain):
            out.append(packed)
        else:
            out.extend([single] * n)
    return out


def _period(events: Bar, steps: int) -> int:
    """Smallest p dividing `steps` such that the bar is p repeated, else `steps`."""
    for p in range(1, steps):
        if steps % p:
            continue
        first = [e for e in events if e[0] < p]
        if not first or any(e[2] is not None and e[2] > p for e in first):
            continue
        k = steps // p
        expected = [(s + j * p, tok, None if end is None else end + j * p)
                     for j in range(k) for s, tok, end in first]
        if list(events) == expected:
            return p
    return steps


def render_bar(events: Bar, steps: int) -> str:
    """One bar as mini-notation: `bd*4`, `[~ sd]*2`, `[eb2@3 eb2@3 eb2@2 ~@2 ...]`."""
    if not events:
        return "~"
    p = _period(events, steps)
    if p < steps:
        inner = _pieces(tuple(e for e in events if e[0] < p), p)
        return f"{_bracket(inner)}*{steps // p}"
    return _bracket(_pieces(events, steps))


def fold_repeats(seq: list[str]) -> list[str]:
    """Consecutive identical bars become one `bar!n` token."""
    tokens = []
    i = 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1] == seq[i]:
            j += 1
        count = j - i + 1
        tokens.append(seq[i] if count == 1 else f"{seq[i]}!{count}")
        i = j + 1
    return tokens


def segment_bars(bars: list[str], max_period: int = 16) -> list[tuple[int, int, list[str]]]:
    """Split bars into (start bar, repeats, block) segments.

    Greedy from the left: at each bar, the block of 2..max_period bars that
    repeats back-to-back covering the most bars wins (ties: the shorter block).
    Bars that start no such repetition are collected into a plain run
    (repeats = 1), where single-bar repeats still fold to `!n`.
    """
    n = len(bars)
    segments: list[tuple[int, int, list[str]]] = []
    free: list[str] = []
    free_start = 0
    i = 0
    while i < n:
        best_p, best_r = 1, 1
        for p in range(2, min(max_period, (n - i) // 2) + 1):
            r = 1
            while i + (r + 1) * p <= n and bars[i + r * p:i + (r + 1) * p] == bars[i:i + p]:
                r += 1
            # A 2+ bar block only counts if it isn't just one bar repeated.
            if r >= 2 and p * r > best_p * best_r and len(set(bars[i:i + p])) > 1:
                best_p, best_r = p, r
        if best_r >= 2:
            if free:
                segments.append((free_start, 1, free))
                free = []
            segments.append((i, best_r, bars[i:i + best_p]))
            i += best_p * best_r
        else:
            if not free:
                free_start = i
            free.append(bars[i])
            i += 1
    if free:
        segments.append((free_start, 1, free))
    return segments


LINE = 90


def lane_mini(bars: list[str]) -> tuple[str, str]:
    """(mini-notation, structure summary) for a lane of rendered bars, one per cycle.

    Uses only mini-notation, so lanes can be stacked with commas:
      bar!n               the same bar n times
      [[b1 b2]!r]@(2r)    a 2-bar block played r times (one bar per cycle)
    A lane that is one loop from start to end is written as just the loop,
    since Strudel repeats it anyway.
    """
    segments = segment_bars(bars)
    if len(segments) == 1:
        _, repeats, block = segments[0]
        tokens = fold_repeats(block)
        if repeats > 1 or len(tokens) == 1:
            # The whole lane is one loop: drop the repeat count.
            tokens = [t.rsplit("!", 1)[0] for t in tokens] if len(tokens) == 1 else tokens
            summary = (f"{len(block)}-bar loop" if len(block) > 1 else "same bar throughout")
            return _wrap_slow(tokens), summary

    parts: list[str] = []
    multiline: list[str] = []
    looped = 0
    for _, repeats, block in segments:
        if repeats == 1:
            for t in fold_repeats(block):
                parts.append(t)
                multiline.append("  " + t)
        else:
            looped += len(block) * repeats
            inner = fold_repeats(block)
            parts.append(f"[[{' '.join(inner)}]!{repeats}]@{len(block) * repeats}")
            multiline.append("  [[\n" + "\n".join("    " + t for t in inner)
                             + f"\n  ]!{repeats}]@{len(block) * repeats}")
    one_line = "<" + " ".join(parts) + ">"
    summary = f"{looped} of {len(bars)} bars in loops" if looped else ""
    if len(one_line) <= LINE:
        return one_line, summary
    return "<\n" + "\n".join(multiline) + "\n>", summary


def _wrap_slow(tokens: list[str]) -> str:
    if len(tokens) == 1:
        return tokens[0]
    one_line = "<" + " ".join(tokens) + ">"
    if len(one_line) <= LINE:
        return one_line
    return "<\n" + "\n".join("  " + t for t in tokens) + "\n>"


# ------------------------------------------------------------------ simplify
#
# Lossy. A transcription of a repeating part is rarely repeated exactly: a
# missed note here, a ghost note there, and no two bars are equal. --simplify
# finds the loop length (1, 2, 4 or 8 bars, at any offset) that describes a
# lane with the fewest distinct parts, merges near-identical loops into one
# part each, and writes the lane as named parts played by arrange().


def _keyset(bar: Bar) -> frozenset:
    """(step, single sound/pitch) pairs; chords are split into their voices."""
    out = set()
    for s, tok, _ in bar:
        for voice in (tok[1:-1].split(",") if tok.startswith("[") else [tok]):
            out.add((s, voice))
    return frozenset(out)


def _similarity(a: tuple[frozenset, ...], b: tuple[frozenset, ...]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        union = x | y
        total += 1.0 if not union else len(x & y) / len(union)
    return total / len(a)


SILENT = -1


def loopify(bars: list[Bar], threshold: float, cost_of) -> tuple[list[tuple[int, int]], list[tuple[Bar, ...]], int]:
    """(runs, parts, changed events).

    runs: (part index or SILENT, cycles) in order; parts: the bars of each part.
    `cost_of(parts, runs)` is the length of the code they would produce; the
    loop length and offset with the lowest cost win (ties: fewer changes).
    """
    n = len(bars)
    keys = [_keyset(b) for b in bars]
    best = None
    for length in (1, 2, 4, 8):
        if length > 1 and n < 2 * length:
            break
        for offset in range(length):
            starts = ([0] if offset else []) + list(range(offset, n, length))
            chunks = [(a, (starts[i + 1] if i + 1 < len(starts) else n) - a) for i, a in enumerate(starts)]
            variants = Counter(tuple(bars[a:a + ln]) for a, ln in chunks)
            variant_keys = {v: tuple(_keyset(b) for b in v) for v in variants}
            parts: list[tuple[Bar, ...]] = []
            assign: dict[tuple[Bar, ...], int] = {}
            for v, _ in sorted(variants.items(), key=lambda kv: -kv[1]):
                if not any(v):
                    assign[v] = SILENT
                    continue
                for i, part in enumerate(parts):
                    if len(part) == len(v) and _similarity(variant_keys[v], variant_keys[part]) >= threshold:
                        assign[v] = i
                        break
                else:
                    assign[v] = len(parts)
                    parts.append(v)
            runs: list[list[int]] = []
            changed = 0
            for a, ln in chunks:
                v = tuple(bars[a:a + ln])
                pid = assign[v]
                if pid != SILENT and parts[pid] != v:
                    changed += sum(len(x ^ y) for x, y in zip(variant_keys[v], variant_keys[parts[pid]]))
                elif pid == SILENT:
                    pass
                if runs and runs[-1][0] == pid:
                    runs[-1][1] += ln
                else:
                    runs.append([pid, ln])
            runs_t = [(pid, c) for pid, c in runs]
            score = (cost_of(parts, runs_t), changed)
            if best is None or score < best[0]:
                best = (score, runs_t, parts, changed)
    return best[1], best[2], best[3]


def bass_cleanup(rows: list[tuple[int, int, int]], fold: bool) -> list[tuple[int, int, int]]:
    """One note at a time (the lowest); with `fold`, moved into the octave around the median.

    Transcribers often double a bass note an octave up, or put a whole section
    an octave off. Lossy: the lowest-note rule is part of --simplify, folding
    is --fold-bass, since a song may change octave on purpose.
    """
    lowest: dict[int, tuple[int, int, int]] = {}
    for row in rows:
        if row[0] not in lowest or row[2] < lowest[row[0]][2]:
            lowest[row[0]] = row
    pitches = sorted(r[2] for r in lowest.values())
    if not pitches:
        return []
    median = pitches[len(pitches) // 2]
    out = []
    for on, off, p in sorted(lowest.values()):
        while fold and p < median - 6:
            p += 12
        while fold and p >= median + 6:
            p -= 12
        out.append((on, off, p))
    return out


# --------------------------------------------------------------- drum mapping

# General MIDI drum notes -> Strudel drum-machine sample names.
GM_DRUMS = {
    35: "bd", 36: "bd",
    37: "rim", 38: "sd", 40: "sd",
    39: "cp",
    42: "hh", 44: "hh", 46: "oh",
    41: "lt", 43: "lt", 45: "mt", 47: "mt", 48: "ht", 50: "ht",
    49: "cr", 52: "cr", 55: "cr", 57: "cr",
    51: "rd", 53: "rd", 59: "rd",
}
DRUM_ORDER = ["bd", "sd", "rim", "cp", "hh", "oh", "lt", "mt", "ht", "cr", "rd", "perc"]


def sound_for(track: dict) -> str:
    name = track["name"].lower()
    instruments = Counter(n.get("instrument", "") for n in track["notes"])
    main = instruments.most_common(1)[0][0] if instruments else ""
    hint = f"{name} {main}"
    if "bass" in hint:
        return '.s("sawtooth").lpf(700).lpq(4).decay(.3).sustain(.5).gain(.8)'
    if "voice" in hint or "vocal" in hint:
        return '.s("triangle").attack(.02).release(.1).gain(.6)'
    if "piano" in hint:
        return '.s("piano").gain(.7)'
    if "pad" in hint or "strings" in hint or "string" in hint or "organ" in hint:
        return '.s("sawtooth").lpf(1800).attack(.08).release(.4).gain(.35)'
    if "lead" in hint:
        return '.s("square").lpf(2500).gain(.4)'
    if "guitar" in hint:
        return '.s("sawtooth").lpf(2200).decay(.4).sustain(.2).gain(.4)'
    return '.s("triangle").gain(.5)'


# ------------------------------------------------------------- audio samples

SAMPLE_PORT = 5432


def cut_bars(stem: Path, bar_times: list[float], out_dir: Path) -> int:
    """Write one WAV per bar of `stem` to out_dir/000.wav, 001.wav ...

    Bars before the start of the audio (a pickup) are padded with silence.
    Uses the standard library for PCM WAV and soundfile for anything else.
    """
    import wave

    try:
        with wave.open(str(stem), "rb") as w:
            channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            data = w.readframes(w.getnframes())
    except wave.Error:
        import numpy as np
        import soundfile as sf

        audio, rate = sf.read(str(stem), dtype="float32", always_2d=True)
        channels, width = audio.shape[1], 2
        data = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()

    frame = channels * width
    total = len(data) // frame
    if out_dir.exists():
        for old in out_dir.glob("*.wav"):
            old.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (a, b) in enumerate(zip(bar_times, bar_times[1:])):
        fa, fb = round(a * rate), round(b * rate)
        chunk = bytearray()
        if fa < 0:
            chunk += bytes((min(fb, 0) - fa) * frame)
        lo, hi = max(fa, 0), min(max(fb, 0), total)
        if hi > lo:
            chunk += data[lo * frame:hi * frame]
        chunk += bytes(((fb - fa) * frame) - len(chunk))
        with wave.open(str(out_dir / f"{i:03d}.wav"), "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(width)
            w.setframerate(rate)
            w.writeframes(bytes(chunk))
    return len(bar_times) - 1


# ------------------------------------------------------------------------ main


def build(data: dict, args) -> tuple[str, str | None, dict]:
    """(main code, audio-stems code or None, stats)."""
    # Grid offset, key and bar numbering always come from all tracks, so they
    # stay the same when --tracks picks only some of them for output.
    tracks = [t for t in data["tracks"] if t["notes"]]
    if not tracks:
        raise SystemExit("No notes to convert.")
    names = [t["name"] for t in tracks]
    selected = set(names)
    if args.tracks:
        wanted = {w.strip().lower() for w in args.tracks.split(",")}
        selected = {n for n in names if n.lower() in wanted}
        if not selected:
            raise SystemExit(f"No track matches --tracks. Tracks: {', '.join(names)}")
    as_audio = set()
    if args.as_audio and args.as_audio.lower() != "none":
        as_audio = {n for n in names if n.lower() in {w.strip().lower() for w in args.as_audio.split(",")}}

    duration = max(n["end"] for t in tracks for n in t["notes"]) + 1.0
    grid = Grid(data.get("grid") or {}, bpm=args.bpm, first_downbeat=args.first_downbeat,
                beats_per_bar=args.beats_per_bar, beat_scale=args.beat_scale,
                duration=duration)
    steps_per_beat = args.grid
    spb = grid.beats_per_bar * steps_per_beat

    def to_steps(t: float) -> float:
        return grid.beat_at(t) * steps_per_beat

    # Systematic latency between the transcription and the beat grid.
    offset = 0.0
    if not args.no_align:
        offset = estimate_offset([to_steps(n["start"]) for t in tracks for n in t["notes"]])

    # Key from duration-weighted pitch classes of all melodic notes.
    histogram = [0.0] * 12
    for t in tracks:
        if t["kind"] == "melodic":
            for n in t["notes"]:
                histogram[n["pitch"] % 12] += max(n["end"] - n["start"], 0.05)
    tonic, mode, confidence = detect_key(histogram)
    if args.key:
        tonic, mode = parse_key(args.key)
    flats = uses_flats(tonic, mode) if args.accidentals == "auto" else args.accidentals == "flats"

    # Quantize every note to (start step, end step, token).
    quantized: dict[str, list[tuple[int, int, str | int]]] = {}
    for t in tracks:
        rows = []
        for n in t["notes"]:
            if n["end"] - n["start"] < args.min_duration and t["kind"] == "melodic":
                continue
            on = round(to_steps(n["start"]) - offset)
            if t["kind"] == "drums":
                rows.append((on, on + 1, GM_DRUMS.get(n["pitch"], "perc")))
            else:
                off = max(on + 1, round(to_steps(n["end"]) - offset))
                pitch = snap_to_scale(n["pitch"], tonic, mode) if args.snap_key else n["pitch"]
                rows.append((on, off, pitch))
        quantized[t["name"]] = rows

    all_on = [r[0] for rows in quantized.values() for r in rows]
    first_bar = math.floor(min(all_on) / spb)
    last_bar = math.floor(max(all_on) / spb)
    start_bar = first_bar + (args.start_bar - 1)
    end_bar = last_bar if args.bars is None else min(last_bar, start_bar + args.bars - 1)
    if start_bar > end_bar:
        raise SystemExit(f"--start-bar is after the last bar ({last_bar - first_bar + 1}).")
    n_bars = end_bar - start_bar + 1
    bar_times = [grid.time_at((start_bar + k) * grid.beats_per_bar) for k in range(n_bars + 1)]
    stats = {"notes": 0, "changed": 0}

    def bars_of(rows, key) -> list[Bar]:
        """Rows -> one tuple of events per bar, with `key` turning a group of
        simultaneous rows into (token, end)."""
        per_bar: list[dict[int, list]] = [defaultdict(list) for _ in range(n_bars)]
        for on, off, tok in rows:
            b = on // spb - start_bar
            if 0 <= b < n_bars:
                per_bar[b][on - (start_bar + b) * spb].append((off - (start_bar + b) * spb, tok))
        out = []
        for bar in per_bar:
            out.append(tuple((s, *key(bar[s])) for s in sorted(bar)))
        return out

    def melodic_key(items):
        pitches = sorted({tok for _, tok in items})
        if args.max_voices and len(pitches) > args.max_voices:
            pitches = pitches[-args.max_voices:]  # keep the top voices
        tokens = [note_name(p, flats) for p in pitches]
        token = tokens[0] if len(tokens) == 1 else "[" + ",".join(tokens) + "]"
        return token, max(off for off, _ in items)

    def drum_key(items):
        sounds = sorted({tok for _, tok in items}, key=DRUM_ORDER.index)
        return (sounds[0] if len(sounds) == 1 else "[" + ",".join(sounds) + "]"), None

    rendered: dict[Bar, str] = {}

    def render(bars) -> list[str]:
        out = []
        for bar in bars:
            if bar not in rendered:
                rendered[bar] = render_bar(bar, spb)
            out.append(rendered[bar])
        return out

    def voices(tok: str) -> list[str]:
        return tok[1:-1].split(",") if tok.startswith("[") else [tok]

    def note_expr(bars) -> str:
        return f"note(`{lane_mini(render(bars))[0]}`)"

    def kit_expr(bars) -> str:
        """Drums as one lane per sound stacked with commas (bd*4, [~ sd]*2, hh*8),
        or all sounds in one lane; whichever is shorter."""
        together = lane_mini(render(bars))[0]
        sounds = sorted({v for bar in bars for _, tok, _ in bar for v in voices(tok)}, key=DRUM_ORDER.index)
        if len(sounds) < 2:
            return f"s(`{together}`)"
        split = [lane_mini(render([tuple((st, snd, None) for st, tok, _ in bar if snd in voices(tok))
                                  for bar in bars]))[0] for snd in sounds]
        joined = ", ".join(split)
        if len(joined) >= len(together):
            return f"s(`{together}`)"
        if "\n" not in joined and len(joined) <= LINE:
            return f"s(`{joined}`)"
        return "stack(\n" + ",\n".join("  s(`" + m.replace("\n", "\n  ") + "`)" for m in split) + "\n)"

    def part_names(n: int) -> list[str]:
        letters = [chr(ord("A") + i) for i in range(26)]
        return (letters + [a + b for a in letters for b in letters])[:n]

    def track_code(t: dict) -> tuple[list[str], str, str] | None:
        """(const lines, expression without sound chain, summary) or None if silent."""
        rows = quantized[t["name"]]
        drums = t["kind"] == "drums"
        in_range = lambda rs: [r for r in rs if 0 <= r[0] // spb - start_bar < n_bars]
        count = len(in_range(rows))
        cleaned = 0
        if (args.simplify or args.fold_bass) and not drums and "bass" in t["name"].lower():
            before = {(on, p) for on, _, p in in_range(rows)}
            rows = bass_cleanup(rows, args.fold_bass)
            cleaned = len(before - {(on, p) for on, _, p in in_range(rows)})
        if not count:
            return None
        stats["notes"] += count
        stats["changed"] += cleaned
        bars = bars_of(rows, drum_key if drums else melodic_key)
        expr_of = kit_expr if drums else note_expr
        unit = "hits" if drums else "notes"

        if not args.simplify:
            summary = "" if drums else lane_mini(render(bars))[1]
            if cleaned:
                summary = f"{cleaned} notes moved/removed by --fold-bass" + (f", {summary}" if summary else "")
            return [], expr_of(bars), f"{count} {unit}" + (f", {summary}" if summary else "")

        def cost(parts, runs):
            return sum(len(expr_of(list(p))) + 20 for p in parts) + 14 * len(runs)

        runs, parts, changed = loopify(bars, args.simplify, cost)
        stats["changed"] += changed
        changed += cleaned
        pct = f"{100 * changed / count:.0f}%"
        if len(runs) == 1 and runs[0][0] != SILENT:
            return [], expr_of(list(parts[0])), (f"{count} {unit}, one {len(parts[0])}-bar loop, "
                                                 f"{changed} {unit} changed ({pct})")
        ident = re.sub(r"\W", "_", t["name"])
        used: list[int] = []
        for pid, _ in runs:
            if pid != SILENT and pid not in used:
                used.append(pid)
        names = dict(zip(used, (f"{ident}_{n}" for n in part_names(len(used)))))
        consts = [f"const {names[pid]} = {expr_of(list(parts[pid]))}" for pid in used]
        entries = [f"[{c}, {'silence' if pid == SILENT else names[pid]}]" for pid, c in runs]
        lines, line = [], ""
        for e in entries:
            if line and len(line) + len(e) + 2 > LINE:
                lines.append(line)
                line = ""
            line = f"{line}, {e}" if line else e
        lines.append(line)
        expr = "arrange(\n  " + ",\n  ".join(lines) + "\n)"
        summary = (f"{count} {unit}, {len(used)} {'part' if len(used) == 1 else 'parts'} in {len(runs)} sections, "
                   f"{changed} {unit} changed ({pct})")
        return consts, expr, summary

    header = [f"// Generated by audio2strudel from: {Path(data.get('source', '?')).name}",
              f"// Tempo {grid.bpm:.2f} BPM, {grid.beats_per_bar}/4, key {key_label(tonic, mode)}"
              + ("" if args.key else f" (correlation {confidence:.2f})"),
              f"// Bars {args.start_bar}-{args.start_bar + n_bars - 1} of {last_bar - first_bar + 1}. "
              f"Bar {args.start_bar} starts at {bar_times[0]:.3f} s in the audio"
              + (" (pickup: before the audio starts)." if bar_times[0] < 0 else ".")]

    lines = list(header)
    if offset:
        ms = offset / steps_per_beat * 60000.0 / grid.bpm
        lines.append(f"// Transcription latency against the beat grid compensated: {ms:+.0f} ms")
    if args.simplify:
        lines.append(f"// Simplified (--simplify {args.simplify}): near-identical loops are made identical"
                     " and grouped into parts (A, B ...) that arrange() plays in order.")
    if as_audio & selected:
        lines.append(f"// {', '.join(sorted(as_audio & selected))}: see {args.audio_file_name}")
    lines.append("// Mute a track by changing '$:' to '_$:'. Add ._pianoroll() to see the notes.")
    lines += ["", f"setcps({grid.bpm:.2f}/60/{grid.beats_per_bar})"]

    order = sorted((t for t in tracks if t["name"] in selected and t["name"] not in as_audio),
                   key=lambda t: (t["kind"] != "drums", "bass" not in t["name"].lower()))
    for t in order:
        code = track_code(t)
        if code is None:
            continue
        consts, expr, summary = code
        chain = '.bank("RolandTR909").gain(.8)' if t["kind"] == "drums" else sound_for(t)
        lines += ["", f"// {t['name']}: {summary}", *consts, f"$: {expr}{chain}"]
    lines.append("")

    # Stems played as audio: cut into one sample per bar, plus their notes as a muted alternative.
    audio_code = None
    audio_tracks = [t for t in tracks if t["name"] in selected and t["name"] in as_audio]
    if audio_tracks:
        work = args.notes_json.resolve().parent
        sample_map: dict[str, list[str]] = {"_base": f"http://localhost:{SAMPLE_PORT}/"}
        blocks = []
        for t in audio_tracks:
            stem = work / "stems" / f"{t['name']}.wav"
            bank = f"{t['name']}_stem"
            code = track_code(t)
            notes_block = ""
            if code is not None:
                consts, expr, summary = code
                notes_block = "\n".join([f"// {t['name']} as notes ({summary}). Remove the '_' to hear them:",
                                         *consts, f"_$: {expr}{sound_for(t)}"])
            if stem.exists():
                count = cut_bars(stem, bar_times, work / "audio" / t["name"])
                sample_map[bank] = [f"audio/{t['name']}/{i:03d}.wav" for i in range(count)]
                blocks.append(f"// {t['name']} as audio: one sample per bar, triggered on the downbeat.\n"
                              f"$: s(\"{bank}\").n(run({count}).slow({count})).clip(1).gain(1)")
            if notes_block:
                blocks.append(notes_block)
        uses_samples = len(sample_map) > 1
        if uses_samples:
            (work / "strudel.json").write_text(json.dumps(sample_map, indent=1), encoding="utf-8")
        audio_lines = header + [
            "//",
            f"// Paste this code AFTER the code from {args.main_file_name} in Strudel.",
        ]
        if uses_samples:
            audio_lines += [
                "// The audio is loaded from your own machine. First start the sample server in a terminal:",
                f"//   .venv\\Scripts\\python serve_samples.py \"{work}\"",
                "// The browser may ask for permission to access devices on your local network: allow it.",
                "",
                f"samples('http://localhost:{SAMPLE_PORT}/strudel.json')",
            ]
        audio_lines += ["", "\n\n".join(blocks), ""]
        audio_code = "\n".join(audio_lines)

    return "\n".join(lines), audio_code, stats


def parse_key(text: str) -> tuple[int, str]:
    parts = text.replace(":", " ").split()
    name = parts[0].lower()
    mode = parts[1].lower() if len(parts) > 1 else "major"
    if mode in ("min", "m", "moll"):
        mode = "minor"
    if mode in ("maj", "dur"):
        mode = "major"
    for names in (NAMES_SHARP, NAMES_FLAT):
        if name in names:
            return names.index(name), mode
    raise SystemExit(f"Unknown key: {text} (use e.g. 'g minor' or 'eb major')")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Convert notes.json to Strudel code.")
    p.add_argument("notes_json", type=Path)
    p.add_argument("-o", "--output", type=Path, help="Default: <notes_json folder>/<name>.strudel.js")
    p.add_argument("--start-bar", type=int, default=1, help="First bar to write (1 = first bar with notes)")
    p.add_argument("--bars", type=int, help="Number of bars (default: rest of the song)")
    p.add_argument("--tracks", help="Only these tracks, e.g. 'bass,drums'")
    p.add_argument("--grid", type=int, default=4, help="Steps per beat: 4 = 16ths, 2 = 8ths, 3 = triplets")
    p.add_argument("--bpm", type=float, help="Constant tempo instead of the tracked beats")
    p.add_argument("--first-downbeat", type=float, help="Time in seconds of a bar line")
    p.add_argument("--beats-per-bar", type=int, help="Beats per bar (default: detected, else 4)")
    p.add_argument("--beat-scale", type=float, default=1.0,
                   help="2 if the detected tempo is half the real one, 0.5 if double")
    p.add_argument("--no-align", action="store_true", help="Do not compensate for latency against the grid")
    p.add_argument("--key", help="Force the key, e.g. 'g minor' (sets flats/sharps and --snap-key)")
    p.add_argument("--accidentals", choices=["auto", "flats", "sharps"], default="auto")
    p.add_argument("--snap-key", action="store_true", help="Move notes outside the key to the nearest scale note")
    p.add_argument("--min-duration", type=float, default=0.04, help="Drop melodic notes shorter than this (seconds)")
    p.add_argument("--max-voices", type=int, default=4, help="Max simultaneous notes per chord (0 = no limit)")
    p.add_argument("--simplify", type=float, metavar="0-1",
                   help="Find loops of 1-8 bars and merge near-identical loops into parts A, B ... played by "
                        "arrange() (lossy). 0.8 = loops sharing 80%% of their notes are made identical")
    p.add_argument("--fold-bass", action="store_true",
                   help="Move bass notes into one octave around the median (fixes octave errors, lossy)")
    p.add_argument("--as-audio", default="vocals",
                   help="Stems played as audio in a separate file instead of notes "
                        "(default: vocals). 'none' = everything as notes in the main file")
    args = p.parse_args(argv)

    data = json.loads(args.notes_json.read_text(encoding="utf-8"))
    stem = Path(data.get("source", "song")).stem
    out = args.output or args.notes_json.with_name(stem + ".strudel.js")
    base = out.name
    for suffix in (".strudel.js", ".js"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    audio_out = out.with_name(base + ".audio.strudel.js")
    args.main_file_name, args.audio_file_name = out.name, audio_out.name

    code, audio_code, stats = build(data, args)
    out.write_text(code, encoding="utf-8")
    print(f"Strudel code written to {out} ({len(code)} characters)", file=sys.stderr)
    if audio_code:
        audio_out.write_text(audio_code, encoding="utf-8")
        print(f"Audio stems written to {audio_out}", file=sys.stderr)
    if args.simplify and stats["notes"]:
        pct = 100 * stats["changed"] / stats["notes"]
        print(f"--simplify changed {stats['changed']} of {stats['notes']} notes ({pct:.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
