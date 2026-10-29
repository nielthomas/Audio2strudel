#!/usr/bin/env python3
"""Audio -> stems -> notes -> Strudel, for testing transcription quality on one song.

  1. ffmpeg converts the input to WAV.
  2. audio-separator splits it into stems with Demucs (htdemucs_ft or htdemucs_6s).
  3. MuScriptor detects the beat grid once on the full mix, then transcribes each
     stem, told which instrument to expect (bass stem -> bass, vocals -> voice, ...).
  4. to_strudel.py turns the notes into Strudel code on that one shared grid.

Everything lands in out/<song>/. Steps whose output already exists are skipped,
so re-running with other converter settings takes seconds.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import to_strudel

# stem -> (track kind, MuScriptor instrument groups; None = let the model decide)
STEMS_4 = {
    "drums": ("drums", ["drums"]),
    "bass": ("melodic", ["electric_bass", "acoustic_bass"]),
    "vocals": ("melodic", ["voice"]),
    "other": ("melodic", None),
}
STEMS_6 = {
    **STEMS_4,
    "guitar": ("melodic", ["acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"]),
    "piano": ("melodic", ["acoustic_piano", "electric_piano"]),
}
SILENCE_DBFS = -45.0


def log(msg: str) -> None:
    print(f"[audio2strudel] {msg}", file=sys.stderr, flush=True)


def check_hf_token(size: str) -> None:
    """Fail before the (slow) separation if the model can't be downloaded."""
    from huggingface_hub import auth_check, get_token
    from huggingface_hub.utils import GatedRepoError

    repo = f"MuScriptor/muscriptor-{size}"
    py = Path(sys.executable)
    if not get_token():
        log("The MuScriptor model requires a (free) HuggingFace account:")
        log(f"  1. Accept the license at https://huggingface.co/{repo}")
        log("  2. Create a read token at https://huggingface.co/settings/tokens")
        log(f"  3. Run:  {py} -c \"from huggingface_hub import login; login()\"")
        raise SystemExit(1)
    try:
        auth_check(repo)
    except GatedRepoError:
        log(f"Your HuggingFace account does not have access to {repo} yet.")
        log(f"  Open https://huggingface.co/{repo} logged in with the same account as the token,")
        log("  and accept the license (access is granted immediately).")
        log("  With a fine-grained token, 'Read access to contents of all public")
        log("  gated repos you can access' must be enabled - or create a token of type Read.")
        raise SystemExit(1)
    except Exception as e:  # offline, proxy, expired token ... the download will tell
        log(f"Warning: could not check access to {repo} ({e.__class__.__name__}). Continuing.")


def to_wav(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg was not found on PATH (Windows: winget install Gyan.FFmpeg)")
    log(f"Converting {src.name} to WAV")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                    "-ac", "2", "-ar", "44100", str(dst)], check=True)


def separate(wav: Path, stem_dir: Path, stems: dict, model_name: str, model_dir: Path) -> dict[str, Path]:
    wanted = {name: stem_dir / f"{name}.wav" for name in stems}
    if all(p.exists() for p in wanted.values()):
        log("Stems already exist, skipping separation")
        return wanted
    from audio_separator.separator import Separator

    log(f"Separating with {model_name} (the model is downloaded the first time)")
    t0 = time.time()
    stem_dir.mkdir(parents=True, exist_ok=True)
    sep = Separator(output_dir=str(stem_dir), output_format="WAV", model_file_dir=str(model_dir))
    sep.load_model(model_filename=model_name)
    sep.separate(str(wav), custom_output_names={n.capitalize(): n for n in stems})
    missing = [n for n, p in wanted.items() if not p.exists()]
    if missing:
        raise SystemExit(f"Separation did not produce these stems: {missing}. Files: {list(stem_dir.iterdir())}")
    log(f"Separation done in {time.time() - t0:.0f} s")
    return wanted


def loudness_dbfs(path: Path) -> float:
    import numpy as np
    import soundfile as sf

    audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    return 20 * np.log10(rms) if rms > 0 else -120.0


def transcribe(model, path: Path, instruments: list[str] | None, midi_out: Path, grid):
    from muscriptor.events import NoteEndEvent, ProgressEvent
    from muscriptor.tokenizer.mt3 import resolve_instrument_names
    from muscriptor.utils.midi import ExportOptions

    names = resolve_instrument_names(instruments) if instruments else None
    events, last = [], -1
    for ev in model.transcribe(str(path), instruments=names):
        if isinstance(ev, ProgressEvent):
            pct = int(100 * ev.completed / max(ev.total, 1))
            if pct // 10 != last // 10:
                log(f"  {path.stem}: {pct}%")
                last = pct
            continue
        events.append(ev)

    notes = [{"pitch": ev.start_event.pitch,
              "start": round(ev.start_event.start_time, 4),
              "end": round(ev.end_time, 4),
              "instrument": ev.start_event.instrument}
             for ev in events if isinstance(ev, NoteEndEvent)]

    # A quantized MIDI file per stem, to listen to / open in a DAW next to the code.
    if notes:
        onsets = [n["start"] for n in notes]
        midi_grid = grid.with_onset_delay(onsets) if grid is not None else None
        midi = model.events_to_midi_bytes(iter(events), beat_grid=midi_grid,
                                          export=ExportOptions(quantize=True, subdivision=4))
        midi_out.write_bytes(midi)
    return notes


def main() -> None:
    p = argparse.ArgumentParser(description="Audio -> stems -> notes -> Strudel (proof of concept)")
    p.add_argument("audio", type=Path, help="Audio file (mp3, wav, flac, m4a ...)")
    p.add_argument("--out", type=Path, default=Path("out"), help="Output folder (default: out)")
    p.add_argument("--mode", choices=["stems", "mix"], default="stems",
                   help="stems: separate first (default). mix: transcribe the full mix directly")
    p.add_argument("--six-stems", action="store_true", help="htdemucs_6s: also guitar and piano")
    p.add_argument("--model", default="medium", choices=["small", "medium", "large"],
                   help="MuScriptor size (default: medium)")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    p.add_argument("--force", action="store_true", help="Run all steps again")
    p.add_argument("--start-bar", type=int, default=1)
    p.add_argument("--bars", type=int, help="Number of bars in the Strudel code (default: the whole song)")
    p.add_argument("--simplify", type=float, metavar="0-1",
                   help="Simplify into loops and parts (lossy). 0.8 is a good starting point")
    p.add_argument("--fold-bass", action="store_true", help="Fix octave errors in the bass (lossy)")
    p.add_argument("--as-audio", default="vocals",
                   help="Stems played as audio instead of notes, e.g. 'vocals,other' (default: vocals)")
    args = p.parse_args()

    src = args.audio.resolve()
    if not src.exists():
        raise SystemExit(f"File not found: {src}")
    work = (args.out / src.stem).resolve()
    if args.force and work.exists():
        shutil.rmtree(work)
    (work / "midi").mkdir(parents=True, exist_ok=True)
    notes_json = work / "notes.json"
    t_start = time.time()

    if notes_json.exists():
        log("notes.json already exists, skipping separation and transcription (use --force)")
    else:
        check_hf_token(args.model)
        mix = work / "mix.wav"
        to_wav(src, mix)

        stems_def = STEMS_6 if args.six_stems else STEMS_4
        if args.mode == "stems":
            model_name = "htdemucs_6s.yaml" if args.six_stems else "htdemucs_ft.yaml"
            stem_paths = separate(mix, work / "stems", stems_def, model_name,
                                  Path.home() / ".cache" / "audio-separator-models")
        else:
            stem_paths = {}

        from muscriptor.transcription_model import TranscriptionModel

        log(f"Loading MuScriptor ({args.model})")
        model = TranscriptionModel.load_model(weights_path=args.model,
                                              device=None if args.device == "auto" else args.device)
        log("Detecting tempo and bar lines on the full mix")
        grid = model.detect_beat_grid_for(str(mix), "best-effort")
        if grid is None:
            log("Warning: no tempo found. Pass --bpm to to_strudel.py afterwards.")
            grid_json = {}
        else:
            grid_json = {"bpm": grid.bpm, "beats_per_bar": grid.beats_per_bar,
                         "first_downbeat": grid.first_downbeat, "steady": grid.steady,
                         "beats": [round(float(b), 4) for b in grid.beats]}
            log(f"  {grid.bpm:.1f} BPM, {grid.beats_per_bar or '?'}/4, "
                f"first downbeat at {grid.first_downbeat:.2f} s"
                + ("" if grid.steady else ", tempo varies"))

        tracks = []
        if args.mode == "stems":
            for name, (kind, instruments) in stems_def.items():
                path = stem_paths[name]
                level = loudness_dbfs(path)
                if level < SILENCE_DBFS:
                    log(f"Skipping {name} (almost silent: {level:.0f} dBFS)")
                    continue
                log(f"Transcribing {name} ({', '.join(instruments) if instruments else 'all instruments'})")
                t0 = time.time()
                notes = transcribe(model, path, instruments, work / "midi" / f"{name}.mid", grid)
                if kind == "drums":
                    notes = [n for n in notes if n["instrument"] == "drums"]
                else:
                    notes = [n for n in notes if n["instrument"] != "drums"]
                log(f"  {len(notes)} notes in {time.time() - t0:.0f} s")
                tracks.append({"name": name, "kind": kind, "notes": notes})
        else:
            log("Transcribing the full mix")
            notes = transcribe(model, mix, None, work / "midi" / "mix.mid", grid)
            by_instrument: dict[str, list] = {}
            for n in notes:
                by_instrument.setdefault(n["instrument"], []).append(n)
            for inst, group in sorted(by_instrument.items(), key=lambda kv: -len(kv[1])):
                tracks.append({"name": inst, "kind": "drums" if inst == "drums" else "melodic",
                               "notes": group})
            log(f"  {len(notes)} notes in {len(tracks)} instruments")

        notes_json.write_text(json.dumps({"source": str(src), "mode": args.mode,
                                          "grid": grid_json, "tracks": tracks}, indent=1),
                              encoding="utf-8")

    out_js = work / f"{src.stem}.strudel.js"
    conv_args = [str(notes_json), "-o", str(out_js), "--start-bar", str(args.start_bar)]
    if args.bars:
        conv_args += ["--bars", str(args.bars)]
    if args.simplify:
        conv_args += ["--simplify", str(args.simplify)]
    if args.fold_bass:
        conv_args += ["--fold-bass"]
    conv_args += ["--as-audio", args.as_audio]
    to_strudel.main(conv_args)

    py = Path(sys.executable)
    audio_js = work / f"{src.stem}.audio.strudel.js"
    log(f"Done in {time.time() - t_start:.0f} s. Output in {work}")
    log(f"  Strudel code : {out_js.name}  (paste at https://strudel.cc)")
    if audio_js.exists():
        log(f"  Audio stems  : {audio_js.name}  (paste after the code; needs the sample server:)")
        log(f"                 \"{py}\" serve_samples.py \"{work}\"")
    log("  Stems / MIDI : stems  midi  (listen to them to see where errors come from)")
    log("  Try other settings without transcribing again, e.g.:")
    log(f"    \"{py}\" to_strudel.py \"{notes_json}\" --simplify 0.8")
    log(f"    \"{py}\" to_strudel.py \"{notes_json}\" --as-audio vocals,other --simplify 0.8")


if __name__ == "__main__":
    main()
