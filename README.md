# audio2strudel – proof of concept

Audio → stems → notes → Strudel code, for testing the quality on one song.

```
song.mp3 ─ffmpeg─► mix.wav ─Demucs (htdemucs_ft)─► drums / bass / vocals / other
                      │                                   │
                      └─ beat tracking (once) ─┐          │ MuScriptor per stem, with
                                               ▼          ▼ an instrument hint
                                          notes.json ──► to_strudel.py ──► song.strudel.js
                                                                      └──► song.audio.strudel.js
                                                                           (vocals as audio)
```

## Requirements

- **uv** – `winget install astral-sh.uv`
- **ffmpeg** – `winget install Gyan.FFmpeg`
- **A free HuggingFace account** that has accepted the license at
  <https://huggingface.co/MuScriptor/muscriptor-medium>. The script asks for your token on the
  first run and checks access before separation starts.
- An NVIDIA GPU is recommended. It also works on CPU, but a song then takes 10–30 minutes.

## Usage

```powershell
.\run.ps1 "C:\Music\song.mp3"                  # lossless: every note exactly as transcribed
.\run.ps1 song.mp3 -Simplify 0.8               # loops and parts: much shorter code (recommended)
.\run.ps1 song.mp3 -AsAudio vocals,other       # vocals and "other" as audio instead of notes
.\run.ps1 song.mp3 -StartBar 9 -Bars 16        # only an excerpt
.\run.ps1 song.mp3 -Mode mix                   # without separation, for comparison
```

Linux/macOS: `./run.sh song.mp3 --simplify 0.8`

Once a song has been transcribed, the notes are saved in `notes.json`. Running `run.ps1` again
with other settings takes a second. `-Force` transcribes from scratch.

The code is written to `out\<song>\<song>.strudel.js` and also copied to the clipboard.

## Vocals (and other stems) as audio

Vocals rarely make sense as Strudel notes, so by default they are **not** in the main file.
Instead, `song.audio.strudel.js` plays the real vocal stem, cut into one sample per bar. Strudel
triggers each sample on the downbeat, so the vocals follow the rest of the code, even if you mute
layers, change the drums or start in the middle of the song.

1. Start the sample server in a terminal (the command is also printed after every run):
   `.venv\Scripts\python serve_samples.py out\song`
2. Paste the main file into Strudel, and paste `song.audio.strudel.js` below it.
3. The browser may ask for permission to access devices on your local network: allow it.

The file also contains the vocal melody as notes, muted with `_$:`, in case you want it.
`-AsAudio vocals,other` does the same for the "other" stem (guitar, synths, keys), which is often
the hardest to transcribe. `-AsAudio none` puts everything as notes in the main file.

## Shorter code: `-Simplify`

Without `-Simplify` the code is lossless: every note is exactly as transcribed. It is compressed
with Strudel's own notation (`bd*4`, `[~ sd]*2`, `x!4`, `<bar!8>`, `[[b1 b2]!4]@8`), and drums are
written as one layer per sound when that is shorter.

But transcriptions almost never repeat exactly. A missing note here and an extra note there
means no two bars are equal. `-Simplify 0.8` finds loops of 1, 2, 4 or 8 bars, makes loops that
share at least 80% of their notes identical, and groups them into named parts:

```js
const bass_A = note(`<[e2@3 e2@3 e2@2 ~@2 e2@3 e2@3] [a2@3 a2@3 a2@2 ~@2 b2@3 b2@3]>`)
const bass_B = note(`<[c3!2 g2!2] [a2 b2]>`)
$: arrange([5, silence], [16, bass_A], [16, bass_B], [4, silence]).s("sawtooth")
```

A lower value gives shorter code and more changed notes. The comment above each layer shows how
many notes were changed. At 0.7, drum fills typically start to disappear. With `-Simplify` the
bass plays one note at a time (the lowest). `-FoldBass` also moves all bass notes into one octave,
which fixes octave errors but can also remove an intentional octave change.

## Output in `out\<song>\`

| File | Contents |
|---|---|
| `<song>.strudel.js` | The Strudel code, one `$:` layer per stem |
| `<song>.audio.strudel.js` | Stems played as audio (default: vocals) + their notes, muted |
| `audio\`, `strudel.json` | Samples per bar for the sample server |
| `stems\*.wav` | The separated stems – listen to see whether errors start in the separation |
| `midi\*.mid` | Quantized MIDI per stem – open in a DAW to see transcription errors |
| `notes.json` | Raw notes (seconds) + beat grid. Everything after this is generated from it |

## Tweak directly with `to_strudel.py`

```powershell
.venv\Scripts\python to_strudel.py out\song\notes.json --simplify 0.8 --start-bar 17 --bars 8
.venv\Scripts\python to_strudel.py out\song\notes.json --beat-scale 2      # tempo detected at half
.venv\Scripts\python to_strudel.py out\song\notes.json --bpm 128 --first-downbeat 0.42
.venv\Scripts\python to_strudel.py out\song\notes.json --key "g minor" --snap-key
.venv\Scripts\python to_strudel.py out\song\notes.json --tracks bass,drums --grid 2
.venv\Scripts\python to_strudel.py out\song\notes.json --simplify 0.8 -o out\song\song_08.strudel.js
```

Each run overwrites its output files; use `-o` to keep several versions side by side.
Run `to_strudel.py -h` for all options.

## What the code does

- **One shared grid.** Tempo and bar lines are detected once on the full mix. All stems are
  quantized to the same beats (following small tempo drifts), so the layers line up in Strudel.
- **Latency compensation.** The systematic delay between the transcription and the beats is
  measured over all notes and subtracted before rounding to 16ths.
- **Key.** Detected with Krumhansl profiles; it decides flats/sharps in note names (`eb2` rather
  than `d#2`). `--snap-key` moves notes outside the scale to the nearest scale note.

## Known limitations

- Notes held across a bar line are cut at the bar line.
- Drums are mapped from General MIDI to `bd sd hh oh cp rim lt mt ht cr rd` in the TR-909 bank.
- The sounds (`sawtooth`, `triangle` ...) are placeholders. Notes and rhythm are what is tested.
- Vocal samples play at their original tempo. If you change `setcps`, they no longer line up.
# Audio2strudel
Audio converted from mp3 to Javascript tha can be run in strudel.cc 
