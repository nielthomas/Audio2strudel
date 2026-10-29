<#
.SYNOPSIS
  Audio -> stems -> notes -> Strudel. Proof of concept for testing the quality on one song.

.EXAMPLE
  .\run.ps1 "C:\Music\song.mp3"

.EXAMPLE
  .\run.ps1 song.mp3 -StartBar 9 -Bars 16          # only an excerpt in the Strudel code

.EXAMPLE
  .\run.ps1 song.mp3 -Simplify 0.8                  # loops and parts: much shorter code

.EXAMPLE
  .\run.ps1 song.mp3 -AsAudio vocals,other          # play vocals and the rest as audio

.EXAMPLE
  .\run.ps1 song.mp3 -Mode mix                      # without separation, for comparison
#>
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Audio,
    [ValidateSet('stems', 'mix')][string]$Mode = 'stems',
    [switch]$SixStems,                                # htdemucs_6s: also guitar and piano
    [ValidateSet('small', 'medium', 'large')][string]$Model = 'medium',
    [int]$StartBar = 1,
    [int]$Bars = 0,                                   # 0 = the whole song
    [double]$Simplify = 0,                            # 0 = lossless. 0.8 = loops and parts (recommended)
    [switch]$FoldBass,                                # fix octave errors in the bass (lossy)
    [string[]]$AsAudio = @('vocals'),                 # stems played as audio; 'none' = everything as notes
    [string]$Out = 'out',
    [switch]$Force,                                   # run all steps again for this song
    [switch]$Cpu,                                     # install the CPU build of PyTorch
    [switch]$Reinstall                                # recreate the Python environment
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$root = $PSScriptRoot
$venv = Join-Path $root '.venv'
$py = Join-Path $venv 'Scripts\python.exe'

function Require($cmd, $hint) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Host "Missing '$cmd'. $hint" -ForegroundColor Red
        exit 1
    }
}
function Check($what) {
    if ($LASTEXITCODE -ne 0) { Write-Host "$what failed (exit $LASTEXITCODE)." -ForegroundColor Red; exit 1 }
}

Require 'uv' 'Install with: winget install astral-sh.uv   (and open a new terminal)'
Require 'ffmpeg' 'Install with: winget install Gyan.FFmpeg   (and open a new terminal)'
if (-not (Test-Path $Audio)) { Write-Host "File not found: $Audio" -ForegroundColor Red; exit 1 }

# --- Python environment (first run only) --------------------------------------
if ($Reinstall -and (Test-Path $venv)) { Remove-Item -Recurse -Force $venv }
if (-not (Test-Path $py)) {
    Write-Host 'First run: creating the Python environment. This takes a few minutes...' -ForegroundColor Cyan
    uv venv --python 3.12 $venv; Check 'uv venv'
    # --torch-backend auto picks the CUDA build of PyTorch when there is an NVIDIA GPU.
    $backend = if ($Cpu) { 'cpu' } else { 'auto' }
    uv pip install --python $py --torch-backend $backend torch torchaudio 'audio-separator[cpu]' audioread muscriptor
    if ($LASTEXITCODE -ne 0) { Remove-Item -Recurse -Force $venv; Check 'Package installation' }
}
& $py -c "import torch; print('PyTorch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (running on CPU, this takes longer)')"

# --- HuggingFace login (the MuScriptor model is free, but its license must be accepted) ---
& $py -c "import sys; from huggingface_hub import get_token; sys.exit(0 if get_token() else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host 'The MuScriptor model requires a free HuggingFace account:' -ForegroundColor Yellow
    Write-Host "  1. Accept the license at https://huggingface.co/MuScriptor/muscriptor-$Model"
    Write-Host '  2. Create a read token at https://huggingface.co/settings/tokens'
    Write-Host '  3. Paste the token below.'
    & $py -c "from huggingface_hub import login; login()"; Check 'HuggingFace login'
}

# --- Run the pipeline ---------------------------------------------------------
if (-not [System.IO.Path]::IsPathRooted($Out)) { $Out = Join-Path $root $Out }
$pipelineArgs = @((Join-Path $root 'pipeline.py'), (Resolve-Path $Audio).Path,
    '--out', $Out, '--mode', $Mode, '--model', $Model, '--start-bar', $StartBar)
if ($Bars -gt 0) { $pipelineArgs += @('--bars', $Bars) }
if ($SixStems) { $pipelineArgs += '--six-stems' }
if ($Force) { $pipelineArgs += '--force' }
if ($Simplify -gt 0) {
    $pipelineArgs += @('--simplify', $Simplify.ToString([System.Globalization.CultureInfo]::InvariantCulture))
}
if ($FoldBass) { $pipelineArgs += '--fold-bass' }
$pipelineArgs += @('--as-audio', ($AsAudio -join ','))

& $py @pipelineArgs; Check 'Pipeline'

$name = [System.IO.Path]::GetFileNameWithoutExtension($Audio)
$js = Join-Path (Join-Path $Out $name) "$name.strudel.js"
if (Test-Path $js) {
    Get-Content -Raw -Encoding UTF8 $js | Set-Clipboard
    Write-Host ''
    Write-Host "The Strudel code has been copied to the clipboard. Paste it at https://strudel.cc" -ForegroundColor Green
    Write-Host "File: $js"
}
$audioJs = Join-Path (Join-Path $Out $name) "$name.audio.strudel.js"
if (Test-Path $audioJs) {
    Write-Host ''
    Write-Host "Audio stems ($($AsAudio -join ', ')): $audioJs" -ForegroundColor Cyan
    Write-Host '  Start the sample server in a new terminal, and paste the file after the main code in Strudel:'
    Write-Host "  & `"$py`" `"$(Join-Path $root 'serve_samples.py')`" `"$(Join-Path $Out $name)`""
}
