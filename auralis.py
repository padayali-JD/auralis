#!/usr/bin/env python3
    
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

# ---------------- Banner ----------------
def banner():
    font = """

 .S_SSSs     .S       S.    .S_sSSs     .S_SSSs    S.       .S    sSSs  
.SS~SSSSS   .SS       SS.  .SS~YS%%b   .SS~SSSSS   SS.     .SS   d%%SP  
S%S   SSSS  S%S       S%S  S%S   `S%b  S%S   SSSS  S%S     S%S  d%S'    
S%S    S%S  S%S       S%S  S%S    S%S  S%S    S%S  S%S     S%S  S%|     
S%S SSSS%S  S&S       S&S  S%S    d*S  S%S SSSS%S  S&S     S&S  S&S     
S&S  SSS%S  S&S       S&S  S&S   .S*S  S&S  SSS%S  S&S     S&S  Y&Ss    
S&S    S&S  S&S       S&S  S&S_sdSSS   S&S    S&S  S&S     S&S  `S&&S   
S&S    S&S  S&S       S&S  S&S~YSY%b   S&S    S&S  S&S     S&S    `S*S  
S*S    S&S  S*b       d*S  S*S   `S%b  S*S    S&S  S*b     S*S     l*S  
S*S    S*S  S*S.     .S*S  S*S    S%S  S*S    S*S  S*S.    S*S    .S*P  
S*S    S*S   SSSbs_sdSSS   S*S    S&S  S*S    S*S   SSSbs  S*S  sSS*S   
SSS    S*S    YSSP~YSSY    S*S    SSS  SSS    S*S    YSSP  S*S  YSS'    
       SP                  SP                 SP           SP           
       Y                   Y                  Y            Y            
                                                                        
by Joby Daniel (Padayali-JD) 
"""
    try:
        width = shutil.get_terminal_size().columns
    except Exception:
        width = 80
    for line in font.splitlines():
        print(line.center(width))

# ---------------- Third-party deps ----------------
try:
    import whisper  # OpenAI Whisper
except Exception as e:
    whisper = None

try:
    from tqdm import tqdm
except Exception:
    # Fallback if tqdm isn't installed
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable or [])
        def update(self, *_args, **_kwargs):
            pass
        def close(self):
            pass

try:
    import ffmpeg  # ffmpeg-python wrapper (nice to have)
except Exception:
    ffmpeg = None

# ---------------- Helpers ----------------
def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess with inherited stdout/stderr."""
    return subprocess.run(cmd, check=check)

def ensure_deps():
    if whisper is None:
        sys.exit("[fatal] The 'openai-whisper' package is not installed. Run: pip install openai-whisper")
    if shutil.which("ffmpeg") is None:
        sys.exit("[fatal] ffmpeg is not installed or not on PATH. See script header for install instructions.")

def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\-\. ]+", "_", name).strip()
    return re.sub(r"\s+", " ", name)

def download_media(url: str, outdir: Path) -> Path:
    """Download media via yt-dlp; return downloaded filepath."""
    if shutil.which("yt-dlp") is None:
        sys.exit("[fatal] 'yt-dlp' is required to download URLs. Install with: pip install yt-dlp")
    outtmpl = str(outdir / "%(title).200B [%(id)s].%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio/best",
        "--no-playlist",
        "-o", outtmpl,
        url,
    ]
    print("[info] Downloading with yt-dlp…")
    run(cmd)
    files = sorted(outdir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        sys.exit("[fatal] Download failed: no file produced")
    return files[0]

def extract_audio(input_path: Path, out_wav: Path, sample_rate: int = 16000) -> Path:
    """Extract mono 16kHz PCM WAV using ffmpeg."""
    print("[info] Extracting audio…")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "wav",
        str(out_wav),
    ]
    run(cmd)
    return out_wav

def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT/VTT timestamp (HH:MM:SS,mmm)."""
    if seconds is None:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_txt(segments: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.get("text", "").strip() + "\n")
    print(f"[ok] Wrote TXT: {out_path}")

def write_srt(segments: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg.get("start", 0.0))
            end = format_timestamp(seg.get("end", seg.get("start", 0.0)))
            text = seg.get("text", "").strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    print(f"[ok] Wrote SRT: {out_path}")

def write_vtt(segments: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = format_timestamp(seg.get("start", 0.0)).replace(",", ".")
            end = format_timestamp(seg.get("end", seg.get("start", 0.0))).replace(",", ".")
            text = seg.get("text", "").strip()
            f.write(f"{start} --> {end}\n{text}\n\n")
    print(f"[ok] Wrote VTT: {out_path}")

def write_json(result: Dict[str, Any], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[ok] Wrote JSON: {out_path}")

# ---------------- Transcription ----------------
def transcribe(
    source: str,
    output_dir: Path,
    model_name: str = "small",
    language: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    ensure_deps()

    output_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="transcribe_tmp_"))

    try:
        if is_url(source):
            media_path = download_media(source, tmpdir)
            base_name = sanitize_filename(media_path.stem)
        else:
            media_path = Path(source)
            if not media_path.exists():
                sys.exit(f"[fatal] Input not found: {media_path}")
            base_name = sanitize_filename(media_path.stem)

        wav_path = tmpdir / f"{base_name}.wav"
        extract_audio(media_path, wav_path)

        print(f"[info] Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)

        print("[info] Transcribing… this can take a while on first run")
        try:
            import torch
            fp16 = torch.cuda.is_available()
        except Exception:
            fp16 = False

        result = model.transcribe(
            str(wav_path),
            language=language,
            fp16=fp16,
            verbose=verbose,
        )

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "id": seg.get("id"),
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", "").strip(),
            })

        outputs = {
            "language": result.get("language"),
            "text": result.get("text", "").strip(),
            "segments": segments,
            "model": model_name,
            "source": str(source),
        }

        txt_path = output_dir / f"{base_name}.txt"
        srt_path = output_dir / f"{base_name}.srt"
        vtt_path = output_dir / f"{base_name}.vtt"
        json_path = output_dir / f"{base_name}.json"

        write_txt(segments, txt_path)
        write_srt(segments, srt_path)
        write_vtt(segments, vtt_path)
        write_json(outputs, json_path)

        print("[done] Transcription complete.")
        return outputs

    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

# ---------------- CLI ----------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcribe local/online videos to text/SRT/VTT/JSON with Whisper")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", "-i", type=str, help="Path to local video/audio file")
    src.add_argument("--url", "-u", type=str, help="YouTube/HTTP(S) URL to download and transcribe")

    p.add_argument("--model", "-m", default="small", type=str,
                   choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                   help="Whisper model size (accuracy vs. speed)")
    p.add_argument("--language", "-l", default=None, type=str,
                   help="Force a language code like 'en', 'hi', 'ta'. Default: auto-detect")
    p.add_argument("--output-dir", "-o", default="transcripts", type=str,
                   help="Where to place output files")
    p.add_argument("--verbose", action="store_true", help="Whisper verbose mode")

    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    source = args.input or args.url
    outdir = Path(args.output_dir)

    try:
        transcribe(
            source=source,
            output_dir=outdir,
            model_name=args.model,
            language=args.language,
            verbose=args.verbose,
        )
        return 0
    except SystemExit as e:
        raise e
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

# ---------------- Entry ----------------
if __name__ == "__main__":
    banner()   # Show banner first
    raise SystemExit(main())
