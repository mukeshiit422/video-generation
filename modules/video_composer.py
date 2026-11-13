"""
Video Composer – merges video track + audio, adds optional elements,
and produces the final Full HD deliverable via FFmpeg.
"""
import os
import subprocess
import tempfile
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger(__name__)


class VideoComposer:
    def __init__(self, cfg: dict):
        self.cfg = cfg.get("video_composition", {})

    # ── Public API ─────────────────────────────────────────────────────────────

    def compose(
        self,
        video_path: str,
        audio_path: str,
        output_path: str = "outputs/final.mp4",
        background_music: str = None,
        music_volume: float = 0.1,
        subtitle_path: str = None,
    ) -> str:
        """Merge video + audio into final file. Returns output path."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if background_music and Path(background_music).exists():
            audio_path = self._mix_audio(audio_path, background_music, music_volume)

        self._merge_video_audio(video_path, audio_path, output_path, subtitle_path)
        logger.info(f"Final video → {output_path}")
        return output_path

    def stitch_clips(
        self,
        clip_paths: list,
        audio_path: str,
        output_path: str = "outputs/final.mp4",
        crossfade_seconds: float = 0.1,
    ) -> str:
        """Concatenate multiple video clips, then merge with a single audio track."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        concat_path = self._concat_clips(clip_paths, crossfade_seconds)
        self._merge_video_audio(concat_path, audio_path, output_path)
        Path(concat_path).unlink(missing_ok=True)
        return output_path

    # ── FFmpeg Helpers ────────────────────────────────────────────────────────

    def _merge_video_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        subtitle_path: str = None,
    ):
        cfg = self.cfg
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
        ]

        vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        if subtitle_path and Path(subtitle_path).exists():
            vf += f",subtitles='{subtitle_path}'"

        cmd += [
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", cfg.get("codec", "libx264"),
            "-preset", cfg.get("preset", "slow"),
            "-crf", str(cfg.get("crf", 18)),
            "-b:v", cfg.get("video_bitrate", "8M"),
            "-c:a", cfg.get("audio_codec", "aac"),
            "-b:a", cfg.get("audio_bitrate", "192k"),
            "-pix_fmt", cfg.get("pixel_format", "yuv420p"),
            "-vf", vf,
            "-shortest",
            str(output_path),
        ]

        logger.debug(f"FFmpeg compose: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.returncode}")

    def _concat_clips(self, clip_paths: list, crossfade_seconds: float) -> str:
        """Concatenate clips with optional crossfade. Returns temp file path."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as list_file:
            for path in clip_paths:
                list_file.write(f"file '{os.path.abspath(path)}'\n")
            list_path = list_file.name

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as out:
            out_path = out.name

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            out_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        Path(list_path).unlink(missing_ok=True)
        return out_path

    def _mix_audio(
        self,
        voice_path: str,
        music_path: str,
        music_volume: float,
    ) -> str:
        """Mix voice audio with background music. Returns temp mixed file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            mixed_path = f.name

        cmd = [
            "ffmpeg", "-y",
            "-i", str(voice_path),
            "-i", str(music_path),
            "-filter_complex",
            f"[0:a]volume=1.0[v];[1:a]volume={music_volume}[m];[v][m]amix=inputs=2:duration=first[out]",
            "-map", "[out]",
            mixed_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return mixed_path

    def get_duration(self, video_path: str) -> float:
        """Return video duration in seconds."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(video_path),
        ]
        import json
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            info = json.loads(result.stdout)
            return float(info["format"]["duration"])
        except Exception:
            return 0.0

    def generate_subtitles(self, segments, output_path: str) -> str:
        """Generate a .srt subtitle file from Segment list."""
        srt_path = output_path.replace(".mp4", ".srt")
        t = 0.0
        lines = []
        for i, seg in enumerate(segments, 1):
            duration = max(1.0, len(seg.text) / 15)
            lines.append(f"{i}")
            lines.append(f"{_srt_time(t)} --> {_srt_time(t + duration)}")
            lines.append(seg.text)
            lines.append("")
            t += duration + seg.pause_after
        Path(srt_path).write_text("\n".join(lines))
        return srt_path


def _srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
