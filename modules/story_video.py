"""
Story Video – script → scene-by-scene narrated video, no character image needed.

  script ─► scenes (sentence groups) ─► per-scene voice-over (Edge-TTS)
         ─► LLM plan: recurring character + style + per-scene image prompts
         ─► Stable Diffusion image per scene (same seed + character text = consistent look)
         ─► Ken Burns zoom/pan per scene + captions ─► crossfades ─► final MP4

Memory (8 GB Macs): phases run one after another so the LLM and Stable
Diffusion are never loaded at the same time.
"""
import random
import subprocess
import textwrap
from pathlib import Path
from typing import Callable, List, Optional

from pydub import AudioSegment

from modules.script_processor import ScriptProcessor, Segment
from utils.logger import setup_logger

logger = setup_logger(__name__)

ROOT = Path(__file__).parent.parent

FORMATS = {
    # name:  (SD width, SD height, video width, video height)
    "16:9": (768, 432, 1920, 1080),
    "9:16": (432, 768, 1080, 1920),
}
FPS = 25
XFADE = 0.6                # seconds of crossfade between scenes
TRANSITIONS = ["fade", "smoothleft", "fadeblack", "smoothright"]

# Best checkpoint per style (see image_generator.MODEL_PROFILES)
MODEL_FOR_STYLE = {
    "anime": "gsdf/Counterfeit-V2.5",
    "photorealistic": "SG161222/Realistic_Vision_V5.1_noVAE",
    "cinematic": "SG161222/Realistic_Vision_V5.1_noVAE",
    "portrait": "SG161222/Realistic_Vision_V5.1_noVAE",
}
DEFAULT_MODEL = "Lykon/dreamshaper-8"

FONT_LATIN = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_DEVANAGARI = "/System/Library/Fonts/Kohinoor.ttc"


def model_for_style(style: str) -> str:
    return MODEL_FOR_STYLE.get(style, DEFAULT_MODEL)


class StoryVideoBuilder:
    def __init__(self, cfg: dict, image_gen, gpu_lock, enhancer=None):
        """
        image_gen: a shared ImageGenerator (keeps SD loaded across jobs)
        gpu_lock:  lock that serialises LLM / SD work with other jobs
        """
        self.cfg = cfg
        self.image_gen = image_gen
        self.gpu_lock = gpu_lock
        if enhancer is None:
            from modules.prompt_enhancer import PromptEnhancer
            enhancer = PromptEnhancer(cfg)
        self.enhancer = enhancer

    # ── Public ─────────────────────────────────────────────────────────────────

    def build(
        self,
        script: str,
        job_dir: Path,
        voice: str = "en-IN-NeerjaNeural",
        rate: str = "+0%",
        language: str = "en",
        fmt: str = "16:9",
        subtitles: bool = True,
        plan: Optional[dict] = None,
        mode: str = "auto",
        progress: Optional[Callable[[int, str], None]] = None,
    ) -> str:
        progress = progress or (lambda pct, msg: None)
        job_dir = Path(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        sd_w, sd_h, out_w, out_h = FORMATS.get(fmt, FORMATS["16:9"])

        scenes = self.split_scenes(ScriptProcessor(self.cfg).process(script))
        logger.info(f"Story video: {len(scenes)} scenes, format {fmt}")

        # 1. Voice-over per scene → exact scene durations
        progress(15, "Generating voice-over…")
        from modules.tts_engine import TTSEngine
        tts = TTSEngine(self.cfg)
        audio_paths, durations = [], []
        for i, scene in enumerate(scenes):
            path = str(job_dir / f"scene_{i:02d}.wav")
            tts.synthesize(scene, None, language, voice, path, rate)
            audio_paths.append(path)
            durations.append(len(AudioSegment.from_wav(path)) / 1000)

        # 2. Plan visuals + one image prompt per scene (LLM loaded once, then freed)
        progress(25, "Planning scenes…")
        texts = [" ".join(s.text for s in scene) for scene in scenes]
        with self.gpu_lock, self.enhancer.session():
            plan = plan or self.enhancer.plan_video(script, mode=mode)
            style = plan["style"]
            presenter = plan["mode"] == "presenter"
            # Each scene prompt comes from that scene's text ALONE: given the character
            # as context, the small LLM just copies it and every scene looks the same.
            # The character (story) is prefixed in code below; the setting is appended.
            prompts = []
            for i, t in enumerate(texts):
                if presenter and self._is_presenter_shot(i, len(texts)):
                    prompts.append(self._presenter_prompt(plan))
                else:
                    r = self.enhancer.enhance(t, style=style)
                    if not presenter and plan["setting"]:
                        r["prompt"] = f"{r['prompt']}, {plan['setting']}"
                    prompts.append(r)
        character = "" if presenter else plan["character"]

        # 3. One image per scene. Same seed + same character text keeps the look consistent.
        model_id = model_for_style(style)
        seed = random.randint(0, 2**31 - 1)
        image_paths = []
        for i, r in enumerate(prompts):
            progress(30 + int(45 * i / len(prompts)), f"Drawing scene {i + 1}/{len(prompts)}…")
            prompt = r["prompt"]
            if character and character.lower()[:25] not in prompt.lower():
                prompt = f"{character}, {prompt}"
            path = str(job_dir / f"scene_{i:02d}.png")
            with self.gpu_lock:
                self.image_gen.generate(
                    prompt, r["negative"], style, sd_w, sd_h, 25, 7.0, seed, model_id, path
                )
            image_paths.append(path)
            logger.info(f"Scene {i + 1}: {prompt[:120]}…")

        # 4. Animate + caption each scene, crossfade, add narration
        progress(78, "Animating scenes…")
        clips = []
        for i, (img, dur, text) in enumerate(zip(image_paths, durations, texts)):
            is_last = i == len(scenes) - 1
            clip_dur = dur + (0.4 if is_last else XFADE)  # extra tail is eaten by the crossfade
            clip = str(job_dir / f"clip_{i:02d}.mp4")
            caps = self._caption_pngs(text, dur, out_w, out_h, job_dir, i) if subtitles else []
            self._scene_clip(img, clip, clip_dur, i, out_w, out_h, caps)
            clips.append(clip)

        progress(92, "Joining scenes…")
        narration = str(job_dir / "audio.wav")
        combined = AudioSegment.silent(duration=0)
        for p in audio_paths:
            combined += AudioSegment.from_wav(p)
        combined.export(narration, format="wav")

        out = str(job_dir / "final.mp4")
        self._join(clips, durations, narration, out)
        logger.info(f"Story video saved → {out}")
        return out

    # ── Presenter scripts ──────────────────────────────────────────────────────
    # No lip-sync here (SadTalker is far too slow/heavy on an 8 GB Mac): the AI
    # presenter opens and closes the video, topic illustrations fill the middle.

    @staticmethod
    def _is_presenter_shot(i: int, n: int) -> bool:
        return i == 0 or (n > 2 and i == n - 1)

    @staticmethod
    def _presenter_prompt(plan: dict) -> dict:
        from modules.prompt_enhancer import DEFAULT_NEGATIVES
        prompt = ", ".join(x for x in (
            plan["character"] or "friendly presenter",
            "looking at camera, warm smile, talking, waist-up shot, centered",
            plan["setting"],
            "soft key light, shallow depth of field, professional video still, sharp focus",
        ) if x)
        return {"prompt": prompt, "negative": DEFAULT_NEGATIVES["photo"], "style": "photorealistic"}

    # ── Scenes ─────────────────────────────────────────────────────────────────

    @staticmethod
    def split_scenes(segments: List[Segment], max_scenes: int = 12) -> List[List[Segment]]:
        """Group sentences into scenes of ~16+ words (~6–8 s of narration each)."""
        import re
        # ScriptProcessor keeps whole paragraphs together; scenes need sentences.
        sentences = []
        for seg in segments:
            parts = [p.strip() for p in re.split(r"(?<=[.!?।])\s+", seg.text) if p.strip()]
            for k, part in enumerate(parts):
                last = k == len(parts) - 1
                sentences.append(Segment(
                    index=len(sentences), text=part,
                    pause_after=seg.pause_after if last else 0.0,
                    emotion=seg.emotion,
                    is_scene_break=seg.is_scene_break and k == 0,
                ))
        segments = sentences
        total_words = sum(len(s.text.split()) for s in segments)
        target = max(16, total_words // max_scenes + 1)
        scenes, cur, words = [], [], 0
        for seg in segments:
            if seg.is_scene_break and cur:          # explicit [SCENE]/[CUT] marker
                scenes.append(cur); cur, words = [], 0
            cur.append(seg)
            words += len(seg.text.split())
            if words >= target:
                scenes.append(cur); cur, words = [], 0
        if cur:
            if scenes and words < 6:                # don't leave a 2-word scene at the end
                scenes[-1].extend(cur)
            else:
                scenes.append(cur)
        return scenes

    # ── Rendering ──────────────────────────────────────────────────────────────

    @staticmethod
    def _motion(i: int, frames: int) -> str:
        """Slow zoom/pan, alternating per scene. Expressions run per output frame (on)."""
        center = "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        return [
            f"z='1+0.12*on/{frames}':{center}",                                   # zoom in
            f"z='1.12-0.12*on/{frames}':{center}",                                # zoom out
            f"z='1.10':x='(iw-iw/zoom)*on/{frames}':y='ih/2-(ih/zoom/2)'",        # pan →
            f"z='1.10':x='(iw-iw/zoom)*(1-on/{frames})':y='ih/2-(ih/zoom/2)'",    # pan ←
        ][i % 4]

    def _scene_clip(self, image: str, out: str, duration: float, i: int,
                    w: int, h: int, captions: list):
        frames = max(1, round(duration * FPS))
        # Work at 2× output size so the zoom stays smooth (no 1-px stepping)
        bg = (f"[0:v]scale={2*w}:{2*h}:flags=lanczos:force_original_aspect_ratio=increase,"
              f"crop={2*w}:{2*h},zoompan={self._motion(i, frames)}:d={frames}:fps={FPS}:s={w}x{h},"
              f"setsar=1,format=yuv420p[v0]")
        chain, last = [bg], "v0"
        for k, (png, start, end) in enumerate(captions):
            chain.append(f"[{last}][{k + 1}:v]overlay=0:0:enable='between(t,{start:.2f},{end:.2f})'[v{k + 1}]")
            last = f"v{k + 1}"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", image]
        for png, _, _ in captions:
            cmd += ["-i", png]
        cmd += ["-filter_complex", ";".join(chain), "-map", f"[{last}]",
                "-frames:v", str(frames), "-r", str(FPS),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Scene {i} render failed: {r.stderr[-500:]}")

    def _join(self, clips: List[str], durations: List[float], audio: str, out: str):
        """Crossfade clips; offsets line each scene up with its narration."""
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        for c in clips:
            cmd += ["-i", c]
        cmd += ["-i", audio]
        if len(clips) == 1:
            vmap, filt = "0:v", []
        else:
            filt, prev, offset = [], "0:v", 0.0
            for k in range(1, len(clips)):
                offset += durations[k - 1]
                label = f"x{k}"
                filt.append(f"[{prev}][{k}:v]xfade=transition={TRANSITIONS[k % len(TRANSITIONS)]}:"
                            f"duration={XFADE}:offset={offset:.3f}[{label}]")
                prev = label
            vmap = f"[{prev}]"
        if filt:
            cmd += ["-filter_complex", ";".join(filt)]
        cmd += ["-map", vmap, "-map", f"{len(clips)}:a",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Joining scenes failed: {r.stderr[-500:]}")

    # ── Captions (PIL; this ffmpeg build has no libass/drawtext) ────────────────

    def _caption_pngs(self, text: str, duration: float, w: int, h: int,
                      job_dir: Path, scene_idx: int) -> list:
        """Split narration into short chunks timed by length; one transparent PNG each."""
        from PIL import Image, ImageDraw, ImageFont

        words = text.split()
        if not words:
            return []
        chunk_words = 7 if w > h else 5          # narrower lines on vertical video
        chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]
        total_chars = sum(len(c) for c in chunks)

        devanagari = any("ऀ" <= ch <= "ॿ" for ch in text)
        font_size = int(min(w, h) * (0.055 if w > h else 0.06))
        try:
            font = ImageFont.truetype(FONT_DEVANAGARI if devanagari else FONT_LATIN, font_size)
        except OSError:
            font = ImageFont.load_default()

        out, t = [], 0.0
        for k, chunk in enumerate(chunks):
            span = duration * len(chunk) / total_chars
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            lines = textwrap.wrap(chunk, width=34 if w > h else 22)
            line_h = int(font_size * 1.25)
            y = int(h * (0.86 if w > h else 0.74)) - line_h * len(lines) // 2
            for line in lines:
                tw = d.textlength(line, font=font)
                x = (w - tw) / 2
                pad = font_size * 0.3
                d.rounded_rectangle([x - pad, y - pad * 0.6, x + tw + pad, y + line_h - pad * 0.2],
                                    radius=int(pad), fill=(0, 0, 0, 140))
                d.text((x, y), line, font=font, fill=(255, 255, 255, 255),
                       stroke_width=max(2, font_size // 18), stroke_fill=(0, 0, 0, 255))
                y += line_h
            path = str(job_dir / f"cap_{scene_idx:02d}_{k:02d}.png")
            img.save(path)
            out.append((path, t, t + span))
            t += span
        return out
