"""
Prompt Enhancer – turns a plain-language description into a Stable Diffusion prompt.

Uses a small local instruct LLM (no paid APIs). SD 1.5 models respond to
comma-separated visual keywords, ignore filler ("create an image of"), don't
know most names, and treat "no X" as "X". The LLM rewrites for all of that.

On an 8 GB Mac the LLM is loaded only for the call and unloaded straight after,
so it never sits in memory alongside the Stable Diffusion pipeline.
"""
import gc
import json
import re
from typing import Optional

from utils.logger import setup_logger
from utils.gpu_utils import get_device, clear_gpu_cache

logger = setup_logger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

STYLES = ["photorealistic", "cinematic", "portrait", "artistic", "anime", "cartoon", "3d_render"]

SYSTEM_PROMPT = f"""You convert a user's plain description into a prompt for the Stable Diffusion 1.5 image model.

Reply with ONLY a JSON object, keys in this order: {{"style": "...", "prompt": "...", "negative": "..."}}

prompt rules:
- Comma-separated keywords and short phrases, 35-70 words. No full sentences.
- Order: main subject first, then concrete visual details (face, body, hair, clothing, colours, pose, action, objects held), then setting/background, then lighting, then medium/camera, then 2-3 quality tags.
- The model does not know names. For any named person, god, character or place, describe how it LOOKS (e.g. Hanuman -> divine vanara with monkey face, orange-red complexion, golden crown, holding golden gada mace).
- Never write negations ("no text", "without people"). Put things to avoid in negative instead.
- Drop filler words like "create an image of", "please", "show me".
- Keep exactly the user's subject and intent. Do not add new people or objects they did not ask for.
- The user may write in Hinglish or with spelling mistakes; understand the meaning and write the prompt in English.
  Hinglish words: ladki = girl/young woman, ladka = boy/young man, aurat = woman, aadmi = man, bachcha = child,
  baith = sitting, khada = standing, pi rahi/pi raha = drinking, kha raha = eating, ke paas = near, me/mein = in.
  Keep the subject's gender exactly as the user said.

negative rules:
- 5-10 comma-separated defects or unwanted things for this image, plus anything the user said they do not want.
- Never put the subject itself, or its body parts, in negative.

style: exactly one of {", ".join(STYLES)}.
- Real people, products, places -> photorealistic (portrait for close-up headshots, cinematic for dramatic scenes).
- Gods, mythology, fantasy, concept art -> artistic, unless the user asks for a photo.
- Anime/manga -> anime. Cartoon/kids/Pixar -> cartoon. 3D objects/renders -> 3d_render.
"""

# One worked example anchors the format far better than rules alone for small models.
FEW_SHOT = [
    {
        "role": "user",
        "content": "a guy talking in a podcast studio with mic, blue lights, should look real, no text on screen",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "style": "photorealistic",
            "prompt": "man in his 30s talking into podcast microphone, short brown hair, light stubble, "
                      "grey sweater, sitting at wooden desk, looking at camera, friendly smile, home podcast "
                      "studio, bookshelves and plants in background, blue ambient light, warm key light on face, "
                      "shallow depth of field, bokeh, 85mm photo, sharp focus, highly detailed",
            "negative": "text, watermark, logo, deformed hands, extra fingers, deformed face, asymmetric eyes, "
                        "blurry, lowres, cartoon, painting, oversaturated, cropped head",
        }),
    },
    {
        "role": "user",
        "content": "ek ladki park me kutte ke saath khel rahi hai, cartoon jaisa",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "style": "cartoon",
            "prompt": "cheerful young girl playing with a happy golden puppy, ponytail, yellow t-shirt, "
                      "blue shorts, running on green grass, park with trees and flowers, sunny day, "
                      "bright colors, 3d cartoon character, pixar style, soft shading, highly detailed",
            "negative": "deformed, extra limbs, bad anatomy, extra fingers, ugly face, blurry, lowres, "
                        "text, watermark, realistic photo, dark, gloomy",
        }),
    },
]

# Used when the model leaves "negative" empty
DEFAULT_NEGATIVES = {
    "photo": "text, watermark, logo, deformed, bad anatomy, extra fingers, deformed hands, deformed face, "
             "asymmetric eyes, blurry, lowres, cartoon, painting, oversaturated",
    "drawn": "text, watermark, logo, deformed, bad anatomy, extra limbs, extra fingers, bad hands, "
             "ugly face, blurry, lowres, jpeg artifacts",
}


class PromptEnhancer:
    def __init__(self, cfg: Optional[dict] = None):
        cfg = (cfg or {}).get("prompt_enhancer", {})
        self.model_id = cfg.get("model", DEFAULT_MODEL)

    def enhance(self, description: str) -> dict:
        """Return {"prompt", "negative", "style"} for a plain-language description."""
        description = description.strip()
        if not description:
            raise ValueError("Empty description")

        raw = self._generate(description)
        result = self._parse(raw)
        if result is None:
            logger.warning(f"Enhancer returned unparseable output, using cleaned input: {raw[:200]!r}")
            result = {"prompt": _strip_filler(description), "negative": "", "style": ""}

        # The user's own words beat a 1.5B model's guess at the style
        style = _style_from_keywords(description) or result.get("style")
        if style not in STYLES:
            style = ""
        result["style"] = style
        drawn = style in DRAWN

        prompt = _clean_list(result["prompt"], max_items=35)
        result["prompt"] = ", ".join(t for t in prompt.split(", ") if not _conflicts_with_style(t, style))

        # Negative = safe default + a few model-specific extras that don't erase the subject
        extras = _clean_list(result["negative"], max_items=20, strip_no=True).split(", ")
        has_person = bool(PERSON_WORDS.search(result["prompt"]))
        # Words the image is supposed to contain: a negative sharing one would erase it
        prompt_words = {w.rstrip("s") for w in re.findall(r"[a-z]+", result["prompt"].lower()) if len(w) > 3}
        prompt_words |= {"landscape", "scenery"} if re.search(r"mountain|forest|sky|sea|field", result["prompt"], re.I) else set()
        extras = [
            t for t in extras
            if t and not (has_person and t.lower().startswith("human"))
            and not (drawn and t.lower() in DRAWN_STYLE_WORDS)
            and not ({w.rstrip("s") for w in re.findall(r"[a-z]+", t.lower())} & prompt_words)
        ][:8]
        base = DEFAULT_NEGATIVES["drawn" if drawn else "photo"]
        result["negative"] = _clean_list(", ".join([base, *extras]), max_items=25)
        logger.info(f"Enhanced prompt: {result['prompt'][:100]}…")
        return result

    # ── internals ──────────────────────────────────────────────────────────────

    def _generate(self, description: str) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = get_device()
        dtype = torch.float16 if device in ("mps",) or "cuda" in device else torch.float32

        logger.info(f"Loading prompt enhancer: {self.model_id}")
        tok = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype).to(device)
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *FEW_SHOT,
                        {"role": "user", "content": description}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=320,
                    do_sample=False,          # deterministic: same description → same prompt
                    # Keep this mild: repetition_penalty / no_repeat_ngram_size also apply to the
                    # instructions + examples, which blocks the JSON keys and the user's own words.
                    # Loops are handled instead by style-first JSON, salvage parsing and _clean_list.
                    repetition_penalty=1.05,
                    pad_token_id=tok.eos_token_id,
                )
            return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        finally:
            # Free the LLM immediately so Stable Diffusion has the memory.
            del model
            gc.collect()
            clear_gpu_cache()

    @staticmethod
    def _parse(raw: str) -> Optional[dict]:
        data = None
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
        if data is None:
            # Truncated / malformed JSON: salvage whichever fields are present
            data = {}
            for key in ("prompt", "negative", "style"):
                m = re.search(rf'"{key}"\s*:\s*"([^"]*)', raw)
                if m:
                    data[key] = m.group(1)
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            return None
        return {
            "prompt": prompt,
            "negative": str(data.get("negative", "")).strip(),
            "style": str(data.get("style", "")).strip().lower(),
        }


DRAWN = {"cartoon", "anime", "artistic", "3d_render"}
DRAWN_STYLE_WORDS = {"cartoon", "anime", "painting", "illustration", "drawing", "3d", "3d render", "unrealistic proportions"}
PERSON_WORDS = re.compile(
    r"\b(man|men|woman|women|girl|boy|person|people|child|kid|lady|guy|businessman|businesswoman|"
    r"creator|teacher|student|doctor|couple|family)\b", re.IGNORECASE)

# Ordered: first match wins. Matched against the user's ORIGINAL description.
STYLE_KEYWORDS = [
    ("anime",          r"\b(anime|manga|chibi)\b"),
    ("cartoon",        r"\b(cartoon|cartoonish|pixar|disney)\b"),
    ("3d_render",      r"\b(3d render|3d model|octane|blender)\b"),
    ("photorealistic", r"\b(photo|photograph|realistic|real life|real looking|dslr)\b"),
    ("artistic",       r"\b(painting|illustration|concept art|digital art|fantasy|mythology|"
                       r"god|goddess|deity|ji|bhagwan|devi|lord|hanuman|krishna|shiva|shiv|ganesh|ganesha|"
                       r"ram|rama|durga|vishnu|buddha)\b"),
]


def _style_from_keywords(description: str) -> Optional[str]:
    for style, pattern in STYLE_KEYWORDS:
        if re.search(pattern, description, re.IGNORECASE):
            return style
    return None


def _conflicts_with_style(tag: str, style: str) -> bool:
    """True for pure style tags ('anime style', '3d render', 'pixar aesthetic') that fight the chosen style."""
    t = tag.lower()
    if not re.fullmatch(
        r"(retro |realistic |cute )?(anime|cartoon|pixar|3d|photo|photorealistic|realistic)"
        r"([- ]?(style|render|rendered|rendering|character|character design|aesthetic|lighting|look))?", t):
        return False
    family = {
        "anime": ("anime",), "cartoon": ("cartoon", "pixar", "3d"), "3d_render": ("3d",),
        "photorealistic": ("photo", "realistic"), "cinematic": ("photo", "realistic"),
        "portrait": ("photo", "realistic"), "artistic": (),
    }.get(style)
    if family is None:  # no style chosen: leave the tag alone
        return False
    return not any(word in t for word in family)


def _clean_list(text: str, max_items: int, strip_no: bool = False) -> str:
    """Normalise a comma list: trim, drop 'no '/'without ' prefixes, de-dupe, cap length.

    Small LLMs sometimes loop ("in space, in space, …") or write "no text" in
    the negative prompt (which then means "text" there too).
    """
    seen, items = set(), []
    for item in text.replace("\n", ",").split(","):
        # Qwen occasionally slips in CJK characters; SD 1.5's tokenizer can't use them
        item = re.sub(r"[^\x00-\x7F\u00C0-\u024F]", "", item)
        item = item.strip().strip(".").strip()
        if strip_no:
            item = re.sub(r"^(no|without|avoid)\s+", "", item, flags=re.IGNORECASE)
        key = item.lower()
        if not item or key in seen or len(item) > 80:
            continue
        seen.add(key)
        items.append(item)
        if len(items) >= max_items:
            break
    return ", ".join(items)


def _strip_filler(text: str) -> str:
    return re.sub(
        r"^\s*(please\s+)?(create|generate|make|draw|show)( me)?( an?)?( image| picture| photo)?( of)?\s*",
        "", text, flags=re.IGNORECASE,
    ).strip()
