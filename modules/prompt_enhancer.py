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
from contextlib import contextmanager
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

    def enhance(self, description: str, context: str = "", style: str = "") -> dict:
        """Return {"prompt", "negative", "style"} for a plain-language description.

        context: visuals that must stay the same across calls (story scenes).
        style:   force this style instead of guessing (keeps a story consistent).
        """
        description = description.strip()
        if not description:
            raise ValueError("Empty description")

        raw = self._generate(description, context)
        result = self._parse(raw)
        if result is None:
            logger.warning(f"Enhancer returned unparseable output, using cleaned input: {raw[:200]!r}")
            result = {"prompt": _strip_filler(description), "negative": "", "style": ""}

        # The user's own words beat a 1.5B model's guess at the style
        style = style or _style_from_keywords(description) or result.get("style")
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

    # Model lifecycle: by default each call loads and unloads the LLM (so it never
    # sits in RAM next to Stable Diffusion). Wrap many calls in `with pe.session():`
    # to load it once — used by the story planner for N scene prompts.

    @contextmanager
    def session(self):
        self._load()
        try:
            yield self
        finally:
            self._unload()

    def _load(self):
        if getattr(self, "_model", None) is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._device = get_device()
        dtype = torch.float16 if self._device == "mps" or "cuda" in self._device else torch.float32
        logger.info(f"Loading prompt enhancer: {self.model_id}")
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype).to(self._device)

    def _unload(self):
        # Free the LLM immediately so Stable Diffusion has the memory.
        self._model = None
        self._tok = None
        gc.collect()
        clear_gpu_cache()

    def _chat(self, messages: list, max_new_tokens: int = 320) -> str:
        import torch
        owns_model = getattr(self, "_model", None) is None
        if owns_model:
            self._load()
        try:
            tok, model = self._tok, self._model
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(self._device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,          # deterministic: same description → same prompt
                    # Keep this mild: repetition_penalty / no_repeat_ngram_size also apply to the
                    # instructions + examples, which blocks the JSON keys and the user's own words.
                    # Loops are handled instead by style-first JSON, salvage parsing and _clean_list.
                    repetition_penalty=1.05,
                    pad_token_id=tok.eos_token_id,
                )
            return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        finally:
            if owns_model:
                self._unload()

    def _generate(self, description: str, context: str = "") -> str:
        user = description
        if context:
            user = (f"Keep these visuals identical in every scene: {context}\n"
                    f"Scene to illustrate: {description}")
        return self._chat([{"role": "system", "content": SYSTEM_PROMPT}, *FEW_SHOT,
                           {"role": "user", "content": user}])

    # ── Video planning (story vs presenter, recurring character) ──────────────

    def describe_character(self, script: str, mode: str) -> str:
        """Visual description of the story's main character, or of a fitting presenter."""
        if mode == "presenter":
            task = ("Describe the person who would present this script on camera: gender, age, "
                    "ethnicity matching the script's language and names (Hindi/Hinglish or Indian names -> Indian), "
                    "hair, clothing suited to the topic, expression.")
            example_in = "Namaste dosto! Aaj hum seekhenge photosynthesis kya hota hai."
            example_out = ("friendly Indian woman teacher in her late 20s, long black hair in a neat braid, "
                           "light blue kurta, small earrings, warm smile")
        else:
            task = ("Describe how the main character of this story looks, so an illustrator can draw "
                    "them the same way in every scene: species or person, age, size, colours, clothing, "
                    "distinctive features. If there is no single main character, describe the main subject.")
            example_in = "A little rabbit named Bunny lived near a big oak tree and loved carrots."
            example_out = ("small white rabbit with long pink ears, big brown eyes, fluffy round tail, "
                           "tiny blue scarf")
        system = (task + " Reply with ONLY the description: 12-30 words, comma-separated, "
                  "visual details only, no names, no actions, no background.")
        raw = self._chat([
            {"role": "system", "content": system},
            {"role": "user", "content": example_in},
            {"role": "assistant", "content": example_out},
            {"role": "user", "content": script[:2000]},
        ], max_new_tokens=80)
        desc = raw.strip().strip('"').splitlines()[0] if raw.strip() else ""
        return _clean_list(desc, max_items=15)

    def plan_video(self, script: str, mode: str = "") -> dict:
        """Return {"mode": "story"|"presenter", "style", "character", "setting"} for a script.

        mode: force "story" or "presenter"; anything else lets the script decide.
        """
        forced = mode if mode in ("story", "presenter") else ""
        raw = self._chat([{"role": "system", "content": PLAN_PROMPT}, *PLAN_FEW_SHOT,
                          {"role": "user", "content": script[:2500]}], max_new_tokens=200)
        data = _loose_json(raw, ("mode", "style", "character", "setting"))
        mode = data.get("mode", "").strip().lower()
        kw_mode = _mode_from_keywords(script, strict=True)
        if forced:
            mode = forced
        elif kw_mode:                     # unambiguous wording beats the small model
            mode = kw_mode
        elif mode not in ("story", "presenter"):
            mode = _mode_from_keywords(script)
        style = _style_from_keywords(script) or data.get("style", "").strip().lower()
        if style not in STYLES:
            style = "photorealistic" if mode == "presenter" else "artistic"
        character = _clean_list(data.get("character", ""), max_items=15)
        if len(character.split()) < 5 or (forced and forced != data.get("mode", "").strip().lower()):
            # Small models often leave this empty inside the JSON; a focused
            # single-task call is far more reliable.
            character = self.describe_character(script, mode)
        character = _tidy_character(character)
        if mode == "presenter" and _is_hindi(script) and "indian" not in character.lower():
            character = f"Indian {character}" if character else "friendly Indian presenter"
        plan = {
            "mode": mode,
            "style": style,
            "character": character,
            "setting": _clean_list(data.get("setting", ""), max_items=8),
        }
        logger.info(f"Video plan: {plan}")
        return plan

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


PLAN_PROMPT = f"""You plan the visuals for a narrated video, given its script.

Reply with ONLY a JSON object, keys in this order: {{"mode": "...", "style": "...", "character": "...", "setting": "..."}}

mode:
- "presenter" if one person speaks directly to the audience: greetings, "I"/"we"/"you", explaining a topic, a tutorial, news, an announcement, a lesson.
- "story" if the script narrates events or a tale about characters, animals, places or history.

character (15-30 words, visual only, comma-separated):
- story: the main recurring character's look (species or person, age, colours, clothing, distinctive features). Empty string if there is no main character.
- presenter: a fitting presenter for this script: gender, age, ethnicity matching the script's language/names/culture (Hindi or Indian names -> Indian), hair, clothing that suits the topic.

setting (5-15 words): the main place or background.

style: exactly one of {", ".join(STYLES)}. Kids stories and fables -> cartoon. Mythology, fantasy, history -> artistic. Anime -> anime. Presenter, news, real-world topics -> photorealistic.
"""

PLAN_FEW_SHOT = [
    {"role": "user", "content": "Once upon a time a little rabbit named Bunny lived near a big oak tree. "
                                "One morning she found a shiny golden key under a leaf."},
    {"role": "assistant", "content": json.dumps({
        "mode": "story", "style": "cartoon",
        "character": "small white rabbit with long pink ears, big brown eyes, fluffy round tail, wearing a tiny blue scarf",
        "setting": "sunny meadow beside a big oak tree, green grass, wildflowers"})},
    {"role": "user", "content": "Namaste dosto! Aaj hum seekhenge ki photosynthesis kya hota hai "
                                "aur plants apna khana kaise banate hain."},
    {"role": "assistant", "content": json.dumps({
        "mode": "presenter", "style": "photorealistic",
        "character": "friendly Indian woman teacher in her late 20s, long black hair in a neat braid, "
                     "light blue kurta, small earrings, warm smile",
        "setting": "bright classroom with green chalkboard and plants"})},
]


def _loose_json(raw: str, keys) -> dict:
    """Parse a JSON object from LLM output; salvage string fields if it's truncated."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return {k: str(v) for k, v in data.items()}
        except json.JSONDecodeError:
            pass
    data = {}
    for key in keys:
        m = re.search(rf'"{key}"\s*:\s*"([^"]*)', raw)
        if m:
            data[key] = m.group(1)
    return data


def _mode_from_keywords(script: str, strict: bool = False) -> str:
    """Direct address ("hello", "you", "aap") → presenter; narration ("she", "once upon") → story.

    strict=True returns "" unless one side is absent entirely (used to override the LLM).
    """
    s = script.lower()
    direct = len(re.findall(r"\b(hello|hi|namaste|welcome|dosto|friends|today we|i am|i'm|my name|"
                            r"you|your|let's|let us|subscribe|aap|hum|aaj)\b", s))
    narrative = len(re.findall(r"\b(once upon|there was|there lived|he|she|they|his|her|their|"
                               r"one day|long ago|story|in the year)\b", s))
    if strict:
        if direct == 0 and narrative > 0:
            return "story"
        if narrative == 0 and direct > 0:
            return "presenter"
        return ""
    return "presenter" if direct > narrative else "story"


def _tidy_character(desc: str) -> str:
    """Keep a character description visual and short.

    The small model sometimes returns a story summary ("Ramu lived in a village.
    Every day he worked…") or bakes a plot object into the look ("golden pot in
    hand"), which would then appear in every scene.
    """
    desc = desc.split(". ")[0]                     # first sentence only
    desc = re.sub(r"\b(lived|lives|worked|works|was|were|went|found|named \w+)\b.*", "", desc,
                  flags=re.IGNORECASE)             # drop narrative tails
    items = [t for t in (x.strip() for x in desc.split(","))
             if t and not re.search(r"\b(in hand|holding|carrying)\b", t, re.IGNORECASE)]
    return ", ".join(items)[:220].strip(" ,")


def _is_hindi(script: str) -> bool:
    if re.search(r"[\u0900-\u097F]", script):   # Devanagari
        return True
    return len(re.findall(r"\b(hai|hain|aap|hum|mein|aaj|dosto|kaise|kya|yeh|nahi|karein)\b",
                          script.lower())) >= 2


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
