"""Rule-based instruction parser for rung 2 (sort_language), built for phrasings it has never seen.

    from aaltonomous.team.instruction import parse
    r = parse("Sort the objects: red into right, blue into left, yellow into center.")
    r.mapping   # {'red': 'right', 'blue': 'left', 'yellow': 'center'}  -> scene.resolve_bin() format
    r.ok        # True when every colour has a bin (or a 'default' covers it)

The output uses the same keys the official success checker understands (aaltonomous/scene/scene.py resolve_bin):
a colour ('red'), a shape ('cube' | 'cylinder' | 'triangle'), a (colour, shape) tuple, or 'default' ("everything
else"). So shape-based held-out instructions ("cubes go left") work as soon as perception reports shapes.

How it works (no templates, no network, < 1 ms):
  1. normalise: lower case, synonyms -> canonical words (crimson -> red, middle/centre -> center, prism -> triangle,
     "whatever is left" -> REST so 'left' is not read as the bin, "the right bins" (= correct) is dropped, ...);
  2. split into sentences, then into segments at commas / and / while / but / then;
  3. walk the segments keeping a buffer: subjects (colours, shapes, REST, pronouns) waiting for a bin and bins
     waiting for a subject; a segment that completes a (subject, bin) pair emits a rule. "respectively" / "in that
     order" zips all subjects with all bins of the sentence. Negations ("not the left bin", "everything except
     red") become constraints;
  4. resolve: explicit rules, then REST / 'default', then elimination (two colours on two different bins -> the
     third colour goes into the third bin).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

COLORS = ("red", "yellow", "blue")
BINS = ("left", "center", "right")
SHAPES = ("cube", "cylinder", "triangle")

# ------------------------------------------------------------------ vocabulary (canonical word <- variants)
_SYN = [
    # multi-word phrases first (applied in order)
    (r"\bleft[\s-]?over(s)?\b", " REST "),
    (r"\b(whatever|what|anything|everything|all that|those that|the ones that|ones that|objects that|items that)"
     r"\s+(is|are|'s|remains?|is still|are still)\s+(left|remaining|over)\b", " REST "),
    (r"\b(whatever|what|everything|anything)'s\s+(still\s+)?left\b", " REST "),
    (r"\b(whatever|what)\s+remains\b", " REST "),
    (r"\b(is|are)\s+(still\s+)?left\b(?!\s*(bin|box|one|side|hand|most|container|basket|tray|crate))", " REST "),
    (r"\b(the\s+)?(rest|remainder|remaining\s+(ones|objects|items|pieces|blocks|colou?rs?)|"
     r"(everything|anything|all)\s+else|other\s+(ones|objects|items|pieces|blocks|colou?rs?)|others)\b", " REST "),
    (r"\b(the\s+)?(remaining|other|last|third|leftover|spare|free|empty|unused)\s+(bin|box|one|container|basket|"
     r"tray|crate|position|spot)\b", " RESTBIN "),
    (r"\bright\s+(away|now|after|then|here|there)\b", " "),
    (r"\b(all\s*right|alright)\b", " "),
    # "the right bins/boxes" (plural) = the correct ones, never the single right bin
    (r"(?<!and )(?<!, )(?<!or )\b(the|their|its|his|her)\s+right\s+(bins|boxes|containers|places|spots|ones|"
     r"positions|baskets)\b", " "),
    (r"\b(correct|proper|appropriate|matching)\s+(bin|box|container|place)s?\b", " "),
    (r"\bfar\s+(left|right)\b", r" \1 "),
    (r"\bleft[\s-]?(most|hand|side)\b", " left "),
    (r"\bright[\s-]?(most|hand|side)\b", " right "),
    (r"\bin\s+(the\s+)?between\b", " center "),
    (r"\b(centre|centered|centred|central|middle|mid|midmost|center[\s-]?most|centermost)\b", " center "),
    (r"\b(crimson|scarlet|cherry|ruby|reds|reddish)\b", " red "),
    (r"\b(golden|gold|lemon|yellows|yellowish)\b", " yellow "),
    (r"\b(navy|azure|cobalt|cyan|blues|bluish|sky\s+blue)\b", " blue "),
    (r"\b(cubes?|cubic|cuboids?|squares?|square[\s-]shaped|boxy)\b", " cube "),
    (r"\b(cylinders?|cylindrical|round(ed)?|circles?|circular|discs?|disks?|pucks?|tubes?)\b", " cylinder "),
    (r"\b(triangles?|triangular|prisms?|wedges?|pyramids?)\b", " triangle "),
    (r"\b(do\s+not|don't|dont|does\s+not|doesn't|doesnt|never|shouldn't|should\s+not|must\s+not|mustn't|"
     r"cannot|can't|isn't|aren't|is\s+not|are\s+not|not)\b", " NOT "),
    (r"\b(except(\s+for)?|excluding|but\s+not|other\s+than|apart\s+from|save\s+for|aside\s+from|besides)\b",
     " EXCEPT "),
    (r"\b(respectively|in\s+(that|the\s+same|this)\s+order|in\s+order)\b", " RESPECTIVELY "),
    (r"\b(everything|all\s+(the\s+)?(objects|items|pieces|blocks|things)|all\s+of\s+them|every\s+(object|item|"
     r"piece|block|thing)|each\s+(object|item|piece|block))\b", " ALL "),
    (r"\b(it|they|them|these|those|which|this\s+one|that\s+one|these\s+ones|those\s+ones)\b", " PRON "),
]
_SYN = [(re.compile(p), r) for p, r in _SYN]

# Viewer-perspective markers: "my left", "from the audience's point of view" -> mirror left/right.
_MIRROR = re.compile(r"\b(my|our|the\s+viewer'?s?|viewer|audience'?s?|spectators?'?|observer'?s?|operator'?s?|"
                     r"human'?s?|person'?s?)\s+(left|right|perspective|point\s+of\s+view|view|side)\b|"
                     r"\bfacing\s+the\s+robot\b|\bfrom\s+the\s+front\b")
_ORDINAL = {"first": "left", "1st": "left", "second": "center", "2nd": "center", "third": "right", "3rd": "right"}
_ORDINAL_RE = re.compile(r"\b(first|1st|second|2nd|third|3rd)\s+(bin|box|container|basket|tray|crate)\b|"
                         r"\b(bin|box|container|basket|tray|crate)\s+(?:number\s+|no\.?\s*|#)?([123])\b")
_SENT_SPLIT = re.compile(r"[.;!?\n]+")
_SEG_SPLIT = re.compile(r",|\b(?:and|while|whereas|but|then|plus|also|after\s+that|finally|lastly|"
                        r"meanwhile|with)\b")
_TOKEN = re.compile(r"\b(red|yellow|blue|cube|cylinder|triangle|left|center|right|REST|RESTBIN|NOT|EXCEPT|"
                    r"RESPECTIVELY|ALL|PRON)\b")


@dataclass
class Parse:
    mapping: dict
    ok: bool
    rules: list = field(default_factory=list)        # (subject, bin) in the order found
    negative: list = field(default_factory=list)     # (subject, bin) that must NOT hold
    notes: list = field(default_factory=list)
    mirrored: bool = False

    def bin_for(self, color, shape=None):
        for k in ((color, shape), color, shape, "default"):
            if k is not None and k in self.mapping:
                return self.mapping[k]
        return None


def normalise(text: str) -> str:
    t = " " + text.lower().replace("’", "'").replace("-", " ") + " "
    t = re.sub(r"[\"()\[\]]", " ", t)
    t = re.sub(r"(\w)\s*:\s*", r"\1 , ", t)          # "left: red" -> segment break
    t = re.sub(r"\s*(->|=>|→|=)\s*", " to ", t)
    for rx, rep in _SYN:
        t = rx.sub(rep, t)
    return re.sub(r"\s+", " ", t).strip()


def _ordinals(t):
    if re.search(r"\b(left|center|right)\b", t):
        return t                                         # positions already named: ordinals mean something else
    t = _ORDINAL_RE.sub(lambda m: " " + (_ORDINAL.get(m.group(1) or "", None)
                                          or {"1": "left", "2": "center", "3": "right"}.get(m.group(4) or "", "")) + " ", t)
    return t


def _tokens(seg):
    """Ordered tokens of a segment with colour+shape merged into one (colour, shape) subject, NOT/EXCEPT attached."""
    raw = _TOKEN.findall(seg)
    out, i = [], 0
    while i < len(raw):
        w = raw[i]
        if w in COLORS and i + 1 < len(raw) and raw[i + 1] in SHAPES and \
                re.search(rf"\b{w}\s+(\w+\s+)?{raw[i + 1]}\b", seg):
            out.append(("subj", (w, raw[i + 1])))
            i += 2
            continue
        if w in COLORS or w in SHAPES:
            out.append(("subj", w))
        elif w in BINS:
            out.append(("bin", w))
        elif w == "REST":
            out.append(("rest", None))
        elif w == "RESTBIN":
            out.append(("restbin", None))
        elif w == "ALL":
            out.append(("all", None))
        elif w == "PRON":
            out.append(("pron", None))
        elif w == "NOT":
            out.append(("not", None))
        elif w == "EXCEPT":
            out.append(("except", None))
        elif w == "RESPECTIVELY":
            out.append(("resp", None))
        i += 1
    return out


def parse(text: str, task: str | None = None) -> Parse:
    """Instruction -> Parse. Never raises. `task='sort_fixed'` (or an empty text) falls back to the fixed mapping
    when nothing could be read."""
    notes = []
    t = normalise(text or "")
    t = _ordinals(t)
    mirrored = bool(_MIRROR.search((text or "").lower()))
    rules, negative = [], []           # (subject, bin); subject: colour | shape | (c, s) | 'default' | 'REST'
    rest_bins = []                     # subjects that go to "the remaining bin"
    excluded_from_all = []

    for sent in _SENT_SPLIT.split(t):
        toks_all = _tokens(sent)
        if not toks_all:
            continue
        if any(k == "resp" for k, _ in toks_all):  # "red, blue and yellow go left, right and center respectively"
            subs = [v for k, v in toks_all if k == "subj"]
            bins = [v for k, v in toks_all if k == "bin"]
            if subs and len(subs) == len(bins):
                rules += list(zip(subs, bins))
                notes.append("respectively")
                continue
        segs = [s for s in _SEG_SPLIT.split(sent) if s and s.strip()]
        subj_buf, bin_buf, last_subj = [], [], []
        last_bin_first = None
        pending_all = False
        for seg in segs:
            toks = _tokens(seg)
            if not toks:
                continue
            neg = False          # NOT applies to the next bin in this segment
            exc = False          # EXCEPT applies to the following subjects in this segment
            s_here, b_here, nb_here = [], [], []
            rest_here = restbin_here = all_here = pron_here = False
            for k, v in toks:
                if k == "not":
                    neg = True
                elif k == "except":
                    exc = True
                elif k == "subj":
                    if exc:
                        excluded_from_all.append(v)
                        all_here = all_here or not s_here      # "all but red" / "except red, everything"
                    else:
                        s_here.append(v)
                elif k == "bin":
                    (nb_here if neg else b_here).append(v)
                    neg = False
                elif k == "rest":
                    rest_here = True
                elif k == "restbin":
                    restbin_here = True
                elif k == "all":
                    all_here = True
                elif k == "pron":
                    pron_here = True
            if exc and not s_here and not all_here:
                all_here = True
            excl_here = [v for v in excluded_from_all if v not in last_subj] if exc else []
            # subjects of this segment (REST/ALL/pronoun only when nothing more specific is named)
            subs = list(s_here)
            if not subs and rest_here:
                subs = ["REST"]
            if not subs and (all_here or pending_all) and (b_here or restbin_here):
                subs = ["default"]
            if not subs and pron_here and last_subj:
                subs = list(last_subj)
            if all_here and not b_here and not s_here:
                pending_all = True
            # negative constraints ("red does not go in the left bin")
            for b in nb_here:
                for s in (subs or subj_buf or last_subj):
                    negative.append((s, b))
            if nb_here and not b_here:
                last_subj = list(subs or subj_buf or last_subj)
                subj_buf = []
                continue
            kinds = [k for k, _ in toks]
            bin_first = "bin" in kinds and "subj" in kinds and kinds.index("bin") < kinds.index("subj")
            if restbin_here and not b_here:
                tgt = subs or subj_buf
                if tgt:
                    rest_bins += tgt
                    last_subj, subj_buf = list(tgt), []
                    continue
            if subs and b_here:
                for s in subj_buf + subs:
                    rules.append((s, b_here[0]))
                last_subj, subj_buf, bin_buf, pending_all = subj_buf + subs, [], [], False
                last_bin_first = (b_here[0] if bin_first else None)
                if excl_here:
                    last_subj = excl_here
                if len(b_here) > 1:
                    notes.append(f"several bins in one segment, used {b_here[0]!r}: {seg.strip()!r}")
            elif subs and last_bin_first and s_here:   # "the left bin is for red, blue ..."
                rules += [(s, last_bin_first) for s in subs]
                last_subj = list(subs)
            elif subs:
                if bin_buf:
                    for s in subs:
                        rules.append((s, bin_buf[0]))
                    last_subj, bin_buf = list(subs), bin_buf[1:]
                else:
                    subj_buf += subs
            elif b_here:
                if subj_buf:
                    for s in subj_buf:
                        rules.append((s, b_here[0]))
                    last_subj, subj_buf = list(subj_buf), []
                else:
                    bin_buf += b_here
        if subj_buf and rules:   # "the left bin is for red and blue": trailing subjects join the last bin
            b = rules[-1][1]
            rules += [(s, b) for s in subj_buf]

    if mirrored:
        flip = {"left": "right", "right": "left", "center": "center"}
        rules = [(s, flip[b]) for s, b in rules]
        negative = [(s, flip[b]) for s, b in negative]
        notes.append("viewer perspective: left/right mirrored")

    # ---------------------------------------------------------------- resolve
    mapping = {}
    for s, b in rules:
        if s == "REST":
            continue
        if s in mapping and mapping[s] != b:
            notes.append(f"conflict for {s}: {mapping[s]} vs {b}; keeping the later one")
        mapping[s] = b
    for s, b in negative:
        if mapping.get(s) == b:
            notes.append(f"{s} -> {b} is negated; dropped")
            del mapping[s]
    used = {b for k, b in mapping.items() if k in COLORS}
    free_bins = [b for b in BINS if b not in used]
    colour_only = all(k in COLORS for k in mapping)
    for s, b in rules:
        if s == "REST" and not colour_only:
            mapping.setdefault("default", b)
        elif s == "REST":
            for c in COLORS:
                if c not in mapping and (c, b) not in negative:
                    mapping[c] = b
    for s in rest_bins:   # "yellow goes in the remaining bin"
        free = [b for b in BINS if b not in {v for k, v in mapping.items() if k in COLORS and k != s}]
        if len(free) == 1:
            mapping[s] = free[0]
    if "default" in mapping:
        for s in excluded_from_all:
            if s not in mapping:
                cands = [b for b in BINS if b != mapping["default"] and (s, b) not in negative]
                if len(cands) == 1:
                    mapping[s] = cands[0]
                else:
                    notes.append(f"{s} excluded from 'everything' but its bin is unknown")
    # elimination: colours without a bin, bins without a colour
    for _ in range(3):
        if "default" in mapping:
            break
        missing = [c for c in COLORS if c not in mapping]
        used = {b for k, b in mapping.items() if k in COLORS}
        free_bins = [b for b in BINS if b not in used]
        if not missing:
            break
        progress = False
        for c in missing:
            cands = [b for b in free_bins if (c, b) not in negative]
            if len(cands) == 1 and (len(missing) == 1 or any(s == c for s, _ in negative)):
                mapping[c] = cands[0]
                notes.append(f"{c} -> {cands[0]} by elimination")
                progress = True
        if not progress:
            break
    # shape rules only: nothing else needed. Colour rules: ok if every colour has a bin.
    has_shape = any(k in SHAPES or isinstance(k, tuple) for k in mapping)
    ok = bool(mapping) and ("default" in mapping or has_shape or all(c in mapping for c in COLORS))
    if not mapping and (task == "sort_fixed" or not (text or "").strip()):
        mapping = {"red": "left", "yellow": "center", "blue": "right"}
        notes.append("nothing parsed: fixed mapping")
        ok = task == "sort_fixed"
    return Parse(mapping, ok, rules, negative, notes, mirrored)


if __name__ == "__main__":
    import sys
    for line in (sys.argv[1:] or sys.stdin):
        r = parse(line.strip())
        print(f"{line.strip()!r}\n  -> {r.mapping}  ok={r.ok}  {'; '.join(r.notes)}")
