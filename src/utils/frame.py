"""
Loader for the Dempster-Shafer frame and the source-class -> focal-element map.

Every module that needs the vision head count, the head names, or the
head-to-focal-element mapping reads it from here. Nothing hardcodes it.
"""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAME_PATH = REPO_ROOT / "configs" / "frame.yaml"
CLASS_MAP_PATH = REPO_ROOT / "configs" / "class_map.yaml"

DISCARD = "discard"


class Frame:
    """The frame of discernment plus each modality's declared focal elements."""

    def __init__(self, classes: dict, vision_heads: dict, modality_heads=None):
        self.classes = classes
        # {modality: {head: [symbols]}} for every modality declared in the
        # frame. validate_modality_bpa() is parameterised over this, so a
        # modality may only commit mass where the frame says it can.
        self.modality_heads = dict(modality_heads or {})
        self.modality_heads["vision"] = dict(vision_heads)
        # Head order is the classifier's output order. Sorted so it matches
        # ImageFolder's alphabetical class_to_idx, which is asserted below.
        self.head_names = sorted(vision_heads)
        self.vision_heads = {h: list(vision_heads[h]) for h in self.head_names}

        unknown = {
            e for els in self.vision_heads.values() for e in els
        } - set(classes)
        if unknown:
            raise ValueError(
                f"{FRAME_PATH}: vision_heads reference focal elements not in "
                f"classes: {sorted(unknown)}"
            )
        if self.head_names != sorted(self.head_names):
            raise ValueError("head order must be sortable for ImageFolder")

    @property
    def num_heads(self) -> int:
        return len(self.head_names)

    def focal_element(self, head: str) -> list:
        """The subset of Theta that this head's mass is assigned to."""
        return self.vision_heads[head]

    def is_singleton(self, head: str) -> bool:
        return len(self.vision_heads[head]) == 1

    def modalities(self) -> list:
        return sorted(self.modality_heads)

    def heads_for(self, modality: str) -> dict:
        """{head: [symbols]} for one modality, in sorted head order."""
        if modality not in self.modality_heads:
            raise KeyError(
                f"{FRAME_PATH}: modality {modality!r} declares no focal "
                f"elements. Known: {self.modalities()}"
            )
        heads = self.modality_heads[modality]
        return {h: list(heads[h]) for h in sorted(heads)}

    def focal_sets_for(self, modality: str) -> set:
        """The frozensets one modality may place mass on, Theta excluded."""
        return {frozenset(v) for v in self.heads_for(modality).values()}

    def __repr__(self):
        heads = ", ".join(
            f"{h}={{{','.join(self.vision_heads[h])}}}" for h in self.head_names
        )
        return f"Frame(Theta={{{','.join(sorted(self.classes))}}}, {heads})"


def load_frame(path=FRAME_PATH) -> Frame:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("classes", "vision_heads"):
        if key not in cfg:
            raise KeyError(f"{path}: missing required key '{key}'")

    # Any "<modality>_heads" block is picked up automatically, so adding a
    # modality is a config change rather than a code change.
    modality_heads = {}
    for key, value in cfg.items():
        if key.endswith("_heads") and isinstance(value, dict):
            modality_heads[key[:-len("_heads")]] = value

    return Frame(cfg["classes"], cfg["vision_heads"], modality_heads)


def load_class_map(path=CLASS_MAP_PATH, frame: Frame = None) -> dict:
    """
    Return {source_name: {source_class: head_or_DISCARD}}.

    Validates that every mapped target is either a known head or the literal
    "discard". A typo'd head name is an error here rather than a mysteriously
    empty class directory later.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "sources" not in cfg:
        raise KeyError(f"{path}: missing required key 'sources'")

    frame = frame or load_frame()
    valid = set(frame.head_names) | {DISCARD}

    out = {}
    for source, mapping in cfg["sources"].items():
        bad = {k: v for k, v in mapping.items() if v not in valid}
        if bad:
            raise ValueError(
                f"{path}: source '{source}' maps to unknown targets {bad}. "
                f"Valid targets: {sorted(valid)}"
            )
        out[source] = dict(mapping)
    return out


def resolve(class_map: dict, source: str, class_name: str) -> str:
    """
    Map one source class to a head, or DISCARD.

    Raises on an unmapped class. A new class appearing in a re-exported dataset
    must stop the pipeline, not vanish into a silent discard.
    """
    if source not in class_map:
        raise KeyError(
            f"{CLASS_MAP_PATH}: no mapping block for source '{source}'"
        )
    mapping = class_map[source]
    if class_name not in mapping:
        raise KeyError(
            f"{CLASS_MAP_PATH}: source '{source}' class {class_name!r} is not "
            f"mapped. Add it explicitly as a head ({sorted(set(mapping.values()) - {DISCARD})}) "
            f"or as '{DISCARD}'. Refusing to guess."
        )
    return mapping[class_name]
