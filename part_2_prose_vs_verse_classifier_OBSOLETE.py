"""
prose_vs_verse_classifier.py
────────────────────────────
Classifies text as PROSE or VERSE using a lightweight local HuggingFace model.

Recommended models (downloaded automatically on first run):
  • Qwen/Qwen2.5-1.5B-Instruct   (~3 GB, great balance of speed & accuracy)
  • Qwen/Qwen2.5-0.5B-Instruct   (~1 GB, fastest, slightly less accurate)
  • microsoft/phi-2               (~5 GB, strong on English literary text)

Install dependencies:
  pip install transformers torch accelerate
"""

import csv
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# ── Configuration ────────────────────────────────────────────────────────────

MODEL_ID = "microsoft/phi-2"   # ← swap to any chat model you prefer

# Use float16 on GPU, float32 on CPU (float16 on CPU causes errors on some HW)
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# ── Label & result types ─────────────────────────────────────────────────────

class Label(str, Enum):
    PROSE   = "prose"
    VERSE   = "verse"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    text:         str
    label:        Label
    raw_response: str
    model_id:     str = field(default=MODEL_ID)

    def __str__(self) -> str:
        preview = self.text.strip()[:70].replace("\n", " ")
        return f"[{self.label.value.upper():>7}]  {preview!r}"


# ── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a literary classifier. Your only job is to decide whether a given
piece of text is PROSE or VERSE.

Definitions:
- PROSE: ordinary written language without metrical structure. Sentences flow
  in paragraphs, without deliberate line breaks for rhythm or rhyme.
- VERSE: text that uses deliberate line breaks, meter, rhyme schemes, or
  rhythmic patterns (poetry, songs, ballads, verse drama, etc.).

Rules:
1. Reply with exactly one word: either  prose  or  verse  (lowercase).
2. Do NOT add explanations, punctuation, or any other text.\
"""


def _build_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Classify the following text:\n\n{text.strip()}"},
    ]


def _parse_label(raw: str) -> Label:
    token = raw.strip().lower().split()[0] if raw.strip() else ""
    if "verse" in token:
        return Label.VERSE
    if "prose" in token:
        return Label.PROSE
    return Label.UNKNOWN


# ── Model loader ─────────────────────────────────────────────────────────────

class ProseVerseClassifier:
    """
    Wraps a HuggingFace causal-LM for single-label prose/verse classification.

    Usage
    -----
    classifier = ProseVerseClassifier()          # loads model once
    result     = classifier.classify(my_text)
    results    = classifier.classify_batch(texts)
    """

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        print(f"Loading model: {model_id}")
        print("(This may take a minute on first run while the model downloads.)\n")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=TORCH_DTYPE,
            device_map="auto",
        )

        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=5,    # we only need one word
            do_sample=False,     # greedy decoding → deterministic
        )
        self._tokenizer = tokenizer
        print(f"✅ Model ready on device: {model.device}\n")

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, text: str) -> ClassificationResult:
        """Classify a single piece of text."""
        prompt = self._tokenizer.apply_chat_template(
            _build_messages(text),
            tokenize=False,
            add_generation_prompt=True,
        )
        output  = self._pipe(prompt)
        raw     = output[0]["generated_text"][len(prompt):].strip()
        return ClassificationResult(
            text=text,
            label=_parse_label(raw),
            raw_response=raw,
            model_id=self.model_id,
        )

    def classify_batch(
        self,
        texts: list[str],
        verbose: bool = True,
    ) -> list[ClassificationResult]:
        """Classify a list of texts, printing progress if verbose=True."""
        results = []
        total   = len(texts)
        for i, text in enumerate(texts, 1):
            if verbose:
                print(f"  [{i}/{total}] classifying…", end=" ", flush=True)
            r = self.classify(text)
            results.append(r)
            if verbose:
                print(f"→ {r.label.value.upper()}  (raw: {r.raw_response!r})")
        return results

    # ── Export helpers ────────────────────────────────────────────────────────

    @staticmethod
    def to_csv(results: list[ClassificationResult], path: str) -> None:
        """Write results to a CSV file."""
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["index", "label", "raw_response", "text"])
            for i, r in enumerate(results, 1):
                writer.writerow([i, r.label.value, r.raw_response.strip(), r.text.strip()])
        print(f"\nResults saved to: {path}")

    @staticmethod
    def print_summary(results: list[ClassificationResult]) -> None:
        """Print a formatted summary table."""
        from collections import Counter
        counts = Counter(r.label for r in results)

        print("\n" + "═" * 64)
        print(f"  {'LABEL':<12}  {'COUNT':>5}  {'%':>6}")
        print("─" * 64)
        total = len(results)
        for label in Label:
            n = counts.get(label, 0)
            pct = 100 * n / total if total else 0
            print(f"  {label.value:<12}  {n:>5}  {pct:>5.1f}%")
        print("═" * 64)

        print("\nDetailed results:")
        print("─" * 64)
        for i, r in enumerate(results, 1):
            preview = r.text.strip()[:65].replace("\n", " ")
            print(f"  {i:>2}. [{r.label.value.upper():>7}]  {preview}…")
        print()


# ── Sample texts ─────────────────────────────────────────────────────────────

SAMPLE_TEXTS: list[tuple[str, str]] = [
    ("verse", """
        To be, or not to be, that is the question:
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune,
        Or to take arms against a sea of troubles
        And by opposing end them.
    """),
    ("verse", """
        Two roads diverged in a yellow wood,
        And sorry I could not travel both
        And be one traveler, long I stood
        And looked down one as far as I could
        To where it bent in the undergrowth.
    """),
    ("prose", """
        It was the best of times, it was the worst of times, it was the age of
        wisdom, it was the age of foolishness, it was the epoch of belief, it was
        the epoch of incredulity, it was the season of Light, it was the season
        of Darkness.
    """),
    ("prose", """
        She walked into the room and immediately sensed that something was wrong.
        The furniture had been rearranged, the curtains drawn tight against the
        afternoon sun, and a faint smell of cigarette smoke lingered in the air,
        though no one in the house had ever smoked.
    """),
    ("prose", """
        And God said, Let there be light: and there was light. And God saw the
        light, that it was good: and God divided the light from the darkness.
    """),
]


# ── CLI entry point ───────────────────────────────────────────────────────────

def _run_demo(classifier: ProseVerseClassifier) -> None:
    print("Running demo on built-in sample texts…\n")
    texts   = [text for _, text in SAMPLE_TEXTS]
    results = classifier.classify_batch(texts)
    classifier.print_summary(results)

    # Accuracy against known labels
    correct = sum(
        r.label.value == expected
        for r, (expected, _) in zip(results, SAMPLE_TEXTS)
        if r.label != Label.UNKNOWN
    )
    print(f"Accuracy on samples: {correct}/{len(SAMPLE_TEXTS)}\n")


def _run_interactive(classifier: ProseVerseClassifier) -> None:
    print("Interactive mode — paste text and press Enter twice to classify.")
    print("Type  quit  to exit.\n")
    while True:
        lines: list[str] = []
        try:
            while True:
                line = input()
                if line.strip().lower() == "quit":
                    print("Bye!")
                    return
                if line == "" and lines:
                    break
                lines.append(line)
        except EOFError:
            break

        if not lines:
            continue

        text   = "\n".join(lines)
        result = classifier.classify(text)
        print(f"\n→ Label: {result.label.value.upper()}  (raw model reply: {result.raw_response!r})\n")


def main(argv: Optional[list[str]] = None) -> None:
    args = argv if argv is not None else sys.argv[1:]

    # Parse a simple --model flag
    model_id = MODEL_ID
    if "--model" in args:
        idx = args.index("--model")
        model_id = args[idx + 1]

    classifier = ProseVerseClassifier(model_id=model_id)

    if "--interactive" in args or "-i" in args:
        _run_interactive(classifier)
    else:
        _run_demo(classifier)

    # Optional CSV export
    if "--csv" in args:
        idx = args.index("--csv")
        out_path = args[idx + 1]
        texts   = [text for _, text in SAMPLE_TEXTS]
        results = classifier.classify_batch(texts, verbose=False)
        ProseVerseClassifier.to_csv(results, out_path)


if __name__ == "__main__":
    main()
