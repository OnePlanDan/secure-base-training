"""
Append-only experiment journal helper.

Usage:
    from journal import append_entry
    append_entry(
        title="Random baseline 3B",
        setup="Qwen2.5-3B, DPO on 2502 random HH-RLHF pairs...",
        results="| Test | Score | ... |",
        conclusions="Random subset performs similarly to full 10k...",
        next_steps="Scale to 7B to check for ceiling effects.",
    )
"""
import re
from datetime import date
from pathlib import Path

JOURNAL_PATH = Path(__file__).parent / "results" / "journal.md"


def _next_entry_number():
    """Read the journal and find the next entry number."""
    if not JOURNAL_PATH.exists():
        return 1
    text = JOURNAL_PATH.read_text()
    entries = re.findall(r"^## Entry (\d+)", text, re.MULTILINE)
    if not entries:
        return 1
    return max(int(n) for n in entries) + 1


def append_entry(title, setup, results, conclusions, next_steps=None):
    """Append a new dated entry to the journal."""
    num = _next_entry_number()
    today = date.today().isoformat()

    lines = [
        "",
        "---",
        "",
        f"## Entry {num} — {title}",
        "",
        f"**Date:** {today}",
        f"**Setup:** {setup}",
        "",
        "**Results:**",
        "",
        results,
        "",
        f"**Conclusions:** {conclusions}",
    ]
    if next_steps:
        lines.append(f"\n**Next steps:** {next_steps}")
    lines.append("")

    entry_text = "\n".join(lines)

    with open(JOURNAL_PATH, "a") as f:
        f.write(entry_text)

    print(f"Journal entry #{num} appended: {title}")
    return num
