#!/usr/bin/env python3
"""Test regex patterns"""

import re

text = """Option 1 — Keep Year 1 as standard annual renewal with no cap clause; fifteen percent (15%) discount if approved; renewal in Year 2 follows standard pricing rules. Option 2 — If the client wants price protection, they must commit upfront to a three (3) year contract now, in which case no increases during Years 1–3 and Year 4 can have a ten percent (10%) cap if explicitly requested and approved. Option 3 — Offer a POC or monthly transition instead for large deals if hesitation is about committing long-term immediately."""

# Try different patterns
patterns = [
    r'Option (\d+)\s*[—–-]\s*([^.]+?\.)',  # Until first period
    r'Option (\d+)\s*[—–-]\s*((?:(?!Option \d).)+)',  # Until next "Option"
    r'Option (\d+)\s*[—–-]\s*(.+?)(?=Option \d|$)',  # Until next Option or end
]

for i, pattern in enumerate(patterns, 1):
    print(f"\nPattern {i}: {pattern}")
    matches = list(re.finditer(pattern, text, re.DOTALL))
    print(f"Found {len(matches)} matches:")
    for match in matches:
        print(f"  Option {match.group(1)}: {match.group(2)[:60]}...")
