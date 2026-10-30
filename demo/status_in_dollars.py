"""yardstick's Claude Code status line for the demo, always in dollars.

On a Claude subscription, Claude Code tells the status line how much of the weekly limit the account has used.
yardstick's line then shows shares of that limit, worked out from the account's own usage. The demo is filmed on a
real account. So this leaves that reading out. The line shows dollars, as it does with an API key.

Claude Code runs it as the status line's command: python -P status_in_dollars.py --home FOLDER
"""

from __future__ import annotations

import io
import json
import sys

from yardstick_predict import claude_code

try:
    event = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace"))
except ValueError:
    event = None
if isinstance(event, dict):
    event.pop("rate_limits", None)
sys.stdin = io.TextIOWrapper(io.BytesIO(json.dumps(event).encode("utf-8")), encoding="utf-8")
sys.exit(claude_code.main([*sys.argv[1:], "status"]))
