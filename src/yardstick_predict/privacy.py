"""Hide keys and passwords in text before it leaves this computer.

Prompts go to Voyage to be embedded. So does the end of the conversation before each one. With LLM analysis on, both
also go to the AI model that reads them. People paste keys, tokens and passwords into prompts. So before any of that
text is sent, each part of it shaped like one is replaced with [hidden].

It knows the common shapes: keys from AI and cloud services, tokens from GitHub, Slack and Stripe, private keys, web
tokens, passwords in web addresses and a value given to a name such as password or api_key. A secret with no known
shape gets through. It's still best not to paste one into a prompt.
"""

from __future__ import annotations

import re

HIDDEN = "[hidden]"

_START = r"(?<![\w-])"
"""A key starts a word. So a longer word with a key's first letters inside it, such as task-notification, isn't one."""

_SHAPES = re.compile(
    "|".join(
        [
            # A private key, from its first line to its last. When the text was cut before its end, all the rest.
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----(?:.*?-----END [A-Z0-9 ]*PRIVATE KEY-----|.*)",
            _START + r"sk-(?:ant|proj)-[\w-]+",  # Anthropic and OpenAI project keys, even when cut short
            _START + r"sk-[\w-]{20,}",  # other OpenAI keys and keys shaped like them
            _START + r"pa-[\w-]{20,}",  # Voyage
            _START + r"gh[pousr]_[A-Za-z0-9]{20,}",  # GitHub
            _START + r"github_pat_\w{20,}",
            _START + r"(?:AKIA|ASIA)[0-9A-Z]{16}",  # AWS
            _START + r"AIza[\w-]{35}",  # Google
            _START + r"xox[abposr]-[A-Za-z0-9-]{10,}",  # Slack
            _START + r"[rsp]k_(?:live|test)_[A-Za-z0-9]{16,}",  # Stripe
            _START + r"eyJ[\w-]{10,}\.eyJ[\w-]{10,}\.[\w-]{10,}",  # a signed web token
            # The start of a key at the very end of a text, as when a cut to length fell inside it
            _START + r"(?:sk-|pa-|gh[pousr]_|github_pat_|xox[abposr]-|[rsp]k_(?:live|test)_|AIza|AKIA|ASIA)[\w-]{4,}\Z",
        ]
    ),
    re.DOTALL,
)

_IN_ADDRESS = re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)")
"""The name and password inside a web address, such as the two before the @ in a database's address."""

_BEARER = re.compile(r"(?i)\b(bearer\s+)[\w.~+/-]{16,}=*")
"""The token after Bearer, as an Authorization header sends it."""

_NAMED = re.compile(
    r"(?i)(\b[\w-]*(?:password|passwd|passphrase|secret|token|api[_-]?key|access[_-]?key|private[_-]?key)[\w-]*"
    r"[\"']?\s*[:=]\s*)"
    r"(?:\"(?!\[hidden\]\")[^\"\n]+\"|'(?!\[hidden\]')[^'\n]+'|(?!\[hidden\])[^\s\"'`,;)\]}]{6,})"
)
"""A value given to a name that says it's secret, such as DB_PASSWORD=... or "api_key": "...". A value in quotes is
hidden whole, spaces and all. One without quotes is hidden up to its first space, when it has six characters or
more. The name stays. So the text still reads the same way."""


def _hide_value(found: re.Match) -> str:
    """The name and what comes before the value as they were, then the value hidden, in its quotes if it had any."""
    name = found.group(1)
    value = found.group(0)[len(name) :]
    quote = value[0] if value[0] in "\"'" else ""
    return name + quote + HIDDEN + quote


def hide_secrets(text: str) -> str:
    """The text with each part shaped like a key, token or password replaced with [hidden].

    Hiding a text twice gives the same text as hiding it once.
    """
    if not text:
        return text
    text = _SHAPES.sub(HIDDEN, text)
    text = _IN_ADDRESS.sub(HIDDEN, text)
    text = _BEARER.sub(lambda found: found.group(1) + HIDDEN, text)
    return _NAMED.sub(_hide_value, text)
