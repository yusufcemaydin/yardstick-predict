"""Keys and passwords in a prompt are hidden before any of its text leaves this computer."""

from __future__ import annotations

import pytest
from builders import ClaudeSession
from conftest import KEY
from test_archives import START, read_claude

from yardstick_predict import embeddings, features, forecast, llm
from yardstick_predict.archives import TEXT_CHARS
from yardstick_predict.features import TaskInfo
from yardstick_predict.privacy import HIDDEN, hide_secrets

# Each one is made up here from parts. So no file in the repository holds a whole one. The privacy test looks for them.
ANTHROPIC = "sk-" + "ant-api03-" + "x" * 40
OPENAI = "sk-" + "proj-" + "y" * 40
VOYAGE = "pa-" + "z" * 40
GITHUB = "gh" + "p_" + "a1" * 18
GITHUB_FINE = "github" + "_pat_" + "b2" * 20
AWS = "AKIA" + "Q" * 16
GOOGLE = "AIza" + "c" * 35
SLACK = "xox" + "b-" + "1234-5678-abcdefghij"
STRIPE = "sk" + "_live_" + "d" * 24
WEB_TOKEN = "eyJ" + "e" * 20 + ".eyJ" + "f" * 20 + "." + "g" * 20
PRIVATE_KEY = "-----BEGIN " + "OPENSSH PRIVATE KEY-----\nAAAA\nBBBB\n-----END OPENSSH PRIVATE KEY-----"
KEYS = [ANTHROPIC, OPENAI, VOYAGE, GITHUB, GITHUB_FINE, AWS, GOOGLE, SLACK, STRIPE, WEB_TOKEN, PRIVATE_KEY]


@pytest.mark.parametrize("secret", KEYS)
def test_a_key_in_a_prompt_is_hidden(secret):
    assert hide_secrets(f"Use this one: {secret} and deploy.") == f"Use this one: {HIDDEN} and deploy."


def test_a_private_key_cut_off_before_its_end_is_hidden_to_the_end():
    assert hide_secrets("Here:\n" + "-----BEGIN " + "RSA PRIVATE KEY-----\nMIIE\nabc") == "Here:\n" + HIDDEN


@pytest.mark.parametrize(
    ("text", "hidden"),
    [
        ("export DB_PASSWORD=hunter2hunter2", f"export DB_PASSWORD={HIDDEN}"),
        ('{"api_key": "abc123def456"}', f'{{"api_key": "{HIDDEN}"}}'),
        ("secret: s3cr3t-value", f"secret: {HIDDEN}"),
        ("postgres://admin:pa55word@localhost:5432/app", f"postgres://{HIDDEN}@localhost:5432/app"),
        ("Authorization: Bearer abcdefghijklmnop1234", f"Authorization: Bearer {HIDDEN}"),
    ],
)
def test_a_password_given_to_a_name_or_in_a_web_address_is_hidden(text, hidden):
    assert hide_secrets(text) == hidden


@pytest.mark.parametrize(
    ("text", "hidden"),
    [
        ('{"password": "correct horse battery staple"}', f'{{"password": "{HIDDEN}"}}'),
        ('password="abc def ghi"', f'password="{HIDDEN}"'),
        ("token: 'a b'", f"token: '{HIDDEN}'"),
    ],
)
def test_a_password_in_quotes_is_hidden_whole(text, hidden):
    assert hide_secrets(text) == hidden
    assert hide_secrets(hidden) == hidden


@pytest.mark.parametrize("secret", [GOOGLE, AWS, VOYAGE, GITHUB])
def test_the_start_of_a_key_at_the_very_end_of_a_text_is_hidden(secret):
    assert hide_secrets("Use " + secret[:-3]) == "Use " + HIDDEN  # as when a cut to length fell inside the key


@pytest.mark.parametrize(
    "text",
    [
        "Add a login page with tests",
        "Set max_tokens=1000 and retry",
        "The tokens: 1.8M, the cost: $0.92",
        "<task-notification-from-a-background-job>",
        "Voyage keys start with pa- and GitHub tokens with ghp_.",
        "sk-learn is what some people call scikit-learn",
        "Read docs/how-it-works.md, then src/yardstick_predict/cli.py:835",
        "",
    ],
)
def test_ordinary_text_stays_as_it_is(text):
    assert hide_secrets(text) == text


def test_hiding_twice_changes_nothing_more():
    once = hide_secrets(f"Try {ANTHROPIC}, OPENAI_API_KEY={OPENAI} and password = hunter2hunter2")
    assert ANTHROPIC not in once and OPENAI not in once and "hunter2" not in once
    assert hide_secrets(once) == once


def test_voyage_gets_the_text_with_its_keys_hidden_and_the_cache_knows_the_text(voyage):
    text = f"Deploy with {ANTHROPIC} today"
    assert len(list(embeddings.embed([text], "voyage-4", KEY))) == 1
    assert [sent for body in voyage.requests for sent in body["input"]] == [f"Deploy with {HIDDEN} today"]
    assert list(embeddings.cached("voyage-4", [text])) == [text]  # so it's never sent or paid for again


LONG_PROMPT = "a " * ((TEXT_CHARS - 20) // 2) + GOOGLE + " and more"
"""A prompt whose cut to TEXT_CHARS falls inside a key. The cut alone would keep most of the key."""


def test_a_key_cut_by_the_prompt_limit_is_hidden_when_records_are_read(tmp_path):
    assert GOOGLE[:20] in LONG_PROMPT[:TEXT_CHARS] and GOOGLE not in LONG_PROMPT[:TEXT_CHARS]
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt(LONG_PROMPT)
    session.reply(5)
    session.save(mtime=START)
    [task] = read_claude(tmp_path).values()
    assert "AIza" not in task.prompt and task.prompt.endswith(f"{HIDDEN} and more")


def test_a_key_cut_by_the_prompt_limit_is_hidden_in_a_forecast_request_and_its_text():
    request = {"version": 1, "tasks": [{"id": "t1", "prompt": LONG_PROMPT, "models": ["claude-opus-5-5"]}]}
    [task], _, _ = forecast.parse(request)
    assert "AIza" not in task.prompt and task.prompt.endswith(f"{HIDDEN} and more")
    text = features.text("prompt", TaskInfo(prompt=LONG_PROMPT, model="claude-opus-5-5"))
    assert text == task.prompt  # the same text training keeps. So the cache knows its vector.


def test_a_key_cut_in_two_by_the_context_limit_is_still_hidden():
    context = "a " * 1000 + ANTHROPIC + " " + "b" * (features.CONTEXT_CHARS - 11)
    assert ANTHROPIC[-10:] in context[-features.CONTEXT_CHARS :]  # the cut alone would keep the key's end
    text = features.text("context", TaskInfo(prompt="Go on", model="claude-opus-5-5", context=context))
    assert ANTHROPIC[-10:] not in text and len(text) == features.CONTEXT_CHARS


def test_the_ai_model_reads_each_task_with_its_keys_hidden():
    context = "a " * 2000 + ANTHROPIC + " " + "b" * (llm.CONTEXT_CHARS - 11)
    message = llm._message([llm.Ask(id="t1", prompt=f"Rotate {GITHUB} today", context=context)])
    assert f"Rotate {HIDDEN} today" in message
    assert GITHUB not in message and ANTHROPIC[-10:] not in message
