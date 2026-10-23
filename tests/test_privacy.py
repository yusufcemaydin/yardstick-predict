"""The repository is public, so no file in it may hold anyone's private data.

The check reads every file git tracks, except binary files and the lock file. It fails on an email address, a home
folder with a real-looking user name, and text shaped like a secret key. When YARDSTICK_PRIVATE_TERMS names a file of
your own private words, one a line, it fails on those too. It names the file and line of each find, never the text
itself, so its output is safe to share. CONTRIBUTING.md says how to run it before every push.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PRIVATE_TERMS = "YARDSTICK_PRIVATE_TERMS"
"""The environment variable that names a file of your own private words."""
SKIPPED = {"uv.lock"}
"""Files never read. The lock file holds only package names, versions, addresses and hashes."""

EMAIL = re.compile(r"[\w.%+-]+@((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})")
NO_REPLY = re.compile(r"\d+\+[A-Za-z0-9-]+@users\.noreply\.github\.com")
"""The address GitHub puts on a commit in place of a real one: an account's number and name. It reaches no one."""
EXAMPLE_DOMAINS = ("example.com", "example.org", "example.net")
"""Domains set aside for examples. They never belong to anyone."""

HOME = re.compile(r"(?<![\w.~-])(?:/Users/|/home/|(?i:[a-z]:\\{1,2}users\\{1,2}))")
"""Where a home folder starts on a Mac, on Linux and on Windows. JSON writes each backslash twice."""
USER_NAME = re.compile(r"[\w.-]+")
EXAMPLE_USERS = ("me", "Jo Smith", "...")
"""Made-up user names that the code and tests use in example paths."""

_START = r"(?<![\w-])"
"""A key starts a word. So a longer word with a key's first letters inside it, such as task-notification, isn't one."""
KEYS = [
    (what, re.compile(pattern))
    for what, pattern in [
        ("an Anthropic API key", _START + r"sk-ant-[\w-]+"),
        ("an OpenAI project key", _START + r"sk-proj-[\w-]+"),
        ("an API key", _START + r"sk-[\w-]{20,}"),
        ("a Voyage API key", _START + r"pa-[\w-]{20,}"),
        ("a GitHub token", _START + r"ghp_[A-Za-z0-9]+"),
        ("a GitHub token", _START + r"github_pat_\w+"),
        ("an AWS access key", _START + r"AKIA[0-9A-Z]{16}"),
        ("a private key", r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    ]
]
"""What each kind of key looks like, the most exact first."""


def private_data(line: str, terms: Sequence[tuple[int, str]] = ()) -> list[str]:
    """What private data a line holds, such as "an email address". Never the data itself."""
    found = []
    if any(not _example_address(match) for match in EMAIL.finditer(line)):
        found.append("an email address")
    if any(_names_a_user(line[match.end() :]) for match in HOME.finditer(line)):
        found.append("a home folder with a user name")
    key = next((what for what, pattern in KEYS if pattern.search(line)), None)
    if key:
        found.append(key)
    folded = line.casefold()
    found += [f"the term on line {number} of your private terms" for number, term in terms if term in folded]
    return found


def _example_address(match: re.Match) -> bool:
    """Whether an email address is GitHub's stand-in for an account, or on a domain set aside for examples."""
    domain = match.group(1).lower()
    examples = any(domain == example or domain.endswith("." + example) for example in EXAMPLE_DOMAINS)
    return examples or bool(NO_REPLY.fullmatch(match.group()))


def _names_a_user(rest: str) -> bool:
    """Whether the text after the start of a home folder begins with a user name that isn't a made-up example.

    Text that can't be a name, such as <name>, $USER or nothing at all, is a placeholder.
    """
    for example in EXAMPLE_USERS:
        if rest.startswith(example) and not USER_NAME.match(rest, len(example)):
            return False
    return bool(USER_NAME.match(rest))


def private_terms() -> list[tuple[int, str]]:
    """Your own private words, from the file YARDSTICK_PRIVATE_TERMS names, each with its line number there.

    Case doesn't matter. Blank lines and lines that start with # are skipped. With the variable unset, there are none.
    """
    name = os.environ.get(PRIVATE_TERMS)
    if not name:
        return []
    try:
        lines = Path(name).expanduser().read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        pytest.fail(f"{PRIVATE_TERMS} names a file that can't be read as text: {error}")
    terms = [(number, line.strip().casefold()) for number, line in enumerate(lines, start=1)]
    return [(number, term) for number, term in terms if term and not term.startswith("#")]


def text_of(path: Path) -> str | None:
    """A file's text, or None when it's binary or gone. A link's text is where it points. It's never followed."""
    if path.is_symlink():
        return os.readlink(path)
    if not path.is_file():
        return None  # deleted, and the deletion not yet committed
    data = path.read_bytes()
    if b"\0" in data:
        return None  # a binary file, such as an image
    return data.decode("utf-8", errors="replace")


def scan(root: Path, names: list[str], terms: Sequence[tuple[int, str]] = ()) -> list[str]:
    """Each line of the named files that holds private data, as "path:line: what's there"."""
    found = []
    for name in names:
        text = None if name in SKIPPED else text_of(root / name)
        for number, line in enumerate((text or "").split("\n"), start=1):
            found += [f"{name}:{number}: {what}" for what in private_data(line, terms)]
    return found


def tracked_files() -> list[str]:
    """Every file git tracks in this checkout. Outside one, such as in an unpacked source package, there's no list."""
    if not (ROOT / ".git").exists():
        pytest.skip("Not a git checkout, so there's no list of tracked files to check.")
    done = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    return [os.fsdecode(name) for name in done.stdout.split(b"\0") if name]


def test_no_tracked_file_holds_private_data():
    found = scan(ROOT, tracked_files(), private_terms())
    if found:
        pytest.fail("Private data in the repository. Take it out, or use a made-up example:\n" + "\n".join(found))


# The check itself -------------------------------------------------------------------------------------------------
# The addresses, folders and keys it must find are put together from pieces, so this file never holds one whole.

AT = "@"


@pytest.mark.parametrize(
    "line",
    [
        "Write to jo" + AT + "mail.test",
        "first.last+news" + AT + "company.co.uk",
        "jo" + AT + "users.noreply.github.com",  # GitHub's kind of address, but without the account's number
        "jo" + AT + "example.community",  # not the example domain, only a name that starts like it
    ],
)
def test_an_email_address_is_found(line):
    assert private_data(line) == ["an email address"]


@pytest.mark.parametrize(
    "line",
    [
        "12345678+jo@users.noreply.github.com",
        "Write to jo@example.com, jo@example.org or jo@mail.example.net.",
        "@pytest.fixture(autouse=True)",
        "- uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0",
        "uv tool install git+https://github.com/someone/something@v0.3.0",
        "weights = inputs @ rng.normal(size=5)",
    ],
)
def test_example_addresses_and_text_that_only_looks_like_one_pass(line):
    assert private_data(line) == []


@pytest.mark.parametrize(
    "line",
    [
        '"cwd": "/Users/' + 'alice/projects/site"',
        "cd /home/" + "bob",
        "C:\\Users\\" + "carol\\Documents",
        '"cwd": "c:\\\\users\\\\' + 'dan\\\\work"',  # as JSON writes it
        "/Users/" + "me2/notes.md",  # a name that only starts like an example
        "/Users/" + "Jo/notes.md",
    ],
)
def test_a_home_folder_with_a_user_name_is_found(line):
    assert private_data(line) == ["a home folder with a user name"]


@pytest.mark.parametrize(
    "line",
    [
        'assert prompt_skills("/Users/me/notes.md says what to do") == set()',
        "python = '/Users/Jo Smith/my $env/bin/python'",
        "A path such as /Users/... isn't one.",
        "Records live in ~/.claude/projects, and in C:\\Users\\<you>\\.codex on Windows.",
        "export PATH=/home/$USER/bin:/Users/",
        "https://example.com/home/page",
    ],
)
def test_example_home_folders_and_placeholders_pass(line):
    assert private_data(line) == []


@pytest.mark.parametrize(
    "key, what",
    [
        ("sk-" + "ant-api03-" + "x" * 30, "an Anthropic API key"),
        ("sk-" + "proj-" + "x" * 8, "an OpenAI project key"),
        ("sk-" + "x" * 20, "an API key"),
        ("pa-" + "x" * 20, "a Voyage API key"),
        ("ghp" + "_" + "x" * 36, "a GitHub token"),
        ("github" + "_pat_" + "x" * 30, "a GitHub token"),
        ("AKIA" + "X" * 16, "an AWS access key"),
        ("-----BEGIN " + "OPENSSH PRIVATE KEY-----", "a private key"),
    ],
)
def test_text_shaped_like_a_key_is_found(key, what):
    assert private_data(f'API_KEY="{key}"') == [what]


@pytest.mark.parametrize(
    "line",
    [
        'KEY = "test-key-never-real"',
        "<task-notification-from-a-background-job>",
        "sk-" + "x" * 19,
        "pa-" + "x" * 19,
        "Voyage keys start with pa- and GitHub tokens with ghp_.",
    ],
)
def test_text_that_isnt_a_key_passes(line):
    assert private_data(line) == []


def test_your_private_terms_are_found_in_any_case(tmp_path, monkeypatch):
    listing = tmp_path / "private-terms.txt"
    listing.write_text("# People and clients\n\nAcme Widgets\n  bluebird  \n", encoding="utf-8")
    monkeypatch.setenv(PRIVATE_TERMS, str(listing))
    terms = private_terms()
    assert terms == [(3, "acme widgets"), (4, "bluebird")]
    assert private_data("Built for ACME WIDGETS, code name Bluebird.", terms) == [
        "the term on line 3 of your private terms",
        "the term on line 4 of your private terms",
    ]
    assert private_data("# People and clients", terms) == []  # a comment in the list is no term


def test_without_the_variable_there_are_no_private_terms_and_a_missing_file_fails(tmp_path, monkeypatch):
    monkeypatch.delenv(PRIVATE_TERMS, raising=False)
    assert private_terms() == []
    monkeypatch.setenv(PRIVATE_TERMS, str(tmp_path / "missing.txt"))
    with pytest.raises(pytest.fail.Exception, match=PRIVATE_TERMS):
        private_terms()


def test_binary_files_the_lock_file_and_deleted_files_are_skipped(tmp_path):
    address = "jo" + AT + "mail.test"
    (tmp_path / "notes.md").write_text(f"First line\nWrite to {address}\n")
    (tmp_path / "uv.lock").write_text(f"{address}\n")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\0\0" + address.encode())
    names = ["notes.md", "uv.lock", "logo.png", "deleted.md"]
    assert scan(tmp_path, names) == ["notes.md:2: an email address"]


def test_a_link_is_checked_for_where_it_points_and_never_followed(tmp_path):
    (tmp_path / "outside.txt").write_text("jo" + AT + "mail.test")
    try:
        (tmp_path / "link").symlink_to("outside.txt")
        (tmp_path / "records").symlink_to("/Users/" + "alice/records")
    except OSError:
        pytest.skip("This computer doesn't let this account make links.")
    assert scan(tmp_path, ["link", "records"]) == ["records:1: a home folder with a user name"]
