"""
FPL authentication with persistent cookie caching.

FPL's login endpoint (users.premierleague.com) is protected by Datadome bot
detection which blocks all automated HTTP clients and headless browsers.
Authentication therefore relies on a session cookie extracted once from the
user's browser.  The cookie is saved to .fpl_cookies.json and reused
automatically on every subsequent run.

To set up authentication, run:
    uv run pointsball setup-auth
"""

import json
import logging
import os
import webbrowser
from pathlib import Path

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_VALIDATE_URL = "https://fantasy.premierleague.com/api/me/"
_DEFAULT_COOKIES_FILE = Path(".fpl_cookies.json")


def _team_id() -> int:
    load_dotenv()
    raw = os.environ.get("FPL_TEAM_ID")
    if not raw:
        raise ValueError("FPL_TEAM_ID is not set in .env")
    return int(raw)


def _save_cookies(cookies: dict, path: Path) -> None:
    path.write_text(json.dumps(cookies))
    logger.debug("Saved cookies to %s.", path)


def _load_cookies(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _session_from_cookies(cookies: dict) -> requests.Session:
    session = requests.Session()
    for name, value in cookies.items():
        session.cookies.set(name, value)
    return session


def _is_valid(session: requests.Session) -> bool:
    """Return True if the session is authenticated."""
    try:
        resp = session.get(_VALIDATE_URL, timeout=10)
        return resp.status_code == 200 and resp.json().get("player") is not None
    except (requests.RequestException, ValueError):
        return False


def _parse_cookie_header(raw: str) -> dict:
    """Parse a raw Cookie header string into a name→value dict."""
    cookies = {}
    for part in raw.split(";"):
        part = part.strip()
        if "=" in part:
            name, _, value = part.partition("=")
            cookies[name.strip()] = value.strip()
    return cookies


def setup_auth(cookies_file: Path = _DEFAULT_COOKIES_FILE) -> None:
    """Interactive wizard to extract and save the FPL session cookie.

    Opens the FPL API in the user's default browser and prompts them to paste
    the Cookie request header.  Only needs to be run once; the cookie is
    saved and reused automatically until it expires (typically months).
    """
    team_id = _team_id()
    url = f"https://fantasy.premierleague.com/api/my-team/{team_id}/"

    print("\n=== Pointsball: One-time FPL authentication setup ===\n")
    print("Step 1: Opening FPL in your browser...")
    webbrowser.open(url)

    print(
        "\nStep 2: In your browser:\n"
        "  a) Make sure you are logged in to fantasy.premierleague.com\n"
        "  b) Open DevTools (F12) → Network tab\n"
        "  c) Refresh the page (Cmd+R / F5)\n"
        "  d) Click the 'my-team' request in the Network list\n"
        "  e) Under 'Request Headers', find the 'Cookie:' header\n"
        "  f) Copy its full value (starts with something like 'pl_profile=...')\n"
    )

    raw = input("Step 3: Paste the Cookie value here and press Enter:\n> ").strip()

    if not raw:
        raise ValueError("No cookie value provided.")

    cookies = _parse_cookie_header(raw)

    if "pl_profile" not in cookies:
        raise ValueError(
            "The pasted value does not contain 'pl_profile'. "
            "Make sure you copied the Cookie header from a request to "
            "fantasy.premierleague.com while logged in."
        )

    session = _session_from_cookies(cookies)
    if not _is_valid(session):
        raise PermissionError(
            "The cookie was parsed but the FPL session is not valid. "
            "Make sure you are logged in to fantasy.premierleague.com and try again."
        )

    _save_cookies(cookies, cookies_file)
    print(f"\nAuthentication successful! Cookies saved to {cookies_file}.")
    print("You won't need to do this again until you log out of FPL.\n")


def get_authenticated_session(cookies_file: Path = _DEFAULT_COOKIES_FILE) -> requests.Session:
    """Return a valid authenticated requests.Session.

    Loads cached cookies from disk when available and still valid.
    Raises RuntimeError with setup instructions if no valid session exists.
    """
    cached = _load_cookies(cookies_file)
    if cached:
        session = _session_from_cookies(cached)
        if _is_valid(session):
            logger.debug("Using cached FPL session from %s.", cookies_file)
            return session
        logger.info("Cached FPL session has expired.")

    raise RuntimeError(
        "No valid FPL session found.\n" "Run the one-time setup wizard:\n\n" "    uv run pointsball setup-auth\n"
    )
