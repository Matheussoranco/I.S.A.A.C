"""Protected browser profiles remain denied when an allowed root is widened."""

from pathlib import Path

from isaac.security.path_policy import is_sensitive_path


def test_chromium_browser_credentials_are_sensitive() -> None:
    paths = [
        Path("home/AppData/Local/Google/Chrome/User Data/Default/Login Data"),
        Path("home/AppData/Local/Microsoft/Edge/User Data/Default/Cookies"),
        Path("home/.config/google-chrome/Default/Login Data"),
        Path("home/.config/chromium/Default/Login Data"),
    ]
    assert all(is_sensitive_path(path) for path in paths)
