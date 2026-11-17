"""Local redirect targets and the UI's message-bearing 303 responses."""

from urllib.parse import quote, urlsplit

from fastapi.responses import RedirectResponse


def _safe_next(target: str | None, fallback: str) -> str:
    """A local path to go back to; anything else (another site) falls back."""
    if not target:
        return fallback
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or not target.startswith("/") or target.startswith("//"):
        return fallback
    return target


def _with_message(url: str, msg: str | None = None, err: str | None = None) -> str:
    base, _, query = url.partition("?")
    params = [p for p in query.split("&") if p and not p.startswith(("msg=", "err="))]
    if msg:
        params.append(f"msg={quote(msg)}")
    if err:
        params.append(f"err={quote(err)}")
    return base + ("?" + "&".join(params) if params else "")


def _redirect(url: str, msg: str | None = None, err: str | None = None) -> RedirectResponse:
    return RedirectResponse(_with_message(url, msg, err), status_code=303)
