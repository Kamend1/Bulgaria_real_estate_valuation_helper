"""
Regression test (2026-09-11, found during a stale-code/wrong-link audit):
rename_conversation_route in both app/routers/assistant.py and
app/routers/market_analyst.py used to redirect with a bare
RedirectResponse(url="/assistant/"/"/analyst/") that dropped the
?report=/?conv= query params -- the one action in the conversation-switcher
feature set that regressed back to the pre-Phase-14.3.2 session-only
behavior every sibling action (new_conversation_route, open_conversation_route)
already avoids via _assistant_redirect/_analyst_redirect. A bare redirect
here means: rename a conversation in one browser tab while a second tab has
since changed the shared session's active report/conversation, and the
first tab silently jumps to whatever the second tab last touched instead of
staying on the conversation just renamed.

Source-text assertions (not a full authenticated HTTP flow) mirror the
established pattern in tests/unit/test_real_scrape_pipeline_wiring.py for
this class of "did the fix actually land in the right function" bug.
"""
import re
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent


def _rename_route_body(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    m = re.search(r"async def rename_conversation_route\(.*?\n(?:.*\n)*?    return .*", src)
    assert m, f"rename_conversation_route not found in {path}"
    return m.group(0)


def test_assistant_rename_redirect_carries_report_and_conv():
    body = _rename_route_body(_ROOT / "app" / "routers" / "assistant.py")
    assert "_assistant_redirect(conv.report_id, conv.id)" in body
    assert 'RedirectResponse(url="/assistant/"' not in body


def test_analyst_rename_redirect_carries_conv():
    body = _rename_route_body(_ROOT / "app" / "routers" / "market_analyst.py")
    assert "_analyst_redirect(conv.id)" in body
    assert 'RedirectResponse(url="/analyst/"' not in body
