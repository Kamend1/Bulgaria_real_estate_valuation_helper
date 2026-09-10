"""
Regression test for a real incident (2026-09-10): both chat message
partials required msg.content to be truthy before rendering ANY of an
assistant message's markup -- including the truncation banner itself. A
message that got cut off by the token cap before producing any visible
text (content="", truncated=True) rendered as literally nothing: no
bubble, no warning, no "Продължи" button. The appraiser saw a completely
empty turn with no indication anything had gone wrong.
"""
from jinja2 import Environment, FileSystemLoader

_ENV = Environment(loader=FileSystemLoader("app/templates"))


def _render(template_name: str, msg: dict) -> str:
    tpl = _ENV.get_template(template_name)
    return tpl.render(messages=[msg], calls=[], total_tokens=0, total_cost=0)


def test_empty_truncated_assistant_message_still_shows_the_warning_banner():
    msg = {"role": "assistant", "content": "", "truncated": True, "tool_calls": None, "created_at": None}
    for template_name in ("assistant/_messages.html", "market_analyst/_messages.html"):
        html = _render(template_name, msg)
        assert "alert-warn" in html, f"{template_name}: truncation banner missing for empty content"
        assert "Продължи" in html, f"{template_name}: continue button missing"
        assert "не успя да генерира видим текст" in html, f"{template_name}: no explanation for the empty bubble"


def test_normal_truncated_assistant_message_still_shows_its_text_and_banner():
    msg = {"role": "assistant", "content": "Частичен отговор", "truncated": True, "tool_calls": None, "created_at": None}
    for template_name in ("assistant/_messages.html", "market_analyst/_messages.html"):
        html = _render(template_name, msg)
        assert "Частичен отговор" in html
        assert "alert-warn" in html


def test_non_truncated_assistant_message_with_content_has_no_banner():
    msg = {"role": "assistant", "content": "Пълен отговор.", "truncated": False, "tool_calls": None, "created_at": None}
    for template_name in ("assistant/_messages.html", "market_analyst/_messages.html"):
        html = _render(template_name, msg)
        assert "Пълен отговор." in html
        assert "alert-warn" not in html
