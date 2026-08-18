from fastapi.templating import Jinja2Templates

from app.services.csrf import get_or_create_csrf_token

templates = Jinja2Templates(directory="app/templates")


def _fmt_number(value) -> str:
    try:
        return f"{int(value):,}".replace(",", " ")
    except (TypeError, ValueError):
        return str(value or "—")


templates.env.filters["format_number"] = _fmt_number
# Called from base.html as csrf_token(request) — request is always in the
# template context because TemplateResponse(request, ...) injects it.
templates.env.globals["csrf_token"] = get_or_create_csrf_token
