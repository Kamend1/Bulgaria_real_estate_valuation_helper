from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/templates")


def _fmt_number(value) -> str:
    try:
        return f"{int(value):,}".replace(",", " ")
    except (TypeError, ValueError):
        return str(value or "—")


templates.env.filters["format_number"] = _fmt_number
