"""RAG generation pipeline for the AI-assisted valuation backbone (Phase 7,
Tier 3).

Guardrails, per the plan doc:
  - Deterministic pre-check gate: retrieval must return >= MIN_COMPARABLES
    before any LLM call happens (no wasted spend on hopeless requests).
  - The suggested value range is computed here in Python from pool/retrieval
    stats -- the model NEVER types out that number; its job is narrative
    text only (rationale, per-comp commentary, reasoning, caveats), written
    in a fixed markdown-section format that gets parsed deterministically.
  - Tool-calling gives the model access to get_market_trend_stats/
    get_pool_stats for additional grounding, but the one number that
    actually gets displayed as "suggested value" never depends on whether
    the model chose to call a tool correctly.
  - Output tokens are capped (MAX_OUTPUT_TOKENS); every run's token usage
    and estimated cost are persisted to ai_valuation_runs -- including
    failed runs (see the try/except around the generation loop below).
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy.orm import Session

from app.db.models import AgentLlmCall, AiValuationRun, AppraisalReport
from app.services.comparable_service import get_pool_with_stats
from app.services.llm.providers import estimate_cost_usd, get_chat_model, resolve_chat_model
from app.services.llm.retriever import retrieve_comparables
from app.services.llm.tools import build_tools

MIN_COMPARABLES = 3
# Bumped again 2026-08-25 (premium-tier audit): a real gpt-5.4-pro run came
# back with the income section truncated mid-word and its last block
# (income_caveats) never written at all. Root cause, confirmed directly
# against the OpenAI Responses API: "-pro" reasoning models bill and cap
# their hidden reasoning tokens out of the SAME max_output_tokens budget as
# the visible completion -- a diagnostic call with this exact system prompt
# shape and the then-current 4200 cap came back `incomplete
# (max_output_tokens)` after spending 3354 of the 4200 tokens (80%) on
# invisible reasoning, leaving only ~850 tokens (2471 chars) for the actual
# 8-section markdown answer. Raised both caps across the board (not just for
# "-pro" models) so every provider/tier has comfortable headroom -- this is
# safe for the cheaper models too, since none of them stop early for hitting
# a ceiling they don't need; actual spend is driven by real usage, not by
# how high this cap is set. These caps are per-call (the final
# narrative-generating call in the tool loop below, though a bad case can
# still add a couple thousand tokens from earlier tool-calling turns on top
# of this), not a hard cost ceiling on their own -- cost is still bounded by
# the cheap-tier model default + rate limiting on the route, not by
# squeezing max_tokens.
MAX_OUTPUT_TOKENS = 6500
# Roughly proportional to MAX_OUTPUT_TOKENS above -- generating both sales
# and income sections (8 headed blocks instead of 4) needs more room even
# before accounting for a reasoning model's hidden token spend on top.
MAX_OUTPUT_TOKENS_WITH_INCOME = 9000
MAX_TOOL_ITERATIONS = 5  # mandatory get_pool_stats + optional get_market_trend_stats + optional compute_income_valuation + final answer, with headroom

_SYSTEM_PROMPT_BASE = """Ти подпомагаш лицензиран оценител на недвижими имоти в България, като
извършваш експертен анализ на пазарни данни за оценяван имот. Пиши със самочувствието на
оценител, който познава пазара -- конкретни, обосновани твърдения, не общи приказки.
Резултатът е ЧЕРНОВА, която ЩЕ БЪДЕ прегледана и редактирана от лицензирания оценител
преди да влезе в доклад -- но твоята роля в тази чернова е да разсъждаваш като експерт,
не да се въздържаш от анализ.

СТРОГИ ПРАВИЛА:
- Ползвай САМО данните, предоставени в съобщението на потребителя и върнати от
  инструментите -- никога не измисляй адреси, цени или характеристики извън предоставените.
- Винаги извиквай get_pool_stats(comparable_type="sale") в началото -- това е ръчно
  потвърденият от оценителя пул сравними (ако има такъв) за същия доклад. Сравни го с
  AI-извлечените сравними по-долу и коментирай изрично в РАЗСЪЖДЕНИЕ, ако се разминават
  съществено -- това е важна проверка за достоверност, не по избор.
- За допълнителен пазарен контекст можеш да извикаш get_market_trend_stats -- никога не
  смятай пазарни агрегати наум.
- Крайният диапазон на стойността по пазарен подход НЕ Е твоя задача -- изчислен е отделно
  и ще бъде показан отделно, извън твоя текст. Твоята задача в РАЗСЪЖДЕНИЕ е да обосновеш
  КЪДЕ в наблюдавания диапазон обектът вероятно се позиционира и защо -- не да изричаш
  конкретна цифра.
- Отговори СТРОГО в зададения по-долу markdown формат, точно тези заглавия, без
  допълнителен текст извън тях, и без да измисляш нови заглавия."""

_SALES_PROMPT_SECTIONS = """
## ОПИСАНИЕ НА ИМОТА
(кратко, конкретно описание на оценявания имот -- обедини структурираните данни за обекта
с предоставеното от оценителя свободно описание по-долу, ако има такова; пиши все едно
представяш имота в началото на доклад, не просто изреждаш полетата)

## ОБОСНОВКА
(защо предоставените сравними са подходящи за обекта)

## КОМЕНТАР ПО СРАВНИМИ
- [id на обявата]: кратък коментар (локация/етаж/състояние/площ спрямо обекта)
(по един ред на всяко сравнимо, точно колкото са предоставени)

## РАЗСЪЖДЕНИЕ
(експертен мост от конкретните характеристики на обекта -- площ, етаж, строителство,
състояние по описанието -- към позицията му спрямо диапазона на сравнимите: горна,
долна или средна част от наблюдаваните цени на кв.м, и защо точно там; включи и
сравнение с get_pool_stats резултата, ако е поискано по-горе; без да изричаш
конкретно число за крайната стойност)

## ОГРАНИЧЕНИЯ
(допускания, ограничения на извадката, какво оценителят трябва да провери допълнително)"""

_INCOME_PROMPT_SECTIONS = """

За доходния подход: извикай compute_income_valuation с наем от предоставените наемни
сравними по-долу И sale_price_per_sqm от предоставените продажни сравними/пула по-горе
(за да се изчислят доходностите -- пропускането на sale_price_per_sqm е позволено, но
означава че gross_yield_pct/net_yield_pct няма да могат да бъдат изчислени, затова
включвай го винаги, когато имаш продажна статистика). Може да отклониш параметрите на
допусканията (cap rate, незаетост, разходи, ръст, хоризонт, терминален cap rate) от
подразбиращите се стойности в рамките на позволените граници (виж описанието на
инструмента), но ВСЯКО отклонение от подразбиращата се стойност ТРЯБВА да бъде обяснено
в ДОХОДЕН РАЗСЪЖДЕНИЕ по-долу -- кой параметър си променил и защо. Инструментът връща и
sensitivity -- таблица на стойността при вариране на cap rate и наем -- коментирай какво
показва тя за чувствителността на извода (напр. колко силно влияе избраната cap rate).
Числовите резултати (NOI, доходност, стойност по капитализация, DCF, sensitivity) НЕ СА
твоя задача да пишеш -- инструментът ги връща и ще бъдат показани отделно.

## ДОХОДЕН ОБОСНОВКА
(защо предоставените наемни сравними са подходящи за обекта)

## ДОХОДЕН КОМЕНТАР
- [id на обявата]: кратък коментар
(по един ред на всяко наемно сравнимо)

## ДОХОДЕН РАЗСЪЖДЕНИЕ
(избраните допускания и защо, особено ако се различават от подразбиращите се стойности;
коментар върху sensitivity резултата -- доколко изводът зависи от избраната cap rate/наем)

## ДОХОДНИ ОГРАНИЧЕНИЯ
(допускания и ограничения специфични за доходния подход)"""

_SECTION_KEYS = {
    "ОПИСАНИЕ НА ИМОТА": "property_description",
    "ОБОСНОВКА": "comparable_selection_rationale",
    "КОМЕНТАР ПО СРАВНИМИ": "comparable_commentary",
    "РАЗСЪЖДЕНИЕ": "value_reasoning",
    "ОГРАНИЧЕНИЯ": "caveats",
}
_INCOME_SECTION_KEYS = {
    "ДОХОДЕН ОБОСНОВКА": "income_rationale",
    "ДОХОДЕН КОМЕНТАР": "income_commentary",
    "ДОХОДЕН РАЗСЪЖДЕНИЕ": "income_reasoning",
    "ДОХОДНИ ОГРАНИЧЕНИЯ": "income_caveats",
}
_ALL_SECTION_KEYS = {**_SECTION_KEYS, **_INCOME_SECTION_KEYS}
_SECTION_PATTERN = re.compile(
    r"##\s*(" + "|".join(re.escape(k) for k in _ALL_SECTION_KEYS) + r")\s*\n(.*?)(?=\n##|\Z)",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class GenerationProgress:
    status: str = "running"   # running | done | error
    step: str = "Извличане на сравними…"
    tokens_so_far: int = 0
    result: dict | None = None
    error: str | None = None


def _extract_text(content) -> str:
    """AIMessage.content is a plain str for OpenAI/Anthropic in the common
    case, but a list of content blocks (e.g. [{"type": "text", "text":
    "..."}]) for langchain-google-genai (and potentially other providers
    later) -- verified empirically, not documented consistently across
    integration packages. Normalizes to plain text, ignoring non-text
    blocks (images, signatures, etc.). Without this, string concatenation
    against a list content crashes with a TypeError on any non-OpenAI/
    Anthropic provider."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _json_safe(value):
    """Recursively converts Decimal/date/datetime -- which raw SQL rows
    (retriever.py, comparable_service's text()-based queries) commonly
    contain -- into plain JSON-serializable types. SQLAlchemy's JSONB
    column has no Decimal-aware encoder by default; without this,
    committing AiValuationRun.output crashes on the first Numeric column."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _persist_call_log(db: Session, run_id, provider: str, model_id: str, call_log: list[dict]) -> None:
    """Bulk-inserts one AgentLlmCall row per entry in call_log (Tier 1,
    2026-08-26), linked to the just-created ai_valuation_runs row. Best-
    effort: a failure here must never take down an otherwise-successful (or
    already-logged-as-failed) generation -- the aggregate ai_valuation_runs
    row is the load-bearing record either way, this is supplementary
    detail."""
    if not call_log:
        return
    try:
        for entry in call_log:
            cost = estimate_cost_usd(model_id, entry["input_tokens"], entry["output_tokens"], provider=provider)
            db.add(AgentLlmCall(
                ai_valuation_run_id=run_id,
                call_label=entry["call_label"],
                provider=provider,
                model=model_id,
                input_tokens=entry["input_tokens"],
                output_tokens=entry["output_tokens"],
                estimated_cost_usd=cost,
            ))
        db.commit()
    except Exception:
        db.rollback()


def _format_comparables(comps: list[dict]) -> str:
    lines = []
    for c in comps:
        dist = c.get("distance")
        closeness = round((1 - float(dist)) * 100) if dist is not None else None
        lines.append(
            f"- id {c['id']}: {c.get('title_city_model') or ''} {c.get('title_geo_2_model') or ''}, "
            f"{c.get('area_sqm_model')} кв.м, {c.get('price_per_sqm_model')} EUR/кв.м, "
            f"{c.get('construction_type_model') or '—'} {c.get('construction_year_model') or ''}"
            + (f", близост {closeness}%" if closeness is not None else "")
        )
    return "\n".join(lines)


def _parse_sections(text: str) -> dict:
    sections = {v: "" for v in _ALL_SECTION_KEYS.values()}
    for match in _SECTION_PATTERN.finditer(text):
        heading = match.group(1).strip().upper()
        key = _ALL_SECTION_KEYS.get(heading)
        if key:
            sections[key] = match.group(2).strip()
    return sections


def _compute_value_range(comps: list[dict], subject_area) -> tuple[float | None, float | None]:
    """Deterministic -- never left to the model. Always derived from the
    SAME AI-retrieved comparables the narrative discusses (comps, not the
    manual pool) -- audit finding 2026-08-25: an earlier version preferred
    the manual pool's stats when non-empty, which silently produced a
    number computed from a DIFFERENT set than the one being narrated
    (confusing when a manual pool already existed, which is the common
    case). The manual pool is still surfaced to the model as an explicit
    cross-reference via the now-mandatory get_pool_stats tool call, not by
    quietly swapping which set the headline number comes from."""
    ppsqms = sorted(float(c["price_per_sqm_model"]) for c in comps if c.get("price_per_sqm_model"))
    if not ppsqms:
        return None, None
    if len(ppsqms) >= 4:
        q = statistics.quantiles(ppsqms, n=4)
        p25, p75 = q[0], q[2]
    else:
        p25, p75 = ppsqms[0], ppsqms[-1]

    if not subject_area:
        return None, None
    area = float(subject_area)
    return round(p25 * area, 0), round(p75 * area, 0)


def generate_valuation_backbone(
    db: Session,
    report: AppraisalReport,
    on_progress: Callable[[GenerationProgress], None] | None = None,
    provider: str | None = None,
    model: str | None = None,
    include_income: bool = False,
) -> GenerationProgress:
    """Synchronous, blocking (real network calls) -- run on a background
    thread from the router, never directly in an async route handler.
    Calls on_progress(...) at each meaningful step so an SSE endpoint can
    relay live status/token-count updates to the browser.

    provider: "openai" | "anthropic" | "google_genai", overriding
        settings.llm_default_provider for this one call -- lets the UI
        offer a per-generation provider choice rather than requiring a
        restart with a different LLM_DEFAULT_PROVIDER in .env.
    model: provider-specific model id, overriding providers.py's cheap-tier
        default for this one call -- lets the UI offer a model-tier choice
        alongside the provider choice (see providers.list_available_models).

    include_income: (Phase 7, Tier 5) also retrieve rent comparables and
        generate an income-approach section (direct capitalization + DCF
        via the compute_income_valuation tool). Independent guardrail from
        the sales section -- insufficient rent comparables degrades the
        income section to "unavailable" without blocking the (always
        required) sales section."""
    progress = GenerationProgress()

    def emit():
        if on_progress:
            on_progress(progress)

    emit()

    comps = retrieve_comparables(db, report, k=6, comparable_type="sale")
    if len(comps) < MIN_COMPARABLES:
        progress.status = "error"
        progress.error = (
            f"Намерени са само {len(comps)} AI-сравними от вид "
            f"\"{report.subject_property_type or '—'}\" (нужни поне {MIN_COMPARABLES}) — "
            f"генерацията е пропусната, за да не се хабят разходи без достатъчно основа."
        )
        emit()
        return progress

    progress.step = "Изчисляване на статистика на пула…"
    emit()
    pool = get_pool_with_stats(db, "sale", report.id)
    pool_stats = pool["stats"]
    value_low, value_high = _compute_value_range(comps, report.subject_area_sqm)

    # Income approach: independent guardrail -- insufficient rent comps
    # degrades this section gracefully rather than failing the whole
    # generation (the sales section above already has what it needs).
    income_available = False
    income_unavailable_reason: str | None = None
    rent_comps: list[dict] = []
    rent_pool_stats: dict | None = None
    if include_income:
        progress.step = "Извличане на наемни сравними…"
        emit()
        rent_comps = retrieve_comparables(db, report, k=6, comparable_type="rent")
        if len(rent_comps) < MIN_COMPARABLES:
            income_unavailable_reason = (
                f"Намерени са само {len(rent_comps)} AI-наемни сравними от вид "
                f"\"{report.subject_property_type or '—'}\" (нужни поне {MIN_COMPARABLES})."
            )
        else:
            rent_pool = get_pool_with_stats(db, "rent", report.id)
            rent_pool_stats = rent_pool["stats"]
            income_available = True

    progress.step = "Изготвяне на наратив (AI)…"
    emit()

    tools = build_tools(db, report)
    # Resolved explicitly rather than sniffed off the constructed model
    # afterward -- OpenAI's class exposes `model_name`, Anthropic's/Google's
    # expose `model`; there's no attribute name that works across all three.
    provider, model_id = resolve_chat_model(provider, model)
    max_tokens = MAX_OUTPUT_TOKENS_WITH_INCOME if income_available else MAX_OUTPUT_TOKENS
    model = get_chat_model(provider, model_id, max_tokens=max_tokens)
    model_with_tools = model.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    subject_desc = (
        f"Вид имот: {report.subject_property_type or '—'}\n"
        f"Гео-категория: {report.subject_geo_category or '—'}\n"
        f"Град/квартал: {report.subject_city or '—'} / {report.subject_neighborhood or '—'}\n"
        f"Площ: {report.subject_area_sqm or '—'} кв.м\n"
        f"Строителство: {report.subject_construction or '—'}, {report.subject_year or '—'} г.\n"
        f"Етаж: {report.subject_floor if report.subject_floor is not None else '—'}"
        + (f"/{report.subject_total_floors}" if report.subject_total_floors else "")
    )
    # Audit finding 2026-08-25: this free-text field existed on the report
    # but was never shown to the model -- the AI could not meaningfully
    # "describe the property" beyond the bare structured fields without it.
    if report.subject_description:
        subject_desc += f"\nСвободно описание от оценителя: {report.subject_description}"
    pool_desc = (
        f"Ръчният пул сравними съдържа {pool_stats['n']} обяви, медиана {pool_stats['median']} EUR/кв.м."
        if pool_stats else
        "Ръчният пул сравними е празен -- базирай коментара си само на AI-извлечените сравними по-долу."
    )

    prompt_parts = [
        f"ОБЕКТ:\n{subject_desc}\n",
        f"AI-ИЗВЛЕЧЕНИ СРАВНИМИ ЗА ПРОДАЖБА (по семантична близост, точно {len(comps)} на брой):\n"
        f"{_format_comparables(comps)}\n",
        f"{pool_desc}\n",
    ]
    if income_available:
        rent_pool_desc = (
            f"Ръчният наемен пул съдържа {rent_pool_stats['n']} обяви, медиана {rent_pool_stats['median']} EUR/кв.м/мес."
            if rent_pool_stats else
            "Ръчният наемен пул е празен -- базирай наема само на AI-извлечените наемни сравними по-долу."
        )
        prompt_parts.append(
            f"AI-ИЗВЛЕЧЕНИ НАЕМНИ СРАВНИМИ (точно {len(rent_comps)} на брой):\n"
            f"{_format_comparables(rent_comps)}\n{rent_pool_desc}\n"
        )
    prompt_parts.append(
        "Ако имаш нужда от по-широк пазарен контекст, можеш да извикаш get_market_trend_stats. "
        "Когато си готов, напиши наратива стриктно по зададения формат."
    )

    system_prompt = _SYSTEM_PROMPT_BASE + _SALES_PROMPT_SECTIONS + (_INCOME_PROMPT_SECTIONS if income_available else "")
    messages: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="\n".join(prompt_parts)),
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    final_response: AIMessage | None = None
    income_valuation_result: dict | None = None

    # Per-call breakdown (Tier 1, 2026-08-26): total_input_tokens/
    # total_output_tokens above are the SUM ai_valuation_runs logs -- this
    # list captures each individual LLM call separately (which tool-loop
    # turn, how many tokens, what it cost) and gets persisted to
    # agent_llm_calls once the run row exists (both success and failure
    # paths below), so "which step ate the budget" is answerable after the
    # fact, not just "how much did the whole thing cost".
    call_log: list[dict] = []

    try:
        for iteration_idx in range(MAX_TOOL_ITERATIONS):
            # Streamed rather than invoke()'d (2026-08-25, cost-visibility
            # audit): a reasoning-heavy "pro"-tier call can take minutes
            # with invoke(), during which tokens_so_far sat frozen and the
            # SSE panel looked stuck -- no way to tell "still working" from
            # "hung". Streaming ticks progress on every chunk instead, for
            # every turn of the tool loop, not just the final answer.
            chunk_accum = None
            last_chunk_usage = {}
            for chunk in model_with_tools.stream(messages):
                chunk_accum = chunk if chunk_accum is None else chunk_accum + chunk
                if chunk.usage_metadata:
                    last_chunk_usage = chunk.usage_metadata
                live_text = _extract_text(chunk_accum.content)
                # Live estimate until the authoritative usage_metadata lands
                # (~4 chars/token, replaced below once usage lands).
                progress.tokens_so_far = total_input_tokens + total_output_tokens + max(len(live_text) // 4, 0)
                emit()
            response: AIMessage = chunk_accum
            call_in = last_chunk_usage.get("input_tokens", 0)
            call_out = last_chunk_usage.get("output_tokens", 0)
            total_input_tokens += call_in
            total_output_tokens += call_out
            call_log.append({
                "call_label": f"tool_loop_{iteration_idx + 1}",
                "input_tokens": call_in,
                "output_tokens": call_out,
            })
            progress.tokens_so_far = total_input_tokens + total_output_tokens
            emit()

            messages.append(response)
            if not response.tool_calls:
                final_response = response
                break

            for call in response.tool_calls:
                tool = tools_by_name.get(call["name"])
                result = tool.invoke(call["args"]) if tool else {"error": f"unknown tool {call['name']}"}
                if call["name"] == "compute_income_valuation":
                    income_valuation_result = result  # last call wins if invoked more than once
                messages.append(ToolMessage(content=json.dumps(result, default=str, ensure_ascii=False), tool_call_id=call["id"]))
        # else: loop exhausted without a final (non-tool-call) answer -- fall
        # through with final_response still None; handled below.

        final_text = _extract_text(final_response.content) if final_response else ""

        # If the model never produced the expected section headers (loop
        # exhausted, or it answered oddly), do one bounded, streamed follow-up
        # with tools unbound so it can't keep calling tools -- this is also the
        # real, token-by-token-visible generation step in the common case where
        # the model answers directly on its first turn (no tool calls needed).
        if "## ОБОСНОВКА" not in final_text.upper():
            progress.step = "Генериране на текст…"
            emit()
            stream_messages = messages if not (messages and isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls) else messages[:-1]
            final_text = ""
            last_chunk_usage = {}
            for chunk in model.stream(stream_messages):
                final_text += _extract_text(chunk.content)
                if chunk.usage_metadata:
                    last_chunk_usage = chunk.usage_metadata
                # Live estimate: ~4 chars/token is a rough but immediate signal;
                # replaced by the authoritative usage_metadata total once the
                # stream's final chunk (which carries it, for OpenAI) arrives.
                progress.tokens_so_far = total_input_tokens + total_output_tokens + max(len(final_text) // 4, 1)
                emit()
            if last_chunk_usage:
                call_in = last_chunk_usage.get("input_tokens", 0)
                call_out = last_chunk_usage.get("output_tokens", 0)
                total_input_tokens += call_in
                total_output_tokens += call_out
                call_log.append({
                    "call_label": "fallback_stream",
                    "input_tokens": call_in,
                    "output_tokens": call_out,
                })
    except Exception as exc:
        # Cost-visibility audit (2026-08-25): a failure partway through the
        # tool-calling loop (rate limit, network error, etc.) used to lose
        # track of whatever tokens the EARLIER, successful turns in this
        # same loop already burned -- nothing was ever written to
        # ai_valuation_runs unless the whole generation completed. Real
        # money, invisible in the cost audit trail. Now logged regardless of
        # outcome, with whatever totals had accumulated before the failure.
        cost = estimate_cost_usd(model_id, total_input_tokens, total_output_tokens, provider=provider)
        run = AiValuationRun(
            report_id=report.id,
            provider=provider,
            model=model_id,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            estimated_cost_usd=cost,
            # Mirrors the success path's result dict shape for these 4 keys
            # (input_tokens/output_tokens/estimated_cost_usd/model) so
            # templates reading ai_valuation_runs.output (not the separate
            # DB columns) show the real spend for a failed run too, not a
            # blank/undefined value.
            output={
                "failed": True,
                "error": str(exc)[:2000],
                "model": model_id,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "estimated_cost_usd": cost,
            },
        )
        db.add(run)
        db.commit()
        _persist_call_log(db, run.id, provider, model_id, call_log)
        progress.status = "error"
        progress.error = (
            f"AI генерацията прекъсна с грешка след използвани "
            f"{total_input_tokens + total_output_tokens} токена"
            + (f" (~${cost:.4f})" if cost is not None else "")
            + f": {exc}"
        )
        emit()
        return progress

    sections = _parse_sections(final_text)

    income_result = None
    if include_income:
        if not income_available:
            income_result = {"available": False, "reason": income_unavailable_reason}
        elif income_valuation_result is None:
            # Rent data was sufficient, but the model never actually called
            # the tool -- degrade gracefully rather than show fabricated
            # numbers or a silent gap.
            income_result = {"available": False, "reason": "AI-ят не изчисли доходна стойност чрез инструмента."}
        else:
            income_result = {
                "available": True,
                "rationale": sections["income_rationale"],
                "commentary": sections["income_commentary"],
                "reasoning": sections["income_reasoning"],
                "caveats": sections["income_caveats"],
                "valuation": income_valuation_result,
                "comparables": rent_comps,
            }

    result = _json_safe({
        "report_id": str(report.id),
        "property_description": sections["property_description"],
        "comparable_selection_rationale": sections["comparable_selection_rationale"],
        "comparable_commentary": sections["comparable_commentary"],
        "value_reasoning": sections["value_reasoning"],
        "caveats": sections["caveats"],
        "suggested_value_range_low": value_low,
        "suggested_value_range_high": value_high,
        "comparables": comps,
        "income": income_result,
    })

    cost = estimate_cost_usd(model_id, total_input_tokens, total_output_tokens, provider=provider)
    result["input_tokens"] = total_input_tokens
    result["output_tokens"] = total_output_tokens
    result["estimated_cost_usd"] = cost
    result["model"] = model_id

    run = AiValuationRun(
        report_id=report.id,
        provider=provider,
        model=model_id,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        estimated_cost_usd=cost,
        output=result,
    )
    db.add(run)
    db.commit()
    _persist_call_log(db, run.id, provider, model_id, call_log)

    progress.status = "done"
    progress.step = "Готово"
    progress.result = result
    progress.tokens_so_far = total_input_tokens + total_output_tokens
    emit()
    return progress
