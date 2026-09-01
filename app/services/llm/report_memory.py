"""Persistent, cross-conversation report memory (Phase 13, 2026-09-01).

Before this module, a specialist's finding lived exactly as long as one
chat turn -- OrchestratorState.findings was written by every specialist
node but never read back by anything (confirmed by grep before writing
this). Two conversations on the same report had zero awareness of each
other's history, and there was no way for a non-chat action (the Report
Compiler) to know what had already been established.

report_agent_findings is append-only (mirrors ListingSnapshot's own
pattern) -- every specialist turn adds a new row rather than overwriting
one blob, so the history of how a finding evolved stays inspectable.
get_report_memory() only surfaces the LATEST row per domain -- callers
that want the full history query the table directly.
"""
from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import ReportAgentFinding


def persist_finding(db: Session, report_id, domain: str, source: str, source_id, summary: str) -> None:
    """Best-effort -- never raises into the caller's turn. A missing/failed
    memory write should degrade to "no memory of this turn" for future
    reads, not fail the chat response the appraiser is waiting on.

    report_id=None is a legitimate no-op, not an error -- _build_tool_loop
    (orchestrator_graph.py) is also reused by the report-agnostic Market
    Analyst (analyst_chain.py), which has no report to attach memory to;
    report_agent_findings.report_id is NOT NULL, so there is nowhere valid
    to write for that caller."""
    if not summary or report_id is None:
        return
    try:
        db.add(ReportAgentFinding(
            report_id=report_id, domain=domain, source=source, source_id=source_id, summary=summary,
        ))
        db.commit()
    except Exception:
        db.rollback()


def get_report_memory(db: Session, report_id) -> dict[str, str]:
    """Latest finding per domain for this report, across ALL conversations
    and compile runs -- what the Supervisor's routing prompt and the Report
    Compiler both see as "already known about this report"."""
    # Correlated subquery for "latest created_at per domain" rather than a
    # window function -- keeps this readable/portable, and report-scoped
    # finding counts are small (dozens, not thousands) so the extra query
    # cost is negligible.
    latest_per_domain = (
        db.query(ReportAgentFinding.domain, func.max(ReportAgentFinding.created_at).label("max_created_at"))
        .filter(ReportAgentFinding.report_id == report_id)
        .group_by(ReportAgentFinding.domain)
        .subquery()
    )
    rows = (
        db.query(ReportAgentFinding)
        .join(
            latest_per_domain,
            (ReportAgentFinding.domain == latest_per_domain.c.domain)
            & (ReportAgentFinding.created_at == latest_per_domain.c.max_created_at),
        )
        .filter(ReportAgentFinding.report_id == report_id)
        .all()
    )
    return {row.domain: row.summary for row in rows}


def format_report_memory(memory: dict[str, str]) -> str:
    """Renders get_report_memory()'s dict as a system-prompt-ready block --
    shared by the supervisor node so routing decisions and the eventual
    synthesis node see the identical text."""
    if not memory:
        return ""
    domain_labels = {
        "income": "Доходен подход", "market": "Пазарен подход",
        "market_analysis": "Пазарен анализ", "legal": "Правно/етично", "auditor": "Критичен преглед",
    }
    lines = "\n".join(f"- {domain_labels.get(d, d)}: {text}" for d, text in memory.items())
    return (
        "\n\nВЕЧЕ ИЗВЕСТНО ЗА ТОЗИ ДОКЛАД (от предишни разговори/действия, не от текущата "
        f"реплика):\n{lines}\n"
        "Ползвай това, за да не пращаш специалист да преоткрива вече установено -- но ако "
        "оценителят изрично поиска ново/обновено изчисление, пак насочи към съответния "
        "специалист."
    )
