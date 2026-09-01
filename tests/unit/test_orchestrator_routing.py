"""
Unit tests for the Supervisor graph's pure routing logic
(app/services/llm/orchestrator_graph.py, Phase 11/13) -- no LLM calls, no DB,
no LangGraph runtime involved. These three functions decide the whole shape
of a turn's cycle (which node runs next, when the hard hop cap kicks in), so
they're worth locking down independently of any live model behavior.
"""
import pytest

from app.services.llm.orchestrator_graph import (
    MAX_SPECIALIST_HOPS,
    RouteDecision,
    _SUPERVISOR_ROUTES,
    _route_after_specialist,
    _route_from_supervisor,
)


def test_route_from_supervisor_uses_next_specialist():
    assert _route_from_supervisor({"next_specialist": "income"}) == "income"
    assert _route_from_supervisor({"next_specialist": "legal"}) == "legal"


def test_route_from_supervisor_defaults_to_answer_when_missing():
    assert _route_from_supervisor({}) == "answer"
    assert _route_from_supervisor({"next_specialist": None}) == "answer"


def test_supervisor_routes_cover_every_route_decision_literal():
    # RouteDecision.next is the only thing the supervisor node can produce
    # (see _supervisor_node_fn) -- if a new literal is ever added there
    # without a matching _SUPERVISOR_ROUTES entry, add_conditional_edges
    # would raise at graph-build time. Catch that mismatch here instead,
    # without needing to build a real graph.
    literal_values = RouteDecision.model_fields["next"].annotation.__args__
    assert set(literal_values) == set(_SUPERVISOR_ROUTES.keys())


def test_supervisor_routes_target_nodes():
    assert _SUPERVISOR_ROUTES["income"] == "income"
    assert _SUPERVISOR_ROUTES["market"] == "market"
    assert _SUPERVISOR_ROUTES["market_analysis"] == "market_analysis"
    assert _SUPERVISOR_ROUTES["legal"] == "legal"
    assert _SUPERVISOR_ROUTES["auditor"] == "auditor"
    assert _SUPERVISOR_ROUTES["answer"] == "direct_answer"
    assert _SUPERVISOR_ROUTES["done"] == "synthesize"


@pytest.mark.parametrize("hops", range(MAX_SPECIALIST_HOPS))
def test_route_after_specialist_returns_to_supervisor_below_cap(hops):
    assert _route_after_specialist({"hops": hops}) == "supervisor"


def test_route_after_specialist_forces_synthesize_at_cap():
    assert _route_after_specialist({"hops": MAX_SPECIALIST_HOPS}) == "synthesize"


def test_route_after_specialist_forces_synthesize_beyond_cap():
    assert _route_after_specialist({"hops": MAX_SPECIALIST_HOPS + 5}) == "synthesize"


def test_route_after_specialist_defaults_hops_to_zero():
    # A state dict missing "hops" entirely (shouldn't happen once the graph
    # is running, since every node returns it, but the .get(..., 0) default
    # is the actual safety net) still routes back to the supervisor.
    assert _route_after_specialist({}) == "supervisor"
