import inspect

from gpt_researcher import GPTResearcher


def test_gpt_researcher_public_contract_used_by_adapter_exists():
    constructor = inspect.signature(GPTResearcher)
    for parameter in ("query", "report_type", "report_source", "parent_query", "verbose"):
        assert parameter in constructor.parameters

    for method in (
        "quick_search",
        "conduct_research",
        "write_report",
        "get_source_urls",
        "get_costs",
    ):
        assert callable(getattr(GPTResearcher, method, None))

