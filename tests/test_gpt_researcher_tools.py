import pytest

import research_tools.gpt_researcher as adapter


class FakeResearcher:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)

    async def quick_search(self, query):
        return [
            {"title": "Primary", "body": "Finding", "href": "https://example.test/1"},
            {"title": "Second", "content": "More", "url": "https://example.test/2"},
        ]

    async def conduct_research(self):
        return None

    async def write_report(self):
        return "# Report\n\nCited result."

    def get_source_urls(self):
        return ["https://example.test/1"]

    def get_costs(self):
        return 0.25


@pytest.mark.asyncio
async def test_both_web_tools_force_report_source_web(monkeypatch):
    FakeResearcher.instances.clear()
    monkeypatch.setattr(adapter, "GPTResearcher", FakeResearcher)
    quick_search, deep_research = adapter.build_web_research_tools(max_search_results=1)

    quick_result = await quick_search("topic")
    report = await deep_research("section", "overall")

    assert len(quick_result.split("URL:")) == 2
    assert "### Sources" in report
    assert all(item.kwargs["report_source"] == "web" for item in FakeResearcher.instances)
    assert FakeResearcher.instances[-1].kwargs["report_type"] == "subtopic_report"
    assert deep_research.cost_tracker.total == pytest.approx(0.5)

