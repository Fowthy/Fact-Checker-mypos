import json
from datetime import datetime, timezone

from excerpt_align import (
    align_issue_excerpts,
    build_fuzzy_index,
    find_excerpt_ranges,
    fuzzy_locate,
    locate_excerpt,
)
from fact_check_request import build_fact_check_body

TEXT = "\n".join([
    "Качването на тегло при хормонален дисбаланс често не е свързано с реално натрупване на мазнини, а със задържане на течности.",
    "В други случаи причината е хранителна.",
    "Той повишава телесната температура, което означава, че организмът изразходва повече енергия дори в покой.",
    "Potrivit Eurostat (https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20260331-2), costul orar a fost de 13,6 euro.",
    "[Image: Harta salariilor pe județe]",
    "Продуктът е без глутен и без ГМО и е подходящ за вегани.",
])


class FakeResponse:
    def __init__(self, body, status_code=200):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


def fake_post(body, status_code=200):
    calls = []

    def post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return FakeResponse(body, status_code)

    return calls, post


def repair_reply(fixes):
    return {
        "choices": [{"message": {"content": json.dumps({"fixes": fixes})}}],
        "usage": {
            "prompt_tokens": 900,
            "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 100},
            "completion_tokens_details": {"reasoning_tokens": 10},
            "cost": 0.004,
        },
    }


def align(issues, post):
    return align_issue_excerpts(TEXT, issues, "openai/gpt-6.1-terra", "k", post)


def test_finds_an_exact_excerpt_as_one_range():
    text = "Before. Exact quoted sentence. After."
    excerpt = "Exact quoted sentence."
    start = text.index(excerpt)

    assert find_excerpt_ranges(text, excerpt) == [(start, start + len(excerpt))]


def test_matches_harmless_case_punctuation_and_whitespace_differences():
    text = "Алое вера – екстракт\nот гел 200:1"

    assert find_excerpt_ranges(text, "алое вера - екстракт от гел 200 : 1") == [(0, len(text))]


def test_maps_non_contiguous_lines_to_separate_source_ranges():
    first = "Капсулата съдържа говежди желатин."
    second = "Капсулна обвивка: хидроксипропил метилцелулоза."
    text = f"{first}\nSeveral unrelated paragraphs.\n{second}"
    second_start = text.index(second)

    assert find_excerpt_ranges(text, f"{first}\n{second}") == [
        (0, len(first)),
        (second_start, second_start + len(second)),
    ]


def test_maps_excerpts_joined_by_an_ai_ellipsis():
    first = "700 мг псилиум в дневната доза;"
    second = "Псилиумът се използва при запек."
    text = f"{first}\nUnquoted content between the claims.\n{second}"

    assert find_excerpt_ranges(text, f"{first}\n… {second}") == [
        (0, len(first)),
        (text.index(second), len(text)),
    ]


def test_does_not_create_a_highlight_from_one_weak_fragment():
    text = "Only this part exists. Some unrelated text."

    assert find_excerpt_ranges(text, "Only this part exists.\nThe model invented this part.") == []


def test_ignores_a_pasted_link_whether_or_not_the_excerpt_repeats_it():
    text = "Potrivit Eurostat (https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20260331-2), costul orar a fost de 13,6 euro."
    without_url = "Potrivit Eurostat, costul orar a fost de 13,6 euro."
    shortened_url = "Potrivit Eurostat (https://ec.europa.eu/…/ddn-20260331-2), costul orar a fost de 13,6 euro."

    for excerpt in (without_url, shortened_url):
        assert find_excerpt_ranges(text, excerpt) == [(0, len(text) - 1)]


def test_treats_romanian_cedilla_and_comma_below_letters_as_the_same_letter():
    text = "Salariul mediu a crescut, iar ţara are cea mai mare inflaţie."
    excerpt = "iar țara are cea mai mare inflație"

    assert find_excerpt_ranges(text, excerpt) == [(text.index("iar ţara"), len(text) - 1)]


def test_locate_excerpt_reports_how_each_excerpt_was_found():
    assert locate_excerpt(TEXT, "В други случаи причината е хранителна.")[0] == "exact"
    assert locate_excerpt(TEXT, "в други случаи причината е хранителна")[0] == "normalized"
    assert locate_excerpt(TEXT, "Potrivit Eurostat, costul orar a fost de 13,6 euro.")[0] == "normalized"
    assert locate_excerpt(TEXT, "Той [прогестеронът] повишава телесната температура")[0] == "brackets"
    assert locate_excerpt(TEXT, "[Image: Harta salariilor pe județe]")[0] == "exact"
    assert locate_excerpt(TEXT, "") is None


def test_fuzzy_locate_finds_a_quote_whose_tail_drifted_from_the_source():
    excerpt = "Качването на тегло при хормонален дисбаланс често не е свързано с реално натрупване на мазнини, а със задържане на течности. Ако килограмите варират с 1-2 кг, това е нормално."
    found = fuzzy_locate(build_fuzzy_index(TEXT), excerpt)

    assert found and found.coverage >= 0.5 and found.words >= 6
    assert TEXT[found.start:found.end] == "Качването на тегло при хормонален дисбаланс често не е свързано с реално натрупване на мазнини, а със задържане на течности"


def test_fuzzy_locate_gives_a_generic_phrase_too_few_words_to_be_accepted():
    found = fuzzy_locate(build_fuzzy_index(TEXT), "Без глутен и без ГМО - Vegan Friendly")

    assert found is None or found.words < 6


def test_fuzzy_locate_rejects_an_excerpt_that_is_not_in_the_text():
    excerpt = "Започнете деня със сок от грейпфрут, портокал и лимон всяка сутрин."

    assert fuzzy_locate(build_fuzzy_index(TEXT), excerpt) is None


def test_align_stores_the_exact_source_passage_and_makes_no_call_when_all_excerpts_are_found():
    calls, post = fake_post(repair_reply([]))
    issues = [
        {"excerpt": "в други случаи причината е хранителна", "issue": "a", "type": "questionable"},
        {"excerpt": "Potrivit Eurostat, costul orar a fost de 13,6 euro.", "issue": "b", "type": "misleading"},
        {"excerpt": "Качването на тегло при хормонален дисбаланс често не е свързано с реално натрупване на мазнини, а със задържане на течности. Ако килограмите варират.", "issue": "c", "type": "misleading"},
    ]
    out = align(issues, post)

    assert out["paths"] == ["normalized", "normalized", "fuzzy"]
    for issue in out["issues"]:
        assert issue["excerpt"] in TEXT
    assert calls == []
    assert out["repair_usage"] is None


def test_align_asks_the_model_to_requote_empty_and_unknown_excerpts_and_accepts_only_real_quotes():
    calls, post = fake_post(repair_reply([
        {"index": 0, "excerpt": "В други случаи причината е хранителна."},
        {"index": 1, "excerpt": "A sentence the model made up."},
    ]))
    issues = [
        {"excerpt": "", "issue": "missing detail", "type": "incomplete"},
        {"excerpt": "https://www.physio-pedia.com/Walking", "issue": "wrong source", "type": "questionable"},
    ]
    out = align(issues, post)

    assert out["paths"] == ["repaired", "none"]
    assert out["issues"][0]["excerpt"] == "В други случаи причината е хранителна."
    assert len(calls) == 1
    assert TEXT in calls[0]["json"]["messages"][1]["content"]
    assert out["repair_usage"] == {
        "input_tokens": 900,
        "output_tokens": 40,
        "input_tokens_details": {"cached_tokens": 100},
        "output_tokens_details": {"reasoning_tokens": 10},
        "cost": 0.004,
    }


def test_align_keeps_the_issues_when_the_repair_call_fails():
    _, post = fake_post({"error": "down"}, 502)
    out = align([{"excerpt": "", "issue": "missing detail", "type": "incomplete"}], post)

    assert out["paths"] == ["none"]
    assert out["repair_error"] == "Excerpt repair: OpenRouter HTTP 502"
    assert out["issues"][0]["issue"] == "missing detail"


def test_request_body_asks_for_web_search_without_a_limit_and_for_exact_excerpts():
    body = build_fact_check_body("openai/gpt-6.1-terra", "Some text.", datetime(2026, 11, 17, 9, 30, tzinfo=timezone.utc))
    system, user = (message["content"] for message in body["input"])

    assert body["tools"] == [
        {"type": "openrouter:web_search", "parameters": {"engine": "exa", "max_results": 10, "max_total_results": 1000}}
    ]
    assert body["stop_server_tools_when"] == [{"type": "max_cost", "max_cost_in_dollars": 5}]
    assert body["reasoning"] == {"effort": "high"}
    assert "Today's date is 2026-11-17." in system
    assert "3. Every issue MUST have a non-empty excerpt." in user
    assert user.endswith("Text to fact-check:\nSome text.\n")
