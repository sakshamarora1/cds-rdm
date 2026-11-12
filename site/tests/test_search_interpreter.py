# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 CERN.
#
# CDS-RDM is free software; you can redistribute it and/or modify it under
# the terms of the GPL-2.0 License; see LICENSE file for more details.

"""Tests for the search interpreter (no LLM needed: its answers are given)."""

from unittest import mock

import pytest
import requests

from cds_rdm.search.interpreter import (
    InterpreterError,
    abbreviations,
    build_messages,
    call_ollama,
    community_matches,
    match_report_prefixes,
    match_vocabulary,
    render,
    report_number_formats,
    report_number_prefixes,
    should_interpret,
)

RESOURCE_TYPES = {
    "publication-dissertation": "Thesis",
    "poster": "Poster",
    "administrative-operationalcircular": "Operational circular",
    "administrative-circular": "Administrative circular",
}
REPORT_FORMATS = {
    "administrative-operationalcircular": ["CERN-OPER-CIRC-##"],
    "administrative-circular": ["CERN-ADMIN-CIRC-##"],
    "publication-dissertation": ["CERN-THESIS-####-###"],
}
COMMUNITIES = {
    "staff association": [("staff-association", "Staff Association")],
    "cern": [("cern-research", "CERN Research")],
    "human resources hr": [("hr", "Human Resources (HR)")],
    "cern courier": [("cern-courier", "CERN Courier")],
}


def find_community(name):
    return COMMUNITIES.get(name, [])


def answer(**fields):
    """An LLM answer with every field empty except ``fields``."""
    return {
        "resource_type": None,
        "creators": [],
        "community": None,
        "records_of": None,
        "identifiers": [],
        "also_written_as": [],
        "year_from": None,
        "year_to": None,
        **fields,
    }


FACETS = {
    "aleph": [
        ("Experiment", 'experiment.id:"ALEPH"', "ALEPH"),
        ("Report numbers", 'identifier:"CERN-ALEPH"', "CERN-ALEPH-…"),
    ],
    "atlas": [("Experiment", 'experiment.id:"ATLAS"', "ATLAS")],
}


def find_facets(name):
    return FACETS.get(name.lower(), [])


def interpret(query, creators_in_index=None, **fields):
    has_creator = (
        None if creators_in_index is None else lambda name: name in creators_in_index
    )
    return render(
        query,
        answer(**fields),
        RESOURCE_TYPES,
        find_community,
        has_creator,
        report_formats=REPORT_FORMATS,
        find_facets=find_facets,
    )


@pytest.mark.parametrize(
    "query,expected",
    [
        ("thesis Mobius", True),
        ("oc6", True),
        ("higgs", False),  # single plain word: searched as typed
        ("title:higgs boson", False),
        ("higgs AND boson", False),
        ('"higgs boson" decay', False),
        ("higg* boson", False),
        ("", False),
        ("word " * 60, False),
    ],
)
def test_should_interpret(query, expected):
    assert should_interpret(query) is expected


def test_thesis_by_creator():
    result = interpret(
        "thesis Mobius",
        resource_type="publication-dissertation",
        creators=["Mobius"],
    )
    # Creator required; the resource type and its words only boost.
    assert result["query"] == (
        '+creator:(+mobius) resource_type.id:"publication-dissertation"^5 thesis'
    )
    assert result["parts"] == [
        {"label": "Resource type", "value": "Thesis"},
        {"label": "Creator", "value": "Mobius"},
    ]


def test_creator_full_name_and_connecting_words():
    result = interpret(
        "theses by Silke Mobius",
        resource_type="publication-dissertation",
        creators=["Silke Mobius"],
    )
    assert result["query"] == (
        '+creator:(+silke +mobius) resource_type.id:"publication-dissertation"^5 theses'
    )


def test_values_not_in_the_search_are_dropped():
    # Small models invent authors, types, communities and years.
    assert (
        interpret(
            "pixel detector upgrade",
            resource_type="poster",
            creators=["ATLAS Collaboration"],
            community="CERN",
            year_from=2020,
        )
        is None
    )


def test_unknown_resource_type_is_dropped():
    assert interpret("thesis Mobius", resource_type="made-up") is None


def test_community_needs_a_full_title_match():
    result = interpret("staff association community", community="Staff Association")
    assert result["query"] == '+community.id:"staff-association"'
    # "CERN" alone must not pick "CERN Research".
    assert interpret("CERN docs", community="CERN") is None


def test_years_must_be_typed_and_not_inside_identifiers():
    result = interpret(
        "posters about FCC from 2020",
        resource_type="poster",
        year_from=2020,
        year_to=2020,
    )
    assert result["query"] == (
        '+FCC +publication_date:[2020 TO 2020] resource_type.id:"poster"^5 posters'
    )
    result = interpret(
        "CERN-THESIS-2021-078 report",
        resource_type="publication-dissertation",
        identifiers=["CERN-THESIS-2021-078"],
        year_from=2021,
    )
    # No year filter, no resource type from the identifier's "THESIS" part, and
    # with an identifier the other words only rank.
    assert result["query"] == '+identifier:"CERN-THESIS-2021-078" report'


def test_short_forms_are_expanded():
    result = interpret(
        "oc6",
        identifiers=["OC6"],
        also_written_as=["Operational Circular No. 6", "CERN-OPER-CIRC-06"],
    )
    assert result["query"] == (
        '+(identifier:"OC6" OR "Operational Circular No. 6" OR "CERN-OPER-CIRC-06")'
    )


def test_identifier_not_in_the_search_becomes_an_alternative():
    result = interpret(
        "operational circular 6",
        resource_type="administrative-operationalcircular",
        identifiers=["OC6"],
    )
    assert result["query"] == (
        '+(6 OR "OC6") '
        'resource_type.id:"administrative-operationalcircular"^5 '
        "(operational circular)"
    )


def test_nothing_understood():
    assert interpret("higgs boson decay") is None
    # A resource type alone leaves nothing that must match.
    assert interpret("posters", resource_type="poster") is None


def test_model_output_cannot_inject_query_syntax():
    # Creators are rebuilt from plain word tokens.
    result = interpret("Mobius x", creators=['Mobius" OR internal_notes.note:x'])
    assert result is None or "internal_notes" not in result["query"]
    # Quoted values are escaped.
    result = interpret("oc6", also_written_as=['OC "6" OR x:y'])
    assert result["query"] == '+(oc6 OR "OC \\"6\\" OR x:y")'


def test_build_messages_lists_repository_facts():
    types = [
        ("administrative-operationalcircular", "Operational circular"),
        ("poster", "Poster"),
    ]
    system, user = build_messages(
        "thesis Mobius",
        types,
        context="Local note.",
        report_formats={"administrative-operationalcircular": ["CERN-OPER-CIRC-##"]},
        communities=["Staff Association"],
    )
    content = system["content"]
    assert "- poster: Poster" in content
    assert "- OC: Operational circular (report numbers CERN-OPER-CIRC-##)" in content
    assert "- Operational circular: CERN-OPER-CIRC-##" in content
    assert "- Staff Association" in content
    assert "Local note." in content
    # The worked example is built from the repository's own series.
    assert '"OC 3" or "OC3" gives' in content
    assert '["Operational circular No. 3", "CERN-OPER-CIRC-03"]' in content
    assert user == {"role": "user", "content": "thesis Mobius"}


def test_build_messages_without_facts_has_no_example():
    system, _ = build_messages("x", [("poster", "Poster")])
    assert "For example" not in system["content"]
    assert "Report number formats" not in system["content"]


def test_report_number_formats():
    formats = report_number_formats(
        [
            *[("oc", f"CERN-OPER-CIRC-{n:02d}-REV{n % 3}") for n in range(1, 12)],
            ("oc", "CERN-OPER-CIRC-NO.6-REV1"),  # under 10% of the type: left out
            ("other", "CERN-CCP-2001-003-Draft"),
            ("other", "CERN-CCP-2002-004"),
            ("article", "26/2001"),  # no letters: not a report number format
        ]
    )
    # Digits become #, revision and draft suffixes are dropped, rare ones too.
    assert formats == {"oc": ["CERN-OPER-CIRC-##"], "other": ["CERN-CCP-####-###"]}


def test_abbreviations_of_multi_word_names():
    assert abbreviations(
        [
            ("a", "Operational circular"),
            ("b", "Conference paper"),
            ("c", "Conference proceeding"),
            ("d", "Thesis"),
        ]
    ) == {
        "OC": ["Operational circular"],
        "CP": ["Conference paper", "Conference proceeding"],
    }


def test_series_alternatives_must_match_the_series_typed():
    # The model gave an operational circular for "AC2": dropped.
    result = interpret(
        "AC2",
        identifiers=["AC2"],
        also_written_as=["Operational circular No. 2", "CERN-OPER-CIRC-02"],
    )
    assert result["query"] == '+identifier:"AC2"'
    result = interpret(
        "AC2",
        identifiers=["AC2"],
        also_written_as=["Administrative circular No. 2", "CERN-ADMIN-CIRC-02"],
    )
    assert result["query"] == (
        '+(identifier:"AC2" OR "Administrative circular No. 2" OR "CERN-ADMIN-CIRC-02")'
    )


def test_unfilled_report_number_formats_are_dropped():
    result = interpret(
        "OC 7", also_written_as=["Operational circular No. 7", "CERN-OPER-CIRC-##"]
    )
    assert result["query"] == '+((OC 7) OR "Operational circular No. 7")'


def test_alternatives_only_rank_when_nothing_else_must_match():
    result = interpret(
        "thesis Mobius",
        resource_type="publication-dissertation",
        creators=["Mobius"],
        also_written_as=["Mobius dissertation", "Mobius thesis"],
    )
    # "Mobius thesis" only reuses typed words: dropped. The other one ranks.
    assert result["query"] == (
        '+creator:(+mobius) resource_type.id:"publication-dissertation"^5 '
        'thesis "Mobius dissertation"'
    )


def test_creators_must_exist():
    result = interpret(
        "thesis on calorimetry",
        creators_in_index={"Mobius"},
        resource_type="publication-dissertation",
        creators=["calorimetry"],
    )
    assert result["query"] == (
        '+calorimetry resource_type.id:"publication-dissertation"^5 thesis'
    )


def test_community_by_acronym_or_full_title():
    result = interpret("HR community notes", community="Human Resources (HR)")
    assert result["query"] == '+notes +community.id:"hr"'
    result = interpret(
        "CERN Courier 2001", community="CERN Courier", year_from=2001, year_to=2001
    )
    assert result["query"] == (
        '+community.id:"cern-courier" +publication_date:[2001 TO 2001]'
    )


def test_open_ended_years_and_years_are_not_identifiers():
    result = interpret(
        "preprints since 2015 2001",
        year_from=2015,
        identifiers=["2001"],
    )
    assert "+publication_date:[2015 TO *]" in result["query"]
    assert "identifier:" not in result["query"]
    assert {"label": "Years", "value": "from 2015"} in result["parts"]


def test_call_ollama_errors_are_interpreter_errors():
    with mock.patch(
        "cds_rdm.search.interpreter.requests.post",
        side_effect=requests.ConnectionError("down"),
    ):
        with pytest.raises(InterpreterError):
            call_ollama([], "http://localhost:11434", "m", 1)

    response = mock.Mock()
    response.json.return_value = {"message": {"content": "not json"}}
    with mock.patch("cds_rdm.search.interpreter.requests.post", return_value=response):
        with pytest.raises(InterpreterError):
            call_ollama([], "http://localhost:11434", "m", 1)


def test_call_ollama_sends_the_schema():
    response = mock.Mock()
    response.json.return_value = {"message": {"content": '{"creators": []}'}}
    with mock.patch(
        "cds_rdm.search.interpreter.requests.post", return_value=response
    ) as post:
        assert call_ollama([], "http://host/", "qwen3:8b", 5) == {"creators": []}
    url, kwargs = post.call_args.args[0], post.call_args.kwargs
    assert url == "http://host/api/chat"
    assert kwargs["json"]["model"] == "qwen3:8b"
    assert kwargs["json"]["format"]["required"]
    assert kwargs["timeout"] == 5


def test_records_of_a_group_match_any_of_its_facets():
    result = interpret("ALEPH records", records_of="ALEPH")
    assert result["query"] == '+(experiment.id:"ALEPH" OR identifier:"CERN-ALEPH")'
    assert result["parts"] == [
        {"label": "Experiment", "value": "ALEPH"},
        {"label": "Report numbers", "value": "CERN-ALEPH-…"},
    ]


def test_records_of_uses_the_typed_part_of_the_name():
    result = interpret("ATLAS publications", records_of="ATLAS Collaboration")
    assert result["query"] == '+experiment.id:"ATLAS"'


def test_records_of_unknown_name_is_ignored():
    assert interpret("Foo records", records_of="Foo") is None


def test_group_given_as_creator_is_read_as_records_of():
    result = interpret(
        "ATLAS theses",
        resource_type="publication-dissertation",
        creators=["ATLAS"],
    )
    assert result["query"] == (
        '+experiment.id:"ATLAS" resource_type.id:"publication-dissertation"^5 theses'
    )
    # Unless the search says "by".
    result = interpret("theses by ATLAS", creators=["ATLAS"])
    assert "creator:(+atlas)" in result["query"]


def test_match_vocabulary():
    accelerators = [("CERN LHC", "CERN LHC", []), ("CERN HL-LHC", "CERN HL-LHC", [])]
    accelerators += [(f"CERN X{i}", f"CERN X{i}", []) for i in range(10)]
    assert match_vocabulary("LHC", accelerators) == [("CERN LHC", "CERN LHC")]
    assert match_vocabulary("lhc accelerator", accelerators, kind="Accelerator") == [
        ("CERN LHC", "CERN LHC")
    ]
    departments = [("EP", "EP - Experimental Physics Department", [])]
    assert match_vocabulary("EP department", departments, kind="Department") == [
        ("EP", "EP - Experimental Physics Department")
    ]
    assert match_vocabulary("experimental physics department", departments) == [
        ("EP", "EP - Experimental Physics Department")
    ]
    assert match_vocabulary("physics", departments) == []
    experiments = [("UA1", "UA1", ["UA-001"])]
    assert match_vocabulary("UA-001", experiments) == [("UA1", "UA1")]


def test_report_number_prefixes():
    numbers = [
        ("a", "CERN-ALEPH-PHYSIC-99-012"),
        ("a", "CERN-ALEPH-PHYSIC-98-001"),
        ("a", "CERN-ALEPH-97-003"),
        ("a", "CERN-ALEPH-96-004"),
        ("b", "CERN-HR-Note-2020-001"),
        ("b", "CERN-HR-Note-2021-002"),
        ("c", "ONCE-ONLY-2001-001"),
    ]
    prefixes = report_number_prefixes(numbers)
    assert prefixes == ["CERN-ALEPH", "CERN-ALEPH-PHYSIC", "CERN-HR-Note"]
    assert match_report_prefixes("ALEPH", prefixes) == ["CERN-ALEPH"]
    assert match_report_prefixes("hr", prefixes) == ["CERN-HR"]
    # The common first segment alone names nothing.
    assert match_report_prefixes("CERN", prefixes) == []


def test_community_matches():
    assert community_matches("HR", "Human Resources (HR)")
    assert community_matches("staff association community", "Staff Association")
    assert not community_matches("CERN", "CERN Research")
