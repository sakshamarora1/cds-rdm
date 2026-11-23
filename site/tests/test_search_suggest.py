# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 CERN.
#
# CDS-RDM is free software; you can redistribute it and/or modify it under
# the terms of the GPL-2.0 License; see LICENSE file for more details.

"""Tests for the search suggestions (spelling fixes and known names)."""

from unittest import mock

import pytest
from flask import Flask, g

from cds_rdm.search import suggest, views
from cds_rdm.search.suggest import (
    is_own_index,
    known_names,
    match_phrase,
    spelling_body,
    spelling_from_response,
    versioned_index_name,
    vocabulary_phrases,
)


def test_spelling_only_from_public_published_records():
    body = spelling_body("higs boson")
    collate = body["suggest"]["spelling"]["phrase"]["collate"]
    assert collate["prune"] is False
    assert collate["query"]["source"]["bool"]["filter"] == [
        {"term": {"access.record": "public"}},
        {"term": {"deletion_status": "P"}},
    ]
    assert body["size"] == 0
    phrase = body["suggest"]["spelling"]["phrase"]
    generators = [g["field"] for g in phrase["direct_generator"]]
    assert generators == [
        "metadata.title",
        "metadata.description",
        "metadata.subjects.subject",
    ]
    match = collate["query"]["source"]["bool"]["must"]["multi_match"]
    assert match["fields"] == generators and match["operator"] == "and"


def test_spelling_from_response():
    response = {"suggest": {"spelling": [{"options": [{"text": "higgs boson"}]}]}}
    assert spelling_from_response("higs boson", response) == "higgs boson"
    # The same text, or no option, is no fix.
    same = {"suggest": {"spelling": [{"options": [{"text": "Higgs Boson"}]}]}}
    assert spelling_from_response("higgs boson", same) is None
    assert (
        spelling_from_response("x", {"suggest": {"spelling": [{"options": []}]}})
        is None
    )
    assert spelling_from_response("x", {}) is None


@pytest.mark.parametrize(
    "text,phrase,expected",
    [
        ("atlas", "ATLAS", (0, 1)),
        ("atlsa higgs", "ATLAS", (0, 1)),  # swapped letters
        ("higgs at atlas", "ATLAS", (2, 3)),
        ("cern lhc papers", "CERN LHC", (0, 2)),
        ("lhc", "CERN LHC", None),  # every word of the name must be typed
        ("btlas", "ATLAS", None),  # the first letter must be right
        ("cms", "CMD", (0, 1)),  # 3 letters allow 1 edit (AUTO)
        ("ab", "AC", None),  # 2 letters allow none
    ],
)
def test_match_phrase(text, phrase, expected):
    assert match_phrase(text, phrase) == expected


def _hit(phrase, label, value, query):
    return {
        "_source": {"phrase": phrase, "label": label, "value": value, "query": query}
    }


def test_known_names_keep_the_other_words():
    hits = [
        _hit("ATLAS", "Experiment", "ATLAS", 'experiment.id:"ATLAS"'),
        # Same value under another name: offered once.
        _hit("atlas", "Experiment", "ATLAS", 'experiment.id:"ATLAS"'),
        # A fuzzy hit from OpenSearch that does not stand for typed words.
        _hit("ALICE", "Experiment", "ALICE", 'experiment.id:"ALICE"'),
    ]
    assert known_names("atlsa higgs", hits) == [
        {
            "label": "Experiment",
            "value": "ATLAS",
            "query": 'experiment.id:"ATLAS" higgs',
        }
    ]


def test_vocabulary_phrases_one_per_name():
    entries = [("EP", "EP - Experimental Physics", ["PH"])]
    docs = list(vocabulary_phrases("Department", "department.id", "id", entries))
    assert sorted(d["phrase"] for d in docs) == [
        "EP",
        "EP - Experimental Physics",
        "Experimental Physics",
        "PH",
    ]
    assert {d["query"] for d in docs} == {'department.id:"EP"'}
    by_title = list(vocabulary_phrases("Beam", "beam", "title", [("b1", "H8", [])]))
    assert by_title[0]["query"] == 'beam:"H8"'


def test_vocabulary_phrases_without_shared_words():
    entries = [(f"a{i}", f"CERN Machine{i}", []) for i in range(10)]
    entries.append(("lhc", "CERN LHC", []))
    phrases = {
        d["phrase"]
        for d in vocabulary_phrases("Accelerator", "accelerator.id", "id", entries)
        if d["query"] == 'accelerator.id:"lhc"'
    }
    assert phrases == {"lhc", "CERN LHC", "LHC"}


def test_only_own_indices_may_be_deleted():
    alias = "cds-rdm-cds-search-phrases"
    name = versioned_index_name(alias)
    assert is_own_index(alias, name)
    assert not is_own_index(alias, alias)
    assert not is_own_index(alias, "cds-rdm-rdmrecords-records-record-v7.0.0")
    assert not is_own_index(alias, f"{alias}-backup")


@pytest.fixture()
def suggest_app():
    """Minimal Flask app with only the search API blueprint."""
    app = Flask("test_suggest")
    app.config.update(
        CDS_SEARCH_SUGGEST_ENABLED=True,
        CDS_SEARCH_INTERPRETER_ENABLED=False,
        CDS_SEARCH_PHRASES_INDEX="cds-search-phrases",
        CDS_SEARCH_SUGGEST_FEW_RESULTS=5,
    )
    app.register_blueprint(views.create_api_bp(app))

    @app.before_request
    def _identity():
        g.identity = mock.sentinel.identity

    return app


def _total(n):
    result = mock.Mock()
    result.total = n
    return result


def _suggest(app, q, totals, spelling="higgs boson", phrases=()):
    """Call the endpoint with OpenSearch and the records service mocked.

    :param totals: records found as typed, then with the spelling fix.
    """
    spelling_response = {
        "suggest": {"spelling": [{"options": [{"text": spelling}] if spelling else []}]}
    }
    phrases_response = {"hits": {"hits": list(phrases)}}
    with (
        mock.patch.object(views, "current_search_client", new=mock.Mock()) as client,
        mock.patch.object(
            views, "current_rdm_records_service", new=mock.Mock()
        ) as service,
    ):
        client.search.side_effect = [phrases_response, spelling_response]
        service.search.side_effect = [_total(n) for n in totals]
        res = app.test_client().get("/search/suggest", query_string={"q": q})
        res.spelling_asked = client.search.call_count == 2
        return res


def test_suggest_spelling_only_if_it_finds_more(suggest_app):
    res = _suggest(suggest_app, "higs boson", totals=[0, 120])
    assert res.json["spelling"] == {"text": "higgs boson", "total": 120}

    res = _suggest(suggest_app, "higs boson", totals=[3, 3])
    assert res.json["spelling"] is None


def test_suggest_no_spelling_fix_for_searches_with_results(suggest_app):
    # "thesis" finds many records: never "fixed" into a commoner word.
    res = _suggest(suggest_app, "thesis", totals=[13000], spelling="this")
    assert res.json["spelling"] is None
    assert not res.spelling_asked


def test_suggest_known_names(suggest_app):
    hits = [_hit("ATLAS", "Experiment", "ATLAS", 'experiment.id:"ATLAS"')]
    res = _suggest(suggest_app, "atlas", totals=[2588], spelling=None, phrases=hits)
    assert res.json["names"] == [
        {"label": "Experiment", "value": "ATLAS", "query": 'experiment.id:"ATLAS"'}
    ]


def test_suggest_without_phrase_index(suggest_app):
    with (
        mock.patch.object(views, "current_search_client", new=mock.Mock()) as client,
        mock.patch.object(
            views, "current_rdm_records_service", new=mock.Mock()
        ) as service,
    ):
        service.search.return_value = _total(0)
        client.search.side_effect = [
            views.search.exceptions.NotFoundError(404, "index_not_found_exception"),
            {"suggest": {"spelling": [{"options": []}]}},
        ]
        res = suggest_app.test_client().get("/search/suggest?q=atlas")
    assert res.status_code == 200
    assert res.json["names"] == []


def test_suggest_query_syntax_is_left_alone(suggest_app):
    with mock.patch.object(views, "current_search_client", new=mock.Mock()) as client:
        res = suggest_app.test_client().get('/search/suggest?q=title:"higgs"')
    assert res.json["spelling"] is None and res.json["names"] == []
    client.search.assert_not_called()


def test_suggest_search_unavailable(suggest_app):
    with mock.patch.object(views, "current_search_client", new=mock.Mock()) as client:
        client.search.side_effect = views.search.exceptions.ConnectionError(
            "N/A", "Connection refused", OSError("refused")
        )
        res = suggest_app.test_client().get("/search/suggest?q=higgs")
    assert res.status_code == 200
    assert res.json["spelling"] is None


def test_suggest_disabled(suggest_app):
    suggest_app.config["CDS_SEARCH_SUGGEST_ENABLED"] = False
    assert suggest_app.test_client().get("/search/suggest?q=x").status_code == 404


def test_build_phrases_index_moves_the_alias_and_deletes_only_own_indices(
    suggest_app,
):
    docs = [{"phrase": "ATLAS", "label": "Experiment", "value": "ATLAS", "query": "q"}]
    alias = "cds-search-phrases"
    old_own, other = f"{alias}-20260101000000", "someone-elses-index"
    with (
        suggest_app.app_context(),
        mock.patch.object(views, "iter_phrase_documents", return_value=iter(docs)),
        mock.patch.object(views, "current_search_client", new=mock.Mock()) as client,
    ):
        client.bulk.return_value = {"errors": False}
        client.indices.get_alias.return_value = {old_own: {}, other: {}}
        name, count = views.build_phrases_index()

    assert count == 1 and is_own_index(alias, name)
    client.indices.create.assert_called_once_with(
        index=name, body=suggest.PHRASES_INDEX_BODY
    )
    actions = client.indices.update_aliases.call_args.kwargs["body"]["actions"]
    assert {"add": {"index": name, "alias": alias}} in actions
    assert {"remove": {"index": old_own, "alias": alias}} in actions
    assert not any(other in str(a) for a in actions)
    client.indices.delete.assert_called_once_with(index=old_own)


def test_build_phrases_index_dry_run_writes_nothing(suggest_app):
    with (
        suggest_app.app_context(),
        mock.patch.object(views, "iter_phrase_documents", return_value=iter([{}])),
        mock.patch.object(views, "current_search_client", new=mock.Mock()) as client,
    ):
        _, count = views.build_phrases_index(dry_run=True)
    assert count == 1
    assert not client.method_calls
