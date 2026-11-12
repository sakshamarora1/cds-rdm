# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 CERN.
#
# CDS-RDM is free software; you can redistribute it and/or modify it under
# the terms of the GPL-2.0 License; see LICENSE file for more details.

"""Semantic search tests (no live OpenSearch or model needed)."""

from unittest import mock

import pytest
from flask import Flask, g
from invenio_search.engine import search

from cds_rdm.semantic_search import views
from cds_rdm.semantic_search.query import (
    InvalidParams,
    build_boosted_id_query,
    build_neural_query,
    validate_params,
)

FIELD = "semantic_chunks_embedding.knn"


def test_build_neural_query_nested():
    body = build_neural_query("dark matter", "m1", FIELD, k=100, size=10)
    assert body["size"] == 10
    must = body["query"]["bool"]["must"][0]
    assert must["nested"]["path"] == "semantic_chunks_embedding"
    assert must["nested"]["score_mode"] == "max"
    neural = must["nested"]["query"]["neural"][FIELD]
    assert neural == {"query_text": "dark matter", "model_id": "m1", "k": 100}
    assert body["query"]["bool"]["filter"] == [{"term": {"access.record": "public"}}]


def test_build_neural_query_flat_field_and_k_at_least_size():
    body = build_neural_query("q", "m1", "embedding", k=5, size=20)
    neural = body["query"]["bool"]["must"][0]["neural"]["embedding"]
    assert neural["k"] == 20


def test_build_boosted_id_query():
    q = build_boosted_id_query([("aaaaa-00001", 0.69718474), ("bbbbb-00002", 0.0)])
    assert q == 'id:"aaaaa-00001"^0.697185 OR id:"bbbbb-00002"^0.000001'
    assert build_boosted_id_query([]) == ""


def test_boosted_id_query_parses_with_rdm_query_parser():
    """The query string survives luqum parsing (used by the RDM query parser)."""
    from luqum.parser import parser

    tree = parser.parse(
        build_boosted_id_query([("aaaaa-00001", 0.5), ("bbbbb-00002", 0.25)])
    )
    assert str(tree) == 'id:"aaaaa-00001"^0.5 OR id:"bbbbb-00002"^0.25'


@pytest.mark.parametrize(
    "q,size,expected",
    [("  hello ", None, ("hello", 10)), ("x", "5", ("x", 5)), ("x", 50, ("x", 50))],
)
def test_validate_params_ok(q, size, expected):
    assert validate_params(q, size, max_size=50) == expected


@pytest.mark.parametrize(
    "q,size", [("", None), ("   ", None), (None, 5), ("x", "a"), ("x", 0), ("x", 51)]
)
def test_validate_params_invalid(q, size):
    with pytest.raises(InvalidParams):
        validate_params(q, size, max_size=50)


@pytest.fixture()
def sem_app():
    """Minimal Flask app with only the semantic search API blueprint."""
    app = Flask("test_semantic")
    app.config.update(
        CDS_SEMANTIC_SEARCH_ENABLED=True,
        CDS_SEMANTIC_SEARCH_INDEX="sem-index",
        CDS_SEMANTIC_SEARCH_MODEL_ID="m1",
        CDS_SEMANTIC_SEARCH_FIELD=FIELD,
        CDS_SEMANTIC_SEARCH_K=100,
        CDS_SEMANTIC_SEARCH_MAX_SIZE=50,
    )
    app.register_blueprint(views.create_api_bp(app))

    @app.before_request
    def _identity():
        g.identity = mock.sentinel.identity

    return app


def _get(app, qs):
    return app.test_client().get(f"/semantic-search{qs}")


def test_view_empty_query(sem_app):
    res = _get(sem_app, "?q=")
    assert res.status_code == 400
    assert "required" in res.json["message"]


def test_view_missing_model_id(sem_app):
    sem_app.config["CDS_SEMANTIC_SEARCH_MODEL_ID"] = None
    res = _get(sem_app, "?q=test")
    assert res.status_code == 503
    assert "MODEL_ID" in res.json["message"]


@pytest.mark.parametrize(
    "exc,expected",
    [
        (search.ConnectionError("N/A", "refused", None), "unreachable"),
        (
            search.NotFoundError(
                404,
                "status_exception",
                {"error": {"root_cause": [{"reason": "Failed to find model"}]}},
            ),
            "Failed to find model",
        ),
        (
            search.NotFoundError(
                404,
                "index_not_found_exception",
                {"error": {"root_cause": [{"reason": "no such index [sem-index]"}]}},
            ),
            "not found",
        ),
        (
            search.RequestError(
                400,
                "search_phase_execution_exception",
                {
                    "error": {
                        "root_cause": [
                            {"reason": "Query vector has invalid dimension: 768"}
                        ]
                    }
                },
            ),
            "Embedding model problem",
        ),
    ],
)
def test_view_backend_errors(sem_app, exc, expected):
    with mock.patch.object(views, "current_search_client", new=mock.Mock()) as client:
        client.search.side_effect = exc
        res = _get(sem_app, "?q=test")
    assert res.status_code == 503
    assert expected in res.json["message"]


def test_view_success_keeps_semantic_order_and_drops_invisible(sem_app):
    os_response = {
        "hits": {
            "hits": [
                {"_source": {"id": "aaaaa-00001"}, "_score": 0.9},
                {"_source": {"id": "bbbbb-00002"}, "_score": 0.8},
                {"_source": {"id": "ccccc-00003"}, "_score": 0.7},
            ]
        }
    }
    # Records service returns only what the identity may see, in its own order.
    service_result = mock.Mock()
    service_result.to_dict.return_value = {
        "hits": {
            "hits": [
                {
                    "id": "ccccc-00003",
                    "metadata": {"title": "C"},
                    "links": {"self_html": "https://x/records/ccccc-00003"},
                },
                {
                    "id": "aaaaa-00001",
                    "metadata": {"title": "A"},
                    "links": {"self_html": "https://x/records/aaaaa-00001"},
                },
            ]
        }
    }
    with (
        mock.patch.object(views, "current_search_client", new=mock.Mock()) as client,
        mock.patch.object(
            views, "current_rdm_records_service", new=mock.Mock()
        ) as service,
    ):
        client.search.return_value = os_response
        service.search.return_value = service_result
        res = _get(sem_app, "?q=dark+matter&size=5")

    assert res.status_code == 200
    assert [h["id"] for h in res.json["hits"]] == ["aaaaa-00001", "ccccc-00003"]
    assert res.json["hits"][0] == {
        "id": "aaaaa-00001",
        "title": "A",
        "score": 0.9,
        "url": "https://x/records/aaaaa-00001",
    }
    # Neural query went to the configured index; permission check used the identity.
    assert client.search.call_args.kwargs["index"] == "sem-index"
    assert service.search.call_args.args[0] is mock.sentinel.identity
    assert "extra_filter" in service.search.call_args.kwargs
