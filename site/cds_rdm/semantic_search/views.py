# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 CERN.
#
# CDS-RDM is free software; you can redistribute it and/or modify it
# under the terms of the GPL-2.0 License; see LICENSE file for more details.

"""Semantic search REST endpoint and UI page (prototype)."""

from flask import Blueprint, abort, current_app, g, jsonify, render_template, request
from invenio_rdm_records.proxies import current_rdm_records_service
from invenio_search import current_search_client
from invenio_search.engine import dsl, search
from invenio_search_ui.searchconfig import search_app_config

from .query import InvalidParams, build_neural_query, validate_params

# Fetch more candidates than requested: some are dropped by the permission check.
CANDIDATE_FACTOR = 3


class SemanticSearchUnavailable(Exception):
    """Search backend or embedding model is not usable."""


def _error(status, message):
    return jsonify({"status": status, "message": message}), status


def _reason(exc):
    """Best-effort extraction of the OpenSearch error reason."""
    info = getattr(exc, "info", None)
    if isinstance(info, dict):
        err = info.get("error", {})
        if isinstance(err, dict):
            causes = err.get("root_cause") or [err]
            return str(causes[0].get("reason", err))
        return str(err)
    return str(exc)


def semantic_candidates(text, n):
    """Run the neural query; return the top ``n`` as ``[(pid, score), ...]``."""
    cfg = current_app.config
    model_id = cfg.get("CDS_SEMANTIC_SEARCH_MODEL_ID")
    if not model_id:
        raise SemanticSearchUnavailable(
            "CDS_SEMANTIC_SEARCH_MODEL_ID is not configured."
        )
    index = cfg["CDS_SEMANTIC_SEARCH_INDEX"]
    body = build_neural_query(
        text,
        model_id=model_id,
        field=cfg["CDS_SEMANTIC_SEARCH_FIELD"],
        k=cfg["CDS_SEMANTIC_SEARCH_K"],
        size=n,
    )
    try:
        res = current_search_client.search(index=index, body=body)
    except search.ConnectionError as e:
        current_app.logger.warning("Semantic search: OpenSearch unreachable: %s", e)
        raise SemanticSearchUnavailable("Search backend is unreachable.")
    except search.NotFoundError as e:
        reason = _reason(e)
        if "index" in getattr(e, "error", "") or "no such index" in reason:
            raise SemanticSearchUnavailable(f"Semantic index '{index}' not found.")
        raise SemanticSearchUnavailable(f"Embedding model problem: {reason}")
    except search.TransportError as e:
        reason = _reason(e)
        current_app.logger.warning("Semantic search failed: %s", reason)
        if "model" in reason.lower() or "dimension" in reason.lower():
            raise SemanticSearchUnavailable(f"Embedding model problem: {reason}")
        raise SemanticSearchUnavailable("Semantic search failed.")

    return [(h["_source"]["id"], h["_score"]) for h in res["hits"]["hits"]]


def resolve_visible(identity, candidates, size):
    """Keep candidates the identity may read, in semantic order, as result dicts.

    The records service applies the usual RDM permission filter plus the
    published / not-deleted / latest-version defaults.
    """
    if not candidates:
        return []
    pids = [pid for pid, _ in candidates]
    result = current_rdm_records_service.search(
        identity,
        params={"size": len(pids)},
        extra_filter=dsl.Q("terms", id=pids),
    )
    by_pid = {hit["id"]: hit for hit in result.to_dict()["hits"]["hits"]}

    out = []
    for pid, score in candidates:
        hit = by_pid.get(pid)
        if hit is None:
            continue
        out.append(
            {
                "id": pid,
                "title": hit.get("metadata", {}).get("title"),
                "score": score,
                "url": hit.get("links", {}).get("self_html"),
            }
        )
        if len(out) == size:
            break
    return out


def semantic_search_view():
    """GET /semantic-search?q=<text>&size=<n>."""
    cfg = current_app.config
    if not cfg.get("CDS_SEMANTIC_SEARCH_ENABLED"):
        abort(404)
    try:
        q, size = validate_params(
            request.args.get("q"),
            request.args.get("size"),
            max_size=cfg["CDS_SEMANTIC_SEARCH_MAX_SIZE"],
        )
    except InvalidParams as e:
        return _error(400, str(e))

    try:
        candidates = semantic_candidates(q, size * CANDIDATE_FACTOR)
    except SemanticSearchUnavailable as e:
        return _error(503, str(e))

    hits = resolve_visible(g.identity, candidates, size)
    return jsonify({"query": q, "total": len(hits), "hits": hits})


def semantic_search_page():
    """The standard RDM search page, backed by the semantic records endpoint."""
    cfg = current_app.config
    if not cfg.get("CDS_SEMANTIC_SEARCH_ENABLED"):
        abort(404)
    search_config = search_app_config(
        "RDM_SEARCH",
        cfg["RDM_FACETS"],
        cfg["RDM_SORT_OPTIONS"],
        "/api/semantic-search/records",
        {"Accept": "application/vnd.inveniordm.v1+json"},
        pagination_options=(10, 20),
        # Same app id as /search, so the same component overrides apply.
        app_id="InvenioAppRdm.Search",
    )
    return render_template(
        "cds_rdm/semantic_search.html", search_app_config=search_config
    )


def create_api_bp(app):
    """Blueprint for the REST endpoint (mounted under /api)."""
    bp = Blueprint("cds_semantic_search_api", __name__)
    bp.add_url_rule("/semantic-search", view_func=semantic_search_view)
    return bp


def create_records_api_bp(app):
    """Blueprint for ``/api/semantic-search/records`` (RDM search format)."""
    from .resources import (
        SemanticSearchRecordResource,
        SemanticSearchRecordResourceConfig,
    )

    ext = app.extensions["invenio-rdm-records"]
    resource = SemanticSearchRecordResource(
        service=ext.records_service,
        config=SemanticSearchRecordResourceConfig.build(app),
    )
    return resource.as_blueprint()


def create_ui_bp(app):
    """Blueprint for the UI page."""
    bp = Blueprint("cds_semantic_search_ui", __name__)
    bp.add_url_rule("/semantic-search", view_func=semantic_search_page)
    return bp
