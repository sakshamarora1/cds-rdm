# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 CERN.
#
# CDS-RDM is free software; you can redistribute it and/or modify it
# under the terms of the GPL-2.0 License; see LICENSE file for more details.

"""RDM-compatible records search resource ranked by semantic similarity.

It returns the same response as ``/api/records``, so the standard InvenioRDM
search React app (facets, sorting, pagination, result items) works on top of it.
"""

from flask import current_app, g
from flask_resources import (
    HTTPJSONException,
    resource_requestctx,
    response_handler,
    route,
)
from invenio_rdm_records.resources import RDMRecordResource, RDMRecordResourceConfig
from invenio_records_resources.resources.records.resource import (
    request_extra_args,
    request_search_args,
)
from invenio_records_resources.resources.records.utils import search_preference
from invenio_search.engine import dsl

from .query import MAX_QUERY_LENGTH, build_boosted_id_query
from .views import SemanticSearchUnavailable, semantic_candidates


class SemanticSearchRecordResourceConfig(RDMRecordResourceConfig):
    """Same serializers and args as the records resource, different URL."""

    blueprint_name = "cds_semantic_search_records"
    url_prefix = "/semantic-search"


class SemanticSearchRecordResource(RDMRecordResource):
    """Only exposes ``GET /semantic-search/records``."""

    def create_url_rules(self):
        """Create the URL rules.

        ``RecordResource`` builds its blueprint without a prefix, so the prefix
        has to be part of the rule.
        """
        return [route("GET", f"{self.config.url_prefix}/records", self.search)]

    @request_extra_args
    @request_search_args
    @response_handler(many=True)
    def search(self):
        """Semantic candidates, then the normal records search over them.

        An empty ``q`` behaves like the normal search (newest records).
        """
        args = dict(resource_requestctx.args)
        text = (args.get("q") or "").strip()
        extra_filter = None
        if text:
            if len(text) > MAX_QUERY_LENGTH:
                raise HTTPJSONException(
                    code=400,
                    description=f"Query is too long (max {MAX_QUERY_LENGTH} characters).",
                )
            try:
                candidates = semantic_candidates(
                    text, current_app.config["CDS_SEMANTIC_SEARCH_CANDIDATES"]
                )
            except SemanticSearchUnavailable as e:
                raise HTTPJSONException(code=503, description=str(e))
            args["q"] = build_boosted_id_query(candidates)
            # Also restrict to the candidates; an empty list matches nothing.
            extra_filter = dsl.Q("terms", id=[pid for pid, _ in candidates])

        hits = self.service.search(
            identity=g.identity,
            params=args,
            search_preference=search_preference(),
            expand=args.get("expand", False),
            extra_filter=extra_filter,
        )
        return hits.to_dict(), 200
