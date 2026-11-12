# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 CERN.
#
# CDS-RDM is free software; you can redistribute it and/or modify it
# under the terms of the GPL-2.0 License; see LICENSE file for more details.

"""Query building and parameter validation for semantic search."""

MAX_QUERY_LENGTH = 500


class InvalidParams(ValueError):
    """Raised when the request parameters are invalid."""


def validate_params(q, size, max_size, default_size=10):
    """Return a cleaned ``(q, size)`` tuple or raise ``InvalidParams``."""
    q = (q or "").strip()
    if not q:
        raise InvalidParams("Query parameter 'q' is required.")
    if len(q) > MAX_QUERY_LENGTH:
        raise InvalidParams(f"Query is too long (max {MAX_QUERY_LENGTH} characters).")

    if size in (None, ""):
        size = default_size
    try:
        size = int(size)
    except (TypeError, ValueError):
        raise InvalidParams("Parameter 'size' must be an integer.")
    if not 1 <= size <= max_size:
        raise InvalidParams(f"Parameter 'size' must be between 1 and {max_size}.")
    return q, size


def build_neural_query(text, model_id, field, k, size):
    """Build a nested neural query; a record scores as its best-matching chunk.

    ``field`` is the full path of the knn_vector, e.g.
    ``semantic_chunks_embedding.knn``; its parent is the nested path. ``k`` is the
    number of nearest *chunks*, so it must be at least ``size``.
    """
    nested_path = field.rsplit(".", 1)[0] if "." in field else None
    neural = {
        "neural": {field: {"query_text": text, "model_id": model_id, "k": max(k, size)}}
    }
    if nested_path:
        neural = {"nested": {"path": nested_path, "score_mode": "max", "query": neural}}
    return {
        "size": size,
        "_source": ["id", "metadata.title"],
        "query": {
            "bool": {
                "must": [neural],
                # The semantic index is a public-only snapshot; filter again anyway.
                "filter": [{"term": {"access.record": "public"}}],
            }
        },
    }


def build_boosted_id_query(candidates):
    """Turn ``[(pid, score), ...]`` into a query string ranking records by score.

    Each clause is a term query on the keyword ``id`` field. Every PID matches
    exactly one document, so BM25 gives every clause the same base score and the
    boost alone decides the order. The records service then sorts by
    ``bestmatch`` in semantic order, while facets, pagination and permissions
    work as in the normal search.
    """
    return " OR ".join(
        f'id:"{pid}"^{max(score, 1e-6):.6f}' for pid, score in candidates
    )
