# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 CERN.
#
# CDS-RDM is free software; you can redistribute it and/or modify it under
# the terms of the GPL-2.0 License; see LICENSE file for more details.

"""REST endpoints helping with a free-text search.

- ``/api/search/interpret``: what an LLM understood the search to mean.
- ``/api/search/suggest``: spelling fixes and known names, no model needed.
"""

import hashlib

from flask import Blueprint, abort, current_app, g, jsonify, request
from flask_principal import AnonymousIdentity
from invenio_access.permissions import any_user, system_identity
from invenio_cache import current_cache
from invenio_communities.proxies import current_communities
from invenio_rdm_records.proxies import current_rdm_records_service
from invenio_rdm_records.records.api import RDMRecord
from invenio_search import current_search_client
from invenio_search.engine import search
from invenio_search.utils import prefix_index
from invenio_vocabularies.proxies import current_service as vocabulary_service

from .interpreter import (
    InterpreterError,
    build_messages,
    call_ollama,
    community_matches,
    match_report_prefixes,
    match_vocabulary,
    quote,
    render,
    report_number_formats,
    report_number_prefixes,
    should_interpret,
)
from .suggest import (
    PHRASES_INDEX_BODY,
    community_phrases,
    is_own_index,
    is_plain_text,
    known_names,
    phrases_body,
    spelling_body,
    spelling_from_response,
    versioned_index_name,
    vocabulary_phrases,
)

# The repository facts given to the model change rarely: rebuild them daily.
FACTS_CACHE_TTL = 24 * 60 * 60
REPORT_NUMBER_SAMPLE = 100  # public records sampled per resource type


def resource_types():
    """``{id: English title}`` of the resource types vocabulary."""
    results = vocabulary_service.read_all(
        system_identity, fields=["id", "title"], type="resourcetypes", max_records=200
    )
    return {
        hit["id"]: hit.get("title", {}).get("en", hit["id"])
        for hit in results.to_dict()["hits"]["hits"]
    }


def _anonymous_identity():
    identity = AnonymousIdentity()
    identity.provides.add(any_user)
    return identity


def sampled_report_numbers():
    """``[(resource_type_id, report_number), ...]`` from a sample of public records."""
    key = "cds-search-interpret:report-numbers:v1"
    sample = current_cache.get(key)
    if sample is not None:
        return sample
    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"access.record": "public"}},
                    {"term": {"metadata.identifiers.scheme": "cdsrn"}},
                ]
            }
        },
        "aggs": {
            "types": {
                "terms": {"field": "metadata.resource_type.id", "size": 200},
                "aggs": {
                    "sample": {
                        "top_hits": {
                            "size": REPORT_NUMBER_SAMPLE,
                            "_source": ["metadata.identifiers"],
                        }
                    }
                },
            }
        },
    }
    response = current_search_client.search(
        index=prefix_index(RDMRecord.index.search_alias), body=body
    )
    sample = [
        (bucket["key"], identifier.get("identifier"))
        for bucket in response["aggregations"]["types"]["buckets"]
        for hit in bucket["sample"]["hits"]["hits"]
        for identifier in hit["_source"].get("metadata", {}).get("identifiers", [])
        if identifier.get("scheme") == "cdsrn"
    ]
    current_cache.set(key, sample, timeout=FACTS_CACHE_TTL)
    return sample


def sampled_report_formats():
    """Report number formats per resource type, from the sampled report numbers.

    ``{resource_type_id: ["CERN-OPER-CIRC-##", ...]}``. Lets the model write the
    report number of a numbered document of any series, without listing the
    series in the prompt.
    """
    return report_number_formats(sampled_report_numbers())


def vocabulary_entries(vocabulary_type):
    """``[(id, English title, [alias, ...]), ...]`` of a vocabulary, cached."""
    key = f"cds-search-interpret:vocabulary:{vocabulary_type}"
    entries = current_cache.get(key)
    if entries is not None:
        return entries
    results = vocabulary_service.read_all(
        system_identity,
        fields=["id", "title", "props"],
        type=vocabulary_type,
        max_records=10000,
    )
    entries = []
    for hit in results.to_dict()["hits"]["hits"]:
        aliases = (hit.get("props") or {}).get("aliases") or ""
        aliases = aliases if isinstance(aliases, list) else aliases.split(",")
        entries.append(
            (
                hit["id"],
                (hit.get("title") or {}).get("en", hit["id"]),
                [a.strip() for a in aliases if a and a.strip()],
            )
        )
    current_cache.set(key, entries, timeout=FACTS_CACHE_TTL)
    return entries


def find_facets(name):
    """What ``name`` can be in the repository: ``[(label, clause, value), ...]``.

    For "NAME records" searches: the communities, CERN vocabulary values
    (``CDS_SEARCH_INTERPRETER_FACETS``) and report number series NAME stands
    for. Records matching any of them are the records of NAME.
    """
    found = []
    for slug, title in find_community(name):
        if community_matches(name, title):
            found.append(("Community", f"community.id:{quote(slug)}", title))
            break
    for vocabulary_type, label, shortcut, use in current_app.config[
        "CDS_SEARCH_INTERPRETER_FACETS"
    ]:
        try:
            entries = vocabulary_entries(vocabulary_type)
        except Exception:  # vocabulary not loaded on this instance
            current_app.logger.warning("Vocabulary %s unavailable", vocabulary_type)
            continue
        for entry_id, title in match_vocabulary(name, entries, kind=label)[:2]:
            value = entry_id if use == "id" else title
            found.append((label, f"{shortcut}:{quote(value)}", title))
    prefixes = report_number_prefixes(sampled_report_numbers())
    for prefix in match_report_prefixes(name, prefixes):
        found.append(("Report numbers", f"identifier:{quote(prefix)}", f"{prefix}-…"))
    return found


def resource_types_in_use():
    """Resource type ids that public records use: the only ones worth offering."""
    key = "cds-search-interpret:types-in-use"
    ids = current_cache.get(key)
    if ids is not None:
        return ids
    response = current_search_client.search(
        index=prefix_index(RDMRecord.index.search_alias),
        body={
            "size": 0,
            "query": {"term": {"access.record": "public"}},
            "aggs": {
                "types": {"terms": {"field": "metadata.resource_type.id", "size": 500}}
            },
        },
    )
    ids = sorted(b["key"] for b in response["aggregations"]["types"]["buckets"])
    current_cache.set(key, ids, timeout=FACTS_CACHE_TTL)
    return ids


def public_community_names(max_pages=5):
    """Titles of the communities anyone can see, sorted."""
    key = "cds-search-interpret:communities"
    names = current_cache.get(key)
    if names is not None:
        return names
    names = set()
    for page in range(1, max_pages + 1):
        results = current_communities.service.search(
            _anonymous_identity(), params={"size": 100, "page": page}
        )
        hits = list(results.hits)
        names.update(hit["metadata"]["title"] for hit in hits)
        if len(hits) < 100:
            break
    names = sorted(names)
    current_cache.set(key, names, timeout=FACTS_CACHE_TTL)
    return names


def find_community(name):
    """Communities the current user can see that best match ``name``."""
    results = current_communities.service.search(
        g.identity, params={"q": name, "size": 5}
    )
    return [(hit["slug"], hit["metadata"]["title"]) for hit in results.hits]


def has_creator(name):
    """Whether any record the current user can see has ``name`` as a creator."""
    words = " ".join(f"+{w}" for w in name.split() if w.isalnum())
    if not words:
        return False
    results = current_rdm_records_service.search(
        g.identity, params={"q": f"creator:({words})", "size": 1}
    )
    return results.total > 0


def _cache_key(model, messages):
    """Per model and prompt: a new prompt or new repository facts re-ask the model."""
    text = "\0".join([model, *(m["content"] for m in messages)])
    return f"cds-search-interpret:{hashlib.sha256(text.encode()).hexdigest()}"


def interpret(query):
    """The LLM's answer for ``query``, cached; ``None`` if the model failed."""
    cfg = current_app.config
    model = cfg["CDS_SEARCH_INTERPRETER_MODEL"]
    in_use = set(resource_types_in_use())
    types = resource_types()
    messages = build_messages(
        query,
        sorted((rid, title) for rid, title in types.items() if rid in in_use),
        cfg["CDS_SEARCH_INTERPRETER_CONTEXT"],
        report_formats=sampled_report_formats(),
        communities=public_community_names(),
    )
    key = _cache_key(model, messages)
    data = current_cache.get(key)
    if data is None:
        try:
            data = call_ollama(
                messages,
                url=cfg["CDS_SEARCH_INTERPRETER_URL"],
                model=model,
                timeout=cfg["CDS_SEARCH_INTERPRETER_TIMEOUT"],
            )
        except InterpreterError as e:
            current_app.logger.warning("Search interpreter unavailable: %s", e)
            return None
        current_cache.set(key, data, timeout=cfg["CDS_SEARCH_INTERPRETER_CACHE_TTL"])
    return data


def interpret_view():
    """``GET /api/search/interpret?q=...``.

    Returns ``{"query": null}`` when the search is better searched as typed: not
    plain text, nothing understood, model unavailable, or no results.
    """
    if not current_app.config.get("CDS_SEARCH_INTERPRETER_ENABLED"):
        abort(404)
    query = " ".join(request.args.get("q", "").split())
    nothing = {"original": query, "query": None, "parts": []}
    if not should_interpret(query):
        return jsonify(nothing)

    data = interpret(query)
    result = (
        render(
            query,
            data,
            resource_types(),
            find_community,
            has_creator,
            report_formats=sampled_report_formats(),
            find_facets=find_facets,
        )
        if data
        else None
    )
    if not result:
        return jsonify(nothing)

    # Never send people to an empty result page: the model's guess may be wrong.
    total = current_rdm_records_service.search(
        g.identity, params={"q": result["query"], "size": 1}
    ).total
    if not total:
        return jsonify(nothing)
    return jsonify({"original": query, **result, "total": total})


# Suggestions
# ===========


def phrases_alias():
    """Prefixed alias of the phrase index."""
    return prefix_index(current_app.config["CDS_SEARCH_PHRASES_INDEX"])


def spelling(text):
    """Corrected ``text`` from public record titles, or ``None``."""
    response = current_search_client.search(
        index=prefix_index(RDMRecord.index.search_alias), body=spelling_body(text)
    )
    return spelling_from_response(text, response)


def find_known_names(text):
    """Known names typed in ``text``; none if the phrase index was not built."""
    try:
        response = current_search_client.search(
            index=phrases_alias(), body=phrases_body(text)
        )
    except search.exceptions.NotFoundError:
        return []
    return known_names(text, response["hits"]["hits"])


def suggest_view():
    """``GET /api/search/suggest?q=...``.

    ``{"original", "spelling": {"text", "total"} | null, "names": [...]}``.
    A spelling fix is only looked for when the search as typed finds few records
    (``CDS_SEARCH_SUGGEST_FEW_RESULTS``), and returned if it finds more: a
    correct but common word ("thesis") is never "fixed".
    """
    cfg = current_app.config
    if not cfg.get("CDS_SEARCH_SUGGEST_ENABLED"):
        abort(404)
    query = " ".join(request.args.get("q", "").split())
    result = {"original": query, "spelling": None, "names": []}
    if not is_plain_text(query):
        return jsonify(result)
    try:
        result["names"] = find_known_names(query)
        typed = current_rdm_records_service.search(
            g.identity, params={"q": query, "size": 1}
        ).total
        fixed = (
            spelling(query) if typed <= cfg["CDS_SEARCH_SUGGEST_FEW_RESULTS"] else None
        )
    except search.exceptions.TransportError as e:
        current_app.logger.warning("Search suggestions unavailable: %s", e)
        return jsonify(result)

    if fixed:
        found = current_rdm_records_service.search(
            g.identity, params={"q": fixed, "size": 1}
        ).total
        if found > typed:
            result["spelling"] = {"text": fixed, "total": found}
    return jsonify(result)


def public_communities(max_pages=20):
    """``[(slug, title), ...]`` of the communities anyone can see."""
    found = []
    for page in range(1, max_pages + 1):
        results = current_communities.service.search(
            _anonymous_identity(), params={"size": 100, "page": page}
        )
        hits = list(results.hits)
        found.extend((hit["slug"], hit["metadata"]["title"]) for hit in hits)
        if len(hits) < 100:
            break
    return found


def iter_phrase_documents():
    """Phrase documents: CERN vocabularies, resource types, public communities."""
    for vocabulary_type, label, shortcut, use in current_app.config[
        "CDS_SEARCH_PHRASES_VOCABULARIES"
    ]:
        try:
            entries = vocabulary_entries(vocabulary_type)
        except Exception:  # vocabulary not loaded on this instance
            current_app.logger.warning("Vocabulary %s unavailable", vocabulary_type)
            continue
        yield from vocabulary_phrases(label, shortcut, use, entries)
    yield from community_phrases(public_communities())


# UPSTREAM(invenio-search): building an index under a new name and moving an
# alias to it is what ``invenio index`` does for record mappings. Use that once
# the phrase index has been tested in dev.
def build_phrases_index(dry_run=False):
    """Build a new phrase index, point the alias at it, delete the old ones.

    Only indices named ``<alias>-<timestamp>`` are deleted, never anything else.
    Returns ``(new index name, number of phrases)``.
    """
    alias = phrases_alias()
    docs = list(iter_phrase_documents())
    name = versioned_index_name(alias)
    if dry_run:
        return name, len(docs)
    client = current_search_client
    client.indices.create(index=name, body=PHRASES_INDEX_BODY)
    body = []
    for doc in docs:
        body.extend([{"index": {"_index": name}}, doc])
    if body:
        response = client.bulk(body=body, refresh=True)
        if response.get("errors"):
            client.indices.delete(index=name)
            raise RuntimeError("Indexing the phrases failed; the alias was not moved.")
    old = [
        index
        for index in (client.indices.get_alias(name=alias, ignore=404) or {})
        if index != name and is_own_index(alias, index)
    ]
    actions = [{"add": {"index": name, "alias": alias}}]
    actions += [{"remove": {"index": index, "alias": alias}} for index in old]
    client.indices.update_aliases(body={"actions": actions})
    for index in old:
        client.indices.delete(index=index)
    return name, len(docs)


def create_api_bp(app):
    """Blueprint for the search interpretation and suggestion endpoints."""
    blueprint = Blueprint("cds_search", __name__)
    blueprint.add_url_rule("/search/interpret", view_func=interpret_view)
    blueprint.add_url_rule("/search/suggest", view_func=suggest_view)
    return blueprint
