# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 CERN.
#
# CDS-RDM is free software; you can redistribute it and/or modify it under
# the terms of the GPL-2.0 License; see LICENSE file for more details.

"""Tests for the CDS record search query parser configuration."""

import pytest
from flask_principal import AnonymousIdentity
from invenio_access.permissions import system_identity, system_permission
from invenio_records_resources.services.records.queryparser import QueryParser
from invenio_records_resources.services.records.queryparser.transformer import (
    RestrictedTerm,
)

from cds_rdm.search import (
    DEFAULT_SEARCH_FIELDS,
    FUZZY_FIELDS,
    WILDCARD_FIELDS,
    CDSSearchFieldTransformer,
)

MAPPING = {
    "doi": "pids.doi.identifier",
    "creator": "metadata.creators.person_or_org.name",
    "internal_notes.note": RestrictedTerm(system_permission),
}


def parse(query, identity=None):
    """Parse ``query`` with the parser configured as in ``invenio.cfg``."""
    parser_cls = QueryParser.factory(
        fields=DEFAULT_SEARCH_FIELDS,
        lenient=True,
        mapping=MAPPING,
        tree_transformer_cls=CDSSearchFieldTransformer,
    )
    return parser_cls(identity).parse(query).to_dict()


def sent(query):
    """The query string sent to OpenSearch, whitespace normalised."""
    return " ".join(parse(query)["query_string"]["query"].split())


def test_plain_query_searches_the_default_fields():
    assert parse("higgs") == {
        "query_string": {
            "query": "higgs",
            "fields": DEFAULT_SEARCH_FIELDS,
            "lenient": True,
        }
    }


@pytest.mark.parametrize(
    "query,expected",
    [
        ("thesis Mobius", "thesis AND Mobius"),
        (
            "creator:mobius thesis",
            "metadata.creators.person_or_org.name:mobius AND thesis",
        ),
        ("a b OR c", "a AND (b OR c)"),
        ('"pixel detector" atlas', '"pixel detector" AND atlas'),
        ("creator:(john doe)", "metadata.creators.person_or_org.name:(john AND doe)"),
        # Explicit operators are kept.
        ("higgs OR boson", "higgs OR boson"),
        ("higgs AND boson", "higgs AND boson"),
        # +/- already say what is required: left alone.
        ("+thesis mobius^5", "+thesis mobius^5"),
        ("higgs -boson", "higgs -boson"),
    ],
)
def test_space_means_and(query, expected):
    assert sent(query) == expected


@pytest.mark.parametrize(
    "query,expected",
    [
        ("10.17181/CERN.ABCD.1234", '"10.17181/CERN.ABCD.1234"'),
        ("doi:10.1007/x", 'pids.doi.identifier:"10.1007/x"'),
        ("CERN-THESIS-2021-078", '"CERN-THESIS-2021-078"'),
        ("state-of-the-art detector", '"state-of-the-art" AND detector'),
        # Left alone: escapes, regular expressions, a leading "-".
        (r"10.1007\/x", r"10.1007\/x"),
        ("/high.*energy/", "/high.*energy/"),
        ("higgs -boson", "higgs -boson"),
    ],
)
def test_compound_words_are_phrases(query, expected):
    assert sent(query) == expected


def test_default_fields_have_no_wildcards():
    # "*" would also search restricted fields such as internal_notes.
    assert not any("*" in field for field in DEFAULT_SEARCH_FIELDS)


def test_restricted_terms_still_restricted(app):
    # A normal user's search on a restricted field is rejected and falls back to a
    # plain multi_match on the text, so it never queries internal_notes.
    assert "multi_match" in parse("internal_notes.note:secret", AnonymousIdentity())
    query = parse("internal_notes.note:secret", system_identity)
    assert query["query_string"]["query"] == "internal_notes.note:secret"


def _in(fields, value):
    fields = [field.replace(":", "\\:") for field in fields]
    return "(" + " OR ".join(f"{field}:{value}" for field in fields) + ")"


@pytest.mark.parametrize(
    "query,expected",
    [
        # No distance typed: OpenSearch's AUTO distance, written out.
        ("higgs~", _in(FUZZY_FIELDS, "higgs~1")),
        ("calorimeter~", _in(FUZZY_FIELDS, "calorimeter~2")),
        ("higgs~2", _in(FUZZY_FIELDS, "higgs~2")),
        ("detect*", _in(WILDCARD_FIELDS, "detect*")),
        ("CERN-THESIS-*", _in(WILDCARD_FIELDS, "CERN-THESIS-*")),
        ("-calor*", "-" + _in(WILDCARD_FIELDS, "calor*")),
        # A field search keeps its field.
        ("creator:higgs~", "metadata.creators.person_or_org.name:higgs~1"),
        ("doi:10.1007*", "pids.doi.identifier:10.1007*"),
        # "*" alone is left to the upstream parser.
        ("*", "*"),
    ],
)
def test_fuzzy_and_wildcard_words_search_few_fields(query, expected):
    assert sent(query) == expected


def _clauses(query):
    """Upper bound of the clauses OpenSearch builds for ``query``.

    A plain word: one per default field. A fuzzy word: up to 50 terms per field.
    A wildcard word: up to 16 terms per field before it becomes a filter.
    """
    text = sent(query)
    plain = sum(
        1 for word in query.split() if not word.endswith("~") and "*" not in word
    )
    fuzzy = text.count("~")
    wildcard = text.count("*")
    return plain * len(DEFAULT_SEARCH_FIELDS) + fuzzy * 50 + wildcard * 16


def test_mixed_search_stays_under_the_clause_limit():
    # Over all default fields, one fuzzy word alone could need 45 * 50 clauses.
    query = "higgs boson pixel tracker upgrade detectr~ calorimetr~ trigr~ calib*"
    assert _clauses(query) < 1024
