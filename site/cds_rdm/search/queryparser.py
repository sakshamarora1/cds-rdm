# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 CERN.
#
# CDS-RDM is free software; you can redistribute it and/or modify it under
# the terms of the GPL-2.0 License; see LICENSE file for more details.

"""Search configuration for the CDS record search bar.

Used with the upstream ``QueryParser`` (see ``invenio.cfg``)::

    QueryParser.factory(
        fields=DEFAULT_SEARCH_FIELDS,
        lenient=True,
        mapping={...},
        tree_transformer_cls=CDSSearchFieldTransformer,
    )
"""

import re
from copy import deepcopy
from decimal import Decimal

from invenio_records_resources.services.records.queryparser import (
    SearchFieldTransformer,
)
from luqum.tree import (
    AndOperation,
    Fuzzy,
    Group,
    OrOperation,
    Phrase,
    Plus,
    Prohibit,
    SearchField,
    Word,
)

DEFAULT_SEARCH_FIELDS = [
    # Mirrors ``index.query.default_field`` of the upstream record mapping
    # (invenio-rdm-records, rdmrecords/records/record-v7.0.0.json), with boosts.
    "id^10",
    "metadata.title^10",
    "metadata.title.original^10",
    "metadata.additional_titles.title^5",
    "metadata.creators.person_or_org.name^5",
    "metadata.creators.person_or_org.family_name^5",
    "metadata.creators.person_or_org.given_name^2",
    "metadata.creators.affiliations.name",
    "metadata.contributors.person_or_org.name^2",
    "metadata.contributors.person_or_org.family_name^2",
    "metadata.contributors.person_or_org.given_name",
    "metadata.contributors.affiliations.name",
    "metadata.identifiers.identifier^10",
    "metadata.subjects.subject^3",
    "metadata.description",
    "metadata.additional_descriptions.description",
    "metadata.contact",
    "metadata.formats",
    "metadata.funding.award.identifiers.identifier",
    "metadata.funding.award.acronym.text",
    "metadata.funding.award.number",
    "metadata.funding.funder.name",
    "metadata.locations.features.place",
    "metadata.locations.features.description",
    "metadata.publication_date",
    "metadata.publisher",
    "metadata.version",
    "metadata.dates.description",
    "metadata.references.reference",
    # Not in the upstream default fields.
    "pids.doi.identifier^10",
    "metadata.related_identifiers.identifier^3",
    "metadata.resource_type.title.en^3",
    "parent.communities.entries.metadata.title^2",
    "custom_fields.cern:experiments.title.en^2",
    "custom_fields.cern:accelerators.title.en^2",
    "custom_fields.cern:departments.title.en",
    "custom_fields.cern:programmes.title.en",
    "custom_fields.cern:beams.title.en",
    "custom_fields.cern:committees.title.en",
    "custom_fields.cern:projects",
    "custom_fields.cern:studies",
    "custom_fields.cern:facilities",
    "custom_fields.meeting:meeting.title",
    "custom_fields.journal:journal.title",
    "custom_fields.thesis:thesis.university",
]
"""Fields searched by a query without field qualifiers, with boosts.

Never use wildcards here: ``*`` would also search fields restricted to
administrators, such as ``internal_notes``.
"""

FUZZY_FIELDS = [
    "metadata.title",
    "metadata.creators.person_or_org.name",
    "metadata.subjects.subject",
    "metadata.description",
]
"""Fields searched by a fuzzy word without a field (``higgs~``).

OpenSearch expands a fuzzy word into up to 50 terms per field, and the whole
query may not have more than 1,024 clauses (``indices.query.bool.max_clause_count``).
Over all of ``DEFAULT_SEARCH_FIELDS`` two fuzzy words were already too many;
over these fields a search can hold several.
"""

WILDCARD_FIELDS = [
    *FUZZY_FIELDS,
    "metadata.additional_titles.title",
    "custom_fields.cern:experiments.title.en",
    "custom_fields.cern:accelerators.title.en",
    "id",
    "metadata.identifiers.identifier",
    "pids.doi.identifier",
]
"""Fields searched by a wildcard word without a field (``detect*``).

Includes the identifiers, for report number prefixes such as ``CERN-THESIS-*``.
"""

# A word the analyzer splits into several tokens: report numbers, DOIs,
# hyphenated words. Not words with wildcards or escapes.
_COMPOUND_WORD_RE = re.compile(r"^[^*?\\\"]+[-/][^*?\\\"]+$")


def _expanded(node, fields):
    """``node`` searched in each of ``fields``: ``(f1:node OR f2:node ...)``.

    A ``:`` in a field name (``custom_fields.cern:experiments``) is escaped, or
    the query string does not parse.
    """
    return Group(
        OrOperation(
            *(SearchField(f.replace(":", "\\:"), deepcopy(node)) for f in fields)
        )
    )


def _is_wildcard(word):
    return ("*" in word or "?" in word) and word.strip("*?") != ""


# UPSTREAM(invenio-records-resources): the field limit for fuzzy and wildcard
# words is not CDS-specific. ``QueryParser.factory`` could take an
# ``expansion_fields`` option. Move it there once a CDS image with this
# transformer has been tested in dev.
class CDSSearchFieldTransformer(SearchFieldTransformer):
    """Field aliases (upstream), plus: every word must match.

    OpenSearch's ``default_operator: AND`` is applied per field when the fields
    use different analyzers: ``thesis Mobius`` would need both words in the
    same field and match nothing. An explicit ``AND`` lets each word match in
    any field.

    Fuzzy and wildcard words without a field search ``FUZZY_FIELDS`` and
    ``WILDCARD_FIELDS`` only.
    """

    def visit_search_field(self, node, context):
        """A word in a field search (``title:detect*``) keeps that field."""
        context["in_search_field"] = True
        yield from super().visit_search_field(node, context)

    def visit_fuzzy(self, node, context):
        """Search ``higgs~`` in a few fields only.

        Without a typed distance, luqum's tree copies write ``higgs~0.5``, which
        OpenSearch reads as 2 edits. Write AUTO's distance instead: 0 edits up to
        2 characters, 1 up to 5, 2 above.
        """
        new_node = node.clone_item()
        new_node.children = list(self.clone_children(node, new_node, context))
        term = new_node.term
        if node._implicit_degree and isinstance(term, Word):
            length = len(term.value)
            new_node.degree = Decimal(0 if length <= 2 else 1 if length <= 5 else 2)
        yield (
            new_node
            if context.get("in_search_field")
            else _expanded(new_node, FUZZY_FIELDS)
        )

    def visit_unknown_operation(self, node, context):
        """Turn ``a b`` into ``a AND b``, unless ``+``/``-`` say what is required."""
        children = list(self.clone_children(node, node.clone_item(), context))
        if any(isinstance(child, (Plus, Prohibit)) for child in children):
            new_node = node.clone_item()
            new_node.children = children
            yield new_node
            return
        # "a b OR c" is "a (b OR c)": keep the group, OpenSearch would read
        # "a AND b OR c" as "+a +b c".
        yield AndOperation(
            *(Group(c) if isinstance(c, OrOperation) else c for c in children)
        )

    def visit_word(self, node, context):
        """Search a word made of several tokens as a phrase.

        ``CERN-THESIS-2021-078`` must match as a whole, not any of its parts,
        and the ``/`` of a DOI would otherwise start a regular expression.
        """
        for new_node in super().visit_word(node, context):
            if isinstance(new_node, Word) and _COMPOUND_WORD_RE.match(new_node.value):
                new_node = Phrase(f'"{new_node.value}"')
            elif (
                isinstance(new_node, Word)
                and _is_wildcard(new_node.value)
                and not context.get("in_search_field")
            ):
                new_node = _expanded(new_node, WILDCARD_FIELDS)
            yield new_node
