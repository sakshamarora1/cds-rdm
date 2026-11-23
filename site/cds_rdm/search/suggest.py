# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 CERN.
#
# CDS-RDM is free software; you can redistribute it and/or modify it under
# the terms of the GPL-2.0 License; see LICENSE file for more details.

"""Search suggestions without a model: spelling fixes and known names.

Two sources, both plain OpenSearch queries:

- **Spelling:** the phrase suggester on the titles, descriptions and subjects
  of the records index. A correction is offered only if it matches a public,
  published record, so words seen only in restricted records are never shown.
- **Known names:** a small index of phrases from the vocabularies and public
  communities ("ATLAS", "CERN LHC", "Thesis"), matched with typos allowed. A
  match offers the search for that experiment, accelerator, community, etc.

Only the phrase index is new; the records index is not changed.
"""

import re
from datetime import datetime, timezone

from .interpreter import quote

_SYNTAX_RE = re.compile(r"[:\"()\[\]{}*?~^\\/+!]|\b(AND|OR|NOT|TO)\b")


def is_plain_text(query, max_length=200):
    """Suggest only for plain words: not field searches, operators or syntax."""
    return bool(query) and len(query) <= max_length and not _SYNTAX_RE.search(query)


PUBLIC_PUBLISHED = [
    {"term": {"access.record": "public"}},
    {"term": {"deletion_status": "P"}},
]

SPELLING_FIELDS = [
    "metadata.title",
    "metadata.description",
    "metadata.subjects.subject",
]
"""Where corrections come from: the words people remember from a record."""

SPELLING_LANGUAGE_FIELD = "metadata.description"
"""Field whose word frequencies rank the corrections: the most running English."""


def spelling_body(text, fields=SPELLING_FIELDS, language_field=SPELLING_LANGUAGE_FIELD):
    """Phrase suggester request for ``text`` over the records index.

    Candidate words come from all ``fields``; a correction is kept only if all
    its words occur in one public, published record, across those fields.
    ``collate`` matters: the dictionary also holds restricted records' words.
    """
    return {
        "size": 0,
        "suggest": {
            "text": text,
            "spelling": {
                "phrase": {
                    "field": language_field,
                    "size": 1,
                    "max_errors": 2,
                    "direct_generator": [
                        {"field": field, "suggest_mode": "always", "min_word_length": 3}
                        for field in fields
                    ],
                    "collate": {
                        "query": {
                            "source": {
                                "bool": {
                                    "must": {
                                        "multi_match": {
                                            "query": "{{suggestion}}",
                                            "fields": fields,
                                            "type": "cross_fields",
                                            "operator": "and",
                                        }
                                    },
                                    "filter": PUBLIC_PUBLISHED,
                                }
                            }
                        },
                        "prune": False,
                    },
                }
            },
        },
    }


def spelling_from_response(text, response):
    """The corrected text, or ``None`` if there is none or it is the same."""
    for entry in response.get("suggest", {}).get("spelling", []):
        for option in entry.get("options", []):
            suggestion = option.get("text", "")
            if suggestion and suggestion.lower() != text.lower():
                return suggestion
    return None


# Known names: the phrase index
# =============================

PHRASES_INDEX_BODY = {
    "settings": {
        "number_of_shards": 1,
        "analysis": {
            "filter": {
                "phrase_edge_ngram": {
                    "type": "edge_ngram",
                    "min_gram": 2,
                    "max_gram": 15,
                }
            },
            "analyzer": {
                "phrase_text": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"],
                },
                "phrase_prefix": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding", "phrase_edge_ngram"],
                },
            },
            "normalizer": {
                "phrase_exact": {
                    "type": "custom",
                    "filter": ["lowercase", "asciifolding"],
                }
            },
        },
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "phrase": {
                "type": "text",
                "analyzer": "phrase_text",
                "fields": {
                    "prefix": {
                        "type": "text",
                        "analyzer": "phrase_prefix",
                        "search_analyzer": "phrase_text",
                    },
                    "exact": {"type": "keyword", "normalizer": "phrase_exact"},
                },
            },
            "label": {"type": "keyword"},
            "value": {"type": "keyword"},
            "query": {"type": "keyword", "index": False},
        },
    },
}
"""One document per name of a value: its id, title and aliases."""


def common_words(titles, share=0.3, min_entries=10):
    """Words most titles of a vocabulary share: "CERN" in "CERN LHC"."""
    if len(titles) < min_entries:
        return set()
    counts = {}
    for title in titles:
        for word in set(title.split()):
            counts[word] = counts.get(word, 0) + 1
    return {word for word, n in counts.items() if n / len(titles) >= share}


def vocabulary_phrases(label, shortcut, use, entries):
    """Phrase documents of one vocabulary.

    :param entries: ``[(id, title, [alias, ...]), ...]``.
    :param use: search the vocabulary value by its ``"id"`` or ``"title"``.

    A title is also indexed without the words most titles share, so "LHC"
    finds "CERN LHC".
    """
    common = common_words([title for _, title, _ in entries])
    for entry_id, title, aliases in entries:
        value = entry_id if use == "id" else title
        query = f"{shortcut}:{quote(value)}"
        short = " ".join(w for w in title.split() if w not in common)
        names = {entry_id, title, short, *aliases, *title.split(" - ")}
        for name in sorted(n.strip() for n in names if n and n.strip()):
            yield {"phrase": name, "label": label, "value": title, "query": query}


def community_phrases(communities):
    """Phrase documents of communities: ``[(slug, title), ...]``."""
    for slug, title in communities:
        query = f"community.id:{quote(slug)}"
        for name in {slug, title}:
            yield {"phrase": name, "label": "Community", "value": title, "query": query}


def versioned_index_name(alias, now=None):
    """``<alias>-<timestamp>``: the build writes a new index, then moves the alias."""
    now = now or datetime.now(timezone.utc)
    return f"{alias}-{now.strftime('%Y%m%d%H%M%S')}"


def is_own_index(alias, name):
    """Whether ``name`` is an index built for ``alias`` (and may be deleted)."""
    suffix = name[len(alias) + 1 :]
    return name.startswith(f"{alias}-") and len(suffix) == 14 and suffix.isdigit()


def phrases_body(text, size=10):
    """Names in the phrase index that ``text`` or a part of it may refer to."""
    return {
        "size": size,
        "_source": ["phrase", "label", "value", "query"],
        "query": {
            "bool": {
                "should": [
                    {"term": {"phrase.exact": {"value": text, "boost": 10}}},
                    {
                        "match": {
                            "phrase": {
                                "query": text,
                                "fuzziness": "AUTO",
                                "prefix_length": 1,
                            }
                        }
                    },
                    {"match": {"phrase.prefix": {"query": text, "boost": 0.5}}},
                ],
                "minimum_should_match": 1,
            }
        },
    }


def _tokens(text):
    return [t for t in "".join(c if c.isalnum() else " " for c in text.lower()).split()]


def _edits(a, b):
    """Damerau-Levenshtein distance (adjacent swaps count as one edit)."""
    d = [
        [i + j if i * j == 0 else 0 for j in range(len(b) + 1)]
        for i in range(len(a) + 1)
    ]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def _auto_fuzziness(word):
    """OpenSearch's ``AUTO``: edits allowed for a word of this length."""
    return 0 if len(word) <= 2 else 1 if len(word) <= 5 else 2


def _close(typed, known):
    return typed[:1] == known[:1] and _edits(typed, known) <= _auto_fuzziness(known)


def match_phrase(text, phrase):
    """The typed words ``phrase`` stands for, or ``None``.

    Every word of the phrase must be typed, in order, with typos allowed.
    Returns ``(start, end)``: the slice of the typed words it covers.
    """
    typed, known = _tokens(text), _tokens(phrase)
    if not known or len(known) > len(typed):
        return None
    for start in range(len(typed) - len(known) + 1):
        if all(_close(typed[start + i], word) for i, word in enumerate(known)):
            return start, start + len(known)
    return None


def known_names(text, hits, max_names=3):
    """Searches for the known names found in ``text``.

    :param hits: phrase index hits, best first.
    :returns: ``[{"label", "value", "query"}, ...]``. ``query`` is the search for
        that name plus the other words typed: "atlsa higgs" becomes
        ``experiment.id:"ATLAS" higgs``.
    """
    typed = _tokens(text)
    found, seen = [], set()
    for hit in hits:
        doc = hit["_source"]
        span = match_phrase(text, doc["phrase"])
        if span is None or doc["query"] in seen:
            continue
        seen.add(doc["query"])
        rest = " ".join(typed[: span[0]] + typed[span[1] :])
        found.append(
            {
                "label": doc["label"],
                "value": doc["value"],
                "query": f"{doc['query']} {rest}".strip(),
            }
        )
        if len(found) == max_names:
            break
    return found
