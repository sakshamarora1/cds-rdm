# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 CERN.
#
# CDS-RDM is free software; you can redistribute it and/or modify it under
# the terms of the GPL-2.0 License; see LICENSE file for more details.

"""Turn a free-text search into a structured query with an LLM.

The LLM only fills in :data:`INTERPRETATION_SCHEMA`. :func:`render` builds the
query from it, keeping only values backed by the words typed and by the
repository, so the LLM never writes query syntax.

No Flask dependencies here; ``views.py`` wires it to the app.
"""

import json
import os
import re
import unicodedata

import requests

INTERPRETATION_SCHEMA = {
    "type": "object",
    "properties": {
        "resource_type": {"type": ["string", "null"]},
        "creators": {"type": "array", "items": {"type": "string"}},
        "community": {"type": ["string", "null"]},
        "records_of": {"type": ["string", "null"]},
        "identifiers": {"type": "array", "items": {"type": "string"}},
        "also_written_as": {"type": "array", "items": {"type": "string"}},
        "year_from": {"type": ["integer", "null"]},
        "year_to": {"type": ["integer", "null"]},
    },
    "required": [
        "resource_type",
        "creators",
        "community",
        "records_of",
        "identifiers",
        "also_written_as",
        "year_from",
        "year_to",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You read a search typed into the CERN Document Server (CDS) search bar and say \
what it refers to, as a JSON object. The search is then run on CDS records.

About CDS records:
- Every record has a resource type (the kind of document, from the list below), a \
title, creators (the authors, cited as authors, stored as "Family, Given"), \
optionally contributors (other people involved, not cited as authors), a \
publication date, a description and identifiers.
- Identifiers include the DOI (format 10.####/..., CDS registers DOIs under \
10.17181), report numbers (also called RN or reference numbers, written in each \
collaboration's own format, listed per resource type below, not always unique), \
CDS record numbers (digits only), arXiv ids (e.g. ####.#####) and INSPIRE ids \
(digits only).
- Every record belongs to one or more communities. A community is a collection \
curated by a group, usually an experiment, a department, a service or a \
project at CERN.

Fields to fill in:
- resource_type: the id of the kind of document the search asks for, or null.
- creators: people or organisations the search gives as authors.
- community: the community the search asks for, written as in the list below, \
or null.
- records_of: the name of who or what the records belong to, when the search \
asks for the records of a group (rule 9), written as in the search, or null.
- identifiers: identifiers written in the search, copied exactly.
- also_written_as: up to 4 other ways the record is likely written in its \
title or report number (see the rules), or empty.
- year_from / year_to: the years the search asks for, or null.

Rules:
1. Fill a field only when the search clearly says it. When unsure, leave it \
empty: every word of the search is searched anyway, so an empty field never \
loses results, but a wrong one hides them.
2. Kind of document: set resource_type when a word or phrase names a kind of \
document, in singular or plural, in any language, or as a common synonym \
(for example a dissertation or a PhD, master's or bachelor's thesis is a \
thesis; slides or a talk are a presentation; a paper in a journal is a \
journal article; a paper not yet peer reviewed is a preprint). Pick the closest \
resource type in the list. A word that is only the topic of the search (what \
the documents are about) is not a kind of document.
3. Creators: only names given as authors: "by NAME", "NAME's thesis", or a \
person's name standing next to a kind of document or a topic. Leave out titles \
such as Dr or Prof. Names of experiments, accelerators, detectors, \
collaborations or places are topics, not creators, unless the search says they \
wrote the document ("by the NAME collaboration").
4. Community: only when the search names a community from the list below, as \
"NAME community", "in NAME", or when the whole search is a community name. \
Otherwise null, even when a word also appears in a community name.
5. Identifiers: anything shaped like a DOI, a report number, an arXiv or \
INSPIRE id. Copy them exactly as typed.
6. Numbered documents: when the search refers to one document of a numbered \
series - a name or an abbreviation followed by a number, such as "NAME 6", \
"ABBR6", "NAME no. 6", "NAME nr 6", "NAME n°6" - fill also_written_as with:
   a. the full name of the series with "No." and the number (for example \
"NAME No. 6"); an abbreviation is usually one of the abbreviations of resource \
type names listed below, and then resource_type is that type;
   b. the report number built from a format listed for that resource type: \
write the number in place of the # placeholders, with exactly as many \
digits as there are # (add leading zeros: ## with 6 gives 06, ### with 6 \
gives 006). Never leave a # in the result.
7. Other short forms: when the search uses an abbreviation for the name of a \
kind of document or series, also give the full form in also_written_as. Do not \
expand or translate topic words.
8. Years: "from YEAR" or "since YEAR" sets year_from; "until YEAR" or "before \
YEAR" sets year_to; "YEAR-YEAR" sets both; a single year asked for sets both to \
that year. Never take a year from inside an identifier or a report number.
9. Records of a group: when the search asks for the records of something - \
"NAME records", "NAME documents", "NAME papers", "NAME publications", "NAME \
material", "records from NAME", "NAME's records" - and NAME is who or what the \
records belong to (an experiment, accelerator, department, community, \
collaboration, committee, programme, beam line or report series), set \
records_of to NAME as typed. Do not also put NAME in creators or community. \
When the word after NAME is also a kind of document ("NAME theses"), set \
resource_type as well.
{context}
Resource types (id: name):
{resource_types}
{report_formats}{communities}"""


def quote(value):
    """Return ``value`` as a query-string phrase."""
    return '"{}"'.format(value.replace("\\", "\\\\").replace('"', '\\"'))


class InterpreterError(Exception):
    """The LLM could not be reached or returned an unusable answer."""


def report_number_formats(report_numbers, per_type=3, min_share=0.1):
    """Most common report number formats per resource type.

    :param report_numbers: ``[(resource_type_id, report_number), ...]`` sampled
        from the records.
    :returns: ``{resource_type_id: [format, ...]}``, digits written as ``#`` and
        suffixes after the last number dropped (revisions, drafts):
        ``CERN-OPER-CIRC-07-REV2`` becomes ``CERN-OPER-CIRC-##``. A report number
        phrase matches the longer stored one, so the suffix is not needed.
    """
    counts = {}
    for resource_type, number in report_numbers:
        if not number or not re.search(r"[A-Za-z]", number):
            continue
        segments = re.sub(r"\d", "#", number.strip()).split("-")
        numbered = [i for i, segment in enumerate(segments) if set(segment) == {"#"}]
        if numbered:
            segments = segments[: numbered[-1] + 1]
        pattern = "-".join(segments)
        type_counts = counts.setdefault(resource_type, {})
        type_counts[pattern] = type_counts.get(pattern, 0) + 1
    formats = {}
    for resource_type, type_counts in counts.items():
        total = sum(type_counts.values())
        ranked = sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))
        kept = [p for p, n in ranked[:per_type] if n / total >= min_share]
        if kept:
            formats[resource_type] = kept
    return formats


def abbreviations(resource_types):
    """``{"OC": ["Operational circular"], ...}``: initials of multi-word names."""
    found = {}
    for _, title in resource_types:
        words = re.findall(r"[A-Za-z]+", title)
        if len(words) >= 2:
            found.setdefault("".join(w[0] for w in words).upper(), []).append(title)
    return found


def _single_number_formats(formats):
    """Formats with one number only (``CERN-OPER-CIRC-##``, not ``X-####-###``)."""
    return [f for f in formats or [] if len(re.findall(r"#+", f)) == 1]


def _numbered_series_example(resource_types, short, report_formats, number=3):
    """A worked example of rule 6 built from the repository's own series.

    Uses the first abbreviation naming a single resource type whose report
    numbers carry one number, so the example changes with the vocabulary and the
    data instead of being written in the prompt.
    """
    by_title = {title: rid for rid, title in resource_types}
    for abbr, titles in sorted(short.items()):
        if len(titles) != 1:
            continue
        rid = by_title[titles[0]]
        formats = _single_number_formats((report_formats or {}).get(rid))
        if not formats:
            continue
        report_number = re.sub(
            r"#+", lambda m: str(number).zfill(len(m.group(0))), formats[0]
        )
        answer = json.dumps(
            {
                "resource_type": rid,
                "also_written_as": [f"{titles[0]} No. {number}", report_number],
            }
        )
        return f'"{abbr} {number}" or "{abbr}{number}" gives {answer}'
    return None


def build_messages(
    query,
    resource_types,
    context=None,
    report_formats=None,
    communities=None,
):
    """Chat messages for ``query``.

    :param resource_types: ``[(id, title), ...]``.
    :param context: extra rules (``CDS_SEARCH_INTERPRETER_CONTEXT``).
    :param report_formats: ``{resource_type_id: [format, ...]}``, see
        :func:`report_number_formats`.
    :param communities: community names the search may refer to.
    """
    titles = dict(resource_types)
    types = "\n".join(f"- {rid}: {title}" for rid, title in resource_types)
    formats = ""
    if report_formats:
        lines = [
            f"- {titles.get(rid, rid)}: {', '.join(patterns)}"
            for rid, patterns in sorted(report_formats.items())
        ]
        formats = "\nReport number formats (# is a digit):\n" + "\n".join(lines) + "\n"
    short = abbreviations(resource_types)
    if short:
        by_title = {title: rid for rid, title in resource_types}
        lines = []
        for abbr, abbr_titles in sorted(short.items()):
            described = []
            for title in abbr_titles:
                single = _single_number_formats(
                    (report_formats or {}).get(by_title[title])
                )
                described.append(
                    f"{title} (report numbers {', '.join(single)})" if single else title
                )
            lines.append(f"- {abbr}: {' or '.join(described)}")
        types += "\n\nAbbreviations of resource type names:\n" + "\n".join(lines)
        example = _numbered_series_example(resource_types, short, report_formats)
        if example:
            types += f"\n\nFor example, following rule 6: {example}"
    names = ""
    if communities:
        names = "\nCommunities:\n" + "\n".join(f"- {name}" for name in communities)
    context = f"\n{context.strip()}\n" if context and context.strip() else ""
    system = SYSTEM_PROMPT.format(
        context=context,
        resource_types=types,
        report_formats=formats,
        communities=names,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]


def call_ollama(messages, url, model, timeout):
    """Ask a local Ollama model for an interpretation; return the parsed JSON."""
    try:
        response = requests.post(
            f"{url.rstrip('/')}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "format": INTERPRETATION_SCHEMA,
                "stream": False,
                "think": False,
                # The prompt lists the resource types, report number formats and
                # communities: more than Ollama's default context window.
                "options": {"temperature": 0, "num_ctx": 8192},
                # Loading the model takes longer than a search should wait: keep
                # it in memory between searches.
                "keep_alive": "30m",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = json.loads(response.json()["message"]["content"])
    except (requests.RequestException, KeyError, ValueError) as e:
        raise InterpreterError(str(e)) from e
    if not isinstance(data, dict):
        raise InterpreterError("Answer is not a JSON object.")
    return data


_PLAIN_QUERY_RE = re.compile(r"^[^:\"()\[\]{}*?~^\\/+!]*$")
_OPERATOR_RE = re.compile(r"\b(AND|OR|NOT|TO)\b")
_SHORT_CODE_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9.]{2,10}$")


def should_interpret(query, max_length=200):
    """Only interpret plain text: no field searches, operators or special syntax.

    Single words are only interpreted when they look like a short code mixing
    letters and digits (``oc6``); other single words are searched as typed.
    """
    query = (query or "").strip()
    if not query or len(query) > max_length:
        return False
    if not _PLAIN_QUERY_RE.match(query) or _OPERATOR_RE.search(query):
        return False
    return len(query.split()) >= 2 or bool(_SHORT_CODE_RE.match(query))


# Words that link the parts of a search and carry no meaning once those parts are
# turned into filters ("theses BY Mobius FROM 2020").
CONNECTING_WORDS = {
    "a",
    "about",
    "after",
    "an",
    "and",
    "at",
    "before",
    "between",
    "by",
    "community",
    "during",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "since",
    "the",
    "to",
    "until",
    "with",
}


def _tokens(text):
    """Lowercase, accent-free word tokens."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return re.findall(r"[a-z0-9]+", text)


def _same_word(a, b):
    """Equal, or the same word with a different ending (thesis/theses)."""
    if a == b:
        return True
    if a.isdigit() or b.isdigit() or min(len(a), len(b)) < 4:
        return False
    prefix = len(os.path.commonprefix([a, b]))
    return prefix >= max(4, min(len(a), len(b)) - 2)


class _Query:
    """The search as typed, and which of its words have been explained."""

    def __init__(self, text):
        self.words = text.split()
        self.word_tokens = [_tokens(word) for word in self.words]
        self.used = set()  # indexes of explained words

    def find(self, text, fuzzy=False):
        """Indexes of unexplained words matching every token of ``text``, or ``None``."""
        found = set()
        for token in _tokens(text):
            matches = [
                i
                for i, tokens in enumerate(self.word_tokens)
                if i not in self.used
                and any(_same_word(token, t) if fuzzy else token == t for t in tokens)
            ]
            if not matches:
                return None
            found.update(matches)
        return found or None

    def find_phrase(self, text):
        """Indexes of consecutive words spelling ``text`` (spaces ignored)."""
        target = "".join(_tokens(text))
        for start in range(len(self.words)):
            joined = ""
            for end in range(start, len(self.words)):
                joined += "".join(self.word_tokens[end])
                if joined == target:
                    return set(range(start, end + 1))
                if not target.startswith(joined):
                    break
        return None

    def remaining_words(self):
        """Words not explained by a filter, without connecting words."""
        return [
            word
            for i, word in enumerate(self.words)
            if i not in self.used
            and not all(t in CONNECTING_WORDS for t in self.word_tokens[i])
        ]


def _strings(values, max_items=5):
    if not isinstance(values, list):
        return []
    cleaned = []
    for value in values:
        if isinstance(value, str) and value.strip() and value.strip() not in cleaned:
            cleaned.append(value.strip())
    return cleaned[:max_items]


RECORD_WORDS = {
    "documents",
    "items",
    "material",
    "materials",
    "publications",
    "records",
}
"""Words that only say "records" in "NAME records": dropped once NAME is resolved."""

TYPE_BOOST = 5
"""Boost of records of the resource type the search names."""

_YEAR_RE = re.compile(r"(?<![\w-])(1[5-9]\d\d|2\d\d\d)(?![\w-])")


def _significant(text):
    return [t for t in _tokens(text) if t not in CONNECTING_WORDS]


def _community_names(title):
    """Ways a search names a community: its bracketed acronyms, then its title.

    "Human Resources (HR)" gives ``["hr"]`` and ``["human", "resources"]``.
    """
    names = [_tokens(acronym) for acronym in re.findall(r"\(([^)]+)\)", title)]
    names.append(_significant(re.sub(r"\([^)]*\)", " ", title)))
    return [name for name in names if name]


def community_matches(name, title):
    """Whether ``name`` is the community ``title`` or its bracketed acronym."""
    target = [t for t in _significant(name) if t != "community"]
    return bool(target) and target in _community_names(title)


def match_vocabulary(name, entries, kind=None, common_share=0.3):
    """Vocabulary entries ``name`` refers to, e.g. "LHC" for "CERN LHC".

    :param entries: ``[(id, title, [alias, ...]), ...]`` of one vocabulary.
    :param kind: what the vocabulary holds ("Department"): the word is dropped
        from ``name``, so "EP department" is "EP".
    :returns: the matching ``(id, title)``, best first.

    ``name`` must equal the id, the title, an alias, or one of them without the
    words most entries share ("CERN" in "CERN LHC", "Department" in "EP -
    Experimental Physics Department"). A part of a longer name does not match:
    "LHC" is not "CERN HL-LHC".
    """
    kind_tokens = _tokens(kind or "")
    target = [
        t for t in _tokens(name) if not any(_same_word(t, k) for k in kind_tokens)
    ]
    if not target or not entries:
        return []
    counts = {}
    for _, title, _ in entries:
        for token in set(_tokens(title)):
            counts[token] = counts.get(token, 0) + 1
    common = (
        {t for t, n in counts.items() if n / len(entries) >= common_share}
        if len(entries) >= 10
        else set()
    )
    short_target = [t for t in target if t not in common] or target
    matches = []
    for entry_id, title, aliases in entries:
        candidates = [entry_id, title, *(aliases or []), *title.split(" - ")]
        for candidate in candidates:
            tokens = _tokens(candidate)
            if tokens == target or [t for t in tokens if t not in common] in (
                target,
                short_target,
            ):
                matches.append((entry_id, title))
                break
    return matches


def report_number_prefixes(report_numbers, min_count=2):
    """Report number prefixes seen at least ``min_count`` times.

    ``CERN-ALEPH-PHYSIC-99-012`` gives ``CERN-ALEPH-PHYSIC``: the segments before
    the first number.
    """
    counts = {}
    for _, number in report_numbers:
        prefix = []
        for segment in (number or "").strip().split("-"):
            if re.search(r"\d", segment):
                break
            prefix.append(segment)
        if prefix:
            key = "-".join(prefix)
            counts[key] = counts.get(key, 0) + 1
    return sorted(p for p, n in counts.items() if n >= min_count)


def match_report_prefixes(name, prefixes, max_matches=3, common_share=0.3):
    """Report number starts naming ``name``: "ALEPH" gives ``CERN-ALEPH``."""
    target = _tokens(name)
    if not target or not prefixes:
        return []
    firsts = {}
    for prefix in prefixes:
        first = prefix.split("-")[0].lower()
        firsts[first] = firsts.get(first, 0) + 1
    common = {f for f, n in firsts.items() if n / len(prefixes) >= common_share}
    found = []
    for prefix in prefixes:
        segments = prefix.split("-")
        for i, segment in enumerate(segments):
            if _tokens(segment) == target and not (
                i == 0 and segment.lower() in common
            ):
                start = "-".join(segments[: i + 1])
                if start not in found:
                    found.append(start)
                break
    return found[:max_matches]


def _community_words(q, title):
    """Words of the search naming the community ``title``, or ``None``.

    The acronym or every significant word of the title must be in the search:
    "CERN" alone does not name "CERN Research".
    """
    for name in _community_names(title):
        found = q.find(" ".join(name))
        if found:
            return found
    return None


def _is_identifier(value):
    """One token with a digit: report numbers, DOIs, arXiv ids, record numbers."""
    return bool(re.fullmatch(r"\S*\d\S*", value))


def _only_typed_words(q, value):
    """True when ``value`` only reuses words of the search (in any order)."""
    typed = [t for tokens in q.word_tokens for t in tokens]
    return all(any(_same_word(t, w) for w in typed) for t in _tokens(value))


def _initials(title):
    return "".join(w[0] for w in re.findall(r"[A-Za-z]+", title)).lower()


def _names_series(q, title):
    """Whether the search names the series ``title``: its words or its initials.

    "operational circular 6", "OC 6" and "oc6" name "Operational circular".
    """
    if q.find(title, fuzzy=True):
        return True
    initials = _initials(title)
    return len(initials) >= 2 and any(
        re.fullmatch(rf"{initials}\d*", t) for tokens in q.word_tokens for t in tokens
    )


def _series_of(alternative, resource_types, report_formats):
    """Title of the series an alternative names ("X No. 6", a report number)."""
    lowered = alternative.lower()
    for title in resource_types.values():
        if lowered.startswith(title.lower()):
            return title
    for rid, formats in (report_formats or {}).items():
        for fmt in formats:
            prefix = fmt.split("#")[0]
            if len(prefix) >= 3 and lowered.startswith(prefix.lower()):
                return resource_types.get(rid)
    return None


def render(
    query,
    data,
    resource_types,
    find_community,
    has_creator=None,
    report_formats=None,
    find_facets=None,
):
    """Build the search query from the LLM's answer.

    Every value must be backed by the search, and by the repository where it can
    be checked, otherwise it is dropped: small models invent authors,
    communities and years. Creators, community and years become required; the
    words they explain are removed from the keywords, all other words are kept
    as typed. The resource type only boosts.

    :param query: the search as typed.
    :param data: the LLM's JSON answer.
    :param resource_types: ``{id: title}`` of the valid resource types.
    :param find_community: ``callable(name) -> [(slug, title), ...]``, the
        communities the person can see that best match ``name``.
    :param has_creator: optional ``callable(name) -> bool``, whether any record
        has this creator; topic words taken for names are dropped.
    :param report_formats: ``{resource_type_id: [format, ...]}``; report
        numbers of a series the search does not name are dropped.
    :param find_facets: optional ``callable(name) -> [(label, clause, value)]``,
        what ``name`` can be in the repository (community, experiment,
        accelerator, report numbers, ...), for "NAME records" searches.
    :returns: ``{"query": str, "parts": [{"label", "value"}]}``, or ``None`` when
        the interpretation adds nothing to the search as typed.
    """
    q = _Query(query)
    parts = []  # what was understood, shown to the user
    required = []  # constraints the search states: creators, community, years
    boosts = []  # ranking hints that must not exclude records

    # Identifiers first, so their parts (CERN-THESIS-...) are not read as words.
    identifiers = []
    alternatives = _strings(data.get("also_written_as"), max_items=4)
    for identifier in _strings(data.get("identifiers")):
        found = (
            q.find_phrase(identifier)
            if _is_identifier(identifier) and not _YEAR_RE.fullmatch(identifier)
            else None
        )
        if found:
            q.used |= found
            identifiers.append(identifier)
        elif identifier not in alternatives:
            alternatives.append(identifier)

    # The resource type and the words naming it only boost: records are not
    # always typed consistently (some circulars are typed "Other"), and "theses"
    # would not match "Thesis" if it were required (there is no stemming).
    resource_type = data.get("resource_type")
    if isinstance(resource_type, str) and resource_type in resource_types:
        title = resource_types[resource_type]
        found = q.find(title, fuzzy=True) or q.find(
            resource_type.rsplit("-", 1)[-1], fuzzy=True
        )
        if found:
            q.used |= found
            boosts.append(f"resource_type.id:{quote(resource_type)}^{TYPE_BOOST}")
            words = [q.words[i] for i in sorted(found)]
            boosts.append(words[0] if len(words) == 1 else f"({' '.join(words)})")
            parts.append({"label": "Resource type", "value": title})

    # "NAME records": NAME may be a community, an experiment, a department, a
    # report number series... Any of what it resolves to may match. A "creator"
    # that is such a group ("ATLAS theses") is read the same way, unless the
    # search says "by".
    records_of = data.get("records_of")
    if find_facets and not (isinstance(records_of, str) and records_of.strip()):
        says_by = any("by" in tokens for tokens in q.word_tokens)
        for creator in [] if says_by else _strings(data.get("creators")):
            if q.find(creator) and find_facets(creator):
                records_of = creator
                data = {
                    **data,
                    "creators": [c for c in data.get("creators") if c != creator],
                }
                break
    if find_facets and isinstance(records_of, str) and _significant(records_of):
        # Only the typed part: the model may complete the name ("FASER" ->
        # "FASER Collaboration").
        typed = [t for t in _significant(records_of) if q.find(t)]
        found = q.find(" ".join(typed)) if typed else None
        facets = find_facets(" ".join(typed)) if found else []
        if facets:
            q.used |= found
            q.used |= {
                i
                for i, tokens in enumerate(q.word_tokens)
                if tokens and all(t in RECORD_WORDS for t in tokens)
            }
            clauses = list(dict.fromkeys(clause for _, clause, _ in facets))
            required.append(
                clauses[0] if len(clauses) == 1 else f"({' OR '.join(clauses)})"
            )
            parts.extend({"label": label, "value": value} for label, _, value in facets)

    for creator in _strings(data.get("creators")):
        found = q.find(creator)
        if not found or found <= q.used:
            continue
        if has_creator is not None and not has_creator(creator):
            continue
        q.used |= found
        # Word by word: names are stored as "Family, Given".
        words = " ".join(f"+{w}" for w in _tokens(creator))
        required.append(f"creator:({words})")
        parts.append({"label": "Creator", "value": creator})

    community = data.get("community")
    if isinstance(community, str) and _tokens(community):
        for slug, title in find_community(" ".join(_tokens(community))):
            found = _community_words(q, title)
            if found:
                q.used |= found | (q.find("community") or set())
                required.append(f"community.id:{quote(slug)}")
                parts.append({"label": "Community", "value": title})
                break

    years = {int(y) for y in _YEAR_RE.findall(query)}
    year_from, year_to = data.get("year_from"), data.get("year_to")
    year_from = year_from if year_from in years else None
    year_to = year_to if year_to in years else None
    if year_from or year_to:
        start, end = year_from or "*", year_to or "*"
        required.append(f"publication_date:[{start} TO {end}]")
        q.used |= {i for i, w in enumerate(q.words) if _YEAR_RE.search(w)}
        if start == end:
            label = str(start)
        elif end == "*":
            label = f"from {start}"
        elif start == "*":
            label = f"until {end}"
        else:
            label = f"{start}–{end}"
        parts.append({"label": "Years", "value": label})

    # Ways to match the item itself, any of which may match: the words left, the
    # identifiers found in the search, and other ways of writing it. When the
    # search contains an identifier, it decides and the other words only rank.
    matches = []
    remaining = q.remaining_words()
    if remaining:
        words = remaining[0] if len(remaining) == 1 else f"({' '.join(remaining)})"
        (boosts if identifiers else matches).append(words)
    for identifier in identifiers:
        matches.append(f"identifier:{quote(identifier)}")
        parts.append({"label": "Identifier", "value": identifier})

    # Drop alternatives that add nothing (only words already typed) or that are
    # an unfilled report number format ("CERN-OPER-CIRC-##").
    # Also drop series names and report numbers of a series the search does not
    # name ("AC2" must not give an operational circular).
    def keep(alternative):
        if _only_typed_words(q, alternative) or "#" in alternative:
            return False
        series = _series_of(alternative, resource_types, report_formats)
        return series is None or _names_series(q, series)

    alternatives = [a for a in alternatives if keep(a)]
    if matches:
        matches.extend(quote(a) for a in alternatives)
    else:
        # Nothing else to match the item: other ways of writing it only rank.
        boosts.extend(quote(a) for a in alternatives)
    if alternatives:
        parts.append({"label": "Also written as", "value": ", ".join(alternatives)})

    if not parts:
        return None  # nothing understood beyond the words themselves
    if matches:
        required.insert(
            0, matches[0] if len(matches) == 1 else f"({' OR '.join(matches)})"
        )
    if not required:
        return None  # only a resource type: nothing left that must match
    return {
        "query": " ".join([*(f"+{clause}" for clause in required), *boosts]),
        "parts": parts,
    }
