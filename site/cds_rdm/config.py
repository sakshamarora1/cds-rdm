# -*- coding: utf-8 -*-
#
# Copyright (C) 2025-2026 CERN.
#
# CDS-RDM is free software; you can redistribute it and/or modify it
# under the terms of the GPL-2.0 License; see LICENSE file for more details.

"""Configuration."""

from invenio_i18n import lazy_gettext as _
from invenio_records_resources.services.records.facets import TermsFacet

# Not working yet
CLC_SYNC_FACETS = {
    "status": {
        "facet": TermsFacet(
            field="status",
            title=_("Status"),
        ),
        "ui": {
            "field": "status",
        },
    },
}
"""Facets/aggregations for CLC user sync results."""

CLC_SYNC_DEFAULT_QUEUE = None
"""Default Celery queue."""

CLC_SYNC_SORT_OPTIONS = {
    "created": dict(
        title=_("Created"),
        fields=["created"],
    ),
    "status": dict(
        title=_("Status"),
        fields=["status"],
    ),
}
"""Definitions of available CLC user sync sort options. """

CLC_SYNC_SEARCH = {
    "facets": ["status"],
    "sort": ["created", "status"],
}
"""CLC user sync search configuration."""

CLC_SYNC_ALLOWED_RESOURCE_TYPES = [
    "publication",
]
"""Allowed resource types for CLC user sync."""

CLC_URL_SYNC = "CHANGE_ME"
"""URL for the CLC endpoint."""

CDS_ILS_IMPORTER_API_KEY = "CHANGE_ME"
"""API key for the CLC importer. This is a placeholder and should be replaced with a real key."""

CDS_COMMUNITIES_REQUIRING_NEW_RECORD_VERSION_REVIEW = []
"""
A list of the UUIDs of communities in which new versions of records will require a community
submission review. For communities not in this list, records will only need a review for their
first version (if the community settings require it).

This only applies to records where the specified community is their parent's default community.
"""

CDS_HARVESTER_USER_EMAIL = None
"""Email of the INSPIRE harvester service user."""

CDS_HARVESTER_SANDBOX_ALLOW_CREATE_PROD_MISSING_RECORDS = False
"""Sandbox only: create a record that already exists on prod but is missing here.

Needs ``CDS_ENVIRONMENT_NAME == "sandbox"``. When both are set we:
1. allow create even if the entry has a CDS DOI
2. ask prod for the parent and version ids from the INSPIRE CDSRDM value
3. create the sandbox record with those same ids
"""

CDS_HARVESTER_PROD_API_URL = "https://repository.cern"
"""Prod base URL used to resolve a CDSRDM id into parent.id and version id."""

CDS_HARVESTER_INSPIRE_API_URL = "https://inspirehep.net/api/literature"
"""Base URL for the INSPIRE literature search API used by the harvester reader."""

CDS_CERN_ARCHIVES_BASE_URL = "https://archives.cern.ch/index.php/"
"""Base URL for CERN Archives (AtoM) links.

Not the final URL — ``archive_link_render`` appends the archive id.
"""

# Search query interpreter
# ========================
CDS_SEARCH_INTERPRETER_ENABLED = False
"""Enable ``/api/search/interpret`` and the "Showing results for" banner.

Free-text searches are sent to the LLM below, which says what they refer to
(resource type, creators, community, years, identifiers). The search page shows
what was understood and offers to search for the words as typed instead.
Enable locally with ``INVENIO_CDS_SEARCH_INTERPRETER_ENABLED=True``.
"""

CDS_SEARCH_INTERPRETER_URL = "http://localhost:11434"
"""Base URL of the Ollama server running the model."""

CDS_SEARCH_INTERPRETER_MODEL = "qwen3:8b"
"""Ollama model used to interpret searches."""

CDS_SEARCH_INTERPRETER_TIMEOUT = 10
"""Seconds to wait for the model before searching for the words as typed."""

CDS_SEARCH_INTERPRETER_CONTEXT = None
"""Extra rules for the model, e.g. local naming conventions.

The resource types, report number formats and communities are read from the
repository and need not be listed here.
"""

CDS_SEARCH_INTERPRETER_CACHE_TTL = 24 * 60 * 60
"""Seconds an interpretation is cached, per search text and model."""

CDS_SEARCH_INTERPRETER_FACETS = [
    # (vocabulary type, label shown, search shortcut, value used: "id" or "title")
    ("experiments", "Experiment", "experiment.id", "id"),
    ("accelerators", "Accelerator", "accelerator.id", "id"),
    ("departments", "Department", "department.id", "id"),
    ("programmes", "Programme", "programme.id", "id"),
    ("committees", "Committee", "committee", "title"),
    ("beams", "Beam", "beam", "title"),
]
"""Vocabularies a "NAME records" search is matched against.

Each entry names a vocabulary behind a CERN custom field and the search shortcut
(see ``CDS_SEARCH_ALIASES`` in ``invenio.cfg``) that searches it.
"""
