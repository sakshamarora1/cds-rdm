// This file is part of CDS RDM
// Copyright (C) 2026 CERN.
//
// CDS RDM is free software; you can redistribute it and/or modify it
// under the terms of the GPL-2.0 License; see LICENSE file for more details.

// Help with a free-text search, above the results:
// - /api/search/interpret (LLM, off by default): if it finds a better query,
//   run it and show what was understood, with a button to search for the
//   words as typed instead.
// - /api/search/suggest (no model): "Did you mean" and known names such as
//   experiments or communities.

import { RDMRecordSearchBarContainer } from "@js/invenio_app_rdm/search/components";
import { i18next } from "@translations/invenio_rdm_records/i18next";
import PropTypes from "prop-types";
import React, { useEffect, useState } from "react";
import { http } from "react-invenio-forms";
import { withState } from "react-searchkit";
import { Button, Label, Loader, Message } from "semantic-ui-react";

const STORAGE_KEY = "cds-search-interpretations-v2";
const INTERPRET_URL = "/api/search/interpret";
const SUGGEST_URL = "/api/search/suggest";

// Kept for the browser tab so the banner survives a reload and "search as
// typed" is remembered. Storage may be unavailable (private mode): then the
// banner only lives until the page is left.
const loadStore = () => {
  try {
    const stored = JSON.parse(window.sessionStorage.getItem(STORAGE_KEY));
    return {
      byQuery: stored?.byQuery || {}, // interpreted query -> {original, parts}
      byOriginal: stored?.byOriginal || {}, // search as typed -> interpreted query
      literal: stored?.literal || [], // searches to keep as typed
      disabled: stored?.disabled || {}, // endpoints not enabled: {interpret, suggest}
    };
  } catch (e) {
    return { byQuery: {}, byOriginal: {}, literal: [], disabled: {} };
  }
};

const saveStore = (store) => {
  try {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(store));
  } catch (e) {
    // Storage full or blocked: not needed for the search itself.
  }
};

// Same rule as the backend: field searches and query syntax are kept as typed.
const looksLikePlainText = (query) => query && !/[:"()[\]{}*?~^\\/+!]/.test(query);

// "Did you mean" and known names for one search.
const Suggestions = ({ suggestion, search }) => {
  const { spelling } = suggestion;
  if (!spelling && !suggestion.names.length) {
    return null;
  }
  return (
    <div className="mt-10" aria-live="polite">
      {spelling && (
        <p className="mb-5">
          {i18next.t("Did you mean")}{" "}
          <Button
            type="button"
            className="link"
            basic
            compact
            size="small"
            onClick={() => search(spelling.text)}
          >
            {spelling.text}
          </Button>
          ?
        </p>
      )}
      {suggestion.names.length > 0 && (
        <p className="mb-5">
          <span className="text-muted">{i18next.t("Search for")} </span>
          {suggestion.names.map((name) => (
            <Label
              key={name.query}
              as="button"
              type="button"
              size="small"
              className="mb-5"
              onClick={() => search(name.query)}
            >
              {name.label}
              <Label.Detail>{name.value}</Label.Detail>
            </Label>
          ))}
        </p>
      )}
    </div>
  );
};

Suggestions.propTypes = {
  suggestion: PropTypes.object.isRequired,
  search: PropTypes.func.isRequired,
};

const QueryInterpretationComponent = ({ currentQueryState, updateQueryState }) => {
  const queryString = (currentQueryState.queryString || "").trim();
  const [store, setStore] = useState(loadStore);
  const [pending, setPending] = useState(false);
  const [suggestions, setSuggestions] = useState({}); // search -> suggestion

  const remember = (next) => {
    saveStore(next);
    setStore(next);
  };

  const search = (query) =>
    updateQueryState({ ...currentQueryState, queryString: query, page: 1 });

  useEffect(() => {
    if (
      store.disabled.interpret ||
      !looksLikePlainText(queryString) ||
      store.byQuery[queryString] ||
      store.literal.includes(queryString)
    ) {
      return undefined;
    }
    const known = store.byOriginal[queryString];
    if (known) {
      search(known);
      return undefined;
    }

    let cancelled = false;
    setPending(true);
    http
      .get(INTERPRET_URL, { params: { q: queryString } })
      .then(({ data }) => {
        if (cancelled) return;
        if (!data.query) {
          remember({ ...store, literal: [...store.literal, queryString] });
          return;
        }
        remember({
          ...store,
          byQuery: {
            ...store.byQuery,
            [data.query]: { original: queryString, parts: data.parts },
          },
          byOriginal: { ...store.byOriginal, [queryString]: data.query },
        });
        search(data.query);
      })
      .catch((error) => {
        // The search as typed stands. Stop asking if the endpoint is off.
        if (!cancelled && error?.response?.status === 404) {
          remember({ ...store, disabled: { ...store.disabled, interpret: true } });
        }
      })
      .finally(() => {
        if (!cancelled) setPending(false);
      });
    return () => {
      cancelled = true;
      setPending(false);
    };
    // Only a new search should trigger an interpretation.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryString]);

  // Suggestions once the interpreter has nothing to add (or is off).
  const interpreted = Boolean(
    store.byQuery[queryString] || store.byOriginal[queryString]
  );
  const wantsSuggestions =
    !store.disabled.suggest &&
    looksLikePlainText(queryString) &&
    !interpreted &&
    (store.disabled.interpret || store.literal.includes(queryString));

  useEffect(() => {
    if (!wantsSuggestions || suggestions[queryString]) {
      return undefined;
    }
    let cancelled = false;
    http
      .get(SUGGEST_URL, { params: { q: queryString } })
      .then(({ data }) => {
        if (!cancelled) setSuggestions((known) => ({ ...known, [queryString]: data }));
      })
      .catch((error) => {
        if (!cancelled && error?.response?.status === 404) {
          remember({ ...store, disabled: { ...store.disabled, suggest: true } });
        }
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryString, wantsSuggestions]);

  if (pending) {
    return (
      <div className="mt-10" aria-live="polite">
        <Loader active inline size="mini" />{" "}
        <span className="text-muted">
          {i18next.t("Looking for what your search refers to…")}
        </span>
      </div>
    );
  }

  const interpretation = store.byQuery[queryString];
  if (!interpretation) {
    const suggestion = wantsSuggestions && suggestions[queryString];
    return suggestion ? <Suggestions suggestion={suggestion} search={search} /> : null;
  }

  const searchAsTyped = () => {
    remember({ ...store, literal: [...store.literal, interpretation.original] });
    search(interpretation.original);
  };

  return (
    <Message info size="small" className="mt-10" aria-live="polite">
      <span>{i18next.t("Showing results for")} </span>
      {interpretation.parts.map((part) => (
        <Label key={`${part.label}-${part.value}`} size="small" className="mb-5">
          {part.label}
          <Label.Detail>{part.value}</Label.Detail>
        </Label>
      ))}
      <div className="mt-5">
        <Button type="button" basic compact size="tiny" onClick={searchAsTyped}>
          {i18next.t('Search for "{{query}}" as typed', {
            query: interpretation.original,
          })}
        </Button>
      </div>
    </Message>
  );
};

QueryInterpretationComponent.propTypes = {
  currentQueryState: PropTypes.object.isRequired,
  updateQueryState: PropTypes.func.isRequired,
};

export const QueryInterpretation = withState(QueryInterpretationComponent);

export const CDSRecordSearchBarContainer = ({ appName }) => (
  <>
    <RDMRecordSearchBarContainer appName={appName} />
    <QueryInterpretation />
  </>
);

CDSRecordSearchBarContainer.propTypes = {
  appName: PropTypes.string,
};

CDSRecordSearchBarContainer.defaultProps = {
  appName: "InvenioAppRdm.Search",
};
