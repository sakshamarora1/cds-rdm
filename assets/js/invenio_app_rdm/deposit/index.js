import React from "react";
import ReactDOM from "react-dom";
import { getInputFromDOM } from "@js/invenio_rdm_records/";
import { RDMDepositForm } from "./RDMDepositForm";
import { OverridableContext, overrideStore } from "react-overridable";

const overriddenComponents = overrideStore.getAll();

ReactDOM.render(
  <OverridableContext.Provider value={overriddenComponents}>
    <RDMDepositForm
      record={getInputFromDOM("deposits-record")}
      preselectedCommunity={getInputFromDOM("deposits-draft-community")}
      files={getInputFromDOM("deposits-record-files")}
      config={getInputFromDOM("deposits-config")}
      permissions={getInputFromDOM("deposits-record-permissions")}
      filesLocked={getInputFromDOM("deposits-record-locked-files")}
      recordRestrictionGracePeriod={getInputFromDOM(
        "deposits-record-restriction-grace-period"
      )}
      allowRecordRestriction={getInputFromDOM("deposits-allow-record-restriction")}
      groupsEnabled={getInputFromDOM("config-groups-enabled")}
    />
  </OverridableContext.Provider>,
  document.getElementById("deposit-form")
);
