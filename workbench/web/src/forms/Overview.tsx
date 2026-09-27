import { useStore } from "zustand";
import { OPTION_KEYS, type Scenario } from "../scenario";
import store from "../store";
import { schemaDescription } from "./fieldHelp";
import { FAMILIES } from "./fieldChoices";
import { CheckInput, NumInput, Screen, SelectInput, TextInput } from "./fields";
import { useScenario } from "./useScenario";

export function OverviewForm({
  scenario,
  description,
  applyFormPatch,
}: {
  scenario: Scenario;
  description: string;
  applyFormPatch: (partial: Partial<Scenario>) => void;
}) {
  const options = scenario.options;
  return (
    <Screen title="Overview">
      {description ? (
        <p className="whitespace-pre-wrap rounded border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200">
          {description}
        </p>
      ) : null}
      <TextInput
        label="title"
        hint={schemaDescription("title")}
        value={scenario.title}
        onChange={(title) => applyFormPatch({ title })}
      />
      <SelectInput
        label="family"
        hint={schemaDescription("scenario.family")}
        value={scenario.family ?? ""}
        options={["", ...FAMILIES].map((value) => ({ value, label: value }))}
        onChange={(family) => applyFormPatch({ family: family.trim() === "" ? undefined : family })}
      />
      <NumInput
        label="iseed"
        hint={schemaDescription("iseed")}
        value={scenario.iseed}
        onChange={(iseed) => applyFormPatch({ iseed: iseed ?? undefined })}
      />
      {OPTION_KEYS.map((key) => (
        <CheckInput
          key={key}
          label={key}
          hint={schemaDescription(key)}
          checked={options[key] === true}
          onChange={(value) => applyFormPatch({ options: { ...options, [key]: value } })}
        />
      ))}
    </Screen>
  );
}

export function Overview() {
  const { scenario, applyFormPatch } = useScenario();
  const description = useStore(store, (s) => s.caseDescription);
  if (scenario == null) return null;
  return (
    <OverviewForm
      scenario={scenario}
      description={description}
      applyFormPatch={applyFormPatch}
    />
  );
}
