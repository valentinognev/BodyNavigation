import { OPTION_KEYS } from "../scenario";
import { CheckInput, NumInput, Screen, TextInput } from "./fields";
import { useScenario } from "./useScenario";

export function Overview() {
  const { scenario, applyFormPatch } = useScenario();
  if (scenario == null) return null;
  const options = scenario.options;
  return (
    <Screen title="Overview">
      <TextInput
        label="title"
        value={scenario.title}
        onChange={(title) => applyFormPatch({ title })}
      />
      <TextInput
        label="family"
        value={scenario.family ?? ""}
        onChange={(family) => applyFormPatch({ family: family.trim() === "" ? undefined : family })}
      />
      <NumInput
        label="iseed"
        value={scenario.iseed}
        onChange={(iseed) => applyFormPatch({ iseed: iseed ?? undefined })}
      />
      {OPTION_KEYS.map((key) => (
        <CheckInput
          key={key}
          label={key}
          checked={options[key] === true}
          onChange={(value) => applyFormPatch({ options: { ...options, [key]: value } })}
        />
      ))}
    </Screen>
  );
}
