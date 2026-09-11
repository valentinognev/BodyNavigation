import { useState } from "react";
import { commitPhases } from "../scenario";
import { Screen, TextInput, inputClass } from "./fields";
import { useScenario } from "./useScenario";

export function Modules() {
  const { scenario, applyFormPatch } = useScenario();
  if (scenario == null) return null;
  const modules = scenario.modules;
  return (
    <Screen title="Modules">
      {modules.map((mod, index) => (
        <div key={index} className="space-y-2 rounded border border-slate-200 p-2 dark:border-slate-700">
          <TextInput
            label="name"
            value={mod.name}
            onChange={(name) => {
              const next = modules.map((m, i) => (i === index ? { ...m, name } : m));
              applyFormPatch({ modules: next });
            }}
          />
          <PhasesInput
            value={mod.phases}
            onCommit={(phases) => {
              const next = modules.map((m, i) => (i === index ? { ...m, phases } : m));
              applyFormPatch({ modules: next });
            }}
          />
          <button
            type="button"
            className={`${inputClass} w-auto`}
            onClick={() => applyFormPatch({ modules: modules.filter((_, i) => i !== index) })}
          >
            Remove
          </button>
        </div>
      ))}
      <button
        type="button"
        className={`${inputClass} w-auto`}
        onClick={() =>
          applyFormPatch({
            modules: [...modules, { name: "", phases: ["def", "init", "exec"] }],
          })
        }
      >
        Add module
      </button>
    </Screen>
  );
}

function PhasesInput({
  value,
  onCommit,
}: {
  value: string[];
  onCommit: (phases: string[]) => void;
}) {
  const idle = value.join(", ");
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState(idle);
  return (
    <label className="grid grid-cols-[9rem_1fr] items-center gap-2 text-sm">
      <span>phases</span>
      <input
        className={inputClass}
        type="text"
        value={focused ? draft : idle}
        onFocus={() => {
          setDraft(idle);
          setFocused(true);
        }}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => {
          onCommit(commitPhases(draft));
          setFocused(false);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            e.currentTarget.blur();
          }
        }}
      />
    </label>
  );
}
