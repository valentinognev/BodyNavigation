import { schemaDescription } from "./fieldHelp";
import { PHASES, moduleNames } from "./fieldChoices";
import { FieldLabel, Screen, SelectInput, TextInput, inputClass } from "./fields";
import { useScenario } from "./useScenario";

const phaseList = PHASES as readonly string[];

function nextPhases(current: string[], phase: string, checked: boolean): string[] {
  const present = new Set(current);
  if (checked) present.add(phase);
  else present.delete(phase);
  const known = phaseList.filter((name) => present.has(name));
  const extra = current.filter((name) => !phaseList.includes(name) && present.has(name));
  return [...known, ...extra];
}

export function Modules() {
  const { scenario, applyFormPatch, program } = useScenario();
  if (scenario == null) return null;
  const modules = scenario.modules;
  const names = program ? moduleNames[program] : undefined;
  return (
    <Screen title="Modules">
      {modules.map((mod, index) => (
        <div key={index} className="space-y-2 rounded border border-slate-200 p-2 dark:border-slate-700">
          {names ? (
            <SelectInput
              label="name"
              hint={schemaDescription("module.name")}
              value={mod.name}
              options={names.map((value) => ({ value, label: value }))}
              onChange={(name) => {
                const next = modules.map((m, i) => (i === index ? { ...m, name } : m));
                applyFormPatch({ modules: next });
              }}
            />
          ) : (
            <TextInput
              label="name"
              hint={schemaDescription("module.name")}
              value={mod.name}
              onChange={(name) => {
                const next = modules.map((m, i) => (i === index ? { ...m, name } : m));
                applyFormPatch({ modules: next });
              }}
            />
          )}
          <PhasesInput
            value={mod.phases}
            onChange={(phases) => {
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
  onChange,
}: {
  value: string[];
  onChange: (phases: string[]) => void;
}) {
  const extra = value.filter((name) => !phaseList.includes(name));
  const shown = [...phaseList, ...extra];
  return (
    <div className="grid grid-cols-[9rem_1fr] items-center gap-2 text-sm">
      <FieldLabel label="phases" hint={schemaDescription("phases")} />
      <span className="flex flex-wrap gap-3">
        {shown.map((phase) => (
          <label key={phase} className="flex items-center gap-1">
            <input
              type="checkbox"
              checked={value.includes(phase)}
              onChange={(e) => onChange(nextPhases(value, phase, e.target.checked))}
            />
            {phase}
          </label>
        ))}
      </span>
    </div>
  );
}
