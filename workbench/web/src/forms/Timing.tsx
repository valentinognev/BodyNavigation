import { useState } from "react";
import { commitNumericRecord } from "../scenario";
import { NumInput, Screen, inputClass } from "./fields";
import { useScenario } from "./useScenario";

export function Timing() {
  const { scenario, applyFormPatch } = useScenario();
  const [newKey, setNewKey] = useState("");
  if (scenario == null) return null;
  return (
    <Screen title="Timing">
      <NumInput
        label="end_time"
        value={scenario.end_time}
        onChange={(end_time) => {
          if (end_time != null) applyFormPatch({ end_time });
        }}
      />
      {Object.entries(scenario.timing).map(([key, value]) => (
        <TimingRow
          key={key}
          name={key}
          value={value}
          onCommit={(draft) => applyFormPatch({ timing: commitNumericRecord(scenario.timing, key, draft) })}
        />
      ))}
      <div className="flex gap-2">
        <input
          className={inputClass}
          type="text"
          placeholder="new timing key"
          value={newKey}
          onChange={(e) => setNewKey(e.target.value)}
        />
        <button
          type="button"
          className={`${inputClass} w-auto`}
          onClick={() => {
            const name = newKey.trim();
            if (name === "" || name in scenario.timing) return;
            applyFormPatch({ timing: { ...scenario.timing, [name]: 0 } });
            setNewKey("");
          }}
        >
          Add
        </button>
      </div>
    </Screen>
  );
}

function TimingRow({
  name,
  value,
  onCommit,
}: {
  name: string;
  value: number;
  onCommit: (draft: string) => void;
}) {
  const idle = String(value);
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState(idle);
  return (
    <label className="grid grid-cols-[9rem_1fr] items-center gap-2 text-sm">
      <span>{name}</span>
      <input
        className={inputClass}
        type="text"
        inputMode="decimal"
        value={focused ? draft : idle}
        onFocus={() => {
          setDraft(idle);
          setFocused(true);
        }}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => {
          onCommit(draft);
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
