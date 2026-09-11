import { useState } from "react";
import { commitParamValue, type Vehicle } from "../scenario";
import { Screen, TextInput, inputClass } from "./fields";
import { useScenario } from "./useScenario";

function patchVehicle(vehicles: Vehicle[], index: number, next: Vehicle): Vehicle[] {
  return vehicles.map((v, i) => (i === index ? next : v));
}

export function Vehicles() {
  const { scenario, applyFormPatch } = useScenario();
  if (scenario == null) return null;
  const vehicles = scenario.vehicles;
  return (
    <Screen title="Vehicles">
      {vehicles.map((vehicle, index) => (
        <VehicleCard
          key={index}
          vehicle={vehicle}
          onChange={(next) => applyFormPatch({ vehicles: patchVehicle(vehicles, index, next) })}
          onRemove={() => applyFormPatch({ vehicles: vehicles.filter((_, i) => i !== index) })}
        />
      ))}
      <button
        type="button"
        className={`${inputClass} w-auto`}
        onClick={() =>
          applyFormPatch({
            vehicles: [...vehicles, { type: "", name: "", params: {}, events: [] }],
          })
        }
      >
        Add vehicle
      </button>
    </Screen>
  );
}

function VehicleCard({
  vehicle,
  onChange,
  onRemove,
}: {
  vehicle: Vehicle;
  onChange: (next: Vehicle) => void;
  onRemove: () => void;
}) {
  const [newParam, setNewName] = useState("");
  return (
    <div className="space-y-2 rounded border border-slate-200 p-2 dark:border-slate-700">
      <TextInput label="type" value={vehicle.type} onChange={(type) => onChange({ ...vehicle, type })} />
      <TextInput label="name" value={vehicle.name} onChange={(name) => onChange({ ...vehicle, name })} />
      <TextInput
        label="family"
        value={vehicle.family ?? ""}
        onChange={(family) =>
          onChange({ ...vehicle, family: family.trim() === "" ? undefined : family })
        }
      />
      <div className="space-y-1">
        <p className="text-sm font-medium">params</p>
        {Object.entries(vehicle.params).map(([key, value]) => (
          <ParamRow
            key={key}
            name={key}
            value={value}
            onCommit={(draft) => onChange({ ...vehicle, params: commitParamValue(vehicle.params, key, draft) })}
          />
        ))}
        <div className="flex gap-2">
          <input
            className={inputClass}
            type="text"
            placeholder="new param"
            value={newParam}
            onChange={(e) => setNewName(e.target.value)}
          />
          <button
            type="button"
            className={`${inputClass} w-auto`}
            onClick={() => {
              const name = newParam.trim();
              if (name === "" || name in vehicle.params) return;
              onChange({ ...vehicle, params: { ...vehicle.params, [name]: 0 } });
              setNewName("");
            }}
          >
            Add
          </button>
        </div>
      </div>
      <div className="flex gap-2">
        <button type="button" className={`${inputClass} w-auto`} disabled title="plan 5">
          Launch MISDC
        </button>
        <button type="button" className={`${inputClass} w-auto`} disabled title="plan 5">
          Launch AID
        </button>
        <button type="button" className={`${inputClass} w-auto`} onClick={onRemove}>
          Remove
        </button>
      </div>
    </div>
  );
}

function ParamRow({
  name,
  value,
  onCommit,
}: {
  name: string;
  value: number | string;
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
