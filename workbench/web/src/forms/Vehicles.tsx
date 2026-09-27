import { useState } from "react";
import { commitParamValue, type Vehicle } from "../scenario";
import { FAMILIES, paramChoices, vehicleTypes, type ParamChoice } from "./fieldChoices";
import { schemaDescription, variableDescription } from "./fieldHelp";
import { FieldLabel, Screen, SelectInput, TextInput, inputClass } from "./fields";
import { useScenario } from "./useScenario";

const buttonClass =
  "w-auto rounded bg-slate-800 px-3 py-1 text-sm text-white disabled:cursor-not-allowed disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900";

function patchVehicle(vehicles: Vehicle[], index: number, next: Vehicle): Vehicle[] {
  return vehicles.map((v, i) => (i === index ? next : v));
}

export function Vehicles() {
  const { scenario, applyFormPatch, program } = useScenario();
  if (scenario == null) return null;
  const vehicles = scenario.vehicles;
  return (
    <Screen title="Vehicles">
      {vehicles.map((vehicle, index) => (
        <VehicleCard
          key={index}
          index={index}
          program={program}
          vehicle={vehicle}
          onChange={(next) => applyFormPatch({ vehicles: patchVehicle(vehicles, index, next) })}
          onRemove={() => applyFormPatch({ vehicles: vehicles.filter((_, i) => i !== index) })}
        />
      ))}
      <button
        type="button"
        className={buttonClass}
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

function vehicleHeading(vehicle: Vehicle, index: number): { title: string; type: string } {
  const name = vehicle.name.trim();
  const type = vehicle.type.trim();
  if (name !== "") return { title: name, type };
  if (type !== "") return { title: type, type: "" };
  return { title: `Vehicle ${index + 1}`, type: "" };
}

function VehicleCard({
  index,
  program,
  vehicle,
  onChange,
  onRemove,
}: {
  index: number;
  program: string | null;
  vehicle: Vehicle;
  onChange: (next: Vehicle) => void;
  onRemove: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [newParam, setNewName] = useState("");
  const heading = vehicleHeading(vehicle, index);
  return (
    <div className="rounded border border-slate-200 dark:border-slate-700">
      <button
        type="button"
        className="flex w-full min-w-0 items-center gap-2 p-2 text-left text-sm font-medium"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        <span aria-hidden="true">{open ? "▾" : "▸"}</span>
        <span className="truncate">{heading.title}</span>
        {heading.type !== "" ? (
          <span className="truncate font-normal text-slate-500 dark:text-slate-400">{heading.type}</span>
        ) : null}
      </button>
      {open ? (
        <div className="space-y-2 px-2 pb-2">
          {program && vehicleTypes[program] ? (
            <SelectInput
              label="type"
              hint={schemaDescription("vehicle.type")}
              value={vehicle.type}
              options={vehicleTypes[program].map((value) => ({ value, label: value }))}
              onChange={(type) => onChange({ ...vehicle, type })}
            />
          ) : (
            <TextInput
              label="type"
              hint={schemaDescription("vehicle.type")}
              value={vehicle.type}
              onChange={(type) => onChange({ ...vehicle, type })}
            />
          )}
          <TextInput
            label="name"
            hint={schemaDescription("vehicle.name")}
            value={vehicle.name}
            onChange={(name) => onChange({ ...vehicle, name })}
          />
          <SelectInput
            label="family"
            hint={schemaDescription("vehicle.family")}
            value={vehicle.family ?? ""}
            options={["", ...FAMILIES].map((value) => ({ value, label: value }))}
            onChange={(family) =>
              onChange({ ...vehicle, family: family.trim() === "" ? undefined : family })
            }
          />
          <div className="space-y-1">
            <p className="text-sm font-medium">
              <FieldLabel label="params" hint={schemaDescription("params")} />
            </p>
            {Object.entries(vehicle.params).map(([key, value]) => (
              <ParamRow
                key={key}
                name={key}
                hint={variableDescription(program, key)}
                value={value}
                choices={program ? paramChoices[program]?.[key] : undefined}
                onCommit={(draft) =>
                  onChange({ ...vehicle, params: commitParamValue(vehicle.params, key, draft) })
                }
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
                className={buttonClass}
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
            <button type="button" className={buttonClass} disabled title="plan 5">
              Launch MISDC
            </button>
            <button type="button" className={buttonClass} disabled title="plan 5">
              Launch AID
            </button>
            <button type="button" className={buttonClass} onClick={onRemove}>
              Remove
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function choiceOptions(choices: ParamChoice[]): { value: string; label: string }[] {
  return choices.map((choice) => {
    const value = String(choice.value);
    return { value, label: choice.label === value ? value : `${value}: ${choice.label}` };
  });
}

function ParamRow({
  name,
  hint,
  value,
  choices,
  onCommit,
}: {
  name: string;
  hint?: string;
  value: number | string;
  choices?: ParamChoice[];
  onCommit: (draft: string) => void;
}) {
  if (choices && choices.length >= 2) {
    return (
      <SelectInput
        label={name}
        hint={hint}
        value={String(value)}
        options={choiceOptions(choices)}
        onChange={onCommit}
      />
    );
  }
  const idle = String(value);
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState(idle);
  return (
    <label className="grid grid-cols-[9rem_1fr] items-center gap-2 text-sm">
      <FieldLabel label={name} hint={hint} />
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
