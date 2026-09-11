import { useState } from "react";
import type { Vehicle, VehicleEvent } from "../scenario";
import { Screen, inputClass } from "./fields";
import { useScenario } from "./useScenario";

function patchVehicle(vehicles: Vehicle[], index: number, next: Vehicle): Vehicle[] {
  return vehicles.map((v, i) => (i === index ? next : v));
}

export function Events() {
  const { scenario, applyFormPatch } = useScenario();
  if (scenario == null) return null;
  return (
    <Screen title="Events">
      {scenario.vehicles.map((vehicle, vIndex) => (
        <div key={vIndex} className="space-y-2 rounded border border-slate-200 p-2 dark:border-slate-700">
          <p className="text-sm font-medium">
            {vehicle.name || vehicle.type || `vehicle ${vIndex + 1}`}
          </p>
          {vehicle.events.map((event, eIndex) => (
            <EventCard
              key={eIndex}
              event={event}
              onChange={(nextEvent) => {
                const events = vehicle.events.map((e, i) => (i === eIndex ? nextEvent : e));
                applyFormPatch({ vehicles: patchVehicle(scenario.vehicles, vIndex, { ...vehicle, events }) });
              }}
              onRemove={() =>
                applyFormPatch({
                  vehicles: patchVehicle(scenario.vehicles, vIndex, {
                    ...vehicle,
                    events: vehicle.events.filter((_, i) => i !== eIndex),
                  }),
                })
              }
            />
          ))}
          <button
            type="button"
            className={`${inputClass} w-auto`}
            onClick={() =>
              applyFormPatch({
                vehicles: patchVehicle(scenario.vehicles, vIndex, {
                  ...vehicle,
                  events: [...vehicle.events, { when: {}, set: {} }],
                }),
              })
            }
          >
            Add event
          </button>
        </div>
      ))}
    </Screen>
  );
}

function EventCard({
  event,
  onChange,
  onRemove,
}: {
  event: VehicleEvent;
  onChange: (next: VehicleEvent) => void;
  onRemove: () => void;
}) {
  return (
    <div className="space-y-2 rounded border border-slate-200 p-2 dark:border-slate-700">
      <JsonObjectField
        label="when"
        value={event.when}
        onCommit={(when) => onChange({ ...event, when })}
      />
      <JsonObjectField
        label="set"
        value={event.set}
        onCommit={(set) => onChange({ ...event, set })}
      />
      <button type="button" className={`${inputClass} w-auto`} onClick={onRemove}>
        Remove
      </button>
    </div>
  );
}

function JsonObjectField({
  label,
  value,
  onCommit,
}: {
  label: string;
  value: Record<string, unknown>;
  onCommit: (next: Record<string, unknown>) => void;
}) {
  const idle = JSON.stringify(value, null, 2);
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState(idle);
  return (
    <label className="block space-y-1 text-sm">
      <span>{label}</span>
      <textarea
        className={`${inputClass} min-h-[5rem] font-mono`}
        value={focused ? draft : idle}
        onFocus={() => {
          setDraft(idle);
          setFocused(true);
        }}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => {
          try {
            const parsed = JSON.parse(draft) as unknown;
            if (parsed != null && typeof parsed === "object" && !Array.isArray(parsed)) {
              onCommit(parsed as Record<string, unknown>);
            }
          } catch {
            /* keep last good object */
          }
          setFocused(false);
        }}
        spellCheck={false}
      />
    </label>
  );
}
