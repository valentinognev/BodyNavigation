import type { Vehicle } from "../scenario";
import { Screen, TextInput } from "./fields";
import { useScenario } from "./useScenario";

const DECK_KEYS = ["aero_deck", "prop_deck", "weather_deck", "sam_deck", "srmb_deck"] as const;
type DeckKey = (typeof DECK_KEYS)[number];

function setDeck(vehicle: Vehicle, key: DeckKey, value: string): Vehicle {
  const next = { ...vehicle };
  const trimmed = value.trim();
  if (trimmed === "") delete next[key];
  else next[key] = trimmed;
  return next;
}

export function Decks() {
  const { scenario, applyFormPatch } = useScenario();
  if (scenario == null) return null;
  return (
    <Screen title="Decks">
      {scenario.vehicles.map((vehicle, index) => (
        <div key={index} className="space-y-2 rounded border border-slate-200 p-2 dark:border-slate-700">
          <p className="text-sm font-medium">
            {vehicle.name || vehicle.type || `vehicle ${index + 1}`}
          </p>
          {DECK_KEYS.map((key) => (
            <TextInput
              key={key}
              label={key}
              value={vehicle[key] ?? ""}
              onChange={(value) => {
                const vehicles = scenario.vehicles.map((v, i) =>
                  i === index ? setDeck(v, key, value) : v,
                );
                applyFormPatch({ vehicles });
              }}
            />
          ))}
        </div>
      ))}
    </Screen>
  );
}
