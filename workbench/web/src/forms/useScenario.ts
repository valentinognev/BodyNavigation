import { useStore } from "zustand";
import store from "../store";

export function useScenario() {
  const scenario = useStore(store, (s) => s.scenario);
  const applyFormPatch = useStore(store, (s) => s.applyFormPatch);
  const program = useStore(store, (s) => s.program);
  return { scenario, applyFormPatch, program };
}
