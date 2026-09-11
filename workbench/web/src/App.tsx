import { useEffect } from "react";
import { useStore } from "zustand";
import { loadCatalogInto } from "./api";
import { Editor } from "./Editor";
import { StartScreen } from "./StartScreen";
import store from "./store";

export default function App() {
  const view = useStore(store, (s) => s.view);

  useEffect(() => {
    void loadCatalogInto(store);
  }, []);

  if (view === "start") {
    return <StartScreen />;
  }

  return <Editor />;
}
