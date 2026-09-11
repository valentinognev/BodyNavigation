import { useState, type KeyboardEvent, type ReactNode } from "react";

export function Screen({ title, children }: { title: string; children: ReactNode }) {
  return (
    <form
      className="space-y-2 text-slate-800 dark:text-slate-100"
      onSubmit={(e) => e.preventDefault()}
    >
      <h2 className="mb-3 text-base font-semibold">{title}</h2>
      {children}
    </form>
  );
}

export function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="grid grid-cols-[9rem_1fr] items-center gap-2 text-sm">
      <span>{label}</span>
      {children}
    </label>
  );
}

const inputClass =
  "w-full rounded border border-slate-300 bg-white px-2 py-1 text-sm disabled:bg-slate-100 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100 dark:disabled:bg-slate-800/60";

export function TextInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <Row label={label}>
      <input
        className={inputClass}
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </Row>
  );
}

function enterBlurs(e: KeyboardEvent<HTMLInputElement>) {
  if (e.key === "Enter") {
    e.preventDefault();
    e.currentTarget.blur();
  }
}

export function NumInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number | null | undefined;
  onChange: (v: number | null) => void;
}) {
  const idle = value == null ? "" : String(value);
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState(idle);
  return (
    <Row label={label}>
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
          const trimmed = draft.trim();
          if (trimmed === "") onChange(null);
          else {
            const num = Number(trimmed);
            if (Number.isFinite(num)) onChange(num);
          }
          setFocused(false);
        }}
        onKeyDown={enterBlurs}
      />
    </Row>
  );
}

export function CheckInput({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <Row label={label}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
    </Row>
  );
}

export { inputClass };
