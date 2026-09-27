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

export function FieldLabel({ label, hint }: { label: string; hint?: string }) {
  return (
    <span
      title={hint}
      className={
        hint
          ? "cursor-help underline decoration-dotted decoration-slate-400 underline-offset-2"
          : undefined
      }
    >
      {label}
    </span>
  );
}

export function Row({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="grid grid-cols-[9rem_1fr] items-center gap-2 text-sm">
      <FieldLabel label={label} hint={hint} />
      {children}
    </label>
  );
}

const controlClass =
  "rounded border border-slate-300 bg-white px-2 py-1 text-sm disabled:bg-slate-100 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100 dark:disabled:bg-slate-800/60";

const inputClass = `w-full ${controlClass}`;

export function TextInput({
  label,
  hint,
  value,
  onChange,
}: {
  label: string;
  hint?: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <Row label={label} hint={hint}>
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
  hint,
  value,
  onChange,
}: {
  label: string;
  hint?: string;
  value: number | null | undefined;
  onChange: (v: number | null) => void;
}) {
  const idle = value == null ? "" : String(value);
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState(idle);
  return (
    <Row label={label} hint={hint}>
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

export function SelectInput({
  label,
  hint,
  value,
  options,
  onChange,
}: {
  label: string;
  hint?: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}) {
  const listed = options.some((option) => option.value === value)
    ? options
    : [{ value, label: value }, ...options];
  return (
    <Row label={label} hint={hint}>
      <select className={inputClass} value={value} onChange={(e) => onChange(e.target.value)}>
        {listed.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </Row>
  );
}

export function PathInput({
  label,
  hint,
  value,
  onChange,
  onBrowse,
}: {
  label: string;
  hint?: string;
  value: string;
  onChange: (v: string) => void;
  onBrowse: () => void;
}) {
  return (
    <Row label={label} hint={hint}>
      <span className="flex min-w-0 gap-2">
        <input
          className={`${controlClass} min-w-0 flex-1`}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
        <button type="button" className={`${controlClass} shrink-0`} onClick={onBrowse}>
          Browse
        </button>
      </span>
    </Row>
  );
}

export function CheckInput({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <Row label={label} hint={hint}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
    </Row>
  );
}

export { inputClass };
