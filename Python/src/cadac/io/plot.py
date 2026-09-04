from pathlib import Path


def write_plot_csv(path, title: str, columns: list[str], rows: list[list[float]]) -> None:
    path = Path(path)
    n = len(columns)
    lines = [title, f"0  0 {n}", ",".join(columns) + ","]
    for row in rows:
        lines.append(",".join(str(value) for value in row) + ",")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
