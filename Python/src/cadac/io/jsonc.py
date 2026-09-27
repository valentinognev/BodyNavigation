import json


def _skip_ws(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i] in " \t\r\n":
        i += 1
    return i


def leading_comment_prefix(text: str) -> str:
    """Comment block before the root value, including markers, ending in a newline."""
    i = _skip_ws(text, 0)
    if i + 1 >= len(text) or text[i] != "/":
        return ""
    start = i
    end = i
    n = len(text)
    while True:
        i = _skip_ws(text, i)
        if i + 1 < n and text[i] == "/" and text[i + 1] == "/":
            i += 2
            while i < n and text[i] not in "\n\r":
                i += 1
            if i < n and text[i] == "\r":
                i += 1
            if i < n and text[i] == "\n":
                i += 1
            end = i
            continue
        if i + 1 < n and text[i] == "/" and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i = n if i + 1 >= n else i + 2
            end = i
            continue
        break
    if end <= start:
        return ""
    prefix = text[start:end]
    if not prefix.endswith("\n"):
        prefix += "\n"
    return prefix


def leading_comment(text: str) -> str | None:
    """Text of the comment before the root value. None when the file has none."""
    prefix = leading_comment_prefix(text)
    if not prefix:
        return None
    parts: list[str] = []
    i = 0
    n = len(prefix)
    while i < n:
        i = _skip_ws(prefix, i)
        if i + 1 < n and prefix[i] == "/" and prefix[i + 1] == "/":
            i += 2
            line_end = prefix.find("\n", i)
            if line_end < 0:
                line_end = n
            parts.append(prefix[i:line_end].strip())
            i = line_end + 1
            continue
        if i + 1 < n and prefix[i] == "/" and prefix[i + 1] == "*":
            i += 2
            end = prefix.find("*/", i)
            body = prefix[i:end if end >= 0 else n]
            lines = [line.strip() for line in body.splitlines()]
            while lines and lines[0] == "":
                lines.pop(0)
            while lines and lines[-1] == "":
                lines.pop()
            if lines:
                parts.append("\n".join(lines))
            i = (end + 2) if end >= 0 else n
            continue
        break
    text_out = "\n".join(part for part in parts if part)
    return text_out or None


def _strip_comments(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    in_string = False
    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "/":
                i += 2
                while i < n and text[i] not in "\n\r":
                    i += 1
                continue
            if nxt == "*":
                i += 2
                while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i = n if i + 1 >= n else i + 2
                continue
        out.append(c)
        i += 1
    return "".join(out)


def loads(text: str) -> object:
    return json.loads(_strip_comments(text))


def load(path) -> object:
    with open(path, encoding="utf-8") as f:
        return loads(f.read())
