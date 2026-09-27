import json


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
