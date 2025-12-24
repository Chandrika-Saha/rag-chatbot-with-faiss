import json

IN_FILE = "meditations.json"     # your current file
OUT_FILE = "quotes_flat.json"    # flattened output

with open(IN_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

out = []
for b_i, book in enumerate(data["books"], start=1):
    title = book["book_title"]
    for p_i, para in enumerate(book["paragraphs"], start=1):
        text = para.strip()
        if not text:
            continue
        out.append({
            "id": f"b{b_i}_p{p_i}",
            "book": title,
            "text": text
        })

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(out)} passages to {OUT_FILE}")
