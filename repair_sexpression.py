# apply_repair.py
import json, re, argparse

def repair_sexpr(expr: str) -> str:
    # Add spaces around brackets and parentheses.
    expr = re.sub(r'([\[\]\(\)])', r' \1 ', expr)

    # Add spaces around commas only when they are outside [...].
    buf, depth = [], 0
    for ch in expr:
        if ch == '[': depth += 1
        elif ch == ']': depth -= 1
        buf.append(' , ' if ch == ',' and depth == 0 else ch)
    expr = ''.join(buf)

    # Add spaces around commas in [ ... ] relations with at least two commas.
    def fix_bracket(m):
        inner = m.group(1)
        if inner.count(',') >= 2:                 # Treat this as a relation.
            inner = re.sub(r'\s*,\s*', ' , ', inner)
        return f'[ {inner.strip()} ]'
    expr = re.sub(r'\[([^\[\]]+)\]', fix_bracket, expr)

    return re.sub(r'\s+', ' ', expr).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Path to beam_test_top_k_predictions.json")
    parser.add_argument("--output", default="beam_test_top_k_predictions_fixed.json")
    args = parser.parse_args()

    data = json.load(open(args.input, encoding="utf8"))
    for item in data:
        preds = item.get("predictions") or item.get("pred", {}).get("predictions")
        if preds:
            preds[0] = repair_sexpr(preds[0])
            print(preds[0])
    json.dump(data, open(args.output, "w", encoding="utf8"),
              ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
