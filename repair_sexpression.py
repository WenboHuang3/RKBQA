# apply_repair.py
import json, re, argparse

def repair_sexpr(expr: str) -> str:
    # 插入括号空格
    expr = re.sub(r'([\[\]\(\)])', r' \1 ', expr)

    # 逗号：仅在不位于 [...] 内时补空格
    buf, depth = [], 0
    for ch in expr:
        if ch == '[': depth += 1
        elif ch == ']': depth -= 1
        buf.append(' , ' if ch == ',' and depth == 0 else ch)
    expr = ''.join(buf)

    # 对含有≥2个逗号的 [ ... ]（关系）补齐逗号两侧空格
    def fix_bracket(m):
        inner = m.group(1)
        if inner.count(',') >= 2:                 # 判定为关系
            inner = re.sub(r'\s*,\s*', ' , ', inner)
        return f'[ {inner.strip()} ]'
    expr = re.sub(r'\[([^\[\]]+)\]', fix_bracket, expr)

    return re.sub(r'\s+', ' ', expr).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="beam_test_top_k_predictions.json 的路径")
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