#!/usr/bin/env python
# save as tools/build_eval_summary.py

import json, argparse, math, pathlib

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def main(beam_file, score_file, test_file, out_file):
    # Read the three source files.
    beam_map   = {x["qid"]: x for x in load_json(beam_file)}
    score_map  = {x["qid"]: x for x in load_json(score_file)}
    q_text_map = {q["QuestionId"]: q["RawQuestion"]
                  for q in load_json(test_file)["Questions"]}

    summary = []
    for qid, s in score_map.items():
        b = beam_map[qid]

        # Mark as correct only when all four metrics equal 1.
        correct = (
            s.get("precision", 0) == 1.0 and
            s.get("recall",    0) == 1.0 and
            s.get("f1",        0) == 1.0 and
            s.get("hit",       0) == 1
        )

        # Select the beam candidate using execute_index.
        exe_idx   = b["execute_index"]
        pred_norm = b["pred"]["predictions"][exe_idx]
        # denormed_pred is a list, usually containing one item.
        pred_denorm = b["denormed_pred"][0] if isinstance(b["denormed_pred"], list) else b["denormed_pred"]

        summary.append({
            "id":                   qid,
            "question":             q_text_map.get(qid, ""),
            "correct":              correct,
            "gt_normed_sexpr":      b["gt_normed_sexpr"],
            "gt_denormed_sexpr":    b["gt_sexpr"],
            "pred_normed_sexpr":    pred_norm,
            "pred_denormed_sexpr":  pred_denorm
        })

    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ Done: {len(summary)} records written to {out_file}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam",  required=True, help="beam_test_top_k_predictions.json_gen_sexpr_results.json")
    ap.add_argument("--score", required=True, help="*_official_format.json_new.json")
    ap.add_argument("--test",  required=True, help="WebQSP.test.json (30% subset)")
    ap.add_argument("--out",   default="data/webqsp_test_eval_summary.json")
    args = ap.parse_args()
    main(args.beam, args.score, args.test, args.out)
