import json, re, os
from tqdm import tqdm
from typing import Optional
from llmtuner.chat import ChatModel
from simple_corrector import make_prescreen_func_from_direct


DIRECT_STATS_JSON = "/mnt/public/algm/wb/data/gpt4o_cot_responses_allerrors_direct.jsonl" 
THRESHOLD = 1
MIN_FREQ  = 5
#prescreen = make_prescreen_func_from_direct(DIRECT_STATS_JSON)
# ── 1. Path configuration ───────────────────────────────────────────────────────────
DATA_DIR   = "/mnt/public/algm/wb/ChatKBQA/"
INPUT_JSON = os.path.join(DATA_DIR, "beam_test_top_k_predictions_with_questions.json")
OUTPUT_JSON = os.path.join(
    DATA_DIR, "beam_test_top_k_predictions_with_llama3allerror_full3fold.json"
)

# ── 2. Initialize the correction LLM ─────────────────────────────────────────────────────
args = dict(
    model_name_or_path="/mnt/public/algm/wb/pretrained/Meta-Llama-3.1-8B-Instruct",
    adapter_name_or_path="/mnt/public/algm/wb/kbqa/ChatKBQA_SFT/Reading/"
                         "LLaMA2-7b/WebQSP_Freebase_NQ_lora_epoch100/checkpoint_full3fold",
    finetuning_type="lora",
    template="llama3",
    num_beams=40
)
# args = dict(
#     model_name_or_path="/mnt/public/algm/wb/pretrained/Llama-2-7b-hf",
#     adapter_name_or_path="/mnt/public/algm/wb/kbqa/ChatKBQA_SFT/Reading/"
#                          "LLaMA2-7b/WebQSP_Freebase_NQ_lora_epoch100/checkpoint_allerrorllama2_16",
#     finetuning_type="lora",
#     template="llama2",
#     num_beams=15
# )
repair_model = ChatModel(args)

# ── 3. Shared prompt template (same as the previous version) ───────────────────────────────
INSTRUCT = ("You are a semantic-parsing expert. For each example you receive:\nA natural-language “Question”.\nA “Predicted logical form” in S-expression.\n\nThink step-by-step and write:\n\n[Correctness] – carefully check every entity, relationship and operators.\nUse numbered reasoning steps that evaluate whether the predicted logical form fully and accurately answers the question, ending with\n  “Tentative verdict: pred likely correct.”  OR\n  “Tentative verdict: pred likely incorrect.”\n\n[Revision]   – give minimal edits and a final <revised_logic_form> … </revised_logic_form> ONLY when the predicted form is judged incorrect.\nPlain ASCII text only.")

TAG_RE = re.compile(
    r"<revised_logic_form>\s*(.*?)\s*</revised_logic_form>",
    flags=re.S
)

def extract_revision(text) -> Optional[str]:
    """Extract the S-expression inside <revised_logic_form> tags, if present."""
    reply_text = text[0].response_text
    m = TAG_RE.search(reply_text)
    return m.group(1).strip() if m else None

# ── 4. Main loop: run correction for each sample ─────────────────────────────────────────
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    samples = json.load(f)
skipped, attempted, revised_cnt = 0,0,0
for sample in tqdm(samples, desc="Repairing"):
    question    = sample["question"]
    pred_lf0    = sample["predictions"][0]
    # if prescreen(pred_lf0 , threshold= THRESHOLD , min_freq= MIN_FREQ):
    #     sample["prescreen_skip"] = True
    #     skipped += 1
    #     continue
    sample["prescreen_skip"] = False
    attempted += 1
    user_prompt = (
        f"{INSTRUCT}\n"
        f"Question: {question}\n"
        f"Predicted logical form: {pred_lf0}"
    )
    messages = [{"role": "user", "content": user_prompt}]

    # Call the LLM.
    reply = repair_model.chat(
        messages,
        num_beams=50,
        do_sample=False
    )



    # Parse and insert the revision.
    revised = extract_revision(reply)
    if revised:
        sample["predictions"].insert(0, revised)
        revised_cnt += 1

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"Done! updated file → {OUTPUT_JSON}")
print(f"[stats] prescreen skipped: {skipped} | attempted repair: {attempted} | revised inserted: {revised_cnt}")
