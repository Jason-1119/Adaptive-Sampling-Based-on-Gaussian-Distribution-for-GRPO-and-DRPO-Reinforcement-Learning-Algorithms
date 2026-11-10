
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from math_verify import parse, verify, LatexExtractionConfig


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def boxed_answer(text):
    # Preserve the project's grading rule: use its last boxed expression.
    start = text.rfind("\\boxed{")
    if start < 0:
        return ""
    start += len("\\boxed{")
    depth = 1
    for end in range(start, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[start:end].strip()
    return ""


def grade(text, gold):
    prediction = boxed_answer(text)
    if not prediction:
        return False, "missing_or_incomplete_box"
    try:
        extraction = [LatexExtractionConfig()]
        target = parse("\\boxed{" + gold + "}", extraction_config=extraction)
        parsed = parse("\\boxed{" + prediction + "}", extraction_config=extraction)
        if not target:
            raise ValueError("Gold answer could not be parsed: " + gold)
        return bool(parsed and verify(target, parsed)), None if parsed else "prediction_parse_failed"
    except ValueError:
        raise
    except Exception as error:
        return False, type(error).__name__ + ": " + str(error)


def normalize_ids(value):
    # Support lists, tensors, BatchEncoding and tokenizers.Encoding.
    if isinstance(value, dict) or hasattr(value, "keys"):
        value = value["input_ids"]
    elif hasattr(value, "ids"):
        value = value.ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected one prompt.")
        value = value[0]
    if not value or not all(isinstance(i, int) for i in value):
        raise TypeError("Tokenizer did not produce a nonempty integer token list.")
    return value


def prompt_ids(tokenizer, problem, config):
    if config.get("prompt_format", "completion") == "chat":
        content = ("Solve the following mathematics problem. Reason step by step "
                   "and put the final answer in \\boxed{...}.\n\n" + problem)
        return normalize_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True))
    text = ("Solve the following mathematics problem. Show your reasoning and "
            "put the final answer in \\boxed{...}.\n\nProblem: " + problem + "\n\nSolution:")
    return normalize_ids(tokenizer.encode(text, add_special_tokens=True))


def main():
    root, evaluation, arm = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    manifest = json.loads((evaluation / "manifest.json").read_text())
    questions = json.loads((evaluation / "questions.json").read_text())
    config = manifest["settings"]["config"]
    maximum = manifest["settings"]["max_new_tokens"]
    output = evaluation / (arm + ".json")
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    payload = dict(arm=arm, fingerprint=fingerprint, results=[])
    if output.exists():
        payload = json.loads(output.read_text())
        if payload["fingerprint"] != fingerprint or payload["arm"] != arm:
            raise ValueError("Existing results belong to different evaluation settings.")
    results = payload["results"]
    if [r["id"] for r in results] != [q["id"] for q in questions[:len(results)]]:
        raise ValueError("Existing results do not match the frozen question order.")
    if len(results) == len(questions):
        print(f"[{arm}] All {len(results)} questions already completed.", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU runtime before evaluating.")
    seed = manifest["settings"]["test_seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[{arm}] Loading model on {torch.cuda.get_device_name(0)} ({dtype}).", flush=True)
    common = dict(revision=config["model_revision"], trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], **common)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    checkpoint = Path(manifest["settings"]["checkpoint"]) if arm != "base" else None
    adapter = arm != "base" and (checkpoint / "adapter_config.json").is_file()
    source = config["model_id"] if arm == "base" or adapter else str(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        source, torch_dtype=dtype, attn_implementation="sdpa",
        **(common if arm == "base" or adapter else {"trust_remote_code": False}))
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
    model.to("cuda").eval()
    # Fresh greedy config avoids inherited temperature/top-p warning flags.
    generation = GenerationConfig(
        max_new_tokens=maximum, do_sample=False, num_beams=1,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, use_cache=True,
    )
    # Validate all prompts and gold answers before expensive generation.
    prompts = []
    for question in questions:
        ids = prompt_ids(tokenizer, question["problem"], config)
        if len(ids) > config.get("max_prompt_tokens", 1024):
            raise ValueError(f"Prompt {question['id']} has {len(ids)} tokens; no silent truncation. "
                             "Increase max_prompt_tokens consistently in a new evaluation if needed.")
        target = parse("\\boxed{" + question["answer"] + "}", extraction_config=[LatexExtractionConfig()])
        if not target:
            raise ValueError("Unparseable gold answer: " + question["id"])
        prompts.append(ids)
    initial_count = len(results)
    started = time.perf_counter()
    for index in range(initial_count, len(questions)):
        question = questions[index]
        ids = torch.tensor([prompts[index]], dtype=torch.long, device="cuda")
        print(f"[{arm}] Generating question {index+1}/{len(questions)}: {question['id']}", flush=True)
        tick = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                       generation_config=generation)
        response = generated[0, ids.shape[1]:].tolist()
        text = tokenizer.decode(response, skip_special_tokens=True)
        correct, grading_error = grade(text, question["answer"])
        ended_with_eos = bool(response and response[-1] == tokenizer.eos_token_id)
        result = dict(
            **question, text=text, extracted_answer=boxed_answer(text), correct=correct,
            tokens=len(response), hit_token_limit=len(response) >= maximum,
            truncated=len(response) >= maximum and not ended_with_eos,
            ended_with_eos=ended_with_eos, grading_error=grading_error,
            seconds=time.perf_counter()-tick,
        )
        results.append(result)
        payload.update(completed=len(results), total=len(questions),
                       accuracy=sum(r["correct"] for r in results)/len(results),
                       mean_tokens=sum(r["tokens"] for r in results)/len(results))
        atomic_json(output, payload)
        elapsed = time.perf_counter()-started
        eta = elapsed/(len(results)-initial_count)*(len(questions)-len(results))
        print(f"[{arm}] {len(results)}/{len(questions)} | this_question_correct={int(correct)} "
              f"| tokens={len(response)} | truncated={result['truncated']} "
              f"| elapsed={elapsed/60:.1f}m | ETA={eta/60:.1f}m", flush=True)
    print(f"[{arm}] Complete: {sum(r['correct'] for r in results)}/{len(results)} correct. Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
