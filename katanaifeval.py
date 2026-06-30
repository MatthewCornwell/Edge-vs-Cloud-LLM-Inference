import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from ifeval import Evaluator, instruction_registry, read_input_examples

csv.field_size_limit(min(sys.maxsize, 2147483647))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="ifeval_subset.jsonl")
    ap.add_argument("--results", required=True, help="benchmark.py results CSV")
    ap.add_argument("--out", help="output CSV path (default: <results>.graded.csv)")
    args = ap.parse_args()

    input_examples = read_input_examples(args.prompts)

    for ex in input_examples:
        ex.kwargs = [
            {k: v for k, v in (kw or {}).items() if v is not None}
            for kw in (ex.kwargs or [])
        ]

    examples_by_prompt = {ex.prompt: ex for ex in input_examples}
    rows = []
    row_index_by_prompt = defaultdict(list)
    with open(args.results, "r", newline="", encoding="utf-8") as f_in:
        rd = csv.DictReader(f_in)
        fieldnames = rd.fieldnames
        for i, row in enumerate(rd):
            rows.append(row)
            prompt_text = row.get("prompt_text") or ""
            if prompt_text and row.get("response_text"):
                row_index_by_prompt[prompt_text].append(i)

    evaluator = Evaluator(instruction_registry)

    extra = [
        "ifeval_strict_pass",
        "ifeval_loose_pass",
        "ifeval_instructions_followed",
        "ifeval_instruction_ids",
    ]


    n_total = n_strict = n_loose = 0
    by_label = {}
    by_label_mode = {}

    for row in rows:
        prompt_text = row.get("prompt_text") or ""
        response_text = row.get("response_text") or ""
        ex = examples_by_prompt.get(prompt_text)

        if not ex or not response_text:
            for k in extra:
                row[k] = ""
            continue

        report, outputs = evaluator.evaluate([ex], {prompt_text: response_text})
        strict_out = outputs["eval_results_strict"][0]
        loose_out = outputs["eval_results_loose"][0]

        strict_pass = bool(strict_out.follow_all_instructions)
        loose_pass = bool(loose_out.follow_all_instructions)

        row["ifeval_strict_pass"] = strict_pass
        row["ifeval_loose_pass"] = loose_pass
        row["ifeval_instructions_followed"] = json.dumps(
            [bool(x) for x in strict_out.follow_instruction_list]
        )
        row["ifeval_instruction_ids"] = json.dumps(list(ex.instruction_id_list))

        n_total += 1
        lbl = row.get("label", "?")
        mode = row.get("mode", "?")
        d = by_label.setdefault(lbl, [0, 0, 0])
        dm = by_label_mode.setdefault((lbl, mode), [0, 0, 0])
        d[2] += 1
        dm[2] += 1
        if strict_pass:
            n_strict += 1
            d[0] += 1
            dm[0] += 1
        if loose_pass:
            n_loose += 1
            d[1] += 1
            dm[1] += 1

    # Write out
    out = args.out or (str(args.results) + ".graded.csv")
    with open(out, "w", newline="", encoding="utf-8") as f_out:
        wr = csv.DictWriter(f_out, fieldnames=list(fieldnames) + extra)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)

    print(f"\nGraded {n_total} rows")
    if n_total:
        print(
            f"Overall: strict={n_strict}/{n_total} ({100*n_strict/n_total:.1f}%) "
            f"loose={n_loose}/{n_total} ({100*n_loose/n_total:.1f}%)"
        )
    print("\nPer label:")
    for lbl, (s, l, t) in sorted(by_label.items()):
        print(f"  {lbl}: strict={s}/{t} ({100*s/t:.1f}%) loose={l}/{t} ({100*l/t:.1f}%)")
    print("\nPer (label, mode):")
    for (lbl, mode), (s, l, t) in sorted(by_label_mode.items()):
        print(
            f"  {lbl}/{mode}: strict={s}/{t} ({100*s/t:.1f}%) "
            f"loose={l}/{t} ({100*l/t:.1f}%)"
        )
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()