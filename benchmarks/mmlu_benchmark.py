import argparse
import json
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from graph.workflow import run_security_workflow

CHOICE_LABELS = ["A", "B", "C", "D"]
SAMPLE_SIZE = 30
RANDOM_SEED = 42


def make_prompt(question: str, choices: list[str]) -> str:
    # 모델이 선택지 문자 하나만 내도록 강하게 고정
    return (
        "다음 객관식 문제의 정답을 고르세요.\n"
        "설명 없이 정답 문자 하나만 출력하세요. (A/B/C/D)\n\n"
        f"문제: {question}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
    )


def extract_choice(text: str) -> str | None:
    # 답변 본문에서 A/B/C/D 하나를 추출
    if not text:
        return None
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else None


def load_mmlu(split: str):
    # MMLU 데이터셋 로드 후 랜덤 시드 고정으로 30개만 샘플링
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets 패키지가 필요합니다. `pip install datasets` 후 다시 실행해 주세요."
        ) from exc

    dataset = load_dataset("cais/mmlu", "all", split=split)
    records = list(dataset)
    random.Random(RANDOM_SEED).shuffle(records)
    return records[: min(SAMPLE_SIZE, len(records))]


def evaluate_mmlu(split: str, output_path: str, verbose: bool = True) -> dict:
    records = list(load_mmlu(split))
    correct = 0
    total = len(records)
    results = []

    # 1) 문제를 에이전트 워크플로우에 넣고 2) 정답률 계산
    for idx, row in enumerate(records, start=1):
        gold = CHOICE_LABELS[int(row["answer"])]
        prompt = make_prompt(row["question"], row["choices"])

        try:
            route, model_output = run_security_workflow(prompt)
            pred = extract_choice(model_output)
        except Exception as exc:  # noqa: BLE001
            route, model_output = "error", f"ERROR: {type(exc).__name__}: {exc}"
            pred = None
        is_correct = pred == gold

        if is_correct:
            correct += 1

        results.append(
            {
                "index": idx,
                "subject": row["subject"],
                "question": row["question"],
                "choices": row["choices"],
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "route": route,
                "model_output": model_output,
            }
        )

        if verbose and (idx % 10 == 0 or idx == len(records)):
            acc = correct / idx if idx else 0.0
            print(f"[{idx}/{len(records)}] accuracy={acc:.4f}")

    # 최종 리포트 출력
    final_acc = correct / total if total else 0.0
    if verbose:
        print("\n=== MMLU Benchmark Result ===")
        print(f"split    : {split}")
        print(f"samples  : {total}")
        print(f"correct  : {correct}")
        print(f"accuracy : {final_acc:.4f}")

    # 샘플별 결과 저장(JSON)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"saved_json: {out}")

    return {
        "split": split,
        "samples": total,
        "correct": correct,
        "accuracy": final_acc,
        "output_path": str(out),
        "results": results,
    }


def main() -> None:
    # 실행 옵션도 최소화: split/출력파일만 받음 (샘플은 30개 고정)
    parser = argparse.ArgumentParser(description="간단한 MMLU 벤치마크")
    parser.add_argument("--split", default="test", help="예: test, validation")
    parser.add_argument(
        "--output",
        default="benchmarks/results/mmlu_results.json",
        help="결과 JSON 저장 경로",
    )
    args = parser.parse_args()

    load_dotenv()
    evaluate_mmlu(split=args.split, output_path=args.output, verbose=True)


if __name__ == "__main__":
    main()
