from fire import Fire
from bigcodebench.evaluate import evaluate
from bigcodebench.generate import run_codegen
import os
def main(
        model: str,
        split: str ="instruct",
        subset:str = "hard",
    ):
    print(f"Starting {model} - {split} - {subset}")
    revision: str = "main"
    extra = "-" + subset if subset != "full" else ""
    backend = "vllm"
    temperature = 0

    # we evaluate locally as it errored out with API
    out_file_name = f"{model}--{revision}--bigcodebench{extra}-{split}--{backend}-{temperature}-1-sanitized_calibrated_eval_results.json"
    out_file_path = "haozhe/" + out_file_name

    if os.path.exists(out_file_path):
        return # already done

    identifier = f"{model}--{revision}--bigcodebench{extra}-{split}--{backend}-{temperature}-1-sanitized_calibrated.jsonl"
    inf_path = "haozhe/inference/" + identifier
    evaluate(samples=inf_path, split=split, subset=subset, 
                # local_execute=True
                )

if __name__ == "__main__":
    Fire(main)
