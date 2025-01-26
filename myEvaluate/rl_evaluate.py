from fire import Fire
from bigcodebench.evaluate import evaluate

import os
def main(
        model: str,
        split: str ="instruct",
        subset:str = "hard",
        **generation_kwargs,
    ):
    revision: str = "main"
    extra = "-" + subset if subset != "full" else ""
    backend = "vllm"
    temperature = 0
    out_file_name = model.replace("/", "--") + f"--{revision}--bigcodebench{extra}-{split}--{backend}-{temperature}-sanitized_calibrated_eval_results.json"
    out_file_path = "bcb_results/" + out_file_name

    if os.path.exists(out_file_path):
        print(f"Already finished {model} - {split} - {subset}")
        return # already done
    
    print(f"Starting {model} - {split} - {subset}")
    evaluate(
        split=split,
        subset=subset,
        n_samples=1,
        model=model,
        backend="vllm",
        # local_execute=True,
        **generation_kwargs,
    )


if __name__ == "__main__":
    Fire(main)
