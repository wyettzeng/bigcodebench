from fire import Fire
from bigcodebench.evaluate import evaluate
from bigcodebench.generate import run_codegen
import os
def main(
        model: str,
        split: str ="instruct",
        subset:str = "hard",
        n_samples: int= 16,
        do_eval: bool = False,
        **generation_kwargs,
    ):
    print(f"Starting {model} - {split} - {subset} - {n_samples} - do eval {do_eval}")
    revision: str = "main"
    extra = "-" + subset if subset != "full" else ""
    backend = "vllm"
    temperature = 0 if n_samples == 1 else 1.0
    if n_samples == 1:
        out_file_name = model.replace("/", "--") + f"--{revision}--bigcodebench{extra}-{split}--{backend}-{temperature}-{n_samples}-sanitized_calibrated_eval_results.json"
        out_file_path = "bcb_results/" + out_file_name

        if os.path.exists(out_file_path):
            return # already done
        
        evaluate(
            split=split,
            subset=subset,
            n_samples=n_samples,
            model=model,
            backend="vllm",
            **generation_kwargs,
        )
    elif not do_eval:
        identifier = model.replace("/", "--") + f"--{revision}--bigcodebench{extra}-{split}--{backend}-{temperature}-{n_samples}-sanitized_calibrated.jsonl"
        inf_path = "bcb_results/" + identifier
        if os.path.exists(inf_path):
            print(f"Skipping {model} - {split} - {subset} - {n_samples} - do eval {do_eval} as file {inf_path} exists")
            return

        run_codegen(
            split=split,
            subset=subset,
            n_samples=n_samples,
            model=model,
            backend="vllm",
            temperature=1.0,
            **generation_kwargs,
        )
    else:
        # we evaluate locally as it errored out with API
        out_file_name = model.replace("/", "--") + f"--{revision}--bigcodebench{extra}-{split}--{backend}-{temperature}-{n_samples}-sanitized_calibrated_eval_results.json"
        out_file_path = "bcb_results/" + out_file_name

        if os.path.exists(out_file_path):
            print(f"Skipping {model} - {split} - {subset} - {n_samples} - do eval {do_eval} as file {out_file_path} exists")
            return # already done

        identifier = model.replace("/", "--") + f"--{revision}--bigcodebench{extra}-{split}--{backend}-{temperature}-{n_samples}-sanitized_calibrated.jsonl"
        inf_path = "bcb_results/" + identifier
        evaluate(samples=inf_path, split=split, subset=subset, 
                 local_execute=True,
                 parallel=16,
                 )

if __name__ == "__main__":
    Fire(main)
