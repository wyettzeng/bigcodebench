from fire import Fire
from bigcodebench.evaluate import evaluate
from bigcodebench.generate import run_codegen

def main(
        model: str,
        split: str ="instruct",
        subset:str = "hard",
        n_samples: int= 16,
        do_eval: bool = False
    ):
    if n_samples == 1:
        evaluate(
            split=split,
            subset=subset,
            n_samples=n_samples,
            model=model,
            backend="vllm",
        )
    elif not do_eval:
        run_codegen(
            split=split,
            subset=subset,
            n_samples=n_samples,
            model=model,
            backend="vllm",
            temperature=1.0
        )
    else:
        # we evaluate locally as it errored out with API
        revision: str = "main"
        extra = "-" + subset if subset != "full" else ""
        backend = "vllm"
        temperature = 0 if n_samples == 1 else 1.0
        identifier = model.replace("/", "--") + f"--{revision}--bigcodebench{extra}-{split}--{backend}-{temperature}-{n_samples}-sanitized_calibrated.jsonl"
        inf_path = "bcb_results/" + identifier

        evaluate(samples=inf_path, split=split, subset=subset, 
                 local_execute=True
                 )

if __name__ == "__main__":
    Fire(main)
