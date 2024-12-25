from fire import Fire
import os
import sys

# Get the directory of the current script
# current_directory = os.path.dirname(os.path.abspath(__file__))
# print(current_directory)
# import pdb; pdb.set_trace()
# current_directory = "/home/ma-user/work/haozhe/code/math_workflow/Mammoth-code/aapublic/evaluation_for_codeRL/bigcodebench"
# # Add it to sys.path
# if current_directory not in sys.path:
#     sys.path.append(current_directory)
base_out = "/home/ma-user/work/haozhe/workspace/aacodeRL/evaluation_for_codeRL/bigcodebench/bcb_results/"
from bigcodebench.evaluate import evaluate
from bigcodebench.generate import run_codegen
import os
def main(
        model: str,
        split: str ="instruct",
        subset:str = "hard",
        n_samples: int= 16,
        do_eval: bool = False
    ):
    print(f"Starting {model} - {split} - {subset} - {n_samples} - do eval {do_eval}")
    revision: str = "main"
    extra = "-" + subset if subset != "full" else ""
    backend = "vllm"
    temperature = 0 if n_samples == 1 else 1.0
    modelname = model.split(os.path.sep)[-1]
    if do_eval:
        # we evaluate locally as it errored out with API
        # out_file_name = model.replace("/", "--") + f"--{revision}--bigcodebench{extra}-{split}--{backend}-{temperature}-{n_samples}-sanitized_calibrated_eval_results.json"
        # out_file_path = base_out + out_file_name

        # if os.path.exists(out_file_path):
        #     return # already done

        identifier = model.split("/")[-1].replace("/", "--") + f"--{revision}--bigcodebench{extra}-{split}--{backend}-{temperature}-{n_samples}-sanitized_calibrated.jsonl"
        inf_path = base_out + identifier
        out_file_path = inf_path.replace("calibrated.jsonl", "calibrated_eval_results.jsonl")
        # if os.path.exists(out_file_path):
        #     return # already done
        # import pdb; pdb.set_trace()
        evaluate(samples=inf_path, split=split, subset=subset, 
                 local_execute=True
                 )
    elif n_samples == 1: # greedy
        identifier = model.split("/")[-1].replace("/", "--") + f"--{revision}--bigcodebench{extra}-{split}--{backend}-{temperature}-{n_samples}-sanitized_calibrated.jsonl"
        inf_path = base_out + identifier
        out_file_path = inf_path.replace("calibrated.jsonl", "calibrated_eval_results.jsonl")
        # if os.path.exists(out_file_path):
        #     return # already done
        
        evaluate(
            split=split,
            subset=subset,
            n_samples=n_samples,
            model=model,
            backend="vllm",
            local_execute=True
            
        )
    else:
        identifier = model.split("/")[-1].replace("/", "--") + f"--{revision}--bigcodebench{extra}-{split}--{backend}-{temperature}-{n_samples}-sanitized_calibrated.jsonl"
    
        inf_path = base_out + identifier
        if os.path.exists(inf_path):
            return

        run_codegen(
            split=split,
            subset=subset,
            n_samples=n_samples,
            model=model,
            backend="vllm",
            temperature=1.0
        )
        

if __name__ == "__main__":
    Fire(main)
