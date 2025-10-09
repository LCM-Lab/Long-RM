
import os, argparse
from evaluating.longreward_bench import LongRewardBenchEvaluator



def evaluate(args):

    model_name = os.path.basename(args.model_path.rstrip("/"))

    if "70b" in model_name.lower() or "72b" in model_name.lower(): 
        batch_size = 500
        tp_size = 8
    elif '13b' in model_name.lower() or '14b' in model_name.lower(): 
        batch_size = 5
        tp_size = 2
    else: 
        batch_size = 5
        tp_size = 1
        
    evaluator = LongRewardBenchEvaluator(
        backend = 'vllm_gpu_batch',
        tasks = ["LongQA", "Summ", "Safety", "ICL", "Cite", "Code", "Math"],
        formats = ["PairWise", "BestOfN"],
        batch_size = batch_size,
        tp_size = tp_size,
        is_disrm = args.is_disrm,
        model_name = args.model_path,
        data_path = args.data_path,
    )

    save_path = f"{os.path.dirname(os.path.abspath(__file__))}/results/{model_name}" \
    if args.save_path is None else args.save_path

    evaluator.inference(save_path = save_path, 
                        do_evalutate = True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type = str, required = True
    )
    parser.add_argument(
        "--is-disrm", action="store_true", default = False
    )
    parser.add_argument(
        "--data-path", type = str, required = True,
        help = "The LongReward Benchmark Data Path"
    )
    parser.add_argument(
        "--save-path", type = str, default = None,
        help = "The Evaluating Results Saving path"
    )
    
    parser.add_argument(
        "--gpus",type = int, nargs = "+", default = [0, 1, 2, 3, 4, 5, 6, 7]
    )


    args = parser.parse_args()
    evaluate(args)
        