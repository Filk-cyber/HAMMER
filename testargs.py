import argparse
def setup_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dataset
    parser.add_argument("--a", type=str, default="llama3")
    parser.add_argument("--idea", type=bool, default=True)
    parser.add_argument("--ablation_study", action="store_true",
                        help="Run ablation study comparing all three scoring functions")
    args = parser.parse_args()
    return args
args=setup_parser()
print(args.a)
print(type(args.a))
print(type(args.idea))
cc=False
cc=args.ablation_study
print("=============")
print(cc)
