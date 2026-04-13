import subprocess

model_list = [
    "ultra_scan_net",
]

dataset_list = [
    "/home/alexandra/Documents/Datasets/BUSI_split"
]

dataset_names = [
    'BUSI',
]

data_len = [
    3,
]

# ✅ Launch validation runs
for model in model_list:
    for i, dataset_path in enumerate(dataset_list):
        experiment_name = f"{model}_{dataset_names[0]}"
        args_path = f"/home/alexandra/Documents/MambaVision/mambavision/output_busi/{experiment_name}/args.yaml" 
        checkpoint_path = f"/home/alexandra/Documents/MambaVision/mambavision/output_busi/{experiment_name}/checkpoint-113.pth.tar"

        cmd = [
            "python3", "val_simple.py",
            "-c", args_path,
            "--loadcheckpoint", checkpoint_path,
            "--infer_only"
        ]

        print(f"\n✅ Validating: {experiment_name}\n")
        subprocess.run(cmd)

# python3 val_simple.py -c /home/alexandra/Documents/MambaVision/best-model-article-1/args.yaml --loadcheckpoint /home/alexandra/Documents/MambaVision/best-model-article-1/model_best.pth.tar --validate_only