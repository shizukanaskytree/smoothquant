import argparse
from huggingface_hub import HfApi, create_repo

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None,
                        help='local model path')
    parser.add_argument('--hf-repo-id', type=str, default=None,
                        help='HF Hub ID, e.g., skytree/smoothquant-models, if None, then we do not upload to HF hub')
    args = parser.parse_args()
    return args

### 上传到 HF hub 很慢.
def main():
    args = parse_args()
    ### tutorial: https://huggingface.co/docs/huggingface_hub/guides/repository
    try:
        create_repo(args.hf_repo_id)
    except:
        print("Repo already created.")

    api = HfApi()
    api.upload_folder(folder_path=args.loal_model_path, repo_id=args.hf_repo_id)
    print(f"{args.model_path} is uploaded to {args.hf_repo_id}")

if __name__ == '__main__':
    main()
