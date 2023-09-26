from huggingface_hub import HfApi
# from huggingface_hub import create_repo

### https://huggingface.co/docs/huggingface_hub/guides/repository
repo_id = "skytree/smoothquant-models"
# create_repo(repo_id) # only run the first time

output_path = "/workspace/outside-docker/smoothquant-prj/smoothquant/examples/int8_models/opt-125m-smoothquant.pt"

api = HfApi()
api.upload_folder(
    folder_path=output_path,
    repo_id=repo_id,
)
