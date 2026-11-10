from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="lmzheng/lmsys-chat-1m",
    filename="data",
    repo_type="dataset"
)

from datasets import load_dataset

dataset = load_dataset("lmzheng/lmsys-chat-1m")
print(dataset)
print(dataset["train"][0])
dataset.save_to_disk("lmsys-chat-1m")