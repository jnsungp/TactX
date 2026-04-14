import os
import torch
from transformers import AutoTokenizer, T5EncoderModel


class T5Embedder:
    TOKENIZER_MAX_LENGTH = 1024

    def __init__(
        self,
        device,
        from_pretrained=None,  # local path
        *,
        cache_dir=None,
        hf_token=None,
        use_text_preprocessing=True,
        t5_model_kwargs=None,
        torch_dtype=None,
        use_offload_folder=None,
        model_max_length=120,
        local_files_only=False,
    ):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.cache_dir = cache_dir

        if t5_model_kwargs is None:
            t5_model_kwargs = {
                "low_cpu_mem_usage": True,
                "torch_dtype": self.torch_dtype,
            }

            if use_offload_folder is not None:
                t5_model_kwargs["offload_folder"] = use_offload_folder
                t5_model_kwargs["device_map"] = {
                    "shared": self.device,
                    "encoder.embed_tokens": self.device,
                    "encoder.block.0": self.device,
                    "encoder.block.1": self.device,
                    "encoder.block.2": self.device,
                    "encoder.block.3": self.device,
                    "encoder.block.4": self.device,
                    "encoder.block.5": self.device,
                    "encoder.block.6": self.device,
                    "encoder.block.7": self.device,
                    "encoder.block.8": self.device,
                    "encoder.block.9": self.device,
                    "encoder.block.10": self.device,
                    "encoder.block.11": self.device,
                    "encoder.block.12": "disk",
                    "encoder.block.13": "disk",
                    "encoder.block.14": "disk",
                    "encoder.block.15": "disk",
                    "encoder.block.16": "disk",
                    "encoder.block.17": "disk",
                    "encoder.block.18": "disk",
                    "encoder.block.19": "disk",
                    "encoder.block.20": "disk",
                    "encoder.block.21": "disk",
                    "encoder.block.22": "disk",
                    "encoder.block.23": "disk",
                    "encoder.final_layer_norm": "disk",
                    "encoder.dropout": "disk",
                }
            else:
                t5_model_kwargs["device_map"] = {
                    "shared": self.device,
                    "encoder": self.device,
                }

        self.use_text_preprocessing = use_text_preprocessing
        self.hf_token = hf_token

        # assert "t5-v1_1-xxl" in from_pretrained
        # model_path = os.path.join(from_pretrained, "pytorch_model.bin")
        # import ipdb; ipdb.set_trace()
        # assert os.path.exists(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            from_pretrained,
            model_max_length=model_max_length,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.model = T5EncoderModel.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **t5_model_kwargs,
        ).eval()
        self.model_max_length = model_max_length

    def get_text_embeddings(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding="longest",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = text_tokens_and_mask["input_ids"].to(self.device)
        attention_mask = text_tokens_and_mask["attention_mask"].to(self.device)
        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )["last_hidden_state"].detach()
        return text_encoder_embs, attention_mask

if __name__ == '__main__':
    import torch
    import yaml
# Use a pipeline as a high-level helper
    from transformers import pipeline

    # pipe = pipeline("text2text-generation", model="google/t5-v1_1-xxl")
    # Load model directly

    # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    # tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl")
    # model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-v1_1-xxl")

    GPU = 0
    # MODEL_PATH = "./data/pretrained_weights/t5-v1_1-xxl"
    MODEL_PATH = "google/t5-v1_1-xxl"
    SAVE_DIR = "/tmp/"

    # Modify this to your task name and instruction
    TASK_NAME = "coke_zero_pouring"
    INSTRUCTION = "grasp a red coke zero can with the right hand, grasp a gray plastic cup with the left hand. pour the coke zero into the cup and then place them back"

    print(os.path.join(SAVE_DIR, f"{TASK_NAME}.pt"))
    # import ipdb; ipdb.set_trace()

    # Note: if your GPU VRAM is less than 24GB, 
    # it is recommanded to enable offloading by specifying an offload directory.
    OFFLOAD_DIR = None  # Specify your offload directory here, ensuring the directory exists.
        
    device = torch.device(f"cuda:{GPU}")
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH, 
        model_max_length=T5Embedder.TOKENIZER_MAX_LENGTH,
        device=device,
        use_offload_folder=OFFLOAD_DIR
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    tokens = tokenizer(
        INSTRUCTION, return_tensors="pt",
        padding="longest",
        truncation=True
    )["input_ids"].to(device)

    tokens = tokens.view(1, -1)
    with torch.no_grad():
        pred = text_encoder(tokens).last_hidden_state.detach().cpu()
    
    assert pred.shape[0] == 1
    pred = pred.reshape(pred.shape[1:])
    
    save_path = os.path.join(SAVE_DIR, f"{TASK_NAME}.pt")
    # We save the embeddings in a dictionary format
    torch.save({
            "name": TASK_NAME,
            "instruction": INSTRUCTION,
            "embeddings": pred
        }, save_path
    )
    
    print(f'\"{INSTRUCTION}\" from \"{TASK_NAME}\" is encoded by \"{MODEL_PATH}\" into shape {pred.shape} and saved to \"{save_path}\"')
