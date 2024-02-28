from transformers import AutoModel, AutoTokenizer
import torch
from pathlib import Path

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

output_dir = Path("onnx_model")
output_dir.mkdir(exist_ok=True, parents=True)
output_model_path = output_dir / "model.onnx"

input_ids = torch.ones((1, 128), dtype=torch.int64)  # Example size, adjust as needed
attention_mask = torch.ones_like(input_ids)

torch.onnx.export(model, 
                  args=(input_ids, attention_mask),
                  f=str(output_model_path),
                  input_names=['input_ids', 'attention_mask'],
                  output_names=['output'],
                  dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},
                                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                                'output': {0: 'batch_size', 1: 'sequence'}},
                  do_constant_folding=True,
                  opset_version=12)  # Usex a higher opset version, e.g., 12 or higher

tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")
