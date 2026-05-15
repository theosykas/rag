from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto"
        )

    def generate(self, query: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the question "
                "based only on the provided context. Be concise.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context[:400]}\n\nQuestion: {query}"
            },
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
        model_input = self.tokenizer(
            [text], return_tensors="pt").to(self.model.device)
        output = self.model.generate(**model_input, max_new_tokens=200)
        output_ids = output[0][len(model_input.input_ids[0]):]
        return self.tokenizer.decode(
            output_ids, skip_special_tokens=True).strip()
