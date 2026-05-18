from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen:
    """. Wrapper class for the local Qwen Causal Language Model.

    Handles tokenization, model loading, hardware device mapping,
    and structured context-grounded text generation using the Hugging Face
    transformers pipeline.
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """. Initializes the Qwen tokenizer and causal language model.

        Args:
            model_name (str): The Hugging Face hub repository identifier
                or local path of the targeted model weights.
                Defaults to "Qwen/Qwen3-0.6B".
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto"
        )

    def generate(self, query: str, context: str) -> str:
        """. Generates a concise response grounded only in the context.

        Constructs a conversation history using a standardized chat template,
        truncates the context string safety window, processes the input tokens,
        and prompts the model to extract and output a faithful answer.

        Args:
            query (str): The question or instruction requested by the user.
            context (str): The raw text or source code extracted from the
                retrieval layer used to answer the query.

        Returns:
            str: The raw generated textual response decoded from the model's
                predicted output token IDs.
        """
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
        decoded_str: str = str(
            self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        )
        return decoded_str
