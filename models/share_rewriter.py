# models/share_rewriter.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

DEFAULT_SHARE_MODEL = "google/flan-t5-small"  # small, fast; replace with custom fine-tuned model when available

class SHARERewriter:
    def __init__(self, model_name: str = DEFAULT_SHARE_MODEL, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def rewrite_instructions(self, original_instructions: str, substitutions: dict) -> str:
        """
        substitutions: dict mapping original ingredient -> substituted ingredient
        Returns rewritten instructions text.
        Approach: Prompt the seq2seq model to rewrite instructions replacing ingredient mentions
        and updating cooking verbs if necessary.
        """
        # Build a short instruction to the model
        subs_lines = "; ".join([f"{o} -> {s}" for o, s in substitutions.items()])
        prompt = (
            "Rewrite the following recipe instructions to match the substitutions. "
            f"Substitutions: {subs_lines}. "
            "If a substitution changes the method (e.g., from dough to batter), adapt the verbs and times accordingly. "
            "Do not change portions unless necessary. Output only the rewritten instructions.\n\n"
            f"Instructions:\n{original_instructions}\n\nRewritten:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text

