#Feed anchor-word-interpolated VA trajectories to a frozen LLM as soft tokens (raw embedding vectors)

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer #HF

from musicbrain.anchors import AnchorSet, interpolation_weights


class FrozenGenerationLLM:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False) #freeze

    def embed_text(self, text: str) -> torch.Tensor:
        """Token embeddings for *text*, shape (1, n_tokens, hidden)."""
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        ids = ids.to(self.device)
        return self.model.get_input_embeddings()(ids) #forward pass on ids

    def word_embedding(self, word: str) -> torch.Tensor:
        """A single embedding vector for *word*.

        Anchor words may tokenize to more than one sub-word piece; when
        that happens this averages the piece embeddings rather than
        picking one arbitrarily, so every anchor is represented by exactly
        one vector for the interpolation in Section 3.3.
        """
        # the leading space matches how this word tokenizes mid-sentence
        # rather than as the first token of a document (see anchors.py discussion)
        ids = self.tokenizer(f" {word}", return_tensors="pt", add_special_tokens=False).input_ids
        ids = ids.to(self.device)
        embeds = self.model.get_input_embeddings()(ids)  # (1, n_pieces, hidden)
        return embeds.mean(dim=1)  # (1, hidden)

    def anchor_embeddings(self, anchor_set: AnchorSet) -> torch.Tensor:
        """Stack of real, in-vocabulary embeddings, one row per anchor word."""
        rows = [self.word_embedding(w) for w in anchor_set.words]
        return torch.cat(rows, dim=0)  # (n_anchors, hidden)

    def interpolate_soft_tokens(
        self, query_va: np.ndarray, anchor_set: AnchorSet, temperature: float = 0.3
    ) -> torch.Tensor:
        """
        Returns shape (n_windows, hidden) -- never a discretely chosen word
        embedding, always a blend
        """
        weights = interpolation_weights(query_va, anchor_set.va, temperature=temperature)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)
        anchor_embeds = self.anchor_embeddings(anchor_set)  # (n_anchors, hidden)
        return weights_t @ anchor_embeds  # (n_windows, hidden)

    def generate_from_soft_trajectory(
        self,
        query_va: np.ndarray,
        anchor_set: AnchorSet,
        prefix_text: str,
        suffix_text: str = "",
        temperature: float = 0.3,
        max_new_tokens: int = 60,
    ) -> str:
        soft_embeds = self.interpolate_soft_tokens(query_va, anchor_set, temperature)
        soft_embeds = soft_embeds.unsqueeze(0).to(self.model.dtype)  # (1, T, hidden)

        prefix_embeds = self.embed_text(prefix_text)
        parts = [prefix_embeds, soft_embeds]
        if suffix_text:
            parts.append(self.embed_text(suffix_text))
        inputs_embeds = torch.cat(parts, dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device) #default mask for no padding

        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False, #deterministic
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # generate() returns only the newly produced tokens when the prompt
        # was supplied as inputs_embeds (no input_ids to prepend).
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
