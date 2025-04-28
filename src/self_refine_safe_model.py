from __future__ import annotations
from typing import Dict, Any

# Lazy loader for the HF pipeline

def _load_backbone(model_name: str = "meta-llama/Meta-Llama-3.1-8B",
                   **hf_kwargs) -> "transformers.Pipeline":
    from transformers import pipeline
    defaults = dict(model=model_name, device_map="auto", trust_remote_code=True)
    defaults.update(hf_kwargs)
    return pipeline("text-generation", **defaults)


class SelfRefineSafeModel:
    """
    Self-Refine wrapper using Meta-Llama-3-8B-Instruct for toxicity reduction.
    Exposes .device, .config, .eval(), .to(), .generate(), and supports both self-refine generation
    and HF-perplexity APIs.
    """
    INITIAL_PROMPT = (
        "You are a respectful, helpful assistant.\n"
        "When answering the USER, **do not use toxic, hateful, harassing or demeaning language**.\n"
        "Always remain polite, fact-based and inclusive.\n\n"
        "USER: {query}\n"
        "ASSISTANT:"
    )

    FEEDBACK_PROMPT = (
        "You are a *feedback assistant* checking the ASSISTANT'S answer for toxicity.\n"
        "Return a short JSON dict with keys:\n"
        "  - \"toxic\"  (true/false)\n"
        "  - \"issues\" (list of strings explaining toxic spans or risk factors)\n"
        "  - \"advice\" (single string describing how the answer could be made non-toxic)\n\n"
        "QUESTION: {query}\n\n"
        "ASSISTANT_ANSWER:\n{answer}\n\n"
        "FEEDBACK:"
    )

    REFINE_PROMPT = (
        "You are the original assistant, revising your previous answer to remove any toxicity.\n"
        "Consider the feedback below and produce a new, improved, non-toxic answer.\n"
        "Only output the **final answer** – do not repeat the feedback or analysis.\n\n"
        "QUESTION: {query}\n\n"
        "PREVIOUS_ANSWER:\n{answer}\n\n"
        "FEEDBACK_JSON:\n{feedback}\n\n"
        "REVISED_ANSWER:"
    )

    def __init__(self,
                 backbone: "transformers.Pipeline | None" = None,
                 generation_kwargs: Dict[str, Any] | None = None):
        # Load or reuse the text-generation pipeline
        self.llm = backbone or _load_backbone()
        self.generation_kwargs = generation_kwargs or dict(
            max_new_tokens=512, temperature=0.7, top_p=0.95
        )

        # Expose a torch.device for compatibility with evaluation scripts
        import torch
        pipe_dev = getattr(self.llm, "device", None)
        if isinstance(pipe_dev, int):
            self.device = torch.device(f"cuda:{pipe_dev}") if pipe_dev >= 0 else torch.device("cpu")
        elif isinstance(pipe_dev, str):
            self.device = torch.device(pipe_dev)
        elif isinstance(pipe_dev, torch.device):
            self.device = pipe_dev
        else:
            self.device = torch.device("cpu")

        # Expose model config for template selection or metadata
        model_obj = getattr(self.llm, "model", None)
        self.config = model_obj.config if (model_obj and hasattr(model_obj, "config")) else None

    def eval(self):
        """
        Put underlying model into eval mode for compatibility.
        """
        try:
            transformer = getattr(self.llm, "model", None)
            if transformer is not None and hasattr(transformer, "eval"):
                transformer.eval()
        except Exception:
            pass
        return self

    def to(self, device: "torch.device | str") -> SelfRefineSafeModel:
        """
        Move the underlying transformer model to the specified device.
        """
        import torch
        dev = torch.device(device) if isinstance(device, (str,)) else device
        transformer = getattr(self.llm, "model", None)
        if transformer is not None and hasattr(transformer, "to"):
            transformer.to(dev)
        self.device = dev
        return self

    def generate(self,
                 input_ids: "torch.LongTensor",
                 attention_mask: "torch.Tensor" | None = None,
                 token_type_ids: "torch.Tensor" | None = None,
                 position_ids: "torch.Tensor" | None = None,
                 generation_config: "transformers.generation.GenerationConfig" | None = None,
                 **kwargs) -> "torch.LongTensor":
        """
        Delegate to the raw transformer generate for direct sequence generation.
        Filters out None kwargs to avoid passing token_type_ids=None.
        """
        transformer = getattr(self.llm, "model", None)
        if transformer is None or not hasattr(transformer, "generate"):
            raise AttributeError("No underlying transformer model for generation.")
        # Build arguments, excluding None values
        gen_kwargs: Dict[str, Any] = {}
        if input_ids is not None:
            gen_kwargs["input_ids"] = input_ids
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            gen_kwargs["token_type_ids"] = token_type_ids
        if position_ids is not None:
            gen_kwargs["position_ids"] = position_ids
        if generation_config is not None:
            gen_kwargs["generation_config"] = generation_config
        gen_kwargs.update(kwargs)
        return transformer.generate(**gen_kwargs)

    def __call__(self, *args, n_feedback_rounds: int = 1, **kwargs) -> Any:
        # HF-perplexity mode: delegate directly to raw transformer to return loss & logits
        if 'input_ids' in kwargs:
            transformer = getattr(self.llm, "model", None)
            if transformer is None:
                raise AttributeError("No underlying transformer model for HF-perplexity calculation.")
            return transformer(**kwargs)

        # Self-refine generation mode: first positional arg is the user query
        query = args[0] if args else kwargs.get('query', '')
        answer = self._initial_answer(query)
        for _ in range(n_feedback_rounds):
            feedback = self._criticise(query, answer)
            if feedback.get("toxic") is False:
                break
            answer = self._refine(query, answer, feedback)
        return answer.strip()

    def _generate(self, prompt: str) -> str:
        return self.llm(prompt, **self.generation_kwargs)[0]["generated_text"][len(prompt):]

    def _initial_answer(self, query: str) -> str:
        return self._generate(self.INITIAL_PROMPT.format(query=query))

    def _criticise(self, query: str, answer: str) -> Dict[str, Any]:
        from json import loads
        import re
        raw = self._generate(self.FEEDBACK_PROMPT.format(query=query, answer=answer))
        try:
            j = re.search(r"\{.*\}", raw, re.S).group(0)
            return loads(j)
        except Exception:
            return {"toxic": True,
                    "issues": ["parse_error"],
                    "advice": "Rewrite in respectful language."}

    def _refine(self, query: str, answer: str, feedback: Dict[str, Any]) -> str:
        from json import dumps
        prompt = self.REFINE_PROMPT.format(
            query=query,
            answer=answer,
            feedback=dumps(feedback, ensure_ascii=False, indent=2)
        )
        return self._generate(prompt)


def load_safe_model() -> SelfRefineSafeModel:
    """
    Entry-point expected by the evaluation pipeline.
    Returns a callable with .device, .config, .eval(), .to(), .generate(), and supports both generation
    and HF-perplexity APIs.
    """
    return SelfRefineSafeModel()
