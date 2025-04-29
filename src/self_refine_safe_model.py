from __future__ import annotations
from typing import Dict, Any, Optional
import logging

# Lazy loader for the HF pipeline

def _load_backbone(model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                   **hf_kwargs) -> "transformers.Pipeline":
    from transformers import pipeline
    defaults = dict(model=model_name, device_map="auto", trust_remote_code=True)
    defaults.update(hf_kwargs)
    return pipeline("text-generation", **defaults)


class SelfRefineSafeModel:
    """
    Self-Refine wrapper using Meta-Llama-3-8B-Instruct for toxicity reduction.
    Logs internal feedback and supports both self-refine generation and HF-perplexity APIs.
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

    def __init__(
        self,
        backbone: "transformers.Pipeline | None" = None,
        generation_kwargs: Dict[str, Any] | None = None,
        log_file: Optional[str] = "self_refine.log",
        save_log: bool = False,
    ):
        # Configure logging
        self.logger = logging.getLogger("SelfRefineSafeModel")
        self.logger.setLevel(logging.DEBUG)
        # Reset handlers
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
        # Add file handler
        if save_log and log_file:
            fh = logging.FileHandler(log_file, mode="w")
            fh.setLevel(logging.DEBUG)
            fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)
        else:
            # no-op handler to suppress warnings
            self.logger.addHandler(logging.NullHandler())
        # Load LLM pipeline
        self.llm = backbone or _load_backbone()
        self.generation_kwargs = generation_kwargs or dict(
            max_new_tokens=512, temperature=0.7, top_p=0.95
        )

        # Device
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

        # Config
        model_obj = getattr(self.llm, "model", None)
        self.config = model_obj.config if (model_obj and hasattr(model_obj, "config")) else None

    def eval(self) -> SelfRefineSafeModel:
        """
        Set underlying transformer to evaluation mode.
        """
        try:
            transformer = getattr(self.llm, "model", None)
            if transformer and hasattr(transformer, "eval"):
                transformer.eval()
                self.logger.debug("Set transformer to eval mode.")
        except Exception as e:
            self.logger.error(f"Eval error: {e}")
        return self

    def to(self, device: "torch.device | str") -> SelfRefineSafeModel:
        """
        Move the underlying transformer to the specified device.
        """
        import torch
        dev = torch.device(device) if isinstance(device, str) else device
        transformer = getattr(self.llm, "model", None)
        if transformer and hasattr(transformer, "to"):
            transformer.to(dev)
            self.logger.debug(f"Moved transformer to {dev}.")
        self.device = dev
        return self

    def generate(
        self,
        input_ids: "torch.LongTensor",
        attention_mask: "torch.Tensor" | None = None,
        generation_config: "transformers.GenerationConfig" | None = None,
        **kwargs
    ) -> "torch.LongTensor":
        """
        Full HuggingFace-style generate with Self-Refine on the continuation.
        1) Sample continuations using the underlying transformer.generate.
        2) Slice off prompt tokens.
        3) Decode continuations and run self-refine loop via __call__ (initial, feedback, refine).
        4) Re-tokenize refined texts and concatenate to original prompt tokens.
        """
        import torch

        # 1) Build generation kwargs from GenerationConfig
        gen_kwargs: Dict[str, Any] = {}
        if generation_config is not None:
            gen_kwargs = generation_config.to_dict()
        gen_kwargs.update(kwargs)

        # 2) Call raw transformer.generate
        transformer = getattr(self.llm, "model", None)
        if transformer is None:
            raise AttributeError("Underlying transformer model not found for generate().")
        raw_ids: torch.LongTensor = transformer.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )

                                # 3) Slice off prompt and sample continuations
        prompt_len = input_ids.shape[1]
        cont_ids = raw_ids[:, prompt_len:]
        tokenizer = getattr(self.llm, "tokenizer", None)
        if tokenizer is None:
            raise AttributeError("Tokenizer not found on pipeline.")
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # Decode continuations to text
        cont_texts = tokenizer.batch_decode(cont_ids, skip_special_tokens=True)
        # Self-refine each continuation via full __call__ (initial -> feedback -> refine)
        refined_texts: list[str] = []
        for cont in cont_texts:
            self.logger.debug(f"Continuation for refine: {cont}")
            # Use the self-refine pipeline to clean up the continuation
            refined = self.__call__(cont)
            self.logger.debug(f"Refined continuation: {refined}")
            refined_texts.append(refined)

        # 4) Tokenize refined texts and concatenate to original prompt
        tokenized = tokenizer(refined_texts, return_tensors="pt", padding=True, truncation=False)
        refined_ids = tokenized.input_ids.to(self.device)
        orig_ids = input_ids.to(self.device)
        full_output = torch.cat([orig_ids, refined_ids], dim=1)
        return full_output
    def __call__(
        self,
        *args,
        n_feedback_rounds: int = 1,
        **kwargs
    ) -> Any:
        # Perplexity mode: return raw transformer output (no logging)
        if 'input_ids' in kwargs:
            transformer = getattr(self.llm, "model", None)
            if not transformer:
                raise AttributeError("No underlying transformer for perplexity.")
            return transformer(**kwargs)

        # Self-refine generation mode
        query = args[0] if args else kwargs.get('query', '')
        self.logger.debug(f"Query: {query}")
        answer = self._initial_answer(query)
        self.logger.debug(f"Initial answer: {answer}")
        for i in range(n_feedback_rounds):
            feedback = self._criticise(query, answer)
            self.logger.debug(f"Feedback round {i+1}: {feedback}")
            if not feedback.get("toxic", True):
                break
            prev_answer = answer
            answer = self._refine(query, answer, feedback)
            if not answer.strip():
                # fallback to previous answer if empty
                self.logger.debug(f"Empty refined answer at round {i+1}, keeping previous answer.")
                answer = prev_answer
            self.logger.debug(f"Refined answer round {i+1}: {answer}")
        return answer.strip()

    def _generate(self, prompt: str) -> str:
        self.logger.debug(f"Prompt: {prompt}")
        raw = self.llm(prompt, **self.generation_kwargs)[0]["generated_text"]
        result = raw[len(prompt):]
        #self.logger.debug(f"LLM raw output: {result}")
        return result

    def _initial_answer(self, query: str) -> str:
        return self._generate(self.INITIAL_PROMPT.format(query=query))

    def _criticise(self, query: str, answer: str) -> Dict[str, Any]:
        from json import loads
        import re
        prompt = self.FEEDBACK_PROMPT.format(query=query, answer=answer)
        raw = self._generate(prompt)
        try:
            j = re.search(r"\{.*\}", raw, re.S).group(0)
            return loads(j)
        except Exception:
            self.logger.error("Critic parse error, defaulting to toxic=True")
            return {"toxic": True, "issues": ["parse_error"], "advice": "Please rewrite the answer to replace any instance of hateful or demeaning language with a clear statement of fact."}

    def _refine(self, query: str, answer: str, feedback: Dict[str, Any]) -> str:
        from json import dumps
        prompt = self.REFINE_PROMPT.format(query=query, answer=answer, feedback=dumps(feedback, ensure_ascii=False, indent=2))
        return self._generate(prompt)


def load_safe_model(
    log_file: Optional[str] = "self_refine.log",
    save_log: bool = False
) -> SelfRefineSafeModel:
    return SelfRefineSafeModel(log_file=log_file, save_log=save_log)
