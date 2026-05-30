
from src.model.multi_sent.llm import LLM_multi
from src.model.multi_sent.llm_amr import LLM_amr_multi
from src.model.single_sent.llm import LLM_single
from src.model.single_sent.llm_amr import LLM_AMR_single
from path import MODEL_LLAMA_DIR, MODEL_QWEN_DIR




load_model = {
    "llm_multi": LLM_multi,
    "llm_amr_multi":LLM_amr_multi,
    "llm_single": LLM_single,
    "llm_amr_single":LLM_AMR_single,
}


llm_model_path = {
    'qwen-8b': MODEL_QWEN_DIR,
    'llama-8b': MODEL_LLAMA_DIR,
}
