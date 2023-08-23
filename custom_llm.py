import os
import zhipuai

from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.chat_models.base import SimpleChatModel
from langchain.schema.messages import BaseMessage
class ZhipuAIAPI(LLM):
    models = ['chatglm_pro', 'chatglm_std', 'chatglm_lite']
    
    model: str = models[2]
    top_p: float = 0.7
    temperature: float = 0.3
    
    full_prompt: list = []
    
    def __init__(self, model):
        super().__init__()
        zhipuai.api_key = os.environ.get('ZHIPUAI_API_KEY')
        self.model = model
    
    @property
    def _llm_type(self) -> str:
        return "ZhipuAIAPI"
    
    def _call(self, prompt: str, stop:Optional[List[str]]=None, 
              run_manager: Optional[CallbackManagerForLLMRun]=None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs is not permitted yet.")
        fmt_prompt = [{"role":"user", "content": f"{prompt}"}]
        self.full_prompt += fmt_prompt
        # params["prompt"] += [{"role": "assistant", "content": f"{Zhipu.answer}" }]

        response = zhipuai.model_api.invoke(
            model=self.model,
            prompt=self.full_prompt,
            top_p=self.top_p,
            temperature=self.temperature
        )
        answer = eval(response['data']['choices'][0]['content']).strip()
        self.full_prompt += [{"role": "assistant", "content": f"{answer}" }]
        return answer

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {'model': self.model,
                'top_p': self.top_p,
                'temperature': self.temperature}



class FakeListChatModel(SimpleChatModel):
    """Fake ChatModel for testing purposes."""

    responses: List
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-list-chat-model"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """First try to lookup in queries, else return 'foo' or 'bar'."""
        response = self.responses[self.i % len(self.responses)]
        self.i += 1
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"responses": self.responses}