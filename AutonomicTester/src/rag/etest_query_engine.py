import json
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.llms import LLM
from llama_index.core import PromptTemplate


class EtestQueryEngine(CustomQueryEngine):
    """E-Test Query Engine."""

    retriever: BaseRetriever
    llm: LLM
    qa_prompt: PromptTemplate
    prompt_dict: dict = None

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)

        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        prompt = self.qa_prompt.format(context_str=context_str, query_str=query_str)
        self.prompt_dict = {
            "num_total_chars": len(prompt),
            "num_context_chars": len(context_str),
            "num_query_chars": len(query_str),
            "prompt_str": prompt,
            "context_str": context_str,
            "query_str": query_str,
        }
        response = self.llm.complete(prompt)
        return str(response)
