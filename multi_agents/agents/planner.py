"""
Planner for Sequential Agentic Market Research Flow (Final Thesis Architecture)
Runs: IoTVerticalAgent → GeoSegmentationAgent → SegmentAgent → PositioningAgent
Maintains context and assembles a thesis-friendly final report.
Each agent receives the user prompt, all required prior context, and retrieved RAG context.
Web Crawler is NOT used; only RAG retrieval from local docs is performed.
"""
from typing import Dict, Any, Optional
from .market_agents import IoTVerticalAgent, GeoSegmentationAgent, SegmentAgent, PositioningAgent
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

DISCLAIMER = "Disclaimer: The following data is synthetic and generated for illustrative purposes only."

class Planner:
    def __init__(self, llm_client, vertical_name: str, region: str, system_architecture: Optional[str] = None, doc_path: str = "./my-docs"):
        self.llm_client = llm_client
        self.vertical_name = vertical_name
        self.region = region
        self.system_architecture = system_architecture
        self.context: Dict[str, Any] = {}
        self.agents = {
            'vertical': IoTVerticalAgent(llm_client),
            'geo': GeoSegmentationAgent(llm_client),
            'segment': SegmentAgent(llm_client),
            'positioning': PositioningAgent(llm_client)
        }
        # Build LlamaIndex index once
        self._build_rag_index(doc_path)

    def _build_rag_index(self, doc_path: str):
        # Load and index documents from the specified folder
        self._rag_documents = SimpleDirectoryReader(doc_path).load_data()
        self._rag_index = VectorStoreIndex.from_documents(self._rag_documents)
        self._rag_query_engine = self._rag_index.as_query_engine()

    def get_rag_context(self, query: str) -> str:
        """
        Retrieve relevant context from the LlamaIndex index for a given query.
        Returns a string with the top 3 results concatenated.
        """
        rag_results = self._rag_query_engine.query(query)
        if hasattr(rag_results, 'response'):
            # LlamaIndex v0.10+ returns a response object
            return rag_results.response.strip()
        elif isinstance(rag_results, str):
            return rag_results.strip()
        elif hasattr(rag_results, '__iter__'):
            # If it's a list of nodes or texts
            return "\n\n".join(str(r) for r in list(rag_results)[:3])
        return ""

    def run(self, user_prompt: str) -> Dict[str, Any]:
        # Step 1: IoT Vertical Agent
        vertical_query = f"Key IoT applications, trends, and challenges in {self.vertical_name}"
        vertical_rag = self.get_rag_context(vertical_query)
        vertical_result = self.agents['vertical'].run(
            user_prompt=user_prompt,
            prior_context=None,
            rag_context=vertical_rag,
            vertical_name=self.vertical_name
        )
        self.context['vertical_result'] = vertical_result

        # Step 2: Geo Segmentation Agent
        geo_query = f"IoT market size, growth, competition, regulations in {self.region} for {self.vertical_name}"
        geo_rag = self.get_rag_context(geo_query)
        geo_result = self.agents['geo'].run(
            user_prompt=user_prompt,
            prior_context={'vertical_result': vertical_result},
            rag_context=geo_rag,
            region=self.region,
            vertical_name=self.vertical_name
        )
        self.context['geo_result'] = geo_result

        # Step 3: Segment Agent
        segment_query = f"Strategic market segments for IoT in {self.region} for {self.vertical_name}, considering previous findings"
        segment_rag = self.get_rag_context(segment_query)
        segment_result = self.agents['segment'].run(
            user_prompt=user_prompt,
            prior_context={
                'vertical_result': vertical_result,
                'geo_result': geo_result
            },
            rag_context=segment_rag
        )
        self.context['segment_result'] = segment_result

        # Step 4: Positioning Agent
        positioning_query = f"Optimal positioning strategies for IoT in {self.region} for {self.vertical_name}, considering segment characteristics"
        positioning_rag = self.get_rag_context(positioning_query)
        positioning_result = self.agents['positioning'].run(
            user_prompt=user_prompt,
            prior_context={
                'vertical_result': vertical_result,
                'geo_result': geo_result,
                'segment_result': segment_result
            },
            rag_context=positioning_rag,
            system_architecture=self.system_architecture
        )
        self.context['positioning_result'] = positioning_result

        # Assemble Final Report
        final_report = self._assemble_report(user_prompt)
        self.context['final_report'] = final_report
        return self.context

    def _assemble_report(self, user_prompt: str) -> str:
        return (
            f"=== IoT Market Research Thesis Report ===\n\n"
            f"User Prompt: {user_prompt}\n\n"
            f"--- IoT Vertical Analysis ---\n{self.context['vertical_result']}\n\n"
            f"--- Geo Segmentation Analysis ---\n{self.context['geo_result']}\n\n"
            f"--- Segment Synthesis ---\n{self.context['segment_result']}\n\n"
            f"--- Strategic Positioning ---\n{self.context['positioning_result']}\n\n"
            f"=== End of Report ==="
        )
