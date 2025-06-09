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
import sys
from datetime import datetime

DISCLAIMER = "Disclaimer: The following data is synthetic and generated for illustrative purposes only."

# Pricing for GPT-4o (EUR)
GPT4O_INPUT_EUR_PER_1M = 4.60
GPT4O_OUTPUT_EUR_PER_1M = 13.80
COST_LIMIT_EUR = 20.0

class Planner:
    def __init__(self, llm_client, vertical_name: str, region: str, system_architecture: Optional[str] = None, doc_path: str = "./my-docs"):
        self.llm_client = llm_client
        self.vertical_name = vertical_name
        self.region = region
        self.system_architecture = system_architecture
        self.context: Dict[str, Any] = {}
        self.agents = {
            'vertical': IoTVerticalAgent(self._llm_with_token_logging("IoT Vertical Agent")),
            'geo': GeoSegmentationAgent(self._llm_with_token_logging("Geo Segmentation Agent")),
            'segment': SegmentAgent(self._llm_with_token_logging("Segment Agent")),
            'positioning': PositioningAgent(self._llm_with_token_logging("Positioning Agent"))
        }
        # Build LlamaIndex index once
        self._build_rag_index(doc_path)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_eur = 0.0

    def _llm_with_token_logging(self, agent_name):
        def wrapper(prompt, **kwargs):
            response = self.llm_client(prompt, return_usage=True, **kwargs)
            # response: (text, usage_dict)
            if isinstance(response, tuple) and len(response) == 2:
                text, usage = response
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', input_tokens + output_tokens)
                # Cost calculation
                input_cost = (input_tokens / 1_000_000) * GPT4O_INPUT_EUR_PER_1M
                output_cost = (output_tokens / 1_000_000) * GPT4O_OUTPUT_EUR_PER_1M
                total_cost = input_cost + output_cost
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_cost_eur += total_cost
                print(f"Agent: {agent_name} | Input tokens: {input_tokens} | Output tokens: {output_tokens} | Total tokens: {total_tokens} | Estimated cost: €{total_cost:.2f}")
                if self.total_cost_eur > COST_LIMIT_EUR:
                    print(f"Safeguard triggered: Estimated cost exceeded €20 — aborting run.")
                    sys.exit(1)
                return text
            else:
                # Fallback: no usage info
                return response
        return wrapper

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

        # Print final summary
        print(f"\n=== Token & Cost Summary ===")
        print(f"Total input tokens: {self.total_input_tokens}")
        print(f"Total output tokens: {self.total_output_tokens}")
        print(f"Total tokens: {self.total_input_tokens + self.total_output_tokens}")
        print(f"Estimated total cost: €{self.total_cost_eur:.2f}")

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

    def format_final_report(self, vertical_result, geo_result, segment_result, positioning_result) -> str:
        """
        Formats the final report in a top-down, actionable, professional style with explicit market variable scoring.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        separator = "\n" + "="*80 + "\n"

        executive_summary = (
            "1. [Key actionable insight 1 from agent results]"
            "\n2. [Key actionable insight 2 from agent results]"
            "\n3. [Key actionable insight 3 from agent results]"
            "\n4. [Clear positioning recommendation]"
            "\n5. [Summary of segment attractiveness]"
        )

        def market_variable_section(variable, explanation, score):
            return (
                f"**{variable}**\n"
                f"Explanation: {explanation}\n"
                f"Attractiveness score: {score}/5\n"
            )

        segment_deep_dive = "\n".join([
            market_variable_section("Market size and growth rate", "[Explanation from segment_result]", "[Score]"),
            market_variable_section("Profitability potential", "[Explanation from segment_result]", "[Score]"),
            market_variable_section("Regulatory requirements", "[Explanation from segment_result]", "[Score]"),
            market_variable_section("Competitive intensity", "[Explanation from segment_result]", "[Score]"),
            market_variable_section("Digital maturity of customers", "[Explanation from segment_result]", "[Score]"),
            market_variable_section("Customer consolidation", "[Explanation from segment_result]", "[Score]"),
            market_variable_section("Technological readiness", "[Explanation from segment_result]", "[Score]"),
        ])

        positioning_section = (
            "## Strategic Positioning Recommendation\n"
            f"{positioning_result}\n"
            "Actionable next steps:\n"
            "- [Step 1]\n"
            "- [Step 2]\n"
        )

        report = (
            f"# IoT Market Segmentation & Positioning Report\n"
            f"**Date:** {now}\n"
            f"{separator}"
            "## 1. Executive Summary\n"
            f"{executive_summary}\n"
            f"{separator}"
            "## 2. Market Context\n"
            "### 2.1 IoT Vertical Analysis\n"
            f"{vertical_result}\n"
            "### 2.2 Geo Segmentation Analysis\n"
            f"{geo_result}\n"
            f"{separator}"
            "## 3. Market Segment Deep Dive\n"
            f"{segment_deep_dive}\n"
            f"{separator}"
            f"{positioning_section}\n"
            f"{separator}"
            "## 5. Appendix\n"
            "- The following data is synthetic and generated for illustrative purposes only.\n"
            "- Sources: See private RAG index documentation.\n"
        )
        return report
