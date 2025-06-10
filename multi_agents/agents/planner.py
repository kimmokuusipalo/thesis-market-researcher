"""
Planner for Sequential Agentic Market Research Flow (Final Thesis Architecture)
Runs: IoTVerticalAgent → GeoSegmentationAgent → SegmentAgent → PositioningAgent
Maintains context and assembles a thesis-friendly final report.
Each agent receives the user prompt, all required prior context, and retrieved RAG context.
Web Crawler is NOT used; only RAG retrieval from local docs is performed.
"""
from typing import Dict, Any, Optional
from .market_agents import IoTVerticalAgent, GeoSegmentationAgent, SegmentAgent, PositioningAgent, CompanyAgent, SegmentRankingAgent
import os
from multi_agents.config import USE_RAG, RAG_ACTIVE_DIRECTORY
from glob import glob
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import sys
from datetime import datetime

DISCLAIMER = "Disclaimer: The following data is synthetic and generated for illustrative purposes only."

# Pricing for GPT-4o (EUR)
GPT4O_INPUT_EUR_PER_1M = 4.60
GPT4O_OUTPUT_EUR_PER_1M = 13.80
COST_LIMIT_EUR = 20.0

class Planner:
    def __init__(self, llm_client, vertical_name: str, region: str, system_architecture: Optional[str] = None, doc_path: str = "RAG"):
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
        # Use config toggle for RAG
        if not USE_RAG:
            print("RAG disabled — no retrieval performed.")
            self._rag_documents = []
            self._rag_index = None
            self._rag_query_engine = None
            return
        # Use RAG_ACTIVE_DIRECTORY if RAG is enabled
        abs_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', RAG_ACTIVE_DIRECTORY))
        print(f"RAG enabled — using folder: {RAG_ACTIVE_DIRECTORY}/")
        from glob import glob
        pdf_files = glob(os.path.join(abs_doc_path, '**', '*.pdf'), recursive=True)
        print(f"RAG Index build: Found {len(pdf_files)} PDF files in {abs_doc_path}")
        for f in pdf_files:
            print(f" - {os.path.relpath(f, abs_doc_path)}")
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        self._rag_documents = SimpleDirectoryReader(abs_doc_path, recursive=True).load_data()
        print(f"RAG Index built from {abs_doc_path} — {len(self._rag_documents)} documents indexed.")
        self._rag_index = VectorStoreIndex.from_documents(self._rag_documents)
        self._rag_query_engine = self._rag_index.as_query_engine()

    def get_rag_context(self, query: str) -> str:
        if not USE_RAG:
            print("RAG disabled for this run — agent will use prior context + user prompt only.")
            return ""
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

    @staticmethod
    def build_geo_filtered_query(original_query: str, segment_geo: str, segment_vertical: str) -> str:
        """
        Returns a geo-filtered query string for get_rag_context().
        The prefix is:
        "Context: This query is about [segment_vertical] in [segment_geo] only. Exclude information about other geographies not equal to [segment_geo]. "
        The rest of the query is then appended.
        """
        prefix = (
            f"Context: This query is about {segment_vertical} in {segment_geo} only. "
            f"Exclude information about other geographies not equal to {segment_geo}. "
        )
        return prefix + original_query

    def _get_contextual_rag_query(self, base_query: str, user_prompt: str) -> str:
        # Add geo-filtering context hint to the query
        geo_hint = f"Context: This query is about the {self.region} {self.vertical_name} IoT market only. Exclude information about other countries. "
        return geo_hint + base_query + f" | User prompt: {user_prompt}"

    def run(self, user_prompt: str, report_filename: str = None) -> Dict[str, Any]:
        import time
        import pandas as pd
        import re
        start_time = time.time()
        # Read private company capabilities if available
        company_capabilities_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', RAG_ACTIVE_DIRECTORY, 'Company_Information', 'company_capabilities.txt'))
        company_capabilities = ""
        company_agent_should_run = os.path.exists(company_capabilities_path)
        if company_agent_should_run:
            with open(company_capabilities_path, 'r') as f:
                company_capabilities = f.read().strip()
            print(f"Company capabilities context loaded from /{RAG_ACTIVE_DIRECTORY}/Company_Information/company_capabilities.txt")
            print("[LOG] Company Agent will be run.")
        else:
            print(f"No company_capabilities.txt found in /{RAG_ACTIVE_DIRECTORY}/Company_Information/")
            print("[LOG] Company Agent will be skipped.")

        # Step 1: IoT Vertical Agent
        vertical_query = self._get_contextual_rag_query(
            f"Key IoT applications, trends, and challenges in {self.vertical_name}", user_prompt)
        vertical_rag = self.get_rag_context(vertical_query)
        vertical_result = self.agents['vertical'].run(
            user_prompt=user_prompt,
            prior_context=None,
            rag_context=vertical_rag,
            vertical_name=self.vertical_name
        )
        self.context['vertical_result'] = vertical_result

        # Step 2: Geo Segmentation Agent
        geo_query = self.build_geo_filtered_query(
            f"IoT market size, growth, competition, regulations in {self.region} for {self.vertical_name}",
            self.vertical_name,
            self.region
        )
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
        segment_query = self._get_contextual_rag_query(
            f"Strategic market segments for IoT in {self.region} for {self.vertical_name}, considering previous findings", user_prompt)
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
        positioning_query = self._get_contextual_rag_query(
            f"Optimal positioning strategies for IoT in {self.region} for {self.vertical_name}, considering segment characteristics", user_prompt)
        positioning_rag = self.get_rag_context(positioning_query)
        positioning_result = self.agents['positioning'].run(
            user_prompt=user_prompt,
            prior_context={
                'vertical_result': vertical_result,
                'geo_result': geo_result,
                'segment_result': segment_result
            },
            rag_context=positioning_rag,
            system_architecture=self.system_architecture,
            company_capabilities=company_capabilities
        )
        self.context['positioning_result'] = positioning_result

        # Step 5: Company Agent (optional)
        if company_agent_should_run:
            if 'company' not in self.agents:
                self.agents['company'] = CompanyAgent(self._llm_with_token_logging("Company Agent"))
            company_result = self.agents['company'].run(
                user_prompt=user_prompt,
                prior_context={
                    'vertical_result': vertical_result,
                    'geo_result': geo_result,
                    'segment_result': segment_result,
                    'positioning_result': positioning_result
                },
                rag_context=company_capabilities
            )
            self.context['company_result'] = company_result
            # Step 6: Segment Ranking Agent (optional)
            if 'segment_ranking' not in self.agents:
                self.agents['segment_ranking'] = SegmentRankingAgent(self._llm_with_token_logging("Segment Ranking Agent"))
            segment_ranking_md = self.agents['segment_ranking'].run(
                prior_context={
                    'segment_result': segment_result,
                    'positioning_result': positioning_result
                },
                company_capabilities=company_capabilities
            )
            self.context['segment_ranking_md'] = segment_ranking_md
            # Export to Excel
            # Parse markdown table to DataFrame
            import re
            import pandas as pd
            # Find the markdown table (starts with | and has at least 2 lines)
            lines = segment_ranking_md.splitlines()
            table_lines = [l for l in lines if l.strip().startswith('|')]
            if table_lines:
                # Join lines and use pandas.read_csv with sep='|', skip first/last empty columns
                table_str = '\n'.join(table_lines)
                # Remove leading/trailing pipes and whitespace
                table_str = '\n'.join([l.strip().strip('|') for l in table_lines])
                from io import StringIO
                df = pd.read_csv(StringIO(table_str), sep='|')
                # Clean up column names and whitespace
                df.columns = [c.strip() for c in df.columns]
                df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                excel_filename = f"Segment_Ranking_{self.region}_{timestamp}.xlsx"
                excel_path = os.path.join("outputs", excel_filename)
                df.to_excel(excel_path, index=False)
                print(f"[LOG] Segment Ranking table exported to Excel: {excel_path}")
            else:
                print("[WARN] Could not parse Segment Ranking markdown table for Excel export.")
        else:
            self.context['company_result'] = None
            self.context['segment_ranking_md'] = None

        # Print final summary
        print(f"\n=== Token & Cost Summary ===")
        print(f"Total input tokens: {self.total_input_tokens}")
        print(f"Total output tokens: {self.total_output_tokens}")
        print(f"Total tokens: {self.total_input_tokens + self.total_output_tokens}")
        print(f"Estimated total cost: €{self.total_cost_eur:.2f}")

        # Assemble Final Report
        final_report = self._assemble_report(user_prompt)
        self.context['final_report'] = final_report

        # Run verification summary
        elapsed = time.time() - start_time
        print("\n✅ All 4 agents completed.")
        print(f"Token usage summary: {self.total_input_tokens + self.total_output_tokens} tokens (input: {self.total_input_tokens}, output: {self.total_output_tokens})")
        if report_filename:
            print(f"Final Report filename: {report_filename}")
        print(f"Time taken: {elapsed:.1f} seconds")
        print("\n--- End of Run Verification ---\n")
        sys.stdout.flush()
        return self.context

    def _assemble_report(self, user_prompt: str) -> str:
        report = (
            f"=== IoT Market Research Thesis Report ===\n\n"
            f"User Prompt: {user_prompt}\n\n"
            f"--- IoT Vertical Analysis ---\n{self.context['vertical_result']}\n\n"
            f"--- Geo Segmentation Analysis ---\n{self.context['geo_result']}\n\n"
            f"--- Segment Synthesis ---\n{self.context['segment_result']}\n\n"
            f"--- Strategic Positioning ---\n{self.context['positioning_result']}\n\n"
        )
        if self.context.get('company_result'):
            report += f"--- Company Validation of Positioning ---\n{self.context['company_result']}\n\n"
        if self.context.get('segment_ranking_md'):
            report += f"--- Segment Ranking Table ---\n{self.context['segment_ranking_md']}\n\n"
        report += f"=== End of Report ==="
        return report
