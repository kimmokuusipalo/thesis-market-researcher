"""
Provides a command line interface for the GPTResearcher class.

Usage:

```shell
python cli.py "<query>" --report_type <report_type> --tone <tone> --query_domains <foo.com,bar.com>
```

"""
import asyncio
import argparse
from argparse import RawTextHelpFormatter
from uuid import uuid4
import os

from dotenv import load_dotenv

from gpt_researcher import GPTResearcher
from gpt_researcher.utils.enum import ReportType, Tone
from backend.report_type import DetailedReport
from multi_agents.agents.planner import Planner
from gpt_researcher.utils.llm import create_chat_completion

# =============================================================================
# CLI
# =============================================================================

cli = argparse.ArgumentParser(
    description="Generate a research report.",
    # Enables the use of newlines in the help message
    formatter_class=RawTextHelpFormatter)

# =====================================
# Arg: Query
# =====================================

cli.add_argument(
    # Position 0 argument
    "query",
    type=str,
    help="The query to conduct research on.")

# =====================================
# Arg: Report Type
# =====================================

choices = [report_type.value for report_type in ReportType]

report_type_descriptions = {
    ReportType.ResearchReport.value: "Summary - Short and fast (~2 min)",
    ReportType.DetailedReport.value: "Detailed - In depth and longer (~5 min)",
    ReportType.ResourceReport.value: "",
    ReportType.OutlineReport.value: "",
    ReportType.CustomReport.value: "",
    ReportType.SubtopicReport.value: "",
    ReportType.DeepResearch.value: "Deep Research"
}

cli.add_argument(
    "--report_type",
    type=str,
    help="The type of report to generate. Options:\n" + "\n".join(
        f"  {choice}: {report_type_descriptions[choice]}" for choice in choices
    ),
    # Deserialize ReportType as a List of strings:
    choices=choices,
    required=True)

# =====================================
# Arg: Tone
# =====================================

cli.add_argument(
    "--tone",
    type=str,
    help="The tone of the report (optional).",
    choices=["objective", "formal", "analytical", "persuasive", "informative",
            "explanatory", "descriptive", "critical", "comparative", "speculative",
            "reflective", "narrative", "humorous", "optimistic", "pessimistic"],
    default="objective"
)

# =====================================
# Arg: Query Domains
# =====================================

cli.add_argument(
    "--query_domains",
    type=str,
    help="A comma-separated list of domains to search for the query.",
    default=""
)

# =============================================================================
# Main
# =============================================================================

async def main(args):
    """
    Run the custom 4-step agentic market research pipeline and write the final report to the output directory.
    """
    # Parse CLI args
    query = args.query
    vertical_name = getattr(args, 'vertical', None) or "Smart Cities"
    region = getattr(args, 'region', None) or "Finland"
    system_architecture = getattr(args, 'system_architecture', None) or "Cloud Platform"

    # Use OpenAI LLM as default (can be replaced with any callable)
    import os
    from openai import OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    def llm_client(prompt):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content

    # Run the Planner
    planner = Planner(llm_client, vertical_name, region, system_architecture)
    context = planner.run(query)
    report = context["final_report"]

    # Write the report to a file
    artifact_filepath = f"outputs/{uuid4()}.md"
    os.makedirs("outputs", exist_ok=True)
    with open(artifact_filepath, "w") as f:
        f.write(report)

    print(f"Report written to '{artifact_filepath}'")

if __name__ == "__main__":
    load_dotenv()
    args = cli.parse_args()
    asyncio.run(main(args))
