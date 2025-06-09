from dotenv import load_dotenv
import sys
import os
import uuid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multi_agents.agents import ChiefEditorAgent
from multi_agents.agents.planner import Planner
import asyncio
import json
from gpt_researcher.utils.enum import Tone
from gpt_researcher.utils.llm import create_chat_completion
from multi_agents.config import RAG_ACTIVE_DIRECTORY

# Run with LangSmith if API key is set
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
load_dotenv()

def open_task():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to task.json
    task_json_path = os.path.join(current_dir, 'task.json')
    
    with open(task_json_path, 'r') as f:
        task = json.load(f)

    if not task:
        raise Exception("No task found. Please ensure a valid task.json file is present in the multi_agents directory and contains the necessary task information.")

    # Override model with STRATEGIC_LLM if defined in environment
    strategic_llm = os.environ.get("STRATEGIC_LLM")
    if strategic_llm and ":" in strategic_llm:
        # Extract the model name (part after the first colon)
        model_name = strategic_llm.split(":", 1)[1]
        task["model"] = model_name
    elif strategic_llm:
        task["model"] = model_name

    return task

def format_segment_name(raw_segment: str) -> str:
    """
    Clean and format a segment name for safe use in filenames.
    - Replaces spaces and dashes with underscores
    - Removes non-alphanumeric/underscore characters
    - Converts to TitleCase with underscores
    """
    import re
    # Replace em/en dashes and hyphens with underscores
    s = re.sub(r"[\s\-–—]+", "_", raw_segment)
    # Remove all non-alphanumeric/underscore
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    # Remove leading/trailing underscores
    s = s.strip("_")
    return s

async def run_research_task(query, websocket=None, stream_output=None, tone=Tone.Objective, headers=None):
    task = open_task()
    task["query"] = query

    chief_editor = ChiefEditorAgent(task, websocket, stream_output, tone, headers)
    research_report = await chief_editor.run_research_task()

    if websocket and stream_output:
        await stream_output("logs", "research_report", research_report, websocket)

    return research_report

async def main():
    import os
    from openai import OpenAI
    from uuid import uuid4
    import sys
    from datetime import datetime
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    def llm_client(prompt, return_usage=False, **kwargs):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}]
        )
        text = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        if return_usage and usage:
            return text, {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            }
        return text
    rag_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RAG'))
    # Use default vertical/region/system_architecture for CLI, but let user_prompt drive context
    vertical_name = os.environ.get("VERTICAL") or "Smart Cities"
    region = os.environ.get("REGION") or "Finland"
    system_architecture = os.environ.get("SYSTEM_ARCHITECTURE") or "Cloud Platform"
    planner = Planner(llm_client, vertical_name, region, system_architecture)
    try:
        while True:
            print("\n==== New Run ====")
            # List available RAG folders (IoT Vertical–Geography pairs):
            active_rag_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', RAG_ACTIVE_DIRECTORY))
            if os.path.exists(active_rag_dir):
                pairs = [d for d in os.listdir(active_rag_dir) if os.path.isdir(os.path.join(active_rag_dir, d))]
                pairs_main = [p for p in pairs if p.lower() != "company_information"]
                print(f"Available RAG folders (IoT Vertical–Geography pairs) in {RAG_ACTIVE_DIRECTORY}/:")
                for p in pairs_main:
                    print(f" - {p}")
                if "Company_Information" in pairs or "company_information" in pairs:
                    print("(Private) Company_Information folder detected.")
            else:
                print(f"No RAG directory found at: {active_rag_dir}")
            print("[LOG] multi_agents/main.py: Pipeline invoked (frontend or CLI)")
            sys.stdout.flush()
            user_prompt = input("Enter your user prompt (or type 'exit' to quit): ")
            if user_prompt.lower() == "exit":
                print("Exiting...")
                sys.exit(0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Final_Report_{timestamp}.txt"
            context = planner.run(user_prompt, report_filename=filename)
            report = context["final_report"]
            os.makedirs("outputs", exist_ok=True)
            with open(f"outputs/{filename}", "w") as f:
                f.write(report)
            print(f"✅ Final Report saved as: outputs/{filename}")
            print("\n==== Run Complete ====\n")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
