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
    vertical_name = os.environ.get("VERTICAL") or "Smart Cities"
    region = os.environ.get("REGION") or "Finland"
    system_architecture = os.environ.get("SYSTEM_ARCHITECTURE") or "Cloud Platform"
    planner = Planner(llm_client, vertical_name, region, system_architecture)
    rag_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RAG'))
    try:
        while True:
            print("\n==== New Run ====")
            # List available RAG folders (Vertical–Geography pairs)
            if os.path.exists(rag_dir):
                pairs = [d for d in os.listdir(rag_dir) if os.path.isdir(os.path.join(rag_dir, d))]
                print("Available RAG folders (IoT Vertical–Geography pairs):")
                for p in pairs:
                    print(f" - {p}")
            else:
                print("No RAG directory found at:", rag_dir)
            user_prompt = input("\nEnter your user prompt (or press Ctrl+C to exit): ")
            context = planner.run(user_prompt)
            report = context["final_report"]
            artifact_filepath = f"outputs/{uuid4()}.md"
            os.makedirs("outputs", exist_ok=True)
            with open(artifact_filepath, "w") as f:
                f.write(report)
            print(f"Report written to '{artifact_filepath}'")
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
