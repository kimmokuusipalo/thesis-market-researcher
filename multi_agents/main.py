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

def extract_vertical_and_region(user_prompt: str):
    """
    Dynamically extract IoT vertical and region from user prompt.
    Returns (vertical_name, region).
    For global analysis, returns 'Global' as region.
    """
    # Check for global/worldwide keywords first
    global_keywords = ["globally", "global", "worldwide", "international", "best markets", "all countries", "all geographies", "all regions"]
    is_global = any(kw in user_prompt.lower() for kw in global_keywords)
    
    if is_global:
        region = "Global"
    else:
        # Try to extract region dynamically
        region = extract_region_from_prompt(user_prompt)
    
    # Extract vertical dynamically
    vertical = extract_vertical_from_prompt(user_prompt)
    
    return vertical, region

def extract_region_from_prompt(user_prompt: str) -> str:
    """
    Dynamically extract region/country from user prompt.
    First checks available RAG folders, then uses comprehensive country list.
    """
    prompt_lower = user_prompt.lower()
    
    # Get available countries from RAG folders
    available_countries = get_available_countries_from_rag()
    
    # First priority: Check available RAG countries
    for country in available_countries:
        if country.lower() in prompt_lower:
            return country
    
    # Second priority: Comprehensive country list
    country_mappings = {
        # Major economies
        "united states": "United States", "usa": "United States", "america": "United States",
        "china": "China", "chinese": "China",
        "germany": "Germany", "german": "Germany",
        "japan": "Japan", "japanese": "Japan",
        "united kingdom": "United Kingdom", "uk": "United Kingdom", "britain": "United Kingdom",
        "france": "France", "french": "France",
        "italy": "Italy", "italian": "Italy",
        "canada": "Canada", "canadian": "Canada",
        "australia": "Australia", "australian": "Australia",
        "south korea": "South Korea", "korea": "South Korea",
        "india": "India", "indian": "India",
        "brazil": "Brazil", "brazilian": "Brazil",
        "russia": "Russia", "russian": "Russia",
        "spain": "Spain", "spanish": "Spain",
        "netherlands": "Netherlands", "dutch": "Netherlands",
        "switzerland": "Switzerland", "swiss": "Switzerland",
        "sweden": "Sweden", "swedish": "Sweden",
        "norway": "Norway", "norwegian": "Norway",
        "denmark": "Denmark", "danish": "Denmark",
        "finland": "Finland", "finnish": "Finland",
        "belgium": "Belgium", "belgian": "Belgium",
        "austria": "Austria", "austrian": "Austria",
        "israel": "Israel", "israeli": "Israel",
        "singapore": "Singapore",
        "hong kong": "Hong Kong",
        "taiwan": "Taiwan",
        "thailand": "Thailand", "thai": "Thailand",
        "malaysia": "Malaysia", "malaysian": "Malaysia",
        "indonesia": "Indonesia", "indonesian": "Indonesia",
        "philippines": "Philippines", "filipino": "Philippines",
        "vietnam": "Vietnam", "vietnamese": "Vietnam",
        "south africa": "South Africa",
        "nigeria": "Nigeria", "nigerian": "Nigeria",
        "egypt": "Egypt", "egyptian": "Egypt",
        "uae": "UAE", "emirates": "UAE", "dubai": "UAE", "abu dhabi": "UAE",
        "saudi arabia": "Saudi Arabia", "saudi": "Saudi Arabia",
        "turkey": "Turkey", "turkish": "Turkey",
        "poland": "Poland", "polish": "Poland",
        "czech republic": "Czech Republic", "czechia": "Czech Republic",
        "hungary": "Hungary", "hungarian": "Hungary",
        "romania": "Romania", "romanian": "Romania",
        "greece": "Greece", "greek": "Greece",
        "portugal": "Portugal", "portuguese": "Portugal",
        "ireland": "Ireland", "irish": "Ireland",
        "new zealand": "New Zealand",
        "mexico": "Mexico", "mexican": "Mexico",
        "argentina": "Argentina", "argentinian": "Argentina",
        "chile": "Chile", "chilean": "Chile",
        "colombia": "Colombia", "colombian": "Colombia",
        "peru": "Peru", "peruvian": "Peru"
    }
    
    # Check for country matches
    for country_key, country_name in country_mappings.items():
        if country_key in prompt_lower:
            return country_name
    
    # Default fallback - use the most common country from RAG if available
    if available_countries:
        return available_countries[0]
    
    return "Finland"  # Final fallback

def extract_vertical_from_prompt(user_prompt: str) -> str:
    """
    Dynamically extract vertical/industry from user prompt.
    First checks available RAG folders, then uses comprehensive vertical mappings.
    """
    prompt_lower = user_prompt.lower()
    
    # Get available verticals from RAG folders
    available_verticals = get_available_verticals_from_rag()
    
    # First priority: Check available RAG verticals (exact match)
    for vertical in available_verticals:
        if vertical.lower() in prompt_lower:
            return vertical
    
    # Second priority: Keyword-based mapping for comprehensive coverage
    vertical_mappings = {
        # Agriculture & Food
        "agriculture": "Smart Agriculture", "farming": "Smart Agriculture", "farm": "Smart Agriculture",
        "livestock": "Smart Agriculture", "crop": "Smart Agriculture", "greenhouse": "Smart Agriculture",
        "aquaculture": "Smart Agriculture", "precision farming": "Smart Agriculture",
        
        # Waste Management
        "waste": "Smart Waste Management", "recycling": "Smart Waste Management", 
        "garbage": "Smart Waste Management", "trash": "Smart Waste Management",
        "sanitation": "Smart Waste Management",
        
        # Manufacturing & Industry
        "manufacturing": "Smart Manufacturing", "factory": "Smart Manufacturing", 
        "industrial": "Smart Manufacturing", "production": "Smart Manufacturing",
        "assembly": "Smart Manufacturing", "automation": "Smart Manufacturing",
        
        # Mining & Extraction
        "mining": "Smart Mining", "mine": "Smart Mining", "extraction": "Smart Mining",
        "oil": "Smart Mining", "gas": "Smart Mining", "coal": "Smart Mining",
        "mineral": "Smart Mining", "quarry": "Smart Mining",
        
        # Energy & Utilities
        "energy": "Smart Energy", "power": "Smart Energy", "electricity": "Smart Energy",
        "grid": "Smart Energy", "solar": "Smart Energy", "wind": "Smart Energy",
        "nuclear": "Smart Energy", "renewable": "Smart Energy",
        "heating": "Smart District Heating", "district heating": "Smart District Heating",
        "cooling": "Smart District Heating", "thermal": "Smart District Heating",
        
        # Transportation & Logistics
        "transport": "Smart Transportation", "traffic": "Smart Transportation", 
        "mobility": "Smart Transportation", "vehicle": "Smart Transportation",
        "logistics": "Smart Logistics", "supply chain": "Smart Logistics", 
        "warehouse": "Smart Logistics", "shipping": "Smart Logistics",
        "aviation": "Smart Transportation", "maritime": "Smart Transportation",
        "rail": "Smart Transportation", "railway": "Smart Transportation",
        
        # Buildings & Infrastructure
        "building": "Smart Buildings", "facility": "Smart Buildings", "office": "Smart Buildings",
        "home": "Smart Buildings", "residential": "Smart Buildings", "commercial": "Smart Buildings",
        "construction": "Smart Buildings", "real estate": "Smart Buildings",
        "infrastructure": "Smart Infrastructure", "bridge": "Smart Infrastructure",
        "tunnel": "Smart Infrastructure", "road": "Smart Infrastructure",
        
        # Water & Environment
        "water": "Smart Water Management", "irrigation": "Smart Water Management",
        "sewage": "Smart Water Management", "wastewater": "Smart Water Management",
        "stormwater": "Smart Water Management", "drain": "Smart Water Management",
        "environment": "Smart Environmental Monitoring", "pollution": "Smart Environmental Monitoring",
        "air quality": "Smart Environmental Monitoring", "weather": "Smart Environmental Monitoring",
        
        # Healthcare
        "health": "Smart Healthcare", "medical": "Smart Healthcare", "hospital": "Smart Healthcare",
        "patient": "Smart Healthcare", "clinical": "Smart Healthcare", "pharmaceutical": "Smart Healthcare",
        
        # Retail & Commerce
        "retail": "Smart Retail", "shopping": "Smart Retail", "store": "Smart Retail",
        "commerce": "Smart Retail", "point of sale": "Smart Retail", "inventory": "Smart Retail",
        
        # Cities & Public Services
        "city": "Smart Cities", "cities": "Smart Cities", "urban": "Smart Cities",
        "municipal": "Smart Cities", "public": "Smart Cities", "government": "Smart Cities",
        "citizen": "Smart Cities", "public safety": "Smart Cities", "emergency": "Smart Cities",
        
        # Finance & Banking
        "finance": "Smart Finance", "banking": "Smart Finance", "payment": "Smart Finance",
        "fintech": "Smart Finance", "insurance": "Smart Finance", "financial": "Smart Finance",
        "financial services": "Smart Finance", "credit": "Smart Finance", "investment": "Smart Finance",
        
        # Education
        "education": "Smart Education", "school": "Smart Education", "university": "Smart Education",
        "learning": "Smart Education", "campus": "Smart Education",
        
        # Tourism & Hospitality
        "tourism": "Smart Tourism", "hotel": "Smart Tourism", "hospitality": "Smart Tourism",
        "travel": "Smart Tourism", "destination": "Smart Tourism",
        
        # Sports & Entertainment
        "sports": "Smart Sports", "stadium": "Smart Sports", "fitness": "Smart Sports",
        "entertainment": "Smart Entertainment", "venue": "Smart Entertainment"
    }
    
    # Check for vertical matches (longest match first)
    sorted_keywords = sorted(vertical_mappings.keys(), key=len, reverse=True)
    for keyword in sorted_keywords:
        if keyword in prompt_lower:
            return vertical_mappings[keyword]
    
    # Third priority: Use RAG-based vertical names if available
    if available_verticals:
        return available_verticals[0]
    
    # Final fallback
    return "Smart Agriculture"

def get_available_countries_from_rag() -> list:
    """Extract available countries from RAG_active folder names only."""
    countries = []
    try:
        from multi_agents.config import RAG_ACTIVE_DIRECTORY
        
        # Only check RAG_active directory (not main RAG)
        active_rag_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', RAG_ACTIVE_DIRECTORY))
        
        if os.path.exists(active_rag_dir):
            folders = [d for d in os.listdir(active_rag_dir) if os.path.isdir(os.path.join(active_rag_dir, d))]
            for folder in folders:
                if folder.lower() != "company_information":
                    # Parse folder names like "Finland_SmartWaste", "Canada_Mining"
                    if "_" in folder:
                        country = folder.split("_")[0]
                        if country not in countries:
                            countries.append(country)
    except Exception as e:
        print(f"Warning: Could not read RAG_active directory: {e}")
    
    return countries

def get_available_verticals_from_rag() -> list:
    """Extract available verticals from RAG_active folder names only."""
    verticals = []
    try:
        from multi_agents.config import RAG_ACTIVE_DIRECTORY
        
        # Only check RAG_active directory (not main RAG)
        active_rag_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', RAG_ACTIVE_DIRECTORY))
        
        if os.path.exists(active_rag_dir):
            folders = [d for d in os.listdir(active_rag_dir) if os.path.isdir(os.path.join(active_rag_dir, d))]
            for folder in folders:
                if folder.lower() != "company_information":
                    # Parse folder names like "Finland_SmartWaste", "Canada_Mining"
                    if "_" in folder:
                        vertical_part = folder.split("_", 1)[1]  # Take everything after first underscore
                        # Convert to proper format
                        if vertical_part == "SmartWaste":
                            vertical_part = "Smart Waste Management"
                        elif vertical_part == "DistrictHeating":
                            vertical_part = "Smart District Heating"
                        elif vertical_part == "Mining":
                            vertical_part = "Smart Mining"
                        elif vertical_part == "StormwaterDrains":
                            vertical_part = "Smart Water Management"
                        elif not vertical_part.startswith("Smart"):
                            vertical_part = f"Smart {vertical_part}"
                        
                        if vertical_part not in verticals:
                            verticals.append(vertical_part)
    except Exception as e:
        print(f"Warning: Could not read RAG_active directory: {e}")
    
    return verticals

def detect_geo_mode(user_prompt: str) -> str:
    multi_keywords = ["best segments", "compare", "multiple markets", "across geographies", "all geographies", "all regions", "all countries", "globally", "global", "worldwide", "international", "best markets"]
    for kw in multi_keywords:
        if kw in user_prompt.lower():
            return "multi"
    return "single"

async def main():
    import os
    from openai import OpenAI
    from uuid import uuid4
    import sys
    from datetime import datetime
    from multi_agents.config import RAG_ACTIVE_DIRECTORY
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
                
                # Show dynamically detected countries and verticals
                available_countries = get_available_countries_from_rag()
                available_verticals = get_available_verticals_from_rag()
                if available_countries:
                    print(f"\nDynamically detected countries: {', '.join(available_countries)}")
                if available_verticals:
                    print(f"Dynamically detected verticals: {', '.join(available_verticals)}")
                print("Note: System supports 40+ countries and 20+ verticals beyond RAG data.")
            else:
                print(f"No RAG directory found at: {active_rag_dir}")
            print("[LOG] multi_agents/main.py: Pipeline invoked (frontend or CLI)")
            sys.stdout.flush()
            user_prompt = input("Enter your user prompt (or type 'exit' to quit): ")
            if user_prompt.lower() == "exit":
                print("Exiting...")
                sys.exit(0)
            vertical_name, region = extract_vertical_and_region(user_prompt)
            geo_mode = detect_geo_mode(user_prompt)
            system_architecture = os.environ.get("SYSTEM_ARCHITECTURE") or "Cloud Platform"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Final_Report_{timestamp}.txt"
            planner = Planner(llm_client, vertical_name, region, system_architecture, geo_mode=geo_mode)
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
