"""
Market Segmentation and Positioning Agents for IoT Thesis
Implements IoTVerticalAgent, GeoSegmentationAgent, SegmentAgent, and PositioningAgent.
Prompts are sourced from agent-prompts.md.
"""
from typing import Optional

class IoTVerticalAgent:
    """
    Agent for identifying and describing key IoT applications and trends in a given vertical.
    """
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.prompt_template = (
            "Role: Industry Expert in IoT verticals.\n"
            "Task: Identify and describe key IoT applications and trends in the {vertical_name} vertical.\n"
            "Instructions:\n"
            "- List main use cases and trends.\n"
            "- Describe market drivers and barriers.\n"
            "- Mark data as synthetic if based on public summaries."
        )

    def run(self, vertical_name: str) -> str:
        prompt = self.prompt_template.format(vertical_name=vertical_name)
        return self.llm_client(prompt)

class GeoSegmentationAgent:
    """
    Agent for analyzing the IoT market landscape in a given country/region and vertical.
    """
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.prompt_template = (
            "Role: IoT Market Analyst for {region}.\n"
            "Task: Analyze the IoT market landscape in {region} for {vertical_name}.\n"
            "Instructions:\n"
            "- Provide synthetic estimates for market size and growth.\n"
            "- List regulatory factors, competitor presence, and key challenges.\n"
            "- Mark data as synthetic."
        )

    def run(self, region: str, vertical_name: str) -> str:
        prompt = self.prompt_template.format(region=region, vertical_name=vertical_name)
        return self.llm_client(prompt)

class SegmentAgent:
    """
    Agent for combining IoT vertical and geographical analysis to define actionable market segments.
    """
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.prompt = (
            "Role: Strategic Market Segment Analyst.\n"
            "Task: Combine IoT vertical and geographical analysis to define actionable market segments.\n"
            "Instructions:\n"
            "- Present key segment characteristics.\n"
            "- Prioritize segments based on opportunity metrics (growth, competition, etc.).\n"
            "- Mark data as synthetic."
        )

    def run(self, vertical_geo_analysis: str) -> str:
        prompt = f"{self.prompt}\n\n{vertical_geo_analysis}"
        return self.llm_client(prompt)

class PositioningAgent:
    """
    Agent for recommending optimal market positioning based on segment analysis and IoT system architecture.
    """
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.prompt = (
            "Role: IoT Strategic Positioning Advisor.\n"
            "Task: Recommend optimal market positioning based on segment analysis and IoT system architecture.\n"
            "Instructions:\n"
            "- Suggest whether to position as Hardware, Middleware, Cloud Platform, or Integrated Solution.\n"
            "- Justify the choice.\n"
            "- Mark data as synthetic."
        )

    def run(self, segment_analysis: str, system_architecture: Optional[str] = None) -> str:
        prompt = self.prompt + "\n\n" + segment_analysis
        if system_architecture:
            prompt += f"\n\nSystem Architecture: {system_architecture}"
        return self.llm_client(prompt)
