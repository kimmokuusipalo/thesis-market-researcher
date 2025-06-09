"""
Planner for Sequential Agentic Market Research Flow
Runs: IoTVerticalAgent → GeoSegmentationAgent → SegmentAgent → PositioningAgent
Maintains context and assembles a thesis-friendly final report.
"""
from typing import Dict, Any
from .market_agents import IoTVerticalAgent, GeoSegmentationAgent, SegmentAgent, PositioningAgent

DISCLAIMER = "Disclaimer: The following data is synthetic and generated for illustrative purposes only."

class Planner:
    def __init__(self, llm_client, vertical_name: str, region: str, system_architecture: str = None):
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

    def run(self, user_prompt: str) -> Dict[str, Any]:
        # Step 1: IoT Vertical Agent
        vertical_input = f"User Prompt: {user_prompt}\n\n"
        vertical_output = self.agents['vertical'].run(self.vertical_name)
        vertical_output = f"{vertical_output}\n\n{DISCLAIMER}"
        self.context['vertical'] = vertical_output

        # Step 2: Geo Segmentation Agent
        geo_input = (
            f"User Prompt: {user_prompt}\n\n"
            f"IoT Vertical Context: {vertical_output}\n"
        )
        geo_output = self.agents['geo'].run(self.region, self.vertical_name)
        geo_output = f"{geo_output}\n\n{DISCLAIMER}"
        self.context['geo'] = geo_output

        # Step 3: Segment Agent
        segment_input = (
            f"User Prompt: {user_prompt}\n\n"
            f"IoT Vertical Context: {vertical_output}\n"
            f"Geo Segmentation Context: {geo_output}\n"
        )
        segment_output = self.agents['segment'].run(f"{vertical_output}\n{geo_output}")
        segment_output = f"{segment_output}\n\n{DISCLAIMER}"
        self.context['segment'] = segment_output

        # Step 4: Positioning Agent
        positioning_input = (
            f"User Prompt: {user_prompt}\n\n"
            f"IoT Vertical Context: {vertical_output}\n"
            f"Geo Segmentation Context: {geo_output}\n"
            f"Segment Context: {segment_output}\n"
        )
        positioning_output = self.agents['positioning'].run(
            f"{vertical_output}\n{geo_output}\n{segment_output}",
            self.system_architecture
        )
        positioning_output = f"{positioning_output}\n\n{DISCLAIMER}"
        self.context['positioning'] = positioning_output

        # Assemble Final Report
        final_report = self._assemble_report(user_prompt)
        self.context['final_report'] = final_report
        return self.context

    def _assemble_report(self, user_prompt: str) -> str:
        return (
            f"=== IoT Market Research Thesis Report ===\n\n"
            f"User Prompt: {user_prompt}\n\n"
            f"--- IoT Vertical Analysis ---\n{self.context['vertical']}\n\n"
            f"--- Geo Segmentation Analysis ---\n{self.context['geo']}\n\n"
            f"--- Segment Synthesis ---\n{self.context['segment']}\n\n"
            f"--- Strategic Positioning ---\n{self.context['positioning']}\n\n"
            f"=== End of Report ==="
        )
