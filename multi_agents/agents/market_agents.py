"""
Market Segmentation and Positioning Agents for IoT Thesis
Implements IoTVerticalAgent, GeoSegmentationAgent, SegmentAgent, PositioningAgent, and CompanyAgent.
Prompts are sourced from agent-prompts.md.
Each agent receives: user_prompt, prior_context, and rag_context.
Web Crawler and Perplexity API are NOT used automatically; only RAG (LlamaIndex) is used for retrieval.
"""
from typing import Optional, Dict

# Move DISCLAIMER here to avoid circular import
DISCLAIMER = "Disclaimer: The following data is synthetic and generated for illustrative purposes only."

class IoTVerticalAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.prompt_template = (
            "# IoT Vertical Analysis\n"
            "Role: Industry Expert in IoT verticals.\n"
            "Task: Identify and describe key IoT applications and trends in the {vertical_name} vertical.\n"
            "Instructions:\n"
            "- List main use cases and trends.\n"
            "- Describe market drivers and barriers.\n"
            "- Mark data as synthetic if based on public summaries."
        )

    def run(self, user_prompt: str, prior_context: Optional[Dict] = None, rag_context: str = "", vertical_name: str = "") -> str:
        prompt = (
            f"{self.prompt_template.format(vertical_name=vertical_name)}\n\n"
            f"User Prompt: {user_prompt}\n\n"
            f"[RAG Context]\n{rag_context}\n"
        )
        result = self.llm_client(prompt)
        return f"{DISCLAIMER}\n\n{result.strip()}"

class GeoSegmentationAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.prompt_template = (
            "# Geo Segmentation Analysis\n"
            "Role: IoT Market Analyst for {region}.\n"
            "Task: Analyze the IoT market landscape in {region} for {vertical_name}.\n"
            "Instructions:\n"
            "- Provide synthetic estimates for market size and growth.\n"
            "- List regulatory factors, competitor presence, and key challenges.\n"
            "- Mark data as synthetic."
        )

    def run(self, user_prompt: str, prior_context: Dict, rag_context: str = "", region: str = "", vertical_name: str = "") -> str:
        vertical_result = prior_context.get("vertical_result", "")
        prompt = (
            f"{self.prompt_template.format(region=region, vertical_name=vertical_name)}\n\n"
            f"User Prompt: {user_prompt}\n\n"
            f"[IoT Vertical Result]\n{vertical_result}\n\n"
            f"[RAG Context]\n{rag_context}\n"
        )
        result = self.llm_client(prompt)
        return f"{DISCLAIMER}\n\n{result.strip()}"

class SegmentAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.prompt_template = (
            """
# Segment Synthesis
Role: Strategic Market Segment Analyst.

Task: Combine IoT vertical and geographical analysis to define actionable market segments for IoT vendors.

Context:
- You will receive:
    - The user prompt
    - IoT Vertical Agent output (vertical characteristics and trends)
    - Geo Segmentation Agent output (geography-specific market dynamics)
    - RAG context (retrieved high-quality documents)

Instructions:
- Using the provided context, analyze and define one or more actionable market segments in the given geography and IoT vertical.
- For each segment, explicitly evaluate the following variables:

    1. Market size and growth rate
    2. Profitability potential
    3. Regulatory requirements (certifications, standards, data privacy laws)
    4. Competitive intensity (market concentration, number of players)
    5. Digital maturity of customers (PoC readiness, full-scale adoption potential)
    6. Customer consolidation (types of buyers, complexity of buying centers)
    7. Technological readiness (existing IoT use, integration capabilities, cloud readiness)

- Present each segment clearly, structured under these variable headings.
- If any variable lacks sufficient information, state so explicitly.

- Remember: The following data is synthetic and generated for illustrative purposes only.
"""
        )

    def run(self, user_prompt: str, prior_context: Dict, rag_context: str = "") -> str:
        vertical_result = prior_context.get("vertical_result", "")
        geo_result = prior_context.get("geo_result", "")
        prompt = (
            f"{self.prompt_template}\n"
            f"User Prompt: {user_prompt}\n\n"
            f"[IoT Vertical Result]\n{vertical_result}\n\n"
            f"[Geo Segmentation Result]\n{geo_result}\n\n"
            f"[RAG Context]\n{rag_context}\n"
        )
        result = self.llm_client(prompt)
        return f"{DISCLAIMER}\n\n{result.strip()}"

class PositioningAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.prompt_template = (
            """
# Strategic Positioning\n"
Role: IoT Strategic Positioning Advisor.\n\n"
Task: Recommend the most appropriate IoT system architecture positioning layer for the vendor, based strictly on the provided segment analysis and market variable scores.\n\n"
Context:\n\n"
You will receive:\n"
- The user prompt\n"
- IoT Vertical Agent output\n"
- Geo Segmentation Agent output\n"
- Segment Agent output (market variable explanations + scores)\n"
- RAG context (retrieved high-quality documents)\n"
- Private company capability description (not to be included in report)\n\n"
Instructions:\n\n"
1. Evaluate each of the following market variables (from the Segment Agent):\n"
    - Market size and growth rate\n"
    - Profitability potential\n"
    - Regulatory requirements\n"
    - Competitive intensity\n"
    - Digital maturity of customers\n"
    - Customer consolidation\n"
    - Technological readiness\n\n"
2. Based strictly on the Segment Agent output and market variable scores, recommend one of the following IoT system architecture positioning layers:\n"
    - Device Layer\n"
    - Middleware Layer\n"
    - Platform / Cloud Layer\n"
    - Multi-layer (end-to-end)\n\n"
3. Justify your recommendation using only the market variable scores and explanations.\n"
4. Do NOT recommend sales actions, partnerships, or general go-to-market advice.\n"
5. Do NOT suggest leveraging capabilities unless it is directly relevant to the system positioning layer.\n"
6. Keep your output clean, technical, and focused on architecture positioning.\n"
7. Begin your output with the disclaimer:\n"
"The following data is synthetic and generated for illustrative purposes only."\n\n"
Remember: This report is public. Do not disclose or reference the private company input directly.\n"""
        )

    def run(self, user_prompt: str, prior_context: Dict, rag_context: str = "", system_architecture: Optional[str] = None, company_capabilities: Optional[str] = None) -> str:
        vertical_result = prior_context.get("vertical_result", "")
        geo_result = prior_context.get("geo_result", "")
        segment_result = prior_context.get("segment_result", "")
        # Compose the prompt, including private company capabilities as a non-output context
        prompt = (
            f"{self.prompt_template}\n"
            f"User Prompt: {user_prompt}\n\n"
            f"[IoT Vertical Result]\n{vertical_result}\n\n"
            f"[Geo Segmentation Result]\n{geo_result}\n\n"
            f"[Segment Synthesis Result]\n{segment_result}\n\n"
            f"[RAG Context]\n{rag_context}\n"
        )
        if company_capabilities:
            prompt += f"\n[Private Company Capabilities] (for LLM context only, do not include in output):\n{company_capabilities}\n"
        if system_architecture:
            prompt += f"\nSystem Architecture: {system_architecture}\n"
        result = self.llm_client(prompt)
        return f"{DISCLAIMER}\n\n{result.strip()}"

class CompanyAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.prompt_template = (
            """
# Company Validation of Positioning
Role: Company Capability Validator.

Task: Evaluate whether the recommended IoT system positioning layer matches the company's capabilities.

Context:
You will receive:
- The user prompt
- IoT Vertical Agent output
- Geo Segmentation Agent output
- Segment Agent output
- Positioning Agent output
- Company capabilities (private, not to be quoted directly)

Instructions:
1. For each market variable, rate from 1 (poor fit) to 5 (excellent fit) how well the company is positioned to compete, with a brief justification:
    - Market size and growth rate
    - Profitability potential
    - Regulatory requirements
    - Competitive intensity
    - Digital maturity of customers
    - Customer consolidation
    - Technological readiness
2. Provide an ultimate recommendation:
    - Strongly recommended
    - Acceptable with caveats
    - Not recommended
3. Justify your recommendation based on the company capabilities and the Positioning Agent's output.
4. Do NOT quote or paraphrase private company input directly.
5. Output must be concise, actionable, and confidential-friendly.
"""
        )

    def run(self, user_prompt: str, prior_context: Dict, rag_context: str = "") -> str:
        vertical_result = prior_context.get("vertical_result", "")
        geo_result = prior_context.get("geo_result", "")
        segment_result = prior_context.get("segment_result", "")
        positioning_result = prior_context.get("positioning_result", "")
        prompt = (
            f"{self.prompt_template}\n"
            f"User Prompt: {user_prompt}\n\n"
            f"[IoT Vertical Result]\n{vertical_result}\n\n"
            f"[Geo Segmentation Result]\n{geo_result}\n\n"
            f"[Segment Synthesis Result]\n{segment_result}\n\n"
            f"[Positioning Agent Result]\n{positioning_result}\n\n"
            f"[Company Capabilities Context]\n{rag_context}\n"
        )
        result = self.llm_client(prompt)
        return f"{DISCLAIMER}\n\n{result.strip()}"

class SegmentRankingAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.prompt_template = (
            """
# Segment Ranking Table
This Segment Ranking Agent is part of a multi-agent system for GenAI-driven market segmentation and positioning in IoT markets, implemented for a Master's thesis using the Design Science Research methodology.

Architecture context:
- The system consists of a Planner coordinating five agents: IoT Vertical Agent, Geo Segmentation Agent, Segment Agent, Positioning Agent, and Segment Ranking Agent.
- The Segment Ranking Agent runs **only if**:
    - company_capabilities.txt is present, AND
    - Segment Agent output includes multiple segments.

Purpose of this agent:
- The Segment Ranking Agent synthesizes the Segment Agent and Positioning Agent outputs, and evaluates the segments against company capabilities.
- It produces an actionable Segment Ranking table to support IoT market entry decisions.
- It provides this table in the Final Report and exports it as an Excel file for decision workshops.

Reason for this structure:
- Supports realistic strategy decision-making.
- Enables prioritization of segments based on capability fit and market attractiveness.
- Demonstrates advanced agentic synthesis in the thesis artefact.

Instructions:
For each segment provided in the Segment Agent output:
- Evaluate the segment against the company's capabilities.
- For each market variable, assign a rating from 1 (very poor fit / high risk) to 5 (excellent fit / opportunity).
- Provide a brief justification for each rating.
- Provide an ultimate recommendation for the segment:
    - Go
    - Further Analyze
    - Not Recommended

Market Variables to Rate:
1. Market Potential
2. Competitive Intensity
3. Regulatory Complexity
4. Technological Readiness
5. Digital Maturity of Customers
6. Fit with Company Capabilities (based on company_capabilities.txt)

Output Format:
For each segment:

Segment Name: [Segment Name]

| Variable                      | Rating (1–5) | Justification |
|-------------------------------|--------------|---------------|
| Market Potential              |              |               |
| Competitive Intensity         |              |               |
| Regulatory Complexity         |              |               |
| Technological Readiness       |              |               |
| Digital Maturity of Customers |              |               |
| Fit with Company Capabilities |              |               |

Ultimate Recommendation: [Go / Further Analyze / Not Recommended]

Final Notes:
- Be concise and actionable.
- Focus on supporting strategic decision making.
- The same table will be exported to Excel — structure your output clearly.

Remember: The following data is synthetic and generated for illustrative purposes only.
"""
        )

    def run(self, prior_context: Dict, company_capabilities: str = "") -> str:
        segment_result = prior_context.get("segment_result", "")
        positioning_result = prior_context.get("positioning_result", "")
        prompt = (
            f"{self.prompt_template}\n"
            f"[Segment Synthesis Result]\n{segment_result}\n\n"
            f"[Positioning Agent Result]\n{positioning_result}\n\n"
            f"[Company Capabilities Context]\n{company_capabilities}\n"
        )
        result = self.llm_client(prompt)
        return result.strip()
