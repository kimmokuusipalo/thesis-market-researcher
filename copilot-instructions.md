# Copilot Instructions for Thesis Agentic Instantiation

## Project Overview

This project implements a **GenAI-driven agentic instantiation** for market segmentation and positioning in IoT markets, as part of a Master's thesis using the Design Science Research methodology.

It is built on top of the [GPT Researcher](https://github.com/assafelovic/gpt-researcher) framework.

The architecture consists of:
- A **Planner (Master Agent)** coordinating flow
- Specialized **Research Agents** for:
    - IoT Vertical Identification
    - Geographical Segmentation
    - Segment Synthesis
    - Strategic Positioning
- Optional **RAG (Retrieval-Augmented Generation)** via LlamaIndex for enhanced contextual grounding.
- Web retrieval is enabled through GPT Researcher’s built-in capabilities.

---

## Architecture Diagram

```plaintext
User → GPT Researcher Planner → IoT Vertical Agent → Geo Segmentation Agent → Segment Agent → [RAG] → Positioning Agent → Final Report