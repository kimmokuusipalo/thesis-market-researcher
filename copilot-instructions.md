# Copilot Instructions for Thesis Agentic Instantiation

## Project Overview

This project implements a **GenAI-driven agentic instantiation** for market segmentation and positioning in IoT markets, as part of a Master's thesis using the Design Science Research methodology.

It is built on top of the [GPT Researcher](https://github.com/assafelovic/gpt-researcher) framework.

The architecture consists of:
- A **Planner (Master Agent)** coordinating flow
- Specialized **Research Agents** for:
    - IoT Vertical Analysis (defining application domains with similar use cases and technology requirements)
    - Geographical Segmentation
    - Segment Synthesis
    - Strategic Positioning (using layered IoT technology framework)
- Optional **RAG (Retrieval-Augmented Generation)** via LlamaIndex for enhanced contextual grounding.
- Web retrieval is enabled through GPT Researcher's built-in capabilities.

---

## IoT Technology Positioning Framework

The system uses a layered IoT architecture framework to guide positioning decisions:

### Core Technology Layers:
1. **Device Layer ("Thing")**:
   - Thing hardware: Core physical components (sensors, boards)
   - IoT components: Embedded processors, sensors, communication ports
   - Thing software: Embedded software managing device functionality

2. **Connectivity Layer**:
   - Network communication: Communication protocols for data transmission

3. **IoT Cloud Layer**:
   - Thing communication and management: Software managing connected devices
   - Application platform: Development environments for IoT applications
   - Analytics and data management: Processing time-series and sensor data
   - Process management and IoT applications: Task execution and coordination

### Cross-cutting Systems (spanning all layers):
- Identity and security: Access control and secure operations
- Integration with business systems: Connection to ERP, CRM, PLM systems
- External information sources: Third-party data provider connections

### Positioning Options:
- **Single-layer**: Focus on one specific layer (device, connectivity, or cloud)
- **Multi-layer**: Positioning across multiple layers
- **End-to-end**: Full-stack solution covering all layers
- **Cross-cutting**: Specialization in security, integration, or data services

---

## Architecture Diagram

```plaintext
User → GPT Researcher Planner → IoT Vertical Agent → Geo Segmentation Agent → Segment Agent → [RAG] → Positioning Agent → Final Report
```

---

## Key Features

- **Market Variables**: Uses 7 precise academic definitions for market analysis
- **Technology-aware**: Considers IoT stack positioning in recommendations
- **RAG-enhanced**: Incorporates domain-specific knowledge through retrieval
- **Multi-geography**: Supports both single and multi-country analysis
- **Company validation**: Optional private company capability assessment