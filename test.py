
import os
from dotenv import load_dotenv

load_dotenv()
from agents import WebSearchTool, Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace, function_tool
import requests
from openai import AsyncOpenAI
from pydantic import BaseModel

# OpenAI API Key Configuration
# The API key can be set via environment variable OPENAI_API_KEY
# or by setting it directly here (not recommended for production)
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.\n"
        "You can do this by running: export OPENAI_API_KEY='your-api-key-here'\n"
        "Or by creating a .env file with OPENAI_API_KEY=your-api-key-here"
    )

# Set OpenAI API key in environment for the agents library
os.environ["OPENAI_API_KEY"] = openai_api_key

# Tool definitions
web_search_preview = WebSearchTool()


# Shared client for file search
client = AsyncOpenAI(api_key=openai_api_key)

@function_tool
def get_epa_water_data(latitude: float, longitude: float) -> str:
    """
    Retrieves water data from the US EPA WATERS API for a given location.
    
    Args:
        latitude: The latitude of the location.
        longitude: The longitude of the location.
        
    Returns:
        A formatted string containing watershed and water quality information.
    """
    import json
    import requests
    try:
        # Step 1: Use EPA Point Indexing Service to find the nearest water features
        url = "https://ordspub.epa.gov/ords/waters10/PointIndexing.Service"
        params = {
            "pGeometry": f"POINT({longitude} {latitude})",
            "pGeometryMod": "WGS84",
            "pPointIndexingMethod": "DISTANCE",
            "pPointIndexingMaxDist": 50,
            "pPointIndexingFcode": "51600",
            "pResolution": "3"
        }
        
        output_parts = []
        resp = requests.get(url, params=params, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            if "output" in data and data["output"] and "ary_flowlines" in data["output"]:
                flowlines = data["output"]["ary_flowlines"]
                if flowlines:
                    first_feature = flowlines[0]
                    gnis_name = first_feature.get("gnis_name") or "Unnamed Stream"
                    comid = first_feature.get("comid")
                    huc12 = first_feature.get("wbd_huc12", "Unknown HUC")
                    reachcode = first_feature.get("reachcode", "Unknown Reach")
                    
                    output_parts.append(f"Found nearby water feature: {gnis_name}")
                    output_parts.append(f"COMID: {comid}")
                    output_parts.append(f"HUC12 Watershed Code: {huc12}")
                    output_parts.append(f"Reachcode: {reachcode}")
                    
                    # Try to get StreamCat metrics
                    sc_url = "https://api.epa.gov/StreamCat/streams/metrics"
                    sc_params = {
                        "comid": comid,
                        "areaOfInterest": "catchment",
                        "name": "WtDep_2008,BFI,Runoff"
                    }
                    try:
                        sc_resp = requests.get(sc_url, params=sc_params, timeout=5)
                        if sc_resp.status_code == 200:
                            sc_data = sc_resp.json()
                            output_parts.append(f"StreamCat Data: {json.dumps(sc_data)[:500]}")
                    except Exception:
                        output_parts.append("Could not retrieve StreamCat metrics.")
                else:
                    output_parts.append("No water features found nearby (empty feature list).")
            else:
                output_parts.append("No water features found in response.")
        else:
            output_parts.append(f"Error querying EPA Point Indexing API: {resp.status_code} {resp.text}")
             
        return "\n".join(output_parts)

    except Exception as e:
        return f"Error querying EPA WATERS API: {str(e)}"

class TopicClassifierAgentSchema(BaseModel):
  classifier: str


class LocationVerificationSchema(BaseModel):
  location: str | None


topic_classifier_agent = Agent(
  name="Topic Classifier Agent",
  instructions="""Classify the user input into one of the following topics only:
- food
- water
- transport
- energy

Provide just the topic name as the output. Do not run this agent until the full query with the location is provided from the previous location verification agent.""",
  model="gpt-4o-mini",
  output_type=TopicClassifierAgentSchema,
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)

food = Agent(
  name="Food",
  instructions="""Act as a specialist in sustainable food practices.

# Response Format
- **User-Friendly & Accessible**: Use clear, simple language suitable for a general audience. Avoid dense paragraphs.
- **Structured Layout**: Use Markdown headers, bullet points, and bold text.
- **Content Flow**:
    1.  **Direct Answer**: Start with a clear, direct summary.
    2.  **The Impact**: Show the CO2 calculation clearly.
    3.  **Comparison**: Compare to the average.
    4.  **Actionable Advice**: Bulleted list of suggestions.

**CRITICAL: You must cite the source for ANY piece of data or fact you mention. Do NOT put citations inline at the end of each line/sentence. INSTEAD, gather all your sources and list them clearly as a bulleted "Sources:" list at the VERY BOTTOM of your response (but above the JSON chart block). Do NOT use any external knowledge.**

ONLY answer from the websites provided below. YOU ARE STRICTLY FORBIDDEN from answering using any other data, inferences, or simulated/imagined facts.
- https://www.un.org/en/climatechange/science/climate-issues/food
- https://patents.google.com/patent/WO2024214112A1
- https://fred.stlouisfed.org/series/EMISSCO2TOTVTCTOUSA
- https://www.oecd.org/en/publications/measuring-carbon-footprints-of-agri-food-products_8eb75706-en/full-report.html
- https://www.usgs.gov/special-topics/water-science-school/science/how-much-water-do-you-use
- https://www.epa.gov/watersense/showerheads
- https://pubs.acs.org/doi/10.1021/acs.est.5c07609
- https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator
- https://www.epa.gov/green-power-markets/green-power-equivalency-calculator
- https://cotap.org/carbon-footprint-calculator/
- https://www.fueleconomy.gov/feg/Find.do?action=sbsSelect

# Steps
- Carefully read and understand the user's query.
- DO NOT simulate or perform any general web search. You are strictly restricted to the static knowledge of the provided URLs.
- Analyze and synthesize the information from ONLY the provided sources.
- Present your answer using appropriate terminology and professional tone.
- Your answer needs to show how much CO2 the person is using per year by doing a calculation using ONLY the provided sources.
- Compare the impact to the average, telling them how much worse what they're doing is than the average.
- Ensure that any reasoning or justifications are included before delivering your final answer.
- Offer 3-4 specific, actionable, and scientifically grounded suggestions to lower their CO2 footprint taking into account the location provided.

# Notes
- Keep paragraphs short (2-3 sentences max).
- Always answer in the role of a specialist.

# Chart Output
If you make a comparison (e.g., User's impact vs Average), you MUST append a JSON block at the VERY END of your response (after all text).
Format:
```json
{
  "chart": {
    "type": "bar",
    "labels": ["You", "Average", "Efficient Option"],
    "values": [120, 150, 80],
    "label": "CO2 Emissions (kg)",
    "title": "Your Footprint vs Average"
  }
}
```
""",
  model="gpt-4o-mini",
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


location_verification = Agent(
  name="Location verification",
  instructions="""Extract the city, state, and country from the user's input or conversation history if present. It is NOT mandatory.

# Steps

1. Analyze the user's input and conversation history to extract the city, state, and country.
2. If location info is found, output it.
3. If no location is found, output empty string. Do NOT ask the user for it.

# Output Format

Output in JSON format as:
{
"location": "[City, State, Country] or empty string"
}
""",
  model="gpt-4o-mini",
  output_type=LocationVerificationSchema,
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


energy = Agent(
  name="Energy",
  instructions="""Act as a specialist in the topic area specified or implied in the user's prompt, but don't say you're a specialist.

# Response Format
- **User-Friendly & Accessible**: Use clear, simple language suitable for a general audience. Avoid dense paragraphs.
- **Structured Layout**: Use Markdown headers, bullet points, and bold text.
- **Content Flow**:
    1.  **Direct Answer**: Start with a clear, direct summary.
    2.  **The Impact**: Show the CO2 calculation clearly.
    3.  **Comparison**: Compare to the average.
    4.  **Actionable Advice**: Bulleted list of suggestions.

**CRITICAL: You must cite the source for ANY piece of data or fact you mention. Do NOT put citations inline at the end of each line/sentence. INSTEAD, gather all your sources and list them clearly as a bulleted "Sources:" list at the VERY BOTTOM of your response (but above the JSON chart block). Do NOT use any external knowledge.**

ONLY answer from the websites provided below. YOU ARE STRICTLY FORBIDDEN from answering using any other data, inferences, or simulated/imagined facts.
- https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator
- https://www.eia.gov/environment/emissions/co2_vol_mass.php
- https://fred.stlouisfed.org/series/EMISSCO2TOTVTCTOUSA
- https://www.oecd.org/en/publications/measuring-carbon-footprints-of-agri-food-products_8eb75706-en/full-report.html
- https://www.usgs.gov/special-topics/water-science-school/science/how-much-water-do-you-use
- https://www.epa.gov/watersense/showerheads
- https://pubs.acs.org/doi/10.1021/acs.est.5c07609
- https://www.epa.gov/green-power-markets/green-power-equivalency-calculator
- https://cotap.org/carbon-footprint-calculator/
- https://www.fueleconomy.gov/feg/Find.do?action=sbsSelect

# Steps
- Carefully read and understand the user's query and identify the relevant field of expertise.
- DO NOT simulate or perform any general web search. You are strictly restricted to the static knowledge of the provided URLs.
- Analyze and synthesize the information from ONLY the provided sources.
- Present your answer using appropriate terminology and professional tone.
- Your answer needs to show how much CO2 the person is using per year by doing a calculation using ONLY the provided sources.
- Compare the impact to the average, telling them how much worse what they're doing is than the average.
- Ensure that any reasoning or justifications are included before delivering your final answer.
- Offer scientifically grounded suggestions on how the user can lower their CO2 footprint taking into account the location provided.

# Notes
- Keep paragraphs short (2-3 sentences max).
- Always answer in the role of a specialist.

# Chart Output
If you make a comparison (e.g., User's impact vs Average), you MUST append a JSON block at the VERY END of your response (after all text).
Format:
```json
{
  "chart": {
    "type": "bar",
    "labels": ["You", "Average", "Efficient Option"],
    "values": [5000, 7500, 4000],
    "label": "Annual kWh Usage",
    "title": "Energy Consumption Comparison"
  }
}
```
""",
  model="gpt-4o-mini",
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


transport = Agent(
  name="Transport",
  instructions="""Act as a specialist in the topic area specified or implied in the user's prompt, but don't say you're a specialist.

# Response Format
- **User-Friendly & Accessible**: Use clear, simple language suitable for a general audience. Avoid dense paragraphs.
- **Structured Layout**: Use Markdown headers, bullet points, and bold text.
- **Content Flow**:
    1.  **Direct Answer**: Start with a clear, direct summary.
    2.  **The Impact**: Show the CO2 calculation clearly.
    3.  **Comparison**: Compare to the average.
    4.  **Actionable Advice**: Bulleted list of suggestions.

**CRITICAL: You must cite the source for ANY piece of data or fact you mention. Do NOT put citations inline at the end of each line/sentence. INSTEAD, gather all your sources and list them clearly as a bulleted "Sources:" list at the VERY BOTTOM of your response (but above the JSON chart block). Do NOT use any external knowledge.**

ONLY answer from the websites provided below. YOU ARE STRICTLY FORBIDDEN from answering using any other data, inferences, or simulated/imagined facts.
- https://data360.worldbank.org/en/dataset/OWID_CB
- https://fred.stlouisfed.org/series/EMISSCO2TOTVTCTOUSA
- https://www.oecd.org/en/publications/measuring-carbon-footprints-of-agri-food-products_8eb75706-en/full-report.html
- https://www.usgs.gov/special-topics/water-science-school/science/how-much-water-do-you-use
- https://www.epa.gov/watersense/showerheads
- https://pubs.acs.org/doi/10.1021/acs.est.5c07609
- https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator
- https://www.epa.gov/green-power-markets/green-power-equivalency-calculator
- https://cotap.org/carbon-footprint-calculator/
- https://www.fueleconomy.gov/feg/Find.do?action=sbsSelect

# Steps
- Carefully read and understand the user's query and identify the relevant field of expertise.
- DO NOT simulate or perform any general web search. You are strictly restricted to the static knowledge of the provided URLs.
- Analyze and synthesize the information from ONLY the provided sources.
- Present your answer using appropriate terminology and professional tone.
- Your answer needs to show how much CO2 the person is using per year by doing a calculation using ONLY the provided sources.
- Compare the impact to the average, telling them how much worse what they're doing is than the average.
- Ensure that any reasoning or justifications are included before delivering your final answer.
- Offer scientifically grounded suggestions on how the user can lower their CO2 footprint taking into account the location provided.

# Notes
- Keep paragraphs short (2-3 sentences max).
- Always answer in the role of a specialist.

# Chart Output
If you make a comparison (e.g., User's impact vs Average), you MUST append a JSON block at the VERY END of your response (after all text).
Format:
```json
{
  "chart": {
    "type": "bar",
    "labels": ["You", "Average", "EV/Train"],
    "values": [2.5, 4.6, 1.2],
    "label": "CO2 per Trip (kg)",
    "title": "Transport Emission Comparison"
  }
}
```
""",
  model="gpt-4o-mini",
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


water = Agent(
  name="Water",
  instructions="""Act as a specialist in the topic area specified or implied in the user's prompt.

# Response Format
- **User-Friendly & Accessible**: Use clear, simple language suitable for a general audience. Avoid dense paragraphs.
- **Structured Layout**: Use Markdown headers, bullet points, and bold text.
- **Content Flow**:
    1.  **Direct Answer**: Start with a clear, direct summary.
    2.  **Local Data**: Incorporate EPA/Drought Monitor data if available.
    3.  **Analysis**: Explain the findings simply.
    4.  **Actionable Advice**: Bulleted list of suggestions.

**CRITICAL: You must cite the source for ANY piece of data or fact you mention. Do NOT put citations inline at the end of each line/sentence. INSTEAD, gather all your sources and list them clearly as a bulleted "Sources:" list at the VERY BOTTOM of your response (but above the JSON chart block). Do NOT use any external knowledge.**

ONLY answer from the websites provided below. YOU ARE STRICTLY FORBIDDEN from answering using any other data, inferences, or simulated/imagined facts.
- https://www.oecd.org/en/publications/measuring-carbon-footprints-of-agri-food-products_8eb75706-en/full-report.html
- https://www.usgs.gov/special-topics/water-science-school/science/how-much-water-do-you-use
- https://www.epa.gov/watersense/showerheads
- https://pubs.acs.org/doi/10.1021/acs.est.5c07609
- https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator
- https://www.epa.gov/green-power-markets/green-power-equivalency-calculator
- https://cotap.org/carbon-footprint-calculator/
- https://www.fueleconomy.gov/feg/Find.do?action=sbsSelect

You have access to the EPA WATERS API via the `get_epa_water_data` tool. Use this tool to get detailed information about local water sheds and quality if the user provides a US location.
1. First, use the `web_search_preview` tool ONLY to find the latitude and longitude of the city/location provided by the user. Do not use it for general facts.
2. Then, use `get_epa_water_data` with these coordinates.
3. **Strictly cite the EPA API for any data retrieved from it.**
4. Incorporate the findings from the EPA API and droughtmonitor.unl.edu website into your response.

# Steps
- Carefully read and understand the user's query.
- Use `web_search_preview` solely for geocoding (finding latitude and longitude) unless absolutely required for a local fact. DO NOT simulate any other web search.
- Use `get_epa_water_data` to get local water metrics.
- DO NOT use external knowledge. Only use the returned data from your tools and your instructions.
- Analyze and synthesize the information found.
- Present your answer as a subject-matter expert, but in a way that is easy for a non-expert to understand.
- Offer 3-4 specific, actionable, and scientifically grounded suggestions to lower the footprint.

# Notes
- Keep paragraphs short (2-3 sentences max).
- Prioritize current and accurate information from your tools.

# Chart Output
If you make a comparison (e.g., User's impact vs Average), you MUST append a JSON block at the VERY END of your response (after all text).
Format:
```json
{
  "chart": {
    "type": "bar",
    "labels": ["You", "Average", "Water Wise"],
    "values": [150, 80, 50],
    "label": "Daily Water Usage (L)",
    "title": "Water Consumption Comparison"
  }
}
```
""",
  model="gpt-4o-mini",
  tools=[web_search_preview, get_epa_water_data],
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)





class WorkflowInput(BaseModel):
  input_as_text: str
  history: list[TResponseInputItem] | None = None
  previous_topic: str | None = None


# Main code entrypoint
async def run_workflow(workflow_input: WorkflowInput):
  with trace("Cameron"):
    # Helper to safe-guard against model_dump errors
    def safe_model_dump(obj):
        if hasattr(obj, 'model_dump'):
            # usage of mode='json' ensures we get python primitives (dicts/lists) 
            # instead of any internal pydantic iterators or custom types
            return obj.model_dump(mode='json')
        if isinstance(obj, dict):
            return obj
        return {}

    def safe_model_dump_json(obj):
        if hasattr(obj, 'model_dump_json'):
            return obj.model_dump_json()
        import json
        if isinstance(obj, dict):
            return json.dumps(obj)
        return str(obj)

    workflow = safe_model_dump(workflow_input)
    
    # Initialize with history if provided, otherwise start fresh
    conversation_history: list[TResponseInputItem] = []
    if workflow.get("history"):
        conversation_history.extend(workflow["history"])
    
    conversation_history.append(
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": workflow["input_as_text"]
          }
        ]
      }
    )

    # Determine topic:
    # 1. If previous_topic is provided and user is confirming, use it.
    # 2. Otherwise, run classification.
    
    classifier = None
    target_topic = workflow.get("previous_topic")
    user_input_lower = workflow["input_as_text"].lower()
    
    confirmation_keywords = ["yes", "please", "comprehensive", "detail", "sure", "ok", "okay", "yeah"]
    is_confirmation = any(keyword in user_input_lower for keyword in confirmation_keywords)
    
    if target_topic and is_confirmation:
        classifier = target_topic
        # Add a hint to the agent that they should provide detailed analysis
        conversation_history.append({
            "role": "system",
            "content": [{"type": "input_text", "text": "User has requested the comprehensive detailed analysis. Please provide it now."}]
        })
    
    
    # Step 1: Run location verification agent (only if we don't know potential classifier yet or just to be safe, 
    # but actually we might want to skip if we are just continuing? 
    # Let's run it to capture any new location info or keep context, but verification is cheap)
    # Actually, if we are confirming, we probably don't need re-verification but let's keep it simple.
    
    location_verification_result_temp = await Runner.run(
      location_verification,
      input=[
        *conversation_history
      ],
      run_config=RunConfig(trace_metadata={
        "__trace_source__": "agent-builder",
        "workflow_id": "wf_68f94a7e17d48190b601b55060f7be580cd9f85a9a7b9c30"
      })
    )

    conversation_history.extend([item.to_input_item() for item in location_verification_result_temp.new_items])

    location_verification_result = {
      "output_text": safe_model_dump_json(location_verification_result_temp.final_output),
      "output_parsed": safe_model_dump(location_verification_result_temp.final_output)
    }
    
    # Check if a valid location was extracted (non-empty location string)
    extracted_location = location_verification_result["output_parsed"].get("location", "").strip()
    
    if not extracted_location:
         # Location not found is okay now, just note it
         pass
    
    
    if not classifier:
        # Step 2: Location found, run topic classifier agent
        topic_classifier_agent_result_temp = await Runner.run(
          topic_classifier_agent,
          input=[
            *conversation_history
          ],
          run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_68f94a7e17d48190b601b55060f7be580cd9f85a9a7b9c30"
          })
        )

        conversation_history.extend([item.to_input_item() for item in topic_classifier_agent_result_temp.new_items])

        topic_classifier_agent_result = {
          "output_text": safe_model_dump_json(topic_classifier_agent_result_temp.final_output),
          "output_parsed": safe_model_dump(topic_classifier_agent_result_temp.final_output)
        }
        
        classifier = topic_classifier_agent_result["output_parsed"].get("classifier", "").lower().strip()
    
    # Step 3: Route to the appropriate specialist agent based on classification
    # classifier variable is already set above either from previous_topic or running the agent
    
    if classifier == "water":
      specialist_result_temp = await Runner.run(
        water,
        input=[*conversation_history],
        run_config=RunConfig(trace_metadata={
          "__trace_source__": "agent-builder",
          "workflow_id": "wf_68f94a7e17d48190b601b55060f7be580cd9f85a9a7b9c30"
        })
      )
      conversation_history.extend([item.to_input_item() for item in specialist_result_temp.new_items])
      history_dump = [safe_model_dump(item) for item in conversation_history]
      return {"output_text": specialist_result_temp.final_output_as(str), "topic": "water", "location": extracted_location, "history": history_dump}
      
    elif classifier == "food":
      specialist_result_temp = await Runner.run(
        food,
        input=[*conversation_history],
        run_config=RunConfig(trace_metadata={
          "__trace_source__": "agent-builder",
          "workflow_id": "wf_68f94a7e17d48190b601b55060f7be580cd9f85a9a7b9c30"
        })
      )
      conversation_history.extend([item.to_input_item() for item in specialist_result_temp.new_items])
      history_dump = [safe_model_dump(item) for item in conversation_history]
      return {"output_text": specialist_result_temp.final_output_as(str), "topic": "food", "location": extracted_location, "history": history_dump}
      
    elif classifier == "transport":
      specialist_result_temp = await Runner.run(
        transport,
        input=[*conversation_history],
        run_config=RunConfig(trace_metadata={
          "__trace_source__": "agent-builder",
          "workflow_id": "wf_68f94a7e17d48190b601b55060f7be580cd9f85a9a7b9c30"
        })
      )
      conversation_history.extend([item.to_input_item() for item in specialist_result_temp.new_items])
      history_dump = [safe_model_dump(item) for item in conversation_history]
      return {"output_text": specialist_result_temp.final_output_as(str), "topic": "transport", "location": extracted_location, "history": history_dump}
      
    elif classifier == "energy":
      specialist_result_temp = await Runner.run(
        energy,
        input=[*conversation_history],
        run_config=RunConfig(trace_metadata={
          "__trace_source__": "agent-builder",
          "workflow_id": "wf_68f94a7e17d48190b601b55060f7be580cd9f85a9a7b9c30"
        })
      )
      conversation_history.extend([item.to_input_item() for item in specialist_result_temp.new_items])
      history_dump = [safe_model_dump(item) for item in conversation_history]
      return {"output_text": specialist_result_temp.final_output_as(str), "topic": "energy", "location": extracted_location, "history": history_dump}
      
    else:
      # Unknown classifier, return the classification result
      history_dump = [safe_model_dump(item) for item in conversation_history]
      return {"output_text": f"I couldn't classify your question into water, food, transport, or energy. Classification: {classifier}", "topic": classifier, "location": extracted_location, "history": history_dump}


# Main entry point
async def main():
    """Main function to run the EcoBot workflow interactively."""
    print("=" * 60)
    print("🌱 Welcome to EcoBot - Your Environmental Impact Assistant 🌱")
    print("=" * 60)
    print("\nI can help you understand your environmental impact in areas of:")
    print("  • Food consumption")
    print("  • Water usage")
    print("  • Transportation")
    print("  • Energy consumption")
    print("\nPlease include your city/location in your question.")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            # Get user input
            user_input = input("🌍 Your question: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using EcoBot! 🌿 Goodbye!")
                break
            
            # Skip empty input
            if not user_input:
                print("Please enter a question.\n")
                continue
            
            print("\n⏳ Processing your request...\n")
            
            # Create workflow input and run
            workflow_input = WorkflowInput(input_as_text=user_input)
            result = await run_workflow(workflow_input)
            
            # Display result
            print("-" * 60)
            print("📊 Response:")
            print("-" * 60)
            
            if isinstance(result, dict):
                if "output_text" in result:
                    print(result["output_text"])
                else:
                    import json
                    print(json.dumps(result, indent=2))
            else:
                print(result)
            
            print("-" * 60)
            print()
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye! 🌿")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again with a different question.\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
