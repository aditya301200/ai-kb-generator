import streamlit as st
import pandas as pd
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# --- 1. Streamlit UI & Setup ---
st.set_page_config(page_title="AI KB Generator", page_icon="📚", layout="centered")

# Secure API Key input in the sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your sk-... key here")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    st.markdown("---")
    st.markdown("Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)")

st.title("📚 AI Knowledge Generator")
st.write("Generate KBs, SOPs, and Troubleshooting guides instantly.")

# Cache the data loading so it doesn't reload on every button click
@st.cache_data
def load_data():
    # Only load the columns we actually need
    columns_to_keep = ['number', 'category', 'subcategory', 'u_symptom', 'priority', 'closed_code']
    try:
        # Using the correct forward slashes for the cloud!
        return pd.read_csv("dataset/servicenow_incidents_5000.csv", usecols=columns_to_keep)
    except FileNotFoundError:
        st.error("Could not find the dataset. Please ensure the 'dataset/servicenow_incidents_5000.csv' file exists in your repository.")
        st.stop()
    except ValueError:
        # Fallback in case the mock data columns don't perfectly match
        return pd.read_csv("dataset/servicenow_incidents_5000.csv")

df = load_data()

# --- 2. Prompt Engineering ---
def generate_document(incident_row, doc_type):
    # Initialize the OpenAI LLM (gpt-4o-mini is fast and cost-effective)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    # Convert the row data into a dictionary
    data = incident_row.to_dict(orient='records')[0]
    
    # Define instructions based on the document type
    if doc_type == "KB Article":
        instructions = "Create a standard Knowledge Base article. Include: Title, Problem Statement, Cause, and Resolution."
    elif doc_type == "SOP":
        instructions = "Create a Standard Operating Procedure. Format as a strict, numbered step-by-step guide for a level 1 agent."
    else: 
        instructions = "Create a Troubleshooting Guide. Include: Symptoms, Verification Steps, and Fix/Workaround."

    # Create the template
    template = """
    You are an expert IT Technical Writer. Analyze the following resolved IT incident and generate documentation based on it.
    
    INSTRUCTIONS:
    {instructions}
    
    TICKET DATA:
    Incident Number: {number}
    Category: {category} / {subcategory}
    Symptom/Issue: {u_symptom}
    Priority: {priority}
    Closing Code/Resolution Notes: {closed_code}
    
    Please provide the final formatted document in Markdown below:
    """
    
    prompt = PromptTemplate(
        input_variables=["instructions", "number", "category", "subcategory", "u_symptom", "priority", "closed_code"],
        template=template
    )
    
    # Create and run the LangChain pipeline
    chain = prompt | llm
    
    # Execute the chain
    response = chain.invoke({
        "instructions": instructions,
        "number": data.get('number', 'N/A'),
        "category": data.get('category', 'N/A'),
        "subcategory": data.get('subcategory', 'N/A'),
        "u_symptom": data.get('u_symptom', 'N/A'),
        "priority": data.get('priority', 'N/A'),
        "closed_code": data.get('closed_code', 'N/A')
    })
    
    # ChatOpenAI returns an AIMessage object, we just want the text content
    return response.content

# --- 3. Main Interface ---
col1, col2 = st.columns([2, 1])
with col1:
    # Use one of the fake incident numbers as an example placeholder
    incident_input = st.text_input("Enter Incident Number (e.g., INC999001):").strip()
with col2:
    doc_selection = st.selectbox("Document Type:", ["KB Article", "SOP", "Troubleshooting Guide"])

if st.button("Generate Document", type="primary"):
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar first.")
    elif not incident_input:
        st.warning("⚠️ Please enter an incident number.")
    else:
        # Search the dataframe for the incident number
        incident_row = df[df['number'] == incident_input]
        
        if incident_row.empty:
            st.error(f"Incident {incident_input} not found in the dataset.")
        else:
            with st.spinner("Analyzing data and generating document via OpenAI..."):
                # Generate the doc
                result = generate_document(incident_row, doc_selection)
                
                # Display Results
                st.success("Generation Complete!")
                st.markdown("---")
                
                st.markdown(result)
                
                st.markdown("---")
                
                # The Download Button
                safe_filename = f"{incident_input}_{doc_selection.replace(' ', '_')}.md"
                st.download_button(
                    label="📥 Download as Markdown (.md)",
                    data=result,
                    file_name=safe_filename,
                    mime="text/markdown"
                )