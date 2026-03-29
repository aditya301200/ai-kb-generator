import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# --- 1. Setup & Configuration ---
# Initialize the local LLM using the ultra-lightweight Llama 3.2 1B model
llm = Ollama(model="llama3.2:1b", temperature=0.3)

# Cache the data loading so it doesn't reload on every button click
@st.cache_data
def load_data():
    # Only load the columns we actually need to save RAM
    columns_to_keep = ['number', 'category', 'subcategory', 'u_symptom', 'priority', 'closed_code']
    try:
        return pd.read_csv("D:\Aditya\Downloads_D\AI Engineer\project_1\dataset\servicenow_incidents_5000.csv", usecols=columns_to_keep)
    except FileNotFoundError:
        st.error("Could not find 'servicenow_incidents_5000.csv'. Please ensure it's in the same folder.")
        st.stop()
    except ValueError:
        # Fallback in case the mock data columns don't perfectly match
        return pd.read_csv("D:\Aditya\Downloads_D\AI Engineer\project_1\dataset\servicenow_incidents_5000.csv")

df = load_data()

# --- 2. Prompt Engineering ---
def generate_document(incident_row, doc_type):
    # Convert the row data into a dictionary
    data = incident_row.to_dict(orient='records')[0]
    
    # Define instructions based on the document type
    if doc_type == "KB Article":
        instructions = "Create a short Knowledge Base article. Include: Title, Problem, Cause, and Resolution."
    elif doc_type == "SOP":
        instructions = "Create a brief Standard Operating Procedure. Use numbered steps."
    else: 
        instructions = "Create a short Troubleshooting Guide. Include: Symptoms and Fix."

    # Keep the prompt concise for the smaller model
    template = """
    You are an IT Technical Writer. Create documentation based on this resolved IT incident.
    
    INSTRUCTIONS: {instructions}
    
    TICKET DATA:
    Number: {number}
    Category: {category} / {subcategory}
    Issue: {u_symptom}
    Priority: {priority}
    Resolution: {closed_code}
    
    Output the final document in Markdown format below:
    """
    
    prompt = PromptTemplate(
        input_variables=["instructions", "number", "category", "subcategory", "u_symptom", "priority", "closed_code"],
        template=template
    )
    
    chain = prompt | llm
    
    response = chain.invoke({
        "instructions": instructions,
        "number": data.get('number', 'N/A'),
        "category": data.get('category', 'N/A'),
        "subcategory": data.get('subcategory', 'N/A'),
        "u_symptom": data.get('u_symptom', 'N/A'),
        "priority": data.get('priority', 'N/A'),
        "closed_code": data.get('closed_code', 'N/A')
    })
    
    return response

# --- 3. Streamlit UI ---
st.set_page_config(page_title="AI KB Generator", layout="centered")

st.title("⚡ Lightweight AI Knowledge Generator")
st.write("Running locally on Llama 3.2 (1B) - Optimized for Low RAM.")

col1, col2 = st.columns([2, 1])
with col1:
    incident_input = st.text_input("Enter Incident Number (e.g., INC12345):").strip()
with col2:
    doc_selection = st.selectbox("Document Type:", ["KB Article", "SOP", "Troubleshooting Guide"])

if st.button("Generate Document", type="primary"):
    if not incident_input:
        st.warning("Please enter an incident number.")
    else:
        # Search the dataframe for the incident number
        incident_row = df[df['number'] == incident_input]
        
        if incident_row.empty:
            st.error(f"Incident {incident_input} not found in the dataset.")
        else:
            with st.spinner("Generating document... (this may take a minute on smaller systems)"):
                result = generate_document(incident_row, doc_selection)
                
                st.success("Generation Complete!")
                st.markdown("---")
                
                # Show the result on the screen
                st.markdown(result)
                
                st.markdown("---")
                
                # --- NEW: The Download Button ---
                # Clean up the file name (e.g., INC12345_KB_Article.md)
                safe_filename = f"{incident_input}_{doc_selection.replace(' ', '_')}.md"
                
                st.download_button(
                    label="📥 Download as Markdown (.md)",
                    data=result,
                    file_name=safe_filename,
                    mime="text/markdown"
                )