import streamlit as st
import cohere
from pdf_extractor import extract_text_from_pdf
from qa_system import answer_question  # For Cohere PDF-based Q&A
from config import API_KEY as cohere_api_key


# Initialize the Cohere client with your API key

cohere_client = cohere.Client(cohere_api_key)

# Set page configuration
st.set_page_config(page_title="PDF Information Extractor & Q&A", page_icon="📄")

# App Title
st.title("📄 PDF Information Extractor & Cohere Q&A System 🤖")


# Find the last period in the response text
def trim_response(response_text):
    
    last_period_index = response_text.rfind('.')
    
    # If there is a period, return the text up to that point
    if last_period_index != -1:
        return response_text[:last_period_index + 1]  # Include the period
    return response_text  # Return the original text if no period is found


# Header for PDF Upload
st.header("Upload a PDF 📄")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    try:
        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(uploaded_file)
        if text:
            st.subheader("Extracted Text Preview:")
            with st.expander("Click to View Extracted Text", expanded=False):
                st.text_area("Extracted Text", text[:2000], height=300)
        else:
            st.error("No text could be extracted from the PDF.")
    except Exception as e:
        st.error(f"Error extracting text: {e}")

    # PDF-specific Q&A (using Cohere)
    st.header("Ask a Question About the PDF 🔍")
    pdf_question = st.text_input("Type your question regarding the PDF content:", placeholder="What is the main topic of the PDF?")
    
    # Integer input for token limit
    input_tokens_pdf = st.number_input(
        "Enter token limit for PDF response (int32):",  # Label for the input
        min_value=1,                                   # Minimum value for token limit
        max_value=2147483647,                          # Maximum value for int32
        value=50,                                      # Default value
        step=1,                                        # Increment step
        format="%d"                                    # Format as integer
    )

    if st.button("Get PDF Answer"):
        if pdf_question and text:
            try:
                # Use Cohere to answer questions based on the PDF content
                answer = answer_question(pdf_question, text, input_tokens_pdf)
                if answer:
                    st.write("**Answer from PDF Content:**")
                    st.success(answer)
                    st.balloons()
                else:
                    st.warning("No answer found for your question.")
            except Exception as e:
                st.error(f"Error answering question: {e}")

    # General Cohere Q&A Section
    st.header("Ask Cohere Anything 🧠")
    cohere_question = st.text_input("Ask any question (not related to PDF):", placeholder="e.g., What is artificial intelligence?")
    
    # Integer input for token limit for general questions
    input_tokens_general = st.number_input(
        "Enter token limit for Cohere response (int32):",  # Label for the input
        min_value=1,                                       # Minimum value for token limit
        max_value=2147483647,                              # Maximum value for int32
        value=50,                                          # Default value
        step=1,                                            # Increment step
        format="%d"                                        # Format as integer
    )

    if st.button("Get Cohere Answer"):
        if cohere_question:
            try:
                # Use Cohere's generate function to get an answer for general questions
                response = cohere_client.generate(
                    model='command-xlarge-nightly',  # or another appropriate model
                    prompt=cohere_question,
                    max_tokens=input_tokens_general,
                    temperature=0.7
                )
                # Display the response

                trimmed_response = trim_response(response.generations[0].text.strip())
                st.write("**Answer from Cohere:**")
                st.success(trimmed_response)
                st.balloons()
            except Exception as e:
                st.error(f"Error with Cohere response: {e}")
        else:
            st.warning("Please enter a question for Cohere.")

# Footer section for aesthetic improvement
st.markdown("---")

