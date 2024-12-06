import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BartForConditionalGeneration

# Load the models and tokenizers
@st.cache_resource
def load_models():
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("Jiraheya/samsum_model_t5_small_10_epochs")
    t5_tokenizer = AutoTokenizer.from_pretrained("Jiraheya/samsum_model_t5_small_10_epochs")
    
    bart_model = BartForConditionalGeneration.from_pretrained("Jiraheya/pegasus_xsum_samsum_model_10epoch")
    bart_tokenizer = AutoTokenizer.from_pretrained("Jiraheya/pegasus_xsum_samsum_model_10epoch")
    
    return t5_model, t5_tokenizer, bart_model, bart_tokenizer

t5_model, t5_tokenizer, bart_model, bart_tokenizer = load_models()

# Set up the Streamlit app
st.title("Dialogue Summarizer Chatbot")

# Create a dropdown for model selection
model_choice = st.selectbox(
    "Choose a model:",
    ("T5-small", "BART-large-cnn")
)

# Create a text area for user input
user_input = st.text_area("Enter your dialogue here:", height=200)

# Create a button to generate summary
if st.button("Generate Summary"):
    if user_input:
        # Prepare input for the model
        input_text = "summarize: " + user_input
        
        if model_choice == "T5-small":
            inputs = t5_tokenizer([input_text], max_length=1024, return_tensors="pt")
            summary_ids = t5_model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=60)
            summary = t5_tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        else:  # BART-large-cnn
            inputs = bart_tokenizer([input_text], max_length=1024, return_tensors="pt")
            summary_ids = bart_model.generate(inputs["input_ids"], num_beams=2, min_length=10, max_length=60)
            summary = bart_tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Display the summary
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some dialogue to summarize.")

# Add information about the app in the sidebar
st.sidebar.subheader("About the App")
st.sidebar.info(
    "This app uses fine-tuned models to summarize dialogues. "
    "Choose a model, enter your dialogue in the text area, and click 'Generate Summary' to get a concise summary."
)
st.sidebar.markdown("Models available:")
st.sidebar.markdown("- T5-small: Jiraheya/samsum_model_t5_small_10_epochs")
st.sidebar.markdown("- BART-large-cnn: Jiraheya/pegasus_xsum_samsum_model_10epoch")