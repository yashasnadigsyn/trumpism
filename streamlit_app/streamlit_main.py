import streamlit as st
import re
import os
from io import StringIO # To read uploaded file content

# --- Core N-gram Logic (Unchanged) ---

def tokenize_words(raw_text: str) -> list:
    """Splits text into words and punctuation tokens."""
    raw_text = raw_text.replace("\n", " ")
    # Consider lowercasing during tokenization for consistency
    tokens = re.findall(r'\w+|[^\w\s]', raw_text.lower())
    return tokens

def construct_n_grams(tokens: list, n: int) -> list:
    """Constructs n-grams from a list of tokens."""
    if not tokens or len(tokens) < n:
        return []
    return [tokens[i : i + n] for i in range(len(tokens) - n + 1)]

def create_ngram_frequency_dict(n_grams_list: list) -> dict:
    """Creates a nested dictionary representing n-gram frequencies."""
    ngram_freq_dict = {}
    if not n_grams_list:
        return ngram_freq_dict

    for ngram in n_grams_list:
        context = ngram[:-1]
        last_word = ngram[-1]

        current_level = ngram_freq_dict
        for word in context:
            current_level = current_level.setdefault(word, {})

        current_level[last_word] = current_level.get(last_word, 0) + 1

    return ngram_freq_dict

def predict_next_word(context: list, ngram_freq_dict: dict, n: int) -> str | None:
    """Predicts the most likely next word based on the context and frequency dictionary."""
    if not context:
        return None

    # Use the last n-1 words from the provided context (lowercase)
    actual_context = [word.lower() for word in context[-(n-1):]]

    if len(actual_context) < n - 1:
        return None # Cannot reliably predict if context is too short

    current_level = ngram_freq_dict
    for word in actual_context:
        if isinstance(current_level, dict) and word in current_level:
            current_level = current_level[word]
        else:
            return None # Context path doesn't exist

    if not isinstance(current_level, dict) or not current_level:
        return None # Reached end of path or invalid structure

    try:
        # Find the word with the highest frequency
        predicted_word = max(current_level, key=current_level.get)
        return predicted_word
    except ValueError: # Handles empty current_level dictionary
        return None

# --- Streamlit Application ---

st.set_page_config(page_title="Trumpism Predictor")
st.title("Trumpism")

# --- Explanation and Attribution at the Top ---
st.markdown(
    """
    This app predicts the next words based on an N-gram model.
    By default, it uses transcripts of Donald Trump's speeches as training data, sourced from the Miller Center.
    **Source:** [Miller Center - Presidential Speeches](https://millercenter.org/the-presidency/presidential-speeches)

    You can optionally upload your own `.txt` file to train the model on different text.
    """
)
st.markdown("---")

# --- Constants and File Handling ---
DEFAULT_FILENAME = "all_txts_combined.txt"

# --- Caching Function for Loading and Processing Data ---
@st.cache_data # Cache based on content hash (for uploaded) or filename (for default), and N
def load_and_process_data(n: int, uploaded_file_content: bytes | None = None, filename: str = DEFAULT_FILENAME):
    """Loads data, tokenizes, builds n-gram dict, and returns stats."""
    raw_text = ""
    source_name = ""
    stats = {"total_tokens": 0, "vocabulary_size": 0}

    if n <= 1:
        return None, stats, f"Error: N must be greater than 1 (received {n})."

    try:
        # 1. Read Text Data
        if uploaded_file_content is not None:
            source_name = "your uploaded file"
            try:
                # Decode assuming UTF-8, provide error handling
                raw_text = uploaded_file_content.decode("utf-8")
            except UnicodeDecodeError:
                return None, stats, f"Error: Could not decode the uploaded file. Please ensure it's UTF-8 encoded."
            except Exception as e:
                 return None, stats, f"Error reading uploaded file: {e}"
        else:
            source_name = f"'{filename}'"
            if not os.path.exists(filename):
                return None, stats, f"Error: Default training file '{filename}' not found in the current directory."
            with open(filename, 'r', encoding='utf-8') as f:
                raw_text = f.read()

        if not raw_text.strip():
            return None, stats, f"Error: The text source ({source_name}) is empty."

        # 2. Tokenize
        tokens = tokenize_words(raw_text) # Already lowercases
        if not tokens:
            return None, stats, f"Error: Could not tokenize the text from {source_name}."
        stats["total_tokens"] = len(tokens)
        stats["vocabulary_size"] = len(set(tokens))

        # 3. Build N-grams
        n_grams_list = construct_n_grams(tokens, n)
        if not n_grams_list:
            return None, stats, f"Error: Not enough tokens ({len(tokens)}) in {source_name} to form {n}-grams. Try a smaller N or a larger text source."

        # 4. Create Frequency Dictionary
        n_grams_freq_dict = create_ngram_frequency_dict(n_grams_list)
        if not n_grams_freq_dict:
            return None, stats, f"Error: Could not build the n-gram frequency model from {source_name} (dictionary is empty)."

        return n_grams_freq_dict, stats, None # Return dictionary, stats, and no error

    except FileNotFoundError:
         return None, stats, f"Error: Default training file '{filename}' not found."
    except Exception as e:
        return None, stats, f"An unexpected error occurred during data processing: {e}"

# --- User Input: File Upload ---
uploaded_file = st.file_uploader(
    "Upload your own training text file (.txt) (OPTIONAL)",
    type=["txt"],
    key="file_uploader"
)

# Determine which data source to use
use_uploaded_file = uploaded_file is not None

# --- User Input: N-gram Configuration ---
n = st.number_input(
    "Input the value of N (e.g., 3 for trigrams):",
    min_value=2,
    value=3, # Default N
    step=1,
    key='n_value'
)

# --- Load and Process Data ---
# Pass file content bytes if uploaded, otherwise use default filename
# The cache will trigger recalc if n changes or if uploaded_file content changes
# or if we switch between uploaded/default
file_content_bytes = uploaded_file.getvalue() if use_uploaded_file else None
ngram_freq_dict, stats, error_msg = load_and_process_data(n, file_content_bytes, DEFAULT_FILENAME)

# --- Display Status and Stats ---
if error_msg:
    st.error(error_msg, icon="🚨")
    st.stop() # Stop execution if model loading failed
elif ngram_freq_dict is None:
    # Should ideally be caught by error_msg, but as a fallback
    st.error("Failed to load or process the data.", icon="❓")
    st.stop()
else:
    # Display success and stats
    data_source_msg = "your uploaded file" if use_uploaded_file else f"'{DEFAULT_FILENAME}'"
    st.success(f"Successfully processed for N={n}.", icon="✅")
    st.info(f"**Training Data Stats:**\n"
            f"- Total Tokens: {stats['total_tokens']:,}\n"
            f"- Vocabulary Size: {stats['vocabulary_size']:,} (unique tokens)",
            icon="📊")

st.markdown("---") # Separator before prediction inputs

# --- User Input: Prediction ---
context = st.text_input(
    f"Enter the starting text (recommend {n-1} words):",
    # Use lowercase for default to match tokenization
    value="Make America",
    key='context_input',
    placeholder=f"Enter {n-1} words..."
)

n_words_to_predict = st.number_input(
    "How many next words to predict?",
    min_value=1,
    max_value=50, # Keep a reasonable max
    value=30,      # Default prediction length
    step=1,
    key='predict_count'
)

predict_button = st.button("Predict Next Words", key='predict_button')

# --- Prediction Logic and Output ---
if predict_button:
    if not context.strip():
        st.warning("Please enter some starting text.", icon="⚠️")
    else:
        # Tokenize context simply by splitting (already lowercased in prediction func)
        context_split = context.split() # Don't lowercase here, predict func handles it

        if not context_split:
             st.warning("Starting text contains no valid words after splitting.", icon="⚠️")
        else:
            # Optional: Warn if context length doesn't match n-1
            if len(context_split) != n - 1:
                st.caption( # Use caption for less intrusive warning
                    f"Note: Input context has {len(context_split)} words. Using the last {min(len(context_split), n-1)} words for {n}-gram prediction.",
                    # icon="ℹ️" # Icon optional for caption
                )

            with st.spinner("Generating prediction..."):
                total_prediction = context_split[:] # Start with user's original casing context
                prediction_stopped = False
                for i in range(n_words_to_predict):
                    # Context for prediction uses the required tail length
                    current_context = total_prediction # Pass the whole list, predict_next_word slices it
                    prediction = predict_next_word(current_context, ngram_freq_dict, n)

                    if prediction:
                        # Append the predicted word (which is lowercase from the model)
                        total_prediction.append(prediction)
                    else:
                        st.warning(
                            f"Prediction stopped after {i} generated words. No further prediction possible for the current context.",
                            icon="⏹️"
                        )
                        prediction_stopped = True
                        break

                st.subheader("Prediction Result:")
                # Simple join, preserves original casing for input, adds lowercase predictions
                st.info(" ".join(total_prediction))