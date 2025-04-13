import re

# Construct n-grams
def tokenize_words(raw_text: str) -> list:
    raw_text = raw_text.replace("\n", " ")
    tokens = re.findall(r'\w+|[^\w\s]', raw_text)
    return tokens

def construct_n_grams(tokens: list, n: int) -> list:
    if len(tokens) < n: return []
    return [tokens[i : i + n] for i in range(len(tokens) - n + 1)]

def create_ngram_frequency_dict(n_grams_list: list) -> dict:
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

def predict_next_word(context: list, ngram_freq_dict: dict, n: int) -> str:
    current_level = ngram_freq_dict
    for word in context:
        if word in current_level:
            current_level = current_level[word]
            # Check if dict
            if not isinstance(current_level, dict):
                print(f"Warning: Unexpected structure found after '{word}'. Context may be too short in training data.")
                return None 
        else:
            return None
        
    if not isinstance(current_level, dict) or not current_level:
        return None
    
    predicted_word = max(current_level, key=current_level.get)
    return predicted_word

raw_text = open("all_txts_combined.txt").read()
tokens = tokenize_words(raw_text)
n = int(input("Input the value of n(n>1): "))
context = input("Enter the input text to predict the next word(no punctuations): ")
context_split = context.split()
n_words_to_predict = int(input("How many next n words to predict: "))
if len(context.split()) != n-1: print(f"For {n}-grams, you have to enter {n-1} words!")
n_grams_ = construct_n_grams(tokens, n)
n_grams_freq_dict = create_ngram_frequency_dict(n_grams_)
total_prediction = context_split
for i in range(n_words_to_predict):
    prediction = predict_next_word(total_prediction[-(n-1):], n_grams_freq_dict, n)
    if prediction:
        total_prediction.append(prediction)
    else:
        print(total_prediction)
        break

print(" ".join(total_prediction))
