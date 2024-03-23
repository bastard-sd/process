import json

from openai import OpenAI
# Placeholder function to simulate interaction with an LLM for getting word categories
def get_category_from_llm(word):
    # Replace this placeholder logic with actual LLM API call
    simulated_categories = {"python": "Programming Language", "dog": "Animal", "apple": "Fruit"}

    # Point to the local server
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

    completion = client.chat.completions.create(
        model="local-model", # this field is currently unused
        messages=[
        {"role": "system", "content": '''You are a Categorizer AI. You are given words of various meanings, genres, themes, and concepts, and your job is to return a category for the word. You are a part of a large project to categorize over 1 million words into categories. These are not grammar categories, but ontological categories. The major categories that you might run into are human description related, so please categorize the term in human centric categories. 
        
        Examples:
            bangs - hair
            mask - facewear
            ribbon - accesssory
            yellow_eyes - eye
            black_legwear - legwear
            blush - expression
            brown_footwear - footwear
            fishing - action
            fishing_line - object
            hammer_and_sickle - weapon
            hat - headgear
            jacket - outerwear
            sitting - action
            simple_background - background
            ice - object
            eye_contact - expression

        OUITPUT: Only output the the single phrase that is the category. Do not output anything else. You are allotted a maximum of 3 words spearated by underscore for your category phrase. Prefer to use fewer words, and only 1 word if possible.'''},
        {"role": "user", "content": f"Given this word, what is the category: {word}"}
        ],
        temperature=0.25,
    )

    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

# Function to save the current state (last processed index) and the master dictionary
def save_state(master_dict, last_index):
    state = {"last_index": last_index}
    with open('categorization_state.json', 'w') as file:
        json.dump(state, file)
    # Save the master dictionary separately
    with open('master_dict.json', 'w') as file:
        json.dump(master_dict, file)

# Function to load the saved state, if it exists
def load_state():
    try:
        with open('categorization_state.json', 'r') as file:
            state = json.load(file)
            last_index = state['last_index']
    except FileNotFoundError:
        last_index = -1  # Indicates no words have been processed yet

    # Load the master dictionary
    try:
        with open('master_dict.json', 'r') as file:
            master_dict = json.load(file)
    except FileNotFoundError:
        master_dict = None

    return master_dict, last_index

# Main function to categorize words, with resume logic and saving the master dictionary
def categorize_words(json_file_path, initial_master_dict=None):
    # Attempt to load the saved state and master dictionary
    master_dict, last_processed_index = load_state()

    # If there's no saved master dictionary, use the initial master dictionary
    if master_dict is None:
        master_dict = initial_master_dict if initial_master_dict is not None else {}

    with open(json_file_path, 'r') as file:
        words = json.load(file)

    # Process each word, starting from the last processed index + 1
    for i in range(last_processed_index + 1, len(words)):
        word = words[i]
        category = get_category_from_llm(word)

        # Add word to the appropriate category in the master dictionary
        if category in master_dict:
            master_dict[category].append(word)
        else:
            master_dict[category] = [word]

        # Save the current state and master dictionary after processing each word
        save_state(master_dict, i)

    return master_dict

# Example usage
json_file_path = '../networks/mldb/classes.json'  # Update this path
initial_master_dict = {}  # Supply the initial master dictionary here if needed
master_dict = categorize_words(json_file_path, initial_master_dict)

print("Finished Processing Master Dictionary")
