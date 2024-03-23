import json

def find_keys_with_substrings(dictionary, substrings):
    """
    Finds keys containing any of the given substrings, returning them and their values.
    
    Parameters:
        dictionary (dict): The dictionary to search through.
        substrings (list): A list of substrings to search for in the keys.
        
    Returns:
        dict: A dictionary with keys containing any of the substrings and their values.
    """
    # Initialize an empty dictionary to store found items
    found_items = {}

    # Iterate over each item in the dictionary
    for key, value in dictionary.items():
        # Check if the key contains any of the substrings
        if any(substring in key for substring in substrings):
            found_items[key] = value
    
    return found_items

def merge_keys(dictionary, keys_to_merge, new_key):
    """
    Merges values from specified keys into a new or existing key.
    
    Parameters:
        dictionary (dict): The dictionary to operate on.
        keys_to_merge (list): A list of keys whose values should be merged.
        new_key (str): The key under which to store the merged values.
    """
    # Ensure the target key exists in the dictionary
    if new_key not in dictionary:
        dictionary[new_key] = []

    for key in keys_to_merge:
        print(key)
        if key in dictionary:
            # Extend the list under new_key with the values from the current key
            if isinstance(dictionary[key], list):
                print(dictionary[key])
                print(dictionary[new_key])
                dictionary[new_key].extend(dictionary[key])
            else:
                dictionary[new_key].append(dictionary[key])
            # Optionally, remove the merged key
            del dictionary[key]

def find_keys_with_substring(dictionary, substring):
    """
    Finds keys containing the given substring, returning them and their values.
    
    Parameters:
        dictionary (dict): The dictionary to search through.
        substring (str): The substring to search for in the keys.
        
    Returns:
        dict: A dictionary with keys containing the substring and their values.
    """
    return {key: value for key, value in dictionary.items() if substring in key}


def load_state():
    # Load the master dictionary
    try:
        with open('combined_master_dict15.json', 'r') as file:
            master_dict = json.load(file)
    except FileNotFoundError:
        master_dict = None

    return master_dict

# Example usage
dictionary = load_state()

# Search by substring example
# substring = "state"
# found_items = find_keys_with_substring(dictionary, substring)

# substrings = ["garden", "plant"]
substrings = ['experience']
found_items = find_keys_with_substrings(dictionary, substrings)

# Save the master dictionary separately
with open('text.json', 'w') as file:
    json.dump(found_items, file)

# Merge keys example
keys_to_merge = found_items.keys()
print(keys_to_merge)

# merge_keys(dictionary, keys_to_merge, "character2")

# # Save the master dictionary separately
# with open('combined_master_dict15.json', 'w') as file:
#     json.dump(dictionary, file)