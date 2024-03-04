from flask import Flask, request, jsonify
from PIL import Image
from concurrent.futures import ThreadPoolExecutor,  as_completed
import spacy
import inflect
from colour import Color
import random
import pprint
import os

pp = pprint.PrettyPrinter(indent=4)

### Spacy Functions

def convert_to_single_line(multiline_string):
    # Split the string into lines
    lines = multiline_string.splitlines()
    
    # Join the lines with a space, but first strip trailing and leading whitespace from each line
    single_line = ' '.join(line.rstrip().lstrip() for line in lines)
    
    return single_line

def filter_words(word_dict, larger_string):
    # Filter the words that are not found in the larger_string
    filtered_words = {key: value for key, value in word_dict.items() if str(value) not in larger_string}
    return filtered_words

def loadinflectmodel():
    ie = inflect.engine()
    return ie

def load_model(model_name, gpu_device_id=None, download=True):
    """
    Loads a spaCy model with the given name and returns it. If the model cannot
    be found, it will be downloaded if download=True.

    vbnet

    Args:
    - model_name (str): the name of the spaCy model to load
    - download (bool): whether to download the model if it cannot be found

    Returns:
    - nlp (spacy.Language): the loaded spaCy model
    """
    original_environ = os.environ.get("CUDA_VISIBLE_DEVICES")

    try:
        if gpu_device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device_id)  # Set the specific GPU
        nlp = spacy.load(model_name)
    finally:
        # If the original CUDA_VISIBLE_DEVICES was None, it means it was not set, so we delete it.
        if original_environ is None:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_environ  # Restore the original environment variable

    return nlp

def concat_pose_phrases(nlp, ie, phrases):
    action_tokens = [
        'sit',
        'stand',
        'lay',
        'kneel',
        'crouch',
        'squat',
        'lean',
        'crawl',
        'jump',
        'walk',
        'run',
        'bend'
    ]
            
    pose_token = 'pose'
    token_to_replace = ''
    new_phrases = []

    for phrase in phrases:
        found = False
        for action in action_tokens:
            if action in phrase:
                if action == 'sit' and 'position' in phrase:
                    continue
                token_to_replace = ie.present_participle(action)
                found = True
                break
        if not found:
            new_phrases.append(phrase)

    if token_to_replace != '':
        for i, phrase in enumerate(new_phrases):
            if pose_token in phrase:
                new_phrases[i] = token_to_replace + ' ' + phrase
    
    return new_phrases

def extract_root_word_phrase_pairs(nlp, input_list):
    tokenized_list = [nlp(text) for text in input_list]
    root_groups = {}

    for tokens in tokenized_list:
        for token in tokens:
            if token.dep_ == 'ROOT':
                root = str(token).lower()
                supplemental_roots = []
                
                # this is where you can make 'translations' one at a time. maybe json this later.
                if root == 'choker':
                    if 'neck bell' in input_list:
                        supplemental_roots.append('bell')
                        
                if root == 'hair':
                    if 'female pubic hair' in tokens.text:
                        root = 'female pubic hair'
                    elif 'male pubic hair' in tokens.text:
                        root = 'male pubic hair'
                    elif 'pubic hair' in tokens.text:
                        root = 'pubic hair'
                    elif 'leg hair' in tokens.text:
                        root = 'leg hair'
                    elif 'chest hair' in tokens.text:
                        root = 'chest hair'
                        
                if root == 'reverse':
                    if 'reverse cowgirl position' in tokens.text:
                        root = 'reverse cowgirl position'
                    elif 'reverse upright straddle' in tokens.text:
                        root = 'reverse upright straddle'
                if root == 'piercing':
                    if 'nipple piercing' in tokens.text:
                        root = 'nipple piercing'
                    elif 'navel piercing' in tokens.text:
                        root = 'navel piercing'

                for supp in supplemental_roots:
                    if supp not in root_groups:
                        root_groups[supp] = set() # Use a set to ensure uniqueness
                    root_groups[supp].add(tokens)
                    
                if supplemental_roots != []:
                    break
            
                if root not in root_groups:
                    root_groups[root] = set() # Use a set to ensure uniqueness
                root_groups[root].add(tokens)
                break

    # Convert sets to lists before returning
    for key, value in root_groups.items():
        root_groups[key] = list(value)

    return root_groups

def is_this_a_color(color):
    try:
        # Converting 'deep sky blue' to 'deepskyblue'
        normalized_color = color.replace(" ", "")
        Color(normalized_color)
        return True
    except ValueError:
        return False

def combine_root_and_phrases(nlp, ie, root, phrases):
    prefixes = []
    suffixes = []
    root_doc = nlp(root)
    plural_root = ie.plural(root) if root_doc[0].tag_ != "NNS" else root
    
    use_plural = False
    for phrase in phrases:
        if str(plural_root) in str(phrase):
            use_plural = True
            break
    root_to_use = plural_root if use_plural else root
    modified_root = root_to_use
    
    for phrase in phrases:
        if 'see-through' in str(phrase):
            if str(root) != 'see':
                prefixes.append(str(phrase))
            else:
                modified_root = str(phrase)
            continue
        if 'hugging own legs' in str(phrase):
            if str(root) != 'hugging':
                if str(phrase) not in str(modified_root):
                    modified_root = str(modified_root) + ' ' + str(phrase)
            else:
                modified_root = str(phrase)
            continue
        if 'female pubic hair' in str(phrase):
            modified_root = 'female pubic hair'
            continue
        if 'male pubic hair' in str(phrase):
            modified_root = 'male pubic hair'
            continue
        if 'pubic hair' in str(phrase):
            modified_root = 'pubic hair'
            continue
        if 'leg hair' in str(phrase):
            modified_root = 'leg hair'
            continue
        if 'chest hair' in str(phrase):
            modified_root = 'chest hair'
            continue
        if 'reverse upright straddle' in str(phrase):
            modified_root = 'reverse upright straddle'
            continue
        if 'reverse cowgirl position' in str(phrase):
            modified_root = 'reverse cowgirl position'
            continue
        if 'cowgirl position' in str(phrase):
            modified_root = 'cowgirl position'
            continue
        doc = nlp(phrase)
        for token in doc:
            if str(token) == str(root_to_use):
                continue
            elif str(token) == str(root):
                continue
            if token.dep_ in ['compound'] and token.pos_ in ['NOUN', 'PROPN']:
                # modified_root = str(modified_root) + ' ' + token.text
                if token.text not in str(modified_root):
                    modified_root = token.text + ' ' + str(modified_root)
            elif token.dep_ in ['amod', 'advmod', 'compound']:
                if is_this_a_color(str(token.text)):
                    if token.text not in prefixes:
                        prefixes.insert(0, token.text)
                else:
                    # right here is the opportunity to ... clean up this prefix list.
                    if token.text not in prefixes:
                        prefixes.append(token.text)
            elif token.pos_ in ['ADP']:
                if str(root_to_use) in str(phrase):
                    new_phrase = str(phrase).replace(str(root_to_use), '').strip()
                elif str(root) in str(phrase):
                    new_phrase = str(phrase).replace(str(root), '').strip()
                else:
                    new_phrase = None

                if new_phrase and str(new_phrase) not in suffixes:
                    suffixes.append(str(new_phrase))
                break
            elif token.pos_ in ['NOUN', 'PROPN']:
                cleaned_phrase = str(phrase).replace('(','').replace(')','')
                if str(root_to_use) in cleaned_phrase:
                    new_phrase = cleaned_phrase.replace(str(root_to_use), '').strip()
                elif str(root) in cleaned_phrase:
                    new_phrase = cleaned_phrase.replace(str(root), '').strip()
                else:
                    new_phrase = cleaned_phrase
                    
                if str(new_phrase) not in str(modified_root):
                    modified_root = str(new_phrase) + ' ' + str(modified_root)
                # Break the inner loop...
                break
            elif token.pos_ in ['VERB']:
                if str(root_to_use) in str(phrase):
                    new_phrase = str(phrase).replace(str(root_to_use), '').strip()
                elif str(root) in str(phrase):
                    new_phrase = str(phrase).replace(str(root), '').strip()
                else:
                    new_phrase = phrase
                    
                if str(new_phrase) not in str(modified_root):
                    modified_root = str(modified_root) + ' ' + str(new_phrase)
                # Break the inner loop...
                break
        else:
            # Continue if the inner loop wasn't broken.
            continue
        # Inner loop was broken, break the outer.
        # break
                
    
    filtered_prefixes = remove_repeated_words(nlp, ie, prefixes)
    # Fake descriptor is usually first descriptor up front.
    for i in range(len(filtered_prefixes)):
        if filtered_prefixes[i] == 'fake':
            filtered_prefixes.pop(i)
            filtered_prefixes.insert(0, 'fake')
    filtered_suffixes = remove_repeated_words(nlp, ie, suffixes)
    # or right here is the opportunity to cleanup the prefix list.
    joined_suffixes = ' and '.join(filtered_suffixes)
    
    return ' '.join(filtered_prefixes + [modified_root] + [joined_suffixes])

def remove_repeated_words(nlp, ie, combined_phrases, blacklist=[]):
    combined_phrases = list(set(combined_phrases))
    for i in range(len(combined_phrases)):
        if " " not in combined_phrases[i] and combined_phrases[i] not in blacklist:
            for j in range(len(combined_phrases)):
                word_doc = nlp(combined_phrases[i])
                plural_noun = ie.plural(combined_phrases[i]) if word_doc[0].tag_ != "NNS" else combined_phrases[i]
                singular_noun = ie.singular_noun(plural_noun)
                
                if singular_noun == False:
                    singular_noun = plural_noun

                if i != j and plural_noun in combined_phrases[j]:
                    combined_phrases[i] = '_REMOVE_'
                    break
                elif i != j and singular_noun in combined_phrases[j]:
                    combined_phrases[i] = '_REMOVE_'
                    break

    combined_phrases = [phrase for i, phrase in enumerate(combined_phrases) if phrase != '_REMOVE_']
    return combined_phrases

def process_gestures(img_json):
    suffix_dict = {}

    # Define the mapping from numerical or percentage values to words
    percentage_to_word = {
        '0-10': 'barely',
        '10-40': 'slightly',
        '40-60': 'half',
        '60-85': 'almost',
        '85-100': 'completely'
    }

    # Iterate over each gesture in the img_json
    for gesture in img_json['gesture']:
        # Get the prefix and suffix of each gesture
        prefix = list(gesture.keys())[0]
        suffix = gesture['gesture']

        # If the prefix is not in the dictionary, initialize an empty list for it
        suffix_dict.setdefault(prefix, []).append(suffix)

        # Now pre-process the rules for each prefix to create `blink` and `body` new prefixes
        for prefix in list(suffix_dict.keys()):  # use list() to avoid RuntimeError
            if prefix == 'face':
                suffixes = suffix_dict[prefix]
                blink_indices = [i for i, suffix in enumerate(suffixes) if 'blink' in suffix]
                head_indices = [i for i, suffix in enumerate(suffixes) if 'head' in suffix]

                # Combining both indices and sorting in reversed order
                combined_indices = sorted(blink_indices + head_indices, reverse=True)

                for i in combined_indices:
                    # Using the index, determine if it's a 'blink' or 'head'
                    suffix = suffixes[i]
                    if 'blink' in suffix:
                        suffix_dict.setdefault('blink', []).append(suffixes.pop(i))
                    elif 'head' in suffix:
                        suffix_dict.setdefault('head', []).append(suffixes.pop(i))

    # Now process the rules for each prefix
    for prefix, suffixes in suffix_dict.items():
        if prefix == 'face':
            if 'facing left' in suffixes and 'facing center' in suffixes:
                suffixes.remove('facing left')
                suffixes.remove('facing center')
                suffixes.append('facing center left')

            if 'facing right' in suffixes and 'facing center' in suffixes:
                suffixes.remove('facing right')
                suffixes.remove('facing center')
                suffixes.append('facing center right')

        if prefix == 'blink':
            if 'blink left eye' in suffixes and 'blink right eye' in suffixes:
                suffixes.remove('blink left eye')
                suffixes.remove('blink right eye')
                suffixes.append('blinking')
            
        if prefix == 'body':
            if 'leaning left' in suffixes and 'leaning right' in suffixes:
                suffixes.remove('leaning left')
                suffixes.remove('leaning right')
                suffixes.append('leaning')

            if 'raise left hand' in suffixes and 'raise right hand' in suffixes:
                suffixes.remove('raise left hand')
                suffixes.remove('raise right hand')
                suffixes.append('raise hands')

            if 'i give up' in suffixes:
                suffixes.remove('i give up')
                
        if prefix == 'iris':
            if 'facing center' in suffixes:
                suffixes.remove('facing center')
                if 'looking center' not in suffixes:
                    suffixes.append('looking center')
            order_dict = {'looking center': 0, 'looking right': 1, 'looking left': 2, 'looking up': 3, 'looking down': 4}
            suffixes.sort(key=lambda x: order_dict.get(x, 5))
            suffixes = [suffixes[0]] + [suffix.replace('looking ', '') for suffix in suffixes[1:]]
            suffix_dict[prefix] = [' and '.join(suffixes)]
            
        if prefix == 'head':
            if 'head up' in suffixes and 'head down' in suffixes:
                suffixes.remove('head down')
                 
    ordered_prefixes = ['face', 'head', 'blink', 'iris']
    processed_string = ' '.join(' '.join(suffix_dict.get(prefix, [])) for prefix in ordered_prefixes if suffix_dict.get(prefix, []))
    
    words = processed_string.split()
    percent = None
    for word in words:
        if '%' in word:
            replace_word = word
            percent = int(word.strip('%'))
            break
        
    if percent is not None:
        percent_key = next((k for k in percentage_to_word.keys() if percent >= int(k.split('-')[0]) and percent <= int(k.split('-')[1])), None)
        if percent_key:
            processed_string = processed_string.replace(replace_word, percentage_to_word[percent_key])
        
    print(processed_string)  # prints 'facing left mouth 15% open   looking center looking up'
    
    
    return processed_string

### Dictionary Manipulation

def flatten(dictionary):
    flat_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    flat_dict.update(flatten(item))
                else:
                    flat_dict[key] = value
        elif isinstance(value, dict):
            flat_dict.update(flatten(value))
        else:
            flat_dict[key] = value
    return flat_dict

def transform_input(input_list, master_dictionary):
    transformed_input = {}
    for input_key, input_value in input_list.items():
        for feature_key, feature_values in master_dictionary.items():
            for feature_value in feature_values:
                if input_key == feature_value:
                    # if feature_key not in transformed_input:
                    #     transformed_input[feature_value] = []
                    # transformed_input[feature_value].append(input_value)
                    transformed_input[feature_value] = input_value
                else:
                    if input_key not in transformed_input:
                        if str(input_key) == '1girl':
                            key = 'girl'
                        elif type(input_key) is int:
                            key = 'number'
                        else:
                            key = input_key
                        # transformed_input[str(key)] = []
                        # transformed_input[str(key)].append(input_value)
                        transformed_input[str(key)] = input_value

    # Adding the 'age' key directly, as it is not part of the transformation
    if 'age' in input_list:
        transformed_input['age'] = input_list['age']
    if 'face_direction' in input_list:
        transformed_input['face_direction'] = input_list['face_direction']
    # if len(meta_json['models']) == 2:
    #     transformed_input['female_name'] = female_name
    #     transformed_input['male_name'] = male_name
    # if len(meta_json['models']) == 3:
    #     transformed_input['female_name'] = female_name
    #     transformed_input['female2_name'] = female2_name
    #     transformed_input['male_name'] = male_name

    return transformed_input

def find_intermediary_keys(transformed_input_list, master_dictionary):
    parent_keys = []

    def traverse_hierarchy(dictionary, key_chain):
        for key, value in dictionary.items():
            new_key_chain = key_chain + [key]
            if isinstance(value, list):
                # Check if any item in the value list is present in the transformed_input_list keys
                if any(item in transformed_input_list for item in value):
                    parent_keys.append(new_key_chain) # Removed the slicing
            elif isinstance(value, dict):
                traverse_hierarchy(value, new_key_chain)

    traverse_hierarchy(master_dictionary, [])
    return [key_chain for key_chain in set(map(tuple, parent_keys))]

master_dictionary = {
    "face": {
        "features": {
            "the_hair":[
                "hairstyle",
                "hair"
            ],
            "the_eyes": [
                "eyes",
                "eyelashes",
                "eyepatch",
                "eye",
                "eyed",
                "sclera",
                "eyebrows",
                "eyewear"
            ],
            "the_mouth": [
                "tooth",
                "teeth",
                "lips",
                "tongue",
                "mouth",
                "lips_parted"
            ],
            "the_rest": [
                "ears",
                "nose",
                "freckles",
                "forehead",
                "cheeks"
            ],
            "the_facial_hair": [
                "sideburns",
                "stubble",
                "mustache",
                "beard",
                "whiskers"
            ],
            "the_makeup": [
                "eyeshadow",
                "blush",
                "lipstick",
                "eyeliner",
                "makeup"
            ]
        },
        "expressions": [
            "grin",
            "stare",
            "smile",
            "frown",
            "jitome",
            "teardrop"
        ]
    },
    "body": {
        "parts": {
            "external": {
                "the_arms": [
                    "shoulders",
                    "arms",
                    "arm",
                    "elbows",
                    "wrists",
                    "hands",
                    "fingers",
                    "thumbs",
                    "fist",
                    "palms",
                    "nails",
                    "nail_polish",
                    "veins"
                ],
                "the_chest": [
                    "chest",
                    "breast",
                    "breasts",
                    "cleavage",
                    "areolae",
                    "nipple",
                    "nipples",
                    "pectorals"
                ],
                "the_torso": [
                    "torso",
                    "abs",
                    "belly",
                    "navel",
                    "navel_piercing",
                    "back",
                    "spine"
                ],
                "the_genitals": [
                    "vagina",
                    "pussy",
                    "cunt",
                    "twat",
                    "labia",
                    "vulva",
                    "penis",
                    "cock",
                    "dick",
                    "erection",
                    "testicles"
                ],
                "the_lower_body": [
                    "anus",
                    "ass",
                    "bottom",
                    "hips",
                    "legs",
                    "knees",
                    "thighs",
                    "shin",
                    "feet",
                    "foot",
                    "barefoot",
                    "toe",
                    "toes",
                    "toenails"
                ]
            },
            "internal": [
                "brain",
                "bone",
                "bones",
                "joints",
                "tendon",
                "limbs",
                "skull"
            ],
            "animal_parts": [
                "tail",
                "fins",
                "paws",
                "snout",
                "antennae",
                "gills",
                "antlers",
                "tentacles",
                "fin"
            ]
        },
        "status": [
            "wet",
            "nude",
            "amputee",
            "bleeding",
            "sleeping",
            "ill"
        ],
        "conditions": [
            "tattoo",
            "mole",
            "scar",
            "wound",
            "bruise",
            "cut",
            "blood",
            "tan",
            "tanline",
            "bulge"
        ]
    }
}

def process_face_and_looking_phrases(face, looking):
    if face == None or face == False:
        return looking
    face_words = face.split()
    looking_words = looking.split()
    
    if face_words[-2] == 'looking' and 'looking' in looking_words:
        looking_words.remove('looking')
        result = face_words + looking_words
        return ' '.join(result)
    else:
        return face + ' ' + looking

def filter_empty_strings(value):
    return [item for item in value if item]

# Custom filter to reject elements from items_to_process in the items list
def reject_items(items, items_to_process):
    return list(item for item in items if item not in items_to_process)

# Custom filter to check if all elements of items_to_process exist in the items list
def is_subset_of(items_to_process, items):
    return all(item in items for item in items_to_process)

def only_elements(value, elements):
    return set(value) == set(elements)

def random_selection(chances):
    total = sum(chances)
    rnd = random.random() * total
    cumulative_chance = 0
    for i, chance in enumerate(chances):
        cumulative_chance += chance
        if rnd < cumulative_chance:
            return i
    return None

def all_filter(value):
    return all(bool(item) for item in value)

def update_and_remove_items(processed_list, category, sub_items):
    for sub_item in sub_items:
        if sub_item in [str(a.text).strip() for a in processed_list[category]]:
            processed_list[sub_item] = [sub_item]
            processed_list[category] = [item for item in processed_list[category] if str(item.text).strip() != sub_item]

## ---  
# Your existing import statements and code, such as model loading, remain here

def process_input_tags(nlp, ie, repeated_word_blacklist, input_list):
    input_list = [item.strip() for item in input_list.split(',') if item.strip()]
    processed_list = extract_root_word_phrase_pairs(nlp, input_list)

    print(processed_list)
    def update_and_remove_items(processed_list, category, sub_items):
        for sub_item in sub_items:
            if sub_item in [str(a.text).strip() for a in processed_list[category]]:
                processed_list[sub_item] = [sub_item]
                processed_list[category] = [item for item in processed_list[category] if str(item.text).strip() != sub_item]

    categories = ['reverse', 'piercing', 'hair']
    sub_items_map = {
        'reverse': ['reverse upright straddle', 'reverse cowgirl position'],
        'piercing': ['nipple piercing', 'navel piercing'],
        'hair': ['leg hair', 'chest hair']
    }

    for category in categories:
        if category in processed_list:
            update_and_remove_items(processed_list, category, sub_items_map[category])

    combined_phrases = [combine_root_and_phrases(nlp, ie, key, phrases).strip() for key, phrases in processed_list.items()]

    filtered_supplemental_phrases = remove_repeated_words(
        nlp, ie, combined_phrases, 
        ['reverse upright straddle', 'reverse cowgirl position', 'nipple piercing', 'navel piercing', 'leg hair', 'chest hair'] + repeated_word_blacklist)
    fully_processed_phrases = concat_pose_phrases(nlp, ie, filtered_supplemental_phrases)
    fully_processed_phrase_dict = extract_root_word_phrase_pairs(nlp, fully_processed_phrases)

    # Iterate through the keys and values in the dictionary
    # And customize particular phrases
    fully_processed_phrase_dict_modified = {}
    for key, value in fully_processed_phrase_dict.items():
        if key == '1girl' and str(value)[1:-1] == '1girl':
            key = 'girl'
        if key == '1boy' and str(value)[1:-1] == '1boy':
            key = 'boy'
        if key == 'polish' and str(value)[1:-1] == 'nail polish':
            key = 'nail_polish'
        if key == 'belt' and str(value)[1:-1] == 'garter belt':
            key = 'garter_belt'
        if key == 'from' and str(value)[1:-1] == 'from behind':
            key = 'from_behind'
        if key == 'clothes' and str(value)[1:-1] == 'open clothes':
            key = 'open_clothes'
        if key == 'shirt' and  'open' in str(value)[1:-1] and 'shirt' in str(value)[1:-1]:
            key = 'open_shirt'
        if key == 'open' and  'open' in str(value)[1:-1] and 'shirt' in str(value)[1:-1]:
            key = 'open_shirt'
        if key == 'parted' and str(value)[1:-1] == 'lips parted':
            key = 'lips_parted'
        if key == 'piercing' and str(value)[1:-1] == 'navel piercing':
            key = 'navel_piercing'
        if key == 'see' and str(value)[1:-1] == 'see-through':
            key = 'see_through'
        if key == 'background' and str(value)[1:-1] == 'blurry background':
            key = 'blurry_background'
        if key == 'support' and str(value)[1:-1] == 'arm support':
            key = 'arm_support'
        if key == 'body' and str(value)[1:-1] == 'full body':
            key = 'full_body'
        if key == 'body' and str(value)[1:-1] == 'upper body':
            key = 'upper_body'
        if key == 'hugging' and str(value)[1:-1] == 'hugging own legs':
            key = 'hugging_own_legs'
        if key == 'pull' and str(value)[1:-1] == 'clothes pull':
            key = 'clothes_pull'
        if key == 'focus' and str(value)[1:-1] == 'solo focus':
            key = 'solo_focus'
        if key == 'crotch' and str(value)[1:-1] == 'pov crotch':
            key = 'pov_crotch'
        if key == 'male pubic hair' and str(value)[1:-1] == 'male pubic hair':
            key = 'male_pubic_hair'
        if key == 'female pubic hair' and str(value)[1:-1] == 'female pubic hair':
            key = 'female_pubic_hair'
        if key == 'pubic hair' and str(value)[1:-1] == 'pubic hair':
            key = 'pubic_hair'
        if key == 'girl' and str(value)[1:-1] == 'girl on top':
            key = 'girl_on_top'
        if key == 'reverse cowgirl position' and 'reverse cowgirl position' in str(value)[1:-1]:
            key = 'reverse_cowgirl_position'
        if key == 'reverse upright straddle' and 'reverse upright straddle' in str(value)[1:-1]:
            key = 'reverse_upright_straddle'
        if key == 'navel piercing' and 'navel piercing' in str(value)[1:-1]:
            key = 'navel_piercing'
        if key == 'nipple piercing' and 'nipple piercing' in str(value)[1:-1]:
            key = 'nipple_piercing'
        if key == 'spread' and str(value)[1:-1] == 'legs spread':
            key = 'legs_spread'
        if key == 'sex' and str(value)[1:-1] == 'sex from behind':
            key = 'sex_from_behind'
        if key == 'girl' and str(value)[1:-1] == 'girl on top':
            key = 'girl_on_top'
        if key == 'position' and str(value)[1:-1] == 'cowgirl position':
            key = 'cowgirl_position'
        if key == 'sucking' and str(value)[1:-1] == 'breast sucking':
            key = 'breast_sucking'
        if key == 'bed' and str(value)[1:-1] == 'bed on':
            key = 'on_bed'
        if key == 'grab' and str(value)[1:-1] == 'ass grab':
            key = 'ass_grab'
        if key == 'behind' and str(value)[1:-1] == 'from behind':
            key = 'from_behind'
        if key == 'straddle' and str(value)[1:-1] == 'upright straddle':
            key = 'upright_straddle'
        if key == 'focus' and str(value)[1:-1] == 'ass focus':
            key = 'ass_focus'
        if key == 'boob' and str(value)[1:-1] == 'side boob':
            key = 'side_boob'
        if key == 'press' and str(value)[1:-1] == 'breast press':
            key = 'breast_press'
        if key == 'penetration' and str(value)[1:-1] == 'imminent penetration':
            key = 'imminent_penetration'
        if key == 'box' and str(value)[1:-1] == 'tissue box':
            key = 'tissue_box'
        if key == 'grab' and str(value)[1:-1] == 'ass torso grab':
            key = 'ass_torso_grab'
        if key == 'close' and str(value)[1:-1] == 'close -up':
            key = 'close_up'
        if key == 'top' and str(value)[1:-1] == 'top bottom top-down -up':
            key = 'face_down_ass_up'
        if key == 'lift' and str(value)[1:-1] == 'leg lift':
            key = 'leg_lift'
        if key == 'leg' and str(value)[1:-1] == 'leg up':
            key = 'leg_up'
        if key == 'difference' and str(value)[1:-1] == 'age difference':
            key = 'age_difference'
        if key == 'juice' and str(value)[1:-1] == 'pussy juice':
            key = 'pussy_juice'
        if key == 'boy' and (str(value)[1:-1] == 'man on top' or str(value)[1:-1] == 'boy on top'):
            key = 'man_on_top'
        if key == 'leg hair' and str(value)[1:-1] == 'leg hair':
            key = 'leg_hair'
        if key == 'chest hair' and str(value)[1:-1] == 'chest hair':
            key = 'chest_hair'
        if key == 'bone' and str(value)[1:-1] == 'prone bone':
            key = 'prone_bone'
        if key == 'head' and str(value)[1:-1] == 'head out of frame':
            key = 'head_out_of_frame'
        if key == 'focus' and str(value)[1:-1] == 'male focus':
            key = 'male_focus'
        if key == 'focus' and str(value)[1:-1] == 'female focus':
            key = 'female_focus'
        if key == 'cum' and str(value)[1:-1] == 'cum in mouth':
            key = 'cum_in_mouth'
        if key == 'penis' and str(value)[1:-1] == 'licking penis':
            key = 'licking_penis'
        if key == 'grab' and str(value)[1:-1] == 'penis grab':
            key = 'penis_grab'
        if key == 'grab' and str(value)[1:-1] == 'breast grab':
            key = 'breast_grab'
        if key == 'bent' and str(value)[1:-1] == 'bent over':
            key = 'bent_over'
        if key == 'closed' and str(value)[1:-1] == 'closed shading eyes':
            key = 'eyes'
        if key == 'removed' and str(value)[1:-1] == 'eyewear removed':
            key = 'eyewear_removed'
        if key == 'drinking' and str(value)[1:-1] == 'drinking wine glass':
            key = 'drinking_wine_glass'
        if key == 'body' and str(value)[1:-1] == 'lower body':
            key = 'lower_body'
            

        # Remove the first and last characters (the square brackets) from each value
        fully_processed_phrase_dict_modified[key] = str(value)[1:-1]

    if 'denim' in fully_processed_phrase_dict_modified and 'pants' in fully_processed_phrase_dict_modified and 'jeans' in fully_processed_phrase_dict_modified:
        fully_processed_phrase_dict_modified.pop('denim')
        fully_processed_phrase_dict_modified.pop('pants')
        fully_processed_phrase_dict_modified.pop('jeans')
        fully_processed_phrase_dict_modified['pants'] = 'denim pants jeans'
        
    if 'lying' in fully_processed_phrase_dict_modified and 'stomach' in fully_processed_phrase_dict_modified:
        fully_processed_phrase_dict_modified.pop('lying')
        fully_processed_phrase_dict_modified.pop('stomach')
        fully_processed_phrase_dict_modified['lying'] = 'lying on stomach'

    if 'lying' in fully_processed_phrase_dict_modified and 'bed' in fully_processed_phrase_dict_modified:
        fully_processed_phrase_dict_modified.pop('lying')
        fully_processed_phrase_dict_modified.pop('bed')
        fully_processed_phrase_dict_modified['bed'] = 'lying on a bed'

    if 'lying' in fully_processed_phrase_dict_modified and 'stomach' in fully_processed_phrase_dict_modified and 'bed' in fully_processed_phrase_dict_modified['stomach']:
        fully_processed_phrase_dict_modified.pop('lying')
        fully_processed_phrase_dict_modified.pop('stomach')
        fully_processed_phrase_dict_modified['lying'] = 'lying on stomach on a bed'

    if 'piercing' in fully_processed_phrase_dict_modified and fully_processed_phrase_dict_modified['piercing'] == 'piercing clitoris':
        fully_processed_phrase_dict_modified['piercing'] = 'clitoris piercing'
        
    if 'on_bed' in fully_processed_phrase_dict_modified and fully_processed_phrase_dict_modified['on_bed'] == 'bed on':
        fully_processed_phrase_dict_modified['on_bed'] = 'on bed'
        
    if 'close_up' in fully_processed_phrase_dict_modified:
        fully_processed_phrase_dict_modified['close_up'] = 'close-up'
    if 'face_down_ass_up' in fully_processed_phrase_dict_modified:
        fully_processed_phrase_dict_modified['face_down_ass_up'] = 'face down ass up'
    if 'man_on_top' in fully_processed_phrase_dict_modified:
        fully_processed_phrase_dict_modified['man_on_top'] = 'man on top'
    if 'fours' in fully_processed_phrase_dict_modified and fully_processed_phrase_dict_modified['fours'] == 'fours':
        fully_processed_phrase_dict_modified['fours'] = 'on all fours'
    if 'side' in fully_processed_phrase_dict_modified and fully_processed_phrase_dict_modified['side'] == 'side from':
        fully_processed_phrase_dict_modified['side'] = 'from the side'
    if 'closed' in fully_processed_phrase_dict_modified and fully_processed_phrase_dict_modified['closed'] == 'one eye closed':
        fully_processed_phrase_dict_modified['eye'] = 'one eye closed'
        fully_processed_phrase_dict_modified.pop('closed')
    if 'crossed' in fully_processed_phrase_dict_modified and fully_processed_phrase_dict_modified['crossed'] == 'legs crossed':
        fully_processed_phrase_dict_modified['legs'] = 'legs crossed'
        fully_processed_phrase_dict_modified.pop('crossed')
    if 'grabbing' in fully_processed_phrase_dict_modified and 'breast_grab' in fully_processed_phrase_dict_modified:
        fully_processed_phrase_dict_modified.pop('grabbing')
    if 'hand' in fully_processed_phrase_dict_modified and fully_processed_phrase_dict_modified['hand'] == 'hand on another\'s head':
        fully_processed_phrase_dict_modified['hand_on_another'] = 'his hand on her head'
    if 'after' in fully_processed_phrase_dict_modified and fully_processed_phrase_dict_modified['after'] == 'fellatio after':
        fully_processed_phrase_dict_modified['after'] = 'after fellatio'
    if 'straps' in fully_processed_phrase_dict_modified and fully_processed_phrase_dict_modified['straps'] == 'garter straps':
        fully_processed_phrase_dict_modified['garter_straps'] = 'garter straps'
    if 'female' in fully_processed_phrase_dict_modified and fully_processed_phrase_dict_modified['female'] == 'skinned dark female':
        fully_processed_phrase_dict_modified['female'] = 'dark skinned female'
    if 'behind' in fully_processed_phrase_dict_modified and fully_processed_phrase_dict_modified['behind'] == 'back behind arms':
        fully_processed_phrase_dict_modified['behind'] = 'arms behind back'
    if 'squeezed' in fully_processed_phrase_dict_modified and 'breasts' in fully_processed_phrase_dict_modified['squeezed']:
        fully_processed_phrase_dict_modified['squeezed'] = 'breasts squeezed together'


    # this might be unnecessary
    # if 'age' in fully_processed_phrase_dict_modified:
    #     fully_processed_phrase_dict_modified['age'] = round(age)
    #     fully_processed_phrase_dict_modified['face_direction'] = face_gesture
    # else:
    #     fully_processed_phrase_dict_modified['age'] = 25
    #     fully_processed_phrase_dict_modified['face_direction'] = False
    # fully_processed_phrase_dict_modified['age'] = 25
    # fully_processed_phrase_dict_modified['face_direction'] = False

    # need input list and male,female by this time. we should pass it in.
    transformed_input_list = transform_input(fully_processed_phrase_dict_modified, flatten(master_dictionary))
    intermediary_keys = find_intermediary_keys(transformed_input_list, master_dictionary)
    flattened_intermediary_keys = [item for sublist in intermediary_keys for item in sublist]
    intermediary_dict = {key: key for key in flattened_intermediary_keys}
    combined_dict = {**intermediary_dict, **transformed_input_list}
    
    combined_phrases2 = [combine_root_and_phrases(nlp, ie, key, phrases).strip() for key, phrases in fully_processed_phrase_dict_modified.items()]

    return ', '.join(fully_processed_phrase_dict_modified.values())

    # # templates
    # template_env = Environment()
    # template_env.filters['filter_empty_strings'] = filter_empty_strings
    # template_env.filters['reject_items'] = reject_items
    # template_env.filters['is_subset_of'] = is_subset_of
    # template_env.filters['only_elements'] = only_elements
    # template_env.filters['random_selection'] = random_selection
    # template_env.filters['all'] = all_filter
    # template_env.filters['process_face_and_looking_phrases'] = process_face_and_looking_phrases


    # face_phrase = template_env.from_string(female_face_template).render(combined_dict)
    # female_face_formatted_phrase= re.sub(' +', ' ', str(face_phrase).replace('\n', ' '))

    # body_phrase =  template_env.from_string(female_body_template).render(combined_dict)
    # female_body_formatted_phrase= re.sub(' +', ' ', str(body_phrase).replace('\n', ' '))

    # female_clothes_phrase =  template_env.from_string(female_clothes_template).render(combined_dict)
    # female_clothes_formatted_phrase= re.sub(' +', ' ', str(female_clothes_phrase).replace('\n', ' '))

    # male_phrase = template_env.from_string(male_body_template).render(combined_dict)
    # male_formatted_phrase= re.sub(' +', ' ', str(male_phrase).replace('\n', ' '))

    # male_clothes_phrase =  template_env.from_string(male_clothes_template).render(combined_dict)
    # male_clothes_formatted_phrase= re.sub(' +', ' ', str(male_clothes_phrase).replace('\n', ' '))

    # sex_phrase = template_env.from_string(sex_template).render(combined_dict)
    # sex_formatted_phrase= re.sub(' +', ' ', str(sex_phrase).replace('\n', ' '))

    # background_phrase =  template_env.from_string(background_template).render(combined_dict)
    # background_formatted_phrase= re.sub(' +', ' ', str(background_phrase).replace('\n', ' '))

    # training_prompts = [
    #     female_face_formatted_phrase.strip(),
    #     female_body_formatted_phrase.strip(),
    #     female_clothes_formatted_phrase.strip(),
    #     male_formatted_phrase.strip(),
    #     male_clothes_formatted_phrase.strip(),
    #     sex_formatted_phrase.strip(),
    #     background_formatted_phrase.strip()
    # ]
    # training_prompts = [prompt for prompt in training_prompts if prompt]

    # # Results
    # final_prompt = ', '.join(training_prompts)
    # remaining_filtered_phrases = filter_words(
    #     fully_processed_phrase_dict_modified, 
    #     female_face_formatted_phrase+
    #     female_body_formatted_phrase+
    #     female_clothes_formatted_phrase+
    #     male_formatted_phrase+
    #     male_clothes_formatted_phrase+
    #     sex_formatted_phrase+
    #     background_formatted_phrase
    #     )

    # print('Excess strings:', remaining_filtered_phrases.values())
    # phrase_blacklist = [
    #     '2boys', 'multiple boys', 'yaoi', 'oto no ko',
    #     'cigarette',
    #     'uncensored', 'wading',
    #     'censored',
    #     'censoring', 'old man', 'fat man', 'man'
    #     'multiple girls', 'close -up','side bed on','bed side on',
    #     'dog', 'fine art parody',
    #     '2girl', 'upright straddle', 'top bottom top-down -up', 'watson cross',
    #     'looking at viewer', 'real world location', 
    #     'licking', 'yuri', 'cunnilingus', 'handjob', 'holding', 'spaghetti', 'pasta', 'wariza',
    #     'anal', 'pasties', 'reverse upright straddle', 'bed on', 'sex toy', 'bar censor',
    #     'back looking at viewer and at another', 'artist name', 'clothed nude female male', 'nude female clothed male', 
    #     'back looking at another and at viewer', 'back looking at another', 'photo background', 'hand on another\'s head',
    #     'science fiction', 'ground motor vehicle', 'motor ground vehicle', 'car interior', 'cockpit', 'fighter jet', 'vehicle focus', 'aircraft', 'airplane', 'car', 'web address', 'watermark', 'weapon', 'bicycle', 'bara',
    #     'fruit', 'cake', 'plate', 'bowl', 'cup', 'lemon', 'fire', 'place bar', 'fat old man', 'old fat man', 'overwatch mercy', 'multiple penise', 'male ass focus', 'back head', 'loli', 'paizuri',
    # ]
    # remaining_filtered_phrases2 = filter_words(remaining_filtered_phrases, ' '.join(phrase_blacklist))
    # remaining_filtered_phrases3 = {key: value for key, value in remaining_filtered_phrases2.items() if value}

    # return final_prompt


