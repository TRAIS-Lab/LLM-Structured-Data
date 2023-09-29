import sys
sys.path.append("./")
from utils.load_products import products_keys_list


def generate_arxiv_prompts(include_options, arxiv_natural_lang_mapping):
    if include_options:
        arxiv_prompts = {
            'subcategory': f"Please predict the most appropriate arXiv Computer Science (CS) sub-category for the paper. Your answer should be chosen from {', '.join(arxiv_natural_lang_mapping.keys())}. The predicted sub-category should be in the format 'cs.XX'.",
            'identifier': f"Please predict the most appropriate original arxiv identifier for the paper. Your answer should be chosen from {', '.join([key.lower() for key in arxiv_natural_lang_mapping.keys()])}. The predicted arxiv identifier should be in the format 'arxiv cs.xx'.",
            'natural language': "Please predict the most appropriate category for the paper. Your answer should be chosen from {}.".format(', '.join(['"{}"'.format(arxiv_natural_lang_mapping[key]) for key in arxiv_natural_lang_mapping.keys()]))
        }
    else:
        arxiv_prompts = {
            'subcategory': "Please predict the most appropriate arXiv Computer Science (CS) sub-category for the paper. The predicted sub-category should be in the format 'cs.XX'.",
            'identifier': "Please predict the most appropriate original arxiv identifier for the paper. The predicted arxiv identifier should be in the format 'arxiv cs.xx'.",
            'natural language': "Please predict the most appropriate category for the paper. Your answer should be chosen from {}.".format(', '.join(['"{}"'.format(arxiv_natural_lang_mapping[key]) for key in arxiv_natural_lang_mapping.keys()]))
        }
    return arxiv_prompts


def generate_system_prompt(source, arxiv_style="subcategory", include_options=False):
    """
    Generate a system prompt based on the given content type and source.
    
    Args:
    - content_type (str): Specifies the type of content (e.g., title, abstract, neighbors).
    - source (str): Specifies the data source (e.g., arxiv, cora, pubmed, product).
    - use_original_arxiv (bool, optional): If set to True, a special prompt for 'arxiv' is used.
    
    Returns:
    - str: Generated system prompt.
    """

    categories = {
        'cora': ["Rule Learning", "Neural Networks", "Case Based", "Genetic Algorithms", "Theory", "Reinforcement Learning", "Probabilistic Methods"],
        'pubmed': ["Type 1 diabetes", "Type 2 diabetes", "Experimentally induced diabetes"]
    }

    arxiv_prompts = generate_arxiv_prompts(include_options, arxiv_natural_lang_mapping)
    
    prompts = {
        'arxiv': arxiv_prompts[arxiv_style],
        'cora': "Please predict the most appropriate category for the paper. Choose from the following categories:\n\n{}",
        'pubmed': "Please predict the most likely type of the paper. Your answer should be chosen from:\n\n{}",
        'product': "Please predict the most likely category of this product from Amazon. Your answer should be chosen from the list:\n\n{}"
    }

    # Fetch the appropriate prompt
    prompt = prompts[source]

    if source in ['cora', 'pubmed']:
        categories_list = "\n".join(categories[source])
        return prompt.format(categories_list)
    elif source == 'product':
        return format(prompt.format("\n".join(products_keys_list)))
    else:
        return format(prompt)

arxiv_natural_lang_mapping = {
    'cs.AI': 'Artificial Intelligence',
    'cs.CL': 'Computation and Language',
    'cs.CC': 'Computational Complexity',
    'cs.CE': 'Computational Engineering, Finance, and Science',
    'cs.CG': 'Computational Geometry',
    'cs.GT': 'Computer Science and Game Theory',
    'cs.CV': 'Computer Vision and Pattern Recognition',
    'cs.CY': 'Computers and Society',
    'cs.CR': 'Cryptography and Security',
    'cs.DS': 'Data Structures and Algorithms',
    'cs.DB': 'Databases',
    'cs.DL': 'Digital Libraries',
    'cs.DM': 'Discrete Mathematics',
    'cs.DC': 'Distributed, Parallel, and Cluster Computing',
    'cs.ET': 'Emerging Technologies',
    'cs.FL': 'Formal Languages and Automata Theory',
    'cs.GL': 'General Literature',
    'cs.GR': 'Graphics',
    'cs.AR': 'Hardware Architecture',
    'cs.HC': 'Human-Computer Interaction',
    'cs.IR': 'Information Retrieval',
    'cs.IT': 'Information Theory',
    'cs.LO': 'Logic in Computer Science',
    'cs.LG': 'Machine Learning',
    'cs.MS': 'Mathematical Software',
    'cs.MA': 'Multiagent Systems',
    'cs.MM': 'Multimedia',
    'cs.NI': 'Networking and Internet Architecture',
    'cs.NE': 'Neural and Evolutionary Computing',
    'cs.NA': 'Numerical Analysis',
    'cs.OS': 'Operating Systems',
    'cs.OH': 'Other Computer Science',
    'cs.PF': 'Performance',
    'cs.PL': 'Programming Languages',
    'cs.RO': 'Robotics',
    'cs.SI': 'Social and Information Networks',
    'cs.SE': 'Software Engineering',
    'cs.SD': 'Sound',
    'cs.SC': 'Symbolic Computation',
    'cs.SY': 'Systems and Control'
}


if __name__ == "__main__":
    # Usage examples
    print(generate_system_prompt("arxiv"))
    print(generate_system_prompt("cora"))
    print(generate_system_prompt("pubmed"))
    print(generate_system_prompt("product"))