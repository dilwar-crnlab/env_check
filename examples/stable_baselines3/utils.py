import numpy as np

def path_vector_to_tokens(path_vector):
    L = path_vector.shape[0]
    # Find indices where the candidate path uses a link (value == 1)
    token_indices = np.where(path_vector == 1)[0]
    # For each index, create a one-hot vector
    tokens = []
    for idx in token_indices:
        token = np.zeros(L, dtype=path_vector.dtype)
        token[idx] = 1.
        tokens.append(token)
    return tokens

def convert_all_paths_to_tokens(path_matrix):
    num_paths = path_matrix.shape[1]
    all_tokens = []
    for i in range(num_paths):
        candidate_path = path_matrix[:, i]
        tokens = path_vector_to_tokens(candidate_path)
        all_tokens.append(tokens)
    return all_tokens