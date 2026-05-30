import penman
import re
# from collections import defaultdict
# import torch
# import pickle
import pandas as pd
from transformers import AutoTokenizer
import torch
# from torch_geometric.data import HeteroData, Batch



def get_indexed(amr_string,sentence):
    """
    Returns a dictionary with string indices where root node is at '0'
    Format: {'0': (node, concept/literal, [start, end]), '1': (...), ...}
    """
    # Parse the AMR
    graph = penman.decode(amr_string)
    
    # Get all nodes from triples (excluding reference triples)
    all_nodes = set()
    for triple in graph.triples:
        source, role, target = triple
        if role != ':instance':
            if source.startswith('"') and source.endswith('"'):
                source = source[1:-1]
            if target.startswith('"') and target.endswith('"'):
                target = target[1:-1]
            all_nodes.add(source)
            all_nodes.add(target)
    if not all_nodes:
        return None, None, None, None,None,None
    # Extract aligned concepts and literals from original AMR string
    # Pattern matches: concept~number or "literal"~number
    pattern = re.compile(r'(?:([^/\s\)]+)~(\d+)|"([^"]+)"~(\d+))')
    matches = pattern.findall(amr_string)
    # print(all_nodes)
    # Create concept/literal to alignment mapping
    # Collect variables present in the graph
    variables_in_graph = {v for (v, _, _) in graph.instances()}      
    alignment_map = {}

    # 1) Instance nodes: capture variable, concept, and alignment index
    # Example: "(n4 / name~20", "(g / game~3"
    # Accept quoted or unquoted concept after '/'
    pattern_instance = re.compile(
        r'\(\s*([^\s()/"]+)\s*/\s*(?:"([^"]+)"|([^\s/")]+))~(\d+)', flags=re.DOTALL
    )
    for var, quoted, unquoted, idx in pattern_instance.findall(amr_string):
        concept = quoted if quoted else unquoted
        idx_int = int(idx)
        alignment_map[var] = (var, concept, [idx_int, idx_int])

    # 2) Quoted literals: capture only those NOT used as instance concepts
    #    We skip matches where the quote is preceded (ignoring whitespace) by '/'
    #    Example kept: :op1 "Oxford"~10
    #    Example skipped: (p2 / "prefecture"~33)
    pattern_quoted = re.compile(r'"([^"]+)"~(\d+)', flags=re.DOTALL)
    for m in pattern_quoted.finditer(amr_string):
        lit = m.group(1)
        idx_int = int(m.group(2))
        # Look back from the opening quote to the previous non-space character
        q_start = m.start()  # index at the opening quote
        j = q_start - 1
        while j >= 0 and amr_string[j].isspace():
            j -= 1
        # If the previous non-space char is '/', this quoted literal is an instance concept → skip
        if j >= 0 and amr_string[j] == '/':
            continue
        # Otherwise, treat as a literal aligned token
        alignment_map[lit] = (lit, lit, [idx_int, idx_int])

    # 3) Unquoted role-literals: token right after a role, followed by ~idx
    # Examples: ":year 1874~18", ":day 14~16", ":month 3~17", ":polarity -~5", ":mode indicative~7", or words like ":quant both~0"
    # We only capture tokens that are NOT quoted and NOT part of a "/ concept~idx" instance.
    pattern_role_literal = re.compile(r':[^\s()"]+\s+([^\s()"]+)~(\d+)', flags=re.DOTALL)
    for token, idx in pattern_role_literal.findall(amr_string):
        idx_int = int(idx)
        # Normalize token (strip quotes if present, though the regex excludes quotes)
        key = token.strip('"')
        # Do not overwrite an existing key (e.g., if already captured as quoted literal)
        if key not in alignment_map:
            alignment_map[key] = (key, key, [idx_int, idx_int])


    # # Post-process alignment_map to drop duplicate literal entries
    # # when a variable with the same alignment index has that literal as its concept.
    # def _dedup_alignment_map(alignment_map, variables_in_graph):
    #     # Collect, per index, the concepts attached to variables
    #     idx_to_var_concepts = defaultdict(set)
    #     for key, (_node_id, concept, span) in alignment_map.items():
    #         if key in variables_in_graph and span and len(span) == 2 and span[0] == span[1]:
    #             idx = span[0]
    #             idx_to_var_concepts[idx].add(concept)

    #     # Any non-variable key whose concept matches a variable concept at the same index is redundant
    #     keys_to_remove = set()
    #     for key, (_node_id, concept, span) in alignment_map.items():
    #         if key in variables_in_graph:
    #             continue
    #         if span and len(span) == 2 and span[0] == span[1]:
    #             idx = span[0]
    #             if concept in idx_to_var_concepts.get(idx, set()):
    #                 keys_to_remove.add(key)

    #     for k in keys_to_remove:
    #         alignment_map.pop(k, None)

    # # Call it right after building alignment_map
    # _dedup_alignment_map(alignment_map, variables_in_graph)

    assert len(all_nodes) == len(alignment_map.keys()), f"Length mismatch: {len(all_nodes)} != {len(alignment_map),} \n amr string : {amr_string} \n sent {sentence} \n all node : {all_nodes} \n alignment_map : {alignment_map}"
    result = []
    
    # Process instance nodes (unpack with 3 variables as you noted)
    for variable, role, concept in graph.instances():
        if variable in all_nodes:

            
            # Find alignment for this concept
            entry = alignment_map.get(variable, [])
                        # Only add if alignment is not empty
            if entry:
                _, _, alignment = entry
                result.append((variable, concept, alignment))
      
    
    # Process literal nodes
    for node in all_nodes:
        if node not in variables_in_graph:
            entry = alignment_map.get(node)
            if entry:
                # entry = (node, node, [idx, idx]) for literals
                _, concept_or_lit, alignment = entry
                result.append((node, concept_or_lit, alignment))
    # print(result)
    assert len(all_nodes) == len(result), f"Length mismatch: {len(all_nodes)} != {len(result),} \n amr string : {amr_string} \n sent {sentence} \n all node : {all_nodes} \n alignment_map : {alignment_map}"
    # Get root and create indexed dictionary
    root = graph.top
    indexed_dict = {}
    
    # Find root node tuple and separate others
    root_tuple = None
    other_tuples = []
    
    for node_tuple in result:
        node_var, concept, alignment = node_tuple
        if node_var == root:
            root_tuple = node_tuple
        else:
            other_tuples.append(node_tuple)
    
    # Indexing logic: if root exists, start from '0', otherwise start from '1'
    if root_tuple:
        # Add root at index '0'
        indexed_dict[0] = root_tuple
        # Add other nodes starting from index '1'
        for i, node_tuple in enumerate(other_tuples, start=1):
            indexed_dict[i] = node_tuple
    # else:
    #     # No root found, index all nodes starting from '1'
    #     for i, node_tuple in enumerate(result, start=1):
    #         indexed_dict[i] = node_tuple

        # Process edges
    # edges = []
    # for edge in graph.edges():
    #     role, source, target = edge
    #     edges.append((source, role, target))

    edges = []
    for triple in graph.triples:
        source, role, target = triple
        if role != ":instance":
            if source.startswith('"') and source.endswith('"'):
                source = source[1:-1]
            if target.startswith('"') and target.endswith('"'):
                target = target[1:-1]

            edges.append((str(source), role[1:], str(target)))
    
    # Create mapping from variable/literal to index
    var_to_index = {}
    for index, (node_var, concept, alignment) in indexed_dict.items():
        var_to_index[node_var] = index
    
    # Replace variable names with indices in edges and filter out edges with missing nodes
    indexed_edges = []
    for source, role, target in edges:
        # Check if both source and target exist in our indexed nodes
        if source in var_to_index and target in var_to_index:
            source_idx = var_to_index[source]
            target_idx = var_to_index[target]
            indexed_edges.append((source_idx, role, target_idx))
        # Skip edges where source or target is not in indexed_nodes
    
    return indexed_dict, indexed_edges, root, root_tuple is not None, var_to_index, result

# input_ids = tokenizer(sent, add_special_tokens=False).input_ids

def word2subwordconvesion(tokens):
    """ word idx in a sentence to subword idx in a document """

    llm_model_path = "/scratch/common_models/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, use_fast=False,)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    wordidx2subwordidx_final = []
    token_id_doc = []
    for doc in tokens:
        wordidx2subwordidx_list_doc = []
        cur_subword_idx = 0
        token_id_sent = []
        for sent in doc:
            # print(sent)
            wordidx2subwordidx_list_sent = []
            word2subword = {}
            token_id_word = []
            for idx,word  in enumerate(sent):
                
                # Tokenize the word
                # print(word)
                tokenized = tokenizer(word, add_special_tokens=False).input_ids
                # print(tokenized)

                # Store the mapping
                # word2subword[word] = tokenized
                wordidx2subwordidx_list_sent.append(list(range(cur_subword_idx,cur_subword_idx+ len(tokenized))))
                cur_subword_idx= cur_subword_idx + len(tokenized)
                token_id_word.extend(tokenized) # each token id word list is one sentene

                # 🔹 Flatten wordidx2subwordidx_list_sent
                flat_indices = [i for sublist in wordidx2subwordidx_list_sent for i in sublist]

                # 🔹 Assert consistency
                assert len(token_id_word) == len(flat_indices), (
                    f"Length mismatch in sentence:\n"
                    f"  len(token_id_word) = {len(token_id_word)}\n"
                    f"  len(flat_indices)  = {len(flat_indices)}\n"
                    f"  token_id_word      = {token_id_word}\n"
                    f"  wordidx2subwordidx = {wordidx2subwordidx_list_sent}"
                )


            token_id_sent.append(token_id_word)    #each token id sent list is one document
            wordidx2subwordidx_list_doc.append(wordidx2subwordidx_list_sent)
        token_id_doc.append(token_id_sent)
        wordidx2subwordidx_final.append(wordidx2subwordidx_list_doc)
    # print(wordidx2subwordidx_final[:3]), print(token_id_doc[:3])
    return wordidx2subwordidx_final, token_id_doc
    


def getconversion_and_emb(sent, tokenizer, word_embedding):
    """ word idx in a sentence to subword idx in a document """

    wordidx2subwordidx_list_sent = []
    emb_sent = []
    cur_subword_idx= 0
    for idx,word  in enumerate(sent):
        
        # Tokenize the word
        # print(word)
        tokenized = tokenizer(word, add_special_tokens=False).input_ids
        # print(tokenized)

        # Process on CPU with no_grad to save memory
        
        emb = word_embedding(torch.tensor(tokenized, device = 'cuda:0'))
        emb_sent.append(emb)

        # Store the mapping
        # word2subword[word] = tokenized
        wordidx2subwordidx_list_sent.append(list(range(cur_subword_idx,cur_subword_idx+ len(tokenized))))
        cur_subword_idx= cur_subword_idx + len(tokenized)
        
    final_emb_sent = torch.cat(emb_sent, dim = 0)
    # Flatten wordidx2subwordidx_list_sent for assertation 
    flat_indices = [i for sublist in wordidx2subwordidx_list_sent for i in sublist]
    # 🔹 Assert consistency
    assert len(final_emb_sent) == len(flat_indices), (
        f"Length mismatch in sentence:\n"
        f"  len(final_emb_sent) = {len(final_emb_sent)}\n"
        f"  len(flat_indices)  = {len(flat_indices)}\n"
    )

    return wordidx2subwordidx_list_sent, final_emb_sent