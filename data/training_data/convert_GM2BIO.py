"""
Convert chemdner training and development data into 
BIO format for bert 

This script assumes the training data and annotations have been placed in ./bc2geneMention/train/train.in and ./bc2geneMention/train/GENE.eval directories. 
"""

import sys
import logging
from chemtok import ChemTokeniser


def annotate_with_IOB(token_dict, citation_dict, annotations):
    """
    Function to actually create the BIO annotated file, of training
    and devlopment CHEMDNER data

    Using Inner outer beginning annotations, 
    I am only using one chem class, as I don't want to slow 
    down the annotations by calling UMLS
    """

    pmid_cnt = 0
    annotation_strings = []
    for pmid in token_dict:
        if pmid not in annotations:
            continue
        end_offset = 0
        beg_offset = 0
        label = "chem"
        # Iterate through the tokens
        # But keep track of which chemicals not considered to be one word
        for i in token_dict[pmid]:
            # Give a other label if the token isn't in the annotations
            iob_label = "O"
            # Get the index of the token in the actual citation
            # Use this as the offset
            citation_index = citation_dict[pmid].index(token_dict[pmid][i][2])
            term = token_dict[pmid][i][2]
            # Alter found terms from citation dict to avoid multiple matches
            # Convert citation to list and then replace term with placeholder so there won't be 
            # any repeat matches with later tokens
            placeholder = ["$" for c in term]
            citation_dict[pmid] = [char for char in citation_dict[pmid]]
            citation_dict[pmid][citation_index: citation_index + len(token_dict[pmid][i][2])] = placeholder
            # check to see if tokens correspond to a multiword chem
            # If there is a multiword chem, the terms after the first
            # term use the Inner label
            if citation_index > beg_offset and citation_index < end_offset and end_offset != 0:
                iob_label = "I-{}".format(label)
            # Otherwise, check the annotations for the token
            # If the token is in there, get the term, label, 
            # and keep track of the offset.
            elif citation_index in annotations[pmid]:
                iob_label = "B-{}".format(label)
                beg_offset = annotations[pmid][citation_index][0]
                end_offset = annotations[pmid][citation_index][1]
                                    
            # Convert citation back to string
            citation_dict[pmid] = "".join(citation_dict[pmid])

            # Save annotation
            annotation_string = "{0}\t{1}\t{2}\n".format(pmid, term, iob_label)
            annotation_strings.append(annotation_string)
        pmid_cnt += 1

        # write each term to file, per line
    with open("BioC_GM_training.txt", "a+", encoding="utf8") as annotations_file:
        [annotations_file.write(s) for s in annotation_strings]


def process_ann(annotations):
    """
    Parse the annotations, pmids, offsets and term from the BioC file

    Return dictionary with id: dictionary of annotations with beginning offset    
    """

    anns = {}
    annotation_dict = {}
    for i, line in enumerate(annotations):
        line = line.split("|")
        start = int(line[1].split(" ")[0])
        end = int(line[1].split(" ")[1])
        term = line[2].strip()
        if i == 0: 
            sent_id = line[0] 
        elif sent_id != line[0]:
            annotation_dict[sent_id] = anns
            sent_id = line[0]
            anns = {}
        # There will be nested annotated entities. Here I only include the longest 
        # entity
        anns[start] = [start, end, term]

    annotation_dict[sent_id] = anns
    return annotation_dict
            
        
def tokenize_text(citation_dict):
    """
    Return dict of pmid: tokens
    """

    # Iterate through the citations and put their tokens into a dictionary:
    # pmid: tokens
    citation_tokens = {}
    for pmid in citation_dict:
        tokens = ChemTokeniser(citation_dict[pmid], clm=False)
        token_dict = {token.start: [token.start, token.end, token.value] for token in tokens.tokens}
        citation_tokens[pmid] = token_dict
    return citation_tokens


def process_text(text):
    """
    Parse the abstracts/titles from the BioC file
    
    Returns dictionary, where each value is a citation, spaced and unspaced
    """

    citation_dict = {}
    citation_dict_spaced = {}
    for line in text:
        # The offsets correspond to the text with no spaces
        # So need two dictionaries, one for Chemtok, with spaces, and another, with no spaces
        # for reference back to the chemtok tokens. 
        citation = "".join(line.split(" ")[1:]).strip()
        citation_spaced = " ".join(line.split(" ")[1:]).strip()
        # Don't deal with titles as these will separate offsets from the abstracts
        citation_dict[line.split(" ")[0]] = citation
        citation_dict_spaced[line.split(" ")[0]] = citation_spaced
    return citation_dict, citation_dict_spaced


def load_data():
    """Open BioCreative GeneMention data"""

    with open("bc2geneMention/train/train.in", "r", encoding="utf8") as f:
        training_data = f.readlines()

    with open("bc2geneMention/train/GENE.eval", "r", encoding="utf8") as f:
        training_ann = f.readlines()

    return training_data, training_ann


def main():
    """
    Convert data to bio.
    Tokenzie text
    Align annotations to tokens, and label as B-chem, I-Inch
    Label everything else as O
    """

    write_file = open("BioC_GM_training.txt", "w")
    write_file.close()
    # Training text and annotations are in separate files
    train_data, train_ann = load_data()
    citation_dict, citation_dict_spaced = process_text(train_data)
    # Use ChemTokeniser from ChemListem to tokenize spaced citation text
    citation_tokens = tokenize_text(citation_dict_spaced)
    annotation_dict = process_ann(train_ann)
    annotate_with_IOB(citation_tokens, citation_dict, annotation_dict)


    
if __name__ == "__main__":
    main()
