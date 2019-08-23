"""
Convert chemdner training and development data into 
BIO format for bert 

This script assumes the BioCreative IV CHEMDNER development and training data have been placed in the same directory as this script.
"""

import sys
import logging
from chemtok import ChemTokeniser


def annotate_with_IOB(token_dict, annotations):
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
        # but keep track of which chemicals not considered to be one word
        for i in token_dict[pmid]:
            # Give a other label if the token isn't in the annotations
            iob_label = "O"
            # check to see if tokens correspond to a multiword chem
            # If there is a multiword chem, the terms after the first
            # term use the Inner label
            if i > beg_offset and i < end_offset and end_offset != 0:
                iob_label = "I-{}".format(label)
            # Otherwise, check the mml content for the spacy token.
            # If the token is in there, get the term, label, 
            # and keep track of the offset.
            elif i in annotations[pmid]:
                iob_label = "B-{}".format(label)
                beg_offset = annotations[pmid][i][0]
                end_offset = annotations[pmid][i][1]
                                    
            term = token_dict[pmid][i][2]
            annotation_string = "{0}\t{1}\t{2}\n".format(pmid, term, iob_label)
            annotation_strings.append(annotation_string)
        pmid_cnt += 1

    with open("chemdner_CDI_training.txt", "a+", encoding="utf8") as annotations_file:
        [annotations_file.write(s) for s in annotation_strings]


def process_ann(annotations):
    """
    Parse the annotations, pmids, offsets and term
    """

    anns = {}
    annotation_dict = {}
    for i, line in enumerate(annotations):
        line = line.split("\t")
        if line[1] == "T":
            continue
        start = int(line[2])
        end = int(line[3])
        term = line[4]
        if i == 0:
            pmid = line[0] 
        elif pmid != line[0]:
            annotation_dict[pmid] = anns
            pmid = line[0]
            anns = {}
        # There will be nested annotated entities. Here I only include the longest 
        # entity
        anns[start] = [start, end, term]

    annotation_dict[pmid] = anns
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
    Parse the abstracts/titles and return the tokens for file
    """

    citation_dict = {}
    for line in text:
        citation = line.split("\t")
        # Don't deal with titles as these will separate offsets from the abstracts
        citation_ta = citation[2].strip()
        citation_dict[citation[0]] = citation_ta
    return citation_dict


def load_data():
    """Open chemdner"""

    with open("training.abstracts.txt", "r", encoding="utf8") as f:
        training_data = f.readlines()

    with open("training.annotations.txt", "r", encoding="utf8") as f:
        training_ann = f.readlines()

    with open("development.abstracts.txt", "r", encoding="utf8") as f:
        dev_data = f.readlines()

    with open("development.annotations.txt", "r", encoding="utf8") as f:
        dev_ann = f.readlines()

    return training_data, training_ann, dev_data, dev_ann


def main():
    """
    Convert data to bio.
    Tokenzie text
    Align annotations to tokens, and label as B-chem, I-Inch
    Label everything else as O
    """

    write_file = open("chemdner_CDI_training.txt", "w")
    write_file.close()
    train_data, train_ann, dev_data, dev_ann = load_data()
    text_data = train_data + dev_data
    annotations = train_ann + dev_ann
    citation_dict = process_text(text_data)
    citation_tokens = tokenize_text(citation_dict)
    annotation_dict = process_ann(annotations)
    annotate_with_IOB(citation_tokens, annotation_dict)

    
if __name__ == "__main__":
    main()
