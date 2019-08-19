"""
Module to convert xlnet tokens to entities for chemical evaluation

Expects xlnet to have produced file with tokenized output per line
"""

import argparse


def get_args():
    """
    Return args for processing tokenized file
    """

    parser = argparse.ArgumentParser(description="Arguments to process file")
    parser.add_argument("--input",
                        dest="input_path",
                        help="Path to output file from xlnet")
    parser.add_argument("--output",
                        dest="output_path",
                        default="xlnet_output",
                        help="Path to output file from xlnet")
    return parser


def read_file(path):
    """ 
    Read tokenized xl output file
    """
 
    print(path)
    with open(path, "r", encoding="utf8") as f:
        print(f)
        predictions = f.readlines()
    return predictions


def process_predictions(predictions, output_path):
    """
    Function to process BERT tokens, IOB predictions
    and create the full entities

    This function assumes the subtokens have not been joined prior to the creation of the file
    """

    # Only use this bit if I haven't reconsituted the sentence piece tokens into full tokens
    entity_pmids = []
    entity_labels = []
    whole_tokens = []
    sub_token = False
    entity_label = ""
    entity_pmid = ""
    prev_label = ""
    token_main = ""
    token_cnt = 0
    # Very annoying loops to reconstitute bert tokens
    for pred in predictions:
        line = pred.split("\t")
        label = line[2].strip()
        pmid = line[0]
        token = line[1]
        if label == "X":
            sub_token = True
            token_sub = token
            token_main += token_sub
        else:
            # Some tokens will have no sub tokens, some will, so I have to keep track
            # of both cases.
            if sub_token == True or (sub_token == False and token_cnt > 0):
                whole_tokens.append(token_main)
                entity_pmids.append(entity_pmid)
                entity_labels.append(entity_label)
            entity_label = label
            entity_pmid = pmid
            token_main = token
            sub_token = False
        token_cnt += 1

    combined_labels = []
    combined_pmids = []
    combined_tokens = []
    i_token_state = False
    b_token_state = False
    o_label_state = False
    b_token = ""
    prev_label = ""
    token_label = ""
    entity_pmid = ""
    i_cnt = 0
    b_cnt = 0
    cnt = 0
    for pmid, token, label in zip(entity_pmids, whole_tokens, entity_labels):
        if label == "O":
            prev_label = "O"
            o_label_state = True
            continue
        elif label.startswith("B"):
            # Account for entities that have B- and I- labels and those that have just B-
            # Check if the loop previously visited the I condition.
            if i_token_state == True or (b_token_state == True and i_token_state == False):
                if b_token != "":
                    combined_labels.append(token_label)
                    combined_pmids.append(entity_pmid)
                    combined_tokens.append(b_token)
            i_token_state = False
            b_token_state = True
            o_label_state = False
            entity_pmid = pmid
            b_token = token
            token_label = label
            b_cnt += 1
        elif label.startswith("I"):
            # Append an inner entity to the previous entity
            i_cnt += 1
            i_token_state = True
            b_token_state = False
            b_token += " " + token
        prev_label = label
        cnt += 1        

    print("Inner and Beginning entity count")
    print(i_cnt, b_cnt)
    with open(output_path,'w') as writer:
        for pmid, token, label in zip(combined_pmids, combined_tokens, combined_labels):
            writer.write("{0}\t{1}\t{2}\n".format(pmid, token, label))


def process_predictions_preprocessed(predictions, output_path):
    """
    Function to process BERT tokens, IOB predictions
    and create the full entities

    This function assumes the subtokens have been joined in the run_ner.py xlnet module
    However, that doesn't join all tokens correctly and while it could be corrected, it 
    is preferred to do it post-that-module.
    """

    ## Here begins the onerous task of parsing the output
    combined_labels = []
    combined_pmids = []
    combined_tokens = []
    i_token_state = False
    b_token_state = False
    o_label_state = False
    b_token = ""
    prev_label = ""
    token_label = ""
    entity_pmid = ""
    i_cnt = 0
    b_cnt = 0
    cnt = 0
    #for pmid, token, label in zip(entity_pmids, whole_tokens, entity_labels):
    for pred in predictions:
        line = pred.split("\t")
        # Handle the first line.
        label = line[2].strip()
        pmid = line[0]
        token = line[1]
        if label == "O":
            prev_label = "O"
            o_label_state = True
            continue
        elif label.startswith("B"):
            # Account for entities that have B- and I- labels and those that have just B-
            # Check if the loop previously visited the I condition.
            if i_token_state == True or (b_token_state == True and i_token_state == False):
                #if "-" in b_token:
                #    # Account for word piece adding space
                #    b_token = "-".join([t.strip() for t in b_token.split("-")])
                #if "/" in b_token:
                #    b_token = "/".join([t.strip() for t in b_token.split("/")])
                #if "(" in b_token:
                #    b_token = "(".join([t.strip() for t in b_token.split("(")])
                #if ")" in b_token:
                #    b_token = ")".join([t.strip() for t in b_token.split(")")])
                combined_labels.append(token_label)
                combined_pmids.append(entity_pmid)
                combined_tokens.append(b_token)
            i_token_state = False
            b_token_state = True
            o_label_state = False
            entity_pmid = pmid
            b_token = token
            token_label = label
            b_cnt += 1
        # Check to see if there are any I- mispredicted. 
        # It is optional to add these to the predictions
        elif label.startswith("I") and o_label_state == True:
            print("No B- before I-")
            print(pmid, token)
            #if "-" in token:
            #    # Account for word piece adding space
            #    token = "-".join([t.strip() for t in token.split("-")])
            #combined_labels.append("B-chem")
            #combined_pmids.append(pmid)
            #combined_tokens.append(token)
        elif label.startswith("I"):
            # Append an inner entity to the previous entity
            i_cnt += 1
            i_token_state = True
            b_token_state = False
            b_token += " " + token
        else:
            print("Unexpected behavior")
            print(pmid, token, label, b_token)
        prev_label = label
        cnt += 1        

    print(i_cnt, b_cnt)
    with open(output_path,'w') as writer:
        for pmid, token, label in zip(combined_pmids, combined_tokens, combined_labels):
            writer.write("{0}\t{1}\t{2}\n".format(pmid, token, label))


def main():
    """
    Read, parse and join xlnet predictions
    """

    predictions = read_file(args.input_path)
    process_predictions(predictions, args.output_path)

if __name__ == "__main__":
    global args
    args = get_args().parse_args()
    main()
