"""
Script to do all the evaluation
of the different tools.

First make sets of the annotated terms in annotated citations.
For each citation, there will be two sets: One with the whole entity,
another with the whole entity and the nested terms.

This module can evaluate based on exact matches, with or without stop words,
and with or without using bootstrap to compute std. dev. of performance.
"""

import os
import glob
import numpy as np
import argparse
import json
import statistics
import random

import matplotlib.pyplot as plt
from leven import levenshtein
#from MTIEvaluator import MTIEvaluator


def get_args():
    """
    Get command line arguments
    """

    parser = argparse.ArgumentParser(description="Arguments to run evaluation")
    parser.add_argument("-l",
                        dest="use_leven",
                        action="store_true",
                        help="If included, use a levenshtein distance threshold to compute precision and recall")
    parser.add_argument("-s",
                        dest="include_stop",
                        action="store_true",
                        help="If included, Do not remove the stop words from the sets of predictions")
    parser.add_argument("-e",
                        dest="error_analysis",
                        action="store_true",
                        help="If included, Write a tool's (hardcoded) false positives and false negatives to file. For mesh analysis all tools are written")
    parser.add_argument("-b",
                        dest="bootstrap",
                        action="store_true",
                        help="If included, compute standard deviation on metrics of bootstrapped annotations")
    parser.add_argument("-m",
                        dest="mesh",
                        action="store_true",
                        help="If included, evaluate tools and manual annotations on mesh")
    parser.add_argument("--mesh_bootstrap",
                        dest="mesh_bootstrap",
                        action="store_true",
                        help="If included, compute standard deviation on metrics of bootstrapped annotations for MeSH")
    return parser


class ProcessMTI():
    """
    Class to load MTI data, that has already been
    processed by MTIEvaluator class in other script
    """

    def get_annotations(self):
        with open('MTI_annotations.json', 'r') as f:
            annotations = json.load(f)
        annotations = {pmid: set(annotations[pmid]) for pmid in annotations}
        return annotations


class ProcessBERTEnsemble():

    def _process_file(self, ann_file):
        """
        Iterate through file, keeping track of pmids
        """

        annotations = {}
        file_ann = []
        for i, line in enumerate(ann_file):
            ann = line.split("\t")
            if i == 0:
                pmid1 = ann[0]
                file_ann.append(ann[1].strip())
            else:
                if pmid1 == ann[0]:
                    file_ann.append(ann[1].strip())
                elif pmid1 != ann[0]:
                    annotations[pmid1] = set(file_ann)
                    file_ann = []
                    pmid1 = ann[0]
                    file_ann.append(ann[1].strip())
        # And get that last one
        annotations[pmid1] = set(file_ann)
        return annotations

    def _combine_annotations(self, ann1, ann2, man_ann):
        """
        Take union of the sets.
        """
        # First add blank sets
        ann1 = adjust_bert_preds(man_ann, ann1)
        ann2 = adjust_bert_preds(man_ann, ann2)
        ensemble_ann = {}
        for pmid in ann1:
            ensemble_ann[pmid] = ann1[pmid].union(ann2[pmid])
        return ensemble_ann

    def get_annotations(self, man_ann, ensemble):
        """
        Load different annotation files
        """

        if ensemble == "bert":
            with open("tool_annotations/bert_512_gm_test.txt", "r", encoding="utf8") as f:
                ann_file1 = f.readlines()
                ann1 = self._process_file(ann_file1)
            with open("tool_annotations/bert_512_chemdner_predictions.txt", "r", encoding="utf8") as f:
                ann_file2 = f.readlines()
                ann2 = self._process_file(ann_file2)

        elif ensemble == "scibert":
            with open("tool_annotations/scibert_chemdner_test.txt", "r", encoding="utf8") as f:
                ann_file1 = f.readlines()
                ann1 = self._process_file(ann_file1)
            with open("tool_annotations/scibert_gene_mention_test.txt", "r", encoding="utf8") as f:
                ann_file2 = f.readlines()
                ann2 = self._process_file(ann_file2)

        elif ensemble == "biobert":
            with open("tool_annotations/biobert_chemdner_test.txt", "r", encoding="utf8") as f:
                ann_file1 = f.readlines()
                ann1 = self._process_file(ann_file1)
            with open("tool_annotations/biobert_gene_mentions_test.txt", "r", encoding="utf8") as f:
                ann_file2 = f.readlines()
                ann2 = self._process_file(ann_file2)

        elif ensemble == "xlnet":
            with open("tool_annotations/xlnet_chemdner_test.txt", "r", encoding="utf8") as f:
                ann_file1 = f.readlines()
                ann1 = self._process_file(ann_file1)
            with open("tool_annotations/xlnet_gm_test.txt", "r", encoding="utf8") as f:
                ann_file2 = f.readlines()
                ann2 = self._process_file(ann_file2)

        return self._combine_annotations(ann1, ann2, man_ann)


class ProcessBERT():

    def _process_file(self, ann_file):
        """
        Iterate through file, keeping track of pmids
        """

        annotations = {}
        file_ann = []
        for i, line in enumerate(ann_file):
            ann = line.split("\t")
            if i == 0:
                pmid1 = ann[0]
                file_ann.append(ann[1].strip())
            else:
                if pmid1 == ann[0]:
                    file_ann.append(ann[1].strip())
                elif pmid1 != ann[0]:
                    annotations[pmid1] = set(file_ann)
                    file_ann = []
                    pmid1 = ann[0]
                    file_ann.append(ann[1].strip())
        # And get that last one
        annotations[pmid1] = set(file_ann)
        return annotations

    def get_annotations(self, model):
        """
        Annotations are all in one file.
        PMIDS indicate new citation.
        """

        if model == "bert":
            with open("tool_annotations/bert_ctb_chem_predictions.txt", "r", encoding="utf8") as f:
                ann_file = f.readlines()
        elif model == "bert_bi":
            with open("tool_annotations/bert_ctb_chem_predictions_bi.txt", "r", encoding="utf8") as f:
                ann_file = f.readlines()
        elif model == "scibert":
            with open("tool_annotations/scibert_ctb_chem_predictions.txt", "r", encoding="utf8") as f:
                ann_file = f.readlines()
        elif model == "cdner_bert":
            with open("tool_annotations/bert_chemdner_test.txt", "r", encoding="utf8") as f:
                ann_file = f.readlines()
        elif model == "cde_cml_bert":
            with open("tool_annotations/bert_512_cde_cml_test.txt", "r", encoding="utf8") as f:
                ann_file = f.readlines()
        elif model == "cde_bert":
            with open("tool_annotations/bert_512_cde_test.txt", "r", encoding="utf8") as f:
                ann_file = f.readlines()
        elif model == "gm_bert":
            with open("tool_annotations/bert_gene_mention_test.txt", "r", encoding="utf8") as f:
                ann_file = f.readlines()
        elif model == "crf_bert":
            with open("tool_annotations/bert_512_chemder_crf_test.txt", "r", encoding="utf8") as f:
                ann_file = f.readlines()
        elif model == "scibert_chemd":
            with open("tool_annotations/scibert_chemdner_test.txt", "r", encoding="utf8") as f:
                ann_file = f.readlines()
        elif model == "xlnet_cdner":
            with open("tool_annotations/xlnet_chemdner_test.txt", "r", encoding="utf8") as f:
                ann_file = f.readlines()
        elif model == "xlnet_gene":
            with open("tool_annotations/xlnet_gm_test.txt", "r", encoding="utf8") as f:
                ann_file = f.readlines()

        annotations = self._process_file(ann_file)
        return annotations


class ProcessChemDataExtractor():

    def _process_file(self, ann_file):
        """
        Iterate through file, keeping track of pmids
        """
        annotations = {}
        file_ann = []
        pmid = ""
        for i, line in enumerate(ann_file):
            if line == "\n":
                new_file = True
                if i > 1:
                    annotations[pmid] = set(file_ann)
            elif new_file == True:
                file_ann = []
                pmid = line.strip()
                new_file = False
            elif new_file == False:
                file_ann.append(line.strip())
        # And get that last one
        annotations[pmid] = set(file_ann)
        return annotations

    def get_annotations(self):
        """
        Annotations are all in one file.
        PMIDS indicate new citation.
        """

        with open("tool_annotations/ChemDataExtractor_annotations.txt", "r", encoding="utf8") as f:
            ann_file = f.readlines()
            annotations = self._process_file(ann_file)
        return annotations


class ProcessPubTator():

    def _process_file(self, ann_file):
        """
        Iterate through file, keeping track of pmids
        """
        annotations = {}
        file_ann = []
        pmid1 = ""
        pmid2 = ""
        for i, line in enumerate(ann_file):
            l = line.split("|")
            if len(l[0]) == 8:
                pmid1 = l[0]
                pmid2 = l[0]
                continue
            if line == "\n":
                annotations[pmid1] = set(file_ann)
                file_ann = []
            elif pmid1 == pmid2:
                ann = line.split("\t")
                pmid2 = ann[0]
                if ann[4].strip() in ["Chemical", "Gene"]:
                    file_ann.append(ann[3].strip())
        # And get that last one
        annotations[pmid1] = set(file_ann)
        return annotations

    def get_annotations(self, version):
        """
        Annotations are all in one file.
        PMIDS indicate new citation.
        """

        if version == "PT":
            with open("tool_annotations/200_annotations_PubTator_5-13-2019.txt", "r", encoding="utf8") as f:
                ann_file = f.readlines()
                annotations = self._process_file(ann_file)
        elif version == "PTC":
            with open("tool_annotations/pubtator_central_export.pubtator", "r", encoding="utf8") as f:
                ann_file = f.readlines()
                annotations = self._process_file(ann_file)

        return annotations


class ProcessLeadMine():

    def _process_file(self, ann_file):
        """
        Iterate through file, keeping track of pmids
        """
        annotations = {}
        file_ann = []
        pmid1 = ""
        #pmid2 = ""
        for i, line in enumerate(ann_file):
            if i != 0:
                ann = line.split("\t")
                if i == 1:
                    pmid1 = ann[1]
                    if ann[0] in ["chemical", "geneprotein", "antibody"]:
                        file_ann.append(ann[7].strip())
                elif i > 1:
                    if pmid1 == ann[1]:
                        if ann[0] in ["chemical", "geneprotein", "antibody"]:
                            file_ann.append(ann[7].strip())
                    elif pmid1 != ann[1]:
                        annotations[pmid1] = set(file_ann)
                        file_ann = []
                        pmid1 = ann[1]
                        if ann[0] in ["chemical", "geneprotein", "antibody"]:
                            file_ann.append(ann[7].strip())
        # And get that last one
        annotations[pmid1] = set(file_ann)
        return annotations

    def get_annotations(self):
        """
        Annotations are all in one file.
        PMIDS are per line
        """
        with open("tool_annotations/NCBI_leadmine.200selectedPMIDs", "r", encoding="utf8") as f:
            ann_file = f.readlines()
            annotations = self._process_file(ann_file)
        return annotations


class ProcessBratCitations():

    def _process_brat(self, citation_ann):
        """
        Process .ann file and return
        set of entities
        """

        annotation = []
        annotation_offset_dict = {}
        for i, line in enumerate(citation_ann):
            try:
                ann = line.strip().split("\t")
                if "T" in ann[0]:
                    annotation.append(ann[2].strip())

            except Exception as e:
                print("Error in processing brat citation")
                print(ann)

        annotation = set(annotation)

        return annotation

    def get_annotations(self, tool):
        """
        Load annotations and return
        the set of entities, depending on
        the tool
        """

        annotations = {}
        tool_dir = {
            "mml_ctb": "tool_annotations/mml_ctb/*.ann",
            "mml_cid_mesh": "tool_annotations/chemical_citations_mml_cid_mesh/*.ann",
            "mml_umls_chem": "tool_annotations/chemical_citations_mml_umls_chem_sg_subset/*.ann",
            "chemlistem": "tool_annotations/chemical_citations_chemlistem/*.ann",
            "lstm_char_embed": "tool_annotations/chemical_citations_tf_ner_chars_lstm_lstm_crf_ema/*ann",
            "lstm_char_embed_chemdner": "tool_annotations/chemical_citations_tf_ner_chars_lstm_lstm_crf_ema_chemdner1/*ann",
            "lstm_char_embed_patents": "tool_annotations/chemical_citations_tf_ner_chars_lstm_lstm_crf_ema_patents/*ann",
            "mml_cid_filtered": "tool_annotations/chemical_citations_mml_cid_filtered/*.ann",
            }

        for f in glob.iglob(tool_dir[tool]):
            pmid = f.split("\\")[1].split(".")[0]
            with open("{}".format(f), "r", encoding="utf8") as citation:
                citation_ann = citation.readlines()
                annotations[pmid] = self._process_brat(set(citation_ann))

        return annotations


class ProcessManualCitations():
    """
    Class to process manual citation data
    and get set of terms for each citation.
    """

    def __init__(self):
        """
        Init variables for counting all entities
        """

        self.org_cnt = 0
        self.inorg_cnt = 0
        self.gene_cnt = 0
        self.pro_cnt = 0
        self.total_cnt = 0
        self.unique_org_cnt = 0
        self.unique_inorg_cnt = 0
        self.unique_gene_cnt = 0
        self.unique_pro_cnt = 0
        self.nested_org_cnt = 0
        self.nested_inorg_cnt = 0
        self.nested_gene_cnt = 0
        self.nested_pro_cnt = 0

    def count_nested_entities(self, nested_entity_classes):
        """
        Count number of entites nested in parent annotations
        """

        if nested_entity_classes != []:
            for ann_class in nested_entity_classes:
                if ann_class == "organic":
                    self.nested_org_cnt += 1
                elif ann_class == "inorganic":
                    self.nested_inorg_cnt += 1
                elif ann_class == "nucleotides":
                    self.nested_gene_cnt += 1
                elif ann_class == "peptides":
                    self.nested_pro_cnt += 1
                else:
                    print("Unexpected behavior")

    def count_unique_entities(self, ann, ann_class):
        """
        Count number of entities in set of each type
        """

        ochem = []
        ichem = []
        pro = []
        gene = []
        for i, ann_type in zip(ann, ann_class):
            if ann_type == "organic":
                ochem.append(i)
            elif ann_type == "inorganic":
                ichem.append(i)
            elif ann_type == "nucleotides":
                gene.append(i)
            elif ann_type == "peptides":
                pro.append(i)

        ochem = set(ochem)
        ichem = set(ichem)
        pro = set(pro)
        gene = set(gene)
        self.unique_org_cnt += len(ochem)
        self.unique_inorg_cnt += len(ichem)
        self.unique_gene_cnt += len(gene)
        self.unique_pro_cnt += len(pro)

    def count_entities(self, ann_class):
        """
        Count entity types
        """

        self.total_cnt += len(ann_class)
        for i in ann_class:
            if i == "organic":
                self.org_cnt += 1
            elif i == "inorganic":
                self.inorg_cnt += 1
            elif i == "nucleotides":
                self.gene_cnt += 1
            elif i == "peptides":
                self.pro_cnt += 1
            else:
                print("Unexpected behavior")

    def entity_count(self):
        return self.org_cnt, self.inorg_cnt, self.gene_cnt, self.pro_cnt, self.total_cnt

    def unique_entity_count(self):
        return self.unique_org_cnt, self.unique_inorg_cnt, self.unique_gene_cnt, self.unique_pro_cnt

    def nested_entity_count(self):
        return self.nested_org_cnt, self.nested_inorg_cnt, self.nested_gene_cnt, self.nested_pro_cnt

    def _process_offsets(self, annotation_offset_dict, pmid):
        """
        Check to see which offsets overlap and return a list of these
        """

        fcnt = 0
        nested_entities = []
        nested_entity_class = []
        for ann1 in annotation_offset_dict:
            if len(annotation_offset_dict[ann1][0]) == 3:
                fcnt += 1
                assert int(annotation_offset_dict[ann1][0][0]) < int(annotation_offset_dict[ann1][0][2]), annotation_offset_dict[ann1]
            for ann2 in annotation_offset_dict:
                # Handle the nested non fragments:
                if (len(annotation_offset_dict[ann1][0]) == 2 and
                    len(annotation_offset_dict[ann2][0]) == 2):
                    # Nests where the start at the same point but one ends earlier:
                    if (int(annotation_offset_dict[ann1][0][0]) == int(annotation_offset_dict[ann2][0][0]) and
                        int(annotation_offset_dict[ann1][0][1]) < int(annotation_offset_dict[ann2][0][1])):
                        nested_entities.append(annotation_offset_dict[ann1][1].strip())
                        nested_entity_class.append(annotation_offset_dict[ann1][2])
                        #print("Nested annotation 1")
                        #print(annotation_offset_dict[ann1], annotation_offset_dict[ann2])
                    # Nests where one starts later but ends at the same point or earlier:
                    elif (
                        int(annotation_offset_dict[ann1][0][0]) > int(annotation_offset_dict[ann2][0][0]) and
                        int(annotation_offset_dict[ann1][0][1]) <= int(annotation_offset_dict[ann2][0][1])):
                        nested_entities.append(annotation_offset_dict[ann1][1].strip())
                        nested_entity_class.append(annotation_offset_dict[ann1][2])
                # Since there are no nested fragments, I don't have to
                # keep track of them because they will exist as their own entities

        return nested_entities, nested_entity_class

    def _process_brat(self, citation_ann, pmid):
        """
        Process .ann file and return
        set of entities
        """

        nested_annotation = []
        ann_class = []
        annotation_offset_dict = {}
        for i, line in enumerate(citation_ann):
            try:
                ann = line.strip().split("\t")
                if "AnnotatorNotes" in ann[1]:
                    continue
                nested_annotation.append(ann[2].strip())
                ann_class.append(ann[1].split(" ")[0])
                #print(ann[1].split(" "))
                offset = ann[1].split(" ")
                # Handle most spans
                # Include class type in annotation offet dict
                if len(offset) == 3:
                    annotation_offset_dict[i] = [offset[1:3], ann[2], ann[1].split(" ")[0]]
                # Handle fragments, which will have a bit
                # indicating the length the fragment skips.
                elif len(offset) == 4 :
                    annotation_offset_dict[i] = [offset[1:4], ann[2], ann[1].split(" ")[0]]
                else:
                    print("Something fishy:", ann)
            # to get non nested, need to check overlap of the rest of the spans
            except Exception as e:
                print(pmid, ann)

        nested_entities, nested_entity_classes = self._process_offsets(annotation_offset_dict, pmid)
        self.count_entities(ann_class)
        self.count_unique_entities(nested_annotation, ann_class)
        self.count_nested_entities(nested_entity_classes)
        #self.count_fragmented_entities
        nested_annotation = set(nested_annotation)
        annotation = set(nested_annotation)
        [annotation.discard(entity) for entity in nested_entities]
        if len(nested_entities) != 0:
            assert len(annotation) < len(nested_annotation)

        return nested_annotation, annotation

    def get_annotations(self):
        """
        Load annotations and return nested
        and unested set
        """

        nested_annotations = {}
        annotations = {}
        for f in os.listdir("manual_annotations/ChemManualHarmonized_after_corrections_v2"):
        #for f in os.listdir("manual_annotations/citations"):
            if f.endswith(".ann"):
                pmid = f.split(".")[0]
                with open("manual_annotations/ChemManualHarmonized_after_corrections_v2/{}".format(f), "r", encoding="utf8") as citation:
                #with open("manual_annotations/citations/{}".format(f), "r", encoding="utf8") as citation:
                    citation_ann = citation.readlines()
                    nested_annotation, annotation = self._process_brat(set(citation_ann), pmid)
                    nested_annotations[pmid] = nested_annotation
                    annotations[pmid] = annotation

        return nested_annotations, annotations


class Evaluate():

    all_found_entities = set()

    def _calculate_levenshtein(self, man_ann, tool_ann, tool):
        """
        Compute metrics using levenshtein distance as threshold

        This is actually not that useful, because mistkaes aren't being corrected, one found entity is being
        edited to another: Beta-globulin-1 to Beta-globulin-2, for example
        """
        tp = 0
        fn = 0
        fp = 0

        # False positives are those in predictions that
        # are greater than threshold away from any entity
        # in the manual annotations
        # However, often the entity can just be changed to a totally different entity,
        # Sox1 to Sox2. If Sox2 is already predicted, I don't want to give
        # relazed credit to sox1 once changed to sox2. This is what the if statement checks for.
        with open("levenshtein_measurements.txt", "a", encoding="utf8") as f:
            for pred in tool_ann:
                match = False
                for ann in man_ann:
                    lev_dist = levenshtein(pred, ann)
                    if len(pred) == 0:
                        print(tool_ann)
                    norm_lev_dist = lev_dist / len(pred)
                    if (norm_lev_dist < 1/3 and ann not in tool_ann) or pred == ann:
                        if not args.bootstrap and norm_lev_dist != 0:
                            f.write("Annotation: {0}, Prediction: {1}, levenshtein distance: {2}, normalized lev: {3}---{4}\n".format(ann, pred, lev_dist, norm_lev_dist, tool))
                        match = True
                        tp += 1
                        break
                if match == False:
                    fp += 1

        # False negatives are those in manual annotations that
        # are greater than threshold away from any entity
        # in the predictions
        for ann in man_ann:
            match = False
            for pred in tool_ann:
                lev_dist = levenshtein(ann, pred)
                norm_lev_dist = lev_dist / len(pred)
                if (norm_lev_dist < 1/3 and pred not in man_ann) or pred == ann:
                    match = True
                    break
            if match == False:
                fn += 1

        return tp, fp, fn

    def get_all_entities(self):
        """
        Return the set of all found entities,
        using all tools.
        """
        return Evaluate.all_found_entities

    def _load_stop_words(self, lowercase=False):
        """
        Load and return set of stopwords
        """

        with open("stop_words.txt", "r", encoding="utf8") as f:
            stop_words = f.readlines()
            if lowercase:
                stop_words = [word.strip().lower() for word in stop_words]
            else:
                stop_words = [word.strip() for word in stop_words]
            stop_words_set = set(stop_words)
            if not lowercase:
                assert len(stop_words_set) == len(stop_words), "Stop words set is shorter than stop word list"

        return stop_words_set

    def _test_stop_words(self, man_ann, tool_ann, stop_words_set):
        """
        Do some tests of stop words set and the manual annotations
        """

        man_ann_reduced = man_ann.difference(stop_words_set)
        # Test if any manual annotations have stop words.
        for i in man_ann:
            assert i not in stop_words_set

        # Test if the set of stopwords is cased correctly
        cased_intersection = stop_words_set.intersection(tool_ann)
        stop_words_lower = set([word.lower() for word in stop_words_set])
        tool_ann_lower = set([word.lower() for word in tool_ann])
        uncased_intersection = stop_words_lower.intersection(tool_ann_lower)
        if len(cased_intersection) != len(uncased_intersection):
            print(uncased_intersection.difference(cased_intersection))
            print(cased_intersection.difference(uncased_intersection))

        #assert len(man_ann_reduced) == len(man_ann)

    def _remove_stop_words(self, tool_ann, stop_words_set):
        """
        Remove words from tool annotation set
        that are in stopword list.

        Also do some tests of stop words.
        """
        tool_ann_reduced = tool_ann.difference(stop_words_set)
        #self._test_stop_words(man_ann, tool_ann, stop_words_set)
        return tool_ann_reduced

    def _recall(self, fn, tp):
        """
        Function to compute recall of total counts
        from entities in documents

        Denominator = relevant documents = in both sets + in manual annotated set
        """
        return tp / (fn + tp)

    def _precision(self, fp, tp):
        """
        Function to compute precision of total counts
        from entities in documents

        Denominator = retrieved documents = in both sets + in tool annotated set
        """
        return tp / (fp + tp)

    def _f_score(self, precision, recall, beta):
        """
        Calculate f_score, using provided beta value.
        """
        return (1 + beta**2) * ((precision*recall) / ((precision * beta**2) + recall))

    def _true_positives(self, man_ann, tool_ann):
        """
        Intersection of manual annotations
        and tool annotation
        """
        #print(man_ann.intersection(tool_ann))
        return len(man_ann.intersection(tool_ann))

    def _false_positives(self, man_ann, tool_ann):
        """
        Count entities in tool ann but not in manual ann.
        """
        return len(tool_ann.difference(man_ann))

    def _false_negatives(self, man_ann, tool_ann):
        """
        Count entities in manual ann but not in tool ann.
        """
        return len(man_ann.difference(tool_ann))

    def _write_results(self, corpus, precision, recall, f_score, beta, tool):
        """
        Output results to file
        """
        if args.use_leven:
            f_name = "result_printouts/results_tool_evaluation_leven.txt"
        else:
            f_name = "result_printouts/results_tool_evaluation.txt"
        with open(f_name, "a+") as f:
            f.write("{0} on {1}\nPrecision: {2}\nRecall: {3}\nF{4}-score: {5}\n\n".format(tool, corpus, precision, recall, beta, f_score))

    def compare_annotations(self, corpus, man_anns, tool_anns, tool):
        """
        Compute precision and recall between manual annotations
        and a tool's annotations
        """

        fn = 0
        fp = 0
        tp = 0

        if tool == "LeadMine":
            man_anns = {pmid: man_anns[pmid] for pmid in man_anns if pmid in tool_anns}
            tool_anns = {pmid: tool_anns[pmid] for pmid in tool_anns if pmid in man_anns}
            assert len(man_anns) == len(tool_anns)
        # if tool == "XLNet ensemble" or "XLNet Gene" or "XLNet chemdner":
        #     for pmid in tool_anns:
        #         xl_ann = list(tool_anns[pmid])
        #         tool_anns[pmid] = set([i for i in xl_ann if "unk" not in i])
        if not args.include_stop:
            stop_words_set = self._load_stop_words()
        for man_ann in man_anns:
            if not args.include_stop:
                tool_anns[man_ann] = self._remove_stop_words(tool_anns[man_ann], stop_words_set)
            # Make set of every entity found by every tool:
            if tool not in ["Nested", "Unnested"]:
                Evaluate.all_found_entities = Evaluate.all_found_entities.union(tool_anns[man_ann])
            if args.use_leven:
                l_tp, l_fp, l_fn = self._calculate_levenshtein(man_anns[man_ann], tool_anns[man_ann], tool)
                tp += l_tp
                fp += l_fp
                fn += l_fn
            else:
                tp += self._true_positives(man_anns[man_ann], tool_anns[man_ann])
                fn += self._false_negatives(man_anns[man_ann], tool_anns[man_ann])
                fp += self._false_positives(man_anns[man_ann], tool_anns[man_ann])

            #Bit for writing false negatives and positives pmid by pmid
            if args.error_analysis:
                if tool == "XLNet Gene":
                    with open("error_analysis/XLgene_false_negatives", "a+", encoding="utf8") as f:
                        f.write("{0}\t{1}\n".format(man_ann, man_anns[man_ann].difference(tool_anns[man_ann])))
                    with open("error_analysis/XLgene_false_positives", "a+", encoding="utf8") as f:
                        f.write("{0}\t{1}\n".format(man_ann, tool_anns[man_ann].difference(man_anns[man_ann])))

        precision = self._precision(fp, tp)
        recall = self._recall(fn, tp)
        beta = 1
        f_score = self._f_score(precision, recall, beta)
        self._write_results(corpus, precision, recall, f_score, beta, tool)
        return np.round(precision, 4), np.round(recall, 4), np.round(f_score, 4)


class EvaluateBootstrap(Evaluate):
    """
    Class for evaluation when bootstrapping citations

    Uses lists of annotations and keys instead of dictionaries.
    """

    def __init__(self):
        """
        Load stop words
        """
        self.stop_words_set = self._load_stop_words()

        if args.use_leven:
            f_name = "result_printouts/results_tool_evaluation_leven_bootstrap.txt"
        else:
            f_name = "result_printouts/results_tool_evaluation_bootstrap.txt"
        results = open(f_name, "w")
        results.close()

    def compare_annotations(self, man_anns, tool_anns, keys, tool):
        """
        Compute precision and recall between manual annotations
        and a tool's annotations
        """

        fn = 0
        fp = 0
        tp = 0

        for i, key in enumerate(keys):
            # Iterate through bootstrapped key list
            # Use iterator number to get tool annotations, and key to get manual annotations
            if not args.include_stop:
                tool_anns[i] = self._remove_stop_words(tool_anns[i], self.stop_words_set)
            if args.use_leven:
                l_tp, l_fp, l_fn = self._calculate_levenshtein(man_anns[key], tool_anns[i], tool)
                tp += l_tp
                fp += l_fp
                fn += l_fn
            else:
                tp += self._true_positives(man_anns[key], tool_anns[i])
                fn += self._false_negatives(man_anns[key], tool_anns[i])
                fp += self._false_positives(man_anns[key], tool_anns[i])

        precision = self._precision(fp, tp)
        recall = self._recall(fn, tp)
        beta = 1
        f_score = self._f_score(precision, recall, beta)

        return np.round(precision, 4), np.round(recall, 4), np.round(f_score, 4)

    def write_results(self, avg_p, avg_r, avg_f, std_p, std_r, std_f, tool_labels):
        """
        Output results to file
        """
        for i, tool in enumerate(tool_labels):
            if args.use_leven:
                f_name = "result_printouts/results_tool_evaluation_leven_bootstrap.txt"
            else:
                f_name = "result_printouts/results_tool_evaluation_bootstrap.txt"
            with open(f_name, "a+") as f:
                f.write("{0}: \nF-score: {1} +/- {2}\nPrecision: {3} +/- {4}\nRecall: {5} +/- {6}\n\n".format(tool, avg_f[i], std_f[i], avg_p[i], std_p[i], avg_r[i], std_r[i]))


class EvaluateMeSH(Evaluate):
    """
    Note: Lower case everything
    """

    def __init__(self):
        """
        Load stopwords and mesh for citations
        """

        if not args.include_stop:
            self.stop_words_set = self._load_stop_words(lowercase=True)

        with open("citations_mesh_matched.json", "r", encoding="utf8") as f:
            self.citation_indexing = json.load(f)
        self.mesh, self.scr, self.mesh_scr = self.parse_mesh()

        if args.use_leven:
            f_name = "result_printouts/results_tool_evaluation_leven_bootstrap_scr.txt"
        else:
            f_name = "result_printouts/results_tool_evaluation_bootstrap_scr.txt"
        results = open(f_name, "w")
        results.write("Tool, F-score, Precision, Recall\n")
        results.close()

    def parse_mesh(self):
        """
        For a given citation, parse the lists of SCRs and MeSH
        """

        mesh_dict = {}
        scr_dict = {}
        mesh_scr_dict = {}

        for pmid in self.citation_indexing:
            mesh_list = self.citation_indexing[pmid][0].split("|")
            mesh_list = [i.lower() for i in mesh_list]
            scr_list = self.citation_indexing[pmid][1].split("|")
            scr_list = [i.lower() for i in scr_list]
            mesh_dict[pmid] = set(mesh_list)
            scr_dict[pmid] = set(scr_list)
            if not args.include_stop:
                mesh_dict[pmid] = self._remove_stop_words(mesh_dict[pmid], self.stop_words_set)
                scr_dict[pmid] = self._remove_stop_words(scr_dict[pmid], self.stop_words_set)
            mesh_scr_dict[pmid] = scr_dict[pmid].union(mesh_dict[pmid])

        return mesh_dict, scr_dict, mesh_scr_dict

    def bootstrap_annotations(self, tool_anns, keys, tool):
        """
        Compute precision and recall between manual annotations
        and a tool's annotations
        """

        fn = 0
        fp = 0
        tp = 0
        mesh = self.scr
        for i, key in enumerate(keys):
            # Iterate through bootstrapped key list
            # Use iterator number to get tool annotations, and key to get manual annotations
            if key == "30880703":
                continue
            if not args.include_stop:
                tool_anns[i] = self._remove_stop_words(tool_anns[i], self.stop_words_set)
            if args.use_leven:
                l_tp, l_fp, l_fn = self._calculate_levenshtein(mesh[key], tool_anns[i], tool)
                tp += l_tp
                fp += l_fp
                fn += l_fn
            else:
                tp += self._true_positives(mesh[key], tool_anns[i])
                fn += self._false_negatives(mesh[key], tool_anns[i])
                fp += self._false_positives(mesh[key], tool_anns[i])

        precision = self._precision(fp, tp)
        recall = self._recall(fn, tp)
        beta = 1
        f_score = self._f_score(precision, recall, beta)

        return np.round(precision, 4), np.round(recall, 4), np.round(f_score, 4)

    def compare_mesh(self, tool_anns, tool):
        """
        Compare Mesh to each tool, including manual annotations
        """

        fn = 0
        fp = 0
        tp = 0
        # Set MeSH type to compare tool annotations to.
        mesh = self.scr
        for pmid in tool_anns:
            # Make lowercase:
            tool_anns[pmid] = set(i.lower() for i in tool_anns[pmid])
            # This one wasn't indexed
            if pmid == "30880703":
                continue
            if not args.include_stop:
                tool_anns[pmid] = self._remove_stop_words(tool_anns[pmid], self.stop_words_set)
            if args.use_leven:
                l_tp, l_fp, l_fn = self._calculate_levenshtein(mesh[pmid], tool_anns[pmid], tool)
                tp += l_tp
                fp += l_fp
                fn += l_fn
            else:
                tp += self._true_positives(mesh[pmid], tool_anns[pmid])
                fn += self._false_negatives(mesh[pmid], tool_anns[pmid])
                fp += self._false_positives(mesh[pmid], tool_anns[pmid])

            if args.error_analysis:
                with open("error_analysis/{}_mesh_false_negatives".format(tool), "a+", encoding="utf8") as f:
                    f.write("{0}\t{1}\n".format(pmid, mesh[pmid].difference(tool_anns[pmid])))
                with open("error_analysis/{}_mesh_false_positives".format(tool), "a+", encoding="utf8") as f:
                    f.write("{0}\t{1}\n".format(pmid, tool_anns[pmid].difference(mesh[pmid])))

        precision = self._precision(fp, tp)
        recall = self._recall(fn, tp)
        beta = 1
        f_score = self._f_score(precision, recall, beta)
        #self._write_results(corpus, precision, recall, f_score, beta, tool)

        return np.round(precision, 4), np.round(recall, 4), np.round(f_score, 4)

    def write_bootstrap_results(self, avg_p, avg_r, avg_f, std_p, std_r, std_f, tool_labels):
        """
        Output results to file
        """
        for i, tool in enumerate(tool_labels):
            if args.use_leven:
                f_name = "result_printouts/results_tool_evaluation_leven_bootstrap_scr.txt"
            else:
                f_name = "result_printouts/results_tool_evaluation_bootstrap_scr.txt"
            with open(f_name, "a+") as f:
                f.write("{0}, {1} +/- {2}, {3} +/- {4}, {5} +/- {6}\n".format(tool, avg_f[i], std_f[i], avg_p[i], std_p[i], avg_r[i], std_r[i]))


def visualize_bar(nested_precision, nested_recall, unnested_precision, unnested_recall, tool_labels):
    """
    Make bar graph of results.
    For each tool show unnested and nested, with
    and without the stop words removed
    """

    opacity = 0.4
    bar_width = 0.2
    n_groups = len(tool_labels)
    index = np.arange(n_groups)
    fig, ax = plt.subplots(figsize=(15, 8))
    #ax[1].legend(prop={'size': 9})
    ax.bar(index - bar_width, nested_precision, bar_width, alpha=opacity, color="b", label="Nested Precision")
    ax.bar(index, nested_recall, bar_width, color="g", alpha=opacity, label="Nested Recall")
    ax.bar(index + bar_width, unnested_precision, bar_width, color="r", alpha=opacity, label="Unnested Precision")
    ax.bar(index + 2*bar_width, unnested_recall, bar_width, color="m", alpha=opacity, label="Unnested Recall")
    ax.set_xlabel('Chemical Recognition Tool')
    ax.set_ylabel('Score')
    ax.set_title('Performance of Chemical Regognition Tools on Corpus')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(tool_labels, rotation=60)
    ax.legend()

    fig.tight_layout()
    if args.use_leven:
        f_name = "figures/evaluation_chem_tools_leven.png"
    else:
        f_name = "figures/evaluation_chem_tools.png"
    fig.savefig(f_name, bbox_inches='tight')
    #plt.show()
    plt.close()


def visualize_table(nested_precision, nested_recall, nested_fscore, tool_labels, unnested_precision=None, unnested_recall=None, unnested_fscore=None):
    """
    Generate a table with all tools performance, including f score and levenshtein score.
    """

    fig, ax = plt.subplots(figsize=(30, 15))

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    cell_text = []
    if unnested_precision == None:
        colLabels = ["Nested F1", "Nested Precision", "Nested Recall"]
    else:
        colLabels = ["Nested F1", "Nested Precision", "Nested Recall", "Unnested F1", "Unnested Precision", "Unnested Recall"]
    for i, tool in enumerate(tool_labels):
        if unnested_precision == None:
            cell_text.append([nested_fscore[i], nested_precision[i], nested_recall[i]])
        else:
            cell_text.append([nested_fscore[i], nested_precision[i], nested_recall[i], unnested_fscore[i], unnested_precision[i], unnested_recall[i]])
    metric_table = ax.table(cellText=cell_text,
                            rowLabels=tool_labels,
                            #rowColours=colors,
                            colWidths=[0.05 for x in colLabels],
                            colLabels=colLabels,
                            loc='center'
                            )
    metric_table.set_fontsize(20)
    metric_table.scale(2, 2)
    #fig.tight_layout()
    if args.use_leven:
        if args.mesh:
            f_name = "figures/evaluation_chem_tools_table_leven_scr.png"
        else:
            f_name = "figures/evaluation_chem_tools_table_leven.png"
    else:
        if args.mesh:
            f_name = "figures/evaluation_chem_tools_table_scr.png"
        else:
            f_name = "figures/evaluation_chem_tools_table.png"
    #plt.xticks([])
    fig.savefig(f_name)
    #plt.show()
    plt.close()


def visualize_bootstrap(avg_nested_precision, avg_nested_recall, avg_nested_fscore, std_p, std_r, std_f, tool_labels):
    """
    Generate a table with all tools performance, including f score and levenshtein score.
    """

    fig, ax = plt.subplots(figsize=(20, 15))

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    cell_text = []
    colLabels = ["Average Nested F1", "Average Nested Precision", "Average Nested Recall", "Standard Deviation F1", "Standard Deviation Precision", "Standard Deviation Recall"]
    for i, tool in enumerate(tool_labels):
        cell_text.append([avg_nested_fscore[i], avg_nested_precision[i], avg_nested_recall[i], std_f[i], std_p[i], std_r[i]])
    metric_table = ax.table(cellText=cell_text,
                            rowLabels=tool_labels,
                            #rowColours=colors,
                            colWidths=[0.1 for x in colLabels],
                            colLabels=colLabels,
                            loc='center'
                            )
    metric_table.set_fontsize(10)
    fig.tight_layout()
    if args.use_leven:
        if args.mesh_bootstrap:
            f_name = "figures/evaluation_chem_tools_table_leven_bootstrap_scr.png"
        else:
            f_name = "figures/evaluation_chem_tools_table_leven_bootstrap.png"
    else:
        if args.mesh_bootstrap:
            f_name = "figures/evaluation_chem_tools_table_bootstrap_scr.png"
        else:
            f_name = "figures/evaluation_chem_tools_table_bootstrap.png"
    #plt.xticks([])
    fig.savefig(f_name), #bbox_inches='tight')
    plt.show()


def adjust_bert_preds(annotations, bert_ann):
    """
    Add blank set for pmids with no predictions
    """
    for i in annotations:
        if i not in bert_ann:
            bert_ann[i] = set()
    return bert_ann


def main():
    """
    Get the manual annotation data and the
    annotated data from each tool
    and compare each.
    """

    if not args.bootstrap:
        if args.use_leven:
            lev_file = open("levenshtein_measurements.txt", "w", encoding="utf8")
            lev_file.close()
            f_name = "result_printouts/results_tool_evaluation_leven.txt"
        else:
            f_name = "result_printouts/results_tool_evaluation.txt"
        results = open(f_name, "w")
        results.close()

    ann_processor = ProcessManualCitations()
    nested_annotations, annotations = ann_processor.get_annotations()
    print("Organic, inorganic, gene protein counts:")
    print(ann_processor.entity_count())
    print(" Unique organic, inorganic, gene protein counts:")
    print(ann_processor.unique_entity_count())
    print(" Nested organic, inorganic, gene protein counts:")
    print(ann_processor.nested_entity_count())
    mml_CTB_ann = ProcessBratCitations().get_annotations(tool="mml_ctb")
    mml_cid_mesh_ann = ProcessBratCitations().get_annotations(tool="mml_cid_mesh")
    mml_cid_filtered_ann = ProcessBratCitations().get_annotations(tool="mml_cid_filtered")
    mml_umls_chem_ann = ProcessBratCitations().get_annotations(tool="mml_umls_chem")
    chemlistem_ann = ProcessBratCitations().get_annotations(tool="chemlistem")
    lstm_char_embed_ann = ProcessBratCitations().get_annotations(tool="lstm_char_embed")
    lstm_char_embed_chemdner_ann = ProcessBratCitations().get_annotations(tool="lstm_char_embed_chemdner")
    lstm_char_embed_patents_ann = ProcessBratCitations().get_annotations(tool="lstm_char_embed_patents")
    chemdataextractor_ann = ProcessChemDataExtractor().get_annotations()
    pubtator_ann = ProcessPubTator().get_annotations("PT")
    ptc_ann = ProcessPubTator().get_annotations("PTC")
    leadmine_ann = ProcessLeadMine().get_annotations()
    bert_ctb_ann = ProcessBERT().get_annotations("bert")
    cde_cml_bert_ann = ProcessBERT().get_annotations("cde_cml_bert")
    cde_bert_ann = ProcessBERT().get_annotations("cde_bert")
    scibert_ann = ProcessBERT().get_annotations("scibert")
    chemdner_bert_ann = ProcessBERT().get_annotations("cdner_bert")
    gene_bert_ann = ProcessBERT().get_annotations("gm_bert")
    ensemble_bert_ann = ProcessBERTEnsemble().get_annotations(annotations, "bert")
    scibert_chemdner_ann = ProcessBERT().get_annotations("scibert_chemd")
    ensemble_scibert_ann = ProcessBERTEnsemble().get_annotations(annotations, "scibert")
    ensemble_biobert_ann = ProcessBERTEnsemble().get_annotations(annotations, "biobert")
    xlnet_cdner_ann = ProcessBERT().get_annotations("xlnet_cdner")
    xlnet_gene_ann = ProcessBERT().get_annotations("xlnet_gene")
    xlnet_ensemble_ann  = ProcessBERTEnsemble().get_annotations(annotations, "xlnet")
    MTI_ann = ProcessMTI().get_annotations()

    bert_ctb_ann = adjust_bert_preds(annotations, bert_ctb_ann)
    cde_cml_bert_ann = adjust_bert_preds(annotations, cde_cml_bert_ann)
    cde_bert_ann = adjust_bert_preds(annotations, cde_bert_ann)
    chemdner_bert_ann = adjust_bert_preds(annotations, chemdner_bert_ann)
    scibert_ann = adjust_bert_preds(annotations, scibert_ann)
    scibert_chemdner_ann = adjust_bert_preds(annotations, scibert_chemdner_ann)
    ensemble_bert_ann = adjust_bert_preds(annotations, ensemble_bert_ann)
    ensemble_scibert_ann = adjust_bert_preds(annotations, ensemble_scibert_ann)
    ensemble_biobert_ann = adjust_bert_preds(annotations, ensemble_biobert_ann)
    gene_bert_ann = adjust_bert_preds(annotations, gene_bert_ann)
    xlnet_cdner_ann = adjust_bert_preds(annotations, xlnet_cdner_ann)
    xlnet_gene_ann = adjust_bert_preds(annotations, xlnet_gene_ann)
    xlnet_ensemble_ann = adjust_bert_preds(annotations, xlnet_ensemble_ann)

    assert len(nested_annotations) == 200, "nested"
    assert len(annotations) == 200, "not nested"
    assert len(mml_CTB_ann) == 200, "mml ctb"
    assert len(mml_cid_mesh_ann) == 200, "cid mesh"
    assert len(mml_cid_filtered_ann) == 200, "filtered cid {}".format(len(mml_cid_filtered_ann))
    assert len(mml_umls_chem_ann) == 200, "umls"
    assert len(chemlistem_ann) == 200, "chemlistem"
    assert len(lstm_char_embed_ann) == 200, "lstm_char_embed"
    assert len(lstm_char_embed_chemdner_ann) == 200, "lstm_char_embed_chemdner {}".format(len(lstm_char_embed_chemdner_ann))
    assert len(lstm_char_embed_patents_ann) == 200, "lstm_char_embed_patents"
    assert len(chemdataextractor_ann) == 200, "cde: {}".format(len(chemdataextractor_ann))
    assert len(pubtator_ann) == 200, "pubtator"
    assert len(ptc_ann) == 200, "ptc_ann"
    assert len(bert_ctb_ann) == 200, "bert_ctb_ann: {}".format(len(bert_ctb_ann))
    assert len(chemdner_bert_ann) == 200, "bert_ann: {}".format(len(chemdner_bert_ann))
    assert len(gene_bert_ann) == 200, "bert_ann: {}".format(len(gene_bert_ann))
    assert len(ensemble_bert_ann) == 200, "bert_ann: {}".format(len(ensemble_bert_ann))
    assert len(ensemble_scibert_ann) == 200, "bert_ann: {}".format(len(ensemble_scibert_ann))
    assert len(ensemble_biobert_ann) == 200, "bert_ann: {}".format(len(ensemble_biobert_ann))
    assert len(cde_cml_bert_ann) == 200, "bert_ann: {}".format(len(cde_cml_bert_ann))
    assert len(cde_bert_ann) == 200, "bert_ann: {}".format(len(cde_bert_ann))
    assert len(scibert_ann) == 200, "scibert_ann: {}".format(len(scibert_ann))
    assert len(scibert_chemdner_ann) == 200, "scibert_chemder_ann: {}".format(len(scibert_chemdner_ann))
    assert len(xlnet_cdner_ann) == 200, "scibert_ann: {}".format(len(xlnet_cdner_ann))
    assert len(xlnet_gene_ann) == 200, "scibert_ann: {}".format(len(xlnet_gene_ann))
    assert len(xlnet_ensemble_ann) == 200, "scibert_ann: {}".format(len(xlnet_ensemble_ann))
    assert len(MTI_ann) == 200, "MTI"

    for i in annotations:
        try:
            assert i in mml_CTB_ann
            assert i in mml_cid_mesh_ann
            assert i in mml_cid_filtered_ann
            assert i in mml_umls_chem_ann
            assert i in chemlistem_ann
            assert i in chemdataextractor_ann
            assert i in pubtator_ann
            assert i in ptc_ann
            assert i in lstm_char_embed_ann, "lstm char"
            assert i in lstm_char_embed_chemdner_ann, "lstm char chemdner"
            assert i in lstm_char_embed_patents_ann, "lstm char patents"
            assert i in leadmine_ann, "leadmine"
            assert i in bert_ctb_ann, "bert"
            assert i in chemdner_bert_ann, "cdner bert"
            assert i in scibert_ann, "scibert"
            assert i in scibert_chemdner_ann, "scibert_chemdner"
            assert i in ensemble_bert_ann, "ensemble bert"
            assert i in ensemble_scibert_ann, "ensemble scibert"
            assert i in ensemble_biobert_ann, "ensemble biobert"
            assert i in gene_bert_ann, "gene bert"
            assert i in cde_bert_ann, "cde bert"
            assert i in xlnet_cdner_ann, "xlnet_cdner_ann"
            assert i in xlnet_gene_ann, "xlnet_gene_ann"
            assert i in xlnet_ensemble_ann, "xlnet_ensemble_ann"
            assert i in MTI_ann, "MTI"
        except AssertionError as e:
            print(i, e)

    tools = {
        "Nested": nested_annotations,
        "Unnested": annotations,
        "MTI": MTI_ann,
        "MML + CTB": mml_CTB_ann,
        "MML + CID MeSH": mml_cid_mesh_ann,
        "MML + CID filtered": mml_cid_filtered_ann,
        "MML + UMLS chemical semantic group": mml_umls_chem_ann,
        "PubTator": pubtator_ann,
        "PubTator Central": ptc_ann,
        "ChemListem": chemlistem_ann,
        "lstm_char_embed": lstm_char_embed_ann,
        "lstm_char_embed_chemdner": lstm_char_embed_chemdner_ann,
        "lstm_char_embed_patents": lstm_char_embed_patents_ann,
        "ChemDataExtractor": chemdataextractor_ann,
        "LeadMine": leadmine_ann,
        "BERT weak supervision": bert_ctb_ann,
        "SciBERT weak supervision": scibert_ann,
        "CDE-ChemListem BERT": cde_cml_bert_ann,
        "CDE BERT": cde_bert_ann,
        "ChemDNER BERT": chemdner_bert_ann,
        "Gene BERT": gene_bert_ann,
        "BERT Ensemble": ensemble_bert_ann,
        "SciBERT chemdner": scibert_chemdner_ann,
        "SciBERT Ensemble": ensemble_scibert_ann,
        "BioBERT Ensemble": ensemble_biobert_ann,
        "XLNet chemdner": xlnet_cdner_ann,
        "XLNet Gene": xlnet_gene_ann,
        "XLNet ensemble": xlnet_ensemble_ann,
        }

    nested_precision = []
    nested_recall = []
    nested_fscore = []
    unnested_precision = []
    unnested_recall = []
    unnested_fscore = []
    tool_labels = []

    if not args.bootstrap:
        for tool in tools:
            print("Evaluating ", tool)
            tool_labels.append(tool)
            p, r, f = Evaluate().compare_annotations("nested", nested_annotations, tools[tool], tool)
            nested_precision.append(p)
            nested_recall.append(r)
            nested_fscore.append(f)
            p, r, f = Evaluate().compare_annotations("unnested", annotations, tools[tool], tool)
            unnested_precision.append(p)
            unnested_recall.append(r)
            unnested_fscore.append(f)
            with open("annotation_sets/{}_set.txt".format(tool), "w", encoding="utf8") as f:
                for ann in tools[tool]:
                    f.write("{0}:{1}\n".format(ann, tools[tool][ann]))

        with open("annotation_sets/nested_all_entities.txt", "w", encoding="utf8") as f:
            all_man_anns = set()
            for ann in nested_annotations:
                all_man_anns = all_man_anns.union(nested_annotations[ann])
            for i in all_man_anns:
                f.write("{}\n".format(i))

        # Metrics for all tools compared to all annotated entites
        found_entities = Evaluate().get_all_entities()
        print("Precision and Recall of set of all entities found by tools:")
        print(len(all_man_anns.intersection(found_entities)))
        print(len(all_man_anns.intersection(found_entities))/(len(all_man_anns.intersection(found_entities))+len(found_entities.difference(all_man_anns))))
        print(len(all_man_anns.intersection(found_entities))/(len(all_man_anns.intersection(found_entities))+len(all_man_anns.difference(found_entities))))

        visualize_bar(nested_precision, nested_recall, unnested_precision, unnested_recall, tool_labels)
        visualize_table(nested_precision, nested_recall, nested_fscore, tool_labels, unnested_precision, unnested_recall, unnested_fscore)

    elif args.bootstrap:
        # Here I compute std dev and percentile bootstrap.
        # Would also be interesting to compute empirical bootstrap
        # https://stats.stackexchange.com/questions/355781/is-it-true-that-the-percentile-bootstrap-should-never-be-used
        # https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf
        bootstrapper = EvaluateBootstrap()
        nested_precision = [[] for i in range(len(tools))]
        nested_recall = [[] for i in range(len(tools))]
        nested_fscore = [[] for i in range(len(tools))]
        tool_labels = [tool for tool in tools]
        random.seed(13)
        for i in range(1000):
            if i % 200 == 0:
                print("Iteration: {}".format(i))
            keys = random.choices(list(nested_annotations), k=200)
            tool_results = {}
            for j, tool in enumerate(tools):
                if tool == "LeadMine":
                    nested_precision[j].append(.5)
                    nested_recall[j].append(.5)
                    nested_fscore[j].append(.5)
                    continue
                temp_ann = [tools[tool][k] for k in keys]
                p, r, f = bootstrapper.compare_annotations(nested_annotations, temp_ann, keys, tool)
                nested_precision[j].append(p)
                nested_recall[j].append(r)
                nested_fscore[j].append(f)

        std_f = [np.round(statistics.stdev(i), 4) for i in nested_fscore]
        std_r = [np.round(statistics.stdev(i), 4) for i in nested_recall]
        std_p = [np.round(statistics.stdev(i), 4) for i in nested_precision]
        avg_f = [np.round(statistics.mean(i), 4) for i in nested_fscore]
        avg_r = [np.round(statistics.mean(i), 4) for i in nested_recall]
        avg_p = [np.round(statistics.mean(i), 4) for i in nested_precision]
        # ci_f = [np.quantile(i, q=(.025, .975)) for i in nested_fscore]
        # ci_r = [np.quantile(i, q=(.025, .975)) for i in nested_recall]
        # ci_p = [np.quantile(i, q=(.025, .975)) for i in nested_precision]
        ci_f = [np.quantile(i, q=(.34, .68)) for i in nested_fscore]
        ci_r = [np.quantile(i, q=(.34, .68)) for i in nested_recall]
        ci_p = [np.quantile(i, q=(.34, .68)) for i in nested_precision]

        visualize_bootstrap(avg_p, avg_r, avg_f, std_p, std_r, std_f, tool_labels)
        bootstrapper.write_results(avg_p, avg_r, avg_f, std_p, std_r, std_f, tool_labels)
        #visualize_bootstrap(avg_p, avg_r, avg_f, ci_p, ci_r, ci_f, tool_labels)
        #print(std_f)
        #print([avg_f[i]-ci_f[i][0] for i in range(len(avg_f))])

    if args.mesh:
        nested_precision = []
        nested_recall = []
        nested_fscore = []
        tool_labels = []
        evaluator = EvaluateMeSH()
        for tool in tools:
            if tool == "LeadMine":
                nested_precision.append(.5)
                nested_recall.append(.5)
                nested_fscore.append(.5)
                unnested_precision.append(.5)
                unnested_recall.append(.5)
                unnested_fscore.append(.5)
                tool_labels.append(tool)
                continue
            print("Evaluating mesh with", tool)
            tool_labels.append(tool)
            p, r, f = evaluator.compare_mesh(tools[tool], tool)
            nested_precision.append(p)
            nested_recall.append(r)
            nested_fscore.append(f)
            # with open("annotation_sets_mesh/{}_mesh_set.txt".format(tool), "w", encoding="utf8") as f:
            #     for ann in tools[tool]:
            #         f.write("{0}:{1}\n".format(ann, tools[tool][ann]))

        visualize_table(nested_precision, nested_recall, nested_fscore, tool_labels)

    if args.mesh_bootstrap:
        # Here I compute std dev and percentile bootstrap.
        # Would also be interesting to compute empirical bootstrap
        # https://stats.stackexchange.com/questions/355781/is-it-true-that-the-percentile-bootstrap-should-never-be-used
        # https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf
        bootstrapper = EvaluateMeSH()
        nested_precision = [[] for i in range(len(tools))]
        nested_recall = [[] for i in range(len(tools))]
        nested_fscore = [[] for i in range(len(tools))]
        tool_labels = [tool for tool in tools]
        random.seed(13)
        for i in range(1000):
            if i % 200 == 0:
                print("Iteration: {}".format(i))
            keys = random.choices(list(nested_annotations), k=200)
            tool_results = {}
            for j, tool in enumerate(tools):
                if tool == "LeadMine":
                    nested_precision[j].append(.5)
                    nested_recall[j].append(.5)
                    nested_fscore[j].append(.5)
                    continue
                temp_ann = [tools[tool][k] for k in keys]
                p, r, f = bootstrapper.bootstrap_annotations(temp_ann, keys, tool)
                nested_precision[j].append(p)
                nested_recall[j].append(r)
                nested_fscore[j].append(f)

        std_f = [np.round(statistics.stdev(i), 4) for i in nested_fscore]
        std_r = [np.round(statistics.stdev(i), 4) for i in nested_recall]
        std_p = [np.round(statistics.stdev(i), 4) for i in nested_precision]
        avg_f = [np.round(statistics.mean(i), 4) for i in nested_fscore]
        avg_r = [np.round(statistics.mean(i), 4) for i in nested_recall]
        avg_p = [np.round(statistics.mean(i), 4) for i in nested_precision]
        # ci_f = [np.quantile(i, q=(.025, .975)) for i in nested_fscore]
        # ci_r = [np.quantile(i, q=(.025, .975)) for i in nested_recall]
        # ci_p = [np.quantile(i, q=(.025, .975)) for i in nested_precision]
        ci_f = [np.quantile(i, q=(.34, .68)) for i in nested_fscore]
        ci_r = [np.quantile(i, q=(.34, .68)) for i in nested_recall]
        ci_p = [np.quantile(i, q=(.34, .68)) for i in nested_precision]

        visualize_bootstrap(avg_p, avg_r, avg_f, std_p, std_r, std_f, tool_labels)
        bootstrapper.write_bootstrap_results(avg_p, avg_r, avg_f, std_p, std_r, std_f, tool_labels)
        #bootstrapper.write_results(avg_p, avg_r, avg_f, std_p, std_r, std_f, tool_labels)
        #visualize_bootstrap(avg_p, avg_r, avg_f, ci_p, ci_r, ci_f, tool_labels)
        #print(std_f)
        #print([avg_f[i]-ci_f[i][0] for i in range(len(avg_f))])


if __name__ == "__main__":
    global args
    args = get_args().parse_args()
    main()
