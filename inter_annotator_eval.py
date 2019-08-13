"""
Module to compute interannotator agreement

Micro F1-score is computed on the entity mentions in the annotated citations.
Cohens kappa is not computed because of the nature of the task.
"""


import os
import sys
from sklearn.metrics import cohen_kappa_score, f1_score

class ProcessManualCitations():
    """
    Class to process manual citation data

    Once get_annotations is called, returns mentions of terms for each citation.
    """

    def _process_brat(self, citation_ann, pmid):
        """
        Process .ann file and return dictionary with offsets beg;end as key

        In the case of fragments the key will be beg;middle;end
        """

        annotation_offset_dict = {}
        for i, line in enumerate(citation_ann):
            try:
                ann = line.strip().split("\t")
                if "AnnotatorNotes" in ann[1]:
                    continue
                offset = ann[1].split(" ")
                ann_class = ann[1].split(" ")[0]
                if len(offset) == 3:
                    key = offset[1] + ";" + offset[2]
                    if key in annotation_offset_dict:
                        print("Unexpected behavior")
                        print(ann[2])
                    annotation_offset_dict[key] = [ann[2], ann_class]
                # Handle fragments, which will have a bit
                # indicating the length the fragment skips.
                elif len(offset) == 4:
                    key = offset[1] + ";" + offset[2] + ";" + offset[3]
                    if key in annotation_offset_dict:
                        print("Unexpected behavior")
                        print(ann[2])
                    annotation_offset_dict[key] = [ann[2], ann_class]


            except Exception as e:
                print(pmid, ann)

        return annotation_offset_dict

    def get_annotations(self, annotator):
        """
        Load annotations and return nested
        and unested set
        """

        nested_annotations = {}
        annotations = {}
        for f in os.listdir("manual_annotations/ChemManual{}".format(annotator)):
        #for f in os.listdir("manual_annotations/citations"):
            if f.endswith(".ann"):
                pmid = f.split(".")[0]
                with open("manual_annotations/ChemManual{0}/{1}".format(annotator, f), "r", encoding="utf8") as citation:
                #with open("manual_annotations/citations/{}".format(f), "r", encoding="utf8") as citation:
                    citation_ann = citation.readlines()
                    #print(citation_ann)
                    nested_annotation = self._process_brat(set(citation_ann), pmid)
                    nested_annotations[pmid] = nested_annotation

        return nested_annotations


def compute_f(ann1, ann2):
    """
    Compute Micro F1 score, using ann1 as the "true" set

    Iterate through citations and for each citation, check to see which
    annotations from ann1 are in ann2. These are tp. The annotations in
    ann1 not in ann1 are fn and the annotations in ann2 not in ann1
    are fp.

    Computed using mentions and offsets, taking into account class disagreements
    """

    tp = 0
    fp = 0
    fn = 0
    for citation in ann1:
        ann2_copy = dict(ann2[citation])
        # Iterate through offsets
        for ann in ann1[citation]:
            if ann in ann2_copy:
                #print(ann1[citation][ann][1])
                #print(ann2_copy[ann][1])
                if ann1[citation][ann][1] == ann2_copy[ann][1]:
                    tp += 1
                    ann2_copy.pop(ann)
            elif ann not in ann2_copy or ann1[citation][ann][1] == ann2_copy[ann][1]:
                fn += 1
        fp += len(ann2_copy)
        #sys.exit()
    beta = 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = (1 + beta**2) * ((precision*recall) / ((precision * beta**2) + recall))

    return f_score


def main():
    """
    Main function to compute inter-annotator metrics
    """

    ann1 = ProcessManualCitations().get_annotations("Max")
    ann2 = ProcessManualCitations().get_annotations("Malvika")
    #gold = ProcessManualCitations().get_annotations("Harmonized_before_corrections")
    gold = ProcessManualCitations().get_annotations("Harmonized_after_corrections_v2")

    f_score = compute_f(ann1, ann2)
    print("F-score, annotator 1 vs annotator 2\n", f_score)
    f_score = compute_f(gold, ann1)
    print("F-score, gold vs annotator 1\n", f_score)
    f_score = compute_f(gold, ann2)
    print("F-score, gold vs annotator 2\n", f_score)


if __name__ == "__main__":
    main()
