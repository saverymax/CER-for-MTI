"""
Script to run the ChemDataExtractor on the
200 pmids.

http://chemdataextractor.org/docs/cem
https://pubs.acs.org/doi/abs/10.1021/acs.jcim.6b00207
"""

import os
from chemdataextractor import Document
from chemdataextractor.reader import PlainTextReader

cde_annotations = open("tool_annotations/ChemDataExtractor_annotations.txt", "w", encoding="utf8")

for d in os.listdir("../citations"):
    with open("../citations/{}".format(d), "rb") as f:
        citation = Document.from_file(f, readers=[PlainTextReader()])
        pmid = d.split(".")[0]
        cde_annotations.write("\n{}\n".format(pmid))
        for ann in citation.cems:
            cde_annotations.write("{}\n".format(ann))

cde_annotations.close()
