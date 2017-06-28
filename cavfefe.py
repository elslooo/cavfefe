import sys

from lib.cv.cli import *
from lib.ds.cli import *
from lib.sc.cli import *
from lib.lm.cli import *

if __name__ == '__main__':
    task = sys.argv[1]

    if task == '--ds-build-vocabulary':
        ds_build_vocabulary()
    elif task == '--cv-prepare':
        cv_prepare()
    elif task == '--cv-train':
        cv_train()
    elif task == '--cv-evaluate':
        cv_evaluate()
    elif task == '--cv-extract-features':
        cv_extract_features()
    elif task == '--sc-prepare':
        sc_prepare()
    elif task == '--sc-train':
        sc_train()
    elif task == '--sc-evaluate':
        sc_evaluate()
    elif task == '--sc-extract-embeddings':
        sc_extract_embeddings()
    elif task == '--lm-prepare':
        lm_prepare()
    elif task == '--lm-train':
        lm_train()
