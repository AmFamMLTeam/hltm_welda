#!bash

sqlite3 hltm_welda/sql/lda.db < hltm_welda/sql/ddl.sql

cd hltm_welda/model
python setup.py build_ext --inplace
