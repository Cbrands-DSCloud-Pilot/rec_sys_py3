#FROM civisanalytics/civis-jupyter-notebooks
FROM civisanalytics/datascience-python
ENV export OPENBLAS_NUM_THREADS = 1 /
ADD implicit2.py /
ADD ratings.py /
ADD test_implicit2.py /
ADD test_ratings.py /
ADD S3_helper.py /
ADD implicit_exe.py /
RUN pip install implicit