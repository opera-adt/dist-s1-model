Traceback (most recent call last):
  File "/gpfs/scratch/opera-dist-ml/users/jmauro/dist-s1-model/trainer.py", line 514, in <module>
    main()
  File "/gpfs/scratch/opera-dist-ml/users/jmauro/dist-s1-model/trainer.py", line 217, in main
    dist_dataset = DistS1Dataset("/scratch/opera-dist-ml/dist-s1-data-updated") ##TODO: add to config
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gpfs/scratch/opera-dist-ml/users/jmauro/dist-s1-model/src/dataset.py", line 15, in __init__
    self.df = self._load_parquet_files()
              ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gpfs/scratch/opera-dist-ml/users/jmauro/dist-s1-model/src/dataset.py", line 25, in _load_parquet_files
    df_list = [pd.read_parquet(pf) for pf in parquet_files]
               ^^^^^^^^^^^^^^^^^^^
  File "/scratch-jpl/opera-dist-ml/users/jmauro/envs/dist-s1-model-gpu/lib/python3.12/site-packages/pandas/io/parquet.py", line 653, in read_parquet
    impl = get_engine(engine)
           ^^^^^^^^^^^^^^^^^^
  File "/scratch-jpl/opera-dist-ml/users/jmauro/envs/dist-s1-model-gpu/lib/python3.12/site-packages/pandas/io/parquet.py", line 68, in get_engine
    raise ImportError(
ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.
A suitable version of pyarrow or fastparquet is required for parquet support.
Trying to import the above resulted in these errors:
 - Missing optional dependency 'pyarrow'. pyarrow is required for parquet support. Use pip or conda to install pyarrow.
 - Missing optional dependency 'fastparquet'. fastparquet is required for parquet support. Use pip or conda to install fastparquet.
