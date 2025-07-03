Traceback (most recent call last):
  File "/gpfs/scratch/opera-dist-ml/users/jmauro/dist-s1-model/trainer.py", line 514, in <module>
    main()
  File "/gpfs/scratch/opera-dist-ml/users/jmauro/dist-s1-model/trainer.py", line 226, in main
    batch_size = train_config['batch_size'],
                 ^^^^^^^^^^^^
NameError: name 'train_config' is not defined. Did you mean: 'load_config'?
