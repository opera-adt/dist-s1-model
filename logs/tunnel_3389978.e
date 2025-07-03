/gpfs/scratch/opera-dist-ml/users/jmauro/dist-s1-model/src/utils.py:996: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  landslide = torch.load(val_config['landslide']['data_path'])
Traceback (most recent call last):
  File "/gpfs/scratch/opera-dist-ml/users/jmauro/dist-s1-model/trainer.py", line 514, in <module>
    main()
  File "/gpfs/scratch/opera-dist-ml/users/jmauro/dist-s1-model/trainer.py", line 377, in main
    train_loss, train_mse, _, _ = run_epoch_tf(
                                  ^^^^^^^^^^^^^
  File "/gpfs/scratch/opera-dist-ml/users/jmauro/dist-s1-model/trainer.py", line 48, in run_epoch_tf
    for batch_idx, (batch, target) in enumerate(dataloader):
                                      ^^^^^^^^^^^^^^^^^^^^^
  File "/scratch-jpl/opera-dist-ml/users/jmauro/envs/dist-s1-model-gpu/lib/python3.12/site-packages/accelerate/data_loader.py", line 567, in __iter__
    current_batch = next(dataloader_iter)
                    ^^^^^^^^^^^^^^^^^^^^^
  File "/scratch-jpl/opera-dist-ml/users/jmauro/envs/dist-s1-model-gpu/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 701, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "/scratch-jpl/opera-dist-ml/users/jmauro/envs/dist-s1-model-gpu/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 757, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/scratch-jpl/opera-dist-ml/users/jmauro/envs/dist-s1-model-gpu/lib/python3.12/site-packages/torch/utils/data/_utils/fetch.py", line 50, in fetch
    data = self.dataset.__getitems__(possibly_batched_index)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/scratch-jpl/opera-dist-ml/users/jmauro/envs/dist-s1-model-gpu/lib/python3.12/site-packages/torch/utils/data/dataset.py", line 420, in __getitems__
    return [self.dataset[self.indices[idx]] for idx in indices]
            ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "/gpfs/scratch/opera-dist-ml/users/jmauro/dist-s1-model/src/dataset.py", line 42, in __getitem__
    with np.load(npz_path, allow_pickle=False) as npz:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/scratch-jpl/opera-dist-ml/users/jmauro/envs/dist-s1-model-gpu/lib/python3.12/site-packages/numpy/lib/_npyio_impl.py", line 454, in load
    fid = stack.enter_context(open(os.fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
PermissionError: [Errno 13] Permission denied: '/scratch/opera-dist-ml/dist-s1-data-updated/dataset_samples_npz/T027-056024-IW3__2025-04-24__1/array_data.npz'
