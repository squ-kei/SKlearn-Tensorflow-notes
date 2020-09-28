An example using interleave to read multiple csv files
```python
def csv_reader_dataset(filepaths, repeat=1, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads) #Here preprocess is a standalone function
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)
```
**tf.data.Dataset by default does not read everything into memory. If the dataset is small enough to fit in memory, you can significantly speed up training by using the cache() method to cache everything to RAM.  
Generally do this after loading and preprocessing, but before shuffling,repeating, batching, and prefetching.**  
```python
dataset = dataset.cache() #cache to memory
dataset = dataset.cache("/path/to/file") #cache to file
```
