import pkg_resources
libs_installed = {pkg.key for pkg in pkg_resources.working_set}

def check_if_lib_is_installed(lib_name):
    return lib_name in libs_installed

def prefetch(queue, input_dataset, transform_f):
    
    iterator = iter(input_dataset)
    
    # enqueue until buffer is full
    try:
        for _ in range(queue.maxsize):
            queue.put_nowait(transform_f(next(iterator)))
    except StopIteration:
        # in this situation, the buffer size was larger than the data
        # in the generator, so we just dont need to do nothing else, the
        # all of the data is inside of the queue
        pass
    
    try:
        while True:
            yield queue.get() # block
            queue.put_nowait(transform_f(next(iterator)))
    except StopIteration:
        # no more items, so lets empty the queue
        while queue.qsize()>0:
            yield queue.get()
    
    # end of the iterator
    #raise StopIteration