from typing import List, Iterable, Set, Union
import inspect

class TransformationTracker:
    
    def __init__(self,
                 transformation_tracker:List[str] = None):
        super().__init__()
        if transformation_tracker is None:
            self._transformation_tracker = list()
        else:
            self._transformation_tracker = transformation_tracker

    def get_transformation_list(self):
        return self._transformation_tracker

class _FunctionIterable:
    # wraper class
    def __init__(self,
                 function_generator) -> None:
        self.function_generator = function_generator
        
    def __iter__(self):
        return iter(self.function_generator())
    
    def is_class(self):
        return False
    
    def __call__(self):
        return self.__iter__()
    
class _ClassIterable:
    
    def __init__(self,
                 class_iterable) -> None:
        self.class_iterable = class_iterable
        
    def is_class(self):
        return True
        
    def __iter__(self):
        return iter(self.class_iterable)
    
    def __call__(self):
        return self.__iter__()


class GenericDataLoader(TransformationTracker):
    
    def __init__(self,
                 python_iterable: Iterable,
                 python_iterable_n_samples:int = None,
                 transformation_tracker:List[str] = None) -> None:
        super().__init__(transformation_tracker)
        if inspect.isgeneratorfunction(python_iterable):
            self.python_iterable = _FunctionIterable(python_iterable)
        else:
            self.python_iterable = _ClassIterable(python_iterable)
            
        self.python_iterable_n_samples = python_iterable_n_samples
        
    def _set_input_dataset(self, input_dataset):
        """
        Set the variable input_dataset, which corresponds to the current
        DataLoader instance
        """
        self.input_dataset=input_dataset

    def _update_samples_frozen_pipeline(self):
        self.n_samples_frozen_pipeline = self._transformation_tracker[:]
    
    def _is_n_samples_correctly_cached(self):
        
        if hasattr(self, "n_samples_frozen_pipeline"):
            # check if any transformation was added since last n_samples check point
            if len(self.n_samples_frozen_pipeline) == len(self._transformation_tracker):
                return True
            else:
                # check if a n_sample recomputation is needed
                if any(filter(lambda x: "batch" in x, self._transformation_tracker[len(self.n_samples_frozen_pipeline):])):
                    return False
                else:
                    # update the frozen pipeline
                    self._update_samples_frozen_pipeline()
                    return True
        else:
            return False 
    
    def get_n_samples(self):
        """
        Returns the number of samples that the DataLoader contains,
        if the DataLoader does not know how many samples it has, then
        the DataLoader will first count the samples and then cache it 
        and return the counter.
        """
        if self._is_n_samples_correctly_cached():
            return self.n_samples
        else:
            # add a logger
            #self.logger.info("this dataset does not have the number of samples in cache so it will take some time to counting")
            n_samples = 0
            for _ in self.input_dataset:
                n_samples += 1
                
            self.n_samples = n_samples
            self._update_samples_frozen_pipeline()
            return self.n_samples
    
    def get_python_iterator_n_samples(self):
        if self.python_iterable_n_samples is not None:
            return self.python_iterable_n_samples
        else:
            # log this operation
            n_samples = 0
            # count
            for _ in self.python_iterable:
                n_samples += 1
        self.python_iterable_n_samples = n_samples
        return n_samples
    
class _DataLoaderIterator:
    def __init__(self, 
                 input_dataset, 
                 transform_f) -> None:
        super().__init__()
        self.input_dataset=input_dataset
        self.transform_f=transform_f
        self.iterator = iter(self._next_gen())
        
    def _next_gen(self):
        for data in self.input_dataset:
            yield self.transform_f(data)
        
    def __next__(self):
        return next(self.iterator)

class LambdaDataLoader(GenericDataLoader):
    
    def __init__(self,
                 input_dataset: Iterable,
                 python_iterable: Iterable,
                 skip_keys: Union[Set,List]=[],
                 transform_f=None,
                 python_iterable_n_samples:int=None,
                 transformation_tracker:List[str] = None) -> None:
        super().__init__(python_iterable=python_iterable,
                         python_iterable_n_samples=python_iterable_n_samples,
                         transformation_tracker=transformation_tracker)
        self.input_dataset = input_dataset
        self.skip_keys=set(skip_keys) if isinstance(skip_keys, list) else skip_keys
        if transform_f is None:
            self.transform_f = lambda x:x
        else:
            self.transform_f = transform_f 

    def _transform(self, data):
        return self.transform_f(data)
    
    def _iter_transform(self, data):
        """
        Runs before the _transform method, used to ensure some
        constrains before and after the transformation
        """
        data_to_skip_transform = {k:data.pop(k) for k in self.skip_keys}
        return self._transform(data) | data_to_skip_transform
    
    def __iter__(self):
        return _DataLoaderIterator(input_dataset=self.input_dataset,
                                   transform_f=self._iter_transform)