import os, random, time, logging
from copy import deepcopy
from typing import Any, List, Optional, Generator, Literal
from argparse import Namespace
from tqdm import tqdm
from multiprocessing.synchronize import Event
import multiprocessing as mp
from evaluating.distributed.environment import \
    (_Marks, Marks,
     Environment as mpenv)


class ProducerConsumerManager:
    def __init__(self,
                 task_info_list : Optional[list] = None,
                 shuffle: bool = True,
                 samples_number: int = None,
                 max_producers: int = 1,
                 max_consumers: int = 1,
                 produce_config: Namespace = Namespace(),
                 consume_config: Namespace = Namespace(),
                 produce_env_config: Optional[Namespace] = None,
                 consume_env_config: Optional[Namespace] = None,
                 start_method: Literal["fork", "spawn"] = "fork",
                 batch_size: int = 1,
                 max_qsize: int = 2000,
                 ):
        assert min(max_producers, max_consumers) > 0

        self.task_info_list = [None]* max_producers \
            if task_info_list is None else deepcopy(task_info_list)
        if shuffle:
            random.shuffle(self.task_info_list)
        self._samples_number_per_producer = 1 if samples_number is None else samples_number // max_producers
        self._estimate_avg_tasks = max(1, len(self.task_info_list) // max_consumers) \
            if samples_number is None else samples_number // max_consumers
        self.max_producers = max_producers
        self.max_consumers = max_consumers
        self.produce_config = produce_config
        self.consume_config = consume_config
        self.produce_env_config = produce_env_config
        self.consume_env_config = consume_env_config

        self.producers = []
        self.consumers = []

        mpenv.set_start_method(start_method, force = True)

        self.manager = mp.Manager()
        self._send_queue = self.manager.Queue() # main -> producer
        self._recv_queue = self.manager.Queue() # main <- producer
        self._task_queue = mp.Queue()    # main -> producer -> consumer 
        self._result_list = self.manager.list()
        self.producers_end_event = [mp.Event() for _ in range(max_producers)]

        self._producers_exception_list = self.manager.list()
        self._consumers_exception_list = self.manager.list()

        self._batch_size = batch_size
        self._max_qsize = max_qsize

    def start(self):
        return self.__enter__()
    
    def terminate(self):
        return self.__exit__()
    
    @property
    def result_list(self):
        if not self.is_finished():
            for _ in self.ckpt_result_lists(realtime = False, do_yield = False):...
        self.join()
        return list(self._result_list)

    def join(self):
        for e in self.producers_end_event:
            e.set()
        if self.is_finished():return
        for process in self.producers + self.consumers:
            if process.is_alive():
                process.join()

    def is_finished(self):
        return hasattr(self,"_is_finished")
        
    def ckpt_result_lists(self, 
                          realtime:bool = True, 
                          ignore_empty:bool = True, 
                          do_yield:bool = True, 
                          save_ckpt_only: bool = False) -> Generator:
        if do_yield and hasattr(self,"_is_calling_ckpt_lists") and self._is_calling_ckpt_lists:
            raise RuntimeError("Nesting calling `ckpt_result_lists` is not allowed!")
        else:
            self._is_calling_ckpt_lists = 1

        if save_ckpt_only: 
            realtime = do_yield = True

        if self.is_finished():
            for i, (l, r) in enumerate(zip(self._ckpt_ids[:-1], self._ckpt_ids[1:])):
                if ignore_empty and l==r:
                    continue
                yield list(self._result_list[l:r]), self._comments[i]
            self._is_calling_ckpt_lists = 0
            delattr(self, "_is_calling_ckpt_lists")
            return

        self._ckpt_ids = [0]
        self._comments = []
        while True:
            signal = self._recv_queue.get()
            
            assert signal in (Marks.ckpt, _Marks.end) or \
            (isinstance(signal, tuple) and len(signal) == 2 and \
             signal[0] is Marks.ckpt)
            comment = signal[1] if isinstance(signal, tuple) else None
            
            if save_ckpt_only:
                temp = list(self._result_list)
                if len(temp) or (not ignore_empty):
                    yield temp, comment
                del self._result_list[:]

            elif realtime:
                temp = list(self._result_list[self._ckpt_ids[-1]:])
                if do_yield and (len(temp) or (not ignore_empty)): 
                    yield temp, comment

            if not save_ckpt_only:
                self._ckpt_ids.append(len(self._result_list))
                self._comments.append(comment)

            if signal is _Marks.end:
                for _ in range(self.max_consumers - 1):
                    assert self._recv_queue.get() is _Marks.end
                self._is_finished = True
                break            
            
            for _ in range(self.max_producers):
                self._send_queue.put(_Marks.recover)

        if (not realtime) and do_yield:
            for l, r in zip(self._ckpt_ids[:-1], self._ckpt_ids[1:]):
                if ignore_empty and l==r:
                    continue
                yield list(self._result_list[l:r])

        if not do_yield: yield


        self._is_calling_ckpt_lists = 0
        delattr(self, "_is_calling_ckpt_lists")
            
        

    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.join()
        else:
            print(f"{exc_type.__name__} in manager - {exc_value}")
        for process in self.producers + self.consumers:
            if len(self._producers_exception_list) + len(self._consumers_exception_list):
                process.terminate()
            else:
                process.join()
                process.terminate()
        self.manager.shutdown()
        return False
    def __enter__(self):
        if self.task_info_list == []:
            logging.warning("[Ignore this warning if you choose not to process any data]\n"
                            "You didn't pass any task information to argument `task_info_list` and"
                            " no data will be processed,\nif you want to run the `produce` process "
                            "without using this argument, \ndo set: `task_info_list = None` rather "
                            "than 'task_info_list = []'.\nSee the comments in `__init__` of the class "
                            "`ProducerConsumerMananger` for more details.")
            self.result_list = []
            return self
        with mpenv.add_process_group(world_size = self.max_consumers):
            for i in range(self.max_consumers):
                self.set_consumer_environment(i, self.consume_env_config)
                process=mpenv.Process(
                    target=self._consumer,
                    args=(self.set_consumer_global_variables, self.consume,
                        self._estimate_avg_tasks, self.max_producers,
                        self._producers_exception_list, self._consumers_exception_list,
                        self._send_queue, self._recv_queue,
                        self._task_queue,self.consume_config, self._result_list,
                        self._batch_size)
                )
                self.consumers.append(process)
                process.start()
        
        with mpenv.add_process_group(world_size = self.max_producers):
            for i, task_info_list in enumerate(self.chunks(self.task_info_list, self.max_producers)):
                self.set_producer_environment(i, self.produce_env_config)
                process=mpenv.Process(
                    target=self._producer,
                    args=(self.set_producer_global_variables, self.produce,
                        self._samples_number_per_producer, self.max_consumers, 
                        self._producers_exception_list, self._consumers_exception_list,
                        self._send_queue, self._recv_queue,
                        task_info_list, self._task_queue, self.produce_config,
                        self._max_qsize, self._batch_size,
                        self.producers_end_event[i])
                )
                
                self.producers.append(process)
                process.start()

        
        assert len(self.producers) == self.max_producers, "Chunk Error!!"

        return self
    
    @classmethod
    def chunks(cls, lst, chunk_num):
        """Yield successive n-sized chunks from lst."""
        chunk_width = len(lst) // chunk_num
        ones = chunk_num - len(lst) % chunk_num 
        p = 0
        for i in range(chunk_num):
            if i == ones: chunk_width += 1
            yield lst[p: p + chunk_width]
            p += chunk_width

    @classmethod
    def set_producer_environment(cls, producer_id: int, produce_env_config: Namespace) -> None:
        if produce_env_config is not None:raise NotImplementedError
    @classmethod
    def set_consumer_environment(cls, consumer_id: int, consume_env_config: Namespace) -> None:
        if consume_env_config is not None:raise NotImplementedError

    @classmethod
    @mpenv.need_rank
    def _producer(cls, set_producer_global_variables, produce, samples_number_per_producer: int, 
                  max_consumers: int, producers_exception_list: List[int], consumers_exception_list: List[int],
                  recv_queue: mp.Queue, send_queue: mp.Queue,
                  task_info_list: list, task_queue: mp.Queue, produce_config: Namespace, 
                  max_qsize: int, batch_size: int,
                  end_event: Event):

        p_name = f"Producer `{getattr(produce,'_global_attr','produce')}`"

        try:
            global_variables = set_producer_global_variables(produce_config)
            for task_id, task_info in tqdm(enumerate(task_info_list), desc=f"{p_name}: {mpenv.get_rank()} ({os.getpid()})"):
                progress_bar = tqdm(desc=f"{p_name}: {mpenv.get_rank()} ({os.getpid()}) on ({task_id + 1}/{len(task_info_list)})",
                                    total = samples_number_per_producer // len(task_info_list))  
                delta = 10
                number_tasks = 0
                for task_sample in produce(task_info, produce_config, global_variables):
                    if len(consumers_exception_list) == max_consumers:
                        raise RuntimeError(f"All Consumers:{consumers_exception_list}, Exit Unexpectedly!")
                    if task_queue.qsize() > max_qsize:
                        while task_queue.qsize() > (max_qsize >> 1):
                            time.sleep(1)

                    if task_sample is Marks.ckpt or \
                        (isinstance(task_sample, tuple) and len(task_sample) == 2 and task_sample[0] is Marks.ckpt): # waiting until all consumers finished
                        mpenv.barrier()
                        if batch_size > 1:
                            for _ in range(mpenv.get_rank(), max_consumers, mpenv.get_world_size()):
                                task_queue.put(_Marks.sep)

                        for _ in range(mpenv.get_rank(), max_consumers, mpenv.get_world_size()):
                            task_queue.put(Marks.ckpt)
                        mpenv.barrier()

                        while task_queue.qsize(): ...
                        mpenv.barrier()

                        while send_queue.qsize(): ...
                        mpenv.barrier()
                        
                        if mpenv.get_rank() == 0:
                            send_queue.put(task_sample)
                        mpenv.barrier()

                        assert recv_queue.get() is _Marks.recover
                        mpenv.barrier()

                        for _ in range(mpenv.get_rank(), max_consumers, mpenv.get_world_size()):
                            task_queue.put(_Marks.recover)
                        mpenv.barrier()

                        continue

                    assert not hasattr(task_sample, "__contains__") or (Marks.ckpt not in task_sample), \
                    "Use ckpt like: 'yield Marks.ckpt' or 'yield Marks.ckpt, obj' with only ONE object!"
                    
                    task_queue.put(task_sample)
                    
                    number_tasks += 1
                    if number_tasks > progress_bar.total:
                        progress_bar.total += delta
                    progress_bar.update(1)
                   
                progress_bar.total = number_tasks
                progress_bar.close()

            if batch_size > 1:
                for _ in range(mpenv.get_rank(), max_consumers, mpenv.get_world_size()):
                    task_queue.put(_Marks.sep)
                    
            for _ in range(mpenv.get_rank(), max_consumers, mpenv.get_world_size()):
                task_queue.put(_Marks.end)

            end_event.wait()    
            
        except Exception as e:
            producers_exception_list.append(os.getpid())
            raise e
    @classmethod
    @mpenv.need_rank
    def _consumer(cls, set_consumer_global_variables, consume, _estimate_avg_tasks: int, max_producers: int,
                  producers_exception_list: List[int], consumers_exception_list: List[int],
                  recv_queue: mp.Queue, send_queue: mp.Queue,
                  task_queue: mp.Queue, consume_config: Namespace, result_list: List,
                  batch_size: int):
        

        c_name = f"Consumer `{getattr(consume,'_global_attr','consume')}`"

        progress_bar = tqdm(desc=f"{c_name}: {mpenv.get_rank()} ({os.getpid()})", total=_estimate_avg_tasks)  
        delta = 10
        number_tasks = 0
        
        try:
            global_variables = set_consumer_global_variables(consume_config)

            if batch_size <= 0: batch_size = float("inf")
            while True:
                if len(producers_exception_list) == max_producers:
                    raise RuntimeError(f"All Producers:{producers_exception_list}, Exit Unexpectedly!")
                if batch_size == 1:
                    task_sample = task_queue.get()
                else:
                    task_sample = None
                    task_samples = []
                    while len(task_samples) < batch_size:
                        task_sample = task_queue.get()
                        if task_sample in (_Marks.end, Marks.ckpt, _Marks.sep):
                            break
                        task_samples.append(task_sample)
                    
                    if len(task_samples):
                        for task_result in consume(task_samples, consume_config, global_variables):
                            result_list.append(task_result)
                            
                        number_tasks += len(task_samples)
                        if number_tasks > progress_bar.total:
                            progress_bar.total += delta
                        progress_bar.update(len(task_samples))

                    if task_sample == _Marks.sep: mpenv.barrier()
                    if task_sample not in (_Marks.end, Marks.ckpt): continue


                if task_sample is _Marks.end:break
                if task_sample is Marks.ckpt:
                    mpenv.barrier()
                    assert task_queue.get() is _Marks.recover
                    
                    mpenv.barrier()
                    
                    continue

                if batch_size != 1: continue

                for task_result in consume(task_sample, consume_config, global_variables):
                    result_list.append(task_result)

                    number_tasks += 1
                    if number_tasks > progress_bar.total:
                        progress_bar.total += delta
                    progress_bar.update(1)
            mpenv.barrier() # essential

        except Exception as e:
            consumers_exception_list.append(os.getpid())
            raise e

        finally:
            mpenv.barrier()
            send_queue.put(_Marks.end)

            progress_bar.total = number_tasks
            progress_bar.close()

    @classmethod
    def set_producer_global_variables(cls, produce_config: Namespace) -> Namespace:
        return Namespace()
    @classmethod
    def set_consumer_global_variables(cls, consume_config: Namespace) -> Namespace:
        return Namespace()

    @classmethod
    def produce(cls, task_info , produce_config: Namespace, glb: Namespace) -> Generator:
        raise NotImplementedError

    @classmethod
    def consume(cls, task_sample, consume_config: Namespace, glb: Namespace) -> Generator:
        raise NotImplementedError
