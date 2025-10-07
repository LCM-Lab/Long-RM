from typing import Type, TypeVar, Callable, List, Tuple, Iterable, Literal, Union
import os, random, itertools
from collections import defaultdict
    
import functools
import torch
import torch.distributed as dist

import os, shutil, dill, json


CALLABLE = TypeVar("CALLABLE")

def makedirs(func):
    @functools.wraps(func)
    def wrapper(obj, path, **kwargs):
        os.makedirs(os.path.dirname(path), exist_ok = True)
        return func(obj, path, **kwargs)
    return wrapper

def mkdir(path: str, exist_ok = True):
    os.makedirs(path, exist_ok = exist_ok)

def read_pkl(path: str) -> object:
    with open(path, "rb") as f:
        content = dill.load(f)
    return content

def read_json(path: str) -> Union[list, dict]:
    with open(path,"r") as f:
        content = json.load(f)
    return content

@makedirs
def save_pkl(obj: object, path: str):
    with open(path, "wb") as f:
        dill.dump(obj, f)

@makedirs
def save_json(obj: Union[list, dict], path: str, **kwargs):
    with open(path,"w") as f:
        json.dump(obj, f, **kwargs) 

def Identity(x): return x

def chunks(lst: list, chunk_num: int):
    """Yield successive n-sized chunks from lst."""
    chunk_width = len(lst) // chunk_num
    ones = chunk_num - len(lst) % chunk_num 
    p = 0
    for i in range(chunk_num):
        if i == ones: chunk_width += 1
        yield lst[p: p + chunk_width]
        p += chunk_width



def counters(obj: Iterable, key: Callable, 
             mapping: Callable = Identity):
    dic = defaultdict(int)
    for k in obj:
        dic[mapping(key(k))] += 1
        
    return dict(dic)


def bucketize(obj: Iterable, bucket_size: int, 
              key: Callable = Identity,
              drop_exceed: bool = False, 
              shuffle: bool = False,
              seed: int = 42):
    '''
    Args:
        obj: An Iterable Object
        bucket_size: The max size one bucket can hold
        key: A function that return the `size` property of an element
        drop_exceed: If one element already exceed the bucket_size, drop it
        shuffle: whether using shuffled or not(sorted)
        seed: if shuffle, control the seed
    '''
    def I(x): return x[0]
    def S(x): return x[1]
    def SI(x): return x[1], x[0]

    indexed = [(i, key(d)) for i, d in enumerate(obj) \
               if (not drop_exceed) or key(d) <= bucket_size]

    if not shuffle: indexed.sort(key = SI)
    else:
        random.seed(seed)
        random.shuffle(indexed)
    
    buckets, indexs = [], []
    tmp_bucket, tmp_index, tmp_size = [], [], 0 
    for x in indexed:
        if tmp_size + S(x) > bucket_size:
            buckets += [tmp_bucket]
            indexs += [tmp_index]

            tmp_bucket, tmp_index, tmp_size = [S(x)], [I(x)], S(x)
        else: 
            tmp_bucket += [S(x)]
            tmp_index += [I(x)]
            tmp_size += S(x)
    
    return buckets, indexs

def randint(a: int, b: int) -> int:
    return random.randint(a, b)

def random_sample(population, k: int) -> List[int]:
    return random.sample(population, k)

def shuffle(lst: list) -> list:
    random.shuffle(lst)
    return lst


def C(m: int, n: int) -> list:
    return list(itertools.combinations(range(m), n))


def cached(target: Type[CALLABLE],
           kwargs: dict = dict(),
           cache_path: str = None) -> CALLABLE:
    if not cache_path:
        cache_path = f"./{target.__name__}.pkl"
    if os.path.exists(cache_path):
        return read_pkl(cache_path)
    obj = target(**kwargs)
    save_pkl(obj, cache_path)
    return obj

def basename(path: str):
    return os.path.basename(path.rstrip("/"))



def path_join(*args):
    return os.path.join(*args)

def remove_path(path: str):
    if os.path.isfile(path):
        if os.path.exists(path):
            os.remove(path)
        return
    if os.path.exists(path):
        shutil.rmtree(path)

def read_path(dir: str, concat_root: bool = True) -> list[str]:
    assert os.path.isdir(dir), f"{dir} is not a directory!"
    list_path = sorted(os.listdir(dir))
    if concat_root:
        list_path = [os.path.join(dir, path) for path in list_path]
    return list_path


def copy_path(src: str, dst: str):
        assert os.path.exists(src)
        shutil.copy(src, dst)

def rank0only_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if dist.get_rank() == 0:
            return func(*args, **kwargs)
    return wrapper

@rank0only_decorator
def rank0print(*values: object, sep: str | None = " ", end: str | None = "\n", file = None, flush: Literal[False] = False) -> None:
    print(*values, sep, end, file, flush)

@rank0only_decorator
def rank0breakpoint():
    breakpoint()

def rankibreakpoint(rank: int):
    if dist.get_rank() == rank:
        breakpoint()





REDUCE_OP = dict(
    mean = dist.ReduceOp.AVG,
    max = dist.ReduceOp.MAX,
    sum = dist.ReduceOp.SUM
)

def all_reduce(data, op: Literal["mean", "max", "sum"] = "mean", group: dist.ProcessGroup = None):
    if isinstance(data, dict):
        ret = {}
        for k,v in data.items():
            ret[k] = all_reduce(v, op)

        return ret

    is_tensor = isinstance(data, torch.Tensor)
    if not is_tensor:
        data = torch.Tensor([data])
    is_cpu_tensor = data.device.type == "cpu"

    if is_cpu_tensor:
        data = data.to(torch.cuda.current_device())
    
    dist.all_reduce(data, op = REDUCE_OP[op], group = group)

    if is_cpu_tensor:
        data = data.cpu()
    
    return data if is_tensor else data.item()

