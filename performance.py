"""Opt-in offline timing; no image, text, identifiers, paths or stdout logging."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter

_current=ContextVar('ocr_stage_metrics',default=None)


@contextmanager
def collect(sink):
    token=_current.set(sink)
    try:yield
    finally:_current.reset(token)


@contextmanager
def stage(name):
    sink=_current.get()
    if sink is None:
        yield
        return
    start=perf_counter();failed=False
    try:yield
    except BaseException:
        failed=True
        raise
    finally:
        metric=sink.setdefault(name,{'seconds':0.0,'calls':0,'failures':0})
        metric['seconds']+=perf_counter()-start
        metric['calls']+=1
        metric['failures']+=int(failed)


def profiled(function):
    @wraps(function)
    def run(*args,profile=None,**kwargs):
        with collect(profile),stage('total'):
            return function(*args,**kwargs)
    return run
