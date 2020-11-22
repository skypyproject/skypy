import pytest


def test_item():
    from skypy.pipeline import Pipeline
    from skypy.pipeline._items import Item

    # set up a mock pipeline
    pipeline = Pipeline({})

    # construct the base class
    item = Item()

    # check `infer` method
    item.infer({})

    # check `depend` method
    deps = item.depend(pipeline)
    assert isinstance(deps, list)

    # check `evaluate` method
    val = item.evaluate(pipeline)
    assert val is None


def test_call():
    from skypy.pipeline import Pipeline
    from skypy.pipeline._items import Call

    # set up a mock pipeline
    pipeline = Pipeline({})

    # function we will call
    def tester(arg1, arg2, *, kwarg1, kwarg2):
        return True

    # invalid construction
    with pytest.raises(TypeError, match='function is not callable'):
        Call(None, [], {})
    with pytest.raises(TypeError, match='args is not a sequence'):
        Call(tester, None, {})
    with pytest.raises(TypeError, match='kwargs is not a mapping'):
        Call(tester, [], None)

    # good construction with no args or kwargs
    call = Call(tester, [], {})

    # call has incomplete args
    with pytest.raises(TypeError, match=r'tester\(\)'):
        call.evaluate(pipeline)

    # infer required args and kwargs
    context = {
        'arg1': 1,
        'arg2': 2,
        'kwarg1': 3,
        'kwarg2': 4,
    }
    call.infer(context)

    # call should be evaluatable now
    result = call.evaluate(pipeline)
    assert result == True
