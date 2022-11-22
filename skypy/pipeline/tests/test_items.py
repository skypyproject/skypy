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
    assert len(deps) == 0

    # check `evaluate` method
    val = item.evaluate(pipeline)
    assert val is None


def test_call():
    from skypy.pipeline import Pipeline
    from skypy.pipeline._items import Call, Ref

    # set up a mock pipeline
    pipeline = Pipeline({})

    # function we will call
    def tester(arg1, arg2, *, kwarg1, kwarg2):
        return arg1, arg2, kwarg1, kwarg2

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

    # good construction with arg1 and kwarg1
    call = Call(tester, [1], {'kwarg1': 3})

    # call still has incomplete args
    with pytest.raises(TypeError, match=r'tester\(\)'):
        call.evaluate(pipeline)

    # infer required arg2 and kwarg2 from context
    context = {
        'arg2': 2,
        'kwarg2': 4,
    }
    call.infer(context)

    # call should be evaluatable now
    result = call.evaluate(pipeline)
    assert result == (1, 2, 3, 4)

    # set up a call with references
    call = Call(tester, [Ref('var1'), 2], {'kwarg1': Ref('var3'), 'kwarg2': 4})

    # set up a pipeline with variables and a call that references them
    pipeline = Pipeline({'var1': 1, 'var3': 3})

    # check dependencies are resolved
    deps = call.depend(pipeline)
    assert deps == ['var1', 'var3']

    # execute the pipeline (sets state) and evaluate the call
    pipeline.execute()
    result = call.evaluate(pipeline)
    assert result == (1, 2, 3, 4)
