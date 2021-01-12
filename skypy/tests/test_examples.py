from skypy.pipeline import load_skypy_yaml, Pipeline


def test_mccl_galaxies():
    config = load_skypy_yaml("../../examples/mccl_galaxies.yml")
    pipeline = Pipeline(config)
    pipeline.execute()

    assert len(pipeline.state) > 0
