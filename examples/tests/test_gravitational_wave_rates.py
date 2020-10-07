from skypy.pipeline import load_skypy_yaml, Pipeline


def test_gravitational_wave_rates():
    config = load_skypy_yaml("../gravitational_wave_rates.yml")
    pipeline = Pipeline(config)
    pipeline.execute()

    assert len(pipeline.state) > 0
