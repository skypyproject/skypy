from skypy.pipeline import load_skypy_yaml, Lightcone, Pipeline


def test_abundance_matching():
    config = load_skypy_yaml("../../examples/abundance_matching.yml")
    lightcone = Lightcone(config)
    lightcone.execute()

    assert len(lightcone.tables) > 0


def test_gravitational_wave_rates():
    config = load_skypy_yaml("../../examples/gravitational_wave_rates.yml")
    pipeline = Pipeline(config)
    pipeline.execute()

    assert len(pipeline.state) > 0


def test_mccl_galaxies():
    config = load_skypy_yaml("../../examples/mccl_galaxies.yml")
    pipeline = Pipeline(config)
    pipeline.execute()

    assert len(pipeline.state) > 0
