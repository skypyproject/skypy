from skypy.pipeline import load_skypy_yaml, Lightcone


def test_abundance_matching():
    config = load_skypy_yaml("../abundance_matching.yml")
    lightcone = Lightcone(config)
    lightcone.execute()

    assert len(lightcone.tables) > 0
