from skypy.Test_subpkg import funcs as f
class TestReturnString:
    def test_hello_world(self):
        x = 'Hello World!'
        assert f.returnString() == x

    def test_ohoh(self):
        x = 'oh oh'
        assert f.returnString('oh oh') == x