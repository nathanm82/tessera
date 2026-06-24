import pytest

from tessera.exceptions import RegistrationError
from tessera.registry import Registry


def test_register_and_create() -> None:
    reg: Registry[dict] = Registry("widget")
    reg.register("box", lambda **kw: {"kind": "box", **kw})
    obj = reg.create("box", size=3)
    assert obj == {"kind": "box", "size": 3}


def test_lookup_is_case_insensitive() -> None:
    reg: Registry[str] = Registry("widget")
    reg.register("Foo", lambda: "ok")
    assert "foo" in reg
    assert reg.create("FOO") == "ok"


def test_duplicate_registration_raises() -> None:
    reg: Registry[int] = Registry("widget")
    reg.register("a", lambda: 1)
    with pytest.raises(RegistrationError):
        reg.register("a", lambda: 2)


def test_unknown_name_lists_known() -> None:
    reg: Registry[int] = Registry("widget")
    reg.register("alpha", lambda: 1)
    with pytest.raises(RegistrationError, match="alpha"):
        reg.create("missing")


def test_names_are_sorted() -> None:
    reg: Registry[int] = Registry("widget")
    reg.register("b", lambda: 1)
    reg.register("a", lambda: 2)
    assert reg.names() == ["a", "b"]
