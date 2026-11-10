import backend.main as entrypoint


def test_explicit_argv_does_not_respawn_for_utf8(monkeypatch):
    monkeypatch.setattr(entrypoint.os, "name", "nt")
    monkeypatch.setattr(
        entrypoint,
        "build_parser",
        lambda: type("Parser", (), {"parse_args": lambda self, argv: None})(),
    )
    monkeypatch.setattr(entrypoint.asyncio, "run", lambda coroutine: 0)
    monkeypatch.setattr(
        entrypoint,
        "_async_main",
        lambda args: type("Awaitable", (), {"__await__": lambda self: iter(())})(),
    )
    monkeypatch.setattr(entrypoint.subprocess, "call", lambda *args: 99)

    assert entrypoint.main(["--query", "test"]) == 0


def test_windows_utf8_restart_command_is_synchronous(monkeypatch):
    captured = {}
    monkeypatch.setattr(entrypoint.sys, "executable", "python.exe")
    monkeypatch.setattr(
        entrypoint.sys, "argv", ["research-assistant", "--query", "题目"]
    )

    def fake_call(argv):
        captured["argv"] = argv
        return 7

    monkeypatch.setattr(entrypoint.subprocess, "call", fake_call)

    result = entrypoint._restart_in_utf8_mode()

    assert result == 7
    assert captured["argv"] == [
        "python.exe",
        "-X",
        "utf8",
        "-m",
        "backend.main",
        "--query",
        "题目",
    ]
