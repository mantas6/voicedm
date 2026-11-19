# NixOS VM test: the module on a server with its firewall on, fake Ollama and
# Chatterbox next to it, and a client machine that checks the API and makes one real
# WebRTC call (Silero VAD and Whisper run on the server, so the hardened unit is
# shown to run inference, not just to start).
{
  pkgs,
  module,
  callClientPython,
}:
let
  # A pinned faster-whisper model, so the sandboxed test needs no network (about 75 MB).
  whisperTinyEn =
    let
      rev = "0d3d19a32d3338f10357c0889762bd8d64bbdeba";
      file =
        name: hash:
        pkgs.fetchurl {
          url = "https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/${rev}/${name}";
          inherit hash;
        };
    in
    pkgs.linkFarm "faster-whisper-tiny.en" {
      "config.json" = file "config.json" "sha256-FLG0IakDSbxVG4gUYUJrVhqHQEnLnkxIZPLKOE9qfMU=";
      "model.bin" = file "model.bin" "sha256-Glr64GpNuRyXXJqdeL5cwRDuTqAirVfVVJLkVQ6Tayo=";
      "tokenizer.json" = file "tokenizer.json" "sha256-kpxSUkCUNtzhs4p10au8teEy0XDY4yTk4E7ZFfotIt8=";
      "vocabulary.txt" = file "vocabulary.txt" "sha256-/3dYh0bTolldMqtbaf/XuVziRBrFdTPLZvw+tXWhFc8=";
    };

  # The question the client speaks, synthesised at build time.
  question = pkgs.runCommand "question.wav" { nativeBuildInputs = [ pkgs.espeak-ng ]; } ''
    espeak-ng -s 140 -w $out "What is the capital of France?"
  '';

  stub = name: port: {
    description = "Fake ${name} for the VM test";
    wantedBy = [ "multi-user.target" ];
    before = [ "voicedm.service" ];
    serviceConfig.ExecStart = "${pkgs.python3}/bin/python3 ${./upstream_stub.py} ${name} ${toString port}";
  };
in
pkgs.testers.runNixOSTest {
  name = "voicedm";

  nodes.server = {
    imports = [ module ];
    services.voicedm = {
      enable = true;
      openFirewall = true;
      ollamaUrl = "http://127.0.0.1:11434";
      chatterboxUrl = "http://127.0.0.1:8004";
      whisperModelPath = whisperTinyEn;
      environment.VOICEDM_TTS_TIMEOUT = 30;
    };
    systemd.services.fake-ollama = stub "ollama" 11434;
    systemd.services.fake-chatterbox = stub "chatterbox" 8004;
    networking.firewall.enable = true;
    virtualisation.memorySize = 2048;
    virtualisation.cores = 2;
  };

  nodes.client = {
    environment.systemPackages = [ pkgs.curl ];
    virtualisation.memorySize = 1024;
  };

  testScript = ''
    import json

    def get(path):
        return json.loads(client.succeed(f"curl -sf http://server:8780{path}"))

    def send(method, path, body):
        return json.loads(client.succeed(
            f"curl -sf -X {method} -H 'content-type: application/json' "
            f"-d '{json.dumps(body)}' http://server:8780{path}"
        ))

    start_all()
    server.wait_for_unit("voicedm.service")
    server.wait_for_open_port(8780)
    client.wait_for_unit("multi-user.target")

    with subtest("health reports both upstreams"):
        health = get("/api/health")
        assert health == {
            "status": "ok", "upstreams": {"ollama": True, "chatterbox": True}, "call": None
        }, health

    with subtest("models and voices come through the proxy"):
        assert get("/api/models") == ["stub-model:latest"], get("/api/models")
        assert get("/api/voices") == ["stub-voice.wav"], get("/api/voices")

    with subtest("sessions are stored in the state directory"):
        created = send("POST", "/api/sessions", {})
        assert created["model"] == "stub-model:latest", created
        assert created["voice"] == "stub-voice.wav", created
        send("PATCH", f"/api/sessions/{created['id']}", {"title": "VM test"})
        assert get(f"/api/sessions/{created['id']}")["title"] == "VM test"
        server.succeed("test -s /var/lib/voicedm/voicedm.db")

    with subtest("the page is served"):
        page = client.succeed("curl -sf http://server:8780/")
        assert "<html" in page.lower(), page[:200]
        client.succeed("curl -sf http://server:8780/app.js > /dev/null")

    with subtest("a call is heard, transcribed and answered"):
        client.succeed(
            "${callClientPython} ${../scripts/call_client.py} --url http://server:8780 "
            f"--session {created['id']} --say ${question} --duration 25 >&2"
        )
        messages = get(f"/api/sessions/{created['id']}")["messages"]
        print(messages)
        user = [m["content"] for m in messages if m["role"] == "user"]
        bot = [m["content"] for m in messages if m["role"] == "assistant"]
        assert any("france" in text.lower() for text in user), messages
        assert bot == ["Paris is the capital of France."], messages
        server.succeed("journalctl -u voicedm | grep -q 'Turn timing (session'")

    with subtest("the service ran hardened, without network access to Hugging Face"):
        print(server.succeed("systemd-analyze security voicedm.service"))
        server.succeed("systemctl show voicedm -p NRestarts --value | grep -qx 0")
        server.fail("journalctl -u voicedm | grep -E 'Traceback|huggingface.co'")
        server.succeed("systemctl restart voicedm")
        server.wait_for_open_port(8780)
        assert get("/api/sessions")[0]["title"] == "VM test"
  '';
}
