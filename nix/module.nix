# NixOS module: services.voicedm. The flake passes its packages in, so `package`
# defaults to this flake's build for the host system.
{ packages }:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  inherit (lib) mkOption types;
  cfg = config.services.voicedm;

  # Linux's default net.ipv4.ip_local_port_range, where aiortc's media sockets land.
  ephemeralUdp = {
    from = 32768;
    to = 60999;
  };

  environment = {
    VOICEDM_HOST = cfg.host;
    VOICEDM_PORT = toString cfg.port;
    VOICEDM_DATA_DIR = "/var/lib/voicedm";
    VOICEDM_OLLAMA_URL = cfg.ollamaUrl;
    VOICEDM_CHATTERBOX_URL = cfg.chatterboxUrl;
    VOICEDM_WHISPER_MODEL =
      if cfg.whisperModelPath != null then "${cfg.whisperModelPath}" else cfg.whisperModel;
    # DynamicUser has no writable home: anything the libraries cache goes to the
    # cache directory (the Whisper model itself lives in VOICEDM_DATA_DIR/models).
    HOME = "/var/lib/voicedm";
    XDG_CACHE_HOME = "/var/cache/voicedm";
    HF_HOME = "/var/cache/voicedm/huggingface";
    HF_HUB_DISABLE_TELEMETRY = "1";
  }
  // lib.optionalAttrs (cfg.stunUrl != null) { VOICEDM_STUN_URL = cfg.stunUrl; }
  // lib.optionalAttrs (cfg.whisperModelPath != null) { HF_HUB_OFFLINE = "1"; }
  // lib.mapAttrs (_: toString) cfg.environment;
in
{
  options.services.voicedm = {
    enable = lib.mkEnableOption "voicedm, a hands-free voice conversation with a local LLM in the browser";

    package = mkOption {
      type = types.package;
      default = packages.${pkgs.stdenv.hostPlatform.system}.default;
      defaultText = lib.literalMD "the voicedm package of this flake for the host system";
      description = "The voicedm package to run.";
    };

    host = mkOption {
      type = types.str;
      default = "0.0.0.0";
      example = "127.0.0.1";
      description = ''
        Address the HTTP server listens on. WebRTC media does not go through it: the
        browser sends audio straight to the host over UDP (see `openFirewall`).
      '';
    };

    port = mkOption {
      type = types.port;
      default = 8780;
      description = "TCP port of the page, the API and the WebRTC offer/answer.";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = false;
      description = ''
        Open the HTTP port (TCP) and the UDP ports for call audio in the firewall.

        aiortc, the WebRTC stack, binds each call's media socket to an ephemeral UDP
        port picked by the kernel and has no setting to pin a range. So this opens the
        whole Linux default ephemeral range, UDP ${toString ephemeralUdp.from}-${toString ephemeralUdp.to}
        (adjust by hand if `net.ipv4.ip_local_port_range` is changed). To open it only
        towards the local network, leave this off and open the same ports on the LAN
        interface (`networking.firewall.interfaces.<name>.allowedUDPPortRanges`).
      '';
    };

    ollamaUrl = mkOption {
      type = types.str;
      default = "http://localhost:11434";
      example = "http://ollama.example.lan:11434";
      description = "Base URL of the Ollama server.";
    };

    chatterboxUrl = mkOption {
      type = types.str;
      default = "http://localhost:8004";
      example = "http://tts.example.lan:8004";
      description = "Base URL of the Chatterbox TTS server.";
    };

    whisperModel = mkOption {
      type = types.str;
      default = "base.en";
      example = "small.en";
      description = ''
        faster-whisper model name. The service downloads it from Hugging Face into
        its state directory on first start, and asks Hugging Face for the current
        revision on every start (the cached copy is used when that fails). Use
        `whisperModelPath` to run without Hugging Face.
      '';
    };

    whisperModelPath = mkOption {
      type = types.nullOr types.path;
      default = null;
      example = lib.literalExpression ''"/srv/models/faster-whisper-base.en"'';
      description = ''
        A directory holding a CTranslate2 Whisper model (`model.bin`, `config.json`,
        `tokenizer.json`, `vocabulary.txt`, as in the Systran/faster-whisper-*
        repositories). When set, it replaces `whisperModel`, and the service never
        contacts Hugging Face (`HF_HUB_OFFLINE=1`), e.g. for offline hosts.
      '';
    };

    stunUrl = mkOption {
      type = types.nullOr types.str;
      default = null;
      example = "stun:stun.example.org:3478";
      description = ''
        Optional STUN server. Not needed when the browser and the host are on the same
        network.
      '';
    };

    environment = mkOption {
      type = types.attrsOf (
        types.oneOf [
          types.str
          types.int
          types.float
        ]
      );
      default = { };
      example = {
        VOICEDM_DEFAULT_MODEL = "example-model:latest";
        VOICEDM_MEMORY_BUDGET = 8000;
        VOICEDM_LOG_LEVEL = "DEBUG";
      };
      description = ''
        Extra environment variables for the service, for the `VOICEDM_*` settings
        without an option of their own: `VOICEDM_DEFAULT_MODEL`,
        `VOICEDM_DEFAULT_VOICE`, `VOICEDM_TTS_TIMEOUT`, `VOICEDM_MEMORY_BUDGET`,
        `VOICEDM_SUMMARY_PROMPT`, `VOICEDM_SUMMARY_TIMEOUT`, `VOICEDM_LOG_LEVEL`.
        They override the values set from the other options. The values end up in the
        Nix store, so do not put secrets here.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    networking.firewall = lib.mkIf cfg.openFirewall {
      allowedTCPPorts = [ cfg.port ];
      allowedUDPPortRanges = [ ephemeralUdp ];
    };

    systemd.services.voicedm = {
      description = "voicedm voice conversation server";
      wantedBy = [ "multi-user.target" ];
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];
      inherit environment;

      serviceConfig = {
        ExecStart = lib.getExe cfg.package;
        Restart = "on-failure";
        RestartSec = 5;
        # Loading the Whisper model (and downloading it on first start) takes a while.
        TimeoutStartSec = "10min";
        # The server ends a running call on SIGTERM and gives it a few seconds.
        TimeoutStopSec = 30;

        DynamicUser = true;
        StateDirectory = "voicedm";
        StateDirectoryMode = "0700";
        CacheDirectory = "voicedm";
        CacheDirectoryMode = "0700";
        WorkingDirectory = "/var/lib/voicedm";
        UMask = "0077";

        # Hardening. Checked with a real call (Silero on onnxruntime, Whisper on
        # CTranslate2, aiortc media) in the NixOS VM test.
        CapabilityBoundingSet = "";
        AmbientCapabilities = "";
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        PrivateDevices = true;
        PrivateUsers = true;
        DevicePolicy = "closed";
        ProtectClock = true;
        ProtectHostname = true;
        ProtectKernelLogs = true;
        ProtectKernelModules = true;
        ProtectKernelTunables = true;
        ProtectControlGroups = true;
        ProtectProc = "invisible";
        # No ProcSubset = "pid": onnxruntime reads /proc/cpuinfo to pick CPU kernels.
        # AF_NETLINK: aiortc lists the host's addresses (getifaddrs) for ICE candidates.
        RestrictAddressFamilies = [
          "AF_INET"
          "AF_INET6"
          "AF_UNIX"
          "AF_NETLINK"
        ];
        RestrictNamespaces = true;
        RestrictRealtime = true;
        RestrictSUIDSGID = true;
        RemoveIPC = true;
        LockPersonality = true;
        # No MemoryDenyWriteExecute: pyOpenSSL's verify callback for the WebRTC DTLS
        # handshake is a cffi callback, which needs write+execute memory.
        SystemCallArchitectures = "native";
        SystemCallFilter = [
          "@system-service"
          "~@privileged"
          "~@resources"
        ];
        SystemCallErrorNumber = "EPERM";
      };
    };
  };
}
