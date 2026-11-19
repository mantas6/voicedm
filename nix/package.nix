# voicedm built from uv.lock with uv2nix: every Python dependency is the exact wheel
# pinned in the lock file, installed into one virtualenv. Binary wheels are patched
# with autoPatchelfHook by pyproject.nix; the overrides below add the system
# libraries some of them expect to find on the host.
{
  lib,
  callPackage,
  python313,
  zlib,
  pyproject-nix,
  uv2nix,
  pyproject-build-systems,
}:
let
  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ../.; };

  overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

  overrides = final: prev: {
    # Only an sdist on PyPI, with a setup.py that does not declare setuptools.
    docopt = prev.docopt.overrideAttrs (old: {
      nativeBuildInputs =
        (old.nativeBuildInputs or [ ]) ++ final.resolveBuildSystem { setuptools = [ ]; };
    });

    voicedm = prev.voicedm.overrideAttrs (old: {
      meta = (old.meta or { }) // {
        description = "Self-hosted, hands-free voice conversation with a local LLM in the browser";
        license = lib.licenses.mit;
        mainProgram = "voicedm";
        platforms = lib.platforms.linux;
      };
    });
  };

  pythonSet = (callPackage pyproject-nix.build.packages { python = python313; }).overrideScope (
    lib.composeManyExtensions [
      pyproject-build-systems.overlays.wheel
      overlay
      overrides
    ]
  );

  # The runtime environment, and one with the dev group (pytest, ruff) for checks.
  venv = pythonSet.mkVirtualEnv "voicedm-env" workspace.deps.default;
  devVenv = pythonSet.mkVirtualEnv "voicedm-dev-env" workspace.deps.all;

  # Only the voicedm entry point from the virtualenv, not every dependency's scripts.
  app = (callPackage pyproject-nix.build.util { }).mkApplication {
    inherit venv;
    package = pythonSet.voicedm;
  };
in
app.overrideAttrs (old: {
  passthru = (old.passthru or { }) // {
    inherit pythonSet venv devVenv;
  };
})
