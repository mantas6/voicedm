{
  description = "voicedm: hands-free voice conversation with a local LLM";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      # packages, nixosModules and checks are added in the Nix milestone
      # (nix/package.nix, nix/module.nix, a NixOS VM test).

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            pkgs.python313
            pkgs.uv
            pkgs.ffmpeg
            pkgs.ruff
          ];
          env = {
            # Use the shell's Python instead of letting uv download one.
            UV_PYTHON = "${pkgs.python313}/bin/python3";
            UV_PYTHON_DOWNLOADS = "never";
            # Binary wheels (av, onnxruntime, ctranslate2) link against these.
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ];
          };
        };
      });

      formatter = forAllSystems (pkgs: pkgs.nixfmt);
    };
}
