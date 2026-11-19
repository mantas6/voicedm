{
  description = "voicedm: hands-free voice conversation with a local LLM";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
    }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      nixosModules = rec {
        voicedm = import ./nix/module.nix { inherit (self) packages; };
        default = voicedm;
      };

      packages = forAllSystems (pkgs: rec {
        voicedm = pkgs.callPackage ./nix/package.nix {
          inherit pyproject-nix uv2nix pyproject-build-systems;
        };
        default = voicedm;
      });

      checks = forAllSystems (
        pkgs:
        let
          inherit (pkgs) lib;
          package = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
          python = "${package.devVenv}/bin/python";
          source = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
              ./pyproject.toml
              ./src
              ./tests
              ./scripts
              ./nix/upstream_stub.py
            ];
          };
          # pytest and ruff from the locked dev environment, against a copy of the
          # source (pytest imports the installed voicedm, as the service does).
          inSource =
            name: script:
            pkgs.runCommand "voicedm-${name}" { } ''
              cp -r ${source} src && chmod -R u+w src && cd src
              export HOME=$TMPDIR
              # The sandbox points this at a missing file; the clients the tests
              # build (never connecting) then fail to load a CA bundle.
              unset SSL_CERT_FILE
              ${script}
              touch $out
            '';
        in
        {
          inherit package;
          pytest = inSource "pytest" "${python} -m pytest -p no:cacheprovider";
          ruff = inSource "ruff" ''
            ${python} -m ruff check --no-cache
            ${python} -m ruff format --check --no-cache
          '';
          vm-test = import ./nix/vm-test.nix {
            inherit pkgs;
            module = self.nixosModules.default;
            callClientPython = python;
          };
        }
      );

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
