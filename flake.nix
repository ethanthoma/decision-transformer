{
  description = "Application packaged using poetry2nix";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixgl.url = "github:nix-community/nixGL";
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix, nixgl }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              allowUnfreePredicate = _: true;
              cudaSupport = true;
            };
            overlays = [ nixgl.overlay ];
          };
          inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication defaultPoetryOverrides;

          python = pkgs.python311;
          pythonVersionNoDot = builtins.replaceStrings [ "." ] [ "" ] python.pythonVersion;
          pythonPackages = builtins.import
            (
              builtins.toFile "pversion" "pkgs: pkgs.python${pythonVersionNoDot}Packages"
            )
            pkgs;

          roms = import ./autorom {
            inherit (pkgs) lib fetchFromGitHub fetchurl writeText;
            inherit (pythonPackages) buildPythonPackage click;
          };

          SDL2 = (import
            (builtins.fetchGit {
              url = "https://github.com/NixOS/nixpkgs/";
              ref = "refs/heads/nixpkgs-unstable";
              rev = "6e3a86f2f73a466656a401302d3ece26fba401d9";
            })
            { inherit system; }).SDL2;
        in
        {
          packages = {
            default = mkPoetryApplication {
              projectDir = self;

              inherit python;

              buildInputs = [
                pkgs.makeWrapper
                SDL2
                pkgs.llvmPackages.libcxx
              ];

              nativeBuildInputs = [
                pkgs.poetry
                pkgs.llvmPackages.clang
                pkgs.pkg-config
              ];

              preferWheels = false;

              overrides = defaultPoetryOverrides.extend
                (final: prev: {
                  tinygrad = prev.tinygrad.overridePythonAttrs
                    (
                      old: {
                        postPatch =
                          ''
                            substituteInPlace tinygrad/engine/jit.py --replace-fail '"CUDA", "NV", "AMD"' '"CUDA", "NV", "AMD", "HSA"'
                            substituteInPlace tinygrad/engine/search.py --replace-fail '"CUDA", "AMD", "NV"' '"CUDA", "AMD", "NV", "HSA"'
                            # patch correct path to opencl
                            substituteInPlace tinygrad/runtime/autogen/opencl.py --replace-fail "ctypes.util.find_library('OpenCL')" "'${pkgs.ocl-icd}/lib/libOpenCL.so'"
                          '';
                      }
                    );
                  gymnasium = prev.gymnasium.overridePythonAttrs
                    (
                      old: {
                        buildInputs = (old.buildInputs or [ ]) ++ [ roms ];
                      }
                    );
                  shimmy = prev.shimmy.overridePythonAttrs
                    (
                      old: {
                        buildInputs = (old.buildInputs or [ ]) ++ [ prev.setuptools ];
                      }
                    );
                });

              postInstall = ''
                ln -s ${SDL2}/lib/libSDL2-2.0.so.0.18.2 $out/lib/libSDL2-2.0.so.0.16.0

                wrapProgram $out/bin/app --set ALE_PY_ROM_DIR "${roms.out}/share/roms/" \
                                         --set LD_LIBRARY_PATH "$out/lib" \
              '';
            };
          };

          devShells.default = pkgs.mkShell {
            inputsFrom = [ self.packages.${system}.default ];
          };
        }
      );
}
