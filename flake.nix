{
  description = "Application packaged using poetry2nix";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
          pkgs = nixpkgs.legacyPackages.${system};
          inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication defaultPoetryOverrides;

          name = "myapp";

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
          packages =
            {
              ${name} =
                mkPoetryApplication {
                  projectDir = self;

                  inherit python;

                  buildInputs = [ pkgs.makeWrapper SDL2 ];

                  nativeBuildInputs = [ pkgs.poetry ];

                  preferWheels = true;

                  overrides = defaultPoetryOverrides.extend
                    (final: prev: {
                      pygame = prev.pygame.overrideAttrs
                        (
                          old: rec {
                            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                              pythonPackages.cython
                              pkgs.pkg-config
                              pkgs.SDL2
                              pkgs.makeWrapper
                              pkgs.autoPatchelfHook
                              pkgs.dpkg
                              pythonPackages.wrapPython
                              pkgs.libGL
                              prev.setuptools
                            ];

                            buildInputs = (old.buildInputs or [ ]) ++ (with pkgs; [
                              freetype
                              libjpeg
                              libpng
                              xorg.libX11
                              portmidi
                              pkgs.SDL2
                              SDL2_image
                              SDL2_mixer
                              SDL2_ttf
                              prev.setuptools
                            ]);

                            src = pkgs.fetchPypi {
                              pname = "pygame";
                              version = "2.5.2";
                              hash = "sha256-wbietdU556xc91UTEl+18vCi2Rix/W6YHyO/CsGxwko=";
                            };

                            preConfigure = '''';

                            preBuild = ''
                              export PYSDL2_DLL_PATH="${pkgs.SDL2}/lib/libSDL2.so"
                            '';

                            postInstall = ''
                              wrapPythonPrograms
                            '';
                          }
                        );
                    }
                    );

                  postInstall = ''
                    ln -s ${SDL2}/lib/libSDL2-2.0.so.0.18.2 $out/lib/libSDL2-2.0.so.0.16.0

                    wrapProgram $out/bin/app --set ALE_PY_ROM_DIR "${roms.out}/share/roms/" \
                                             --set LD_LIBRARY_PATH "$out/lib"
                  '';
                };

              default = self.packages.${system}.${name};
            };

          devShells.default = pkgs.mkShell {
            inputsFrom = [ self.packages.${system}.${name} ];
          };

          devShells.poetry = pkgs.mkShell {
            packages = [ pkgs.poetry ];
          };
        });
}
