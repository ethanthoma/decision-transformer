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
        in
        {
          packages = {
            myapp =
              let
                python = pkgs.python311;

                roms = import ./autorom {
                  inherit (pkgs) lib fetchFromGitHub fetchurl writeText;
                  inherit (pkgs.python311Packages) buildPythonPackage click;
                };
              in
              mkPoetryApplication
                rec {
                  projectDir = self;

                  inherit python;

                  nativeBuildInputs = [ pkgs.poetry ];

                  overrides = defaultPoetryOverrides.extend
                    (final: prev: {
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

                  env.ALE_ROM_DIR = "${roms.out}/share/roms/";

                  postInstall = ''
                    mkdir -p $out/lib/python${pkgs.lib.versions.majorMinor python.version}/site-packages/ale_py/roms
                    cp -R ${roms.out}/share/roms/* $out/lib/python${pkgs.lib.versions.majorMinor python.version}/site-packages/ale_py/roms/
                    echo $out/lib/python${pkgs.lib.versions.majorMinor python.version}/site-packages/ale_py/roms
                  '';
                };
            default = self.packages.${system}.myapp;
          };

          devShells.default = pkgs.mkShell {
            inputsFrom = [ self.packages.${system}.myapp ];
          };

          devShells.poetry = pkgs.mkShell {
            packages = [ pkgs.poetry ];
          };
        });
}
