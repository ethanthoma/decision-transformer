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

          roms = import ./autorom {
            inherit (pkgs) lib fetchFromGitHub fetchurl writeText;
            inherit (pkgs.python311Packages) buildPythonPackage click;
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
            ${name} =
              mkPoetryApplication
                {
                  projectDir = self;

                  inherit python;

                  buildInputs = [ pkgs.makeWrapper SDL2 ];

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
