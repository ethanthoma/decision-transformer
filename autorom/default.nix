{ lib
, writeText
, buildPythonPackage
, fetchFromGitHub
, fetchurl
, click
}:

let
  roms = fetchurl {
    url = "https://gist.githubusercontent.com/jjshoots/61b22aefce4456920ba99f2c36906eda/raw/00046ac3403768bfe45857610a3d333b8e35e026/Roms.tar.gz.b64";
    sha256 = "sha256-Asp3fBZHanL6NmgKK6ePJMOsMbIVUDNUml83oGUxF94=";
  };

  romsFile = writeText "roms_as_base64_tar_gz.b64" (builtins.readFile roms);
in
buildPythonPackage rec {
  pname = "autorom";
  version = "0.6.1";
  format = "setuptools";

  src = fetchFromGitHub {
    owner = "Farama-Foundation";
    repo = "AutoROM";
    rev = "v${version}";
    sha256 = "sha256-fC5OOXAnnP4x4j/IbpG0YdTz5F5pgyY0tumNjyrQ8FM=";
  };

  env.ALE_ROM_DIR = "$out/share/roms/";

  preConfigure = ''
    cd ./packages/AutoROM.accept-rom-license
  '';

  patches = [ ./autorom.patch ];

  postPatch = ''
    substituteInPlace src/AutoROM.py \
      --replace 'roms_as_base64_tar_gz.b64' '${romsFile}'
  '';

  propagatedBuildInputs = [ click ];

  postInstall = ''
    mkdir -p $out/share/roms
    mv $out/lib/python3.*/site-packages/AutoROM/roms/* $out/share/roms
  '';

  doCheck = false;

  meta = with lib; {
    description = "AutoROM installer for Atari ROMs";
    homepage = "https://github.com/Farama-Foundation/AutoROM";
    license = licenses.mit;
    maintainers = with maintainers; [ ethanthoma ];
    platforms = platforms.all;
  };
}


