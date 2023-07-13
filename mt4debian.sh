#!/bin/bash

# Copyright 2022, MetaQuotes Ltd.

# MetaTrader download url
URL="https://download.mql5.com/cdn/web/metaquotes.software.corp/mt4/mt4oldsetup.exe"
# Wine version to install: stable or devel
WINE_VERSION="stable"

# Prepare: switch to 32 bit and add Wine key
sudo dpkg --add-architecture i386
wget -nc https://dl.winehq.org/wine-builds/winehq.key
sudo mv winehq.key /usr/share/keyrings/winehq-archive.key

# Get Debian version and trim to major only
OS_VER=$(lsb_release -r |cut -f2 |cut -d "." -f1)
# Choose repository based on Debian version
if (( $OS_VER >= 12)); then
  wget -nc https://dl.winehq.org/wine-builds/debian/dists/bookworm/winehq-bookworm.sources
  sudo mv winehq-bookworm.sources /etc/apt/sources.list.d/
elif (( $OS_VER < 12 )) && (( $OS_VER >= 11 )); then
  wget -nc https://dl.winehq.org/wine-builds/debian/dists/bullseye/winehq-bullseye.sources
  sudo mv winehq-bullseye.sources /etc/apt/sources.list.d/
elif (( $OS_VER <= 10 )); then
  wget -nc https://dl.winehq.org/wine-builds/debian/dists/buster/winehq-buster.sources
  sudo mv winehq-buster.sources /etc/apt/sources.list.d/
fi

# Update package and install Wine
sudo apt update
sudo apt upgrade
sudo apt install --install-recommends winehq-$WINE_VERSION

# Download MetaTrader
wget $URL -O mt4setup.exe

# Set environment to Windows 10
WINEPREFIX=~/.mt4 WINEARCH=win32 winecfg -v=win10
# Start MetaTrader installer in 32 bit environment
WINEPREFIX=~/.mt4 WINEARCH=win32 wine mt4setup.exe
