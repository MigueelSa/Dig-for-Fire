#!/bin/bash
rm -rf build dist
pyinstaller --clean --noconfirm --onefile digforfire.spec

