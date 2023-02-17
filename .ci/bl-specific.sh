#!/bin/bash

# Beamline-specific steps.

sudo mkdir -p -v /nsls2/xf08bm/shared/config/runengine-metadata
sudo chown -Rv $USER: /nsls2/

# Temporary solution until the bloptools#5 is merged/released.
#
#   https://github.com/NSLS-II/bloptools/pull/5
#
mkdir ~/src/ && src/
git clone https://github.com/thomaswmorris/bloptools.git
cd bloptools
git checkout gp-opt
