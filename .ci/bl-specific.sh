#!/bin/bash

# For reference: https://www.gnu.org/software/bash/manual/html_node/The-Set-Builtin.html.
set -vxeuo pipefail

# Beamline-specific steps.

sudo mkdir -p -v /nsls2/xf08bm/shared/config/runengine-metadata
sudo chown -Rv $USER: /nsls2/

# Temporary solution until the bloptools#5 is merged/released.
#
#   https://github.com/NSLS-II/bloptools/pull/5
#
mkdir -v $HOME/src/
cd $HOME/src/
git clone https://github.com/thomaswmorris/bloptools.git
cd bloptools
git checkout gp-opt
