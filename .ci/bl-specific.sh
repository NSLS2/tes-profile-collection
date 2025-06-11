#!/bin/bash

# For reference: https://www.gnu.org/software/bash/manual/html_node/The-Set-Builtin.html.
set -vxeo pipefail

# Beamline-specific steps.

sudo mkdir -p -v /nsls2/xf08bm/shared/config/runengine-metadata
sudo chown -Rv $USER: /nsls2/

