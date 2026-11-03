#!/usr/bin/env bash
# Use ./bumpVersion.sh VERSION
## Script upgrades the version numbers in the places necessary across the repo.
## It will add the current and next version to appropiate locations.
## Used for after a release to start work on next version.

set -e -u -o pipefail

if [[ -n "${DEBUG-}" ]]; then
  set -x
fi

# Use docs dir as freezeAtVersion.py currently only works from that dir
cd "$(dirname "$0")/../docs/"

# Get the current version from _config.yml
current_version=$(sed -n 's/current_version:\s*//p' _config.yml)

echo "Current version: ${current_version}"
if [  $# -eq 0  ]
  then
    echo "Next version not specified. Usage ./{$0} VERSION"
    exit 1
fi

next_version=${1}
dev_name=${dev_name:-'dev1'}

# Freeze docs
python3 bin/freezeAtVersion.py --version "${current_version}"

# Update version in _config.yml
sed -i 's/\(current_version:\).*/\1 '"${next_version}"'/' _config.yml

# Update previous_versions.yml
underscored_current_version=$(echo "${current_version}" | tr . _)
sed -i '1i- title: '"${current_version}" _data/previous_versions.yml
sed -i '2i\ \ url: /docs/'"${underscored_current_version}" _data/previous_versions.yml

# Add new version number to release notes
sed -i "1i ## Version ${next_version}\\n" ../release_notes.md
