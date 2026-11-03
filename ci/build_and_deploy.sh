#!/bin/bash

if [[ $TRAVIS_BRANCH == 'master' ]]; then
  ./gradlew clean build publish -PnexusUser=$NEXUS_USER -PnexusPassword=$NEXUS_PASSWORD -Psigning.keyId=$SIGNING_KEY_ID -Psigning.password=$SIGNING_PASSWORD -Psigning.secretKeyRingFile=../ci/secret-keys-keanu.gpg --info --stacktrace
else
  ./gradlew clean build --info --stacktrace
fi
