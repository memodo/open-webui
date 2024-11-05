#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export NETWORK_MODE=host
else
    export NETWORK_MODE=bridge
fi

docker compose -f docker-compose.chroma.yml up