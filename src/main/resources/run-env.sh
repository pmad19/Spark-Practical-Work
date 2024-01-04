#!/bin/bash

export APP_NAME=bigdata-2023-2024-g12
export FOLDER=
export PATH_PROJECT=$(realpath $(dirname $0))/../../..

export SPARK_PROPERTIES=$PATH_PROJECT/src/main/resources/spark/properties.conf