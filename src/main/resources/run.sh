#!/bin/bash

. ./run-env.sh

spark-submit \
    --properties-file $SPARK_PROPERTIES \
    --master local[*] \
    --py-files $APP_NAME-python.zip \
    $PATH_PROJECT/src/main/python/core/main.py